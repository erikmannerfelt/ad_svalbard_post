import matplotlib.pyplot as plt
import matplotlib.patheffects
import pandas as pd
import geopandas as gpd
import shapely.ops
import numpy as np

from pathlib import Path

CACHE_DIR = (Path(__file__).absolute() / "../cache").resolve()
CACHE_DIR.mkdir(exist_ok=True)
CRS_EPSG = 32633

def make_coastline_intervals() -> gpd.GeoDataFrame:
    cache_path = CACHE_DIR / "coastline_intervals.arrow"

    if cache_path.is_file():
        return gpd.read_feather(cache_path).set_index("key")  # type: ignore
    coastlines = {}
    for year in range(2013, 2025):
        coastline = gpd.read_file(f"zip://input/Coast_annual.zip/Coast{year}.shp").to_crs(CRS_EPSG).dissolve().iloc[0]["geometry"]

        coastlines[year] = coastline

    c_13_18 = shapely.ops.unary_union([g for yr, g in coastlines.items() if yr < 2019]) 
    c_19_24 = shapely.ops.unary_union([g for yr, g in coastlines.items() if yr >= 2019]) 
    c_13_24 = shapely.ops.unary_union([c_13_18, c_19_24])

    out = gpd.GeoDataFrame(
        pd.DataFrame.from_records(
            [
                {
                    "key": "13_18",
                    "name": "2013-2018",
                    "geometry": c_13_18,
                },
                {
                    "key": "19_24",
                    "name": "2019-2024",
                    "geometry": c_19_24,
                },
                {
                    "key": "13_24",
                    "name": "2013-2024",
                    "geometry": c_13_24,
                }
            ]
        ),
        geometry="geometry",
        crs=CRS_EPSG,
    )

    out.to_feather(cache_path)

    return gpd.read_feather(cache_path).set_index("key") # type: ignore

        


def main():
    coasts = make_coastline_intervals()
    coasts["start_year"] = coasts["name"].str.split("-", expand=True).iloc[:, 0].astype(int)
    coasts["end_year"] = coasts["name"].str.split("-", expand=True).iloc[:, 1].astype(int)

    # rgi7_orig = gpd.read_file("input/RGI2000-v7.0-G-07_svalbard_jan_mayen.zip").to_crs(CRS_EPSG)
    rgi7_orig = gpd.read_file("zip:///home/erikmann/Projects/UiO/phd_thesis/data/RGI_V7_Surge_Database.zip/RGI_V7_Surge_Database/RGI2000-v7.0-G-07_svalbard_jan_mayen_Surge_Database.shp").to_crs(CRS_EPSG)
    rgi7_orig["geometry"] = rgi7_orig["geometry"].buffer(0)
    rgi7_orig["modified"] = False

    outline_corrections = gpd.read_file("shapes/outline_corrections.geojson")
    outline_corrections["is_extension"] = outline_corrections["fix_type"].str.contains("partition_override")

    for key, coastline in coasts.iterrows():
        rgi7 = rgi7_orig.copy()
        for _, item in outline_corrections.sort_values(["is_extension", "priority"]).iterrows():
            if any(
                   [
                       item["first_active_year"] > coastline["end_year"],
                       item["last_active_year"] < coastline["start_year"],
                    ]
                ):
                continue
            outline_idx = rgi7[rgi7["rgi_id"] == item["rgi_id"]].index.tolist()

            if len(outline_idx) != 1:
                raise RuntimeError(f"Problematic correction. Matched none or too many indexes.\n{item}")

            overlaps = rgi7["geometry"].overlaps(item["geometry"])
        
            rgi7.loc[overlaps, "geometry"] = rgi7.loc[overlaps, "geometry"].difference(item["geometry"])
            rgi7.loc[outline_idx[0], "geometry"] = rgi7.loc[outline_idx[0], "geometry"].union(item["geometry"])
            rgi7.loc[overlaps, "modified"] = True

        rgi7["geometry"] = rgi7["geometry"].intersection(coastline["geometry"])

        rgi7.to_feather(f"cache/rgi7_{key}.arrow")
        


def sample_rasters(redo: bool = False):

    cache_path = CACHE_DIR / "outlines_sampled.arrow"

    if cache_path.is_file() and not redo:
        return gpd.read_feather(cache_path)
    import rasterio
    import rasterio.features
    name_overrides = {
        "RGI2000-v7.0-G-07-01383": "Storisstraumen",
        "RGI2000-v7.0-G-07-01385": "Austfonna Basin-2",
        "RGI2000-v7.0-G-07-01379": "Austfonna Basin-7",
        "RGI2000-v7.0-G-07-01381": "Austfonna Basin-5",

    }
    outlines = gpd.read_file("cache/rgi7_13_24.arrow")
    outlines["id"] = outlines["rgi_id"].str.split("-", expand=True).iloc[:, -1].astype(int)
    outlines = outlines[~outlines.geometry.is_empty]

    outlines["glac_name"] = outlines.apply(lambda row: name_overrides.get(row["rgi_id"], row["glac_name"]), axis="columns")

    glacier_zones = gpd.read_file("shapes/glacier_zones.geojson").to_crs(outlines.crs).set_index("zone_label")
    outlines = gpd.sjoin(outlines, glacier_zones)

    min_year = 2010
    def is_surging(string):

        results = [False]
        for part in str(string).split(";"):
            try:
                year = int(part.strip())
                results.append(year >= min_year)
            except:
                continue
        return any(results)
    outlines["surging"] = (outlines["S_Term"] + ";" + outlines["S_Onset"].str.replace("2024", "").str.replace("2025", "")).apply(is_surging)

    data = {}
    trend_dir = Path("/home/erikmann/temp/")
    with rasterio.open("input/trend_2013-2024_slope_20260409.tif") as raster:
        rasterized = rasterio.features.rasterize([(outline["geometry"], outline["id"]) for _, outline in outlines.iterrows()], out_shape=(raster.height, raster.width), fill=0, transform=raster.transform)
        periglacial = rasterized == 0
        rasterized = rasterized[~periglacial]

        data[f"dhdt_13_24"] = raster.read(1, masked=True)[~periglacial].filled(np.nan)
        res = raster.res[0]

    with rasterio.open(trend_dir / "trend_2013-2024_accel_20260309B.tif") as raster:
        data["accel"] = (raster.read(1, masked=True)[~periglacial].astype("float32") * raster.scales[0]).filled(np.nan)

    for short, long in [("13_18", "2013-2018"), ("19_24", "2019-2024")]:
        with rasterio.open(f"input/trend_{long}_slope_20260409.tif") as raster:

            data[f"dhdt_{short}"] = raster.read(1, masked=True)[~periglacial].filled(np.nan)
    
    outlines["area"] = np.nan
    outlines["dh"] = np.nan
    for idx, outline in outlines.iterrows():
        mask = rasterized == outline["id"]
        area = np.count_nonzero(mask) * res ** 2
        if area == 0:
            continue

        outlines.loc[idx, "area"] = area

        for key, arr in data.items():
            outlines.loc[idx, key] = np.nanmean(arr[mask])
            outlines.loc[idx, f"{key}_positive_v"] = np.nansum(arr[mask][arr[mask] > 0.]) * res ** 2
            outlines.loc[idx, f"{key}_positive_area"] = np.count_nonzero(arr[mask] > 0.1) * res ** 2

        # Manual bias correction at Kvitøyjøkulen
        if outline["rgi_id"] == "RGI2000-v7.0-G-07-01583":
            outlines.loc[idx, "dh"] += 1.5

    outlines.to_feather(cache_path)
    return gpd.read_feather(cache_path)


def get_dhdt():

    outlines = sample_rasters()
    glacier_zones = gpd.read_file("shapes/glacier_zones.geojson").to_crs(outlines.crs).set_index("zone_label")

    coasts = make_coastline_intervals().loc[["13_24"]].explode().to_crs(glacier_zones.crs)
    coasts = coasts[coasts.geometry.area > 1e8]
    coasts = coasts.dissolve().simplify(100)

    glacier_zones.geometry = glacier_zones.geometry.intersection(coasts.geometry[0])

    # glacier_zones["color"] = glacier_zones["zone_label"].apply(
    #     {
    #         "NW": "#3d8673",
    #         "N": "#174940",
    #         "NE": "#cbebda",
    #         "C": "#83cda9",
    #         "S": "#7e3404",
    #         "B": "#bb6a07",
    #         "V": "#f6ebb6",
    #         "A": "#ddbd5c",
    #     }.get
    # )
    # glacier_zones.to_file("shapes/glacier_zones.geojson")
    # return
            
    per_zone_grouped = outlines.groupby("zone_label")
    glacier_zones["area"] = per_zone_grouped["area"].sum()

    # keys = [str(col) for col in outlines.columns if "dhdt" in col or "accel" in col]
    to_v_keys = ["dhdt_13_24", "dhdt_13_18", "dhdt_19_24", "accel"]

    # per_zone = outlines[["zone_name"]].first()
    # per_zone["area"] = 

    # for key in data:
    #     per_zone = per_zone_grouped[


    for key in to_v_keys:
        dv_col = key.replace("dhdt", "dvdt").replace("accel", "accel_v")
        outlines[dv_col] = outlines[key] * outlines["area"]

        glacier_zones[key] = per_zone_grouped[key].mean()
        glacier_zones[dv_col] = glacier_zones[key] * glacier_zones["area"] 


        nonsurging_positive_volume = outlines.query("~surging")[f"{key}_positive_v"].sum() / 1e9
        positive_nonsurge = 100 * outlines.query("~surging")[f"{key}_positive_area"].sum() / outlines.query("~surging")["area"].sum()

        change_rate = outlines[dv_col].sum()
        surging_change_rate = outlines.query('surging')[dv_col].sum()
        nonsurging_change_rate = outlines.query('~surging')[dv_col].sum()

        print(dv_col)
        yr_unit = "yr²" if "accel" in key else "yr" 
        print(f"Nonsurging positive sum: {nonsurging_positive_volume:.2f} km³ / {yr_unit}" )
        print(f"Nonsurging positive area: {positive_nonsurge:.2f}%")
        print(f"Change rate: {change_rate / 1e9:.2f} km³ / {yr_unit}")
        print(f"\tSurging change rate: {surging_change_rate / 1e9:.2f} km³ / {yr_unit} ({100 * surging_change_rate / change_rate:.2f}%)")
        print(f"\tNon-surging change rate: {nonsurging_change_rate / 1e9:.2f} km³ / {yr_unit} ({100 * nonsurging_change_rate / change_rate:.2f}%)")

        for _, zone in glacier_zones.iterrows():
            print(f"- {zone['zone_name']}:\t{zone[dv_col] / 1e9:.2f} km³ / {yr_unit} ({100 * zone[dv_col] / glacier_zones[dv_col].sum():.2f}%)") 

            

       
        # print(glacier_zones[dv_col] / 1e9)

        
        print("\n\n")


    print(outlines.groupby("surging")["area"].sum() / 1e6)

    
    # for i, (label, zone) in enumerate(glacier_zones.iterrows()):
    #     plt.scatter(zone["accel_v"] / 1e9, zone["dvdt_13_24"] / 1e9, s=0.5 * zone["area"] / 1e6, c=zone["color"], zorder=i)
    #     plt.annotate(label, (zone["accel_v"] / 1e9, zone["dvdt_13_24"] / 1e9), ha="center", va="center", zorder=i + 1)
    # plt.ylim(-5, 0)
    # plt.xlim(-1.5, 0.25)
    # plt.show()
    #
    fig = plt.figure(figsize=(5, 5))
    inset = plt.gca().inset_axes([0., 0.6, 0.3, 0.4])
    # inset.set_axis_off()
    glacier_zones.plot(color=glacier_zones["color"], ax=inset)
    for i, (label, zone) in enumerate(glacier_zones.iterrows()):
        plt.scatter(zone["dvdt_13_18"] / 1e9, zone["dvdt_19_24"] / 1e9, s=0.3 * zone["area"] / 1e6, c=zone["color"], zorder=i + 1, edgecolor="#ccc")

        xy_text = (zone["dvdt_13_18"] / 1e9, zone["dvdt_19_24"] / 1e9)

        if label == "NW":
            xy_text = (xy_text[0] - 0.35, xy_text[1] - 0.28)
        elif label == "N":
            xy_text = (xy_text[0] - 0., xy_text[1] - 0.15)
            

        text_kwargs = {"ha": "center", "va": "center", "path_effects":[matplotlib.patheffects.withStroke(foreground="white", linewidth=1)]}
        plt.annotate(label,xy_text,zorder=i + 2, **text_kwargs)
        inset.annotate(label, (zone.geometry.centroid.x, zone.geometry.centroid.y), **text_kwargs)


    vlim = [-6, 0.2]
    plt.fill_between(vlim, vlim, [max(vlim)] * 2, color="#018571", alpha=0.2)  
    plt.text(0.6, 0.7, "Less negative", transform=plt.gca().transAxes, color="#018571", ha="right", fontsize=12)
    plt.text(0.9, 0.5, "More negative", transform=plt.gca().transAxes, color="#a6611a", ha="right", fontsize=12)
    plt.fill_between(vlim, vlim, [min(vlim)] * 2, color="#a6611a", alpha=0.2)  
    plt.plot(vlim, vlim, color="#333", linestyle="--", zorder=0)
    plt.ylim(vlim)
    plt.xlim(vlim)

    inset.set_xticks([])
    inset.set_yticks([])
    plt.xlabel("Volume change rate 2013-2018 (km³ / a)")
    plt.ylabel("Volume change rate 2019-2024 (km³ / a)")
    plt.tight_layout()
    Path("figures/").mkdir(exist_ok=True)

    plt.savefig("figures/perzone_volume_change.svg")
    # plt.show()
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    inset = plt.gca().inset_axes([0., 0.5, 0.4, 0.5])
    # inset.set_axis_off()
    glacier_zones.plot(color=glacier_zones["color"], ax=inset)
    for i, (label, zone) in enumerate(glacier_zones.iterrows()):
        plt.scatter(zone["dhdt_13_18"], zone["dhdt_19_24"], s=0.3 * zone["area"] / 1e6, c=zone["color"], zorder=i + 1, edgecolor="#ccc")

        xy_text = (zone["dhdt_13_18"], zone["dhdt_19_24"])

        # if label == "NW":
        #     xy_text = (xy_text[0] - 0.35, xy_text[1] - 0.28)
        # elif label == "N":
        #     xy_text = (xy_text[0] - 0., xy_text[1] - 0.15)
            

        text_kwargs = {"ha": "center", "va": "center", "path_effects":[matplotlib.patheffects.withStroke(foreground="white", linewidth=1)]}
        plt.annotate(label,xy_text,zorder=i + 2, **text_kwargs)
        inset.annotate(label, (zone.geometry.centroid.x, zone.geometry.centroid.y), **text_kwargs)


    vlim = [-1.05, -0.08]
    plt.fill_between(vlim, vlim, [max(vlim)] * 2, color="#018571", alpha=0.2)  
    plt.text(0.68, 0.82, "Less\nnegative", transform=plt.gca().transAxes, color="#018571", ha="center", fontsize=12)
    plt.text(0.87, 0.68, "More\nnegative", transform=plt.gca().transAxes, color="#a6611a", ha="center", fontsize=12)
    plt.fill_between(vlim, vlim, [min(vlim)] * 2, color="#a6611a", alpha=0.2)  
    plt.plot(vlim, vlim, color="#333", linestyle="--", zorder=0)
    plt.ylim(vlim)
    plt.xlim(vlim)

    inset.set_xticks([])
    inset.set_yticks([])
    plt.xlabel("Elevation change rate 2013-2018 (km³ / a)")
    plt.ylabel("Elevation change rate 2019-2024 (km³ / a)")
    plt.tight_layout()
    Path("figures/").mkdir(exist_ok=True)

    plt.savefig("figures/perzone_elevation_change.svg")
    plt.show()
    # print(outlines["dv"].sum() / 1e9)
    # print((outlines["dhdt2"] * outlines["area"]).sum() / 1e9)
    # print(outlines["surging"].sum(), outlines.shape[0])

    # with pd.option_context("display.max_rows", 20, "display.height", 20,):
    # pd.set_option("display.max_rows", 30)
    # print(outlines.sort_values("dv")[["glac_name", "dv", "dh", "id"]].iloc[:30])
    # print(outlines.groupby("surging")["dv"].sum() / 1e9)
    # print(outlines.groupby("surging")["dh"].mean())

    # outlines.plot(column="dv", cmap="RdBu", vmin=-1e9, vmax=1e9)
    # outlines.plot(column="dhdt", cmap="RdBu", vmin=-3, vmax=3)
    # outlines.plot(column="dhdt2", cmap="PuOr", vmin=-0.5, vmax=0.5)

    # plt.show()
    return
    
    for i, (issurging, items) in enumerate(outlines.groupby("surging")):
        plt.subplot(2, 1, i + 1)
        
        
        plt.hist(items["dh"].dropna(), bins=np.linspace(-5, 1, 20 if issurging else 100), color="red" if issurging else "blue")
        plt.yscale("log")
    # empty = hist[hist == 0]
    # plt.bar(bins[~empty], np.log10(hist[~empty])) 
    plt.show()

if __name__ == "__main__":
    main()
