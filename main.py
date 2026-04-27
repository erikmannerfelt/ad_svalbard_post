import matplotlib.pyplot as plt
import matplotlib.patheffects
import matplotlib.cm
import matplotlib.colors
import mpl_toolkits.axes_grid1.inset_locator
import pandas as pd
import geopandas as gpd
import shapely.ops
import numpy as np
import scipy.interpolate
import base64, io
import xml.etree.ElementTree as ET
from PIL import Image

from record_information import record_information, format_float

from pathlib import Path

CACHE_DIR = (Path(__file__).absolute() / "../cache").resolve()
CACHE_DIR.mkdir(exist_ok=True)
CRS_EPSG = 32633

DHDT_VMIN = 2
DHDT_VMAX = 1
DHDT_COLORS = [
    (-DHDT_VMIN, "#400912"),
    (-0.75 * DHDT_VMIN, "#630d1c"),
    (-0.40 * DHDT_VMIN, "#cd721c"),
    (-0., "#eeeeec"),
    (DHDT_VMAX, "#6497e3"),
]

DHDT_NORMALIZER = matplotlib.colors.Normalize(vmin=-DHDT_VMIN, vmax=DHDT_VMAX, clip=True)
DHDT_SM = matplotlib.cm.ScalarMappable(
    norm=DHDT_NORMALIZER,
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list("dhdt", [(DHDT_NORMALIZER(a), b) for a, b in DHDT_COLORS]),
)

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
        

def get_neff_model():
    neff_data = pd.read_csv("input/vgm_neff_cirq_numerical.csv", index_col=0).squeeze()
    model = scipy.interpolate.interp1d(neff_data.index, neff_data, fill_value="extrapolate")
    neff_model = lambda a: model(np.clip(a, a_min=0, a_max=1.2e8)) # Clamp by the largest gaussian range
    return neff_model
    

def sample_rasters(redo: bool = False) -> gpd.GeoDataFrame:

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
    neff_model = get_neff_model()

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

    date = "20260424"
    data = {}
    with rasterio.open(f"input/trend_2013-2024_slope_{date}.tif") as raster:
        rasterized = rasterio.features.rasterize([(outline["geometry"], outline["id"]) for _, outline in outlines.iterrows()], out_shape=(raster.height, raster.width), fill=0, transform=raster.transform)
        periglacial = rasterized == 0
        rasterized = rasterized[~periglacial]

        bounds =raster.bounds

        # data[f"dhdt_13_24"] = raster.read(1, masked=True)[~periglacial].filled(np.nan)
        res = raster.res[0]

    # with rasterio.open(trend_dir / "trend_2013-2024_accel_20260309B.tif") as raster:
    #     data["accel"] = (raster.read(1, masked=True)[~periglacial].astype("float32") * raster.scales[0]).filled(np.nan)
    #
    for short, kind, long in [
        ("13_18", "slope", "2013-2018"),
        ("19_24", "slope_tcorr", "2019-2024"),
        ("13_24", "slope", "2013-2024"),
        # ("13_24", "accel", "2013-2024"),
    ]:
        # date = "20260409" if kind == "slope" else "20260309B"
        base_key = f"{kind.replace('_tcorr', '')}_{short}"
        paths = {
            base_key: Path(f"input/trend_{long}_{kind}_{date}.tif"),
        }
           
        paths[base_key + "_err_unscaled"] = paths[base_key].with_stem(paths[base_key].stem.replace(kind, kind + "_err"))
        for key, filepath in paths.items():
            with rasterio.open(filepath) as raster:
                window = rasterio.windows.from_bounds(*bounds, transform=raster.transform)
                data[key] = (raster.read(1, masked=True, window=window, boundless=True)[~periglacial].astype("float32") * raster.scales[0]).filled(np.nan)

    data["accel_13_24"] = (data["slope_19_24"] - data["slope_13_18"]) / 6
    data["accel_13_24_err_unscaled"] = np.hypot(data["slope_19_24_err_unscaled"], data["slope_13_18_err_unscaled"]) / 6

    
    outlines["area"] = np.nan
    for idx, outline in outlines.iterrows():
        mask = rasterized == outline["id"]
        area = np.count_nonzero(mask) * res ** 2
        if area == 0:
            continue

        outlines.loc[idx, "area"] = area
        outlines.loc[idx, "neff"] = neff_model(area)

        for key, arr in data.items():
            outlines.loc[idx, key] = np.nanmean(arr[mask])
            # Manual bias correction at Kvitøyjøkulen
            if outline["rgi_id"] == "RGI2000-v7.0-G-07-01583":
                if key == "slope_13_24":
                    outlines.loc[idx, key] += 1.5

            if "_err_unscaled" in key:
                outlines.loc[idx, key] *= 2  # Convert to 2sigma
                outlines.loc[idx, key.replace("_unscaled", "")] = outlines.loc[idx, key] / (outlines.loc[idx, "neff"] ** 0.5)
            else:
                outlines.loc[idx, f"{key}_positive_vol"] = np.nansum(arr[mask][arr[mask] > 0.]) * res ** 2
                outlines.loc[idx, f"{key}_positive_area"] = np.count_nonzero(arr[mask] > 0.1) * res ** 2


    outlines.to_feather(cache_path)
    return gpd.read_feather(cache_path)


def plot_regional_dhdt_fig(glacier_zones):
    all_params = [
        {
            "xcol": "slope_13_18",
            "ycol": "slope_19_24",
            "col_suffix": "",
            "unit": "rate",
            "vlim": [-1.5, 0.2],
            "lessneg_text_xy": (0.68, 0.82),
            "moreneg_text_xy": (0.87, 0.68),
            "out_stem": "perzone_elevation_change",
        },
        {
            "xcol": "slope_13_18",
            "ycol": "slope_19_24",
            "unit": "rate",
            "col_suffix": "_nonsurging",
            "vlim": [-1.15, 0.2],
            "lessneg_text_xy": (0.68, 0.82),
            "moreneg_text_xy": (0.87, 0.68),
            "out_stem": "perzone_elevation_change_nonsurging",
        },
        {
            "xcol": "slope_13_18_vol",
            "ycol": "slope_19_24_vol",
            "unit": "vol_rate",
            "col_suffix": "",
            "vlim": [-10.5, 1],
            "lessneg_text_xy": (0.68, 0.82),
            "moreneg_text_xy": (0.87, 0.68),
            "out_stem": "perzone_volume_change",
        },
        {
            "xcol": "slope_13_18_vol",
            "ycol": "slope_19_24_vol",
            "unit": "vol_rate",
            "col_suffix": "_nonsurging",
            "vlim": [-6, 1],
            "lessneg_text_xy": (0.68, 0.82),
            "moreneg_text_xy": (0.87, 0.68),
            "out_stem": "perzone_volume_change_nonsurging",
        },
    ]

    for params in all_params:
        fig = plt.figure(figsize=(5, 4.9))

        xcol_interval = f"20{params['xcol'].split('_')[1]}-20{params['xcol'].split('_')[2]}"
        ycol_interval = f"20{params['ycol'].split('_')[1]}-20{params['ycol'].split('_')[2]}"

        unit_scale = {"rate": 1., "vol_rate": 1e-9}.get(params["unit"])
        unit = {"vol_rate": "km$^{3}$ a$^{-1}$", "rate": "m a$^{-1}$"}.get(params["unit"])
        axis_label = {"vol_rate": "Volume change rate", "rate": "Elevation change rate"}.get(params["unit"])
        plt.title(f"Regional {axis_label.lower()} " + ("(non-surging)" if "nonsurging" in params["out_stem"] else ""))
        inset = plt.gca().inset_axes([0., 0.5, 0.4, 0.5])
        # inset.set_axis_off()
        glacier_zones.plot(color=glacier_zones["color"], ax=inset)
        for i, (label, zone) in enumerate(glacier_zones.sort_values("area" + params["col_suffix"], ascending=False).iterrows()):
            plt.errorbar(
                x=zone[params["xcol"] + params["col_suffix"]] * unit_scale,
                y=zone[params["ycol"] + params["col_suffix"]] * unit_scale,
                yerr=zone[params["xcol"] + "_err" + params["col_suffix"]] * unit_scale,
                xerr=zone[params["ycol"] + "_err" + params["col_suffix"]] * unit_scale,
                color="black",
                marker="o",
                markersize=0.8 * zone["area" + params["col_suffix"]] / 1e8,
                markerfacecolor=zone["color"],
                markeredgecolor="#ccc",
                barsabove=True,
                alpha=1.0,
                zorder=i + 1,
            )

            xy_text = (zone[params["xcol"] + params["col_suffix"]] * unit_scale, zone[params["ycol"] + params["col_suffix"]] * unit_scale)

            # if label == "NW":
            #     xy_text = (xy_text[0] - 0.35, xy_text[1] - 0.28)
            # elif label == "N":
            #     xy_text = (xy_text[0] - 0., xy_text[1] - 0.15)
            

            text_kwargs = {"ha": "center", "va": "center", "path_effects":[matplotlib.patheffects.withStroke(foreground="black", linewidth=1)], "color": "white"}
            plt.annotate(label,xy_text,zorder=i + 2, **text_kwargs)
            inset.annotate(label, (zone.geometry.centroid.x, zone.geometry.centroid.y), **text_kwargs)


        plt.fill_between(params["vlim"], params["vlim"], [max(params["vlim"])] * 2, color="#018571", alpha=0.2)  
        plt.text(*params["lessneg_text_xy"], "Less\nnegative", transform=plt.gca().transAxes, color="#018571", ha="center", fontsize=12)
        plt.text(*params["moreneg_text_xy"], "More\nnegative", transform=plt.gca().transAxes, color="#a6611a", ha="center", fontsize=12)
        plt.fill_between(params["vlim"], params["vlim"], [min(params["vlim"])] * 2, color="#a6611a", alpha=0.2)  
        plt.plot(params["vlim"], params["vlim"], color="#333", linestyle="--", zorder=0)
        plt.ylim(params["vlim"])
        plt.xlim(params["vlim"])

        inset.set_xticks([])
        inset.set_yticks([])
        plt.xlabel(f"{axis_label} {xcol_interval} ({unit})")
        plt.ylabel(f"{axis_label} {ycol_interval} ({unit})")
        plt.tight_layout()
        Path("figures/").mkdir(exist_ok=True)

        plt.savefig(f"figures/{params['out_stem']}.svg")
        plt.show()
        plt.close()

def get_dhdt():

    # neff_data = pd.read_csv("input/vgm_neff_cirq_numerical.csv", index_col=0).squeeze()
    # plt.plot(neff_data.index, neff_data ** 0.5)
    # neff_data = pd.read_csv("input/vgm_neff_cirq_theoretical.csv", index_col=0).squeeze()
    # plt.plot(neff_data.index, neff_data ** 0.5)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    # return

    outlines = sample_rasters()

    to_v_keys = ["slope_13_24", "slope_13_18", "slope_19_24", "accel_13_24"]
    for key in to_v_keys:
        for suffix in ["", "_err", "_err_unscaled"]:
            outlines[key + "_vol" + suffix] = outlines[key + suffix] * outlines["area"]
        
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

    per_zone_nonsurging_grouped = outlines.query("~surging").groupby("zone_label")
    # glacier_zones[f"area_nonsurging"] =per_zone_nonsurging_grouped["area"].sum()
    # glacier_zones_nonsurging = glacier_zones.copy()
    # glacier_zones_nonsurging["area"] = per_zone_nonsurging_grouped["area"].sum()

    # keys = [str(col) for col in outlines.columns if "dhdt" in col or "accel" in col]

    # per_zone = outlines[["zone_name"]].first()
    # per_zone["area"] = 

    # for key in data:
    #     per_zone = per_zone_grouped[

    # neff_model = lambda a: np.clip(get_neff_model()(a), a_min=0, a_max=outlines["area_km2"].max())
    neff_model = get_neff_model()
    # o

    record_information(
        {
            "units": {
                "vol_rate": "km$^{3}$~a$^{-1}$",
                "vol_accel": "km$^{3}$~a$^{-2}$"
            },
            "parameters": {
                "positive_change_threshold": 0.1,
            },
        }
    )

    for key in to_v_keys:
        vol_col = key + "_vol"
        err_col = key + "_err"
        vol_err_col = key + "_vol_err"

        for suffix, zone_grouped in [("", per_zone_grouped), ("_nonsurging", per_zone_nonsurging_grouped)]:
            glacier_zones["area" + suffix] = per_zone_grouped["area"].sum()
            glacier_zones[vol_col + suffix] = zone_grouped[vol_col].sum()
            glacier_zones[vol_err_col + suffix] = per_zone_grouped[vol_err_col + "_unscaled"].sum() / (neff_model(per_zone_grouped["area"].sum()) ** 0.5)
            # glacier_zones[vol_err_col + suffix] = zone_grouped[vol_err_col].sum()

            glacier_zones[key + suffix] = glacier_zones[f"{vol_col}{suffix}"] / glacier_zones[f"area{suffix}"]
            glacier_zones[err_col + suffix] = glacier_zones[vol_err_col] / glacier_zones[f"area{suffix}"]


        nonsurging_positive_volume = outlines.query("~surging")[f"{key}_positive_vol"].sum() / 1e9
        positive_nonsurge = 100 * outlines.query("~surging")[f"{key}_positive_area"].sum() / outlines.query("~surging")["area"].sum()

        change_rate = outlines[vol_col].sum()
        change_rate_err = outlines[vol_err_col].sum() / (neff_model(outlines["area"].sum()) ** 0.5)
        surging_change_rate = outlines.query('surging')[vol_col].sum()
        surging_change_rate_err = outlines.query('surging')[vol_err_col].sum() / (neff_model(outlines.query("surging")["area"].sum()) ** 0.5)
        nonsurging_change_rate = outlines.query('~surging')[vol_col].sum()
        nonsurging_change_rate_err = outlines.query('~surging')[vol_err_col].sum() / (neff_model(outlines.query("~surging")["area"].sum()) ** 0.5)

        per_zone = {}
        for zone_label, zone in glacier_zones.iterrows():
            per_zone[zone_label] = {
                "name": zone["zone_name"],
                "vol_rate": zone[vol_col] / 1e9,
                "vol_rate_err": zone[vol_err_col] / 1e9,
                "percentage_of_total": 100 * zone[vol_col] / glacier_zones[vol_col].sum(),
            }
            # print(f"- {zone['zone_name']}:\t{zone[vol_col] / 1e9:.2f}±{zone[vol_err_col] / 1e9:.2f} km³ / {yr_unit} ({100 * zone[vol_col] / glacier_zones[vol_col].sum():.2f}%)") 

        record_information(
            {
                "changes": {
                    key.replace("13", "start").replace("18", "mid").replace("19", "mid").replace("24", "end"):{
                        "vol_rate": change_rate / 1e9,
                        "vol_rate_err": change_rate_err / 1e9,
                        "nonsurging_positive_vol_rate": nonsurging_positive_volume / 1e9,
                        "nonsurging_positive_area_percent": positive_nonsurge,
                        "surging": {
                            "vol_rate": surging_change_rate / 1e9,
                            "vol_rate_err": surging_change_rate_err / 1e9,
                            "percentage_of_total": round(100 * surging_change_rate / change_rate),  
                        },
                        "nonsurging": {
                            "vol_rate": nonsurging_change_rate / 1e9,
                            "vol_rate_err": nonsurging_change_rate_err / 1e9,
                            "percentage_of_total": round(100 * nonsurging_change_rate / change_rate),  
                        },
                        "regions": per_zone,
                    }
                },

            }
        )

        print(vol_col)
        yr_unit = "yr²" if "accel" in key else "yr" 
        print(f"Nonsurging positive sum: {nonsurging_positive_volume:.2f} km³ / {yr_unit}" )
        print(f"Nonsurging positive area: {positive_nonsurge:.2f}%")
        print(f"Change rate: {change_rate / 1e9:.2f}±{change_rate_err / 1e9:.2f} km³ / {yr_unit}")
        print(f"\tSurging change rate: {surging_change_rate / 1e9:.2f}±{surging_change_rate_err / 1e9:.2f} km³ / {yr_unit} ({100 * surging_change_rate / change_rate:.2f}%)")
        print(f"\tNon-surging change rate: {nonsurging_change_rate / 1e9:.2f}±{nonsurging_change_rate_err / 1e9:.2f} km³ / {yr_unit} ({100 * nonsurging_change_rate / change_rate:.2f}%)")

        for _, zone in glacier_zones.iterrows():
            print(f"- {zone['zone_name']}:\t{zone[vol_col] / 1e9:.2f}±{zone[vol_err_col] / 1e9:.2f} km³ / {yr_unit} ({100 * zone[vol_col] / glacier_zones[vol_col].sum():.2f}%)") 
        
        print("\n\n")

    areas = outlines.groupby("surging")["area"].sum() / 1e6
    record_information(
        {
            "area": {
                "start_end": {
                    "all": round(areas.sum()),
                    "surging": round(areas[True]),
                    "surging_percent": round(100 * areas[True] / areas.sum()),
                    "nonsurging": round(areas[False]),
                },
            },
        }

    )
    
    plot_regional_dhdt_fig(glacier_zones)
    
    fig = plt.figure(figsize=(4, 3))
    axes: list[plt.Axes] = fig.subplots(2, 1, sharex=True, sharey=False).ravel().tolist() # type: ignore
    fig.subplots_adjust(left=0.135, bottom=0.165, right=0.995, top=0.98, hspace=0.1)

    for i, (issurging, items) in enumerate(outlines.groupby("surging")):
        axis = axes[i]
        vals = items["slope_13_24"].dropna()
        axis.hist(vals, bins=np.linspace(-5, 1, 30 if issurging else 100), color="#ff6666" if issurging else "#6699ff")
        if i == 0:
            axis.set_yscale("log")

        axis.set_ylabel("Count")
        if i == 1:
            axis.set_xlabel("Elevation change rate 2013-2024 (m a$^{-1}$)")

        plt.text(0.02, 0.95, ("Surging " if issurging else "Non-surging ") + f"(n={vals.shape[0]})", va="top", transform=axis.transAxes)

    plt.savefig("figures/surging_vs_nonsurging_hist.svg")
    # plt.tight_layout()
    plt.show()

def svg_png_to_jpg(svg_path: Path | str, out_path: Path | str | None = None, *, quality=90):
    svg_ns = "http://www.w3.org/2000/svg"
    xlink_ns = "http://www.w3.org/1999/xlink"

    ET.register_namespace("", svg_ns)
    ET.register_namespace("xlink", xlink_ns)
    out_path = svg_path if out_path is None else out_path

    tree = ET.parse(svg_path)
    root = tree.getroot()

    for img in root.findall(f".//{{{svg_ns}}}image"):
        href = img.get(f"{{{xlink_ns}}}href")
        if not href or not href.startswith("data:image/png;base64,"):
            continue

        data = base64.b64decode(href.split(",", 1)[1])
        im = Image.open(io.BytesIO(data))

        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")

        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, subsampling=0)

        img.set(
            f"{{{xlink_ns}}}href",
            "data:image/jpeg;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"),
        )

    tree.write(out_path, encoding="utf-8", xml_declaration=True)

def dhdt_overview_fig():
    import rasterio
    import rasterio.features

    coasts = make_coastline_intervals()

    fig = plt.figure(figsize=(8.3, 5.3))
    axes: list[plt.Axes] = fig.subplots(1, 2, sharex=True, sharey=True).ravel().tolist() # type: ignore

    all_params = [
        {
            "filepath": "input/trend_2013-2018_slope_20260424.tif",
            "title": "2013–2018",
            "coast": coasts.loc["13_18"],
        },
        {
            "filepath": "input/trend_2019-2024_slope_20260424.tif",
            "title": "2019–2024",
            "coast": coasts.loc["19_24"],
        }
    ]

    for i, params in enumerate(all_params):
        with rasterio.open(params["filepath"]) as raster:
            img = DHDT_SM.to_rgba(raster.read(1, masked=True).astype("float32").filled(0) * raster.scales[0])
            ocean = rasterio.features.rasterize(
                (params["coast"].geometry,),
                out_shape=(raster.height, raster.width),
                fill=0,
                transform=raster.transform
            ) == 0

            img[ocean, :] = 1.

            axes[i].imshow(
                img,
                extent=(raster.bounds.left, raster.bounds.right, raster.bounds.bottom, raster.bounds.top),
            )


        axes[i].set_xlim(4e5, 7.4e5)
        axes[i].set_ylim(8.5e6, 8.95e6)
        axes[i].set_xlabel("Easting (m; UTM 33N)")
        axes[i].set_title(params["title"])
        axes[i].ticklabel_format(scilimits=(0, 0))
            

    def add_inset(axis, left=0.6):
        # Add map color bar
        inset = axis.inset_axes((left, 0.02, 0.1, 0.15))
        cbar = plt.colorbar(DHDT_SM,cax=inset, pad=0.02)
        # text = plt.text(0.1, 1.25, "dH dt$^{-1}$", transform=inset.transAxes)
        text2 = plt.text(1.15, 0.5, "m a$^{-1}$", ha="left", va="center", transform=inset.transAxes )
        cbar.set_ticks([DHDT_NORMALIZER.vmin, DHDT_NORMALIZER.vmax], labels=[f"≤{DHDT_NORMALIZER.vmin:.0f}", f"≥{DHDT_NORMALIZER.vmax:.0f}"]) # type: ignore
        return [text2, inset]
        

    inset_items = add_inset(axes[1])
    axes[0].set_ylabel("Northing (m; UTM 33N)")

    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.98, top=0.95, wspace=0.1)
    # plt.show()
    out_path = "figures/perinterval_slope_overview.svg"
    plt.savefig(out_path, dpi=400)
    svg_png_to_jpg(out_path)

    # Make a new figure for only Austfonna
    for axis in axes:
        axis.set_xlim(614000, 740000)
        axis.set_ylim(8790000, 8950000)
    for item in inset_items:
        item.remove()
    add_inset(axes[1], left=0.75)

    out_path2 = "figures/perinterval_slope_austfonna.svg"
    plt.savefig(out_path2, dpi=400)
    svg_png_to_jpg(out_path2)
    
    plt.show()

    
    
def hypsometric():
    import rasterio as rio
    import rasterio.features

    outlines = sample_rasters()
    glacier_zones = gpd.read_file("shapes/glacier_zones.geojson").to_crs(outlines.crs).set_index("zone_label")

    glacier_zones["id"] = np.arange(glacier_zones.shape[0]) + 1
    elev_bins = np.linspace(-0.01, 1200.01, 11)
    elev_bin_centers = (elev_bins[1:] + elev_bins[:-1]) / 2

    rasters = {
        "dem": "input/trend_2013-2024_intercept.tif",
        "2013-2018": "input/trend_2013-2018_slope.tif",
        "2019-2024": "input/trend_2019-2024_slope.tif",
    }

    with rio.open(rasters["dem"]) as raster:
        scale = raster.scales[0]

    with rio.open(rasters["dem"], overview_level=3) as raster:

        transform = raster.transform
        bounds = raster.bounds
        dem_arr = (raster.read(1, masked=True).astype("float32") * scale).filled(0)

        zones_rasterized: np.ndarray = rasterio.features.rasterize(
            [(zone["geometry"], zone["id"]) for _, zone in glacier_zones.iterrows()],
            out_shape=dem_arr.shape,
            transform=raster.transform,
            dtype="uint8",
        )

        outlines_rasterized: np.ndarray = rasterio.features.rasterize(
            outlines.query("~surging")["geometry"].values,
            out_shape=dem_arr.shape,
            transform=raster.transform,
        ) == 1

        zones_rasterized = zones_rasterized[outlines_rasterized]

        dem_digitized = np.digitize(dem_arr, elev_bins)[outlines_rasterized]

        del dem_arr


    dhdt_arrs = {}

    for key in ["2013-2018", "2019-2024"]:
        with rio.open(rasters[key]) as raster:
            scale = raster.scales[0]

        with rio.open(rasters[key], overview_level=3) as raster:

            window = rio.windows.from_bounds(*bounds, transform=raster.transform)

            dhdt_arrs[key] = (raster.read(1, window=window, boundless=True, masked=True)[outlines_rasterized] * scale).filled(np.nan)

    per_zone = {}
    for label, row in glacier_zones.iterrows():
        zone_mask = zones_rasterized == row["id"]

        vals = []
        for idx in np.unique(dem_digitized):
            if ((idx - 1) >= elev_bin_centers.shape[0]):
                continue
            mask = zone_mask & (dem_digitized == idx)

            if np.count_nonzero(mask) == 0:
                continue
            entry = {}
            for key in dhdt_arrs:
                med = np.nanmedian(dhdt_arrs[key][mask])
                entry["elevation"] = elev_bin_centers[idx - 1]
                entry[f"{key}_med"] = med
                entry[f"{key}_nmad"] = 1.4826 * np.nanmedian(np.abs(dhdt_arrs[key][mask] - med))

            vals.append(entry)

        per_zone[label] = pd.DataFrame.from_records(vals).set_index("elevation")


    colors = {"2013-2018": DHDT_SM.to_rgba(1.), "2019-2024": DHDT_SM.to_rgba(-1)}

    fig = plt.figure(figsize=(8, 4))
    axes = fig.subplots(2, glacier_zones.shape[0] // 2, sharex=True, sharey=True)
    for col, (label, zone) in enumerate(glacier_zones.iterrows()):
        axis: plt.Axes = axes.ravel()[col]
        for row, interval in enumerate(dhdt_arrs.keys()):

            axis.set_title(str(zone["zone_name"]))
            axis.errorbar(
                per_zone[label].index,
                per_zone[label][f"{interval}_med"],
                per_zone[label][f"{interval}_nmad"],
                color=colors[interval], 
                alpha=0.5,
                fmt="none",
            )
            axis.scatter(
                per_zone[label].index,
                per_zone[label][f"{interval}_med"],
                marker="s",
                edgecolor="#777",
                color=colors[interval], 
                label=interval,
            )

        xlim = axis.get_xlim()
        axis.hlines(0, *xlim, color="#777", linestyles="--", alpha=0.5)
        axis.set_xlim(xlim)

        if col in [0, 4]:
            axis.set_ylabel("dH / dt (m a$^{-1}$)")

        if (col >= glacier_zones.shape[0] // 2):
            axis.set_xlabel("Elevation (m a.s.l.)")

        if (col + 1) == axes.size:
            axis.legend(loc="lower right")


    plt.ylim(-3, 0.8)

    plt.subplots_adjust(top=0.937, bottom=0.121, left=0.073, right=0.991, hspace=0.219, wspace=0.061)
    # plt.xscale("log")

    plt.savefig("figures/perzone_hypsometric_signal.svg")
    plt.show()

            

        
        


        



if __name__ == "__main__":
    main()
