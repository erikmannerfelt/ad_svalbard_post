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
import tqdm
import dataclasses

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


def surge_overlaps_interval(term_value, onset_value, min_year, max_year) -> bool:
    """Return whether any surge episode overlaps an interval.

    Episodes are encoded as semicolon-separated years. A termination value of
    ``Ongoing`` or ``Not observed`` is treated as open-ended.

    An episode overlaps an interval when it starts on or before ``max_year``
    and ends after ``min_year``.

    >>> surge_overlaps_interval("1998;2021", "1991;2017", 2013, 2018)
    True
    >>> surge_overlaps_interval("1998;2021", "1991;2017", 2019, 2024)
    True
    >>> surge_overlaps_interval("2014", "2011", 2019, 2024)
    False
    >>> surge_overlaps_interval("2014", "2011", 2013, 2018)
    True
    >>> surge_overlaps_interval("2025", "2025", 2013, 2024)
    False
    >>> surge_overlaps_interval("2013", "2011", 2013, 2024)
    False
    >>> surge_overlaps_interval("Not observed", "2003", 2019, 2024)
    True
    >>> surge_overlaps_interval("Not observed", "1990", 2019, 2024)
    False
    """

    def _tokens(value):
        if pd.isna(value):
            return []
        return [part.strip() for part in str(value).split(";") if part.strip()]

    onset_tokens = _tokens(onset_value)
    term_tokens = _tokens(term_value)

    for idx, onset_token in enumerate(onset_tokens):
        try:
            onset_year = int(onset_token)
        except ValueError:
            continue

        if onset_year > max_year:
            continue

        term_token = term_tokens[idx] if idx < len(term_tokens) else None
        if term_token is None:
            term_year = np.inf
        else:
            term_lower = term_token.lower()
            if term_lower in {"", "n/a", "not observed", "ongoing"}:
                term_year = np.inf
            else:
                try:
                    term_year = int(term_token)
                except ValueError:
                    term_year = np.inf

        if onset_year <= max_year and term_year > min_year:
            return True

    return False

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

        


def make_outlines():
    cache_path = CACHE_DIR / "rgi7_outlines.arrow"

    intervals = [
        ("13_18", 2013, 2018),
        ("19_24", 2019, 2024),
        ("13_24", 2013, 2024),
    ]

    if cache_path.is_file():
        out = gpd.read_feather(cache_path)
        for interval, _, _ in intervals:
            out[f"area_{interval}"] = out[f"geometry_{interval}"].area
        return out
    coasts = make_coastline_intervals()

    rgi7_orig = gpd.read_file("zip://input/RGI_V7_Surge_Database.zip/RGI_V7_Surge_Database/RGI2000-v7.0-G-07_svalbard_jan_mayen_Surge_Database.shp").to_crs(CRS_EPSG)
    rgi7_orig["geometry"] = rgi7_orig["geometry"].buffer(0)

    field_overrides = {
        "RGI2000-v7.0-G-07-01379": {
            "S_Onset": "2015",
            "S_Term": "Ongoing",
            "glac_name": "Austfonna Basin-7",
        },
        "RGI2000-v7.0-G-07-01383": {"glac_name": "Storisstraumen"},
        "RGI2000-v7.0-G-07-01385": {"glac_name": "Austfonna Basin-2"},
        "RGI2000-v7.0-G-07-01381": {"glac_name": "Austfonna Basin-5"},
        "RGI2000-v7.0-G-07-00560": { # Tinkarpbreen
            "S_Onset": "2012",
            "S_Term": "Ongoing",

        },
    }

    for rgi_id, overrides in field_overrides.items():
        idx = rgi7_orig[rgi7_orig["rgi_id"] == rgi_id].index
        for key, value in overrides.items():
            rgi7_orig.loc[idx, key] = value

    outline_corrections = gpd.read_file("shapes/outline_corrections.geojson")
    outline_corrections["is_extension"] = outline_corrections["fix_type"].str.contains("partition_override")

    out = rgi7_orig.drop(columns=["geometry"]).copy()
    rgi_id_to_idx = rgi7_orig.reset_index().set_index("rgi_id")["index"].to_dict()

    for key, min_year, max_year in tqdm.tqdm(intervals):
        coastline = coasts.loc[key]
        rgi7 = rgi7_orig.copy()
        rgi7["modified"] = False
        active_corrections = outline_corrections.loc[
            ~(
                (outline_corrections["first_active_year"] > max_year)
                | (outline_corrections["last_active_year"] < min_year)
            )
        ].sort_values(["is_extension", "priority"])

        for _, item in active_corrections.iterrows():
            outline_idx = rgi_id_to_idx.get(item["rgi_id"])
            if outline_idx is None:
                continue
            outline_idx = [outline_idx]

            if len(outline_idx) != 1:
                raise RuntimeError(f"Problematic correction. Matched none or too many indexes.\n{item}")

            overlaps = pd.Series(False, index=rgi7.index)
            try:
                candidate_idx = rgi7.sindex.query(item["geometry"], predicate="intersects")
                if len(candidate_idx):
                    overlaps.loc[candidate_idx] = rgi7.loc[candidate_idx, "geometry"].overlaps(item["geometry"]).to_numpy()
            except Exception:
                overlaps = rgi7["geometry"].overlaps(item["geometry"])
        
            rgi7.loc[overlaps, "geometry"] = rgi7.loc[overlaps, "geometry"].difference(item["geometry"])
            rgi7.loc[outline_idx[0], "geometry"] = rgi7.loc[outline_idx[0], "geometry"].union(item["geometry"])
            rgi7.loc[overlaps, "modified"] = True

        rgi7["geometry"] = rgi7["geometry"].intersection(coastline["geometry"])

        out[f"geometry_{key}"] = gpd.GeoSeries(rgi7["geometry"].values, crs=CRS_EPSG)
        out[f"modified_{key}"] = rgi7["modified"].values
        out[f"surging_{key}"] = [
            surge_overlaps_interval(term_value, onset_value, min_year, max_year)
            for term_value, onset_value in zip(rgi7["S_Term"], rgi7["S_Onset"])
        ]

    out = gpd.GeoDataFrame(out, geometry="geometry_13_24", crs=CRS_EPSG)
    for interval, _, _ in intervals:
        out[f"area_{interval}"] = out[f"geometry_{interval}"].area
    out.to_feather(cache_path)

    return out
        

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
    import rasterio.vrt
    neff_model = get_neff_model()

    outlines_path = CACHE_DIR / "rgi7_outlines.arrow"
    if not outlines_path.is_file():
        make_outlines()

    outlines = gpd.read_feather(outlines_path)
    outlines["id"] = outlines["rgi_id"].str.split("-", expand=True).iloc[:, -1].astype(int)


    glacier_zones = gpd.read_file("shapes/glacier_zones.geojson").to_crs(outlines.crs).set_index("zone_label")
    outlines = gpd.sjoin(outlines, glacier_zones).reset_index(drop=True)

    date = "20260424"
    intervals = [
        ("13_18", "slope", "2013-2018", "geometry_13_18"),
        ("19_24", "slope_tcorr", "2019-2024", "geometry_19_24"),
        ("13_24", "slope", "2013-2024", "geometry_13_24"),
    ]

    rasterized_by_short = {}
    periglacial_by_short = {}
    res_by_short = {}
    data_by_short = {}
    base_keys = []

    for short, kind, long, geom_col in intervals:
        base_key = f"{kind.replace('_tcorr', '')}_{short}"
        base_keys.append(base_key)
        paths = {
            base_key: Path(f"input/trend_{long}_{kind}_{date}.tif"),
        }
        # paths[base_key + "_spatial_err_unscaled"] = paths[base_key + "_temporal_err"].with_stem(paths[base_key + "_temporal_err"].stem.replace("temporal", "spatial").replace("_tcorr", ""))
        paths[base_key + "_spatial_err_unscaled"] = paths[base_key].parent / f"trend_{long}_{kind.replace('_tcorr', '')}_spatial_err.tif"
        paths[base_key + "_temporal_err"] = paths[base_key].parent / f"trend_{long}_{kind}_temporal_err.tif"

        data_by_short[short] = {}

        with rasterio.open(paths[base_key]) as raster:
            sampling_bounds = raster.bounds
            rasterized = rasterio.features.rasterize(
                [(geom, outline_id) for geom, outline_id in zip(outlines[geom_col], outlines["id"]) if geom is not None and not geom.is_empty],
                out_shape=(raster.height, raster.width),
                fill=0,
                transform=raster.transform,
            )
            periglacial = rasterized == 0
            rasterized_by_short[short] = rasterized[~periglacial]
            periglacial_by_short[short] = periglacial
            res_by_short[short] = raster.res[0]

            transform = raster.transform
            shape = rasterized.shape

        for key, filepath in paths.items():
            with rasterio.open(filepath) as raster:
                scale = raster.scales[0]
            with rasterio.open(filepath, overview_level=3 if "2026" not in filepath.stem else None) as orig_raster:

                with rasterio.vrt.WarpedVRT(orig_raster, transform=transform, height=rasterized.shape[0], width=rasterized.shape[1]) as raster:
                    # window = rasterio.windows.from_bounds(*sampling_bounds, transform=raster.transform)
                    data_by_short[short][key] = (raster.read(1, masked=True)[~periglacial_by_short[short]].astype("float32") * scale).filled(np.nan)

    for short, _, _, _ in intervals:
        outlines[f"area_{short}"] = np.nan
        outlines[f"neff_{short}"] = np.nan

    # i = 0
    for idx, outline in tqdm.tqdm(outlines.iterrows(), total=outlines.shape[0]):
        # i+= 1
        # if i > 50:
        #     break
        for short, _, _, _ in intervals:
            mask = rasterized_by_short[short] == outline["id"]
            area = np.count_nonzero(mask) * res_by_short[short] ** 2
            if area == 0:
                continue

            outlines.loc[idx, f"area_{short}"] = area
            outlines.loc[idx, f"neff_{short}"] = neff_model(area)

            for key, arr in data_by_short[short].items():
                if "temporal_err" in key:
                    outlines.loc[idx, key] = np.nanmean(
                        np.sqrt(
                            np.clip(
                                (arr[mask] ** 2) - (data_by_short[short][key.replace("temporal_err", "spatial_err_unscaled")][mask] ** 2),
                                a_min=0,
                                a_max=np.inf,
                            )
                        )
                    )
                else:
                    outlines.loc[idx, key] = np.nanmean(arr[mask])

                if "slope" in key and "_err" not in key:
                    outlines.loc[idx, f"{key}_positive_vol"] = np.nansum(arr[mask][arr[mask] > 0.]) * res_by_short[short] ** 2
                    outlines.loc[idx, f"{key}_positive_area"] = np.count_nonzero(arr[mask] > 0.1) * res_by_short[short] ** 2


    for key in base_keys:
        outlines[[f"{key}_spatial_err_unscaled", f"{key}_temporal_err"]] *= 2 # Convert to 2sigma


    outlines["accel_13_24"] = (outlines["slope_19_24"] - outlines["slope_13_18"]) / 6
    for err_col in ["spatial_err_unscaled", "temporal_err"]:
        outlines[f"accel_13_24_{err_col}"] = np.hypot(outlines[f"slope_13_18_{err_col}"], outlines[f"slope_19_24_{err_col}"])


    key_to_interval = lambda s: "_".join(s.split("_")[-2:])
    for key in ["accel_13_24", *base_keys]:
        outlines[f"{key}_spatial_err"] = outlines[f"{key}_spatial_err_unscaled"] / (outlines[f"neff_{key_to_interval(key)}"] ** 0.5)

        outlines[f"{key}_err"] = np.hypot(outlines[f"{key}_spatial_err"], outlines[f"{key}_temporal_err"])

        for suffix in ["", "_err", "_spatial_err", "_spatial_err_unscaled", "_temporal_err"]:
            outlines[key + "_vol" + suffix] = outlines[key + suffix] * outlines[f"area_{key_to_interval(key)}"]
            
        # if not pd.isna(outlines.loc[idx, "slope_13_18"]) and not pd.isna(outlines.loc[idx, "slope_19_24"]):
        #     outlines.loc[idx, "accel_13_24"] = (outlines.loc[idx, "slope_19_24"] - outlines.loc[idx, "slope_13_18"]) / 6
        # if not pd.isna(outlines.loc[idx, "slope_13_18_spatial_err_unscaled"]) and not pd.isna(outlines.loc[idx, "slope_19_24_spatial_err_unscaled"]):
        #     outlines.loc[idx, "accel_13_24_spatial_err_unscaled"] = np.hypot(outlines.loc[idx, "slope_19_24_spatial_err_unscaled"], outlines.loc[idx, "slope_13_18_spatial_err_unscaled"]) / 6
        #     outlines.loc[idx, "accel_13_24_temporal_err"] = np.hypot(outlines.loc[idx, "slope_19_24_temporal_err"], outlines.loc[idx, "slope_13_18_temporal_err"]) / 6
        #     outlines.loc[idx, "accel_13_24_spatial_err"] = outlines.loc[idx, "accel_13_24_spatial_err_unscaled"] / (outlines.loc[idx, "neff_13_24"] ** 0.5)

        # if not pd.isna(outlines.loc[idx, "accel_13_24"]):
        #     outlines.loc[idx, "accel_13_24_positive_vol"] = outlines.loc[idx, "accel_13_24"] * outlines.loc[idx, "area_13_24"]


    outlines.to_feather(cache_path)
    return gpd.read_feather(cache_path)


def plot_regional_dhdt_fig(all_changes, show: bool = True):
    glacier_zones = refine_glacier_zones()
    all_params = [
        {
            "xcol": "slope_start_mid",
            "ycol": "slope_mid_end",
            "partition": "all",
            "unit": "elev_rate",
            "vlim": [-1.55, 0.2],
            "lessneg_text_xy": (0.68, 0.82),
            "moreneg_text_xy": (0.87, 0.68),
            "out_stem": "perzone_elevation_change",
        },
        {
            "xcol": "slope_start_mid",
            "ycol": "slope_mid_end",
            "unit": "elev_rate",
            "partition": "nonsurging",
            "vlim": [-1.5, 0.2],
            "lessneg_text_xy": (0.68, 0.82),
            "moreneg_text_xy": (0.87, 0.68),
            "out_stem": "perzone_elevation_change_nonsurging",
        },
        {
            "xcol": "slope_start_mid",
            "ycol": "slope_mid_end",
            "unit": "vol_rate",
            "partition": "all",
            "vlim": [-10.5, 1],
            "lessneg_text_xy": (0.68, 0.82),
            "moreneg_text_xy": (0.87, 0.68),
            "out_stem": "perzone_volume_change",
        },
        {
            "xcol": "slope_start_mid",
            "ycol": "slope_mid_end",
            "unit": "vol_rate",
            "partition": "nonsurging",
            "vlim": [-6, 1],
            "lessneg_text_xy": (0.68, 0.82),
            "moreneg_text_xy": (0.87, 0.68),
            "out_stem": "perzone_volume_change_nonsurging",
        },
    ]

    for params in all_params:
        fig = plt.figure(figsize=(5, 4.9))

        interval_translation = {
            "start_end": "2013-2024",
            "start_mid": "2013-2018",
            "mid_end": "2019-2024",
        }

        xcol_interval = interval_translation[params["xcol"].replace("slope_", "")]
        ycol_interval = interval_translation[params["ycol"].replace("slope_", "")]
        unit =all_changes["units"][params["unit"]]
        axis_label = {"vol_rate": "Volume change rate", "elev_rate": "Elevation change rate"}.get(params["unit"])
        plt.title(f"Regional {axis_label.lower()} " + ("(non-surging)" if "nonsurging" in params["out_stem"] else ""))
        inset = plt.gca().inset_axes([0., 0.5, 0.4, 0.5])
        glacier_zones.plot(color=glacier_zones["color"], ax=inset)

        xdata = all_changes["changes"][params["xcol"]][params["partition"]]["per_zone"]
        ydata = all_changes["changes"][params["ycol"]][params["partition"]]["per_zone"]
        for i, (label, zone_xdata) in enumerate(xdata.items()):
            zone = glacier_zones.loc[label]
            zone_ydata = ydata[label]
            plt.errorbar(
                x=zone_xdata[params["unit"]],
                y=zone_ydata[params["unit"]],
                xerr=zone_xdata[params["unit"] + "_err"],
                yerr=zone_ydata[params["unit"] + "_err"],
                color=np.array(matplotlib.colors.to_rgb(zone["color"])) * 0.6,
                marker="o",
                markersize=0.8 * zone_xdata["area"] / 1e2,
                markerfacecolor=zone["color"],
                markeredgecolor="#ccc",
                barsabove=True,
                alpha=1.0,
                zorder=i + 1,
            )

            # xy_text = (zone[params["xcol"] + params["col_suffix"]], zone[params["ycol"] + params["col_suffix"]])
            xy_text = (zone_xdata[params["unit"]], zone_ydata[params["unit"]])

            # if label == "NW":
            #     xy_text = (xy_text[0] - 0.35, xy_text[1] - 0.28)
            # elif label == "N":
            #     xy_text = (xy_text[0] - 0., xy_text[1] - 0.15)
            

            text_kwargs = {"ha": "center", "va": "center", "path_effects":[matplotlib.patheffects.withStroke(foreground="black", linewidth=1)], "color": "white"}
            plt.annotate(label,xy_text,zorder=i + 300, **text_kwargs)
            inset.annotate(label, (zone.geometry.centroid.x, zone.geometry.centroid.y), **text_kwargs)


        plt.fill_between(params["vlim"], params["vlim"], [max(params["vlim"])] * 2, color=DHDT_SM.to_rgba(1), alpha=0.2)  
        plt.text(*params["lessneg_text_xy"], "Less\nnegative", transform=plt.gca().transAxes, color=np.array(DHDT_SM.to_rgba(1)[:3]) * 0.7, ha="center", fontsize=12)
        plt.text(*params["moreneg_text_xy"], "More\nnegative", transform=plt.gca().transAxes, color=np.array(DHDT_SM.to_rgba(-1)[:3]) * 0.7, ha="center", fontsize=12)
        plt.fill_between(params["vlim"], params["vlim"], [min(params["vlim"])] * 2, color=DHDT_SM.to_rgba(-1), alpha=0.2)  
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
        if show:
            plt.show()
        plt.close()


def refine_glacier_zones():
    cache_path = CACHE_DIR / "refined_glacier_zones.arrow"

    if cache_path.is_file():
        return gpd.read_feather(cache_path)

    glacier_zones = gpd.read_file("shapes/glacier_zones.geojson").set_index("zone_label")

    coasts = make_coastline_intervals().loc[["13_24"]].explode().to_crs(glacier_zones.crs)
    coasts = coasts[coasts.geometry.area > 1e8]
    coasts = coasts.dissolve().simplify(100)

    glacier_zones.geometry = glacier_zones.geometry.intersection(coasts.geometry[0])

    glacier_zones.to_feather(cache_path)
    return gpd.read_feather(cache_path)
    

def get_statistics():

    outlines = sample_rasters()
    to_v_keys = ["slope_13_24", "slope_13_18", "slope_19_24", "accel_13_24"]
    key_to_interval = lambda s: "_".join(s.split("_")[-2:])

    # Tinkarpbreen
    outlines.loc[outlines["rgi_id"] == "RGI2000-v7.0-G-07-00560", ["surging_13_18", "surging_13_24", "surging_19_24"]] = True

    glacier_zones = refine_glacier_zones()
        
    neff_model = get_neff_model()

    all_stats = {
            "units": {
                "vol_rate": "km$^{3}$~a$^{-1}$",
                "vol_accel": "km$^{3}$~a$^{-2}$",
                "elev_rate": "m~a$^{-1}$"
            },
            "parameters": {
                "positive_change_threshold": 0.1,
            },
        }


    all_stats["changes"] = {}
    for key in to_v_keys:
        interval = key_to_interval(key)

        vol_col = key + "_vol"
        err_col = key + "_err"
        vol_err_col = key + "_vol_err"

        changes = {}
        for partition, query in [("all", ""), ("surging", f"surging_{interval}"), ("nonsurging", f"~surging_{interval}")]:

            for zone_label in ["allzones", *glacier_zones.index]:
                if partition == "all":
                    subset = outlines
                else:
                    subset = outlines.query(query)
                if zone_label != "allzones":
                    subset = subset.query(f"zone_label == '{zone_label}'")

                total_area = np.max([subset[f"area_{interval}"].sum(), 1e-5])

                vol_rate = subset[vol_col].sum()
                # vol_rate_err = subset[vol_err_col].sum() / (neff_model(total_area) ** 0.5)
                vol_rate_err = max(subset[vol_col + "_spatial_err_unscaled"].sum() / (neff_model(total_area) ** 0.5), subset[vol_col + "_temporal_err"].sum()) 

                new_changes = {
                    "vol_rate": vol_rate / 1e9,
                    "vol_rate_err": vol_rate_err / 1e9,
                    "area": total_area / 1e6,
                    "elev_rate": vol_rate / total_area,
                    "elev_rate_err": vol_rate_err / total_area,
                }
                if "accel" not in key:
                    new_changes["positive_vol"]= subset[f"{key}_positive_vol"].sum() / 1e9
                    new_changes["positive_area"]= subset[f"{key}_positive_area"].sum() / 1e6

                    new_changes["positive_area_frac"] =round (100* new_changes["positive_area"] / new_changes["area"])

                if partition != "all" or zone_label != "allzones": 
                    denom = "all" if zone_label == "allzones" else partition
                    new_changes["vol_rate_percent"] = round(100 * new_changes["vol_rate"] / changes[denom]["vol_rate"])
                    new_changes["area_percent"] = round(100 * new_changes["area"] / changes[denom]["area"])

                if zone_label == "allzones":
                    changes[partition] = new_changes
                else:
                    if "per_zone" not in changes[partition]:
                        changes[partition]["per_zone"] = {}

                    changes[partition]["per_zone"][zone_label] = new_changes
        all_stats["changes"][key.replace("13", "start").replace("18", "mid").replace("19", "mid").replace("24", "end")] = changes


    record_information(all_stats) # type: ignore
    return all_stats


def make_figs(show: bool = False):
    changes_stats = get_statistics()
    outlines = sample_rasters()
    fig = plt.figure(figsize=(4, 3))
    for i, (issurging, items) in enumerate(outlines.groupby("surging_13_24")):

        xvals = np.full(items.shape[0], float(issurging))
        # xvals += np.random.default_rng(0).normal(scale=0.03, size=xvals.size)

        plt.violinplot([np.log10(np.clip(-items["slope_13_24_vol"], a_min=1e-6, a_max=np.inf))], positions=[float(issurging)])
        # plt.scatter(xvals, items["slope_13_24_vol"]) 

    plt.ylim(10, 3)
    plt.xticks([0, 1], ["Nonsurging", "Surging"])
    yticks = plt.gca().get_yticks()
    plt.yticks(yticks, labels=["10$^{" + str(int(ytick)) + "}$" for ytick in yticks])
    plt.ylabel("Volume change rate (km³ / a)")
    plt.tight_layout()
    plt.savefig("figures/surging_vs_nonsurging_vol_violin.svg")
    plt.close()
    
    
    plot_regional_dhdt_fig(all_changes=changes_stats, show=False)
    
    fig = plt.figure(figsize=(4, 3))
    axes: list[plt.Axes] = fig.subplots(2, 1, sharex=True, sharey=False).ravel().tolist() # type: ignore
    fig.subplots_adjust(left=0.135, bottom=0.165, right=0.995, top=0.98, hspace=0.1)

    for i, (issurging, items) in enumerate(outlines.groupby("surging_13_24")):
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

    
    
def hypsometric(show: bool = True):
    import rasterio as rio
    import rasterio.features

    outlines = sample_rasters()
    zone_meta = outlines[["zone_label", "zone_name"]].drop_duplicates().set_index("zone_label")
    zone_ids = {label: i + 1 for i, label in enumerate(zone_meta.index)}
    id_to_label = {i: label for label, i in zone_ids.items()}

    elev_bins = np.linspace(-0.01, 1200.01, 11)
    elev_bin_centers = (elev_bins[1:] + elev_bins[:-1]) / 2

    @dataclasses.dataclass
    class Config:
        interval_name: str
        short: str
        filepath: Path
        color: np.ndarray

    intervals = [
        Config("2013-2018", "13_18", Path("input/trend_2013-2018_slope.tif"), DHDT_SM.to_rgba(1.)),
        Config("2019-2024", "19_24", Path("input/trend_2019-2024_slope.tif"), DHDT_SM.to_rgba(-1.)),
    ]
    intercept_path = "input/trend_2013-2024_intercept.tif"

    with rio.open(intercept_path) as dem_raster_full:
        scale = dem_raster_full.scales[0]  # Only the non-overview band has this information
    with rio.open(intercept_path, overview_level=3) as dem_raster:
        dem_arr = (dem_raster.read(1, masked=True).astype("float32") * scale).filled(0)
        dem_transform = dem_raster.transform

    per_zone = {}
    for config in intervals:
        geom_col = f"geometry_{config.short}"
        nonsurging = outlines.loc[~outlines[f"surging_{config.short}"] & outlines[geom_col].notna()].copy()
        nonsurging = nonsurging[~nonsurging[geom_col].is_empty]

        with rio.open(config.filepath) as dhdt_raster_full:
            scale = dhdt_raster_full.scales[0] # Only the non-overview band has this information
        with rio.open(config.filepath, overview_level=3) as dhdt_raster:
            zone_raster = rasterio.features.rasterize(
                [(geom, zone_ids[label]) for geom, label in zip(nonsurging[geom_col], nonsurging["zone_label"])],
                out_shape=dem_arr.shape,
                transform=dem_transform,
                fill=0,
                dtype="uint8",
            )
            valid = zone_raster > 0
            dhdt_arr = (dhdt_raster.read(1, masked=True).astype("float32") * scale).filled(np.nan)
            if dhdt_arr.shape != dem_arr.shape:
                raise RuntimeError(f"Hypsometric raster shape mismatch for {config.interval_name}")

        frame = pd.DataFrame(
            {
                "zone_label": [id_to_label[i] for i in zone_raster[valid]],
                "elev_idx": np.digitize(dem_arr, elev_bins)[valid],
                "dhdt": dhdt_arr[valid],
            }
        )
        frame = frame[(frame["elev_idx"] >= 1) & (frame["elev_idx"] <= elev_bin_centers.shape[0])]

        zone_results = {}
        for label in zone_meta.index:
            subset = frame[frame["zone_label"] == label]
            vals = []
            for idx in np.unique(subset["elev_idx"]):
                vals_here = subset.loc[subset["elev_idx"] == idx, "dhdt"].to_numpy(dtype=float)
                if vals_here.size == 0:
                    continue
                med = np.nanmedian(vals_here)
                vals.append(
                    {
                        "elevation": elev_bin_centers[idx - 1],
                        f"{config.short}_med": med,
                        f"{config.short}_nmad": 1.4826 * np.nanmedian(np.abs(vals_here - med)),
                    }
                )
            zone_results[label] = (
                pd.DataFrame.from_records(vals).set_index("elevation")
                if vals
                else pd.DataFrame(columns=[f"{config.short}_med", f"{config.short}_nmad"]).set_index(pd.Index([], name="elevation"))
            )
        per_zone[config.short] = zone_results

    fig = plt.figure(figsize=(8, 4))
    ncols = int(np.ceil(len(zone_meta) / 2))
    axes = np.atleast_1d(fig.subplots(2, ncols, sharex=True, sharey=True)).ravel()
    for col, (label, zone) in enumerate(zone_meta.iterrows()):
        axis: plt.Axes = axes[col]
        axis.set_title(str(zone["zone_name"]))
        for config in intervals:
            data = per_zone[config.short][label]
            if data.empty:
                continue
            axis.errorbar(
                data.index,
                data[f"{config.short}_med"],
                data[f"{config.short}_nmad"],
                color=config.color,
                alpha=0.5,
                fmt="none",
            )
            axis.scatter(
                data.index,
                data[f"{config.short}_med"],
                marker="s",
                edgecolor="#777",
                color=config.color,
                label=config.interval_name,
            )

        xlim = axis.get_xlim()
        axis.hlines(0, *xlim, color="#777", linestyles="--", alpha=0.5)
        axis.set_xlim(xlim)

        if col in [0, 4]:
            axis.set_ylabel("dH / dt (m a$^{-1}$)")

        if col >= ncols:
            axis.set_xlabel("Elevation (m a.s.l.)")

        if (col + 1) == len(zone_meta):
            axis.legend(loc="lower right")

    for axis in axes[len(zone_meta):]:
        axis.set_visible(False)

    plt.ylim(-3, 0.8)
    plt.subplots_adjust(top=0.937, bottom=0.121, left=0.073, right=0.991, hspace=0.219, wspace=0.061)
    plt.savefig("figures/perzone_hypsometric_signal.svg")

    if show:
        plt.show()

            

        
        


        



if __name__ == "__main__":
    make_outlines()
