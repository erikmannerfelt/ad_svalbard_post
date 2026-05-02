from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CACHE_DIR
from . import outlines, tools


def get_neff_model():
    import scipy.interpolate

    neff_data = pd.read_csv("input/vgm_neff_cirq_numerical.csv", index_col=0).squeeze()
    model = scipy.interpolate.interp1d(neff_data.index, neff_data, fill_value="extrapolate")

    return lambda a: model(np.clip(a, a_min=0, a_max=1.2e8))


def sample_rasters(redo: bool = False) -> gpd.GeoDataFrame:
    cache_path = CACHE_DIR / "outlines_sampled.arrow"

    if cache_path.is_file() and not redo:
        return gpd.read_feather(cache_path)

    import rasterio
    import rasterio.features
    import rasterio.vrt

    neff_model = get_neff_model()

    outlines_df=  outlines.make_outlines()
    outlines_df["id"] = outlines_df["rgi_id"].str.split("-", expand=True).iloc[:, -1].astype(int)
    glacier_zones = gpd.read_file("shapes/glacier_zones.geojson").to_crs(outlines_df.crs).set_index("zone_label")
    outlines_df = gpd.sjoin(outlines_df, glacier_zones).reset_index(drop=True)

    date = "20260424"
    interval_raster_kinds = [
        (tools.get_interval("13_18"), "slope"),
        (tools.get_interval("19_24"), "slope_tcorr"),
        (tools.get_interval("13_24"), "slope"),
    ]
    rasterized_by_short = {}
    periglacial_by_short = {}
    res_by_short = {}
    data_by_short = {}
    base_keys = []

    for interval, kind in interval_raster_kinds:
        base_key = f"{kind.replace('_tcorr', '')}_{interval.short}"
        base_keys.append(base_key)

        paths = {
            base_key: Path(f"input/trend_{interval.filename_label}_{kind}_{date}.tif"),
        }
        paths[base_key + "_spatial_err_unscaled"] = paths[base_key].parent / f"trend_{interval.filename_label}_{kind.replace('_tcorr', '')}_spatial_err.tif"
        paths[base_key + "_temporal_err"] = paths[base_key].parent / f"trend_{interval.filename_label}_{kind}_temporal_err.tif"
        data_by_short[interval.short] = {}

        with rasterio.open(paths[base_key]) as raster:
            rasterized = rasterio.features.rasterize(
                [(geom, outline_id) for geom, outline_id in zip(outlines_df[interval.geometry_col], outlines_df["id"]) if geom is not None and not geom.is_empty],
                out_shape=(raster.height, raster.width),
                fill=0,
                transform=raster.transform,
            )
            periglacial = rasterized == 0
            rasterized_by_short[interval.short] = rasterized[~periglacial]
            periglacial_by_short[interval.short] = periglacial
            res_by_short[interval.short] = raster.res[0]
            transform = raster.transform

        for key, filepath in paths.items():
            with rasterio.open(filepath) as raster:
                scale = raster.scales[0]
            with rasterio.open(filepath, overview_level=3 if "2026" not in filepath.stem else None) as orig_raster:
                with rasterio.vrt.WarpedVRT(orig_raster, transform=transform, height=rasterized.shape[0], width=rasterized.shape[1]) as raster:
                    data_by_short[interval.short][key] = (raster.read(1, masked=True)[~periglacial_by_short[interval.short]].astype("float32") * scale).filled(np.nan)

    for interval, _ in interval_raster_kinds:
        outlines_df[f"area_{interval.short}"] = np.nan
        outlines_df[f"neff_{interval.short}"] = np.nan

    for idx, outline in outlines_df.iterrows():
        for interval, _ in interval_raster_kinds:
            mask = rasterized_by_short[interval.short] == outline["id"]
            area = np.count_nonzero(mask) * res_by_short[interval.short] ** 2
            if area == 0:
                continue

            outlines_df.loc[idx, f"area_{interval.short}"] = area
            outlines_df.loc[idx, f"neff_{interval.short}"] = neff_model(area)

            for key, arr in data_by_short[interval.short].items():
                if "temporal_err" in key:
                    outlines_df.loc[idx, key] = np.nanmean(np.sqrt(np.clip((arr[mask] ** 2) - (data_by_short[interval.short][key.replace("temporal_err", "spatial_err_unscaled")][mask] ** 2), a_min=0, a_max=np.inf)))
                else:
                    outlines_df.loc[idx, key] = np.nanmean(arr[mask])
                if "slope" in key and "_err" not in key:
                    outlines_df.loc[idx, f"{key}_positive_vol"] = np.nansum(arr[mask][arr[mask] > 0.]) * res_by_short[interval.short] ** 2
                    outlines_df.loc[idx, f"{key}_positive_area"] = np.count_nonzero(arr[mask] > 0.1) * res_by_short[interval.short] ** 2

    for key in base_keys:
        outlines_df[[f"{key}_spatial_err_unscaled", f"{key}_temporal_err"]] *= 2

    outlines_df["accel_13_24"] = (outlines_df["slope_19_24"] - outlines_df["slope_13_18"]) / 6
    for err_col in ["spatial_err_unscaled", "temporal_err"]:
        outlines_df[f"accel_13_24_{err_col}"] = np.hypot(outlines_df[f"slope_13_18_{err_col}"], outlines_df[f"slope_19_24_{err_col}"])

    for key in ["accel_13_24", *base_keys]:
        key_to_interval = tools.key_to_interval(key)
        outlines_df[f"{key}_spatial_err"] = outlines_df[f"{key}_spatial_err_unscaled"] / (outlines_df[f"neff_{key_to_interval}"] ** 0.5)
        outlines_df[f"{key}_err"] = np.hypot(outlines_df[f"{key}_spatial_err"], outlines_df[f"{key}_temporal_err"])

        for suffix in ["", "_err", "_spatial_err", "_spatial_err_unscaled", "_temporal_err"]:
            outlines_df[key + "_vol" + suffix] = outlines_df[key + suffix] * outlines_df[f"area_{key_to_interval}"]

    outlines_df.to_feather(cache_path)

    return gpd.read_feather(cache_path)
