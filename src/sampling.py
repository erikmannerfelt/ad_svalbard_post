from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import tqdm

from . import outlines, tools
from .config import CACHE_DIR, INPUT_DIR, POSITIVE_CHANGE_THRESHOLD


def get_neff_model():
    import scipy.interpolate

    # Written by `adsvalbard.variogram.main()` together with `variogram_model.csv`, so the
    # n_eff matches the variogram reported in the SI. The older table in `input/old/` came
    # from a superseded variogram and gave up to ~26x larger n_eff.
    neff_data = tools.read_aux_csv("vgm_neff_cirq.csv", index_col=0).squeeze()
    model = scipy.interpolate.interp1d(neff_data.index, neff_data)

    # The table spans 1 m2 to 35000 km2, which covers every glacier and the archipelago.
    return lambda a: model(np.clip(a, a_min=neff_data.index.min(), a_max=neff_data.index.max()))


def _nanmean_or_warn(values: np.ndarray, *, rgi_id: str, glac_name: str, interval_short: str, key: str) -> float:
    if values.size == 0:
        warnings.warn(f"Missing sampled values for {rgi_id} ({glac_name}) in {interval_short} for {key}; storing NaN.", RuntimeWarning, stacklevel=2)
        return float("nan")
    if np.isnan(values).all():
        warnings.warn(f"All sampled values are NaN for {rgi_id} ({glac_name}) in {interval_short} for {key}; storing NaN.", RuntimeWarning, stacklevel=2)
        return float("nan")
    return float(np.nanmean(values))


#: The rate raster each interval's coverage is measured against, matching the products
#: `sample_rasters()` samples. The suffix-less 2019-2024 raster is the temporally
#: bias-corrected one; the uncorrected variant is `trend_2019-2024_slope_uncorr.tif`.
_COVERAGE_RASTER_KIND: Final[dict[str, str]] = {
    "13_18": "slope",
    "19_24": "slope",
    "13_24": "slope",
}


def compute_coverage_fractions(overview_level: int | None = 2) -> dict[str, float]:
    """Fraction of each interval's glacierized area that carries a valid fitted rate.

    This is the `observational_coverage_percentage` of the GlaMBIE-2 submission: the
    share of pixels inside the period-specific outlines where the trend fit produced a
    value. It is derived rather than stored because it is cheap (seconds, at the default
    overview) and because a hand-maintained copy silently goes stale whenever the
    outlines change.

    Returns
    -------
    dict
        Percentages (0-100) keyed by `Interval.short`.

    Notes
    -----
    No interpolation or gap-filling is applied anywhere in the pipeline, so the
    uncovered remainder is implicitly assigned its own glacier's mean rate.
    """
    import rasterio
    import rasterio.features

    outlines_df = outlines.make_outlines()
    coverage: dict[str, float] = {}

    for interval in tools.iter_intervals():
        kind = _COVERAGE_RASTER_KIND[interval.short]
        path = INPUT_DIR / f"trend_{interval.filename_label}_{kind}.tif"
        with rasterio.open(path, overview_level=overview_level) as raster:
            inside = rasterio.features.rasterize(
                [
                    (geom, 1)
                    for geom in outlines_df[interval.geometry_col]
                    if geom is not None and not geom.is_empty
                ],
                out_shape=(raster.height, raster.width),
                fill=0,
                dtype="uint8",
                transform=raster.transform,
            ).astype(bool)
            valid = ~raster.read(1, masked=True).mask & inside

        total_px = int(inside.sum())
        if total_px == 0:
            raise RuntimeError(f"No glacierized pixels found for {interval.short} in {path.name}")
        coverage[interval.short] = 100.0 * int(valid.sum()) / total_px

    return coverage


def sample_rasters(redo: bool = False, overview_level: int | None = 2, use_tqdm: bool = True) -> gpd.GeoDataFrame:
    cache_path = CACHE_DIR / "outlines_sampled.arrow"

    if cache_path.is_file() and not redo:
        return gpd.read_feather(cache_path)

    import rasterio
    import rasterio.features
    import rasterio.windows

    neff_model = get_neff_model()

    outlines_df = outlines.make_outlines()
    outlines_df["id"] = outlines_df["rgi_id"].str.split("-", expand=True).iloc[:, -1].astype(int)

    interval_raster_kinds = [
        (tools.get_interval("13_18"), "slope"),
        (tools.get_interval("19_24"), "slope"),
        (tools.get_interval("13_24"), "slope"),
    ]
    rasterized_by_short = {}
    periglacial_by_short = {}
    res_by_short = {}
    data_by_short = {}
    base_keys = []

    for interval, kind in interval_raster_kinds:
        base_key = f"{kind}_{interval.short}"
        base_keys.append(base_key)

        paths = {
            base_key: Path(f"input/trend_{interval.filename_label}_{kind}.tif"),
        }
        paths[base_key + "_baseline_err_unscaled"] = paths[base_key].parent / f"trend_{interval.filename_label}_{kind}_baseline_err.tif"
        paths[base_key + "_fit_err"] = paths[base_key].parent / f"trend_{interval.filename_label}_{kind}_fit_err.tif"
        data_by_short[interval.short] = {}

        with rasterio.open(paths[base_key], overview_level=overview_level) as raster:
            rasterized = rasterio.features.rasterize(
                [(geom, outline_id) for geom, outline_id in zip(outlines_df[interval.geometry_col], outlines_df["id"]) if geom is not None and not geom.is_empty],
                out_shape=(raster.height, raster.width),
                fill=0,
                dtype="int32",
                transform=raster.transform,
            )
            periglacial = rasterized == 0
            rasterized_by_short[interval.short] = rasterized[~periglacial]
            periglacial_by_short[interval.short] = periglacial
            res_by_short[interval.short] = raster.res[0]
            crs = raster.crs
            base_shape = (raster.height, raster.width)
            base_bounds = raster.bounds

        for key, filepath in paths.items():
            # Overview reads lose the per-band scale metadata, so read scale from the full raster first.
            with rasterio.open(filepath) as full_raster:
                scale = full_raster.scales[0]
                full_crs = full_raster.crs

            with rasterio.open(filepath, overview_level=overview_level) as raster:
                if raster.crs != crs or raster.crs != full_crs:
                    raise RuntimeError(f"Raster CRS mismatch for {filepath.name}")
                if key == base_key:
                    data = raster.read(1, masked=True)
                else:
                    window = rasterio.windows.from_bounds(*base_bounds, transform=raster.transform)
                    window = window.round_offsets().round_lengths()
                    data = raster.read(1, window=window, masked=True)
                    if data.shape != base_shape:
                        raise RuntimeError(f"Raster window shape mismatch for {filepath.name}: expected {base_shape}, found {data.shape}")
                data_by_short[interval.short][key] = (data[~periglacial_by_short[interval.short]].astype("float32") * scale).filled(np.nan)

    for interval, _ in interval_raster_kinds:
        outlines_df[f"area_{interval.short}"] = np.nan
        outlines_df[f"neff_{interval.short}"] = np.nan

    for idx, outline in tqdm.tqdm(outlines_df.iterrows(), total=outlines_df.shape[0], desc="Applying sampled raster data", disable=not use_tqdm):
        for interval, _ in interval_raster_kinds:
            mask = rasterized_by_short[interval.short] == outline["id"]
            area = np.count_nonzero(mask) * res_by_short[interval.short] ** 2
            if area == 0:
                continue

            outlines_df.loc[idx, f"area_{interval.short}"] = area
            outlines_df.loc[idx, f"neff_{interval.short}"] = neff_model(area)

            interval_data = data_by_short[interval.short]
            baseline_key = next(key for key in interval_data if key.endswith("_baseline_err_unscaled"))
            fit_key = next(key for key in interval_data if key.endswith("_fit_err"))

            for key, arr in interval_data.items():
                values = arr[mask]
                if key == fit_key:
                    baseline_values = interval_data[baseline_key][mask]
                    excess_values = np.sqrt(np.clip((values ** 2) - (baseline_values ** 2), a_min=0, a_max=np.inf))
                    outlines_df.loc[idx, key.replace("fit_err", "excess_err")] = _nanmean_or_warn(
                        excess_values,
                        rgi_id=outline["rgi_id"],
                        glac_name=outline.get("glac_name", ""),
                        interval_short=interval.short,
                        key=key.replace("fit_err", "excess_err"),
                    )
                outlines_df.loc[idx, key] = _nanmean_or_warn(
                    values,
                    rgi_id=outline["rgi_id"],
                    glac_name=outline.get("glac_name", ""),
                    interval_short=interval.short,
                    key=key,
                )
                if "slope" in key and "_err" not in key:
                    # Volume and area share the threshold, so they describe the same pixels.
                    positive = values > POSITIVE_CHANGE_THRESHOLD
                    outlines_df.loc[idx, f"{key}_positive_vol"] = np.nansum(values[positive]) * res_by_short[interval.short] ** 2
                    outlines_df.loc[idx, f"{key}_positive_area"] = np.count_nonzero(positive) * res_by_short[interval.short] ** 2

    # for key in base_keys:
    #     outlines_df[[f"{key}_baseline_err_unscaled", f"{key}_fit_err", f"{key}_excess_err"]] *= 2

    # The sub-period midpoints are six years apart. The errors must be divided by the
    # same six as the value, or the acceleration uncertainty is inflated sixfold.
    outlines_df["accel_13_24"] = (outlines_df["slope_19_24"] - outlines_df["slope_13_18"]) / 6
    for err_col in ["baseline_err_unscaled", "fit_err", "excess_err"]:
        outlines_df[f"accel_13_24_{err_col}"] = np.hypot(outlines_df[f"slope_13_18_{err_col}"], outlines_df[f"slope_19_24_{err_col}"]) / 6

    for key in ["accel_13_24", *base_keys]:
        key_to_interval = tools.key_to_interval(key)
        outlines_df[f"{key}_baseline_err"] = outlines_df[f"{key}_baseline_err_unscaled"] / (outlines_df[f"neff_{key_to_interval}"] ** 0.5)
        outlines_df[f"{key}_err"] = np.hypot(outlines_df[f"{key}_baseline_err"], outlines_df[f"{key}_excess_err"])

        for suffix in ["", "_err", "_fit_err", "_baseline_err", "_baseline_err_unscaled", "_excess_err"]:
            outlines_df[key + "_vol" + suffix] = outlines_df[key + suffix] * outlines_df[f"area_{key_to_interval}"]

    outlines_df.to_feather(cache_path)

    return gpd.read_feather(cache_path)
