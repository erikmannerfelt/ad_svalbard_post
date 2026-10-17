from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from . import (
    comparisons,
    coreg,
    coverage,
    gap_classifier,
    outlines,
    positive_area,
    reporting,
    sampling,
    submarine,
    tools,
)
from .config import POSITIVE_CHANGE_THRESHOLD

_PER_GLACIER_EXPORTS = [
    {"key": "storisstraumen", "glac_name": "Storisstraumen", "surging_13_24": True},
    {"key": "negribreen", "glac_name": "Negribreen", "surging_13_24": True},
    {"key": "stonebreen", "glac_name": "Stonebreen", "surging_13_24": True},
    {"key": "kvitoyjokulen", "glac_name": "Kvitøyjøkulen", "surging_13_24": False},
    {"key": "brasvellbreen", "glac_name": "Bråsvellbreen", "surging_13_24": False},
]


def _per_glacier_stats(row: pd.Series, key: str) -> dict[str, float]:
    interval = tools.key_to_interval(key)
    out = {
        "vol_rate": float(row[f"{key}_vol"] / 1e9),
        "vol_rate_err": float(row[f"{key}_vol_err"] / 1e9),
        "area": float(row[f"area_{interval}"] / 1e6),
        "elev_rate": float(row[key]),
        "elev_rate_err": float(row[f"{key}_err"]),
    }
    return out


def _validate_per_glacier_exports(sampled: pd.DataFrame) -> None:
    ranked_surging = (
        sampled.loc[sampled["surging_13_24"]]
        .sort_values("slope_13_24_vol", ascending=True)["glac_name"]
        .head(3)
        .tolist()
    )
    ranked_nonsurging = (
        sampled.loc[~sampled["surging_13_24"]]
        .sort_values("slope_13_24_vol", ascending=True)["glac_name"]
        .head(2)
        .tolist()
    )

    expected_surging = [item["glac_name"] for item in _PER_GLACIER_EXPORTS[:3]]
    expected_nonsurging = [item["glac_name"] for item in _PER_GLACIER_EXPORTS[3:]]

    if ranked_surging != expected_surging:
        raise RuntimeError(
            f"Unexpected top 2013-2024 surging volume-loss glaciers: expected {expected_surging}, found {ranked_surging}"
        )
    if ranked_nonsurging != expected_nonsurging:
        raise RuntimeError(
            f"Unexpected top 2013-2024 nonsurging volume-loss glaciers: expected {expected_nonsurging}, found {ranked_nonsurging}"
        )


def _build_per_glacier_exports(
    sampled: pd.DataFrame, key: str
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for spec in _PER_GLACIER_EXPORTS:
        slug = spec["key"]
        row_matches = sampled.loc[sampled["glac_name"].eq(spec["glac_name"])]
        if row_matches.empty:
            raise RuntimeError(f"Could not find glacier row for {spec['glac_name']}")
        if row_matches.shape[0] > 1:
            raise RuntimeError(
                f"Glacier name {spec['glac_name']} is not unique; cannot export per-glacier stats safely"
            )

        row = row_matches.iloc[0]
        if bool(row["surging_13_24"]) != bool(spec["surging_13_24"]):
            raise RuntimeError(f"Unexpected surging state for {spec['glac_name']}")

        out[slug] = _per_glacier_stats(row, key)
    return out


def _histogram_mode(values: pd.Series | np.ndarray, bins: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    counts, edges = np.histogram(arr, bins=bins)
    idx = int(np.argmax(counts))
    return float((edges[idx] + edges[idx + 1]) / 2)


def read_variogram_model() -> pd.DataFrame:
    return tools.read_aux_csv("variogram_model.csv")


def read_empirical_variogram() -> pd.DataFrame:
    return tools.read_aux_csv("empirical_variogram.csv", index_col=0)


# Adapted from https://github.com/mmaelicke/scikit-gstat/blob/c717c15ecdb5ab258efe306850e6c8602044d24b/skgstat/models.py
def variogram_spherical(h: np.ndarray, r: float, c0: float) -> np.ndarray:
    h_arr = np.asarray(h, dtype=float)
    a = r / 1.0
    ratio = h_arr / a
    return np.where(h_arr <= r, c0 * ((1.5 * ratio) - (0.5 * ratio**3.0)), c0)


# Adapted from https://github.com/mmaelicke/scikit-gstat/blob/c717c15ecdb5ab258efe306850e6c8602044d24b/skgstat/models.py
def variogram_gaussian(h: np.ndarray, r: float, c0: float) -> np.ndarray:
    h_arr = np.asarray(h, dtype=float)
    a = r / 2.0
    return c0 * (1.0 - np.exp(-(h_arr**2 / a**2)))


# Adapted from https://github.com/GlacioHack/xdem/blob/v0.0.13/xdem/spatialstats.py#L1574-L1609
def make_variogram_model(params: pd.DataFrame | None = None):
    model_funs = {
        "spherical": variogram_spherical,
        "gaussian": variogram_gaussian,
    }

    if params is None:
        params = read_variogram_model()

    def vgm_model(h):
        h_arr = np.asarray(h, dtype=float)
        out = np.zeros_like(h_arr, dtype=float)
        for _, row in params.iterrows():
            out += model_funs[str(row["model"])](
                h_arr, float(row["range"]), float(row["psill"])
            )
        return out

    return vgm_model


def variogram_correlation(h, params: pd.DataFrame | None = None):
    if params is None:
        params = read_variogram_model()
    vgm_model = make_variogram_model(params)
    sill = float(params["psill"].sum())
    return 1.0 - vgm_model(h) / sill


def _format_distance(distance_m: float) -> str:
    if distance_m < 1000:
        return f"{int(np.rint(distance_m))}~m"
    return f"{distance_m / 1000:.1f}~km"


def variogram_correlation_breakpoints(
    params: pd.DataFrame | None = None,
) -> dict[str, str]:
    if params is None:
        params = read_variogram_model()

    h_max = max(50000.0, float(params["range"].max()) * 10.0)
    hs = np.geomspace(1.0, h_max, 50000)
    corr = np.asarray(variogram_correlation(hs, params=params), dtype=float)

    thresholds = {"seventyfive": 0.75, "twentyfive": 0.25, "five": 0.05}
    out: dict[str, str] = {}
    for key, threshold in thresholds.items():
        idx = np.flatnonzero(corr <= threshold)
        if idx.size == 0:
            out[key] = _format_distance(float(hs[-1]))
        else:
            out[key] = _format_distance(float(hs[int(idx[0])]))
    return out


def _get_variogram_statistics() -> dict[str, object]:
    model = read_variogram_model()
    component_names = ["first", "second", "third"]
    model_name_map = {"spherical": "spherical", "gaussian": "Gaussian"}

    variogram = {}
    for component_name, (_, row) in zip(
        component_names, model.iterrows(), strict=False
    ):
        variogram[component_name] = {
            "model": model_name_map.get(str(row["model"]), str(row["model"])),
            "range": int(np.rint(float(row["range"]))),
            "psill": f"{float(row['psill']):.3f}",
        }
    variogram["correlation"] = variogram_correlation_breakpoints(model)
    return {"variogram": variogram}


_SUBMARINE_INTERVALS = (
    ("13_18", "slope_start_mid"),
    ("19_24", "slope_mid_end"),
    ("13_24", "slope_start_end"),
)


_TMB_PARTITIONS = ("all", "surging", "nonsurging")


def _get_submarine_and_tmb_stats(
    changes: dict[str, object], sampled: pd.DataFrame
) -> tuple[dict[str, object], dict[str, object]]:
    """Archipelago-wide submarine volume/mass change and total mass balance (TMB), per interval and surge class.

    Submarine rates come from `submarine.get_submarine_changes()`, i.e. the
    `SvalbardSubmarineRetreat` product, reduced with
    `submarine.aggregate_submarine` (mean and half-range over the bed models,
    on each partition's own total). Each glacier's interval-specific surge
    classification is read from `sampled` (already includes the Tinkarpbreen
    override applied in `get_statistics()`), so the `"surging"`/`"nonsurging"`
    split matches `changes[stats_key]`. TMB combines above-water and submarine
    volume at their own densities: `comparisons.common.RHO_AW`/`RHO_AW_ERR`
    (850+/-60 kg/m^3) above water and `submarine.RHO_SUB` (900+/-0 kg/m^3)
    below, matching `main.tex`'s TMB equation. The two volume errors are
    independent (different data), so they combine in quadrature.

    The `"all"` partition also carries every model's own total (excluded
    models included, under `models`) and each ensemble model's interpolated
    share, for the SI table. The `*_percent` fields of the submarine category
    can be negative or exceed 100%, since a partition's net submarine term can
    have the opposite sign to the archipelago's; prefer quoting the rates.
    """
    submarine_stats: dict[str, object] = {}
    tmb_stats: dict[str, object] = {}
    rho_sub_gt_per_km3 = submarine.RHO_SUB / 1000.0
    product = submarine.read_submarine_product()

    for interval_short, stats_key in _SUBMARINE_INTERVALS:
        per_glacier = submarine.get_submarine_changes(
            interval_short, product
        ).per_glacier
        surging_col = tools.get_interval(interval_short).surging_col
        surge_map = sampled.set_index("rgi_id")[surging_col]
        is_surging = (
            per_glacier.index.to_series().map(surge_map).fillna(False).astype(bool)
        )
        partition_masks = {
            "all": pd.Series(True, index=per_glacier.index),
            "surging": is_surging,
            "nonsurging": ~is_surging,
        }

        submarine_partitions: dict[str, object] = {}
        tmb_partitions: dict[str, object] = {}
        for partition, mask in partition_masks.items():
            subset = per_glacier.loc[mask]
            vol_rate, vol_rate_err = (
                value / 1e9 for value in submarine.aggregate_submarine(subset)
            )
            mass_rate = vol_rate * rho_sub_gt_per_km3
            mass_rate_err = vol_rate_err * rho_sub_gt_per_km3

            sub_entry: dict[str, object] = {
                "vol_rate": vol_rate,
                "vol_rate_err": vol_rate_err,
                "mass_rate": mass_rate,
                "mass_rate_err": mass_rate_err,
            }
            if partition == "all":
                sub_entry["models"] = {
                    submarine.MODEL_LABELS[model]: {"vol_rate": total / 1e9}
                    for model, total in submarine.model_totals(subset).items()
                }
                for model in submarine.BED_MODELS:
                    sub_entry["models"][submarine.MODEL_LABELS[model]][
                        "interpolated_percent"
                    ] = round(100 * submarine.interpolated_share(subset, model))
                sub_entry["retreat_area"] = float(subset["retreat_area"].sum() / 1e6)
                sub_entry["advance_area"] = float(subset["advance_area"].sum() / 1e6)

            above_water = changes[stats_key][partition]
            aw_vol_rate = float(above_water["vol_rate"])
            aw_vol_rate_err = float(above_water["vol_rate_err"])
            aw_mass_rate = aw_vol_rate * comparisons.common.RHO_AW
            aw_mass_rate_err = float(
                np.hypot(
                    aw_vol_rate_err * comparisons.common.RHO_AW,
                    aw_vol_rate * comparisons.common.RHO_AW_ERR,
                )
            )

            total_vol_rate = aw_vol_rate + vol_rate
            total_vol_rate_err = float(np.hypot(aw_vol_rate_err, vol_rate_err))
            total_mass_rate = aw_mass_rate + mass_rate
            total_mass_rate_err = float(np.hypot(aw_mass_rate_err, mass_rate_err))

            tmb_entry: dict[str, object] = {
                "vol_rate": total_vol_rate,
                "vol_rate_err": total_vol_rate_err,
                "mass_rate": total_mass_rate,
                "mass_rate_err": total_mass_rate_err,
                "above_water_vol_rate": aw_vol_rate,
                "above_water_vol_rate_err": aw_vol_rate_err,
                "above_water_mass_rate": aw_mass_rate,
                "above_water_mass_rate_err": aw_mass_rate_err,
                # The density term of above_water_mass_rate_err on its own.
                "density_mass_rate_err": abs(
                    aw_vol_rate * comparisons.common.RHO_AW_ERR
                ),
                "submarine_vol_rate": vol_rate,
                "submarine_vol_rate_err": vol_rate_err,
                "submarine_mass_rate": mass_rate,
                "submarine_mass_rate_err": mass_rate_err,
                "submarine_of_total_percent": round(100 * vol_rate / total_vol_rate),
            }

            if partition != "all":
                sub_entry["vol_rate_percent"] = round(
                    100 * vol_rate / submarine_partitions["all"]["vol_rate"]
                )
                sub_entry["mass_rate_percent"] = round(
                    100 * mass_rate / submarine_partitions["all"]["mass_rate"]
                )
                tmb_entry["vol_rate_percent"] = round(
                    100 * total_vol_rate / tmb_partitions["all"]["vol_rate"]
                )
                tmb_entry["mass_rate_percent"] = round(
                    100 * total_mass_rate / tmb_partitions["all"]["mass_rate"]
                )

            submarine_partitions[partition] = sub_entry
            tmb_partitions[partition] = tmb_entry

        submarine_stats[stats_key] = submarine_partitions
        tmb_stats[stats_key] = tmb_partitions

    submarine_stats["validation"] = _get_bed_validation_stats()
    return submarine_stats, tmb_stats


def _get_bed_validation_stats() -> dict[str, object]:
    """Raw bed models against IBCAO soundings at the fronts, for the Millan exclusion argument.

    `own` is each model on its own sounded cells; `common` is the largest
    shared-cell set in the upstream table (the like-for-like comparison).
    Keys are `submarine.MODEL_LABELS`; `depth_ratio` is model over measured
    depth, `rmse` and `median_bias` in m (negative bias: too deep).
    """
    validation = submarine.read_bed_model_validation()
    cell_sets = validation["cell_set"].unique()
    common = max(
        (cell_set for cell_set in cell_sets if cell_set != "own"),
        key=lambda cell_set: cell_set.count("+"),
    )
    out: dict[str, object] = {}
    for key, cell_set in (("own", "own"), ("common", common)):
        rows = validation.loc[validation["cell_set"] == cell_set]
        out[key] = {
            submarine.MODEL_LABELS[row.model]: {
                "n_cells": int(row.n_cells),
                "depth_ratio": float(row.depth_ratio),
                "rmse": round(row.rmse_m),
                "median_bias": round(row.median_bias_m),
            }
            for row in rows.itertuples()
        }
    return out


def _subperiod_ratio(start: dict, end: dict) -> dict[str, object]:
    ratio = end["vol_rate"] / start["vol_rate"]
    return {
        # One decimal, as cited in the text ("a factor of 3.9").
        "vol_rate_ratio": f"{ratio:.1f}",
        "vol_rate_increase_percent": round(100 * (ratio - 1)),
    }


def _get_subperiod_ratios(changes: dict) -> dict[str, object]:
    """Ratio of the 2019-2024 to the 2013-2018 volume change rate.

    Parameters
    ----------
    changes
        The `changes` statistics, with `slope_start_mid` and `slope_mid_end`.

    Returns
    -------
    Per partition (all, surging, nonsurging) and zone, `vol_rate_ratio` (one
    decimal; >1 means more negative in 2019-2024) and
    `vol_rate_increase_percent`. The surge classification is made per period,
    so the surging ratios compare different sets of glaciers. Zones with no
    volume change in either period (no glaciers in the partition) are skipped.
    """
    start, end = changes["slope_start_mid"], changes["slope_mid_end"]
    out: dict[str, object] = {}
    for partition in ("all", "surging", "nonsurging"):
        per_zone = {
            zone: _subperiod_ratio(start[partition]["per_zone"][zone], zone_end)
            for zone, zone_end in end[partition]["per_zone"].items()
            if zone_end["vol_rate"] != 0
            and start[partition]["per_zone"][zone]["vol_rate"] != 0
        }
        out[partition] = _subperiod_ratio(start[partition], end[partition]) | {
            "per_zone": per_zone
        }
    return out


def get_statistics():
    sampled = sampling.sample_rasters()
    to_v_keys = ["slope_13_24", "slope_13_18", "slope_19_24", "accel_13_24"]
    intervals = tools.iter_intervals()
    sampled.loc[
        sampled["rgi_id"] == "RGI2000-v7.0-G-07-00560",
        [i.surging_col for i in intervals],
    ] = True
    _validate_per_glacier_exports(sampled)

    glacier_zones = outlines.refine_glacier_zones()
    neff_model = sampling.get_neff_model()

    all_stats = {
        "units": {
            "vol_rate": "km$^{3}$~a$^{-1}$",
            "vol_accel": "km$^{3}$~a$^{-2}$",
            "elev_rate": "m~a$^{-1}$",
        },
        "parameters": {
            "positive_change_threshold": POSITIVE_CHANGE_THRESHOLD,
            "vertcoreg_min_comparisons": coreg.VERTCOREG_MIN_COMPARISONS,
        },
    }

    all_stats["changes"] = {}
    rigidcoreg = coreg.read_rigidcoreg_results()
    vertcoreg = coreg.read_vertcoreg_results()
    vertcoreg_meta = coreg.read_vertcoreg_meta()
    vertcoreg_support = coreg.read_vertcoreg_support()
    # Off-glacier shares are constant per region (same points every year); coverage varies
    support_per_zone = vertcoreg_support.groupby("vertcoreg_zone").first()
    stable_in_region_percent = (
        100
        * support_per_zone["n_stable_in_region"]
        / support_per_zone["n_points_in_region"]
    )
    stable_in_box_percent = (
        100 * support_per_zone["n_stable"] / support_per_zone["n_points"]
    )
    stable_covered_percent = (
        100 * vertcoreg_support["n_stable_covered"] / vertcoreg_support["n_stable"]
    )

    all_stats["coregistration"] = {
        "counts": coreg.get_coregistration_counts(),
        "rigidcoreg": {
            "stable_terrain_percent_median": float(
                (rigidcoreg["stable_fraction"] * 100).median()
            ),
            "stable_nmad_median": float(rigidcoreg["stable_nmad"].median()),
            "icp_slope_m_per_km_median": float(
                rigidcoreg["icp_slope_m_per_km"].median()
            ),
            "icp_slope_m_per_km_mode": _histogram_mode(
                rigidcoreg["icp_slope_m_per_km"], np.linspace(0, 0.8, 150)
            ),
            "horizontal_shift_m_median": float(
                rigidcoreg["rigidcoreg_horizontal_shift_m"].median()
            ),
        },
        "vertcoreg": {
            "ramp_m_per_km_median": float(vertcoreg["ramp_m_per_km"].median()),
            "ramp_m_per_km_mode": _histogram_mode(
                vertcoreg["ramp_m_per_km"], np.linspace(0, 0.3, 60)
            ),
            "stable_in_region_percent_min": round(
                float(stable_in_region_percent.min())
            ),
            "stable_in_region_percent_max": round(
                float(stable_in_region_percent.max())
            ),
            "stable_in_box_percent_min": round(float(stable_in_box_percent.min())),
            "stable_in_box_percent_max": round(float(stable_in_box_percent.max())),
            "stable_covered_percent_median": round(
                float(stable_covered_percent.median())
            ),
            "stable_covered_percent_min": round(float(stable_covered_percent.min())),
            "between_pair_nmad_pre_mean": float(vertcoreg_meta["nmad_pre"].mean()),
            "between_pair_nmad_post_mean": float(vertcoreg_meta["nmad_post"].mean()),
        },
    }
    all_stats["uncertainty"] = _get_variogram_statistics()

    for key in to_v_keys:
        interval = "_".join(key.split("_")[-2:])
        vol_col = key + "_vol"
        changes = {}

        for partition, query in [
            ("all", ""),
            ("surging", f"surging_{interval}"),
            ("nonsurging", f"~surging_{interval}"),
        ]:
            for zone_label in ["allzones", *glacier_zones.index]:
                subset = sampled if partition == "all" else sampled.query(query)
                if zone_label != "allzones":
                    subset = subset.query(f"zone_label == '{zone_label}'")

                total_area = np.max([subset[f"area_{interval}"].sum(), 1e-5])
                vol_rate = subset[vol_col].sum()
                baseline_err = subset[vol_col + "_baseline_err_unscaled"].sum() / (
                    neff_model(total_area) ** 0.5
                )
                excess_err = subset[vol_col + "_excess_err"].sum()
                vol_rate_err = np.hypot(baseline_err, excess_err)

                new_changes = {
                    "vol_rate": vol_rate / 1e9,
                    "vol_rate_err": vol_rate_err / 1e9,
                    "area": total_area / 1e6,
                    "elev_rate": vol_rate / total_area,
                    "elev_rate_err": vol_rate_err / total_area,
                }
                if "accel" not in key:
                    new_changes["positive_vol"] = (
                        subset[f"{key}_positive_vol"].sum() / 1e9
                    )
                    new_changes["positive_area"] = (
                        subset[f"{key}_positive_area"].sum() / 1e6
                    )
                    new_changes["positive_area_frac"] = round(
                        100 * new_changes["positive_area"] / new_changes["area"]
                    )

                if partition != "all" or zone_label != "allzones":
                    denom = "all" if zone_label == "allzones" else partition
                    new_changes["vol_rate_percent"] = round(
                        100 * new_changes["vol_rate"] / changes[denom]["vol_rate"]
                    )
                    new_changes["area_percent"] = round(
                        100 * new_changes["area"] / changes[denom]["area"]
                    )

                if zone_label == "allzones":
                    changes[partition] = new_changes
                else:
                    if "per_zone" not in changes[partition]:
                        changes[partition]["per_zone"] = {}
                    changes[partition]["per_zone"][zone_label] = new_changes

        changes["per_glacier"] = _build_per_glacier_exports(sampled, key)

        if key == "slope_13_24":
            stats_key = "slope_start_end"
        elif key == "slope_13_18":
            stats_key = "slope_start_mid"
        elif key == "slope_19_24":
            stats_key = "slope_mid_end"
        else:
            stats_key = "accel_start_end"

        all_stats["changes"][stats_key] = changes

    all_stats["changes"]["ratio_mid_start"] = _get_subperiod_ratios(
        all_stats["changes"]
    )

    all_stats["classifier"] = gap_classifier.get_classifier_stats()
    gap_classifier.write_classifier_importance_table()

    all_stats["submarine"], all_stats["tmb"] = _get_submarine_and_tmb_stats(
        all_stats["changes"], sampled
    )

    all_stats["coverage"] = coverage.get_coverage_stats()
    all_stats["positive_sensitivity"] = positive_area.get_sensitivity_stats(sampled)

    all_stats["comparisons"] = comparisons.get_comparisons(all_stats["changes"])
    comparisons.write_comparison_table(all_stats["changes"])

    reporting.write_statistics_outputs(all_stats)
    return all_stats


def compute_hypsometric_profiles():
    import rasterio as rio
    import rasterio.features

    sampled = sampling.sample_rasters()
    zone_meta = (
        sampled[["zone_label", "zone_name"]].drop_duplicates().set_index("zone_label")
    )
    zone_meta = zone_meta.loc[zone_meta.index.notna() & zone_meta["zone_name"].notna()]
    zone_ids = {label: i + 1 for i, label in enumerate(zone_meta.index)}
    id_to_label = {i: label for label, i in zone_ids.items()}

    elev_bins = np.linspace(-0.01, 1200.01, 11)
    elev_bin_centers = (elev_bins[1:] + elev_bins[:-1]) / 2

    @dataclasses.dataclass
    class Config:
        interval: tools.Interval
        filepath: str

    interval_map = {i.short: i for i in tools.iter_intervals()}
    intervals = [
        Config(interval_map["13_18"], "input/trend_2013-2018_slope.tif"),
        Config(interval_map["19_24"], "input/trend_2019-2024_slope.tif"),
    ]

    intercept_path = "input/trend_2013-2024_intercept.tif"
    with rio.open(intercept_path) as dem_raster_full:
        scale = dem_raster_full.scales[0]

    with rio.open(intercept_path, overview_level=3) as dem_raster:
        dem_arr = (dem_raster.read(1, masked=True).astype("float32") * scale).filled(0)
        dem_transform = dem_raster.transform

    per_zone = {}
    for config in intervals:
        geom_col = config.interval.geometry_col
        nonsurging = sampled.loc[
            ~sampled[config.interval.surging_col]
            & sampled[geom_col].notna()
            & sampled["zone_label"].notna()
        ].copy()
        nonsurging = nonsurging[~nonsurging[geom_col].is_empty]

        with rio.open(config.filepath) as dhdt_raster_full:
            scale = dhdt_raster_full.scales[0]

        with rio.open(config.filepath, overview_level=3) as dhdt_raster:
            zone_raster = rasterio.features.rasterize(
                [
                    (geom, zone_ids[label])
                    for geom, label in zip(
                        nonsurging[geom_col], nonsurging["zone_label"]
                    )
                ],
                out_shape=dem_arr.shape,
                transform=dem_transform,
                fill=0,
                dtype="uint8",
            )
            valid = zone_raster > 0
            dhdt_arr = (
                dhdt_raster.read(1, masked=True).astype("float32") * scale
            ).filled(np.nan)
            if dhdt_arr.shape != dem_arr.shape:
                raise RuntimeError(
                    f"Hypsometric raster shape mismatch for {config.interval.display_label}"
                )

        frame = pd.DataFrame(
            {
                "zone_label": [id_to_label[i] for i in zone_raster[valid]],
                "elev_idx": np.digitize(dem_arr, elev_bins)[valid],
                "dhdt": dhdt_arr[valid],
            }
        )
        frame = frame[
            (frame["elev_idx"] >= 1) & (frame["elev_idx"] <= elev_bin_centers.shape[0])
        ]

        zone_results = {}
        for label in zone_meta.index:
            subset = frame[frame["zone_label"] == label]
            vals = []
            for idx in np.unique(subset["elev_idx"]):
                vals_here = subset.loc[subset["elev_idx"] == idx, "dhdt"].to_numpy(
                    dtype=float
                )
                if vals_here.size == 0:
                    continue
                med = np.nanmedian(vals_here)
                vals.append(
                    {
                        "elevation": elev_bin_centers[idx - 1],
                        f"{config.interval.short}_med": med,
                        f"{config.interval.short}_nmad": 1.4826
                        * np.nanmedian(np.abs(vals_here - med)),
                    }
                )

            zone_results[label] = (
                pd.DataFrame.from_records(vals).set_index("elevation")
                if vals
                else pd.DataFrame(
                    columns=[
                        f"{config.interval.short}_med",
                        f"{config.interval.short}_nmad",
                    ]
                ).set_index(pd.Index([], name="elevation"))
            )
        per_zone[config.interval.short] = zone_results

    return {
        "zone_meta": zone_meta,
        "per_zone": per_zone,
        "intervals": intervals,
        "elev_bins": elev_bins,
        "elev_bin_centers": elev_bin_centers,
    }
