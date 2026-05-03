from __future__ import annotations

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import coreg, outlines, reporting, sampling, tools


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
            out += model_funs[str(row["model"]) ](h_arr, float(row["range"]), float(row["psill"]))
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


def variogram_correlation_breakpoints(params: pd.DataFrame | None = None) -> dict[str, str]:
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
    for component_name, (_, row) in zip(component_names, model.iterrows(), strict=False):
        variogram[component_name] = {
            "model": model_name_map.get(str(row["model"]), str(row["model"])),
            "range": int(np.rint(float(row["range"]))),
            "psill": f"{float(row['psill']):.3f}",
        }
    variogram["correlation"] = variogram_correlation_breakpoints(model)
    return {"variogram": variogram}


def get_statistics():
    sampled = sampling.sample_rasters()
    to_v_keys = ["slope_13_24", "slope_13_18", "slope_19_24", "accel_13_24"]
    intervals = tools.iter_intervals()
    sampled.loc[sampled["rgi_id"] == "RGI2000-v7.0-G-07-00560", [i.surging_col for i in intervals]] = True

    glacier_zones = outlines.refine_glacier_zones()
    neff_model = sampling.get_neff_model()

    all_stats = {
        "units": {
            "vol_rate": "km$^{3}$~a$^{-1}$",
            "vol_accel": "km$^{3}$~a$^{-2}$",
            "elev_rate": "m~a$^{-1}$",
        },
        "parameters": {"positive_change_threshold": 0.1, "vertcoreg_min_comparisons": coreg.VERTCOREG_MIN_COMPARISONS},
    }

    all_stats["uncertainty"] = _get_variogram_statistics()

    all_stats["changes"] = {}
    rigidcoreg = coreg.read_rigidcoreg_results()
    vertcoreg = coreg.read_vertcoreg_results()
    vertcoreg_meta = coreg.read_vertcoreg_meta().copy()
    vertcoreg_meta["support_percent"] = vertcoreg_meta["n_stable_points"] / vertcoreg_meta["n_points"] * 100

    all_stats["coregistration"] = {
        "counts": coreg.get_coregistration_counts(),
        "rigidcoreg": {
            "stable_terrain_percent_median": float((rigidcoreg["stable_fraction"] * 100).median()),
            "stable_nmad_median": float(rigidcoreg["stable_nmad"].median()),
            "icp_slope_m_per_km_median": float(rigidcoreg["icp_slope_m_per_km"].median()),
            "icp_slope_m_per_km_mode": _histogram_mode(rigidcoreg["icp_slope_m_per_km"], np.linspace(0, 0.8, 150)),
            "horizontal_shift_m_median": float(rigidcoreg["rigidcoreg_horizontal_shift_m"].median()),
        },
        "vertcoreg": {
            "ramp_m_per_km_median": float(vertcoreg["ramp_m_per_km"].median()),
            "ramp_m_per_km_mode": _histogram_mode(vertcoreg["ramp_m_per_km"], np.linspace(0, 0.3, 60)),
            "stable_terrain_percent_mean": float(vertcoreg_meta["support_percent"].mean()),
            "between_pair_nmad_pre_mean": float(vertcoreg_meta["nmad_pre"].mean()),
            "between_pair_nmad_post_mean": float(vertcoreg_meta["nmad_post"].mean()),
        },
    }

    for key in to_v_keys:
        interval = "_".join(key.split("_")[-2:])
        vol_col = key + "_vol"
        changes = {}

        for partition, query in [("all", ""), ("surging", f"surging_{interval}"), ("nonsurging", f"~surging_{interval}")]:
            for zone_label in ["allzones", *glacier_zones.index]:
                subset = sampled if partition == "all" else sampled.query(query)
                if zone_label != "allzones":
                    subset = subset.query(f"zone_label == '{zone_label}'")

                total_area = np.max([subset[f"area_{interval}"].sum(), 1e-5])
                vol_rate = subset[vol_col].sum()
                baseline_err = subset[vol_col + "_baseline_err_unscaled"].sum() / (neff_model(total_area) ** 0.5)
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
                    new_changes["positive_vol"] = subset[f"{key}_positive_vol"].sum() / 1e9
                    new_changes["positive_area"] = subset[f"{key}_positive_area"].sum() / 1e6
                    new_changes["positive_area_frac"] = round(100 * new_changes["positive_area"] / new_changes["area"])

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

        if key == "slope_13_24":
            stats_key = "slope_start_end"
        elif key == "slope_13_18":
            stats_key = "slope_start_mid"
        elif key == "slope_19_24":
            stats_key = "slope_mid_end"
        else:
            stats_key = "accel_start_end"

        all_stats["changes"][stats_key] = changes

    reporting.write_statistics_outputs(all_stats)
    return all_stats


def compute_hypsometric_profiles():
    import rasterio as rio
    import rasterio.features

    sampled = sampling.sample_rasters()
    zone_meta = sampled[["zone_label", "zone_name"]].drop_duplicates().set_index("zone_label")
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
        nonsurging = sampled.loc[~sampled[config.interval.surging_col] & sampled[geom_col].notna() & sampled["zone_label"].notna()].copy()
        nonsurging = nonsurging[~nonsurging[geom_col].is_empty]

        with rio.open(config.filepath) as dhdt_raster_full:
            scale = dhdt_raster_full.scales[0]

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
                raise RuntimeError(f"Hypsometric raster shape mismatch for {config.interval.display_label}")

        frame = pd.DataFrame({"zone_label": [id_to_label[i] for i in zone_raster[valid]], "elev_idx": np.digitize(dem_arr, elev_bins)[valid], "dhdt": dhdt_arr[valid]})
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
                vals.append({"elevation": elev_bin_centers[idx - 1], f"{config.interval.short}_med": med, f"{config.interval.short}_nmad": 1.4826 * np.nanmedian(np.abs(vals_here - med))})

            zone_results[label] = pd.DataFrame.from_records(vals).set_index("elevation") if vals else pd.DataFrame(columns=[f"{config.interval.short}_med", f"{config.interval.short}_nmad"]).set_index(pd.Index([], name="elevation"))
        per_zone[config.interval.short] = zone_results

    return {"zone_meta": zone_meta, "per_zone": per_zone, "intervals": intervals, "elev_bins": elev_bins, "elev_bin_centers": elev_bin_centers}
