from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import tools


VERTCOREG_MIN_COMPARISONS = 60


def _vertical_change_m_per_km(matrix: object) -> float:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (4, 4):
        return float("nan")
    rot = arr[:3, :3]
    cos_theta = (np.trace(rot) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.tan(theta) * 1000.0)


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_json_member(zip_file: ZipFile, name: str) -> dict:
    with zip_file.open(name) as fh:
        return json.loads(fh.read())


def _parse_rigidcoreg_member(name: str, obj: dict) -> dict:
    file_stem = Path(name).stem
    dem_stem = file_stem.removesuffix("_dem_coreg")
    parts = dem_stem.split("_")
    year = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else pd.NA
    sensor = parts[1] if len(parts) > 1 else pd.NA

    steps = obj.get("steps", []) or []
    icp_step = next((step for step in steps if isinstance(step, dict) and step.get("name") == "affine.ICP"), steps[0] if steps else {})
    nk_step = next((step for step in steps if isinstance(step, dict) and step.get("name") == "affine.NuthKaab"), steps[1] if len(steps) > 1 else {})

    matrix = (((icp_step or {}).get("meta") or {}).get("_meta") or {}).get("matrix")
    matrix_arr = np.asarray(matrix, dtype=float) if matrix is not None else np.full((4, 4), np.nan)
    icp_shift_east_m = _safe_float(matrix_arr[0, 3]) if matrix_arr.shape == (4, 4) else float("nan")
    icp_shift_north_m = _safe_float(matrix_arr[1, 3]) if matrix_arr.shape == (4, 4) else float("nan")
    icp_horizontal_shift_m = float(np.hypot(icp_shift_east_m, icp_shift_north_m))
    icp_slope_m_per_km = _vertical_change_m_per_km(matrix_arr)

    nk_meta = (nk_step or {}).get("meta") or {}
    nk_inner = nk_meta.get("_meta") or {}
    nuth_kaab_shift_east_px = _safe_float(nk_inner.get("offset_east_px"))
    nuth_kaab_shift_north_px = _safe_float(nk_inner.get("offset_north_px"))
    nuth_kaab_resolution_m = _safe_float(nk_inner.get("resolution"))
    nuth_kaab_shift_east_m = nuth_kaab_shift_east_px * nuth_kaab_resolution_m
    nuth_kaab_shift_north_m = nuth_kaab_shift_north_px * nuth_kaab_resolution_m

    rigidcoreg_shift_east_m = icp_shift_east_m + nuth_kaab_shift_east_m
    rigidcoreg_shift_north_m = icp_shift_north_m + nuth_kaab_shift_north_m
    rigidcoreg_horizontal_shift_m = float(np.hypot(rigidcoreg_shift_east_m, rigidcoreg_shift_north_m))

    bounds = obj.get("bounds", {}) or {}

    return {
        "result_stem": file_stem,
        "file_stem": dem_stem,
        "year": year,
        "sensor": sensor,
        "icp_slope_m_per_km": icp_slope_m_per_km,
        "icp_shift_east_m": icp_shift_east_m,
        "icp_shift_north_m": icp_shift_north_m,
        "icp_horizontal_shift_m": icp_horizontal_shift_m,
        "nuth_kaab_shift_east_px": nuth_kaab_shift_east_px,
        "nuth_kaab_shift_north_px": nuth_kaab_shift_north_px,
        "nuth_kaab_resolution_m": nuth_kaab_resolution_m,
        "nuth_kaab_shift_east_m": nuth_kaab_shift_east_m,
        "nuth_kaab_shift_north_m": nuth_kaab_shift_north_m,
        "rigidcoreg_shift_east_m": rigidcoreg_shift_east_m,
        "rigidcoreg_shift_north_m": rigidcoreg_shift_north_m,
        "rigidcoreg_horizontal_shift_m": rigidcoreg_horizontal_shift_m,
        "stable_nmad": _safe_float(obj.get("stable_nmad")),
        "stable_fraction": _safe_float(obj.get("stable_fraction")),
        "stable_bias": _safe_float(obj.get("stable_bias")),
        "bounds_left": _safe_float(bounds.get("left")),
        "bounds_bottom": _safe_float(bounds.get("bottom")),
        "bounds_right": _safe_float(bounds.get("right")),
        "bounds_top": _safe_float(bounds.get("top")),
    }


def read_rigidcoreg_results() -> pd.DataFrame:
    records = []
    bad_dems = tools.read_aux_text("excluded_dems_rigidcoreg.txt").splitlines()
    with tools.read_aux_zip("rigidcoreg_results.zip") as zip_file:
        for name in zip_file.namelist():
            if any(s in name for s in bad_dems):
                continue
            if not name.endswith(".json"):
                continue
            obj = _read_json_member(zip_file, name)
            records.append(_parse_rigidcoreg_member(name, obj))
    return pd.DataFrame.from_records(records)


def _parse_vertcoreg_member_name(name: str) -> tuple[int, str, bool]:
    stem = Path(name).stem
    parts = stem.split("_")
    year = int(parts[2])
    vertcoreg_zone = "_".join(parts[3:-1] if parts[-1] == "meta" else parts[3:])
    is_meta = stem.endswith("_meta")
    return year, vertcoreg_zone, is_meta


def read_vertcoreg_results() -> pd.DataFrame:
    records = []
    bad_dems = tools.read_aux_text("excluded_dems_rigidcoreg.txt").splitlines()
    with tools.read_aux_zip("vertcoreg_results.zip") as zip_file:
        for name in zip_file.namelist():
            if not name.endswith(".csv"):
                continue
            year, vertcoreg_zone, _ = _parse_vertcoreg_member_name(name)
            with zip_file.open(name) as fh:
                data = pd.read_csv(fh).rename(
                    columns={
                        "Unnamed: 0": "file_stem",
                        "a": "ramp_east_m_per_m",
                        "b": "ramp_north_m_per_m",
                        "c": "vertical_offset_m",
                    }
                )
            data = data[~data["file_stem"].isin(bad_dems)]
            data["year"] = year
            data["vertcoreg_zone"] = vertcoreg_zone
            data = data.loc[data["n_comparisons"] > VERTCOREG_MIN_COMPARISONS].copy()
            data["ramp_m_per_km"] = np.hypot(data["ramp_east_m_per_m"], data["ramp_north_m_per_m"]) * 1000
            records.append(data)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def read_vertcoreg_meta() -> pd.DataFrame:
    records = []
    with tools.read_aux_zip("vertcoreg_results.zip") as zip_file:
        for name in zip_file.namelist():
            if not name.endswith("_meta.json"):
                continue
            year, vertcoreg_zone, _ = _parse_vertcoreg_member_name(name)
            obj = _read_json_member(zip_file, name)
            records.append(
                {
                    "year": year,
                    "vertcoreg_zone": vertcoreg_zone,
                    "n_points": _safe_float(obj.get("n_points")),
                    "n_stable_points": _safe_float(obj.get("n_stable_points")),
                    "stable_terrain_median_pre": _safe_float(obj.get("stable_terrain_median_pre")),
                    "stable_terrain_median_post": _safe_float(obj.get("stable_terrain_median_post")),
                    "nmad_pre": _safe_float(obj.get("nmad_pre")),
                    "nmad_post": _safe_float(obj.get("nmad_post")),
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def get_coregistration_counts() -> dict[str, int]:
    rigid_ids = set(read_rigidcoreg_results()["file_stem"])
    vert_ids = set(read_vertcoreg_results()["file_stem"])
    return {
        "rigidcoreg": len(rigid_ids),
        "vertcoreg": len(vert_ids),
        "overlap": len(rigid_ids & vert_ids),
        "combined_unique": len(rigid_ids | vert_ids),
        "vertcoreg_only": len(vert_ids - rigid_ids),
    }
