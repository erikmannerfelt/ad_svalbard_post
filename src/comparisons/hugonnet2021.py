"""Comparison with the ASTER elevation-change series of Hugonnet et al. (2021).

Regional and per-glacier products for RGI region 7 (Svalbard and Jan Mayen). Two
properties make this the cleanest external comparison available: the same
850 kg m^-3 volume-to-mass conversion we use, and released 1-sigma uncertainties
(doubling them reproduces the published 95% interval of -10.5 +/- 1.7 Gt a^-1).

Unlike the other modules here, these files have no stable public download URL, so
they live in ``input/comparisons/hugonnet2021`` rather than being fetched.

The regional offset over 2013-2018 invites the reading that ASTER under-resolves
surge signals. :func:`surge_specificity` and :func:`subset_comparison` test that
directly and it does not hold: our rates are 35-45% more negative on surge-affected
and other glaciers alike, and excluding the large surges does not close the gap.

Glaciers are linked through the full RGI7-RGI6 overlap table
(:func:`common.load_rgi6_links`): each of our glaciers gets the overlap-area-weighted
mean of the RGI6 rates that overlap it, which covers practically all our glacier area.

:func:`method_specificity` asks the same of the co-registration method: on glaciers
not affected by surging, the offset is the same where the strips were globally
deramped as where they were rigidly co-registered.
"""

from __future__ import annotations

import zipfile
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

from .. import sampling
from ..config import CRS_EPSG, INPUT_DIR
from . import common

DATA_DIR: Final = common.COMPARISON_DIR / "hugonnet2021"
CUMUL_NAME: Final = "dh_07_rgi60_reg_cumul.csv"
PERGLA_NAME: Final = "dh_07_rgi60_pergla_rates.csv.zip"

#: Hugonnet publishes no 2013-2018 period; these five annual periods span it exactly.
ANNUAL_PERIODS_13_18: Final[tuple[str, ...]] = (
    "2013-01-01_2014-01-01",
    "2014-01-01_2015-01-01",
    "2015-01-01_2016-01-01",
    "2016-01-01_2017-01-01",
    "2017-01-01_2018-01-01",
)

#: 1-sigma dh/dt errors of their published 5-year windows bracket this range.
ELEV_ERR_RANGE: Final[tuple[float, float]] = (0.035, 0.090)

#: Manually drawn regions where the strips are globally deramped instead of rigidly
#: co-registered (``adsvalbard.mosaicking``), in the auxiliary files.
DERAMP_REGIONS_NAME: Final = "bad_coreg_regions.geojson"

#: Large surges of 2013-2018, excluded together in :func:`subset_comparison`
LARGE_SURGES: Final[tuple[str, ...]] = (
    "Storisstraumen",
    "Negribreen",
    "Stonebreen",
    "Nathorstbreen",
)


def load_regional_cumulative() -> pd.DataFrame:
    """Load the monthly cumulative regional change series, indexed by date."""
    path = DATA_DIR / CUMUL_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. These files have no public URL; copy them into {DATA_DIR}."
        )
    return pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()


def region_area_km2() -> float:
    """Glacier area of RGI region 7 as used by Hugonnet et al., in km^2."""
    return float(load_regional_cumulative()["area"].iloc[0]) / 1e6


def window_rates(start: str, end: str) -> dict[str, float]:
    """Mean rates between two dates, from the cumulative series.

    Differencing the cumulative series is preferable to averaging the published
    annual rates, whose errors are strongly correlated in time and whose multi-year
    windows do not align with our sub-periods.

    Returns
    -------
    dict
        ``elev_rate`` (m a^-1), ``vol_rate`` (km^3 a^-1), ``mass_rate`` (Gt a^-1)
        and ``span_years``.
    """
    cumulative = load_regional_cumulative()
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    index = cumulative.index.union([first, last])
    span = (last - first).days / 365.25
    out: dict[str, float] = {"span_years": span}
    for column, key, scale in (
        ("dh", "elev_rate", 1.0),
        ("dvol", "vol_rate", 1e-9),
        ("dm", "mass_rate", 1.0),
    ):
        series = cumulative[column].reindex(index).interpolate("time")
        out[key] = float(series[last] - series[first]) * scale / span
    return out


def load_per_glacier_13_18() -> pd.Series:
    """Per-glacier 2013-2018 elevation-change rates (m a^-1), from the annual periods."""
    path = DATA_DIR / PERGLA_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; copy the Hugonnet per-glacier rates there."
        )
    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as handle:
            rates = pd.read_csv(handle)
    annual = rates[rates["period"].isin(ANNUAL_PERIODS_13_18)]
    grouped = annual.groupby("rgiid")["dhdt"].agg(["mean", "size"])
    if not (grouped["size"] == len(ANNUAL_PERIODS_13_18)).all():
        raise ValueError(
            "some glaciers lack the full set of annual periods for 2013-2018"
        )
    return grouped["mean"].rename("hugonnet_elev_rate")


def matched_glaciers() -> pd.DataFrame:
    """Our 2013-2018 per-glacier results with Hugonnet's rate over the same glacier.

    Each of our glaciers gets the overlap-area-weighted mean of the rates of the RGI6
    glaciers that overlap it. Glaciers split or merged between the inventories are
    thereby handled without a matching heuristic.

    Returns
    -------
    pandas.DataFrame
        Our ``sampled`` columns for 2013-2018 plus ``hugonnet_elev_rate`` and
        ``hugonnet_vol_rate`` (over our area, km^3 a^-1), ``our_vol_rate``,
        ``elev_diff`` (ours minus theirs, m a^-1) and ``vol_diff`` (km^3 a^-1).
    """
    links = common.load_rgi6_links()
    links["hugonnet_elev_rate"] = links["rgi6_id"].map(load_per_glacier_13_18())
    links = links.dropna(subset=["hugonnet_elev_rate"])
    links["weighted"] = links["hugonnet_elev_rate"] * links["overlap_area_km2"]
    per_rgi7 = links.groupby("rgi7_id")[["weighted", "overlap_area_km2"]].sum()
    hugonnet_rate = (per_rgi7["weighted"] / per_rgi7["overlap_area_km2"]).rename(
        "hugonnet_elev_rate"
    )

    ours = pd.DataFrame(gpd.read_feather(common.SAMPLED_PATH))
    ours = ours.drop(columns=[c for c in ours.columns if c.startswith("geometry")])
    merged = ours.merge(hugonnet_rate, left_on="rgi_id", right_index=True)
    merged["area_km2"] = merged["area_13_18"] / 1e6
    merged["surging"] = merged["surging_13_18"]
    merged["our_elev_rate"] = merged["slope_13_18"]
    merged["our_vol_rate"] = merged["slope_13_18_vol"] / 1e9
    merged["hugonnet_vol_rate"] = (
        merged["hugonnet_elev_rate"] * merged["area_km2"] / 1000.0
    )
    merged["elev_diff"] = merged["our_elev_rate"] - merged["hugonnet_elev_rate"]
    merged["vol_diff"] = merged["our_vol_rate"] - merged["hugonnet_vol_rate"]
    return merged


def compare_subset(subset: pd.DataFrame) -> dict[str, float]:
    """Total 2013-2018 volume change of a set of glaciers in both records.

    Our uncertainty is aggregated as in ``statistics.get_statistics`` (baseline
    reduced by the number of effective samples, excess summed linearly). Theirs is
    the area times the range of their 1-sigma elevation-change errors, which gives
    a range of significances.

    Returns
    -------
    dict
        ``our_vol_rate``, ``our_vol_rate_err``, ``hugonnet_vol_rate`` (km^3 a^-1),
        ``area`` (km^2), ``percent_difference`` (ours relative to theirs),
        ``sigma_low``/``sigma_high`` (difference over the combined 1-sigma error) and
        ``percent_less_negative`` (theirs relative to ours) with
        ``percent_less_negative_err``, the larger combined 1-sigma error relative to
        ours, so that their ratio is ``sigma_low``.
    """
    area_m2 = float(subset["area_13_18"].sum())
    baseline = float(subset["slope_13_18_vol_baseline_err_unscaled"].sum()) / (
        sampling.get_neff_model()(area_m2) ** 0.5
    )
    excess = float(subset["slope_13_18_vol_excess_err"].sum())
    ours = float(subset["our_vol_rate"].sum())
    ours_err = float(np.hypot(baseline, excess)) / 1e9
    theirs = float(subset["hugonnet_vol_rate"].sum())
    difference = abs(ours - theirs)
    theirs_errs = [err * area_m2 / 1e9 for err in ELEV_ERR_RANGE]
    return {
        "our_vol_rate": ours,
        "our_vol_rate_err": ours_err,
        "hugonnet_vol_rate": theirs,
        "area": area_m2 / 1e6,
        "percent_difference": 100.0 * (ours - theirs) / theirs,
        "sigma_low": difference / float(np.hypot(ours_err, theirs_errs[1])),
        "sigma_high": difference / float(np.hypot(ours_err, theirs_errs[0])),
        "percent_less_negative": 100.0 * (ours - theirs) / ours,
        "percent_less_negative_err": 100.0
        * float(np.hypot(ours_err, theirs_errs[1]))
        / abs(ours),
    }


def subset_comparison() -> dict[str, dict[str, float]]:
    """:func:`compare_subset` for all glaciers and with surges excluded in turn."""
    merged = matched_glaciers()
    return {
        "all": compare_subset(merged),
        "nostorisstraumen": compare_subset(
            merged[merged["glac_name"] != "Storisstraumen"]
        ),
        "nolargesurges": compare_subset(
            merged[~merged["glac_name"].isin(LARGE_SURGES)]
        ),
        "nonsurging": compare_subset(merged[~merged["surging"]]),
        "surging": compare_subset(merged[merged["surging"]]),
    }


def surge_specificity() -> dict[str, float]:
    """Test whether the offset from Hugonnet tracks surge status. It does not.

    Returns
    -------
    dict
        Median offsets by class, the Mann-Whitney p-value, and the share of the
        total volume difference contributed by Storisstraumen alone.
    """
    merged = matched_glaciers()
    surging = merged[merged["surging"]]
    quiescent = merged[~merged["surging"]]
    total = float(merged["vol_diff"].sum())
    storisstraumen = float(
        merged.loc[merged["glac_name"] == "Storisstraumen", "vol_diff"].iloc[0]
    )
    without = surging[surging["glac_name"] != "Storisstraumen"]
    return {
        "n_surging": len(surging),
        "n_quiescent": len(quiescent),
        "median_diff_surging": float(surging["elev_diff"].median()),
        "median_diff_quiescent": float(quiescent["elev_diff"].median()),
        "median_diff_surging_without_storisstraumen": float(
            without["elev_diff"].median()
        ),
        "mannwhitney_p": float(
            stats.mannwhitneyu(surging["elev_diff"], quiescent["elev_diff"]).pvalue
        ),
        "total_vol_diff": total,
        "storisstraumen_vol_diff": storisstraumen,
        "storisstraumen_share_percent": 100.0 * storisstraumen / total,
    }


def method_specificity() -> dict[str, dict[str, float]]:
    """Test whether the offset from Hugonnet depends on the co-registration method.

    A glacier counts as deramped when most of its 2013-2018 area lies inside the
    deramping regions; the 2 km blending margin around them is ignored. Surge-affected
    glaciers are left out, since three of the largest offsets (Storisstraumen,
    Austfonna Basin-2 and Stonebreen) are surges inside the regions and would
    otherwise be attributed to the method. Percent differences are not reported,
    since the deramped ice caps' rates are close to zero in Hugonnet's product.

    Returns
    -------
    dict
        Per method (``deramped``, ``rigid``): ``n``, ``area`` (km^2) and the
        area-weighted mean ``elev_diff`` (ours minus theirs, m a^-1).
    """
    regions = gpd.read_file(
        f"/vsizip/{INPUT_DIR / 'aux_files.zip'}/{DERAMP_REGIONS_NAME}"
    ).to_crs(CRS_EPSG)
    sampled = gpd.read_feather(common.SAMPLED_PATH).set_geometry("geometry_13_18")
    sampled = sampled.to_crs(CRS_EPSG)
    inside_fraction = pd.Series(
        (
            sampled.geometry.intersection(regions.union_all()).area
            / sampled.geometry.area
        ).to_numpy(),
        index=sampled["rgi_id"],
    )

    merged = matched_glaciers()
    nonsurging = merged[~merged["surging"]]
    deramped = nonsurging["rgi_id"].map(inside_fraction).fillna(0.0) > 0.5
    results = {}
    for name, subset in (
        ("deramped", nonsurging[deramped]),
        ("rigid", nonsurging[~deramped]),
    ):
        area = float(subset["area_km2"].sum())
        results[name] = {
            "n": len(subset),
            "area": area,
            "elev_diff": float((subset["elev_diff"] * subset["area_km2"]).sum()) / area,
        }
    return results


def offset_geography() -> pd.DataFrame:
    """Per-zone breakdown of the offset, excluding the Storisstraumen outlier."""
    merged = matched_glaciers()
    rest = merged[merged["glac_name"] != "Storisstraumen"]
    return (
        rest.groupby("zone_label")
        .agg(
            n=("elev_diff", "size"),
            area_km2=("area_km2", "sum"),
            vol_diff=("vol_diff", "sum"),
            median_elev_diff=("elev_diff", "median"),
        )
        .sort_values("vol_diff")
    )


def summary() -> dict[str, object]:
    """Numbers cited in the manuscript, for ``tables/statistics.tex``."""
    reference = window_rates("2013-01-01", "2018-01-01")
    specificity = surge_specificity()
    merged = matched_glaciers()
    subsets = {
        name: {
            "our_vol_rate": round(values["our_vol_rate"], 2),
            "our_vol_rate_err": round(values["our_vol_rate_err"], 2),
            "hugonnet_vol_rate": round(values["hugonnet_vol_rate"], 2),
            "percent_difference": round(values["percent_difference"]),
            "sigma_low": round(values["sigma_low"], 1),
            "sigma_high": round(values["sigma_high"], 1),
            "percent_less_negative": round(values["percent_less_negative"]),
            "percent_less_negative_err": round(values["percent_less_negative_err"]),
        }
        for name, values in subset_comparison().items()
    }
    return {
        "elev_rate_start_mid": round(reference["elev_rate"], 3),
        "vol_rate_start_mid": round(reference["vol_rate"], 2),
        "mass_rate_start_mid": round(reference["mass_rate"], 2),
        "area": round(region_area_km2()),
        "n_matched": len(merged),
        "matched_area": round(float(merged["area_km2"].sum())),
        "mannwhitney_p": round(specificity["mannwhitney_p"], 3),
        "storisstraumen_share_percent": round(
            specificity["storisstraumen_share_percent"]
        ),
        "subsets": subsets,
        "method": {
            name: {
                "n": values["n"],
                "area": round(values["area"]),
                "elev_diff": round(values["elev_diff"], 2),
            }
            for name, values in method_specificity().items()
        },
    }
