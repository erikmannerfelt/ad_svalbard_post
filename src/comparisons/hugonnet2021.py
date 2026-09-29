"""Comparison with the ASTER elevation-change series of Hugonnet et al. (2021).

Regional and per-glacier products for RGI region 7 (Svalbard and Jan Mayen). Two
properties make this the cleanest external comparison available: the same
850 kg m^-3 volume-to-mass conversion we use, and released 1-sigma uncertainties
(doubling them reproduces the published 95% interval of -10.5 +/- 1.7 Gt a^-1).

Unlike the other modules here, these files have no stable public download URL, so
they live in ``input/comparisons/hugonnet2021`` rather than being fetched.

The regional offset over 2013-2018 invites the reading that ASTER under-resolves
surge signals, because it happens to equal the surge-related excess loss we resolve.
:func:`surge_specificity` tests that directly and it does not hold: the offset is
statistically indistinguishable between surge-affected and quiescent glaciers.
"""

from __future__ import annotations

import zipfile
from typing import Final

import pandas as pd
from scipy import stats

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


def load_per_glacier_13_18() -> pd.DataFrame:
    """Per-glacier 2013-2018 elevation-change rates, built from the annual periods."""
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
    grouped = annual.groupby("rgiid").agg(
        hugonnet_elev_rate=("dhdt", "mean"),
        n_periods=("dhdt", "size"),
        perc_area_meas=("perc_area_meas", "mean"),
    )
    if not (grouped["n_periods"] == len(ANNUAL_PERIODS_13_18)).all():
        raise ValueError(
            "some glaciers lack the full set of annual periods for 2013-2018"
        )
    return grouped.drop(columns="n_periods")


def matched_glaciers() -> pd.DataFrame:
    """Join our 2013-2018 per-glacier rates to Hugonnet's via the official RGI6 links.

    Returns
    -------
    pandas.DataFrame
        One row per linked glacier, with ``elev_diff`` (ours minus theirs, m a^-1)
        and ``vol_diff`` (km^3 a^-1).
    """
    links = common.load_rgi6_links()[["rgi7_id", "rgi6_id"]]
    ours = common.load_our_glaciers("13_18")
    merged = links.merge(ours, left_on="rgi7_id", right_on="rgi_id").merge(
        load_per_glacier_13_18(), left_on="rgi6_id", right_index=True
    )
    merged["elev_diff"] = merged["our_elev_rate"] - merged["hugonnet_elev_rate"]
    merged["vol_diff"] = merged["elev_diff"] * merged["area_km2"] / 1000.0
    return merged


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
        "linked_area_km2": float(merged["area_km2"].sum()),
    }


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
    austfonna = float(merged.loc[merged["zone_label"] == "A", "vol_diff"].sum())
    return {
        "elev_rate_start_mid": round(reference["elev_rate"], 3),
        "vol_rate_start_mid": round(reference["vol_rate"], 2),
        "mass_rate_start_mid": round(reference["mass_rate"], 2),
        "area": round(region_area_km2()),
        "n_linked": len(merged),
        "linked_area": round(specificity["linked_area_km2"]),
        "mannwhitney_p": round(specificity["mannwhitney_p"], 3),
        "storisstraumen_share_percent": round(
            specificity["storisstraumen_share_percent"]
        ),
        "austfonna_share_percent": round(
            100.0 * austfonna / specificity["total_vol_diff"]
        ),
    }
