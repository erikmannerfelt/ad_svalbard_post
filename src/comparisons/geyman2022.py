"""Comparison with the 1936-2010 photogrammetric reconstruction of Geyman et al. (2022).

Their supplementary spreadsheet carries per-glacier 1936-2010 elevation and mass
changes, and per-glacier twenty-first-century projections under three emission
scenarios. Glaciers are joined to ours on GLIMS ID, which both inventories carry, so
no spatial matching heuristic is needed.

Three properties of the source must be respected.

* Their published statistics **exclude Austfonna**, because there is no 1936 photo
  coverage of its eastern part. Our side of any comparison must exclude it too, which
  is what :func:`our_comparable_rate` does.
* The widely quoted -0.67 m a^-1 is *not* their historical rate. It is their most
  modest (RCP2.6) 2010-2100 projection. The observed 1936-2010 rate is -0.35 m a^-1.
* **The per-glacier projection columns must not be aggregated.** They contain
  physically implausible outliers (up to +27 m a^-1) and 9-17% of glaciers carry a
  positive projected rate, so an area-weighted mean of them comes out *positive* and
  bears no relation to the published archipelago values. The paper says as much:
  the predictions are "not necessarily accurate for any individual glacier". Use
  :data:`PROJECTIONS`, which holds the published archipelago figures.
  :func:`validate_against_publication` documents this so it is not rediscovered.

The per-glacier *mass* columns, by contrast, reproduce the published total to within
2%, which is what gives confidence that the file is being read correctly.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from .. import reporting, tools
from . import common

URL: Final = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41586-021-04314-4/MediaObjects/41586_2021_4314_MOESM3_ESM.xlsx"
)
FILENAME: Final = "41586_2021_4314_MOESM3_ESM.xlsx"

#: Published archipelago figures, all excluding Austfonna.
OBSERVED_ELEV_RATE: Final = common.Estimate(-0.35, 0.03)  # m a^-1, 1936-2010
OBSERVED_MASS_RATE: Final = common.Estimate(-7.75, 0.20)  # Gt a^-1, 1936-2010

#: Published 2010-2100 projected thinning rates by emission scenario (m a^-1).
PROJECTIONS: Final[dict[str, float]] = {"rcp26": -0.67, "rcp45": -0.73, "rcp85": -0.92}

PROJECTION_COLUMNS: Final[dict[str, str]] = {
    "rcp26": "dhdt_2010_2100_scenario_2.6 (m yr-1)",
    "rcp45": "dhdt_2010_2100_scenario_4.5 (m yr-1)",
    "rcp85": "dhdt_2010_2100_scenario_8.5 (m yr-1)",
}

ELEV_COLUMN: Final = "dhdt_1936_2010 (m yr-1)"
MASS_COLUMN: Final = "mass_change_method1_1936_2010 (kg)"

#: Span of the 1936-2010 reconstruction, used to turn cumulative mass into a rate.
SPAN_YEARS: Final = 74.0


def load(refresh_download: bool = False) -> pd.DataFrame:
    """Load the per-glacier supplementary table, restricted to valid reconstructions.

    The first spreadsheet row holds a merged section banner and the second holds the
    real column names, so the header is taken from the second row.
    """
    path = tools.fetch(URL, common.COMPARISON_DIR / FILENAME, refresh_download=refresh_download)
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None)
    frame = raw.iloc[2:].copy()
    frame.columns = [str(name).strip() for name in raw.iloc[1]]
    frame = frame.reset_index(drop=True)
    numeric = [
        ELEV_COLUMN,
        MASS_COLUMN,
        "area_2010 (m2)",
        "isTidewater",
        "valid_1936_reconstruction",
        *PROJECTION_COLUMNS.values(),
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["glims_id"] = frame["GLIMSID"].astype(str).str.strip().str.strip("'")
    frame["area_km2"] = frame["area_2010 (m2)"] / 1e6
    valid = (frame["valid_1936_reconstruction"] == 1) & (frame["area_km2"] > 0)
    return frame[valid].reset_index(drop=True)


def validate_against_publication(refresh_download: bool = False) -> dict[str, float]:
    """Check what in the spreadsheet does and does not reproduce the published values.

    Confirms that the mass columns and the volume-integrated elevation rate match the
    paper, and records that area-weighting the projection columns does not. Intended
    to be read, not to gate anything.
    """
    frame = load(refresh_download=refresh_download)
    area_km2 = float(frame["area_km2"].sum())
    mass_rate = float(frame[MASS_COLUMN].sum()) / 1e12 / SPAN_YEARS
    # The published -0.35 m a^-1 is volume-integrated, not a mean of glacier rates.
    integrated_elev = mass_rate / common.RHO_AW / area_km2 * 1000.0
    out = {
        "n_glaciers": len(frame),
        "area_km2": area_km2,
        "mass_rate_from_per_glacier": mass_rate,
        "published_mass_rate": OBSERVED_MASS_RATE.value,
        "volume_integrated_elev_rate": integrated_elev,
        "published_elev_rate": OBSERVED_ELEV_RATE.value,
        "area_weighted_mean_of_glacier_rates": common.area_weighted(
            frame[ELEV_COLUMN], frame["area_km2"]
        ),
    }
    for scenario, column in PROJECTION_COLUMNS.items():
        series = frame[column].dropna()
        out[f"{scenario}_area_weighted_UNUSABLE"] = common.area_weighted(
            series, frame.loc[series.index, "area_km2"]
        )
        out[f"{scenario}_max_UNUSABLE"] = float(series.max())
        out[f"{scenario}_percent_positive"] = 100.0 * float((series > 0).mean())
    return out


def matched_glaciers(
    interval: str = "13_24", refresh_download: bool = False
) -> pd.DataFrame:
    """Join our per-glacier rates to Geyman's on GLIMS ID.

    Only the historical columns are carried through; the projection columns are
    deliberately dropped, for the reason given in the module docstring.
    """
    theirs = load(refresh_download=refresh_download)
    ours = common.load_our_glaciers(interval)
    ours["glims_id"] = ours["glims_id"].astype(str).str.strip().str.strip("'")
    merged = ours.merge(
        theirs[
            ["glims_id", ELEV_COLUMN, MASS_COLUMN, "area_km2", "isTidewater"]
        ].rename(
            columns={
                ELEV_COLUMN: "geyman_elev_rate",
                MASS_COLUMN: "geyman_mass_change_kg",
                "area_km2": "geyman_area_km2",
            }
        ),
        on="glims_id",
        how="inner",
    )
    merged["elev_ratio"] = merged["our_elev_rate"] / merged["geyman_elev_rate"]
    return merged


def our_comparable_rate(interval: str = "13_24") -> float:
    """Our volume-integrated elevation rate excluding Austfonna, in m a^-1.

    Geyman's published statistics exclude Austfonna, so a like-for-like comparison
    must exclude our Austfonna zone as well. Volume integration matches how their
    published archipelago rate is formed.
    """
    ours = common.load_our_glaciers(interval)
    outside = ours[ours["zone_label"] != "A"]
    return common.area_weighted(outside["our_elev_rate"], outside["area_km2"])


def projection_benchmark() -> pd.DataFrame:
    """Compare our observed rates with their *published* projected century averages.

    Returns
    -------
    pandas.DataFrame
        One row per interval, giving our Austfonna-excluded rate, its ratio to the
        1936-2010 observed baseline, and its percentage of each scenario projection.
    """
    rows = []
    for interval, label in (
        ("13_18", "2013-2018"),
        ("19_24", "2019-2024"),
        ("13_24", "2013-2024"),
    ):
        rate = our_comparable_rate(interval)
        row: dict[str, float | str] = {
            "interval": label,
            "our_elev_rate_excl_austfonna": rate,
            "ratio_to_1936_2010": abs(rate) / abs(OBSERVED_ELEV_RATE.value),
        }
        for scenario, projected in PROJECTIONS.items():
            row[f"percent_of_{scenario}"] = 100.0 * abs(rate) / abs(projected)
        rows.append(row)
    return pd.DataFrame(rows).set_index("interval")


def historical_contrast(interval: str = "13_24") -> dict[str, float]:
    """How much faster glaciers are thinning now than over 1936-2010, glacier by glacier.

    Restricted to glaciers outside Austfonna so that the comparison matches the
    domain Geyman et al. report on.
    """
    merged = matched_glaciers(interval)
    outside = merged[merged["zone_label"] != "A"]
    ours = common.area_weighted(outside["our_elev_rate"], outside["area_km2"])
    theirs = common.area_weighted(outside["geyman_elev_rate"], outside["area_km2"])
    finite = outside[np.isfinite(outside["elev_ratio"])]
    return {
        "n": len(outside),
        "area_km2": float(outside["area_km2"].sum()),
        "our_area_weighted": ours,
        "geyman_area_weighted": theirs,
        "ratio_of_area_weighted": ours / theirs,
        "median_per_glacier_ratio": float(finite["elev_ratio"].median()),
        "percent_thinning_faster_now": 100.0
        * float((outside["our_elev_rate"] < outside["geyman_elev_rate"]).mean()),
    }


def summary() -> dict[str, object]:
    """Numbers cited in the manuscript, for ``tables/statistics.tex``."""
    benchmark = projection_benchmark()
    contrast = historical_contrast()
    return {
        "observed_elev_rate": OBSERVED_ELEV_RATE.value,
        "observed_mass_rate": OBSERVED_MASS_RATE.value,
        "projection_rcp_twentysix": PROJECTIONS["rcp26"],
        "projection_rcp_fortyfive": PROJECTIONS["rcp45"],
        "projection_rcp_eightyfive": PROJECTIONS["rcp85"],
        "our_elev_rate_excl_austfonna_start_end": round(
            float(benchmark.loc["2013-2024", "our_elev_rate_excl_austfonna"]), 2
        ),
        "our_elev_rate_excl_austfonna_mid_end": round(
            float(benchmark.loc["2019-2024", "our_elev_rate_excl_austfonna"]), 2
        ),
        "percent_of_rcp_eightyfive_start_end": round(
            float(benchmark.loc["2013-2024", "percent_of_rcp85"])
        ),
        "percent_of_rcp_eightyfive_mid_end": round(
            float(benchmark.loc["2019-2024", "percent_of_rcp85"])
        ),
        "ratio_to_baseline_start_end": reporting.format_float(
            float(benchmark.loc["2013-2024", "ratio_to_1936_2010"]), 1
        ),
        "ratio_to_baseline_mid_end": reporting.format_float(
            float(benchmark.loc["2019-2024", "ratio_to_1936_2010"]), 1
        ),
        "n_matched": contrast["n"],
        "matched_area": round(contrast["area_km2"]),
        "percent_thinning_faster_now": round(contrast["percent_thinning_faster_now"]),
        "median_per_glacier_ratio": reporting.format_float(
            contrast["median_per_glacier_ratio"], 1
        ),
    }
