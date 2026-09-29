"""Comparison with the WGMS annual mass-change estimates for Svalbard and Jan Mayen.

The WGMS `amce` product temporally downscales multi-year geodetic elevation changes
with annual in-situ measurements. The method is described by Dussaillant et al.
(2025); the dataset itself is versioned separately and is reissued as the underlying
Fluctuations of Glaciers database grows.

**Which version to use.** The 2026-02-10 release names Dussaillant et al. (2025) as
its method paper, so the two are not competing estimates: one is the method, the
other is the current data. Over the window Dussaillant et al. published
(2015-2024) the newer release gives -24.64 Gt a^-1 against their -24.9, a difference
of about 1%. The newer release is preferred here because it supplies an annual
series, which lets us compute means over *our* sub-periods rather than only over the
window that happened to be published, and because it extends to 2025. Cite
Dussaillant et al. (2025) for the method and the dataset DOI for the numbers.

The annual series also provides an independent check on the central claim of this
study, since the two sub-period means can be computed directly from it.
"""

from __future__ import annotations

import io
from typing import Final

import numpy as np
import pandas as pd

from . import common

VERSION: Final = "2026-02-10"
URL: Final = f"https://wgms.ch/downloads/wgms-amce-{VERSION}.zip"
FILENAME: Final = f"wgms-amce-{VERSION}.zip"
REGION_MEMBER: Final = "region/SJM.csv"
REGION_CODE: Final = "SJM"  # Svalbard and Jan Mayen

#: As published by Dussaillant et al. (2025) for 2015-2024, for cross-reference.
DUSSAILLANT_2015_2024: Final = common.Estimate(-24.9, 9.4)  # Gt a^-1


def load_regional_series(refresh_download: bool = False) -> pd.DataFrame:
    """Load the annual regional mass-change series for Svalbard and Jan Mayen.

    Returns
    -------
    pandas.DataFrame
        Indexed by hydrological year, with ``gt``, ``gt_sigma``, ``mwe``,
        ``mwe_sigma`` and ``area_km2``.
    """
    path = common.fetch(URL, FILENAME, refresh_download=refresh_download)
    payload = common.read_zip_member(path, REGION_MEMBER)
    return pd.read_csv(io.StringIO(payload.decode()), comment="#").set_index("year")


def window_mean(
    start: int, end: int, refresh_download: bool = False
) -> dict[str, float]:
    """Mean mass-change rate over a range of hydrological years, inclusive.

    Two uncertainties are returned because the annual errors are neither independent
    nor perfectly correlated, and the dataset does not state their covariance. The
    independent bound divides by sqrt(n); the correlated bound keeps the mean annual
    sigma. The truth lies between, and the correlated bound is the conservative one
    to quote.
    """
    series = load_regional_series(refresh_download=refresh_download)
    window = series.loc[start:end]
    if window.empty:
        raise ValueError(f"no data for {start}-{end}")
    mean_sigma = float(window["gt_sigma"].mean())
    return {
        "mass_rate": float(window["gt"].mean()),
        "sigma_independent": mean_sigma / float(np.sqrt(len(window))),
        "sigma_correlated": mean_sigma,
        "specific_balance": float(window["mwe"].mean()),
        "n_years": len(window),
        "area_km2": float(window["area_km2"].mean()),
    }


def version_comparison() -> dict[str, float]:
    """Quantify how much the current release differs from the published version.

    Compares this release over Dussaillant et al.'s published 2015-2024 window with
    their published value, which is the basis for preferring the newer data.
    """
    window = window_mean(2015, 2024)
    return {
        "amce_2015_2024": window["mass_rate"],
        "dussaillant_published_2015_2024": DUSSAILLANT_2015_2024.value,
        "difference": window["mass_rate"] - DUSSAILLANT_2015_2024.value,
        "percent_difference": 100.0
        * abs(window["mass_rate"] - DUSSAILLANT_2015_2024.value)
        / abs(DUSSAILLANT_2015_2024.value),
    }


def sub_period_contrast() -> dict[str, float]:
    """Independent check on this study's central claim, from the annual series.

    Our headline result is that the archipelago-wide loss rate rose sharply between
    2013-2018 and 2019-2024. Because this product resolves individual years, that
    contrast can be evaluated in it directly.
    """
    first = window_mean(2013, 2018)
    second = window_mean(2019, 2024)
    full = window_mean(2013, 2024)
    return {
        "start_mid_mass_rate": first["mass_rate"],
        "mid_end_mass_rate": second["mass_rate"],
        "start_end_mass_rate": full["mass_rate"],
        "ratio": second["mass_rate"] / first["mass_rate"],
        "difference": second["mass_rate"] - first["mass_rate"],
        "start_mid_sigma_correlated": first["sigma_correlated"],
        "mid_end_sigma_correlated": second["sigma_correlated"],
    }


def summary() -> dict[str, object]:
    """Numbers cited in the manuscript, for ``tables/statistics.tex``."""
    contrast = sub_period_contrast()
    version = version_comparison()
    full = window_mean(2013, 2024)
    return {
        "version": VERSION,
        "mass_rate_start_mid": round(contrast["start_mid_mass_rate"], 1),
        "mass_rate_mid_end": round(contrast["mid_end_mass_rate"], 1),
        "mass_rate_start_end": round(contrast["start_end_mass_rate"], 1),
        "sigma_start_end": round(full["sigma_correlated"], 1),
        "ratio": round(contrast["ratio"], 1),
        "area": round(full["area_km2"]),
        "amce_dussaillant_window": round(version["amce_2015_2024"], 1),
        "dussaillant_published": DUSSAILLANT_2015_2024.value,
        "version_percent_difference": round(version["percent_difference"]),
    }
