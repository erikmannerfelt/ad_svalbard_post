"""Comparison with the ICESat-2 estimate of Wang et al. (2022).

There is no data publication for this study, so the module carries the values
transcribed from their Tables 1 and 2 (Svalbard row) together with the reconversion
that makes them comparable with ours.

Their headline mass rate is about half ours, but most of that gap is not
observational. Their reported *volume* change differs from ours by only about
1.2 sigma; the headline gap opens up because their preferred mass conversion uses a
zonal firn/ice scheme whose implied bulk factor is only ~600 kg m^-3. Reconverting
their own volume change at the 850 kg m^-3 of Huss (2013) removes roughly 40% of the
apparent discrepancy. Huss (2013) also warns that conversion factors outside
750-900 kg m^-3 arise from short 1-3 year observation windows, which is exactly
their 2.24-year span.
"""

from __future__ import annotations

from typing import Final

from . import common

#: ATL11 cycles 3-11 span 29 March 2019 to 23 June 2021 (their Section 3.1).
SPAN_YEARS: Final = 2.24
WINDOW: Final = "29 Mar 2019 - 23 Jun 2021"

VOLUME_RATE: Final = common.Estimate(-24.759, 7.598)  # km^3 a^-1
ELEV_RATE: Final = common.Estimate(-0.775, 0.225)  # m a^-1
AREA_KM2: Final = 33771.0

#: Their three conversion schemes; scheme 3 is the one they headline.
MASS_RATE_917: Final = common.Estimate(-22.704, 6.968)
MASS_RATE_FIRN: Final = common.Estimate(-12.627, 3.875)
MASS_RATE_HEADLINE: Final = common.Estimate(-14.843, 4.024)


def implied_bulk_density(mass_rate: common.Estimate = MASS_RATE_HEADLINE) -> float:
    """Bulk volume-to-mass factor implied by a reported mass and volume pair, kg m^-3."""
    return abs(mass_rate.value / VOLUME_RATE.value) * 1000.0


def reconverted_mass_rate() -> common.Estimate:
    """Their reported volume change, converted at the density this study uses."""
    return VOLUME_RATE.to_mass()


def density_attribution(our_mass_rate: float) -> dict[str, float]:
    """Split the apparent mass discrepancy into density and residual components.

    Parameters
    ----------
    our_mass_rate
        Our 2019-2024 mass rate in Gt a^-1, for the comparison.
    """
    reconverted = reconverted_mass_rate()
    published_gap = abs(our_mass_rate - MASS_RATE_HEADLINE.value)
    common_gap = abs(our_mass_rate - reconverted.value)
    return {
        "implied_bulk_density": implied_bulk_density(),
        "reconverted_mass_rate": reconverted.value,
        "reconverted_mass_rate_err": reconverted.error or float("nan"),
        "published_gap": published_gap,
        "gap_at_common_density": common_gap,
        "percent_explained_by_density": 100.0
        * (published_gap - common_gap)
        / published_gap,
    }


def summary(
    our_volume_rate: common.Estimate, our_mass_rate: float
) -> dict[str, object]:
    """Numbers cited in the manuscript, for ``tables/statistics.tex``.

    Parameters
    ----------
    our_volume_rate
        Our 2019-2024 volume rate, for the sigma comparison.
    our_mass_rate
        Our 2019-2024 mass rate in Gt a^-1.
    """
    attribution = density_attribution(our_mass_rate)
    return {
        "window": WINDOW,
        "span_years": SPAN_YEARS,
        "vol_rate": VOLUME_RATE.value,
        "vol_rate_err": VOLUME_RATE.error,
        "elev_rate": ELEV_RATE.value,
        "mass_rate_headline": MASS_RATE_HEADLINE.value,
        "implied_bulk_density": round(attribution["implied_bulk_density"]),
        "reconverted_mass_rate": round(attribution["reconverted_mass_rate"], 1),
        "percent_explained_by_density": round(
            attribution["percent_explained_by_density"]
        ),
        "sigma_from_ours_volume": round(
            common.sigma_difference(our_volume_rate, VOLUME_RATE), 1
        ),
        "area": int(AREA_KM2),
    }
