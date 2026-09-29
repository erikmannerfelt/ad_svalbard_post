"""Comparison with the CryoSat-2 swath-altimetry budget of Morris et al. (2020).

This is the direct precedent for the surge/non-surge partition that this study
extends: they split Svalbard's 2011-2017 mass loss into a non-surging and a surging
component, using the same 850 kg m^-3 conversion we do.

Their Norwegian Polar Institute data publication ships the gridded 1 km elevation
change, the ice fraction, the subregion map and -- most usefully -- **the surge mask
they used**, all in EPSG:32633, the projection of our own product. That makes it
possible to recompute the partition on a common grid under both surge definitions
and so measure how much of the difference between their 31% and our 49% is
definitional rather than observational (:func:`partition_on_common_grid`).

Note that their surge share counts actively surging *ice* delineated as basin
outlines, whereas ours counts whole *glaciers* affected by an observed surge
overlapping the interval. Ours is necessarily the broader category.
"""

from __future__ import annotations

import io
import zipfile
from typing import Final

import numpy as np
import pandas as pd
import rasterio
import rasterio.features

from . import common

URL: Final = "https://api.npolar.no/dataset/5776cb31-5858-48b6-a9cf-412106c34987/_file/_all?format=zip"
FILENAME: Final = "npolar.no-dataset-5776cb31-5858-48b6-a9cf-412106c34987.zip"

DHDT_MEMBER: Final = "Svalbard_dhdt_extrapolated_33X.tif"
SURGE_MASK_MEMBER: Final = "Svalbard_surge_mask_33X.tif"
ICE_FRACTION_MEMBER: Final = "Svalbard_ice_fraction_1km_33X.tif"
REGIONAL_MEMBER: Final = "Svalbard_dMdt_array.csv"

#: Published Table 1 figures for the whole archipelago, 2011-2017.
TOTAL_MASS_RATE: Final = common.Estimate(-16.0, 3.0)  # Gt a^-1
NONSURGING_MASS_RATE: Final = common.Estimate(-11.02)  # Gt a^-1
SURGING_MASS_RATE: Final = common.Estimate(-4.98)  # Gt a^-1
NONSURGING_SPECIFIC_BALANCE: Final = -0.37  # m w.e. a^-1
STORISSTRAUMEN_MASS_RATE: Final = -3.56  # Gt a^-1
AREA_KM2: Final = 33773.0


def archive(refresh_download: bool = False):
    """Return the opened data-publication archive."""
    return zipfile.ZipFile(
        common.fetch(URL, FILENAME, refresh_download=refresh_download)
    )


def load_regional_partition(refresh_download: bool = False) -> pd.DataFrame:
    """Per-subregion mass-change rates split into total, non-surging and surging.

    This is the machine-readable form of their Table 1. Values differ from the
    printed table by a few hundredths of a Gt a^-1, which is rounding.
    """
    with archive(refresh_download=refresh_download) as opened:
        frame = pd.read_csv(
            io.BytesIO(opened.read(REGIONAL_MEMBER)),
            header=None,
            names=["subregion", "total", "nonsurging", "surging"],
        )
    return frame.set_index("subregion")


def _read_grid(member: str, refresh_download: bool = False) -> tuple[np.ndarray, dict]:
    """Read one raster member, returning the array and its rasterio profile."""
    with (
        archive(refresh_download=refresh_download) as opened,
        rasterio.open(io.BytesIO(opened.read(member))) as source,
    ):
        return source.read(1), source.profile


def partition_on_common_grid(interval: str = "13_18") -> dict[str, float]:
    """Recompute the surge partition of *their* data under *our* surge definition.

    Their gridded elevation change is integrated over their ice fraction on their own
    1 km grid, then split two ways: by the surge mask they published, and by
    rasterising our surge-affected glacier outlines onto the same grid. The
    difference between the two shares isolates the definitional component of the
    31%-versus-49% gap, because the underlying observations are identical.

    Returns
    -------
    dict
        Volume and mass rates for each partition, and the surge share under each
        definition, in percent of the archipelago total.
    """
    import geopandas as gpd

    dhdt, profile = _read_grid(DHDT_MEMBER)
    surge_mask, _ = _read_grid(SURGE_MASK_MEMBER)
    ice_fraction, _ = _read_grid(ICE_FRACTION_MEMBER)

    cell_area_km2 = abs(profile["transform"].a * profile["transform"].e) / 1e6
    valid = np.isfinite(dhdt) & np.isfinite(ice_fraction) & (ice_fraction > 0)

    outlines = gpd.read_feather(common.SAMPLED_PATH).set_geometry(
        f"geometry_{interval}"
    )
    outlines = outlines.to_crs(profile["crs"])
    surging = outlines[outlines[f"surging_{interval}"]]
    our_mask = rasterio.features.rasterize(
        ((geometry, 1) for geometry in surging.geometry),
        out_shape=dhdt.shape,
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
    ).astype(bool)

    # Volume change per cell in km^3 a^-1: dh/dt [m] x ice area [km^2] / 1000.
    volume = np.where(valid, dhdt * ice_fraction * cell_area_km2 / 1000.0, 0.0)
    total = float(volume.sum())
    their_surge = float(volume[valid & (surge_mask > 0)].sum())
    our_surge = float(volume[valid & our_mask].sum())

    return {
        "total_vol_rate": total,
        "total_mass_rate": total * common.RHO_AW,
        "their_surge_vol_rate": their_surge,
        "our_surge_vol_rate": our_surge,
        "their_surge_share_percent": 100.0 * their_surge / total,
        "our_surge_share_percent": 100.0 * our_surge / total,
        "ice_area_km2": float(ice_fraction[valid].sum() * cell_area_km2),
        "their_surge_area_km2": float(
            ice_fraction[valid & (surge_mask > 0)].sum() * cell_area_km2
        ),
        "our_surge_area_km2": float(
            ice_fraction[valid & our_mask].sum() * cell_area_km2
        ),
    }


def validate_grid_integration(refresh_download: bool = False) -> pd.DataFrame:
    """Check that integrating their published grid reproduces their published totals.

    It does, exactly, for five of the six subregions. Austfonna is the exception:
    the grid integrates to about -6.6 Gt a^-1 against their published -4.5. That is
    the Storisstraumen surge advance, which they state was accounted for separately
    by digitising pre- and post-surge calving fronts rather than through the grid.
    The whole-archipelago difference (-18.0 versus the published -16.0 Gt a^-1) is
    entirely this one correction, so grid-derived totals should not be quoted as
    theirs without saying so.

    Returns
    -------
    pandas.DataFrame
        Per-subregion integrated and published mass rates, in Gt a^-1.
    """
    dhdt, profile = _read_grid(DHDT_MEMBER, refresh_download=refresh_download)
    ice_fraction, _ = _read_grid(ICE_FRACTION_MEMBER)
    division, _ = _read_grid("Svalbard_division_33X.tif")
    cell_area_km2 = abs(profile["transform"].a * profile["transform"].e) / 1e6
    valid = np.isfinite(dhdt) & np.isfinite(ice_fraction) & (ice_fraction > 0)

    published = load_regional_partition()["total"]
    rows = []
    for index, subregion in enumerate(published.index, start=1):
        selected = valid & (division == index)
        volume = float(
            (dhdt[selected] * ice_fraction[selected] * cell_area_km2 / 1000.0).sum()
        )
        rows.append(
            {
                "subregion": subregion,
                "integrated_mass_rate": volume * common.RHO_AW,
                "published_mass_rate": float(published.iloc[index - 1]),
                "area_km2": float(ice_fraction[selected].sum() * cell_area_km2),
            }
        )
    frame = pd.DataFrame(rows).set_index("subregion")
    frame["difference"] = frame["integrated_mass_rate"] - frame["published_mass_rate"]
    return frame


def cross_checks(our_stats: dict[str, object]) -> dict[str, float]:
    """The two comparisons strong enough to lead with in the manuscript.

    Their non-surging specific balance against ours, and the share of the archipelago
    total carried by Storisstraumen in each study.

    Parameters
    ----------
    our_stats
        The ``changes`` block of this study's statistics.
    """
    our_nonsurge_specific = (
        float(our_stats["slope_start_mid"]["nonsurging"]["elev_rate"]) * common.RHO_AW
    )
    our_storisstraumen = float(
        our_stats["slope_start_end"]["per_glacier"]["storisstraumen"]["vol_rate"]
    )
    our_total = float(our_stats["slope_start_end"]["all"]["vol_rate"])
    return {
        "morris_nonsurging_specific": NONSURGING_SPECIFIC_BALANCE,
        "our_nonsurging_specific_13_18": our_nonsurge_specific,
        "nonsurging_specific_difference": abs(
            our_nonsurge_specific - NONSURGING_SPECIFIC_BALANCE
        ),
        "morris_storisstraumen_share_percent": 100.0
        * STORISSTRAUMEN_MASS_RATE
        / TOTAL_MASS_RATE.value,
        "our_storisstraumen_share_percent": 100.0 * our_storisstraumen / our_total,
        "morris_surge_share_percent": 100.0
        * SURGING_MASS_RATE.value
        / TOTAL_MASS_RATE.value,
    }


def summary(our_stats: dict[str, object]) -> dict[str, object]:
    """Numbers cited in the manuscript, for ``tables/statistics.tex``."""
    checks = cross_checks(our_stats)
    partition = partition_on_common_grid()
    return {
        "total_mass_rate": TOTAL_MASS_RATE.value,
        "total_mass_rate_err": TOTAL_MASS_RATE.error,
        "nonsurging_mass_rate": NONSURGING_MASS_RATE.value,
        "surging_mass_rate": SURGING_MASS_RATE.value,
        "surge_share_percent": round(checks["morris_surge_share_percent"]),
        "nonsurging_specific": NONSURGING_SPECIFIC_BALANCE,
        "our_nonsurging_specific": round(checks["our_nonsurging_specific_13_18"], 2),
        "storisstraumen_mass_rate": STORISSTRAUMEN_MASS_RATE,
        "storisstraumen_share_percent": round(
            checks["morris_storisstraumen_share_percent"]
        ),
        "our_storisstraumen_share_percent": round(
            checks["our_storisstraumen_share_percent"]
        ),
        "area": int(AREA_KM2),
        "regridded_their_surge_share_percent": round(
            partition["their_surge_share_percent"]
        ),
        "regridded_our_surge_share_percent": round(
            partition["our_surge_share_percent"]
        ),
    }


def load_monthly_series(refresh_download: bool = False) -> dict[str, pd.DataFrame]:
    """Load the per-subregion monthly cumulative mass-change series, 2011-2017.

    Columns are cumulative Gt for the subregion total, its non-surging ice and its
    surging ice. Vestfonna has no surging component and so carries one column fewer.
    Fitting a linear trend to these reproduces their published rates to about 2%.
    """
    files = {
        "Austfonna": "Aust_dM_array.csv",
        "BarentsEdge": "BarEd_dM_array.csv",
        "Northeast": "NESbn_dM_array.csv",
        "Northwest": "NWSbn_dM_array.csv",
        "South": "SoSbn_dM_array.csv",
        "Vestfonna": "Vest_dM_array.csv",
    }
    series = {}
    with archive(refresh_download=refresh_download) as opened:
        for name, member in files.items():
            frame = pd.read_csv(io.BytesIO(opened.read(member)), header=None)
            columns = ["year", "month", "total", "nonsurging"]
            if frame.shape[1] == 5:
                columns.append("surging")
            frame.columns = columns
            if "surging" not in frame:
                frame["surging"] = 0.0
            frame["decimal_year"] = frame["year"] + (frame["month"] - 0.5) / 12.0
            series[name] = frame
    return series


def period_sensitivity(start: float = 2013.0, end: float = 2018.0) -> dict[str, float]:
    """How much of the surge share depends on their window rather than their data.

    Their record covers 2011-2017 while our first sub-period is 2013-2018, so the
    obvious explanation for a lower surge share is that their window predates the
    peak of the Storisstraumen surge. Refitting their own monthly series over the
    overlapping years tests that, and it moves the surge share by only about one
    percentage point: **period is not the explanation**.
    """
    series = load_monthly_series()

    def trend(frame: pd.DataFrame, column: str, low: float, high: float) -> float:
        window = frame[(frame["decimal_year"] >= low) & (frame["decimal_year"] < high)]
        return float(np.polyfit(window["decimal_year"], window[column], 1)[0])

    out: dict[str, float] = {}
    for label, (low, high) in {
        "full": (2011.0, 2018.0),
        "overlap": (start, end),
    }.items():
        total = sum(trend(frame, "total", low, high) for frame in series.values())
        surging = sum(trend(frame, "surging", low, high) for frame in series.values())
        out[f"{label}_total_mass_rate"] = total
        out[f"{label}_surging_mass_rate"] = surging
        out[f"{label}_surge_share_percent"] = 100.0 * surging / total
    out["surge_share_shift"] = (
        out["overlap_surge_share_percent"] - out["full_surge_share_percent"]
    )
    return out


def observation_coverage(interval: str = "13_18") -> dict[str, float]:
    """Fraction of each class resting on real observations rather than extrapolation.

    Their published grid comes in an extrapolated and an unextrapolated version, so
    the two can be differenced to find where CryoSat-2 actually measured. Surge
    basins turn out to be *better* observed than the archipelago average, which rules
    out sparse coverage as the reason their surge signal is smaller than ours.
    """
    import geopandas as gpd

    extrapolated, profile = _read_grid(DHDT_MEMBER)
    unextrapolated, _ = _read_grid("Svalbard_dhdt_unextrapolated_33X.tif")
    ice_fraction, _ = _read_grid(ICE_FRACTION_MEMBER)
    cell_area_km2 = abs(profile["transform"].a * profile["transform"].e) / 1e6
    valid = np.isfinite(extrapolated) & np.isfinite(ice_fraction) & (ice_fraction > 0)
    observed = np.isfinite(unextrapolated)

    outlines = gpd.read_feather(common.SAMPLED_PATH).set_geometry(
        f"geometry_{interval}"
    )
    outlines = outlines.to_crs(profile["crs"])
    our_mask = rasterio.features.rasterize(
        (
            (geometry, 1)
            for geometry in outlines[outlines[f"surging_{interval}"]].geometry
        ),
        out_shape=extrapolated.shape,
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
    ).astype(bool)

    out: dict[str, float] = {}
    for label, selected in (
        ("surge", valid & our_mask),
        ("nonsurge", valid & ~our_mask),
        ("all", valid),
    ):
        area = float(ice_fraction[selected].sum() * cell_area_km2)
        covered = float(ice_fraction[selected & observed].sum() * cell_area_km2)
        out[f"{label}_area_km2"] = area
        out[f"{label}_observed_percent"] = 100.0 * covered / area
    return out
