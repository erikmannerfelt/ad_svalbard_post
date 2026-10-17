"""Shared helpers for comparing this study with published Svalbard estimates.

Each sibling module wraps one published dataset. The conventions they share are
defined here: where downloaded source data is cached, how it is fetched, the value
type used to carry a rate with its uncertainty, and the RGI6-to-RGI7 glacier links
needed to join any RGI6-indexed product to our outlines.

Source archives are downloaded on demand rather than committed, so that the
comparison is reproducible from the URLs alone. A local mirror directory is
consulted first, which keeps a fresh checkout from re-downloading files that are
already on disk (some are >100 MB).
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from ..config import INPUT_DIR, PROJECT_DIR

COMPARISON_DIR: Final = INPUT_DIR / "comparisons"

# Files a user may already have downloaded by hand are picked up from here rather
# than fetched again. Nothing is required to be present.
MIRROR_DIR: Final = PROJECT_DIR.parent / "resources"

#: The full RGI7-RGI6 overlap table, with clusters of connected glaciers. The table
#: shipped inside the RGI7 archive holds only 36 one-to-one links for this region.
RGI6_LINKS_FILENAME: Final = "RGI2000-v7.0-G-07_svalbard_jan_mayen-rgi6_links.csv"
RGI6_LINKS_URL: Final = (
    "https://cluster.klima.uni-bremen.de/~fmaussion/misc/rgi7_data/rgi7_rgi6_links/"
    + RGI6_LINKS_FILENAME
)

SAMPLED_PATH: Final = PROJECT_DIR / "cache" / "outlines_sampled.arrow"

#: Volume-to-mass conversion used by this study and by most sources compared here.
RHO_AW: Final = 0.850  # Gt per km^3
RHO_AW_ERR: Final = 0.060


@dataclass(frozen=True)
class Estimate:
    """A rate with a 1-sigma uncertainty, in whatever unit the caller is using."""

    value: float
    error: float | None = None

    def __str__(self) -> str:
        if self.error is None:
            return f"{self.value:.3f}"
        return f"{self.value:.3f} +/- {self.error:.3f}"

    def to_mass(self) -> Estimate:
        """Convert a volume rate (km^3 a^-1) to a mass rate (Gt a^-1).

        Propagates the volume and density uncertainties in quadrature.
        """
        value = self.value * RHO_AW
        if self.error is None:
            return Estimate(value, abs(self.value) * RHO_AW_ERR)
        error = float(np.hypot(self.error * RHO_AW, self.value * RHO_AW_ERR))
        return Estimate(value, error)


def sigma_difference(a: Estimate, b: Estimate) -> float:
    """Difference between two estimates in units of their combined 1-sigma.

    Treats the uncertainties as independent, which is appropriate when the estimates
    come from different sensors and processing chains.
    """
    if a.error is None or b.error is None:
        raise ValueError("both estimates need an uncertainty to compare in sigma")
    return abs(a.value - b.value) / float(np.hypot(a.error, b.error))


def fetch(url: str, filename: str, refresh_download: bool = False) -> Path:
    """Return a local path to ``filename``, downloading from ``url`` if needed.

    Resolution order: the cached copy under ``input/comparisons``; then the local
    mirror directory; then the network. The download is written to a uniquely named
    temporary sibling and installed with :meth:`Path.replace`, so an interrupted
    fetch cannot leave a truncated file that looks usable.

    Parameters
    ----------
    url
        Source URL, recorded in the calling module next to the dataset it serves.
    filename
        Name to store the file under in the cache directory.
    refresh_download
        Re-download even if a cached copy exists.

    Returns
    -------
    Path
        Location of the local file.
    """
    target = COMPARISON_DIR / filename
    if target.is_file() and not refresh_download:
        return target

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    mirrored = MIRROR_DIR / filename
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    try:
        if mirrored.is_file() and not refresh_download:
            shutil.copyfile(mirrored, temporary)
        else:
            print(f"downloading {filename} from {url}")
            with (
                urllib.request.urlopen(url) as response,
                open(temporary, "wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_zip_member(archive: Path, member: str) -> bytes:
    """Read one member of a zip archive into memory."""
    with zipfile.ZipFile(archive) as opened, opened.open(member) as handle:
        return handle.read()


def load_rgi6_links(refresh_download: bool = False) -> pd.DataFrame:
    """Load the RGI7-to-RGI6 overlap links for region 7.

    One row per overlapping pair, including many-to-many cases where glaciers were
    split or merged between the inventories.

    Returns
    -------
    pandas.DataFrame
        Columns ``rgi7_id``, ``rgi6_id``, ``overlap_area_km2``, the two reciprocal
        area fractions and ``cluster_id`` (connected group of linked glaciers).
    """
    path = fetch(RGI6_LINKS_URL, RGI6_LINKS_FILENAME, refresh_download=refresh_download)
    links = pd.read_csv(path)
    return links[
        [
            "rgi7_id",
            "rgi6_id",
            "overlap_area_km2",
            "rgi7_area_fraction",
            "rgi6_area_fraction",
            "cluster_id",
        ]
    ]


def load_our_glaciers(interval: str = "13_18") -> pd.DataFrame:
    """Load our per-glacier results with centroids, for one canonical interval.

    Parameters
    ----------
    interval
        Interval shorthand, one of ``13_18``, ``19_24`` or ``13_24``.
    """
    import geopandas as gpd

    if not SAMPLED_PATH.is_file():
        raise FileNotFoundError(f"sampled outlines not found: {SAMPLED_PATH}")
    outlines = gpd.read_feather(SAMPLED_PATH).set_geometry(f"geometry_{interval}")
    outlines["centroid_north_km"] = outlines.geometry.centroid.y / 1000.0
    outlines["area_km2"] = outlines[f"area_{interval}"] / 1e6
    columns = [
        "rgi_id",
        "glims_id",
        "glac_name",
        "zone_label",
        "term_type",
        f"surging_{interval}",
        f"slope_{interval}",
        "area_km2",
        "centroid_north_km",
    ]
    return pd.DataFrame(outlines[columns]).rename(
        columns={f"surging_{interval}": "surging", f"slope_{interval}": "our_elev_rate"}
    )


def area_weighted(values: pd.Series, areas: pd.Series) -> float:
    """Area-weighted mean, which is the correct aggregation for elevation rates."""
    return float(np.average(values, weights=areas))
