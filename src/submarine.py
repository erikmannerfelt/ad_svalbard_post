"""Submarine (below-sea-level) glacier volume change from marine-front retreat and advance.

Reads the published products of the sibling `SvalbardSubmarineRetreat` repo
rather than computing anything geometric here. That repo builds, per RGI7
tidewater glacier, the retreat and advance polygons between the NPI coastlines
of 2012, 2018 and 2023, merges six bed/ice-thickness products into gap-free
100 m beds (IBCAO soundings at sea, the model's own inversion under former
ice, a harmonic fill in between), and integrates each bed's depth below sea
level over each polygon (`submarine_volume_change.arrow`). Its `AGENTS.md`
and `README.md` document the construction.

**Bed models.** Four enter the estimate (`BED_MODELS`); the central value is
their mean and the uncertainty is half their range, both evaluated on the
*aggregated* total of whatever subset is reported (glacier, surge class,
archipelago). Bed-model error is systematic across neighbouring glaciers, so
it is never combined in quadrature across glaciers. With four models the
half-range is about one standard deviation in expectation, and it describes a
small, non-random set of published models more honestly than a sample
standard deviation would.

Two models are sampled upstream but excluded (`EXCLUDED_BED_MODELS`):

- SVIFT 1.0 is superseded by SVIFT 1.1. It matters for comparison with
  earlier estimates, though: the previous version of this module used
  G. Moholdt's precomputed polygons, whose depths are SVIFT-1.0-like, and gave
  -0.32 and -1.58 km^3/a for 2012-2018 and 2018-2023. The change *areas* agree
  with the new product to <3%; the difference is entirely depth (40 vs 65 m
  mean retreat depth for 2012-2018).
- Millan et al. (2022) is nonphysical at the glacier margins, which is exactly
  where this term is evaluated. The evidence is the upstream
  `bed_model_validation.csv` (`read_bed_model_validation`).

**Dates.** Rates divide by whole coastline years (6, 5, 11 a). The coastline
files carry no acquisition dates; the per-front dates available elsewhere
(July-September, the "2012" coastline partly 2011 imagery) change the rates by
at most 4.4%, well inside the bed-model spread (see the upstream `AGENTS.md`).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import geopandas as gpd
import pandas as pd

from . import tools
from .config import CRS_EPSG, PROJECT_DIR

#: Solid ice below the water line; see `main.tex`, "Conversion to mass change".
RHO_SUB: Final = 900.0

SUBMARINE_REPO_DIR: Final = PROJECT_DIR.parent / "SvalbardSubmarineRetreat"
SUBMARINE_PRODUCT_PATH: Final = (
    SUBMARINE_REPO_DIR / "products" / "submarine_volume_change.arrow"
)
BED_VALIDATION_PATH: Final = (
    SUBMARINE_REPO_DIR / "products" / "bed_model_validation.csv"
)

#: Bed models whose mean and half-range form the submarine estimate.
BED_MODELS: Final = ("svift_v11", "maffezzoli", "frank", "vanpelt")

#: Bed models present in the product but deliberately left out, with why.
EXCLUDED_BED_MODELS: Final = MappingProxyType(
    {
        "svift_v10": "superseded by SVIFT 1.1",
        "millan": "nonphysical at the glacier margins (see bed_model_validation.csv)",
    }
)

#: Letters-only labels for TeX macro keys (`reporting.to_tex_macros` cannot
#: take the digits in `svift_v11`).
MODEL_LABELS: Final = MappingProxyType(
    {
        "svift_v11": "svift",
        "svift_v10": "sviftold",
        "maffezzoli": "maffezzoli",
        "millan": "millan",
        "frank": "frank",
        "vanpelt": "vanpelt",
    }
)

#: Coastline years bounding each elevation-change interval. Coastlines are
#: late-summer, so 2012-2018 is the closest match to 2013-2018 (spring 2013 to
#: spring 2019), and so on.
COASTLINE_YEARS: Final = MappingProxyType(
    {"13_18": (2012, 2018), "19_24": (2018, 2023), "13_24": (2012, 2023)}
)

_CHANGE_SIGN: Final = MappingProxyType({"retreat": -1.0, "advance": 1.0})


def read_submarine_product() -> gpd.GeoDataFrame:
    """Read and validate `SUBMARINE_PRODUCT_PATH`.

    Returns
    -------
    GeoDataFrame
        One row per (glacier, interval, change type), EPSG:32633. The
        `<model>_volume_change_m3` columns are unsigned magnitudes (m^3);
        `change_type` gives the sign.
    """
    if not SUBMARINE_PRODUCT_PATH.is_file():
        raise FileNotFoundError(
            f"{SUBMARINE_PRODUCT_PATH} not found; build it in SvalbardSubmarineRetreat "
            "with volume.make_submarine_volume_product()"
        )
    product = gpd.read_feather(SUBMARINE_PRODUCT_PATH)

    required = {"rgi_id", "year_from", "year_to", "change_type", "area_m2"}
    required |= {
        f"{model}_{column}"
        for model in (*BED_MODELS, *EXCLUDED_BED_MODELS)
        for column in ("volume_change_m3", "interpolated_share")
    }
    missing = sorted(required - set(product.columns))
    if missing:
        raise ValueError(f"submarine product lacks columns {missing}")
    if product.crs is None or product.crs.to_epsg() != CRS_EPSG:
        raise ValueError(f"submarine product CRS is {product.crs}, expected {CRS_EPSG}")
    unknown_types = set(product["change_type"]) - set(_CHANGE_SIGN)
    if unknown_types:
        raise ValueError(f"unexpected change_type values {sorted(unknown_types)}")
    intervals = set(zip(product["year_from"], product["year_to"], strict=True))
    absent = set(COASTLINE_YEARS.values()) - intervals
    if absent:
        raise ValueError(
            f"submarine product lacks coastline intervals {sorted(absent)}"
        )
    return product


def read_bed_model_validation() -> pd.DataFrame:
    """The upstream comparison of raw bed models against IBCAO soundings at the fronts.

    Columns `cell_set` (``"own"`` or a ``+``-joined set of models sharing the
    same cells), `model`, `n_cells`, `median_bias_m`, `mean_bias_m`,
    `rmse_m` and `depth_ratio` (model over measured depth).
    """
    return pd.read_csv(BED_VALIDATION_PATH)


@dataclass(frozen=True)
class SubmarineChanges:
    """Per-glacier submarine volume change for one interval.

    Attributes
    ----------
    per_glacier
        Indexed by `rgi_id`. One signed volume-rate column per model in
        `BED_MODELS` and `EXCLUDED_BED_MODELS` (m^3/a, negative for net
        retreat), plus `retreat_area` and `advance_area` (m^2) and, per
        model, `<model>_gross_volume` (m^3, retreat plus advance, unsigned,
        not a rate) and `<model>_interpolated_volume` (the part of it resting
        on the harmonic fill rather than on a sounding or an inversion). Use
        `aggregate_submarine` to reduce any subset of rows to an estimate.
    dt_yr
        Whole years between the bounding coastlines.
    """

    per_glacier: pd.DataFrame
    dt_yr: float


def get_submarine_changes(
    interval_short: str, product: gpd.GeoDataFrame | None = None
) -> SubmarineChanges:
    """Per-glacier signed submarine volume rates for `interval_short`.

    Parameters
    ----------
    interval_short
        One of `COASTLINE_YEARS`, e.g. ``"13_18"``.
    product
        A frame from `read_submarine_product`; read if not given.
    """
    tools.get_interval(interval_short)  # validates the short code
    if product is None:
        product = read_submarine_product()
    year_from, year_to = COASTLINE_YEARS[interval_short]
    dt_yr = float(year_to - year_from)

    rows = product.loc[
        (product["year_from"] == year_from) & (product["year_to"] == year_to)
    ]
    sign = rows["change_type"].map(_CHANGE_SIGN)
    models = (*BED_MODELS, *EXCLUDED_BED_MODELS)

    signed = pd.DataFrame(
        {model: sign * rows[f"{model}_volume_change_m3"] / dt_yr for model in models}
    )
    gross = pd.DataFrame(
        {f"{model}_gross_volume": rows[f"{model}_volume_change_m3"] for model in models}
    )
    interpolated = pd.DataFrame(
        {
            f"{model}_interpolated_volume": rows[f"{model}_volume_change_m3"]
            * rows[f"{model}_interpolated_share"]
            for model in models
        }
    )
    area = pd.DataFrame(
        {
            f"{change_type}_area": rows["area_m2"].where(
                rows["change_type"] == change_type, 0.0
            )
            for change_type in _CHANGE_SIGN
        }
    )
    per_glacier = (
        pd.concat([signed, gross, interpolated, area], axis=1)
        .groupby(rows["rgi_id"])
        .sum()
    )
    return SubmarineChanges(per_glacier=per_glacier, dt_yr=dt_yr)


def aggregate_submarine(per_glacier_subset: pd.DataFrame) -> tuple[float, float]:
    """Submarine volume rate and uncertainty for a set of glaciers (m^3/a).

    Each model's rate is summed over the subset first; the estimate is the
    mean of those totals over `BED_MODELS` and the uncertainty half their
    range. See the module docstring for why.

    >>> frame = pd.DataFrame({"svift_v11": [-1.0, -1.0], "maffezzoli": [-0.5, -0.5],
    ...                       "frank": [-1.0, 0.0], "vanpelt": [-1.5, -0.5]})
    >>> aggregate_submarine(frame)
    (-1.5, 0.5)
    """
    totals = per_glacier_subset.loc[:, list(BED_MODELS)].sum()
    return float(totals.mean()), float((totals.max() - totals.min()) / 2)


def model_totals(per_glacier_subset: pd.DataFrame) -> dict[str, float]:
    """Every model's summed rate over the subset (m^3/a), excluded models included."""
    return {
        model: float(per_glacier_subset[model].sum())
        for model in (*BED_MODELS, *EXCLUDED_BED_MODELS)
    }


def interpolated_share(per_glacier_subset: pd.DataFrame, model: str) -> float:
    """Fraction of `model`'s gross (retreat plus advance) volume that rests on the harmonic fill."""
    gross = per_glacier_subset[f"{model}_gross_volume"].sum()
    if gross == 0:
        return float("nan")
    return float(per_glacier_subset[f"{model}_interpolated_volume"].sum() / gross)
