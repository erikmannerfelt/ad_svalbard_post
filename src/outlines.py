from __future__ import annotations

from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.ops

from . import tools
from .config import CACHE_DIR, CRS_EPSG, INPUT_DIR, PROJECT_DIR
from .tools import ForceRedo

#: Svalbard Surge Database V5 (Harcourt et al.): RGI 7.0 outlines for region 07 with
#: surge attributes. Pinned to the V5 record, not the concept DOI, so a new version
#: cannot change the outlines silently. V6 is expected to make several entries in
#: `make_outlines()`'s ``field_overrides`` redundant.
SURGE_DATABASE_URL: Final = (
    "https://zenodo.org/records/18033216/files/RGI_V7_Surge_Database.zip?download=1"
)
SURGE_DATABASE_PATH: Final = INPUT_DIR / "RGI_V7_Surge_Database.zip"
SURGE_DATABASE_LAYER: Final = (
    "RGI_V7_Surge_Database/RGI2000-v7.0-G-07_svalbard_jan_mayen_Surge_Database.shp"
)
COASTLINE_PATH: Final = INPUT_DIR / "Coast_annual.zip"
GLACIER_ZONES_PATH: Final = PROJECT_DIR / "shapes" / "glacier_zones.geojson"
OUTLINE_CORRECTIONS_PATH: Final = PROJECT_DIR / "shapes" / "outline_corrections.geojson"


def surge_overlaps_interval(term_value, onset_value, min_year, max_year) -> bool:
    def _tokens(value):
        if pd.isna(value):
            return []
        return [part.strip() for part in str(value).split(";") if part.strip()]

    onset_tokens = _tokens(onset_value)
    term_tokens = _tokens(term_value)
    for idx, onset_token in enumerate(onset_tokens):
        try:
            onset_year = int(onset_token)
        except ValueError:
            continue
        if onset_year > max_year:
            continue
        term_token = term_tokens[idx] if idx < len(term_tokens) else None
        if term_token is None:
            term_year = np.inf
        else:
            term_lower = term_token.lower()
            if term_lower in {"", "n/a", "not observed", "ongoing"}:
                term_year = np.inf
            else:
                try:
                    term_year = int(term_token)
                except ValueError:
                    term_year = np.inf
        if onset_year <= max_year and term_year > min_year:
            return True
    return False


def _write_feather_atomic(frame: gpd.GeoDataFrame, cache_path: Path) -> None:
    """Serialize to a temporary sibling, then atomically install it at `cache_path`.

    These caches take minutes to rebuild, so an interrupted write must not be able to
    leave a truncated file that later looks usable. The suffix stays `.arrow` because
    the serializer infers its format from it.
    """
    tmp_path = cache_path.with_suffix(f".tmp{cache_path.suffix}")
    try:
        frame.to_feather(tmp_path)
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def make_coastline_intervals(force_redo: ForceRedo = False) -> gpd.GeoDataFrame:
    """Union the annual coastlines into one land polygon per interval.

    The union is the *maximum* land extent over the interval, so for a retreating
    tidewater front it is effectively the extent at the start of the interval.
    """
    cache_path = CACHE_DIR / "coastline_intervals.arrow"

    if cache_path.is_file() and not force_redo:
        return gpd.read_feather(cache_path).set_index("key")  # type: ignore

    coastlines = {}
    for year in range(2013, 2025):
        coastline = (
            gpd.read_file(f"zip://{COASTLINE_PATH}!Coast{year}.shp")
            .to_crs(CRS_EPSG)
            .dissolve()
            .iloc[0]["geometry"]
        )
        coastlines[year] = coastline

    c_13_18 = shapely.ops.unary_union([g for yr, g in coastlines.items() if yr < 2019])
    c_19_24 = shapely.ops.unary_union([g for yr, g in coastlines.items() if yr >= 2019])
    c_13_24 = shapely.ops.unary_union([c_13_18, c_19_24])

    interval_to_geometry = {
        "13_18": c_13_18,
        "19_24": c_19_24,
        "13_24": c_13_24,
    }

    records = []
    for interval in tools.iter_intervals():
        records.append(
            {
                "key": interval.short,
                "name": interval.display_label,
                "geometry": interval_to_geometry[interval.short],
            }
        )

    out = gpd.GeoDataFrame(pd.DataFrame.from_records(records), geometry="geometry", crs=CRS_EPSG)

    _write_feather_atomic(out, cache_path)

    return gpd.read_feather(cache_path).set_index("key")  # type: ignore


def _read_outline_cache(cache_path: Path) -> gpd.GeoDataFrame:
    """Read the outline cache and derive the per-interval areas from its geometries.

    Areas are derived on read rather than stored so they cannot silently disagree with
    the geometries they came from.
    """
    out = gpd.read_feather(cache_path)
    for interval in tools.iter_intervals():
        out[interval.area_col] = out[interval.geometry_col].area
    return out  # type: ignore[return-value]


def _validate_corrections(
    corrections: gpd.GeoDataFrame,
    rgi7: gpd.GeoDataFrame,
    probably_tidewater: pd.Series,
) -> None:
    """Check each correction against what the pipeline will actually do with it.

    Two failure modes are worth catching before a multi-minute rebuild:

    1. An `rgi_id` that matches no RGI 7.0 outline. The correction would silently do
       nothing, which is indistinguishable from it having been applied.
    2. A `tidewater_extension` whose target is excluded from the coastline clip by
       `probably_tidewater`. Its seaward part would then be unioned in *literally*
       rather than trimmed back to the coastline, inventing ice over open water. See
       `make_outlines` for why the two fix types are treated differently.
    """
    known = set(rgi7["rgi_id"])
    unknown = sorted(set(corrections["rgi_id"]) - known)
    if unknown:
        raise RuntimeError(f"outline_corrections.geojson targets unknown rgi_id(s): {unknown}")

    clipped = set(rgi7.loc[probably_tidewater, "rgi_id"])
    unclipped_domains = sorted(
        corrections.loc[
            corrections["fix_type"].str.contains("tidewater_extension")
            & ~corrections["rgi_id"].isin(clipped),
            "rgi_id",
        ]
    )
    if unclipped_domains:
        raise RuntimeError(
            "tidewater_extension targets that are not coastline-clipped, so their seaward "
            f"part would be added literally: {unclipped_domains}"
        )


def make_outlines(
    force_redo: ForceRedo = False, refresh_download: bool = False
) -> gpd.GeoDataFrame:
    """Build period-specific glacier outlines from RGI 7.0, corrections and annual coastlines.

    Returns one geometry per glacier per interval (``geometry_13_18`` etc.), each being the
    RGI 7.0 outline with any active corrections from ``shapes/outline_corrections.geojson``
    applied, then — for glaciers that are or may be tidewater — clipped to the union of that
    interval's annual coastlines.

    The two extension types mean different things
    ---------------------------------------------
    ``land_extension`` is a **literal union**: the polygon delineates ice to add, and must be
    drawn tightly to it, because every square metre inside it is added to the target.

    ``tidewater_extension`` is an **attribution domain**: the polygon marks the region within
    which land that the coastline says exists should be attributed to the target, wherever
    the coastline extends beyond the RGI 7.0 outline. It does not delineate ice, and is drawn
    deliberately large — typically well out over open water.

    Both are applied by the same union below; what separates them is the coastline clip that
    follows. For a tidewater target the union is temporary and generous (the target briefly
    extends over ocean and over its own existing area), and the clip trims it back to exactly
    the land inside the domain.

    The practical consequence for a ``tidewater_extension`` is that a huge polygon adding
    almost no area is the expected outcome, not a sign the correction failed: most of a domain
    either already lies inside the target's own outline, or lies seaward of every annual
    coastline in the interval and is clipped away — correctly, since nothing was ice there in
    any year. Concretely, the 29 domains added on 2026-09-05 span 600 km² of polygon and
    changed total glacier area by +5.5 km², none of it taken from another glacier.

    Nothing in the mechanism ties a ``fix_type`` to the clip, though — the clip depends on the
    *target* satisfying ``probably_tidewater``, not on how the polygon was labelled. A domain
    drawn for a land-terminating glacier would therefore have its seaward part added
    literally, inventing ice over open water. ``_validate_corrections`` rejects that case
    rather than leaving it to be noticed downstream.

    Parameters
    ----------
    force_redo
        ``False`` reads ``cache/rgi7_outlines.arrow``; ``"self"`` recomputes these outlines
        while reusing the cached coastline intervals; ``"all"`` also rebuilds those. A full
        rebuild takes a few minutes.
    refresh_download
        Re-download the surge database (`SURGE_DATABASE_URL`) before a rebuild. It is
        otherwise downloaded only if missing.
    """
    cache_path = CACHE_DIR / "rgi7_outlines.arrow"
    intervals = tools.iter_intervals()

    if cache_path.is_file() and not force_redo:
        return _read_outline_cache(cache_path)

    coasts = make_coastline_intervals(force_redo="all" if force_redo == "all" else False)

    surge_database = tools.fetch(SURGE_DATABASE_URL, SURGE_DATABASE_PATH, refresh_download=refresh_download)
    rgi7_orig = gpd.read_file(f"zip://{surge_database}!{SURGE_DATABASE_LAYER}").to_crs(CRS_EPSG)
    rgi7_orig["geometry"] = rgi7_orig["geometry"].buffer(0)

    base_cols = [
        "rgi_id",
        "glims_id",
        "utm_zone",
        "term_type",
        "glac_name",
        "glac_name_",
        "slope_deg",
        "aspect_deg",
        "S_Direct",
        "S_Indirect",
        "S_All",
        "S_Onset",
        "S_Term",
        "geometry",
    ]

    field_overrides = {
        "RGI2000-v7.0-G-07-00741": {"term_type": 1},  # Scheelebreen became tidewater during the study period; treat it as tidewater throughout.
        "RGI2000-v7.0-G-07-01312": {"glac_name": "Lilliehöökbreen"},
        "RGI2000-v7.0-G-07-01379": {"S_Onset": "2015", "S_Term": "Ongoing", "glac_name": "Austfonna Basin-7"},
        "RGI2000-v7.0-G-07-01383": {"glac_name": "Storisstraumen"},
        "RGI2000-v7.0-G-07-01385": {"glac_name": "Austfonna Basin-2"},
        "RGI2000-v7.0-G-07-01381": {"glac_name": "Austfonna Basin-5"},
        "RGI2000-v7.0-G-07-01384": {"glac_name": "Bråsvellbreen"},
        "RGI2000-v7.0-G-07-00560": {"S_Onset": "2012", "S_Term": "Ongoing"},  # dH/dt observation of Tinkarpbreen
        "RGI2000-v7.0-G-07-00888": {"S_Onset": "2016", "S_Term": "2019"},  # Sveitsarfonna mirroring Penckbreen
        "RGI2000-v7.0-G-07-00899": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},  # Zawadzkibreen mirroring Nathorstbreen
        "RGI2000-v7.0-G-07-00902": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},  # Polakkbreen mirroring Nathorstbreen
        "RGI2000-v7.0-G-07-00912": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},  # Dobrowolskibreen mirroring Nathorstbreen
    }
    for rgi_id, overrides in field_overrides.items():
        idx = rgi7_orig[rgi7_orig["rgi_id"] == rgi_id].index
        for key, value in overrides.items():
            rgi7_orig.loc[idx, key] = value

    probably_tidewater = rgi7_orig["term_type"].eq(1) | rgi7_orig["zmin_m"].le(25)

    glacier_zones = gpd.read_file(GLACIER_ZONES_PATH).to_crs(CRS_EPSG)[["zone_label", "zone_name", "geometry"]]
    centroids = gpd.GeoDataFrame(rgi7_orig[["rgi_id"]].copy(), geometry=rgi7_orig.geometry.centroid, crs=CRS_EPSG)
    zone_matches = gpd.sjoin(centroids, glacier_zones, how="left", predicate="within")
    if zone_matches.index.duplicated().any() or zone_matches["zone_label"].isna().any():
        missing = zone_matches.loc[zone_matches["zone_label"].isna(), "rgi_id"].tolist()
        raise RuntimeError(f"Problematic zone assignment. Matched none or too many zones. Missing: {missing}")

    outline_corrections = gpd.read_file(OUTLINE_CORRECTIONS_PATH)
    # Sort key: extensions (False) apply first, partition overrides (True) land on top.
    outline_corrections["is_partition_override"] = outline_corrections["fix_type"].str.contains("partition_override")
    _validate_corrections(outline_corrections, rgi7_orig, probably_tidewater)

    outlines_df = rgi7_orig[base_cols].drop(columns=["geometry"]).copy()
    outlines_df[["zone_label", "zone_name"]] = zone_matches[["zone_label", "zone_name"]].to_numpy()
    rgi_id_to_idx = rgi7_orig.reset_index().set_index("rgi_id")["index"].to_dict()

    for interval in intervals:
        coastline = coasts.loc[interval.short]
        rgi7 = rgi7_orig.copy()
        rgi7["modified"] = False

        active_corrections = outline_corrections.loc[~((outline_corrections["first_active_year"] > interval.end_year) | (outline_corrections["last_active_year"] < interval.start_year))].sort_values(["is_partition_override", "priority"])
        for _, item in active_corrections.iterrows():
            target_idx = rgi_id_to_idx[item["rgi_id"]]

            # `overlaps` is deliberately strict: a correction wholly inside another glacier
            # is not a partition of it and must not be cut out of it.
            overlaps = pd.Series(False, index=rgi7.index)
            candidate_idx = rgi7.sindex.query(item["geometry"], predicate="intersects")
            if len(candidate_idx):
                overlaps.loc[candidate_idx] = rgi7.loc[candidate_idx, "geometry"].overlaps(item["geometry"]).to_numpy()

            rgi7.loc[overlaps, "geometry"] = rgi7.loc[overlaps, "geometry"].difference(item["geometry"])
            rgi7.loc[target_idx, "geometry"] = rgi7.loc[target_idx, "geometry"].union(item["geometry"])
            rgi7.loc[overlaps, "modified"] = True

        # Clip back to the coastline. This is what turns the unioned correction domains
        # above into an attribution rule (see the docstring): whatever land the coastline
        # places inside a domain stays with the target, and the rest of the domain — ocean,
        # or area the target already held — contributes nothing.
        orig_area = rgi7.loc[probably_tidewater, "geometry"].area
        clipped = rgi7.loc[probably_tidewater, "geometry"].intersection(coastline["geometry"])
        coast_modified = (orig_area - clipped.area) > 1.
        rgi7.loc[coast_modified.index[coast_modified], "modified"] = True
        rgi7.loc[probably_tidewater, "geometry"] = clipped
        outlines_df[interval.geometry_col] = gpd.GeoSeries(rgi7["geometry"].values, crs=CRS_EPSG)
        outlines_df[interval.modified_col] = rgi7["modified"].values
        outlines_df[interval.surging_col] = [surge_overlaps_interval(term_value, onset_value, interval.start_year, interval.end_year) for term_value, onset_value in zip(rgi7["S_Term"], rgi7["S_Onset"])]

    outlines_df = gpd.GeoDataFrame(outlines_df, geometry=tools.get_interval("13_24").geometry_col, crs=CRS_EPSG)

    _write_feather_atomic(outlines_df, cache_path)

    return _read_outline_cache(cache_path)


def refine_glacier_zones(force_redo: ForceRedo = False) -> gpd.GeoDataFrame:
    """Clip the glacier zones to the mainland coastlines, dropping small offshore slivers."""
    cache_path = CACHE_DIR / "refined_glacier_zones.arrow"

    if cache_path.is_file() and not force_redo:
        return gpd.read_feather(cache_path)

    glacier_zones = gpd.read_file(GLACIER_ZONES_PATH).set_index("zone_label")
    coasts = make_coastline_intervals(force_redo="all" if force_redo == "all" else False).loc[["13_24"]].explode().to_crs(glacier_zones.crs)
    coasts = coasts[coasts.geometry.area > 1e8]
    coasts = coasts.dissolve().simplify(100)

    glacier_zones.geometry = glacier_zones.geometry.intersection(coasts.geometry[0])

    _write_feather_atomic(glacier_zones, cache_path)

    return gpd.read_feather(cache_path)
