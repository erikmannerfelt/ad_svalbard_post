from __future__ import annotations

import shapely.ops
import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm

from .config import CACHE_DIR, CRS_EPSG
from . import tools


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


def make_coastline_intervals() -> gpd.GeoDataFrame:
    cache_path = CACHE_DIR / "coastline_intervals.arrow"

    if cache_path.is_file():
        return gpd.read_feather(cache_path).set_index("key")  # type: ignore

    coastlines = {}
    for year in range(2013, 2025):
        coastline = (
            gpd.read_file(f"zip://input/Coast_annual.zip/Coast{year}.shp")
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

    out.to_feather(cache_path)

    return gpd.read_feather(cache_path).set_index("key")  # type: ignore


def make_outlines(redo: bool = False):
    cache_path = CACHE_DIR / "rgi7_outlines.arrow"
    intervals = tools.iter_intervals()

    if cache_path.is_file() and not redo:
        out = gpd.read_feather(cache_path)
        for interval in intervals:
            out[interval.area_col] = out[interval.geometry_col].area
        return out

    coasts = make_coastline_intervals()

    rgi7_orig = gpd.read_file("zip://input/RGI_V7_Surge_Database.zip/RGI_V7_Surge_Database/RGI2000-v7.0-G-07_svalbard_jan_mayen_Surge_Database.shp").to_crs(CRS_EPSG)
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

    glacier_zones = gpd.read_file("shapes/glacier_zones.geojson").to_crs(CRS_EPSG)[["zone_label", "zone_name", "geometry"]]
    centroids = gpd.GeoDataFrame(rgi7_orig[["rgi_id"]].copy(), geometry=rgi7_orig.geometry.centroid, crs=CRS_EPSG)
    zone_matches = gpd.sjoin(centroids, glacier_zones, how="left", predicate="within")
    if zone_matches.index.duplicated().any() or zone_matches["zone_label"].isna().any():
        missing = zone_matches.loc[zone_matches["zone_label"].isna(), "rgi_id"].tolist()
        raise RuntimeError(f"Problematic zone assignment. Matched none or too many zones. Missing: {missing}")

    outline_corrections = gpd.read_file("shapes/outline_corrections.geojson")
    outline_corrections["is_extension"] = outline_corrections["fix_type"].str.contains("partition_override")

    outlines_df = rgi7_orig[base_cols].drop(columns=["geometry"]).copy()
    outlines_df[["zone_label", "zone_name"]] = zone_matches[["zone_label", "zone_name"]].to_numpy()
    rgi_id_to_idx = rgi7_orig.reset_index().set_index("rgi_id")["index"].to_dict()

    for interval in intervals:
        coastline = coasts.loc[interval.short]
        rgi7 = rgi7_orig.copy()
        rgi7["modified"] = False

        active_corrections = outline_corrections.loc[~((outline_corrections["first_active_year"] > interval.end_year) | (outline_corrections["last_active_year"] < interval.start_year))].sort_values(["is_extension", "priority"])
        for _, item in active_corrections.iterrows():
            outline_idx = rgi_id_to_idx.get(item["rgi_id"])
            if outline_idx is None:
                continue

            outline_idx = [outline_idx]
            if len(outline_idx) != 1:
                raise RuntimeError(f"Problematic correction. Matched none or too many indexes.\n{item}")

            overlaps = pd.Series(False, index=rgi7.index)
            try:
                candidate_idx = rgi7.sindex.query(item["geometry"], predicate="intersects")
                if len(candidate_idx):
                    overlaps.loc[candidate_idx] = rgi7.loc[candidate_idx, "geometry"].overlaps(item["geometry"]).to_numpy()
            except Exception:
                overlaps = rgi7["geometry"].overlaps(item["geometry"])
            rgi7.loc[overlaps, "geometry"] = rgi7.loc[overlaps, "geometry"].difference(item["geometry"])
            rgi7.loc[outline_idx[0], "geometry"] = rgi7.loc[outline_idx[0], "geometry"].union(item["geometry"])
            rgi7.loc[overlaps, "modified"] = True

        # Keep only the coastline-constrained geometries for glaciers that are likely tidewater.
        orig_area = rgi7.loc[probably_tidewater, "geometry"].area
        clipped = rgi7.loc[probably_tidewater, "geometry"].intersection(coastline["geometry"])
        coast_modified = (orig_area - clipped.area) > 1.
        rgi7.loc[coast_modified.index[coast_modified], "modified"] = True
        rgi7.loc[probably_tidewater, "geometry"] = clipped
        outlines_df[interval.geometry_col] = gpd.GeoSeries(rgi7["geometry"].values, crs=CRS_EPSG)
        outlines_df[interval.modified_col] = rgi7["modified"].values
        outlines_df[interval.surging_col] = [surge_overlaps_interval(term_value, onset_value, interval.start_year, interval.end_year) for term_value, onset_value in zip(rgi7["S_Term"], rgi7["S_Onset"])]

    outlines_df = gpd.GeoDataFrame(outlines_df, geometry=tools.get_interval("13_24").geometry_col, crs=CRS_EPSG)
    for interval in intervals:
        outlines_df[interval.area_col] = outlines_df[interval.geometry_col].area

    outlines_df.to_feather(cache_path)

    return outlines_df


def refine_glacier_zones(redo: bool = False):
    cache_path = CACHE_DIR / "refined_glacier_zones.arrow"

    if cache_path.is_file() and not redo:
        return gpd.read_feather(cache_path)

    glacier_zones = gpd.read_file("shapes/glacier_zones.geojson").set_index("zone_label")
    coasts = make_coastline_intervals().loc[["13_24"]].explode().to_crs(glacier_zones.crs)
    coasts = coasts[coasts.geometry.area > 1e8]
    coasts = coasts.dissolve().simplify(100)

    glacier_zones.geometry = glacier_zones.geometry.intersection(coasts.geometry[0])

    glacier_zones.to_feather(cache_path)

    return gpd.read_feather(cache_path)
