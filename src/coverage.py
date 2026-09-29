"""Annual mosaic coverage, per-pixel source and acquisition season.

The annual mosaics carry, per pixel and year, the number of DEMs in the median
(``count``), their median day of year (``doy``) and which product they came from
(``quality``), plus the most common source over the record (``quality_mode``). This
module samples those bands on a coarse grid and summarizes them for the Supporting
Information: the per-year coverage (Figure S1), the source fractions and map
(Text S5, Figure S4), and the bias that the uneven acquisition season can introduce
into integer-year trends (Text S1).

The rasters' own overviews are average-resampled, which corrupts categorical and
day-of-year values (26% of 2018 overview DOYs fall below day 60, against 0.15% at full
resolution), so the sampling reads full-resolution blocks and keeps every
``SAMPLE_STRIDE``-th pixel instead.
"""

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import tqdm.contrib.concurrent
import xarray as xr

from . import outlines, tools
from .config import CACHE_DIR, INPUT_DIR, TABLE_DIR

YEARS: Final = tuple(range(2013, 2025))
#: Every 16th 5 m pixel, i.e. an 80 m grid. The rasters' 256 px blocks are multiples
#: of this, so block-wise striding lands on the same pixels as a global stride.
SAMPLE_STRIDE: Final = 16
#: Quality flag values, from ``adsvalbard.mosaicking.load_and_mosaic_dem``.
QUALITY_LABELS: Final = {
    1: "Heavily filtered, co-registered",
    2: "Lightly filtered, co-registered",
    3: "Bias-corrected only",
    4: "Globally deramped",
}
#: Macro-safe (letters only) keys for the quality flags.
QUALITY_KEYS: Final = {1: "heavy", 2: "light", 3: "biascorr", 4: "deramped"}
#: Day-of-year bounds of the main melt season (1 June to 1 September). An annual
#: median acquired before the start is treated as end-of-winter and one after the end
#: as end-of-summer, with a linear ramp between. A deliberately simple proxy for how
#: much of a summer's surface lowering an annual median contains.
MELT_START_DOY: Final = 152
MELT_END_DOY: Final = 245
#: Day-of-year values below this are not physical for optical stereo in the polar
#: night and occur only in a few blended pixels (0.15% in 2018); they are dropped.
MIN_VALID_DOY: Final = 60


def _strided_read(path: Path, stride: int = SAMPLE_STRIDE) -> np.ndarray:
    """Read every ``stride``-th pixel of band 1 at full resolution, block by block."""
    import rasterio

    with rasterio.open(path) as raster:
        out = np.zeros(
            (-(-raster.height // stride), -(-raster.width // stride)),
            dtype=raster.dtypes[0],
        )
        for _, window in raster.block_windows(1):
            block = raster.read(1, window=window)
            row0 = -(-window.row_off // stride)
            col0 = -(-window.col_off // stride)
            sub = block[
                row0 * stride - window.row_off :: stride,
                col0 * stride - window.col_off :: stride,
            ]
            out[row0 : row0 + sub.shape[0], col0 : col0 + sub.shape[1]] = sub
    return out


def sample_annual_mosaics(force_redo: tools.ForceRedo = False) -> xr.Dataset:
    """Sample the annual count, day-of-year and quality bands on an 80 m grid.

    Parameters
    ----------
    force_redo
        ``False`` uses the cache. ``"self"`` and ``"all"`` both resample, since the
        inputs are source rasters rather than derived caches.

    Returns
    -------
    Dataset with ``count``, ``doy`` and ``quality`` (``year``, ``y``, ``x``) and
    ``quality_mode`` (``y``, ``x``), in EPSG:32633. ``x``/``y`` are the map
    coordinates of the upper-left corner of each sampled 5 m pixel; the ``transform``
    attribute gives the corresponding 80 m grid.
    """
    import rasterio

    cache_path = CACHE_DIR / "annual_mosaic_samples.nc"
    if cache_path.is_file() and not force_redo:
        return xr.load_dataset(cache_path)

    paths = {
        (kind, year): INPUT_DIR / f"dem_{year}_{kind}.tif"
        for kind in ("count", "doy", "quality")
        for year in YEARS
    }
    paths[("quality_mode", 0)] = INPUT_DIR / "dem_quality_mode.tif"
    keys = list(paths)
    arrays = tqdm.contrib.concurrent.thread_map(
        _strided_read,
        [paths[key] for key in keys],
        max_workers=12,
        desc="Sampling annual mosaic bands",
    )
    sampled = dict(zip(keys, arrays, strict=True))

    with rasterio.open(paths[("quality_mode", 0)]) as raster:
        transform = raster.transform * rasterio.Affine.scale(SAMPLE_STRIDE)
        crs = raster.crs.to_string()
    shape = sampled[("quality_mode", 0)].shape
    x = transform.c + transform.a * np.arange(shape[1])
    y = transform.f + transform.e * np.arange(shape[0])

    data_vars = {
        kind: (("year", "y", "x"), np.stack([sampled[(kind, year)] for year in YEARS]))
        for kind in ("count", "doy", "quality")
    }
    data_vars["quality_mode"] = (("y", "x"), sampled[("quality_mode", 0)])
    dataset = xr.Dataset(
        data_vars,
        coords={"year": list(YEARS), "y": y, "x": x},
        attrs={"crs": crs, "transform": list(transform)[:6], "stride": SAMPLE_STRIDE},
    )

    # Interrupting a multi-minute write must not leave a usable-looking file.
    tmp_path = cache_path.with_suffix(".tmp.nc")
    try:
        dataset.to_netcdf(
            tmp_path,
            encoding={name: {"zlib": True, "complevel": 4} for name in data_vars},
        )
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return xr.load_dataset(cache_path)


def glacier_mask(dataset: xr.Dataset) -> np.ndarray:
    """Rasterize the 2013--2024 union glacier outlines onto the sample grid."""
    import rasterio
    import rasterio.features

    outlines_df = outlines.make_outlines()
    geometries = outlines_df[tools.get_interval("13_24").geometry_col]
    return rasterio.features.rasterize(
        ((geom, 1) for geom in geometries if geom is not None and not geom.is_empty),
        out_shape=(dataset.sizes["y"], dataset.sizes["x"]),
        transform=rasterio.Affine(*dataset.attrs["transform"]),
        fill=0,
        dtype="uint8",
    ).astype(bool)


def melt_fraction(doy: np.ndarray) -> np.ndarray:
    """Fraction of the melt season elapsed at each day of year (0 before, 1 after).

    Examples
    --------
    >>> melt_fraction(np.array([100, 152, 198, 245, 280])).round(2).tolist()
    [0.0, 0.0, 0.49, 1.0, 1.0]
    """
    fraction = (doy.astype("float64") - MELT_START_DOY) / (
        MELT_END_DOY - MELT_START_DOY
    )
    return np.clip(fraction, 0.0, 1.0)


def seasonal_trend_leverage(melt: pd.Series) -> dict[str, float]:
    """Spurious trend per meter of summer surface lowering, for each interval.

    If an annual median acquired a fraction ``m_t`` of the way through the melt season
    is lowered by ``S * m_t`` for a summer lowering ``S`` (m), an ordinary least-squares
    fit against integer years is biased by ``-S * L``, with
    ``L = sum((t - mean(t)) * m_t) / sum((t - mean(t))**2)`` in a^-1.

    Examples
    --------
    >>> years = pd.Series([0.0] * 3 + [1.0] * 3 + [0.0] * 6, index=range(2013, 2025))
    >>> {k: round(v, 3) for k, v in seasonal_trend_leverage(years).items()}
    {'start_mid': 0.257, 'mid_end': 0.0, 'start_end': -0.031}
    """
    leverage = {}
    for interval in tools.iter_intervals():
        subset = melt.loc[interval.start_year : interval.end_year]
        t = subset.index.to_numpy(dtype=float)
        dt = t - t.mean()
        leverage[interval.stats_key] = float(
            np.sum(dt * subset.to_numpy()) / np.sum(dt**2)
        )
    return leverage


def get_coverage_stats(force_redo: tools.ForceRedo = False) -> dict[str, object]:
    """Summarize coverage, source and acquisition season over glaciers.

    Also writes ``tables/seasonal_coverage.csv`` and the SI table
    ``tables/seasonal_coverage.tex``.

    Returns
    -------
    ``source`` holds the percentage of glacier area whose most common source over the
    record is each product; ``leverage`` the seasonal trend leverage per interval
    (m a^-1 of bias per meter of summer lowering, see :func:`seasonal_trend_leverage`).
    """
    dataset = sample_annual_mosaics(force_redo=force_redo)
    glaciers = glacier_mask(dataset)

    mode = dataset["quality_mode"].to_numpy()[glaciers]
    mode = mode[mode > 0]
    source = {
        QUALITY_KEYS[flag]: round(100 * float(np.mean(mode == flag)), 1)
        for flag in QUALITY_KEYS
    }

    rows = []
    for year in YEARS:
        count = dataset["count"].sel(year=year).to_numpy()[glaciers]
        doy = dataset["doy"].sel(year=year).to_numpy()[glaciers]
        doy = doy[(count > 0) & (doy >= MIN_VALID_DOY)]
        rows.append(
            {
                "year": year,
                "coverage_percent": 100 * float(np.mean(count > 0)),
                "median_count": float(np.median(count[count > 0])),
                "summer_percent": 100 * float(np.mean(doy >= MELT_START_DOY)),
                "median_doy": float(np.median(doy)),
                "melt_fraction": float(np.mean(melt_fraction(doy))),
            }
        )
    season = pd.DataFrame(rows).set_index("year")
    season.round(3).to_csv(TABLE_DIR / "seasonal_coverage.csv")
    _write_seasonal_table(season)

    return {
        "source": source,
        "leverage": {
            k: round(v, 3)
            for k, v in seasonal_trend_leverage(season["melt_fraction"]).items()
        },
        "melt_start_doy": MELT_START_DOY,
        "melt_end_doy": MELT_END_DOY,
    }


def _write_seasonal_table(season: pd.DataFrame) -> Path:
    """Write the per-year coverage and season table for the Supporting Information."""
    lines = [
        "% Autogenerated by src.coverage.get_coverage_stats -- do not edit.",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Year & Glacier coverage (\%) & Median DEM count & Summer medians (\%) & Median day of year \\",
        r"\hline",
    ]
    for year, row in season.iterrows():
        lines.append(
            f"{year} & {row['coverage_percent']:.1f} & {row['median_count']:.0f} & "
            f"{row['summer_percent']:.0f} & {row['median_doy']:.0f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    path = TABLE_DIR / "seasonal_coverage.tex"
    path.write_text("\n".join(lines) + "\n")
    return path
