"""Sensitivity of the positive-elevation-change area to its threshold and to noise.

The area with rates above ``POSITIVE_THRESHOLD`` is a pixel count, so unlike the mean
rates it is not averaged down by aggregation: the per-pixel error (~0.1 m a^-1) is of the
same size as the threshold, and most of it is correlated over 5-16 km, the scale of the
ice-cap interiors where the positive area lies. Two checks bound how much this matters:

1. The observed fractions recomputed for the thresholds in ``THRESHOLDS``.
2. The fractions recomputed after adding ``N_REALIZATIONS`` noise fields drawn from the
   error model of the main text (Section 2.6): per pixel, ``sigma_baseline`` times a
   Gaussian random field with the stable-terrain variogram's correlation, plus
   ``sigma_excess`` times a single standard-normal draw per realization, since
   ``sigma_excess`` is treated as fully systematic.

The noise is added to the observed field, which already contains noise, so the
simulated spread is an upper bound on the noise-induced uncertainty of the fractions.
"""

from typing import Final

import geopandas as gpd
import numpy as np
import scipy.fft
import tqdm
import xarray as xr

from . import outlines, statistics, tools
from .config import CACHE_DIR, INPUT_DIR

#: Rate threshold (m a^-1) of the reported positive area; ``statistics`` reports the same.
POSITIVE_THRESHOLD: Final = 0.1
THRESHOLDS: Final = (0.0, 0.05, 0.1, 0.2)
N_REALIZATIONS: Final = 100
RANDOM_SEED: Final = 20261016
#: 40 m, as in ``sampling.sample_rasters``, so the unperturbed fractions equal the
#: reported ones.
OVERVIEW_LEVEL: Final = 2
#: Padding (m) of the periodic simulation grid. The correlation of the longest variogram
#: component is below 1e-4 at this distance, so wrap-around correlation is negligible.
PADDING_M: Final = 30_000.0
INTERVAL_SHORTS: Final = ("13_18", "19_24")
PARTITIONS: Final = ("all", "surging", "nonsurging")


def _read_interval(
    interval: tools.Interval, outlines_df: gpd.GeoDataFrame
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Read the rate and error rasters of one interval and rasterize its outlines.

    Returns
    -------
    The ``slope``, ``baseline_err`` and ``excess_err`` arrays (m a^-1, NaN outside data),
    the glacier id per pixel (0 outside glaciers) and the pixel size in meters.
    """
    import rasterio
    import rasterio.features
    import rasterio.windows

    paths = {
        "slope": INPUT_DIR / f"trend_{interval.filename_label}_slope.tif",
        "baseline_err": INPUT_DIR
        / f"trend_{interval.filename_label}_slope_baseline_err.tif",
        "fit_err": INPUT_DIR / f"trend_{interval.filename_label}_slope_fit_err.tif",
    }
    with rasterio.open(paths["slope"], overview_level=OVERVIEW_LEVEL) as raster:
        glacier_id = rasterio.features.rasterize(
            [
                (geom, outline_id)
                for geom, outline_id in zip(
                    outlines_df[interval.geometry_col], outlines_df["id"], strict=True
                )
                if geom is not None and not geom.is_empty
            ],
            out_shape=(raster.height, raster.width),
            fill=0,
            dtype="int32",
            transform=raster.transform,
        )
        res = float(raster.res[0])
        bounds = raster.bounds
        shape = (raster.height, raster.width)

    data = {}
    for key, path in paths.items():
        # Overview reads lose the per-band scale metadata (see `sampling.sample_rasters`).
        with rasterio.open(path) as full_raster:
            scale = full_raster.scales[0]
        with rasterio.open(path, overview_level=OVERVIEW_LEVEL) as raster:
            window = rasterio.windows.from_bounds(*bounds, transform=raster.transform)
            window = window.round_offsets().round_lengths()
            values = raster.read(1, window=window, masked=True)
        if values.shape != shape:
            raise RuntimeError(
                f"Shape mismatch for {path.name}: expected {shape}, found {values.shape}"
            )
        data[key] = (values.astype("float32") * scale).filled(np.nan)

    fit_err = data.pop("fit_err")
    data["excess_err"] = np.sqrt(
        np.clip(fit_err**2 - data["baseline_err"] ** 2, a_min=0, a_max=None)
    )
    return data, glacier_id, res


def circulant_filter(shape: tuple[int, int], res: float) -> np.ndarray:
    """Spectral filter that turns white noise into the variogram-correlated field.

    Uses circulant embedding: the correlation function from
    ``statistics.variogram_correlation`` is evaluated on a periodic grid of ``shape``
    with wrap-around distances, and the square root of its (real, non-negative) 2-D
    Fourier transform is the filter. ``irfft2(filter * rfft2(w))`` for unit white
    noise ``w`` then has unit variance and the variogram's correlation.

    Parameters
    ----------
    shape
        Periodic (padded) grid shape in pixels.
    res
        Pixel size in meters.

    Returns
    -------
    Array of the ``rfft2`` output shape, float32.
    """
    distance_y = np.minimum(np.arange(shape[0]), shape[0] - np.arange(shape[0])) * res
    distance_x = np.minimum(np.arange(shape[1]), shape[1] - np.arange(shape[1])) * res
    distance = np.hypot(distance_y[:, None], distance_x[None, :])
    correlation = np.asarray(
        statistics.variogram_correlation(distance), dtype="float32"
    )
    del distance

    eigenvalues = scipy.fft.rfft2(correlation, workers=-1).real
    # Truncation of the correlation leaves tiny negative eigenvalues; they carry no power.
    return np.sqrt(np.clip(eigenvalues, a_min=0, a_max=None)).astype("float32")


def _fractions_above(
    values: np.ndarray, threshold: float, partition_masks: dict[str, np.ndarray]
) -> list[float]:
    """Percentage of each partition's pixels above ``threshold``, in ``PARTITIONS`` order.

    NaN (no data) compares False, so data gaps count as not positive but stay in the
    denominator, as in ``sampling.sample_rasters``.
    """
    above = values > threshold
    return [
        100
        * np.count_nonzero(above[partition_masks[key]])
        / np.count_nonzero(partition_masks[key])
        for key in PARTITIONS
    ]


def simulate_positive_fractions(
    sampled: gpd.GeoDataFrame, force_redo: tools.ForceRedo = False
) -> xr.Dataset:
    """Positive-area fractions for several thresholds and under simulated noise.

    Parameters
    ----------
    sampled
        Per-glacier table from ``sampling.sample_rasters``, with any surge-class
        overrides already applied (as in ``statistics.get_statistics``). Only
        ``rgi_id`` and the intervals' surge columns are used.
    force_redo
        ``False`` uses the cache. ``"self"`` and ``"all"`` both recompute, since the
        inputs are source rasters.

    Returns
    -------
    Dataset in percent of the partition's glacier area: ``observed`` (``interval``,
    ``partition``, ``threshold``) and ``simulated`` (``interval``, ``partition``,
    ``realization``), the latter at ``POSITIVE_THRESHOLD``.
    """
    cache_path = CACHE_DIR / "positive_area_sensitivity.nc"
    if cache_path.is_file() and not force_redo:
        return xr.load_dataset(cache_path)

    outlines_df = outlines.make_outlines()
    outlines_df["id"] = (
        outlines_df["rgi_id"].str.split("-", expand=True).iloc[:, -1].astype(int)
    )
    surge_by_rgi = sampled.set_index("rgi_id")
    rng = np.random.default_rng(RANDOM_SEED)

    observed = np.full((len(INTERVAL_SHORTS), len(PARTITIONS), len(THRESHOLDS)), np.nan)
    simulated = np.full((len(INTERVAL_SHORTS), len(PARTITIONS), N_REALIZATIONS), np.nan)
    filter_cache: dict[tuple[tuple[int, int], float], np.ndarray] = {}

    for i, interval_short in enumerate(INTERVAL_SHORTS):
        interval = tools.get_interval(interval_short)
        data, glacier_id, res = _read_interval(interval, outlines_df)

        is_surging = np.zeros(int(glacier_id.max()) + 1, dtype=bool)
        surging = surge_by_rgi.loc[outlines_df["rgi_id"], interval.surging_col]
        is_surging[outlines_df["id"].to_numpy()] = surging.fillna(False).to_numpy(bool)
        on_glacier = glacier_id > 0
        masks = {
            "all": on_glacier,
            "surging": on_glacier & is_surging[glacier_id],
            "nonsurging": on_glacier & ~is_surging[glacier_id],
        }
        # Only glacier pixels are counted, so keep 1-D views of those.
        slope = data["slope"][on_glacier]
        # A missing error leaves the pixel unperturbed rather than dropping it.
        baseline_err = np.nan_to_num(data["baseline_err"][on_glacier])
        excess_err = np.nan_to_num(data["excess_err"][on_glacier])
        partition_masks = {key: mask[on_glacier] for key, mask in masks.items()}

        for k, threshold in enumerate(THRESHOLDS):
            observed[i, :, k] = _fractions_above(slope, threshold, partition_masks)

        pad = int(np.ceil(PADDING_M / res))
        padded_shape = (
            scipy.fft.next_fast_len(glacier_id.shape[0] + pad, real=True),
            scipy.fft.next_fast_len(glacier_id.shape[1] + pad, real=True),
        )
        if (padded_shape, res) not in filter_cache:
            filter_cache[(padded_shape, res)] = circulant_filter(padded_shape, res)
        spectral_filter = filter_cache[(padded_shape, res)]

        for r in tqdm.trange(N_REALIZATIONS, desc=f"Simulating noise {interval_short}"):
            white = rng.standard_normal(padded_shape, dtype="float32")
            field = scipy.fft.irfft2(
                spectral_filter * scipy.fft.rfft2(white, workers=-1),
                s=padded_shape,
                workers=-1,
            )
            correlated = field[: glacier_id.shape[0], : glacier_id.shape[1]][on_glacier]
            noise = baseline_err * correlated + excess_err * rng.standard_normal()
            simulated[i, :, r] = _fractions_above(
                slope + noise, POSITIVE_THRESHOLD, partition_masks
            )

    dataset = xr.Dataset(
        {
            "observed": (("interval", "partition", "threshold"), observed),
            "simulated": (("interval", "partition", "realization"), simulated),
        },
        coords={
            "interval": list(INTERVAL_SHORTS),
            "partition": list(PARTITIONS),
            "threshold": list(THRESHOLDS),
            "realization": np.arange(N_REALIZATIONS),
        },
        attrs={
            "units": "percent of partition glacier area",
            "positive_threshold_m_per_a": POSITIVE_THRESHOLD,
            "random_seed": RANDOM_SEED,
            "overview_level": OVERVIEW_LEVEL,
        },
    )
    # Fast to rebuild relative to the simulation, but a half-written file would look valid.
    tmp_path = cache_path.with_suffix(".tmp.nc")
    try:
        dataset.to_netcdf(tmp_path)
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return xr.load_dataset(cache_path)


def get_sensitivity_stats(
    sampled: gpd.GeoDataFrame, force_redo: tools.ForceRedo = False
) -> dict[str, object]:
    """Ranges of the positive-area fractions under threshold changes and noise.

    Returns
    -------
    Per statistics key (``slope_start_mid``, ``slope_mid_end``) and partition: the
    observed fraction, its range over ``THRESHOLDS`` (``thrlow``/``thrhigh``) and the
    2.5th-97.5th percentile range over the noise realizations (``simlow``/``simhigh``),
    all in whole percent like the reported fractions. Also the thresholds, the number
    of realizations and, per partition, the percentage of realizations in which the
    2013-2018 fraction exceeds the 2019-2024 one (``declinepercent``).
    """
    dataset = simulate_positive_fractions(sampled, force_redo=force_redo)
    stats_keys = {"13_18": "slope_start_mid", "19_24": "slope_mid_end"}

    out: dict[str, object] = {
        "nrealizations": N_REALIZATIONS,
        "thresholdlow": f"{min(THRESHOLDS):.2f}",
        "thresholdhigh": f"{max(THRESHOLDS):.2f}",
    }
    for interval_short, stats_key in stats_keys.items():
        per_partition = {}
        for partition in PARTITIONS:
            observed = dataset["observed"].sel(
                interval=interval_short, partition=partition
            )
            simulated = dataset["simulated"].sel(
                interval=interval_short, partition=partition
            )
            low, high = np.percentile(simulated.to_numpy(), [2.5, 97.5])
            per_partition[partition] = {
                "observed": round(float(observed.sel(threshold=POSITIVE_THRESHOLD))),
                "thrlow": round(float(observed.min())),
                "thrhigh": round(float(observed.max())),
                "simlow": round(float(low)),
                "simhigh": round(float(high)),
            }
        out[stats_key] = per_partition

    # Realizations are independent between the periods, so pair them by index.
    simulated = dataset["simulated"]
    out["declinepercent"] = {
        partition: round(
            100
            * float(
                (
                    simulated.sel(interval="13_18", partition=partition)
                    > simulated.sel(interval="19_24", partition=partition)
                ).mean()
            )
        )
        for partition in PARTITIONS
    }
    return out
