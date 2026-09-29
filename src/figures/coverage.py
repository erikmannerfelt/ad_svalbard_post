"""Supporting-Information maps of annual coverage and per-pixel source."""

import itertools

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from .. import coverage
from ..config import FIGURE_DIR

#: Upper edges of the DEM-count classes; the last class is open-ended.
COUNT_BOUNDS = (1, 2, 3, 4, 6, 8, 12, 16, 24)
NO_DATA_COLOR = "#d9d9d9"
QUALITY_COLORS = {1: "#3b6e8f", 2: "#9ec5d9", 3: "#e08a3c", 4: "#7d5ba6"}


def _extent(dataset) -> tuple[float, float, float, float]:
    """Plot extent (left, right, bottom, top) of the sample grid."""
    step = float(dataset["x"][1] - dataset["x"][0])
    return (
        float(dataset["x"][0]),
        float(dataset["x"][-1]) + step,
        float(dataset["y"][-1]) - step,
        float(dataset["y"][0]),
    )


def _data_window(land: np.ndarray, pad: int = 20) -> tuple[slice, slice]:
    """Row and column slices around the pixels that ever carry data."""
    rows = np.flatnonzero(land.any(axis=1))
    cols = np.flatnonzero(land.any(axis=0))
    return (
        slice(max(rows[0] - pad, 0), rows[-1] + pad),
        slice(max(cols[0] - pad, 0), cols[-1] + pad),
    )


def plot_annual_dem_count(show: bool = False) -> None:
    """Per-year DEM count of the annual medians (SI Figure S1)."""
    dataset = coverage.sample_annual_mosaics()
    glaciers = coverage.glacier_mask(dataset)
    land = dataset["quality_mode"].to_numpy() > 0
    rows, cols = _data_window(land)
    dataset = dataset.isel(y=rows, x=cols)
    land, glaciers = land[rows, cols], glaciers[rows, cols]
    extent = _extent(dataset)

    # Plot the count class rather than the count, so each class gets one magma color.
    n_classes = len(COUNT_BOUNDS)
    cmap = plt.get_cmap("magma", n_classes)
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, n_classes), cmap.N)
    background = matplotlib.colors.ListedColormap(["white", NO_DATA_COLOR])

    fig, axes = plt.subplots(3, 4, figsize=(8.3, 8.6), sharex=True, sharey=True)
    for axis, year in zip(axes.ravel(), coverage.YEARS, strict=True):
        count = dataset["count"].sel(year=year).to_numpy()
        doy = dataset["doy"].sel(year=year).to_numpy()
        axis.imshow(land, cmap=background, extent=extent, interpolation="nearest")
        image = axis.imshow(
            np.ma.masked_array(np.digitize(count, COUNT_BOUNDS) - 1, mask=count < 1),
            cmap=cmap,
            norm=norm,
            extent=extent,
            interpolation="nearest",
        )
        valid = glaciers & (count > 0)
        glacier_doy = doy[valid & (doy >= coverage.MIN_VALID_DOY)]
        axis.set_title(str(year), fontsize=9, loc="left")
        axis.text(
            0.98,
            0.02,
            f"{100 * valid.sum() / glaciers.sum():.0f}% glaciers\n"
            f"{100 * np.mean(glacier_doy >= coverage.MELT_START_DOY):.0f}% summer",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.97, bottom=0.08, wspace=0.03, hspace=0.1
    )
    colorbar_axis = fig.add_axes((0.25, 0.04, 0.5, 0.015))
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("DEMs in annual median")
    colorbar.set_ticks(np.arange(n_classes))
    colorbar.set_ticklabels(
        [
            str(a) if b - a == 1 else f"{a}–{b - 1}"
            for a, b in itertools.pairwise(COUNT_BOUNDS)
        ]
        + [f"≥{COUNT_BOUNDS[-1]}"]
    )
    colorbar.ax.tick_params(labelsize=7)

    fig.savefig(FIGURE_DIR / "annual_dem_count.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_source_map(show: bool = False) -> None:
    """Most common per-pixel source over 2013--2024 (SI Figure S4)."""
    dataset = coverage.sample_annual_mosaics()
    mode = dataset["quality_mode"].to_numpy()
    rows, cols = _data_window(mode > 0)
    dataset = dataset.isel(y=rows, x=cols)
    mode = mode[rows, cols]

    flags = sorted(coverage.QUALITY_LABELS)
    cmap = matplotlib.colors.ListedColormap([QUALITY_COLORS[f] for f in flags])
    norm = matplotlib.colors.BoundaryNorm(np.arange(0.5, len(flags) + 1), cmap.N)

    fig, axis = plt.subplots(figsize=(6.5, 6.8))
    axis.imshow(
        np.ma.masked_equal(mode, 0),
        cmap=cmap,
        norm=norm,
        extent=_extent(dataset),
        interpolation="nearest",
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=QUALITY_COLORS[f]) for f in flags]
    axis.legend(
        handles,
        [coverage.QUALITY_LABELS[f] for f in flags],
        loc="lower right",
        fontsize=8,
        frameon=False,
    )
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "coreg_source_map.png", dpi=300)
    if show:
        plt.show()
    plt.close(fig)
