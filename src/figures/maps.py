"""Shared drawing helpers for the multi-panel dH/dt map figures.

Layouts are specified in mm from the figure's top-left corner, as in a print layout, so the
helpers take the figure size in mm to convert to figure fractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.cm
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.windows

from .. import outlines

MM_PER_INCH: Final = 25.4
# The SVG embeds the maps resampled to this resolution. The rasters are read at 10 m, which is
# finer than one 400 dpi dot at the map scales used.
DPI: Final = 400
OVERVIEW_LEVEL: Final = 0

FRAME_LINEWIDTH: Final = 0.8
TEXT_COLOR: Final = "#555"
MARKER_COLOR: Final = "red"
LABEL_HALO: Final = (
    matplotlib.patheffects.withStroke(foreground="white", linewidth=1.5),
)
LABEL_BOX: Final = {
    "facecolor": "white",
    "edgecolor": "black",
    "linewidth": 0.6,
    "boxstyle": "square,pad=0.15",
}


@dataclass(frozen=True)
class FigureSizeMM:
    width: float
    height: float

    def figsize(self) -> tuple[float, float]:
        return (self.width / MM_PER_INCH, self.height / MM_PER_INCH)

    def rect(
        self, left: float, top: float, width: float, height: float
    ) -> tuple[float, float, float, float]:
        """Convert a box in mm from the top-left corner to a figure-fraction rect."""
        return (
            left / self.width,
            1.0 - (top + height) / self.height,
            width / self.width,
            height / self.height,
        )


@dataclass(frozen=True)
class LocatorArea:
    """A map extent (EPSG:32633, m) to mark on a locator map."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    label: str
    # Offset of the label from the extent's centre, in locator-map metres
    label_offset: tuple[float, float]


def read_trend_rgba(
    filepath: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    mappable: matplotlib.cm.ScalarMappable,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read and colour a trend raster within a map extent.

    Returns
    -------
    The RGBA image (no data transparent) and its ``imshow`` extent (left, right, bottom, top).
    """
    # The band scale is lost in the overview reads, so read it from the full-resolution raster
    with rasterio.open(filepath) as raster:
        scale = raster.scales[0]
    with rasterio.open(filepath, overview_level=OVERVIEW_LEVEL) as raster:
        window = rasterio.windows.from_bounds(
            xlim[0], ylim[0], xlim[1], ylim[1], transform=raster.transform
        )
        window = window.round_offsets(op="floor").round_lengths(op="ceil")
        values = raster.read(1, window=window, masked=True)
        left, bottom, right, top = rasterio.windows.bounds(window, raster.transform)

    rgba = mappable.to_rgba(values.astype("float32").filled(0.0) * scale)
    rgba[np.ma.getmaskarray(values), 3] = 0.0
    return rgba, (left, right, bottom, top)


def add_map_axes(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    filepath: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    mappable: matplotlib.cm.ScalarMappable,
) -> plt.Axes:
    """Add a framed, tickless axis showing a trend raster within the given extent."""
    axis = fig.add_axes(rect)
    img, extent = read_trend_rgba(filepath, xlim, ylim, mappable)
    axis.imshow(img, extent=extent, interpolation="antialiased")
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(FRAME_LINEWIDTH)
    return axis


def add_scale_bar(
    axis: plt.Axes,
    ticks_km: tuple[int, ...],
    y_frac: float = 0.07,
    x_frac: float = 0.035,
) -> None:
    """Draw a ticked scale bar on a map axis in metres, by default in the lower-left corner."""
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    x0 = xmin + x_frac * (xmax - xmin)
    y0 = ymin + y_frac * (ymax - ymin)
    tick_height = 0.025 * (ymax - ymin)
    kwargs = {"color": TEXT_COLOR, "lw": 0.8, "solid_capstyle": "butt"}
    axis.plot([x0, x0 + ticks_km[-1] * 1e3], [y0, y0], **kwargs)
    for km in ticks_km:
        x = x0 + km * 1e3
        axis.plot([x, x], [y0, y0 + tick_height], **kwargs)
        axis.text(
            x,
            y0 + 1.6 * tick_height,
            f"{km} km" if km == ticks_km[-1] else str(km),
            ha="left" if km == ticks_km[-1] else "center",
            va="bottom",
            fontsize=7,
            color=TEXT_COLOR,
            path_effects=LABEL_HALO,
        )


def add_panel_label(
    fig: plt.Figure, size: FigureSizeMM, label: str, left_mm: float, top_mm: float
) -> None:
    fig.text(
        (left_mm + 0.6) / size.width,
        1.0 - (top_mm + 0.5) / size.height,
        f"{label})",
        ha="left",
        va="top",
        fontsize=9,
        bbox=LABEL_BOX,
        zorder=10,
    )


def add_corner_text(axis: plt.Axes, text: str) -> None:
    """Boxed text in the upper-right corner of an axis, e.g. the interval of a map."""
    axis.text(
        0.985,
        0.975,
        text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=LABEL_BOX,
        zorder=10,
    )


def add_locator(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    areas: tuple[LocatorArea, ...],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> plt.Axes:
    """Add a locator map of land and glaciers with the given areas marked in red."""
    axis = fig.add_axes(rect)
    # Rasterized: written as vectors, the outlines' vertices would dominate the SVG
    coast = outlines.make_coastline_intervals().loc[["13_18"]]
    glaciers = outlines.make_outlines()["geometry_13_18"]
    coast.plot(ax=axis, color="#a7a7a7", linewidth=0, rasterized=True)
    glaciers.plot(ax=axis, color="#b0bbcf", linewidth=0, rasterized=True)

    for area in areas:
        axis.add_patch(
            matplotlib.patches.Rectangle(
                (area.xlim[0], area.ylim[0]),
                area.xlim[1] - area.xlim[0],
                area.ylim[1] - area.ylim[0],
                facecolor=MARKER_COLOR,
                edgecolor="none",
            )
        )
        axis.text(
            np.mean(area.xlim) + area.label_offset[0],
            np.mean(area.ylim) + area.label_offset[1],
            area.label,
            ha="center",
            va="center",
            fontsize=7,
            color="darkred",
            path_effects=LABEL_HALO,
        )

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_aspect("equal")
    axis.set_facecolor("white")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(FRAME_LINEWIDTH)
    return axis


def add_dhdt_colorbar(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    mappable: matplotlib.cm.ScalarMappable,
) -> None:
    """Add a framed box with an open-ended dH/dt colorbar labelled at its saturation limits."""
    box = fig.add_axes(rect)
    box.set_xticks([])
    box.set_yticks([])
    for spine in box.spines.values():
        spine.set_linewidth(FRAME_LINEWIDTH)

    norm = mappable.norm
    cax = box.inset_axes((0.12, 0.1, 0.22, 0.8))
    cbar = fig.colorbar(mappable, cax=cax, extend="both")
    cbar.set_ticks(
        [norm.vmin, norm.vmax],
        labels=[f"≤{norm.vmin:.0f}".replace("-", "−"), f"≥{norm.vmax:.0f}"],
    )
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_linewidth(0.6)
    box.text(
        0.42,
        0.5,
        "m a$^{-1}$",
        transform=box.transAxes,
        ha="left",
        va="center",
        fontsize=7,
    )
