"""SI figure of the temporal bias correction of the 2019–2024 trend in two areas lacking 2024 DEMs.

One row per area (Isachsenfonna, eastern Austfonna), with the uncorrected rate, the corrected
rate and the correction (corrected minus uncorrected). The rates use the main-text colour
scale. The dashed line in the correction panels outlines where the correction applies. Styled after `surge_examples`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from ..config import FIGURE_DIR, INPUT_DIR
from . import maps
from .common import DHDT_SM, svg_png_to_jpg

UNCORR_PATH: Final = INPUT_DIR / "trend_2019-2024_slope_uncorr.tif"
CORR_PATH: Final = INPUT_DIR / "trend_2019-2024_slope.tif"

# The correction is clamped at 0.6 m/a (1 m/a on Isachsenfonna)
CORRECTION_LIMIT: Final = 1.0
CORRECTION_SM: Final = matplotlib.cm.ScalarMappable(
    norm=matplotlib.colors.Normalize(
        vmin=-CORRECTION_LIMIT, vmax=CORRECTION_LIMIT, clip=True
    ),
    cmap="PuOr",
)
# The rasters store rates as int16 with a scale of ~0.0015 m/a, so smaller differences are
# rounding. Sizes are in 10 m pixels: a closing of 5 pixels and a minimum area of 1 km2.
FOOTPRINT_MIN_CORRECTION: Final = 0.005
FOOTPRINT_CLOSING: Final = 5
FOOTPRINT_MIN_AREA_PX: Final = 10_000
OUTLINE_KWARGS: Final = {"colors": "black", "linewidths": 0.6, "linestyles": "--"}

FIG_WIDTH_MM: Final = 140.0
PANEL_WIDTH_MM: Final = FIG_WIDTH_MM / 3


@dataclass(frozen=True)
class CorrectionArea:
    locator: maps.LocatorArea
    scale_bar_km: tuple[int, ...]

    @property
    def panel_height_mm(self) -> float:
        xlim, ylim = self.locator.xlim, self.locator.ylim
        return PANEL_WIDTH_MM * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])


CORRECTION_AREAS: Final = (
    # Isachsenfonna
    CorrectionArea(
        maps.LocatorArea(
            (450_698, 472_018), (8_771_325, 8_801_383), "a–c", (0, -40_000)
        ),
        (0, 5, 10),
    ),
    # Eastern Austfonna
    CorrectionArea(
        maps.LocatorArea(
            (690_480, 723_641), (8_881_068, 8_924_224), "d–f", (0, -45_000)
        ),
        (0, 5, 10),
    ),
)
FIG_SIZE: Final = maps.FigureSizeMM(
    width=FIG_WIDTH_MM,
    height=sum(area.panel_height_mm for area in CORRECTION_AREAS),
)

LOCATOR_XLIM: Final = (420_000.0, 780_000.0)
LOCATOR_YLIM: Final = (8_580_000.0, 8_960_000.0)
LOCATOR_WIDTH_MM: Final = 22.0
LOCATOR_HEIGHT_MM: Final = (
    LOCATOR_WIDTH_MM
    * (LOCATOR_YLIM[1] - LOCATOR_YLIM[0])
    / (LOCATOR_XLIM[1] - LOCATOR_XLIM[0])
)
# Lower-left corner of the bottom-left panel
LOCATOR_BOX_MM: Final = (
    0.0,
    FIG_SIZE.height - LOCATOR_HEIGHT_MM,
    LOCATOR_WIDTH_MM,
    LOCATOR_HEIGHT_MM,
)
# Colorbar boxes at the right edges of the top row's middle and right panels, below their labels
CBAR_SIZE_MM: Final = (17.0, 16.5)
DHDT_CBAR_BOX_MM: Final = (2 * PANEL_WIDTH_MM - CBAR_SIZE_MM[0], 7.0, *CBAR_SIZE_MM)
CORRECTION_CBAR_BOX_MM: Final = (FIG_WIDTH_MM - CBAR_SIZE_MM[0], 7.0, *CBAR_SIZE_MM)


def read_correction(
    xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float]]:
    """Read the correction (corrected minus uncorrected 2019–2024 rate, m a-1) in an extent.

    Returns
    -------
    The correction (masked where either rate is missing) and its ``imshow`` extent.
    """
    uncorr, extent = maps.read_trend_values(UNCORR_PATH, xlim, ylim)
    corr, _ = maps.read_trend_values(CORR_PATH, xlim, ylim)
    return corr - uncorr, extent


def correction_footprint(correction: np.ma.MaskedArray) -> np.ndarray:
    """Area where the correction applies, cleaned of specks, holes and zero crossings.

    The closing bridges the narrow band where a smoothly varying correction passes zero, and
    only connected areas of at least `FOOTPRINT_MIN_AREA_PX` are kept, which drops small
    patches on bare rock.
    """
    applied = np.abs(correction.filled(0.0)) > FOOTPRINT_MIN_CORRECTION
    applied = scipy.ndimage.binary_closing(applied, iterations=FOOTPRINT_CLOSING)
    applied = scipy.ndimage.binary_fill_holes(applied)
    labels, n_labels = scipy.ndimage.label(applied)
    sizes = scipy.ndimage.sum_labels(applied, labels, index=np.arange(1, n_labels + 1))
    large_labels = np.flatnonzero(sizes >= FOOTPRINT_MIN_AREA_PX) + 1
    return np.isin(labels, large_labels)


def add_outline(axis: plt.Axes, mask: np.ndarray, extent: tuple[float, ...]) -> None:
    left, right, bottom, top = extent
    xs = np.linspace(left, right, mask.shape[1])
    ys = np.linspace(top, bottom, mask.shape[0])
    axis.contour(xs, ys, mask.astype("float32"), levels=[0.5], **OUTLINE_KWARGS)


def add_correction_axes(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    correction: np.ma.MaskedArray,
    extent: tuple[float, float, float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> plt.Axes:
    """Add a map of the correction (corrected minus uncorrected rate)."""
    axis = fig.add_axes(rect)
    rgba = CORRECTION_SM.to_rgba(correction.filled(0.0))
    rgba[np.ma.getmaskarray(correction), 3] = 0.0
    axis.imshow(rgba, extent=extent, interpolation="antialiased")
    maps.style_map_axis(axis, xlim, ylim)
    return axis


def plot_tcorr_examples(show: bool = False) -> None:
    fig = plt.figure(figsize=FIG_SIZE.figsize())
    labels = iter("abcdef")

    top_mm = 0.0
    for area in CORRECTION_AREAS:
        xlim, ylim = area.locator.xlim, area.locator.ylim
        height_mm = area.panel_height_mm
        correction, extent = read_correction(xlim, ylim)
        footprint = correction_footprint(correction)
        for col_i, corner_text in enumerate(("Uncorrected", "Corrected", "Correction")):
            left_mm = col_i * PANEL_WIDTH_MM
            rect = FIG_SIZE.rect(left_mm, top_mm, PANEL_WIDTH_MM, height_mm)
            if col_i < 2:
                filepath = UNCORR_PATH if col_i == 0 else CORR_PATH
                axis = maps.add_map_axes(fig, rect, filepath, xlim, ylim, DHDT_SM)
            else:
                axis = add_correction_axes(fig, rect, correction, extent, xlim, ylim)
                # Only here: on the rate maps, the outline would hide the seam it is meant to show
                add_outline(axis, footprint, extent)
            maps.add_corner_text(axis, corner_text)
            maps.add_panel_label(fig, FIG_SIZE, next(labels), left_mm, top_mm)
            if col_i == 0:
                # Keep the bottom row's scale bar clear of the locator map
                is_bottom = top_mm + height_mm >= LOCATOR_BOX_MM[1]
                x_frac = (
                    (LOCATOR_WIDTH_MM + 3.0) / PANEL_WIDTH_MM if is_bottom else 0.035
                )
                maps.add_scale_bar(axis, area.scale_bar_km, x_frac=x_frac)
        top_mm += height_mm

    maps.add_dhdt_colorbar(fig, FIG_SIZE.rect(*DHDT_CBAR_BOX_MM), DHDT_SM)
    maps.add_dhdt_colorbar(fig, FIG_SIZE.rect(*CORRECTION_CBAR_BOX_MM), CORRECTION_SM)
    maps.add_locator(
        fig,
        FIG_SIZE.rect(*LOCATOR_BOX_MM),
        tuple(area.locator for area in CORRECTION_AREAS),
        LOCATOR_XLIM,
        LOCATOR_YLIM,
    )
    maps.add_panel_label(fig, FIG_SIZE, "g", LOCATOR_BOX_MM[0], LOCATOR_BOX_MM[1])

    out_path = FIGURE_DIR / "tcorr_examples.svg"
    fig.savefig(out_path, dpi=maps.DPI)
    svg_png_to_jpg(out_path)
    if show:
        plt.show()
    plt.close(fig)
