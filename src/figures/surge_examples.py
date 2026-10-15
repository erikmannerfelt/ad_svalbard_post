"""SI figure of dH/dt over the three largest surges of the study period.

One row per surge (Storisstraumen, Negribreen, Stonebreen), with the 2013–2018 rates on the
left and the 2019–2024 rates on the right. The colour scale is three times that of the
main-text maps. The extents are approximations of those in the lost original layout; each row's
height follows from its extent's aspect ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt

from ..config import FIGURE_DIR, INPUT_DIR
from . import maps
from .common import DHDT_NORMALIZER, DHDT_SM, svg_png_to_jpg

# Three times the main-text range, so the surge signals do not saturate
SURGE_SCALE_FACTOR: Final = 3.0
SURGE_SM: Final = matplotlib.cm.ScalarMappable(
    norm=matplotlib.colors.Normalize(
        vmin=SURGE_SCALE_FACTOR * DHDT_NORMALIZER.vmin,
        vmax=SURGE_SCALE_FACTOR * DHDT_NORMALIZER.vmax,
        clip=True,
    ),
    cmap=DHDT_SM.cmap,
)

FIG_WIDTH_MM: Final = 140.0
PANEL_WIDTH_MM: Final = FIG_WIDTH_MM / 2


@dataclass(frozen=True)
class SurgeArea:
    locator: maps.LocatorArea
    scale_bar_km: tuple[int, ...]

    @property
    def panel_height_mm(self) -> float:
        xlim, ylim = self.locator.xlim, self.locator.ylim
        return PANEL_WIDTH_MM * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])


SURGE_AREAS: Final = (
    # Storisstraumen
    SurgeArea(
        maps.LocatorArea(
            (658_089, 729_291), (8_818_241, 8_879_582), "a, b", (0, -42_000)
        ),
        (0, 10, 20),
    ),
    # Negribreen
    SurgeArea(
        maps.LocatorArea(
            (563_655, 594_871), (8_721_323, 8_740_699), "c, d", (0, 22_000)
        ),
        (0, 5, 10),
    ),
    # Stonebreen
    SurgeArea(
        maps.LocatorArea(
            (694_402, 718_810), (8_634_138, 8_652_591), "e, f", (0, 22_000)
        ),
        (0, 5, 10),
    ),
)
FIG_SIZE: Final = maps.FigureSizeMM(
    width=FIG_WIDTH_MM, height=sum(area.panel_height_mm for area in SURGE_AREAS)
)

LOCATOR_XLIM: Final = (470_000.0, 780_000.0)
LOCATOR_YLIM: Final = (8_590_000.0, 8_920_000.0)
LOCATOR_WIDTH_MM: Final = 26.0
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
# dH/dt colorbar box, at the right edge of the top-right panel below its interval label
DHDT_CBAR_BOX_MM: Final = (FIG_WIDTH_MM - 17.0, 7.0, 17.0, 16.5)

INTERVALS: Final = (
    ("2013–2018", INPUT_DIR / "trend_2013-2018_slope.tif"),
    ("2019–2024", INPUT_DIR / "trend_2019-2024_slope.tif"),
)


def plot_surge_examples(show: bool = False) -> None:
    fig = plt.figure(figsize=FIG_SIZE.figsize())
    labels = iter("abcdef")

    top_mm = 0.0
    for area in SURGE_AREAS:
        height_mm = area.panel_height_mm
        for col_i, (interval_label, filepath) in enumerate(INTERVALS):
            left_mm = col_i * PANEL_WIDTH_MM
            axis = maps.add_map_axes(
                fig,
                FIG_SIZE.rect(left_mm, top_mm, PANEL_WIDTH_MM, height_mm),
                filepath,
                area.locator.xlim,
                area.locator.ylim,
                SURGE_SM,
            )
            maps.add_corner_text(axis, interval_label)
            maps.add_panel_label(fig, FIG_SIZE, next(labels), left_mm, top_mm)
            if col_i == 0:
                # Keep the bottom row's scale bar clear of the locator map
                is_bottom = top_mm + height_mm >= LOCATOR_BOX_MM[1]
                x_frac = (
                    (LOCATOR_WIDTH_MM + 3.0) / PANEL_WIDTH_MM if is_bottom else 0.035
                )
                maps.add_scale_bar(axis, area.scale_bar_km, x_frac=x_frac)
        top_mm += height_mm

    maps.add_dhdt_colorbar(fig, FIG_SIZE.rect(*DHDT_CBAR_BOX_MM), SURGE_SM)
    maps.add_locator(
        fig,
        FIG_SIZE.rect(*LOCATOR_BOX_MM),
        tuple(area.locator for area in SURGE_AREAS),
        LOCATOR_XLIM,
        LOCATOR_YLIM,
    )
    maps.add_panel_label(fig, FIG_SIZE, "g", LOCATOR_BOX_MM[0], LOCATOR_BOX_MM[1])

    out_path = FIGURE_DIR / "surging_examples.svg"
    fig.savefig(out_path, dpi=maps.DPI)
    svg_png_to_jpg(out_path)
    if show:
        plt.show()
    plt.close(fig)
