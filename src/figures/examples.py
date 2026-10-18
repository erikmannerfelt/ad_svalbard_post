"""Main-text figure of elevation-change examples (maps + point timeseries).

Four example areas are shown in rows: the 2013–2018 rate map (left), the annual timeseries at
the red dot (middle) and the 2019–2024 rate map (right). A locator map of the four areas sits
in the lower-right corner. The map extents are those of the QGIS layout the figure was first
composed in, so that the recreated figure frames the same areas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import geopandas as gpd
import matplotlib.patches
import matplotlib.pyplot as plt

from .. import tools
from ..config import FIGURE_DIR, INPUT_DIR
from . import maps, timeseries
from .common import DHDT_SM, svg_png_to_jpg

FIG_SIZE: Final = maps.FigureSizeMM(width=180.0, height=4 * 41.55)
CELL_WIDTH_MM: Final = 60.0
CELL_HEIGHT_MM: Final = 41.55
N_ROWS: Final = 4


@dataclass(frozen=True)
class ExampleArea:
    """One row of the figure: a map extent and the point timeseries shown in it."""

    point_key: str
    locator: maps.LocatorArea


EXAMPLE_AREAS: Final = (
    ExampleArea(
        "kronebreen_terminus",
        maps.LocatorArea(
            (421_625.3, 475_488.6),
            (8_732_554.7, 8_769_855.1),
            "a–c",
            (0, 36_000),
        ),
    ),
    ExampleArea(
        "storbreen_acc",
        maps.LocatorArea(
            (521_043.7, 565_357.3),
            (8_548_541.5, 8_579_228.7),
            "d–f",
            (52_000, 0),
        ),
    ),
    ExampleArea(
        "negribreen_central",
        maps.LocatorArea(
            (552_776.5, 597_090.1),
            (8_709_855.3, 8_740_542.4),
            "g–i",
            (0, 36_000),
        ),
    ),
    ExampleArea(
        "austfonna_north",
        maps.LocatorArea(
            (642_931.8, 696_795.2),
            (8_892_515.0, 8_929_815.4),
            "j–l",
            (0, -40_000),
        ),
    ),
)
LOCATOR_XLIM: Final = (390_762.9, 782_158.9)
LOCATOR_YLIM: Final = (8_477_170.0, 8_959_371.1)
# Locator position and size (left, top, width, height) in mm from the figure's top-left corner
LOCATOR_BOX_MM: Final = (120.0, 103.42, 27.34, 33.68)
# dH/dt colorbar box, straddling the g/j boundary in the left column
DHDT_CBAR_BOX_MM: Final = (0.0, 104.0, 17.0, 31.0)

SCALE_BAR_KM: Final = (0, 5, 10)


def _plot_count_colorbar(axis: plt.Axes) -> None:
    cax = axis.inset_axes((0.8, 0.45, 0.04, 0.48))
    sm = plt.cm.ScalarMappable(norm=timeseries.COUNT_NORM, cmap=timeseries.COUNT_CMAP)
    cbar = axis.figure.colorbar(sm, cax=cax, extend="max")
    cbar.set_ticks([1, 10], labels=["1", "≥10"])
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label("Count", fontsize=6, labelpad=1)


def plot_elevation_change_examples(show: bool = False) -> None:
    points = tools.read_aux_csv("point_timeseries_trends.csv").set_index("key")
    point_locations = gpd.read_file(
        f"/vsizip/{INPUT_DIR / 'aux_files.zip'}/point_timeseries_trends_points.geojson"
    ).set_index("key")

    fig = plt.figure(figsize=FIG_SIZE.figsize())
    raster_paths = (
        INPUT_DIR / "trend_2013-2018_slope.tif",
        INPUT_DIR / "trend_2019-2024_slope.tif",
    )
    labels = iter("abcdefghijkl")
    timeseries_axes = []

    for row_i, area in enumerate(EXAMPLE_AREAS):
        top_mm = row_i * CELL_HEIGHT_MM
        point = point_locations.loc[area.point_key].geometry

        for col_i, filepath in zip((0, 2), raster_paths):
            axis = maps.add_map_axes(
                fig,
                FIG_SIZE.rect(
                    col_i * CELL_WIDTH_MM, top_mm, CELL_WIDTH_MM, CELL_HEIGHT_MM
                ),
                filepath,
                area.locator.xlim,
                area.locator.ylim,
                DHDT_SM,
            )
            axis.plot(
                point.x, point.y, "o", color=maps.MARKER_COLOR, ms=5.5, mec="none"
            )
            if col_i == 0:
                # Lift the scale bar of the row above the dH/dt colorbar box clear of it
                cbar_top_mm = DHDT_CBAR_BOX_MM[1]
                row_bottom_mm = top_mm + CELL_HEIGHT_MM
                if top_mm < cbar_top_mm < row_bottom_mm:
                    maps.add_scale_bar(
                        axis,
                        SCALE_BAR_KM,
                        y_frac=(row_bottom_mm - cbar_top_mm) / CELL_HEIGHT_MM + 0.06,
                    )
                else:
                    maps.add_scale_bar(axis, SCALE_BAR_KM)

        ts_axis = fig.add_axes(
            FIG_SIZE.rect(
                CELL_WIDTH_MM + 10.5,
                top_mm + 1.5,
                CELL_WIDTH_MM - 12.0,
                CELL_HEIGHT_MM - 6.5,
            )
        )
        timeseries.draw_panel(
            ts_axis,
            points.loc[area.point_key],
            timeseries.COUNT_CMAP,
            timeseries.COUNT_NORM,
        )
        timeseries_axes.append(ts_axis)

        for col_i in range(3):
            left_mm = col_i * CELL_WIDTH_MM
            # Shift a left-column label that would sit under the dH/dt colorbar box
            if (
                col_i == 0
                and DHDT_CBAR_BOX_MM[1]
                < top_mm
                < DHDT_CBAR_BOX_MM[1] + DHDT_CBAR_BOX_MM[3]
            ):
                left_mm = DHDT_CBAR_BOX_MM[2]
            # Move the label of the cell under the locator to its right, at the same height
            if (
                col_i == 2
                and LOCATOR_BOX_MM[1] < top_mm < LOCATOR_BOX_MM[1] + LOCATOR_BOX_MM[3]
            ):
                left_mm = LOCATOR_BOX_MM[0] + LOCATOR_BOX_MM[2]
            maps.add_panel_label(fig, FIG_SIZE, next(labels), left_mm, top_mm)

    # Cell frames for the timeseries column, matching the map frames
    for row_i in range(N_ROWS):
        fig.add_artist(
            matplotlib.patches.Rectangle(
                FIG_SIZE.rect(
                    CELL_WIDTH_MM, row_i * CELL_HEIGHT_MM, CELL_WIDTH_MM, CELL_HEIGHT_MM
                )[:2],
                CELL_WIDTH_MM / FIG_SIZE.width,
                CELL_HEIGHT_MM / FIG_SIZE.height,
                transform=fig.transFigure,
                fill=False,
                linewidth=maps.FRAME_LINEWIDTH,
                edgecolor="black",
                zorder=5,
            )
        )

    _plot_count_colorbar(timeseries_axes[2])
    timeseries_axes[3].legend(
        handles=timeseries.legend_handles(),
        loc="lower center",
        fontsize=6,
        frameon=False,
        handlelength=1.8,
    )
    maps.add_dhdt_colorbar(fig, FIG_SIZE.rect(*DHDT_CBAR_BOX_MM), DHDT_SM)
    maps.add_locator(
        fig,
        FIG_SIZE.rect(*LOCATOR_BOX_MM),
        tuple(area.locator for area in EXAMPLE_AREAS),
        LOCATOR_XLIM,
        LOCATOR_YLIM,
    )
    maps.add_panel_label(fig, FIG_SIZE, "m", LOCATOR_BOX_MM[0], LOCATOR_BOX_MM[1])

    out_path = FIGURE_DIR / "elevation_change_examples.svg"
    fig.savefig(out_path, dpi=maps.DPI)
    svg_png_to_jpg(out_path)
    if show:
        plt.show()
    plt.close(fig)
