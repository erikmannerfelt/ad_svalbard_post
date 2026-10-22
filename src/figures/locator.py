"""Globe locator map showing where Svalbard lies in the Arctic.

Drawn in an orthographic projection with a radial sheen, in the style of the locator
in Mannerfelt's PhD thesis (originally made in QGIS). Countries are the Natural Earth
1:50m admin-0 polygons.
"""

from __future__ import annotations

from typing import Final

import geopandas as gpd
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import shapely

from .. import tools
from ..config import INPUT_DIR

NATURAL_EARTH_URL: Final = (
    "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
)
#: Spherical radius (m) shared by the clipping and display projections.
SPHERE_RADIUS: Final = 6_371_000.0
#: View centre (longitude, latitude), chosen so the North Pole, Svalbard, Greenland and
#: Scandinavia are all well inside the visible hemisphere.
DEFAULT_CENTER: Final = (20.0, 68.0)
#: Svalbard bounding box (lon_min, lat_min, lon_max, lat_max), Kvitøya included.
SVALBARD_BOUNDS: Final = (9.5, 76.2, 34.0, 80.9)
LAND_COLOR: Final = "#8c8c8c"
BORDER_COLOR: Final = "#3a3a3a"
HIGHLIGHT_COLOR: Final = "#e41a1c"


def _orthographic_crs(center: tuple[float, float]) -> str:
    return f"+proj=ortho +lon_0={center[0]} +lat_0={center[1]} +R={SPHERE_RADIUS} +units=m +no_defs"


def visible_countries(
    center: tuple[float, float] = DEFAULT_CENTER,
) -> gpd.GeoDataFrame:
    """Country polygons clipped to the hemisphere facing ``center``, in orthographic m.

    The orthographic projection is undefined on the far side of the globe, so the
    polygons are first clipped in an azimuthal equidistant projection with the same
    centre, where the visible hemisphere is exactly a disk of radius pi/2 * R.
    """
    path = tools.fetch(NATURAL_EARTH_URL, INPUT_DIR / "ne_50m_admin_0_countries.zip")
    countries = gpd.read_file(f"zip://{path}")
    countries = countries[countries["CONTINENT"] != "Antarctica"]
    # Densify so that long straight edges follow the curvature of the projection.
    countries = countries.set_geometry(
        shapely.segmentize(countries.geometry.to_numpy(), 0.5)
    )

    aeqd = (
        f"+proj=aeqd +lon_0={center[0]} +lat_0={center[1]} +R={SPHERE_RADIUS} "
        "+units=m +no_defs"
    )
    # Just inside the horizon, where the orthographic projection is still finite.
    hemisphere = shapely.Point(0, 0).buffer(0.999 * np.pi / 2 * SPHERE_RADIUS, 256)
    clipped = countries.to_crs(aeqd).clip(hemisphere)
    clipped = clipped[~clipped.geometry.is_empty]
    clipped = clipped.set_geometry(
        shapely.segmentize(clipped.geometry.to_numpy(), 50_000)
    )
    projected = clipped.to_crs(_orthographic_crs(center))
    return projected.set_geometry(shapely.make_valid(projected.geometry.to_numpy()))


def _radial_shading(resolution: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Ocean colour and a darkening overlay on a unit disk, lit from the upper left.

    Returns an RGB image for the ocean and an RGBA overlay (black with varying alpha)
    that darkens the limb, including over land.
    """
    coords = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(coords, coords[::-1])
    radius = np.hypot(x, y)
    light_distance = np.hypot(x + 0.35, y - 0.4)

    ocean_value = np.clip(1.0 - 0.13 * light_distance**1.6, 0.0, 1.0)
    ocean = np.repeat(ocean_value[:, :, None], 3, axis=2)

    limb = np.clip(radius, 0, 1) ** 6
    overlay = np.zeros((resolution, resolution, 4))
    overlay[:, :, 3] = np.clip(0.06 * light_distance + 0.30 * limb, 0, 0.45)
    return ocean, overlay


def draw_globe(
    axis: plt.Axes,
    center: tuple[float, float] = DEFAULT_CENTER,
    highlight_bounds: tuple[float, float, float, float] = SVALBARD_BOUNDS,
) -> None:
    """Draw the globe into ``axis`` and frame ``highlight_bounds`` (lon/lat) on it."""
    extent = (-SPHERE_RADIUS, SPHERE_RADIUS, -SPHERE_RADIUS, SPHERE_RADIUS)
    clip_circle = matplotlib.patches.Circle(
        (0, 0), SPHERE_RADIUS, transform=axis.transData
    )

    ocean, overlay = _radial_shading()
    ocean_image = axis.imshow(ocean, extent=extent, zorder=0, interpolation="bilinear")
    ocean_image.set_clip_path(clip_circle)

    visible_countries(center).plot(
        ax=axis,
        color=LAND_COLOR,
        edgecolor=BORDER_COLOR,
        linewidth=0.12,
        zorder=1,
    )

    overlay_image = axis.imshow(
        overlay, extent=extent, zorder=2, interpolation="bilinear"
    )
    overlay_image.set_clip_path(clip_circle)

    lon_min, lat_min, lon_max, lat_max = highlight_bounds
    box = gpd.GeoSeries(
        [
            shapely.segmentize(
                shapely.box(lon_min, lat_min, lon_max, lat_max).exterior, 0.5
            )
        ],
        crs=f"+proj=longlat +R={SPHERE_RADIUS} +no_defs",
    ).to_crs(_orthographic_crs(center))
    box.plot(ax=axis, color=HIGHLIGHT_COLOR, linewidth=1.6, zorder=3)

    axis.add_patch(
        matplotlib.patches.Circle(
            (0, 0), SPHERE_RADIUS, fill=False, edgecolor="#777", linewidth=0.4, zorder=4
        )
    )
    axis.set_xlim(-SPHERE_RADIUS * 1.01, SPHERE_RADIUS * 1.01)
    axis.set_ylim(-SPHERE_RADIUS * 1.01, SPHERE_RADIUS * 1.01)
    axis.set_aspect("equal")
    axis.set_axis_off()
