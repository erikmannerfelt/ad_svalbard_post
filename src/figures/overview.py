from __future__ import annotations

import matplotlib.pyplot as plt

from .. import outlines
from .common import DHDT_NORMALIZER, DHDT_SM, svg_png_to_jpg


def dhdt_overview_fig(show: bool = True):
    import rasterio
    import rasterio.features

    coasts = outlines.make_coastline_intervals()
    fig = plt.figure(figsize=(8.3, 5.3))
    axes = fig.subplots(1, 2, sharex=True, sharey=True).ravel().tolist()
    all_params = [
        {"filepath": "input/trend_2013-2018_slope.tif", "title": "2013–2018", "coast": coasts.loc["13_18"]},
        {"filepath": "input/trend_2019-2024_slope_tcorr.tif", "title": "2019–2024", "coast": coasts.loc["19_24"]},
    ]
    for i, params in enumerate(all_params):

        scale = rasterio.open(params["filepath"]).scales[0]
        with rasterio.open(params["filepath"], overview_level=3) as raster:
            img = DHDT_SM.to_rgba(raster.read(1, masked=True).astype("float32").filled(0) * scale)
            ocean = rasterio.features.rasterize((params["coast"].geometry,), out_shape=(raster.height, raster.width), fill=0, transform=raster.transform) == 0
            img[ocean, :] = 1.0
            axes[i].imshow(img, extent=(raster.bounds.left, raster.bounds.right, raster.bounds.bottom, raster.bounds.top))
        axes[i].set_xlim(4e5, 7.4e5)
        axes[i].set_ylim(8.5e6, 8.95e6)
        axes[i].set_xlabel("Easting (m; UTM 33N)")
        axes[i].set_title(params["title"])
        axes[i].ticklabel_format(scilimits=(0, 0))

    def add_inset(axis, left=0.6):
        inset = axis.inset_axes((left, 0.02, 0.1, 0.15))
        cbar = plt.colorbar(DHDT_SM, cax=inset, pad=0.02, extend="both")
        text2 = plt.text(1.15, 0.5, "m a$^{-1}$", ha="left", va="center", transform=inset.transAxes)
        cbar.set_ticks([DHDT_NORMALIZER.vmin, DHDT_NORMALIZER.vmax], labels=[f"≤{DHDT_NORMALIZER.vmin:.0f}", f"≥{DHDT_NORMALIZER.vmax:.0f}"])
        return [text2, inset]

    inset_items = add_inset(axes[1])
    axes[0].set_ylabel("Northing (m; UTM 33N)")
    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.98, top=0.95, wspace=0.1)
    out_path = "figures/perinterval_slope_overview.svg"
    plt.savefig(out_path, dpi=400)
    svg_png_to_jpg(out_path)
    for axis in axes:
        axis.set_xlim(614000, 740000)
        axis.set_ylim(8790000, 8950000)
    for item in inset_items:
        item.remove()
    add_inset(axes[1], left=0.75)
    out_path2 = "figures/perinterval_slope_austfonna.svg"
    plt.savefig(out_path2, dpi=400)
    svg_png_to_jpg(out_path2)
    if show:
        plt.show()

    plt.close()
