from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.patheffects
import pandas as pd
import numpy as np

from .. import outlines
from .. import statistics
from . import locator
from .common import DHDT_NORMALIZER, DHDT_SM, svg_png_to_jpg


def dhdt_overview_fig(show: bool = True):
    import rasterio
    import rasterio.features

    coasts = outlines.make_coastline_intervals()
    glacier_zones = outlines.refine_glacier_zones()

    # glacier_zones.loc["S", "color"] = "#7b3294"
    # glacier_zones.loc["B", "color"] = "#855b78"


    data = statistics.compute_hypsometric_profiles()
    zone_meta = data["zone_meta"].drop(index="K", errors="ignore")
    per_zone = data["per_zone"]
    intervals = data["intervals"]

    import matplotlib.pyplot as plt


    xmin = 402_000
    xmax = 767_217
    # The top edge cuts off Sjuøyane (unglaciated islets) to free the corner for the globe
    ymin = 8_468_000
    ymax = 8_950_000
    box_aspect = (ymax - ymin) / (xmax - xmin)  # height / width

    # Kvitøya extent (glacier bounds plus a margin) and the inset's lower-left corner in map coordinates
    kvitoya_xlim = (809_000, 848_000)
    kvitoya_ylim = (8_930_000, 8_970_000)
    kvitoya_anchor = (712_000, 8_928_000)
    kvitoya_offset = (kvitoya_anchor[0] - kvitoya_xlim[0], kvitoya_anchor[1] - kvitoya_ylim[0])

    fig = plt.figure(figsize=(8.3, 6.6)) # TODO: GLR allows only 7.09 inch wide figures

    outer = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[5.3, 0.9],
        left=0.055,
        right=0.985,
        bottom=0.065,
        top=0.955,
        hspace=0.035,
    )


    # ------------------------------------------------------------------
    # Two large maps
    # ------------------------------------------------------------------

    gs_maps = outer[0].subgridspec(
        nrows=1,
        ncols=2,
        wspace=0.035,
    )

    ax_map1 = fig.add_subplot(gs_maps[0, 0])
    ax_map2 = fig.add_subplot(gs_maps[0, 1])

    for ax in (ax_map1, ax_map2):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Coordinate units are metres in both directions
        ax.set_aspect("equal", adjustable="box")

        # Height / width of the physical axes
        ax.set_box_aspect(box_aspect)

    map_axes = [ax_map1, ax_map2]
    

    # ------------------------------------------------------------------
    # Eight small panels in one row
    # ------------------------------------------------------------------

    gs_small = outer[1].subgridspec(
        nrows=1,
        ncols=8,
        wspace=0.12,
    )

    ax_small = [fig.add_subplot(gs_small[0, 0])]

    for i in range(1, 8):
        ax_small.append(
            fig.add_subplot(
                gs_small[0, i],
                sharey=ax_small[0],
            )
        )

    for ax in ax_small:
        ax.label_outer()

    # zone_inset = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax, width="50%", height="50%", loc="upper left", borderpad=0, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes)
    zone_inset = fig.add_axes((0., 0.23, 0.16, 0.16 * box_aspect))


    all_axes = [*map_axes,zone_inset, *ax_small]
    # The map labels are moved right of the globe in the corner of panel a, the same in b.
    # They and the globe are top-aligned with the Kvitøya inset, which sticks out above
    # the map frame.
    map_label_x = 0.27
    map_top_y = (kvitoya_ylim[1] + kvitoya_anchor[1] - kvitoya_ylim[0] - ymin) / (ymax - ymin)
    for label, ax in zip("abcdefghijk", all_axes):
        ax.text(
            map_label_x if ax in map_axes else 0.02, map_top_y if ax in map_axes else 0.98, f"{label})",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )


    # fig = plt.figure(figsize=(8.3, 5.3))
    # axes = fig.subplots(1, 2, sharex=True, sharey=True).ravel().tolist()
    all_params = [
        {"filepath": "input/trend_2013-2018_slope.tif", "title": "2013–2018", "coast": coasts.loc["13_18"]},
        {"filepath": "input/trend_2019-2024_slope.tif", "title": "2019–2024", "coast": coasts.loc["19_24"]},
    ]
    for i, params in enumerate(all_params):

        scale = rasterio.open(params["filepath"]).scales[0]
        with rasterio.open(params["filepath"], overview_level=3) as raster:
            img = DHDT_SM.to_rgba(raster.read(1, masked=True).astype("float32").filled(0) * scale)
            ocean = rasterio.features.rasterize((params["coast"].geometry,), out_shape=(raster.height, raster.width), fill=0, transform=raster.transform) == 0
            img[ocean, :] = 1.0
            img_extent = (raster.bounds.left, raster.bounds.right, raster.bounds.bottom, raster.bounds.top)
            map_axes[i].imshow(img, extent=img_extent)

        # Kvitøya lies ~60 km east of the frame. Show it in an inset at the same map scale,
        # placed just north of Austfonna at roughly its true northing.
        kvitoya_width = kvitoya_xlim[1] - kvitoya_xlim[0]
        kvitoya_height = kvitoya_ylim[1] - kvitoya_ylim[0]
        kvitoya_ax = map_axes[i].inset_axes(
            (kvitoya_anchor[0], kvitoya_anchor[1], kvitoya_width, kvitoya_height),
            transform=map_axes[i].transData,
        )
        kvitoya_ax.imshow(img, extent=img_extent)
        kvitoya_ax.set_xlim(kvitoya_xlim)
        kvitoya_ax.set_ylim(kvitoya_ylim)
        kvitoya_ax.set_xticks([])
        kvitoya_ax.set_yticks([])
        for spine in kvitoya_ax.spines.values():
            spine.set_edgecolor("#555")
            spine.set_linewidth(0.6)
        kvitoya_ax.text(-0.04, 0.5, "Kvitøya", transform=kvitoya_ax.transAxes, ha="right", va="center", fontsize=7, color="#555")
        map_axes[i].annotate(
            "",
            xy=(xmax - 2e3, kvitoya_anchor[1] + kvitoya_height / 2),
            xytext=(kvitoya_anchor[0] + kvitoya_width, kvitoya_anchor[1] + kvitoya_height / 2),
            xycoords="data",
            arrowprops={"arrowstyle": "->", "color": "#555", "linewidth": 0.6, "shrinkA": 0, "shrinkB": 0},
        )
        map_axes[i].set_xlim(xmin, xmax)
        map_axes[i].set_ylim(ymin, ymax)
        map_axes[i].set_aspect("equal")
        map_axes[i].set_axis_off()
        map_axes[i].text(map_label_x + 0.07, map_top_y, params["title"], transform=map_axes[i].transAxes, fontsize=12, va="top")
        map_axes[i].ticklabel_format(scilimits=(0, 0))

        if i == 0:
            # Globe locator in the open sea northwest of Spitsbergen.
            globe_width = 0.25
            globe_ax = map_axes[i].inset_axes(
                (0.0, map_top_y - globe_width / box_aspect, globe_width, globe_width / box_aspect)
            )
            locator.draw_globe(globe_ax)

            scale_km = 100
            scale_xmax = map_axes[i].get_xlim()[1] - 1e4
            scale_y = map_axes[i].get_ylim()[0] + 7e4
            map_axes[i].plot([scale_xmax - scale_km * 1000, scale_xmax], [scale_y] * 2, color="#555")
            for label, xcoord in [
                ("0 km", scale_xmax - scale_km * 1000),
                (f"{scale_km} km", scale_xmax),
            ]:
                map_axes[i].text(
                    xcoord,
                    scale_y - 3e3,
                    label,
                    transform=map_axes[i].transData,
                    fontsize=8,
                    ha="center",
                    va="top",
                    color="#555",
                )
                map_axes[i].plot([xcoord] * 2, [scale_y, scale_y + 1e4], color="#555")
            

    def add_inset_cbar(axis, left=0.6):
        inset = axis.inset_axes((left, 0.08, 0.1, 0.15))
        cbar = plt.colorbar(DHDT_SM, cax=inset, pad=0.02, extend="both")
        text2 = plt.text(1.15, 0.5, "m a$^{-1}$", ha="left", va="center", transform=inset.transAxes)
        cbar.set_ticks([DHDT_NORMALIZER.vmin, DHDT_NORMALIZER.vmax], labels=[f"≤{DHDT_NORMALIZER.vmin:.0f}", f"≥{DHDT_NORMALIZER.vmax:.0f}"])
        return [text2, inset]

    inset_items = add_inset_cbar(map_axes[1])

    # Draw Kvitøya where the inset sits in the main maps, framed the same way
    glacier_zones.loc[["K"], "geometry"] = glacier_zones.loc[["K"]].translate(*kvitoya_offset)
    glacier_zones.plot(color=glacier_zones["color"], ax=zone_inset)
    zone_inset.add_patch(
        matplotlib.patches.Rectangle(
            kvitoya_anchor,
            kvitoya_xlim[1] - kvitoya_xlim[0],
            kvitoya_ylim[1] - kvitoya_ylim[0],
            fill=False,
            edgecolor="#555",
            linewidth=0.4,
        )
    )
    zone_inset.set_xticks([])
    zone_inset.set_yticks([])
    for label, zone in glacier_zones.iterrows():
            text_kwargs = {
                "ha": "center",
                "va": "center",
                "path_effects": [matplotlib.patheffects.withStroke(foreground="black", linewidth=1)],
                "color": "white",
            }

            zone_inset.annotate(str(label), (zone.geometry.centroid.x, zone.geometry.centroid.y),fontsize=8, **text_kwargs)


    if zone_meta.index[-1] == "B":
        zone_meta = zone_meta.iloc[np.r_[np.arange(zone_meta.shape[0] - 2), [-1, -2]]]

    for i, (label, zone) in enumerate(zone_meta.iterrows()):
        if pd.isna(zone["zone_name"]):
            continue
        axis: plt.Axes = ax_small[i]
        axis.set_title(str(zone["zone_name"]).replace(" / ", " /\n"), fontsize=8)

        for interval in intervals:
            d = per_zone[interval.interval.short][label]
            if d.empty:
                continue

            color = DHDT_SM.to_rgba(1.0 if interval.interval.short == "13_18" else -1.0)
            axis.errorbar(d.index, d[f"{interval.interval.short}_med"], d[f"{interval.interval.short}_nmad"], color=color, alpha=0.5, fmt="none")

            axis.scatter(d.index, d[f"{interval.interval.short}_med"], marker="s", edgecolor="#777", color=color, label=interval.interval.display_label)

        xlim = axis.get_xlim()
        axis.hlines(0, *xlim, color="#777", linestyles="--", alpha=0.5)
        axis.set_xlim(xlim)
        axis.set_ylim(-4.2, 0.99)

        axis.set_xticks([0, 750 if xlim[1] > 1000. else 500])

        if i == 0:
            axis.set_ylabel("dH dt$^{-1}$ (m a$^{-1}$)", labelpad=0)
        # if col >= ncols:
        #     axis.set_xlabel("Elevation (m a.s.l.)")
        if i == (len(zone_meta) - 1):
            axis.legend(fontsize=7,
                # labelspacing=0.2,     # Reduces vertical space between rows
                handletextpad=0.2,    # Reduces horizontal space between icon and text
                borderaxespad=0.4,    # Reduces space between legend border and axes
                handlelength=1.5,      # Shortens the length of the legend lines
                loc="lower right")
    fig.text(0.5, 0.005, "Elevation (m a.s.l.)", ha="center", va="bottom", transform=fig.transFigure, fontsize=10)

    # axes[0].set_ylabel("Northing (m; UTM 33N)")
    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.9, top=0.95, wspace=0)
    out_path = "figures/perinterval_slope_overview.svg"
    plt.savefig(out_path, dpi=400)
    svg_png_to_jpg(out_path)


    return
    for axis in axes:
        axis.set_xlim(614000, 740000)
        axis.set_ylim(8790000, 8950000)
    for item in inset_items:
        item.remove()
    add_inset_cbar(axes[1], left=0.75)
    out_path2 = "figures/perinterval_slope_austfonna.svg"
    plt.savefig(out_path2, dpi=400)
    svg_png_to_jpg(out_path2)
    if show:
        plt.show()

    plt.close()
