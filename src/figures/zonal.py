from __future__ import annotations

import matplotlib.patheffects
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator
import numpy as np

from ..config import FIGURE_DIR
from ..outlines import refine_glacier_zones
from ..statistics import get_statistics
from .common import DHDT_SM


def plot_zone_dhdt_fig(show: bool = True):
    all_changes = get_statistics()
    glacier_zones = refine_glacier_zones()

    all_params = [
        {"xcol": "slope_start_mid", "ycol": "slope_mid_end", "partition": "all", "unit": "elev_rate", "vlim": [-1.55, 0.2], "lessneg_text_xy": (0.68, 0.82), "moreneg_text_xy": (0.87, 0.68), "out_stem": "perzone_elevation_change"},
        {"xcol": "slope_start_mid", "ycol": "slope_mid_end", "unit": "elev_rate", "partition": "nonsurging", "vlim": [-1.5, 0.2], "lessneg_text_xy": (0.68, 0.82), "moreneg_text_xy": (0.87, 0.68), "out_stem": "perzone_elevation_change_nonsurging"},
        {"xcol": "slope_start_mid", "ycol": "slope_mid_end", "unit": "vol_rate", "partition": "all", "vlim": [-10.5, 1], "lessneg_text_xy": (0.68, 0.82), "moreneg_text_xy": (0.87, 0.68), "out_stem": "perzone_volume_change"},
        {"xcol": "slope_start_mid", "ycol": "slope_mid_end", "unit": "vol_rate", "partition": "nonsurging", "vlim": [-6, 1], "lessneg_text_xy": (0.68, 0.82), "moreneg_text_xy": (0.87, 0.68), "out_stem": "perzone_volume_change_nonsurging"},
    ]

    for params in all_params:
        fig = plt.figure(figsize=(5, 4.9))
        interval_translation = {"start_end": "2013-2024", "start_mid": "2013-2018", "mid_end": "2019-2024"}
        xcol_interval = interval_translation[params["xcol"].replace("slope_", "")]
        ycol_interval = interval_translation[params["ycol"].replace("slope_", "")]
        unit = all_changes["units"][params["unit"]].replace("~", " ")
        axis_label = {"vol_rate": "Volume change rate", "elev_rate": "Elevation change rate"}.get(params["unit"])
        ax = plt.gca()

        ax.set_title(f"Zonal {axis_label.lower()} " + ("(non-surging)" if "nonsurging" in params["out_stem"] else ""))
        ax.set_ylim(params["vlim"])
        ax.set_xlim(params["vlim"])
        # inset = plt.gca().inset_axes([0.0, 0.5, 0.4, 0.5])
        inset = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax, width="50%", height="50%", loc="upper left", borderpad=0, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes)
        glacier_zones.plot(color=glacier_zones["color"], ax=inset)
        inset.set_xticks([])
        inset.set_yticks([])

        xdata = all_changes["changes"][params["xcol"]][params["partition"]]["per_zone"]
        ydata = all_changes["changes"][params["ycol"]][params["partition"]]["per_zone"]

        for i, (label, zone_xdata) in enumerate(xdata.items()):
            zone = glacier_zones.loc[label]
            zone_ydata = ydata[label]

            ax.errorbar(
                x=zone_xdata[params["unit"]],
                y=zone_ydata[params["unit"]],
                xerr=zone_xdata[params["unit"] + "_err"],
                yerr=zone_ydata[params["unit"] + "_err"],
                color=np.array(matplotlib.colors.to_rgb(zone["color"])) * 0.6,
                marker="o",
                markersize=0.8 * zone_xdata["area"] / 1e2,
                markerfacecolor=zone["color"],
                markeredgecolor="#ccc",
                barsabove=True,
                alpha=1.0,
                zorder=i + 1,
            )

            xy_text = (zone_xdata[params["unit"]], zone_ydata[params["unit"]])
            text_kwargs = {
                "ha": "center",
                "va": "center",
                "path_effects": [matplotlib.patheffects.withStroke(foreground="black", linewidth=1)],
                "color": "white",
            }

            ax.annotate(label, xy_text, zorder=i + 300, **text_kwargs)
            inset.annotate(label, (zone.geometry.centroid.x, zone.geometry.centroid.y), **text_kwargs)

        ax.fill_between(params["vlim"], params["vlim"], [max(params["vlim"])] * 2, color=DHDT_SM.to_rgba(1), alpha=0.2)
        ax.text(*params["lessneg_text_xy"], "Less\nnegative", transform=ax.transAxes, color=np.array(DHDT_SM.to_rgba(1)[:3]) * 0.7, ha="center", fontsize=12)
        ax.text(*params["moreneg_text_xy"], "More\nnegative", transform=ax.transAxes, color=np.array(DHDT_SM.to_rgba(-1)[:3]) * 0.7, ha="center", fontsize=12)
        ax.fill_between(params["vlim"], params["vlim"], [min(params["vlim"])] * 2, color=DHDT_SM.to_rgba(-1), alpha=0.2)
        ax.plot(params["vlim"], params["vlim"], color="#333", linestyle="--", zorder=0)
        ax.set_xlabel(f"{axis_label} {xcol_interval} ({unit})")
        ax.set_ylabel(f"{axis_label} {ycol_interval} ({unit})")
        fig.subplots_adjust(top=0.928,
            bottom=0.134,
            left=0.171,
            right=0.948,
            hspace=0,
            wspace=0)
        # fig.tight_layout()


        plt.savefig(FIGURE_DIR / f"{params['out_stem']}.svg")

        if show:
            plt.show()

        plt.close()
