from __future__ import annotations

import matplotlib.patheffects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .. import coreg
from ..config import FIGURE_DIR


VERTCOREG_ZONE_NAMES = {
    "austfonna": "Austfonna",
    "isachsenfonna": "Isachsenfonna",
    "kvitoya": "Kvitøya",
    "lomonosovfonna": "Lomonosovfonna",
    "stonebreen": "Stonebreen",
    "vestfonna": "Vestfonna",
}


def plot_rigidcoreg_histograms(show: bool = True):
    data = coreg.read_rigidcoreg_results()

    fig = plt.figure(figsize=(8, 4))
    axes: list[plt.Axes] = list(fig.subplots(2, 2).ravel())

    n_bins = 150
    axes[0].hist(data["icp_slope_m_per_km"], bins=np.linspace(0, 0.8, n_bins), color="#55558e")
    axes[0].set_xlabel("ICP slope correction (m km$^{-1}$)")
    axes[1].hist(data["rigidcoreg_horizontal_shift_m"], bins=np.linspace(0, 30, n_bins), color="#8e5555")
    axes[1].set_xlabel("Horizontal correction magnitude (m)")
    axes[2].hist(data["stable_fraction"] * 100, bins=np.linspace(0, 100, n_bins), color="#555")
    axes[2].set_xlabel("Stable terrain (%)")
    axes[3].hist(data["stable_nmad"], bins=np.linspace(0, 7, n_bins), color="#555")
    axes[3].set_xlabel("Stable terrain NMAD (m)")

    plt.tight_layout()
    for i, axis in enumerate(axes):
        axis.annotate(
            "abcd"[i],
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(5, -5),
            textcoords="offset pixels",
            ha="left",
            va="top",
            path_effects=[matplotlib.patheffects.withStroke(foreground="white", linewidth=2)],
        )

    fig.savefig(FIGURE_DIR / "rigidcoreg_histograms.svg")

    if show:
        plt.show()

    plt.close()


def plot_vertcoreg_summary(show: bool = True):
    data = coreg.read_vertcoreg_results()
    meta = coreg.read_vertcoreg_meta().copy()

    meta["support_percent"] = meta["n_stable_points"] / meta["n_points"] * 100

    zones = [zone for zone in VERTCOREG_ZONE_NAMES if zone in set(meta["vertcoreg_zone"].dropna())]
    pre_color = "#666"
    post_color = "#c77"
    marker_edge = "#666"

    fig = plt.figure(figsize=(8, 4))
    outer = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], wspace=0.18, hspace=0.35)
    ax_ramp = fig.add_subplot(outer[0, 0])
    ax_support = fig.add_subplot(outer[1, 0])
    ax_nmad = fig.add_subplot(outer[:, 1])

    ax_ramp.hist(data["ramp_m_per_km"], bins=np.linspace(0, 0.3, 60), color="#55558e")
    ax_ramp.set_xlabel("Ramp magnitude (m km$^{-1}$)")
    ax_ramp.set_ylabel("Count")

    year_min = int(meta["year"].min())
    year_max = int(meta["year"].max())
    year_count = year_max - year_min + 1
    zone_gap = 3
    zone_stride = year_count + zone_gap
    zone_centers = []

    for zone_index, zone in enumerate(zones):
        subset = meta[meta["vertcoreg_zone"] == zone].sort_values("year")
        x = (subset["year"] - year_min).to_numpy(dtype=float) + zone_index * zone_stride
        for xpos, pre, post in zip(x, subset["nmad_pre"], subset["nmad_post"], strict=False):
            ax_nmad.plot([xpos, xpos], [pre, post], color="#bbb", linewidth=0.8, zorder=1)
        ax_nmad.scatter(x, subset["nmad_pre"], s=18, color=pre_color, edgecolors=marker_edge, linewidths=0.3, zorder=2)
        ax_nmad.scatter(x, subset["nmad_post"], s=18, color=post_color, edgecolors=marker_edge, linewidths=0.3, zorder=3)
        zone_centers.append(zone_index * zone_stride + (year_count - 1) / 2)

    ax_nmad.set_ylabel("Between-pair NMAD (m)")
    ax_nmad.set_xticks(zone_centers)
    ax_nmad.set_xticklabels([])
    for center, zone in zip(zone_centers, zones, strict=False):
        ax_nmad.annotate(
            VERTCOREG_ZONE_NAMES[zone],
            xy=(center, 0),
            xycoords=ax_nmad.get_xaxis_transform(),
            xytext=(7, -3),
            textcoords="offset points",
            ha="right",
            va="top",
            rotation=15,
            clip_on=False,
        )
    ax_nmad.set_xlabel("")
    ax_nmad.set_xlim(-0.5, (len(zones) - 1) * zone_stride + year_count - 0.5)
    ax_nmad.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=pre_color, markeredgecolor=marker_edge, markeredgewidth=0.3, markersize=5, label="Before"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=post_color, markeredgecolor=marker_edge, markeredgewidth=0.3, markersize=5, label="After"),
        ],
        loc="upper right",
        fontsize=8,
        frameon=False,
    )

    for zone_index, zone in enumerate(zones):
        subset = meta[meta["vertcoreg_zone"] == zone]
        mean = float(subset["support_percent"].mean())
        ax_support.bar(zone_index, mean, color="#555", width=0.7, edgecolor="none")
    ax_support.set_ylabel("Stable terrain (%)")
    ax_support.set_xlim(-0.5, len(zones) - 0.5)
    ax_support.set_ylim(0, 100)

    for axis, positions in [(ax_support, range(len(zones))), (ax_nmad, zone_centers)]:
        axis.set_xlabel("")
        axis.set_xticks(list(positions))
        axis.set_xticklabels([])
        for pos, zone in zip(positions, zones, strict=False):
            axis.annotate(
                VERTCOREG_ZONE_NAMES[zone],
                xy=(pos, 0),
                xycoords=axis.get_xaxis_transform(),
                xytext=(7, -3),
                textcoords="offset points",
                ha="right",
                va="top",
                rotation=15,
                clip_on=False,
            )

    for label, axis in zip("abc", [ax_ramp, ax_support, ax_nmad], strict=False):
        axis.annotate(
            label,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(5, -5),
            textcoords="offset pixels",
            ha="left",
            va="top",
            path_effects=[matplotlib.patheffects.withStroke(foreground="white", linewidth=2)],
        )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.96, wspace=0.18, hspace=0.35)
    fig.savefig(FIGURE_DIR / "vertcoreg_summary.svg")

    if show:
        plt.show()

    plt.close(fig)
