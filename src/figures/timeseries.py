from __future__ import annotations

import matplotlib.colors
import matplotlib.patheffects
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from matplotlib.lines import Line2D

from .. import tools
from ..config import FIGURE_DIR
from .common import svg_png_to_jpg


YEARS = list(range(2013, 2025))
FALLBACK_2013_REFERENCE_X = 2013.49
FALLBACK_2019_REFERENCE_X = 2019.60


def plot_point_timeseries_trends(show: bool = True):
    data = tools.read_aux_csv("point_timeseries_trends.csv")

    fig = plt.figure(figsize=(8, 5))
    axes = np.atleast_1d(fig.subplots(3, 2, sharex=True)).ravel()
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=1, vmax=10)

    def parse_float(value):
        if value in (None, ""):
            return np.nan
        return float(value)

    def get_reference_x(row, year, fallback):
        doy = parse_float(row.get(f"{year}_doy"))
        if np.isfinite(doy):
            return year + (doy - 1.0) / 365.25
        return float(fallback)

    def plot_trend(axis, slope, intercept, ref_x, start_year, end_year, color, alpha=1.0, lw=1.6):
        xs = np.linspace(start_year, end_year, 200)
        ys = intercept + slope * (xs - ref_x)
        axis.plot(xs, ys, color=color, alpha=alpha, lw=lw)

    def choose_major_step(ymin, ymax):
        span = max(ymax - ymin, 1.0)
        for step in (5.0, 10.0, 20.0, 25.0, 50.0, 100.0):
            if span / step <= 6:
                return step
        return 100.0

    for idx, (_, row) in enumerate(data.iterrows()):
        axis: plt.Axes = axes[idx]
        label = row["label"]

        xs = []
        ys = []
        yerr = []
        for year in YEARS:
            dem = parse_float(row.get(f"{year}_dem"))
            nmad = parse_float(row.get(f"{year}_nmad"))
            doy = parse_float(row.get(f"{year}_doy"))
            x = year + (doy - 1.0) / 365.25 if np.isfinite(doy) else np.nan
            xs.append(x)
            ys.append(dem)
            yerr.append(nmad)

        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        counts = np.asarray([parse_float(row.get(f"{year}_count")) for year in YEARS], dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)

        axis.errorbar(xs[valid], ys[valid], yerr=yerr[valid] * 2, fmt="none", ecolor="0.2", capsize=2, lw=0.9, alpha=0.8)
        axis.scatter(xs[valid], ys[valid], c=cmap(norm(np.clip(counts[valid], 1, 10))), s=20, edgecolors="0.15", linewidths=0.25, zorder=3)

        ref_2013_x = get_reference_x(row, 2013, FALLBACK_2013_REFERENCE_X)
        ref_2019_x = get_reference_x(row, 2019, FALLBACK_2019_REFERENCE_X)

        plot_trend(axis, parse_float(row.get("2013-2024_slope")), parse_float(row.get("2013-2024_intercept")), ref_2013_x, 2013.0, 2025.0, color="0.65", alpha=0.9, lw=2.0)
        plot_trend(axis, parse_float(row.get("2013-2018_slope")), parse_float(row.get("2013-2018_intercept")), ref_2013_x, 2013.0, 2019.0, color="black")
        plot_trend(axis, parse_float(row.get("2019-2024_slope")), parse_float(row.get("2019-2024_intercept")), ref_2019_x, 2019.0, 2025.0, color="black")

        # axis.set_title(label, fontsize=8)
        axis.text(0.5, 0.98, label, fontsize=8, transform=axis.transAxes, ha="center", va="top")
        if idx % 2 == 0:
            axis.set_ylabel("Elevation (m a.sl.)")

        y_min = np.nanmin(ys)
        y_max = np.nanmax(ys)
        if np.isfinite(y_min) and np.isfinite(y_max):
            pad = max(5.0, (y_max - y_min) * 0.15)
            axis.set_ylim(y_min - pad, y_max + pad)
        major_step = choose_major_step(*axis.get_ylim())
        axis.yaxis.set_major_locator(MultipleLocator(major_step))
        axis.yaxis.set_minor_locator(MultipleLocator(2.5))

        axis.grid(True, which="major", axis="y", alpha=0.35, linewidth=0.95)
        axis.grid(True, which="minor", axis="y", alpha=0.35, linewidth=0.65)
        axis.grid(True, which="major", axis="x", alpha=0.15)
        axis.tick_params(axis="y", which="minor", left=True, length=2.5, labelleft=False, color="0.5")

        if label == "Kronebreen terminus retreat":
            cax = axis.inset_axes((0.06, 0.08, 0.08, 0.5))
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation="vertical", extend="max")
            cbar.set_ticks([1, 10])
            cbar.set_ticklabels(["1", "≥10"])
            cbar.ax.tick_params(labelsize=6, length=2)
            cbar.outline.set_linewidth(0.6)
            cbar.set_label("Count", labelpad=-5)

    for axis in axes[len(data):]:
        axis.axis("off")

    for axis in axes[-2:]:
        axis.set_xlabel("Year")

    legend_handles = [
        Line2D([0], [0], color="k", marker="o", lw=1, ms=4, label="Annual medians ± 2x NMAD"),
        Line2D([0], [0], color="black", lw=1.6, label="2013–2018 / 2019–2024 trend"),
        Line2D([0], [0], color="0.65", lw=2.0, label="2013–2024 trend"),
    ]
    axes[-1].legend(handles=legend_handles, loc="lower left", fontsize=7)

    for label, axis in zip("abcdef", axes, strict=False):
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

    fig.subplots_adjust(left=0.08, bottom=0.12, right=0.99, top=0.96, wspace=0.13, hspace=0.1)
    out_path = FIGURE_DIR / "point_timeseries_trends.svg"
    fig.savefig(out_path, bbox_inches="tight")
    svg_png_to_jpg(out_path)

    if show:
        plt.show()

    plt.close(fig)
