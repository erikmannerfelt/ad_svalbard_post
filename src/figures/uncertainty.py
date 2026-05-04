from __future__ import annotations

import geopandas as gpd
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from ..config import FIGURE_DIR
from .. import sampling, statistics, tools
from .common import svg_png_to_jpg


def plot_terrain_err(show: bool = True):
    data = tools.read_aux_csv("binned_terrain_err_trend_2019-2024_slope.csv", index_col=0)
    data["count"] = data["count"].astype(int)
    xcol = "terr_slope"
    ycol = "terr_curvature"

    for col in ["terr_slope", "terr_curvature", "easting", "northing"]:
        data[col] = data[col].apply(tools.pandas_str_to_interval)

    e_bins = pd.IntervalIndex(sorted(data["easting"].dropna().unique()))
    n_bins = pd.IntervalIndex(sorted(data["northing"].dropna().unique()))
    e_merged = [pd.Interval(e_bins[0].left, e_bins[1].right, closed=e_bins[0].closed), pd.Interval(e_bins[2].left, e_bins[3].right, closed=e_bins[2].closed)]
    n_merged = [pd.Interval(n_bins[0].left, n_bins[1].right, closed=n_bins[0].closed), pd.Interval(n_bins[2].left, n_bins[3].right, closed=n_bins[2].closed)]
    data["easting_merged"] = data["easting"].map({e_bins[0]: e_merged[0], e_bins[1]: e_merged[0], e_bins[2]: e_merged[1], e_bins[3]: e_merged[1]})
    data["northing_merged"] = data["northing"].map({n_bins[0]: n_merged[0], n_bins[1]: n_merged[0], n_bins[2]: n_merged[1], n_bins[3]: n_merged[1]})

    data["_nmad_weighted"] = data["nmad"] * data["count"]
    data = data.groupby(["easting_merged", "northing_merged", xcol, ycol], as_index=False).agg(nd=("nd", "first"), count=("count", "sum"), _nmad_weighted=("_nmad_weighted", "sum"))
    data["nmad"] = data["_nmad_weighted"] / data["count"]
    data = data.drop(columns=["_nmad_weighted"])

    data["e_mid"] = pd.IntervalIndex(data["easting_merged"]).mid
    data["n_mid"] = pd.IntervalIndex(data["northing_merged"]).mid
    e_cols = np.sort(data["e_mid"].dropna().unique())
    n_cols = np.sort(data["n_mid"].dropna().unique())

    data = data[
        np.logical_and.reduce(
            (
                data.nd == 4,
                np.isfinite(pd.IntervalIndex(data["terr_slope"]).mid),
                np.isfinite(pd.IntervalIndex(data["terr_curvature"]).mid),
            )
        )
    ]

    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-1.3, -0.6), cmap="cividis")
    edge_to_percentile = {}
    ticks = {}
    tick_labels = {}
    for col in [xcol, ycol]:
        edges = np.unique(np.r_[pd.IntervalIndex(data[col]).right, pd.IntervalIndex(data[col]).left])
        percentiles = np.linspace(0, 100, edges.size)
        edge_to_percentile[col] = lambda val, edges=edges, percentiles=percentiles: np.interp(val, edges, percentiles)
        ticks[col] = percentiles
        tick_labels[col] = [f"{0 if edge < 1e-2 else edge:.2g}" for edge in edges]

    max_top_count = 0
    max_right_count = 0
    for n_mid in n_cols[::-1]:
        sub0 = data[data["n_mid"] == n_mid]
        for e_mid in e_cols:
            subset = sub0[sub0["e_mid"] == e_mid]
            top_max = subset.groupby(xcol, sort=False)["count"].sum().max()
            right_max = subset.groupby(ycol, sort=False)["count"].sum().max()
            max_top_count = max(max_top_count, 0 if pd.isna(top_max) else top_max)
            max_right_count = max(max_right_count, 0 if pd.isna(right_max) else right_max)

    fig = plt.figure()
    outer = fig.add_gridspec(n_cols.shape[0], e_cols.shape[0], wspace=0.25, hspace=0.25)

    for i, n_mid in enumerate(n_cols[::-1]):
        sub0 = data[data["n_mid"] == n_mid]
        for j, e_mid in enumerate(e_cols):
            subset = sub0[sub0["e_mid"] == e_mid]
            panel = outer[i, j].subgridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1], wspace=0.05, hspace=0.05)
            ax_top: plt.Axes = fig.add_subplot(panel[0, 0])
            ax: plt.Axes = fig.add_subplot(panel[1, 0], sharex=ax_top)
            ax_right: plt.Axes = fig.add_subplot(panel[1, 1], sharey=ax)

            top_counts = subset.groupby(xcol, sort=False)["count"].sum()
            right_counts = subset.groupby(ycol, sort=False)["count"].sum()

            for interval, count in top_counts.items():
                left = edge_to_percentile[xcol](interval.left)
                right = edge_to_percentile[xcol](interval.right)
                ax_top.bar(left + (right - left) / 2, count, width=right - left, align="center", color="#666", edgecolor="#444", linewidth=0.5)

            for interval, count in right_counts.items():
                bottom = edge_to_percentile[ycol](interval.left)
                top = edge_to_percentile[ycol](interval.right)
                ax_right.barh(bottom + (top - bottom) / 2, count, height=top - bottom, align="center", color="#666", edgecolor="#444", linewidth=0.5)

            for _, row in subset.iterrows():
                ax.add_patch(
                    plt.Rectangle(
                        (edge_to_percentile[xcol](row[xcol].left), edge_to_percentile[ycol](row[ycol].left)),
                        width=edge_to_percentile[xcol](row[xcol].right) - edge_to_percentile[xcol](row[xcol].left),
                        height=edge_to_percentile[ycol](row[ycol].right) - edge_to_percentile[ycol](row[ycol].left),
                        facecolor=sm.to_rgba(np.log10(np.clip(row["nmad"], a_min=1e-3, a_max=np.inf))),
                        edgecolor="#777",
                        linewidth=1,
                    )
                )

            for col, axis in [(xcol, ax.xaxis), (ycol, ax.yaxis)]:
                axis.set_ticks(ticks[col], tick_labels[col], fontsize=8)
                axis._set_lim(0, 100, auto=False)

            if i == n_cols.shape[0] - 1:
                ax.set_xlabel("Slope (°)")
            if j == 0:
                ax.set_ylabel("Curvature (100/m)")

            ax_top.set_xlim(0, 100)
            ax_right.set_ylim(0, 100)
            ax_top.set_ylim(0, max_top_count)
            ax_right.set_xlim(0, max_right_count)

            quad_label = ("N" if i == 0 else "S") + ("W" if j == 0 else "E")
            ax_top.text(0.02, 0.95, quad_label, transform=ax_top.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")

            ax_top.tick_params(axis="x", labelbottom=False)
            ax_top.tick_params(axis="y", left=False, labelleft=False)
            ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
            ax_right.tick_params(axis="y", labelleft=False, labelright=False)

            ax_top.spines["right"].set_visible(False)
            ax_top.spines["top"].set_visible(False)
            ax_right.spines["top"].set_visible(False)
            ax_right.spines["right"].set_visible(False)

    cax = fig.add_axes((0.43, 0.53, 0.14, 0.025))
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("NMAD (m a$^{-1}$)", fontsize=8)
    ticks2 = np.linspace(sm.norm.vmin, sm.norm.vmax, 3)
    cbar.set_ticks(ticks2, labels=[f"{10**tick:.1g}" for tick in ticks2])

    fig.subplots_adjust(top=0.995, bottom=0.085, left=0.09, right=0.995, hspace=0.045, wspace=0.2)
    fig.savefig(FIGURE_DIR / "binned_terrain_err.svg")

    if show:
        plt.show()

    plt.close()


def plot_patch_method_vs_vgm(show: bool = True):
    outlines = sampling.sample_rasters()
    intervals = tools.iter_intervals()
    fig = plt.figure(figsize=(8, 5))
    axes = fig.subplots(len(intervals), 3, sharex=True, sharey=True)

    for i, interval in enumerate(intervals):
        patch = tools.read_aux_csv(f"patch_method_trend_{interval.filename_label}_slope.csv")
        area_col = f"area_{interval.short}"

        for j, (col, name) in enumerate([(f"slope_{interval.short}_baseline_err", r"$\sigma_{baseline}$ " + interval.display_label), (f"slope_{interval.short}_excess_err", r"$\sigma_{excess}$ " + interval.display_label), (f"slope_{interval.short}_err", f"Total {interval.display_label}")]):
            ax: plt.Axes = axes[i, j]
            ax.set_title(name)
            ax.scatter(patch["exact_areas"] / 1e6, patch["nmad"] * 2, marker="x", c="k", zorder=2, label="Patch method")
            surging = outlines.query(f"surging_{interval.short}")
            nonsurging = outlines.query(f"~surging_{interval.short}")
            ax.scatter(nonsurging[area_col] / 1e6, nonsurging[col], alpha=0.4, edgecolor="none", s=20, label="Nonsurging", rasterized=True)
            ax.scatter(surging[area_col] / 1e6, surging[col], alpha=0.4, edgecolor="none", s=20, label="Surging")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlim(0.03, 1500)
            ax.set_ylim(5e-3, 1.7)
            if j == 0 and i == 1:
                ax.set_ylabel("Integrated uncertainty (m)")
            if j == 0 and i == (len(intervals) - 1):
                ax.legend(fontsize=8)
            if i == (len(intervals) - 1):
                ax.set_xlabel("Area (km²)")

    fig.tight_layout(h_pad=0.1)
    out_path = FIGURE_DIR / "patch_method_vs_vgm.svg"
    fig.savefig(out_path, dpi=300)
    svg_png_to_jpg(out_path)

    if show:
        plt.show()

    plt.close()


def plot_baseline_err_variogram(show: bool = True):
    variogram = statistics.read_empirical_variogram()
    params = statistics.read_variogram_model()
    vgm_model = statistics.make_variogram_model(params)

    variogram["bin_width"] = np.r_[[0], np.diff(variogram.index)]
    fig = plt.figure(figsize=(8.3 * 0.81, 5 * 0.81))
    axes = fig.subplots(2, 1, sharex=True, height_ratios=[0.3, 0.7])
    axes[0].bar(variogram.index, variogram["count"], width=variogram["bin_width"] * 1.1, align="edge", color="gray")
    axes[1].plot(variogram.index, vgm_model(variogram.index), label="Variogram model", color="black")
    axes[1].errorbar(variogram.index, variogram["exp"], yerr=variogram["err_exp"], fmt="x", label="Empirical variogram", color="royalblue")

    axes[1].set_ylim(1e-3, np.nanmax(vgm_model(variogram.index)) * 1.25)
    for r in params["range"].values:
        axes[1].vlines(r, *axes[1].get_ylim(), color="gray", linestyles="--")

    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Bin count")
    axes[1].set_ylabel("Variance (m²)")
    axes[1].legend(loc="lower right")
    axes[1].set_xlabel("Spatial lag (m)")

    print(params)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "baseline_err_variogram.svg")

    if show:
        plt.show()

    plt.close()
