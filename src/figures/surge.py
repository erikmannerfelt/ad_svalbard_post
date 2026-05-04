from __future__ import annotations

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd

from ..config import CACHE_DIR, FIGURE_DIR
from ..tools import key_to_interval


def _plot_surging_vs_nonsurging_volume_violin(outlines: gpd.GeoDataFrame, show: bool = True):
    fig = plt.figure(figsize=(4, 3))

    for issurging, items in outlines.groupby("surging_13_24"):
        plt.violinplot([np.log10(np.clip(-items["slope_13_24_vol"], a_min=1e-6, a_max=np.inf))], positions=[float(issurging)])

    plt.ylim(10, 3)
    plt.xticks([0, 1], ["Nonsurging", "Surging"])
    yticks = plt.gca().get_yticks()
    plt.yticks(yticks, labels=["10$^{" + str(int(ytick)) + "}$" for ytick in yticks])
    plt.ylabel("Volume change rate (km³ / a)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "surging_vs_nonsurging_vol_violin.svg")

    if show:
        plt.show()

    plt.close()


def _plot_surging_vs_nonsurging_hist(outlines: gpd.GeoDataFrame, show: bool = True):
    fig = plt.figure(figsize=(4, 3))
    axes = fig.subplots(2, 1, sharex=True, sharey=False).ravel().tolist()
    fig.subplots_adjust(left=0.135, bottom=0.165, right=0.995, top=0.98, hspace=0.1)

    for i, (issurging, items) in enumerate(outlines.groupby("surging_13_24")):
        axis = axes[i]
        vals = items["slope_13_24"].dropna()
        axis.hist(vals, bins=np.linspace(-5, 1, 30 if issurging else 100), color="#ff6666" if issurging else "#6699ff")

        if i == 0:
            axis.set_yscale("log")

        axis.set_ylabel("Count")
        if i == 1:
            axis.set_xlabel("Elevation change rate 2013-2024 (m a$^{-1}$)")

        plt.text(0.02, 0.95, ("Surging " if issurging else "Non-surging ") + f"(n={vals.shape[0]})", va="top", transform=axis.transAxes)

    plt.savefig(FIGURE_DIR / "surging_vs_nonsurging_hist.svg")

    if show:
        plt.show()

    plt.close()


def plot_surge_nosurge_bar(show: bool = True):
    outlines = gpd.read_feather(CACHE_DIR / "outlines_sampled.arrow")
    val_cols = ["slope_13_18", "slope_19_24"]
    titles = ["2013-2018", "2019-2024"]
    surge_colors = {
        True: {"even_color": matplotlib.colors.to_rgba("#c76b2a"), "odd_color": matplotlib.colors.to_rgba("#e0a36a")},
        False: {"odd_color": matplotlib.colors.to_rgba("#2b8cbe"), "even_color": matplotlib.colors.to_rgba("#66c2d7")},
    }
    fig = plt.figure(figsize=(8, 3))
    axes = fig.subplots(1, 2, sharey=True)

    def _prepare_stack(data: pd.DataFrame, vol_col: str) -> pd.DataFrame:
        data = data[["glac_name", vol_col]].copy()
        data["stack_vol"] = -data[vol_col].astype(float)
        data = data.sort_values("stack_vol", ascending=True)
        data["bottom"] = data["stack_vol"].cumsum().shift(fill_value=0.0)
        data["mid"] = data["bottom"] + data["stack_vol"] / 2
        return data

    def _stack_to_rgba_strip(heights: np.ndarray, even_color: np.ndarray, odd_color: np.ndarray) -> np.ndarray:
        if heights.size == 0:
            return np.zeros((1, 1, 4), dtype=float)
        bottoms = np.r_[0.0, np.cumsum(heights[:-1])]
        total_height = float(heights.sum())
        n_rows = max(1024, min(8192, int(np.ceil(total_height)) if total_height > 0 else 1024))
        row_edges = np.linspace(0.0, total_height, n_rows + 1)
        row_height = row_edges[1] - row_edges[0]
        colors = np.vstack((even_color, odd_color))
        rgba_sum = np.zeros((n_rows, 4), dtype=float)
        weights = np.zeros(n_rows, dtype=float)
        for idx, (bottom, height) in enumerate(zip(bottoms, heights, strict=False)):
            if height <= 0:
                continue
            top = bottom + height
            color = colors[idx % 2]
            start = np.searchsorted(row_edges, bottom, side="right") - 1
            stop = np.searchsorted(row_edges, top, side="left") - 1
            start = int(np.clip(start, 0, n_rows - 1))
            stop = int(np.clip(stop, 0, n_rows - 1))
            if start == stop:
                overlap = top - bottom
                rgba_sum[start] += overlap * color
                weights[start] += overlap
                continue
            first_overlap = row_edges[start + 1] - bottom
            rgba_sum[start] += first_overlap * color
            weights[start] += first_overlap
            if stop > start + 1:
                rgba_sum[start + 1:stop] += row_height * color
                weights[start + 1:stop] += row_height
            last_overlap = top - row_edges[stop]
            rgba_sum[stop] += last_overlap * color
            weights[stop] += last_overlap
        strip = np.zeros((n_rows, 1, 4), dtype=float)
        valid = weights > 0
        strip[valid, 0, :] = rgba_sum[valid] / weights[valid, None]
        strip[valid, 0, 3] = 1.0
        return strip

    for i, val_col in enumerate(val_cols):
        ax: plt.Axes = axes.ravel()[i]
        ax.set_title(titles[i])
        vol_col = val_col + "_loss"
        outlines[vol_col] = outlines.geometry.area * outlines[val_col] / 1e9
        for surging, data in outlines.groupby(f"surging_{key_to_interval(val_col)}"):
            data = _prepare_stack(data, vol_col)
            stack_vol = data["stack_vol"].to_numpy(dtype=float)
            rgba_strip_args = np.asarray(surge_colors[surging]["even_color"], dtype=float), np.asarray(surge_colors[surging]["odd_color"], dtype=float)
            # Ugly hack that works: if the count is even, flip the colors so it ends with the same color regardless of evenness
            if data.shape[0] % 2 == 0:
                rgba_strip_args = rgba_strip_args[::-1]
            strip = _stack_to_rgba_strip(stack_vol, *rgba_strip_args)
                
            total_height = float(stack_vol.sum())
            x = float(int(surging))
            extent = (x - 0.38, x + 0.38, 0.0, -total_height)
            ax.imshow(strip, extent=extent, origin="lower", aspect="auto", interpolation="bilinear")
            ax.add_patch(plt.Rectangle((extent[0], extent[2]), width=extent[1] - extent[0], height=extent[3] - extent[2], facecolor="none", edgecolor="#777", linewidth=1))
            for _, row in data.loc[data["stack_vol"] > 1].iterrows():
                ax.annotate(row["glac_name"], (x, -row["mid"]), ha="center", va="center", fontsize=9)
        ax.xaxis.set_ticks([0, 1], ["Nonsurging", "Surging"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-24, 0)
        ax.axhline(0, color="#444", linewidth=0.6)
        if i == 0:
            ax.set_ylabel("Volume change rate (km$^3$ a$^{-1}$)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "surge_nosurge_bar.svg", dpi=300)

    if show:
        plt.show()


def plot_surging_vs_nonsurging_figures(show: bool = True):
    outlines = gpd.read_feather(CACHE_DIR / "outlines_sampled.arrow")
    _plot_surging_vs_nonsurging_volume_violin(outlines, show=show)
    _plot_surging_vs_nonsurging_hist(outlines, show=show)
