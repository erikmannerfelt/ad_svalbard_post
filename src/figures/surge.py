from __future__ import annotations

import matplotlib.colors
import matplotlib.patches
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd

from ..config import CACHE_DIR, FIGURE_DIR
from ..outlines import refine_glacier_zones
from ..tools import key_to_interval
from .. import tools


def _plot_surging_vs_nonsurging_volume_violin(outlines: gpd.GeoDataFrame, show: bool = True):
    fig = plt.figure(figsize=(4, 3))

    interval = "13_24"
    for issurging, items in outlines.groupby(f"surging_{interval}"):
        plt.violinplot([np.log10(np.clip(-items[f"slope_{interval}_vol"], a_min=1e-6, a_max=np.inf))], positions=[float(issurging)])

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
    interval = "13_24"

    for i, (issurging, items) in enumerate(outlines.groupby(f"surging_{interval}")):
        axis = axes[i]
        vals = items[f"slope_{interval}"].dropna()
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


def plot_postsurge_anomaly_by_time(show: bool = True):
    interval = "13_24"
    outlines = gpd.read_feather(CACHE_DIR / "outlines_sampled.arrow")
    outlines = outlines.loc[outlines[f"slope_{interval}"].notna() & outlines["zone_label"].notna() & outlines[f"area_{interval}"].notna()].copy()

    def _parse_years(value):
        if pd.isna(value):
            return []
        for part in str(value).split(";"):
            token = part.strip()
            if not token:
                continue
            low = token.lower()
            if low in {"n/a", "not observed", "ongoing"}:
                continue
            try:
                yield int(token)
            except ValueError:
                continue

    def _prepare_postsurge_frame(frame: gpd.GeoDataFrame):
        frame = frame.copy()
        frame["term_years"] = frame["S_Term"].apply(lambda value: list(_parse_years(value)))
        frame["onset_years"] = frame["S_Onset"].apply(lambda value: list(_parse_years(value)))
        frame["never_surged"] = frame["term_years"].str.len().eq(0) & frame["onset_years"].str.len().eq(0)
        frame["last_term_pre2013"] = frame["term_years"].apply(lambda years: max([year for year in years if year < 2013], default=np.nan))
        frame["years_since_last_surge_2013"] = 2013 - frame["last_term_pre2013"]
        frame["log_area"] = np.log10(frame[f"area_{interval}"])

        large = frame.loc[frame[f"area_{interval}"] > 1e6].copy()
        surged = large.loc[large["last_term_pre2013"].notna() & large["years_since_last_surge_2013"].notna() & ~large[f"surging_{interval}"].fillna(False)].copy()
        never = large.loc[large["never_surged"]].copy()

        if surged.empty or never.empty:
            raise RuntimeError("Not enough large-glacier surge or control data to build post-surge anomaly figure")

        never_zone_medians = never.groupby("zone_label", observed=False)[f"slope_{interval}"].median()
        surged = surged.loc[surged["zone_label"].isin(never_zone_medians.index)].copy()
        never = never.loc[never["zone_label"].isin(never_zone_medians.index)].copy()

        if surged.empty or never.empty:
            raise RuntimeError("No overlapping zones between surged and never-surged large glaciers")

        surged["zone_control_median"] = surged["zone_label"].map(never_zone_medians)
        never["zone_control_median"] = never["zone_label"].map(never_zone_medians)
        surged["anomaly"] = surged[f"slope_{interval}"] - surged["zone_control_median"]
        never["anomaly"] = never[f"slope_{interval}"] - never["zone_control_median"]

        bins = [0, 20, 40, 60, 80, 100, 120]
        labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100+"]
        surged["age_bin"] = pd.cut(surged["years_since_last_surge_2013"], bins=bins, labels=labels, include_lowest=True, right=True)
        surged = surged.loc[surged["age_bin"].notna()].copy()

        if surged.empty:
            raise RuntimeError("No large pre-2013 surge-terminated glaciers fell into the requested age bins")

        return surged, never, labels

    def _bootstrap_median_ci(values: pd.Series, n_iter: int = 10000, seed: int = 0):
        values = values.dropna().to_numpy(dtype=float)
        if values.size == 0:
            return np.nan, np.nan, np.nan
        rng = np.random.default_rng(seed)
        samples = np.empty(n_iter, dtype=float)
        for i in range(n_iter):
            samples[i] = float(np.median(rng.choice(values, size=values.size, replace=True)))
        return float(np.quantile(samples, 0.025)), float(np.median(samples)), float(np.quantile(samples, 0.975))

    def _sample_null_medians(zone_control_map: dict[str, np.ndarray], zone_counts: dict[str, int], n_iter: int = 10000, seed: int = 0):
        rng = np.random.default_rng(seed)
        samples = np.full(n_iter, np.nan, dtype=float)
        zones = [zone for zone, count in zone_counts.items() if count > 0 and zone in zone_control_map and zone_control_map[zone].size > 0]
        if not zones:
            return samples
        for i in range(n_iter):
            vals = []
            for zone in zones:
                pool = zone_control_map[zone]
                count = zone_counts[zone]
                vals.append(rng.choice(pool, size=count, replace=pool.size < count))
            samples[i] = float(np.median(np.concatenate(vals)))
        return samples

    surged, never, labels = _prepare_postsurge_frame(outlines)

    glacier_zones = refine_glacier_zones()
    zone_labels = sorted(surged["zone_label"].dropna().unique().tolist())
    zone_colors = {label: glacier_zones.loc[label, "color"] for label in zone_labels if label in glacier_zones.index}

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.subplots_adjust(left=0.12, bottom=0.16, right=0.98, top=0.93)

    x_positions = np.array([10, 30, 50, 70, 90, 110], dtype=float)
    box_edges = [0, 20, 40, 60, 80, 100, 120]

    control_zone_map = {zone: grp["anomaly"].dropna().to_numpy(dtype=float) for zone, grp in never.groupby("zone_label", observed=False)}

    legend_handles = []
    seen_legend = set()
    for x, label, left, right in zip(x_positions, labels, box_edges[:-1], box_edges[1:], strict=False):
        subset = surged.loc[surged["age_bin"] == label].copy()
        if subset.empty:
            continue

        zone_counts = subset["zone_label"].value_counts().to_dict()
        null_samples = _sample_null_medians(control_zone_map, zone_counts, n_iter=8000, seed=42 + int(x))
        null_samples = null_samples[np.isfinite(null_samples)]
        if null_samples.size:
            null_q16, null_q50, null_q84 = np.quantile(null_samples, [0.16, 0.5, 0.84])
            null_q025, _, null_q975 = np.quantile(null_samples, [0.025, 0.5, 0.975])
            ax.add_patch(matplotlib.patches.Rectangle((left, null_q025), right - left, null_q975 - null_q025, facecolor="#d9d9d9", edgecolor="none", alpha=0.45, zorder=0))
            ax.add_patch(matplotlib.patches.Rectangle((left, null_q16), right - left, null_q84 - null_q16, facecolor="#bdbdbd", edgecolor="none", alpha=0.7, zorder=1))
            ax.hlines(null_q50, left, right, color="#666", linewidth=0.8, zorder=2)

        obs_low, obs_med, obs_high = _bootstrap_median_ci(subset["anomaly"], n_iter=6000, seed=100 + int(x))
        ax.errorbar(x, obs_med, yerr=[[obs_med - obs_low], [obs_high - obs_med]], fmt="o", color="#333", ecolor="#333", elinewidth=1.2, capsize=3, markersize=5.5, zorder=5)

        for j, (_, row) in enumerate(subset.iterrows()):
            color = zone_colors.get(row["zone_label"], (0.35, 0.35, 0.35, 1.0))
            point_x = min(float(row["years_since_last_surge_2013"]), 120.0)
            ax.scatter(point_x, row["anomaly"], s=9, color=color, alpha=0.75, edgecolors="white", linewidths=0.2, zorder=4)
            if row["zone_label"] not in seen_legend:
                seen_legend.add(row["zone_label"])
                legend_handles.append(matplotlib.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.3, markersize=5, label=row["zone_label"]))

        ax.annotate(f"n={len(subset)}", (x, 0.02), xycoords=(ax.transData, ax.transAxes), ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="#444", linewidth=0.8, zorder=3)
    ax.set_xticks(x_positions, labels)
    ax.set_xlabel("Years since last observed surge termination")
    ax.set_ylabel("Zonal dH dt$^{-1}$ anomaly (m a$^{-1}$)")

    if legend_handles:
        ax.legend(handles=legend_handles, title="Zone", fontsize=8, title_fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    ax.set_xlim(0, 120)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "postsurge_anomaly_by_time.svg", dpi=300)

    if show:
        plt.show()

    plt.close(fig)
