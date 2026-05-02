import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.ticker
import zipfile
import io
import numpy as np
import scipy.interpolate
import geopandas as gpd

from pathlib import Path

PROJECT_DIR = Path(__file__).absolute().parent.parent
INPUT_DIR = PROJECT_DIR / "input"
FIGURE_DIR = PROJECT_DIR / "figures"
CACHE_DIR = PROJECT_DIR / "cache/"

# Function to convert IntervalIndex written to str in csv back to pd.Interval
# from: https://github.com/pandas-dev/pandas/issues/28210
def _pandas_str_to_interval(istr: str) -> float | pd.Interval:
    if isinstance(istr, float):
        return np.nan
    else:
        c_left = istr[0] == "["
        c_right = istr[-1] == "]"
        closed = {(True, False): "left", (False, True): "right", (True, True): "both", (False, False): "neither"}[
            c_left, c_right
        ]
        left, right = map(float, istr[1:-1].split(","))
        try:
            return pd.Interval(left, right, closed)
        except Exception:
            return np.nan

def plot_terrain_err():
    with zipfile.ZipFile(INPUT_DIR / "aux_files.zip") as zip_file:
        data = pd.read_csv(io.BytesIO(zip_file.read("binned_terrain_err_trend_2019-2024_slope.csv")), index_col=0)
    data["count"] = data["count"].astype(int)
    xcol = "terr_slope"
    ycol = "terr_curvature"

    for col in ["terr_slope", "terr_curvature", "easting", "northing"]:
        data[col] = data[col].apply(_pandas_str_to_interval)

    # Merge the four east/north bins into a 2x2 grid for plotting.
    e_bins = pd.IntervalIndex(sorted(data["easting"].dropna().unique()))
    n_bins = pd.IntervalIndex(sorted(data["northing"].dropna().unique()))
    e_merged = [pd.Interval(e_bins[0].left, e_bins[1].right, closed=e_bins[0].closed), pd.Interval(e_bins[2].left, e_bins[3].right, closed=e_bins[2].closed)]
    n_merged = [pd.Interval(n_bins[0].left, n_bins[1].right, closed=n_bins[0].closed), pd.Interval(n_bins[2].left, n_bins[3].right, closed=n_bins[2].closed)]
    data["easting_merged"] = data["easting"].map({e_bins[0]: e_merged[0], e_bins[1]: e_merged[0], e_bins[2]: e_merged[1], e_bins[3]: e_merged[1]})
    data["northing_merged"] = data["northing"].map({n_bins[0]: n_merged[0], n_bins[1]: n_merged[0], n_bins[2]: n_merged[1], n_bins[3]: n_merged[1]})

    # Keep NMAD as a count-weighted average while summing the sample counts.
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

    sm = matplotlib.cm.ScalarMappable(
        # norm=matplotlib.colors.Normalize(*np.nanpercentile(np.log10(np.clip(data["nmad"].values, a_min=1e-3, a_max=np.inf)), [1, 99])),
        norm=matplotlib.colors.Normalize(-1.3, -0.6),
        cmap="cividis",
    )
    edge_to_percentile = {}
    ticks = {}
    tick_labels = {}
    for col in [xcol, ycol]:
        edges = np.unique(np.r_[
            pd.IntervalIndex(data[col]).right,
            pd.IntervalIndex(data[col]).left
        ])
        percentiles = np.linspace(0, 100, edges.size)
        edge_to_percentile[col] = scipy.interpolate.interp1d(edges, percentiles) 

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
    # Put northing on rows (top to bottom: north to south) and easting on columns.
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
                ax_top.bar(
                    left + (right - left) / 2,
                    count,
                    width=right - left,
                    align="center",
                    color="#666",
                    edgecolor="#444",
                    linewidth=0.5,
                )

            for interval, count in right_counts.items():
                bottom = edge_to_percentile[ycol](interval.left)
                top = edge_to_percentile[ycol](interval.right)
                ax_right.barh(
                    bottom + (top - bottom) / 2,
                    count,
                    height=top - bottom,
                    align="center",
                    color="#666",
                    edgecolor="#444",
                    linewidth=0.5,
                )

            for _, row in subset.iterrows():
                ax.add_patch(plt.Rectangle(
                    (edge_to_percentile[xcol](row[xcol].left), edge_to_percentile[ycol](row[ycol].left)),
                    width=edge_to_percentile[xcol](row[xcol].right) - edge_to_percentile[xcol](row[xcol].left),
                    height=edge_to_percentile[ycol](row[ycol].right) - edge_to_percentile[ycol](row[ycol].left),
                    facecolor=sm.to_rgba(np.log10(np.clip(row["nmad"], a_min=1e-3, a_max=np.inf))),
                    
                    edgecolor="#777",
                    linewidth=1,
                ))

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
    ticks = np.linspace(sm.norm.vmin, sm.norm.vmax, 3)
    cbar.set_ticks(ticks, labels=[f"{10**tick:.1g}" for tick in ticks])


    fig.subplots_adjust(top=0.995,
        bottom=0.085,
        left=0.09,
        right=0.995,
        hspace=0.045,
        wspace=0.2
    )

    fig.savefig(FIGURE_DIR / "binned_terrain_err.svg")

    plt.show()


def plot_surge_nosurge_bar(show: bool = True):

    outlines = gpd.read_file(CACHE_DIR / "outlines_sampled.arrow")

    key_to_interval = lambda s: "_".join(s.split("_")[-2:])
    val_cols = ["slope_13_18", "slope_19_24"]
    titles = ["2013-2018", "2019-2024"]

    surge_colors = {
        True: {
            "even_color": matplotlib.colors.to_rgba("#c76b2a"),
            "odd_color": matplotlib.colors.to_rgba("#e0a36a"),
        },
        False: {
            "odd_color": matplotlib.colors.to_rgba("#2b8cbe"),
            "even_color": matplotlib.colors.to_rgba("#66c2d7"),
        }
    }

    fig = plt.figure(figsize=(8, 3))
    axes = fig.subplots(1, 2, sharey=True)

    def _prepare_stack(data: pd.DataFrame, vol_col: str) -> pd.DataFrame:
        data = data[["glac_name", vol_col]].copy()
        data["stack_vol"] = -data[vol_col].astype(float)
        data = data.sort_values("stack_vol", ascending=True)
        data["bottom"] = data["stack_vol"].cumsum().shift(fill_value=0.0); data["mid"] = data["bottom"] + data["stack_vol"] / 2
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
            strip = _stack_to_rgba_strip(
                stack_vol,
                np.asarray(surge_colors[surging]["even_color"], dtype=float),
                np.asarray(surge_colors[surging]["odd_color"], dtype=float),
            )
            total_height = float(stack_vol.sum())
            x = float(int(surging))

            extent= (x - 0.38, x + 0.38, 0., -total_height)
            ax.imshow(
                strip,
                extent=extent,
                origin="lower",
                aspect="auto",
                interpolation="bilinear",
            )
            ax.add_patch(
                plt.Rectangle(
                    (extent[0], extent[2]),
                    width=extent[1] - extent[0],
                    height=extent[3] - extent[2], 
                    facecolor="none",
                    edgecolor="#777",
                    linewidth=1,
                )
            )

            for _, row in data.loc[data["stack_vol"] > 1].iterrows():
                ax.annotate(
                    row["glac_name"],
                    (x, -row["mid"]),
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        ax.xaxis.set_ticks([0, 1], ["Nonsurging", "Surging"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-23, 0)
        ax.axhline(0, color="#444", linewidth=0.6)

        if i == 0:
            ax.set_ylabel("Volume change rate (km$^3$ a$^{-1}$)")
    fig.tight_layout()

    fig.savefig(FIGURE_DIR / "surge_nosurge_bar.svg", dpi=300)

    

    if show:
        plt.show()


def plot_patch_method_vs_vgm():

    intervals = [
        ("13_18", "2013-2018"),
        ("19_24", "2019-2024"),
        ("13_24", "2013-2024"),
    ]
    short, long = intervals[0]
    rgi = gpd.read_feather("cache/outlines_sampled.arrow")

    fig = plt.figure(figsize=(8, 5))
    axes = fig.subplots(len(intervals), 3, sharex=True, sharey=True)
    for i, (short, long) in enumerate(intervals):
        with zipfile.ZipFile(INPUT_DIR / "aux_files.zip") as zip_file:
            patch = pd.read_csv(io.BytesIO(zip_file.read(f"patch_method_trend_{long}_slope.csv")))
        for j, (col, name) in enumerate([(f"slope_{short}_spatial_err", r"$\sigma_{baseline}$ " + long),(f"slope_{short}_temporal_err", r"$\sigma_{excess}$ " + long),(f"slope_{short}_err", f"Total {long}")]):
            ax: plt.Axes = axes[i, j]

            ax.set_title(name)
            ax.scatter(patch["exact_areas"] / 1e6, patch["nmad"] * 2, marker="x", c="k", zorder=2, label="Patch method")
            surging = rgi.query(f"surging_{short}")
            nonsurging = rgi.query(f"~surging_{short}")
            ax.scatter(nonsurging["area_km2"], nonsurging[col], alpha=0.4, edgecolor="none",s=20, label="Nonsurging")
            ax.scatter(surging["area_km2"], surging[col], alpha=0.4, edgecolor="none",s=20, label="Surging")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

            ax.set_xlim(0.03, 1500)
            ax.set_ylim(5e-3, 1.7)
            if j == 0:
                if i == 1:
                    ax.set_ylabel("Uncertainty (m)")
                if i == (len(intervals) - 1):
                    ax.legend(fontsize=8)

            if i == (len(intervals) - 1):
                ax.set_xlabel("Area (km²)")
    fig.tight_layout(h_pad=0.1)
    fig.savefig("figures/patch_method_vs_vgm.svg")
    plt.show()

    
