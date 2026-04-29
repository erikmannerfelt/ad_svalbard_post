import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import zipfile
import io
import numpy as np
import scipy.interpolate

from pathlib import Path

PROJECT_DIR = Path(__file__).absolute().parent.parent
INPUT_DIR = PROJECT_DIR / "input"
FIGURE_DIR = PROJECT_DIR / "figures"

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
        data = pd.read_csv(io.BytesIO(zip_file.read("binned_terrain_err_trend_2013-2024_slope.csv")), index_col=0)
    data["count"] = data["count"].astype(int)

    for col in ["terr_slope", "terr_curvature", "easting", "northing"]:
        data[col] = data[col].apply(_pandas_str_to_interval)

    data["e_mid"] = pd.IntervalIndex(data["easting"]).mid
    data["n_mid"] = pd.IntervalIndex(data["northing"]).mid
    e_cols = data["e_mid"].dropna().unique()[[0, -1]]
    n_cols = data["n_mid"].dropna().unique()[[0, -1]]

    data = data[
            np.logical_and.reduce(
                (
                    data.nd == 4,
                    np.isfinite(pd.IntervalIndex(data["terr_slope"]).mid),
                    np.isfinite(pd.IntervalIndex(data["terr_curvature"]).mid),
                    np.isin(data["e_mid"], e_cols),
                    np.isin(data["n_mid"], n_cols),
                )
            )
        ]

    xcol = "terr_slope"
    ycol = "terr_curvature"
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(*np.nanpercentile((data["nmad"].values), [1, 99])),
        cmap="viridis",
    )
    edge_to_percentile = {}
    percentile_to_edge = {}
    ticks = {}
    tick_labels = {}
    for col in [xcol, ycol]:
        edges = np.unique(np.r_[
            pd.IntervalIndex(data[col]).right,
            pd.IntervalIndex(data[col]).left
        ])
        percentiles = np.linspace(0, 100, edges.size)
        edge_to_percentile[col] = scipy.interpolate.interp1d(edges, percentiles) 
        percentile_to_edge[col] = scipy.interpolate.interp1d(percentiles, edges) 

        ticks[col] = percentiles
        tick_labels[col] = [f"({edge:.1f})" for edge in edges]

    fig = plt.figure()
    axes = fig.subplots(e_cols.shape[0], n_cols.shape[0])

    for i, (_, sub0) in enumerate(data.groupby("e_mid")):
        for j, (_, subset) in enumerate(sub0.groupby("n_mid")):
            ax: plt.Axes = axes[i, j]


            for _, row in subset.iterrows():
                ax.add_patch(plt.Rectangle(
                    (edge_to_percentile[xcol](row[xcol].left), edge_to_percentile[ycol](row[ycol].left)),
                    width=edge_to_percentile[xcol](row[xcol].right) - edge_to_percentile[xcol](row[xcol].left),
                    height=edge_to_percentile[ycol](row[ycol].right) - edge_to_percentile[ycol](row[ycol].left),
                    facecolor=sm.to_rgba((row["nmad"])),
                    
                    edgecolor="#777",
                    linewidth=1,
                ))


            for col, axis in [(xcol, ax.xaxis), (ycol, ax.yaxis)]:
                axis.set_ticks(ticks[col], tick_labels[col], fontsize=8)
                axis._set_lim(0, 100, auto=False)

    plt.show()
