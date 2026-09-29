from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


WORKDIR = Path(__file__).resolve().parent
CSV_PATH = WORKDIR / "point_timeseries_trends.csv"
SVG_PATH = WORKDIR / "point_timeseries_trends.svg"

YEARS = list(range(2013, 2025))
TREND_INTERVALS = ["2013-2018", "2019-2024", "2013-2024"]

QUALITY_LABELS = {
    "q95": "q95",
    "q75": "q75",
    "bias": "bias",
    "vertcorr": "vertcorr",
}

FALLBACK_2013_REFERENCE_X = 2013.49
FALLBACK_2019_REFERENCE_X = 2019.60


def read_csv_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value):
    if value in (None, ""):
        return np.nan
    return float(value)


def get_reference_x(row, year, fallback):
    doy = parse_float(row.get(f"{year}_doy"))
    if np.isfinite(doy):
        return year + (doy - 1.0) / 365.25
    return float(fallback)


def plot_trend(ax, slope, intercept, ref_x, start_year, end_year, color, alpha=1.0, lw=1.6):
    xs = np.linspace(start_year, end_year, 200)
    ys = intercept + slope * (xs - ref_x)
    ax.plot(xs, ys, color=color, alpha=alpha, lw=lw)


def choose_major_step(ymin, ymax):
    span = max(ymax - ymin, 1.0)
    for step in (5.0, 10.0, 20.0, 25.0, 50.0, 100.0):
        if span / step <= 6:
            return step
    return 100.0


def count_to_color(count):
    return min(max(count, 1.0), 10.0)


def main():
    rows = read_csv_rows(CSV_PATH)
    ncols = 2
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 5), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=1, vmax=10)
    last_scatter = None

    for idx, (ax, row) in enumerate(zip(axes, rows)):
        label = row["label"]
        quality = QUALITY_LABELS.get(row.get("quality_mode", ""), row.get("quality_mode", ""))

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

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        yerr = np.array(yerr, dtype=float)
        counts = np.array([parse_float(row.get(f"{year}_count")) for year in YEARS], dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)

        ax.errorbar(xs[valid], ys[valid], yerr=yerr[valid], fmt="none", ecolor="0.2", capsize=2, lw=0.9, alpha=0.8)
        colors_for_points = cmap(norm(np.clip(counts[valid], 1, 10)))
        last_scatter = ax.scatter(xs[valid], ys[valid], c=colors_for_points, s=20, edgecolors="0.15", linewidths=0.25, zorder=3)

        ref_2013_x = get_reference_x(row, 2013, FALLBACK_2013_REFERENCE_X)
        ref_2019_x = get_reference_x(row, 2019, FALLBACK_2019_REFERENCE_X)

        plot_trend(ax, parse_float(row.get("2013-2024_slope")), parse_float(row.get("2013-2024_intercept")), ref_2013_x, 2013.0, 2025.0, color="0.65", alpha=0.9, lw=2.0)
        plot_trend(ax, parse_float(row.get("2013-2018_slope")), parse_float(row.get("2013-2018_intercept")), ref_2013_x, 2013.0, 2019.0, color="black")
        plot_trend(ax, parse_float(row.get("2019-2024_slope")), parse_float(row.get("2019-2024_intercept")), ref_2019_x, 2019.0, 2025.0, color="black")

        ax.set_title(f"{label} | quality={quality}", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Elevation (m a.sl.)")
        else:
            ax.set_ylabel("")

        y_min = np.nanmin(ys)
        y_max = np.nanmax(ys)
        if np.isfinite(y_min) and np.isfinite(y_max):
            pad = max(5.0, (y_max - y_min) * 0.15)
            ax.set_ylim(y_min - pad, y_max + pad)
        major_step = choose_major_step(*ax.get_ylim())
        ax.yaxis.set_major_locator(MultipleLocator(major_step))
        ax.yaxis.set_minor_locator(MultipleLocator(2.5))
        ax.tick_params(axis="y", which="minor", left=True, length=2.5, labelleft=False, color="0.5")
        ax.grid(True, which="major", axis="y", alpha=0.35, linewidth=0.95)
        ax.grid(True, which="minor", axis="y", alpha=0.18, linewidth=0.65)
        ax.grid(True, which="major", axis="x", alpha=0.15)

        if label == "Kronebreen terminus retreat":
            cax = inset_axes(ax, width="7%", height="50%", loc="lower left", borderpad=0.8)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation="vertical", extend="max")
            cbar.set_ticks([1,10])
            cbar.set_ticklabels(["1","≥10"])
            cbar.ax.tick_params(labelsize=6, length=2)
            cbar.outline.set_linewidth(0.6)

    for ax in axes[len(rows):]:
        ax.axis("off")

    for ax in axes[-ncols:]:
        ax.set_xlabel("Year")

    legend_handles = [
        Line2D([0], [0], color="k", marker="o", lw=1, ms=4, label="Annual medians ± NMAD"),
        Line2D([0], [0], color="black", lw=1.6, label="2013–2018 / 2019–2024 trend"),
        Line2D([0], [0], color="0.65", lw=2.0, label="2013–2024 trend"),
    ]
    axes[-1].legend(handles=legend_handles, loc="lower left", fontsize=7)

    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved figure to {SVG_PATH}")


if __name__ == "__main__":
    main()
