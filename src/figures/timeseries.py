from __future__ import annotations

from typing import Final

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

YEARS = list(range(2013, 2025))
FALLBACK_2013_REFERENCE_X = 2013.49
FALLBACK_2019_REFERENCE_X = 2019.60

# Annual DEM count, saturating at 10 DEMs per year
COUNT_CMAP: Final = plt.get_cmap("viridis")
COUNT_NORM: Final = matplotlib.colors.Normalize(vmin=1, vmax=10)


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


def draw_panel(axis: plt.Axes, row, cmap, norm, marker_size: float = 20) -> None:
    """Draw one point's annual medians, NMAD error bars and the three fitted trends."""
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

    axis.errorbar(xs[valid], ys[valid], yerr=yerr[valid], fmt="none", ecolor="0.2", capsize=2, lw=0.9, alpha=0.8)
    axis.scatter(xs[valid], ys[valid], c=cmap(norm(np.clip(counts[valid], 1, 10))), s=marker_size, edgecolors="0.15", linewidths=0.25, zorder=3)

    ref_2013_x = get_reference_x(row, 2013, FALLBACK_2013_REFERENCE_X)
    ref_2019_x = get_reference_x(row, 2019, FALLBACK_2019_REFERENCE_X)

    plot_trend(axis, parse_float(row.get("2013-2024_slope")), parse_float(row.get("2013-2024_intercept")), ref_2013_x, 2013.0, 2025.0, color="0.65", alpha=0.9, lw=2.0)
    plot_trend(axis, parse_float(row.get("2013-2018_slope")), parse_float(row.get("2013-2018_intercept")), ref_2013_x, 2013.0, 2019.0, color="black")
    plot_trend(axis, parse_float(row.get("2019-2024_slope")), parse_float(row.get("2019-2024_intercept")), ref_2019_x, 2019.0, 2025.0, color="black")

    axis.set_ylabel("Elevation (m a.s.l.)", fontsize=7, labelpad=2)

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
    axis.tick_params(axis="both", which="major", labelsize=6)

    axis.set_xlim(2012.5, 2024.9)
    axis.xaxis.set_major_locator(MultipleLocator(4))


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="k", marker="o", lw=1, ms=4, label="Annual medians ± NMAD"),
        Line2D([0], [0], color="black", lw=1.6, label="2013–2018 / 2019–2024 trend"),
        Line2D([0], [0], color="0.65", lw=2.0, label="2013–2024 trend"),
    ]
