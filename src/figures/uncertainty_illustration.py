"""Conceptual SI figure of the two uncertainty components, sigma_baseline and sigma_excess.

The figure is an illustration, not a calculation: the point scatter, the fan widths and
the bar shares are set directly. Fans and bars agree with each other through
sigma_fit^2 = sigma_baseline^2 + sigma_excess^2, and the fans pivot on the mean date,
since both sigmas are rate uncertainties.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from ..config import FIGURE_DIR

BASELINE_COLOR: Final = "#3f6fb0"
EXCESS_COLOR: Final = "#c8553d"
FIT_COLOR: Final = "#2b2b2b"

#: Fan half-widths, as rates in elevation per time step
SIGMA_BASELINE: Final = 0.25
#: Baseline share 1 / 1.7^2, about 35%
SIGMA_FIT_NONLINEAR: Final = 1.7 * SIGMA_BASELINE
#: Baseline share of a few percent
SIGMA_FIT_SURGE: Final = 2.0

N_DEMS: Final = 12


def _stable(t: np.ndarray) -> np.ndarray:
    return np.zeros_like(t)


def _linear(t: np.ndarray) -> np.ndarray:
    return -1.5 * t


def _nonlinear(t: np.ndarray) -> np.ndarray:
    # Thickening that peaks near the middle and turns to thinning, as in main Fig. 2
    return -0.3 * t - 0.16 * t**2


def _surge(t: np.ndarray) -> np.ndarray:
    # Rapid drawdown of a reservoir area within a year or two
    return -0.6 * t - 42.0 / (1 + np.exp(-(t - 1.0) * 4))


def _scatter(t: np.ndarray) -> np.ndarray:
    """Point scatter without long-range structure, so the linear case reads as linear."""
    # The seed was picked for residuals that alternate around the line
    noise = np.random.default_rng(549).normal(0, 1, t.size)
    noise -= np.polyval(np.polyfit(t, noise, 1), t)
    return 0.3 * noise / noise.std()


def _fan(
    axis: plt.Axes,
    t: np.ndarray,
    center: float,
    slope: float,
    sigma: float,
    color: str,
    alpha: float,
) -> None:
    """Trend lines within +-sigma of ``slope``, pivoting on the mean date."""
    fan_t = np.linspace(t.min() - 0.4, t.max() + 0.4, 50)
    axis.fill_between(
        fan_t,
        center + (slope - sigma) * fan_t,
        center + (slope + sigma) * fan_t,
        color=color,
        alpha=alpha,
        linewidth=0,
    )


def _plot_timeseries_panels(
    fig: plt.Figure, gridspec: matplotlib.gridspec.GridSpec
) -> list[plt.Axes]:
    t = np.arange(N_DEMS, dtype=float)
    t -= t.mean()
    fine_t = np.linspace(t.min(), t.max(), 200)
    scatter = _scatter(t)

    panels: tuple[tuple[str, Callable[[np.ndarray], np.ndarray], float | None], ...] = (
        ("Stable terrain", _stable, None),
        ("Glacier: linear", _linear, None),
        ("Glacier: nonlinear", _nonlinear, SIGMA_FIT_NONLINEAR),
        ("Glacier: surge", _surge, SIGMA_FIT_SURGE),
    )
    axes = []
    for i, (title, history, sigma_fit) in enumerate(panels):
        axis = fig.add_subplot(gridspec[i])
        elevation = history(t) + scatter
        slope, intercept = np.polyfit(t, elevation, 1)
        if sigma_fit is not None:
            _fan(axis, t, intercept, slope, sigma_fit, EXCESS_COLOR, 0.2)
        _fan(axis, t, intercept, slope, SIGMA_BASELINE, BASELINE_COLOR, 0.4)
        if history is not _stable:
            axis.plot(
                fine_t, history(fine_t), color="#888", linewidth=0.8, linestyle=":"
            )
        axis.plot(t, np.polyval([slope, intercept], t), color=FIT_COLOR, linewidth=1)
        axis.scatter(t, elevation, s=8, color="k", zorder=3)

        if history is _stable:
            axis.set_ylim(-3.0, 3.0)
            axis.text(
                0.02,
                0.95,
                r"$\pm\sigma_{baseline}$",
                transform=axis.transAxes,
                color=BASELINE_COLOR,
                fontsize=8,
                va="top",
            )
        elif history is _surge:
            # Limits follow the data, so the fans may be cut at the ends
            pad = 0.12 * np.ptp(elevation)
            axis.set_ylim(elevation.min() - pad, elevation.max() + pad)
        else:
            # The linear and nonlinear panels share one elevation span, so their fans
            # can be compared directly
            linear = _linear(t) + scatter
            half_span = 0.62 * np.ptp(linear)
            middle = 0.5 * (elevation.min() + elevation.max())
            axis.set_ylim(middle - half_span, middle + half_span)
        if history is _nonlinear:
            axis.text(
                0.98,
                0.95,
                r"$\pm\sigma_{fit}$",
                transform=axis.transAxes,
                color=EXCESS_COLOR,
                fontsize=8,
                va="top",
                ha="right",
            )

        # The surge spans a much larger elevation range than the other panels
        tick_labels = ["Lower", "Higher"] if history is _surge else ["Low", "High"]
        axis.set_yticks(axis.get_ylim(), tick_labels)
        axis.set_xticks([])
        axis.tick_params(labelsize=7, length=0)
        axis.set_title(title, fontsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axes.append(axis)

    axes[-1].set_xlabel("Time", fontsize=8)
    return axes


def _plot_variance_shares(axis: plt.Axes) -> None:
    baseline_share = 100 * np.array(
        [
            1.0,
            SIGMA_BASELINE**2 / SIGMA_FIT_NONLINEAR**2,
            SIGMA_BASELINE**2 / SIGMA_FIT_SURGE**2,
        ]
    )
    positions = np.arange(baseline_share.size)
    axis.bar(positions, baseline_share, color=BASELINE_COLOR, width=0.6)
    axis.bar(
        positions,
        100 - baseline_share,
        bottom=baseline_share,
        color=EXCESS_COLOR,
        width=0.6,
    )
    axis.text(
        0,
        50,
        r"$\sigma^2_{baseline}$",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
    )
    axis.text(
        1,
        65,
        r"$\sigma^2_{excess}$",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
    )
    axis.set_xticks(positions, ["Linear", "Nonlinear", "Surge"])
    axis.set_ylim(0, 100)
    axis.set_yticks([])
    axis.set_ylabel(r"Share of $\sigma^2_{fit}$", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.set_title(
        r"$\sigma^2_{fit} = \sigma^2_{baseline} + \sigma^2_{excess}$", fontsize=8
    )
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _plot_aggregation(axis: plt.Axes) -> None:
    area = np.logspace(-1, 3, 80)
    baseline = 1 / np.sqrt(1 + area / 0.3)
    excess = np.full_like(area, 0.12)
    axis.plot(
        area,
        baseline,
        color=BASELINE_COLOR,
        linewidth=1.8,
        label=r"$\sigma_{baseline}$ / $\sqrt{n_{eff}}$",
    )
    axis.plot(
        area,
        excess,
        color=EXCESS_COLOR,
        linewidth=1.8,
        label=r"$\sigma_{excess}$, fully systematic",
    )
    axis.plot(
        area,
        np.hypot(baseline, excess),
        color=FIT_COLOR,
        linewidth=1,
        linestyle="--",
        label="Total",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_yticks([])
    axis.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_xticks([area[0], area[-1]], ["Small", "Large"])
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.tick_params(axis="x", length=0, labelsize=7)
    axis.set_xlabel("Glacier area", fontsize=8)
    axis.set_ylabel("Glacier-wide uncertainty", fontsize=8)
    axis.legend(fontsize=7, frameon=False, loc="lower left")
    axis.set_title("Aggregation over a glacier", fontsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot_uncertainty_illustration(show: bool = True) -> None:
    """Draw the conceptual sigma_baseline / sigma_excess figure of the SI."""
    fig = plt.figure(figsize=(7.2, 7.4))
    left = fig.add_gridspec(
        4, 1, left=0.09, right=0.47, top=0.96, bottom=0.06, hspace=0.38
    )
    right = fig.add_gridspec(
        2, 1, left=0.6, right=0.99, top=0.96, bottom=0.06, hspace=0.4
    )

    timeseries_axes = _plot_timeseries_panels(fig, left)
    fig.text(0.015, 0.51, "Elevation", rotation=90, va="center", fontsize=8)
    share_axis = fig.add_subplot(right[0])
    _plot_variance_shares(share_axis)
    aggregation_axis = fig.add_subplot(right[1])
    _plot_aggregation(aggregation_axis)

    for label, axis in zip(
        "abcdef", [*timeseries_axes, share_axis, aggregation_axis], strict=True
    ):
        axis.text(
            0.0,
            1.03,
            f"{label})",
            transform=axis.transAxes,
            fontsize=9,
            ha="right",
            va="bottom",
        )

    fig.savefig(FIGURE_DIR / "uncertainty_illustration.svg")
    if show:
        plt.show()
    plt.close(fig)
