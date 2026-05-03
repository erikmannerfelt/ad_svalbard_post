from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import FIGURE_DIR
from .. import statistics
from .common import DHDT_SM


def plot_hypsometric_profiles(show: bool = True):
    data = statistics.compute_hypsometric_profiles()
    zone_meta = data["zone_meta"].drop(index="K", errors="ignore")
    per_zone = data["per_zone"]
    intervals = data["intervals"]

    fig = plt.figure(figsize=(8, 4))
    ncols = int(np.ceil(len(zone_meta) / 2))
    axes = np.atleast_1d(fig.subplots(2, ncols, sharex=True, sharey=True)).ravel()

    for col, (label, zone) in enumerate(zone_meta.iterrows()):
        if pd.isna(zone["zone_name"]):
            continue
        axis: plt.Axes = axes[col]
        axis.set_title(str(zone["zone_name"]))

        for interval in intervals:
            d = per_zone[interval.interval.short][label]
            if d.empty:
                continue

            color = DHDT_SM.to_rgba(1.0 if interval.interval.short == "13_18" else -1.0)
            axis.errorbar(d.index, d[f"{interval.interval.short}_med"], d[f"{interval.interval.short}_nmad"], color=color, alpha=0.5, fmt="none")

            axis.scatter(d.index, d[f"{interval.interval.short}_med"], marker="s", edgecolor="#777", color=color, label=interval.interval.display_label)

        xlim = axis.get_xlim()
        axis.hlines(0, *xlim, color="#777", linestyles="--", alpha=0.5)
        axis.set_xlim(xlim)

        if col in [0, 4]:
            axis.set_ylabel("dH / dt (m a$^{-1}$)")
        if col >= ncols:
            axis.set_xlabel("Elevation (m a.s.l.)")
        if (col + 1) == len(zone_meta):
            axis.legend(loc="lower right")

    for axis in axes[len(zone_meta):]:
        axis.set_visible(False)

    plt.ylim(-3, 0.8)
    plt.subplots_adjust(top=0.937, bottom=0.121, left=0.073, right=0.991, hspace=0.219, wspace=0.061)

    plt.savefig(FIGURE_DIR / "perzone_hypsometric_signal.svg")

    if show:
        plt.show()
