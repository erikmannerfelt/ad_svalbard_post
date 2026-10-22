"""Regenerate the statistics and figures of the manuscript.

    python main.py             # statistics, then figures
    python main.py stats       # tables/statistics.{json,tex}, tables/comparisons.tex, ...
    python main.py figures     # figures/
"""

from __future__ import annotations

import argparse

from src import figures, statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "step",
        nargs="?",
        choices=("all", "stats", "figures"),
        default="all",
        help="what to regenerate (default: all)",
    )
    args = parser.parse_args()

    if args.step in ("all", "stats"):
        statistics.get_statistics()
    if args.step in ("all", "figures"):
        figures.make_all_figures(show=False)


if __name__ == "__main__":
    main()
