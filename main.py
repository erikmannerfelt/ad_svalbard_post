from __future__ import annotations

from src import figures


def make_figs(show: bool = False):
    figures.make_all_figures(show=show)


if __name__ == "__main__":
    make_figs()
