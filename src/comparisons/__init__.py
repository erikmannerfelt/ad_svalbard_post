"""Comparisons of this study with published Svalbard mass-balance estimates.

One module per published source. Each exposes a ``summary()`` returning the values
the manuscript cites, and :func:`get_comparisons` collects them into the nested
structure that :mod:`src.reporting` turns into ``tables/statistics.tex`` macros.

Source data is downloaded on demand from the URL recorded in each module, so the
comparison is reproducible without committing large files. The one exception is
Hugonnet et al. (2021), whose regional and per-glacier products have no stable
public URL and therefore live in ``input/comparisons/hugonnet2021``.

Densities differ between sources and are the single largest cause of apparent
disagreement, so each module states the convention its source used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from ..config import TABLE_DIR
from . import common, geyman2022, hugonnet2021, morris2020, wang2022, wgms_amce

__all__ = [
    "common",
    "get_comparisons",
    "geyman2022",
    "hugonnet2021",
    "morris2020",
    "wang2022",
    "wgms_amce",
]

STATISTICS_PATH: Final = TABLE_DIR / "statistics.json"


def _load_our_changes(changes: dict[str, object] | None) -> dict[str, object]:
    """Return this study's ``changes`` block, reading it from disk if not supplied.

    Passing it in avoids a write-then-read round trip when this is called from
    :func:`src.statistics.get_statistics`, which has the values in hand already.
    """
    if changes is not None:
        return changes
    if not STATISTICS_PATH.is_file():
        raise FileNotFoundError(
            f"{STATISTICS_PATH} not found; run statistics.get_statistics() first."
        )
    with open(STATISTICS_PATH) as handle:
        return json.load(handle)["changes"]


def get_comparisons(
    changes: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Collect every source's summary into one nested dictionary.

    Each source is attempted independently so that a missing local file or an
    unreachable host degrades that one entry rather than the whole set; the reason
    is recorded in place of the values.

    Parameters
    ----------
    changes
        This study's ``changes`` statistics. Read from ``tables/statistics.json``
        when omitted.
    """
    our_changes = _load_our_changes(changes)
    mid_end = our_changes["slope_mid_end"]["all"]
    ours = {
        "volume_rate_mid_end": common.Estimate(
            float(mid_end["vol_rate"]), float(mid_end["vol_rate_err"])
        ),
        "mass_rate_mid_end": float(mid_end["vol_rate"]) * common.RHO_AW,
    }
    # Keys must be letters only: reporting.to_tex_macros builds LaTeX macro names
    # from them, and TeX control sequences cannot contain digits.
    builders = {
        "morris": lambda: morris2020.summary(our_changes),
        "hugonnet": hugonnet2021.summary,
        "geyman": geyman2022.summary,
        "wgmsamce": wgms_amce.summary,
        "wang": lambda: wang2022.summary(
            ours["volume_rate_mid_end"], ours["mass_rate_mid_end"]
        ),
    }
    collected: dict[str, dict[str, object]] = {}
    for name, builder in builders.items():
        try:
            collected[name] = builder()
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"comparison '{name}' unavailable: {error}")
            collected[name] = {"unavailable": type(error).__name__}
    return collected


#: Rows of the Supporting Information comparison table, in the order they appear.
#: Numbers are pulled from each module so the table cannot drift from the analysis;
#: the comparability note is prose that is not derivable and so is stated here.
#: ``period`` is the span of the source dataset, whereas ``rate_period`` is the window
#: the tabulated rate refers to. The two differ where a module evaluates an annual or
#: multi-period series over our own intervals, so the rate is labeled with its window
#: rather than silently inheriting the dataset span. ``summary``, ``rate_key`` and
#: ``rate_err_key`` name the module summary and its fields; empty means not tabulated.
TABLE_SOURCES: Final[tuple[dict[str, str], ...]] = (
    {
        "citation": r"\citet{morris_spread_2020}",
        "summary": "morris",
        "rate_key": "total_mass_rate",
        "rate_err_key": "total_mass_rate_err",
        "rate_period": "2011--2017",
        "period": "2011--2017",
        "method": "CryoSat-2 SARIn swath altimetry, 1~km$^2$ plane fits",
        "density": r"\textbf{850$\pm$60} (as ours)",
        "note": (
            "Direct precedent for the surge/non-surge partition. Non-surge specific "
            "balance $-0.37$ vs our $-0.39$~m~w.e.~a$^{-1}$; Storisstraumen 22\\% of "
            "their total vs our 19\\%. Their 31\\% counts actively surging \\emph{ice}, "
            "ours counts surge-affected \\emph{glaciers}."
        ),
    },
    {
        "citation": r"\citet{hugonnet_accelerated_2021}",
        "summary": "hugonnet",
        "rate_key": "mass_rate_start_mid",
        "rate_err_key": "",
        "rate_period": "2013--2018",
        "period": "2000--2020",
        "method": "ASTER stereo DEMs, gap-interpolated, temporally regularised",
        "density": r"\textbf{850} (as ours)",
        "note": (
            "Same density and released 1$\\sigma$ errors. Our 2013--2018 offset is "
            "1.5--1.8$\\sigma$. Per-glacier the offset is \\emph{not} surge-specific."
        ),
    },
    {
        "citation": r"\citet{dussaillant_annual_2025}",
        "summary": "wgmsamce",
        "rate_key": "mass_rate_start_end",
        "rate_err_key": "sigma_start_end",
        "rate_period": "2013--2024",
        "period": "1976--2025",
        "method": "Temporal downscaling of geodetic data with in-situ observations",
        "density": r"\textbf{850$\pm$60}",
        "note": (
            "Annual series, so it can be evaluated over our exact sub-periods. "
            "Independently reproduces the sub-period increase."
        ),
    },
    {
        "citation": r"\citet{wang_accelerated_2022}",
        "summary": "wang",
        "rate_key": "mass_rate_headline",
        "rate_err_key": "",
        "rate_period": "2019--2021",
        "period": "2.24~a from 29 Mar 2019",
        "method": "ICESat-2 ATL11 altimetry, hypsometric extrapolation",
        "density": r"\textbf{600 effective} (zonal firn/ice)",
        "note": (
            "Volume rate differs from ours by only 1.2$\\sigma$; the headline mass gap "
            "is largely a density artefact."
        ),
    },
    {
        "citation": r"\citet{geyman_historical_2022}",
        "summary": "geyman",
        "rate_key": "observed_mass_rate",
        "rate_err_key": "",
        "rate_period": "1936--2010",
        "period": "1936--2010",
        "method": "Historical aerial photogrammetry vs.\\ modern DEM",
        "density": r"850$\pm$60",
        "note": (
            r"\textbf{Excludes Austfonna}, so our side must too. Projections for "
            "2010--2100 are $-0.67$ (RCP2.6) to $-0.92$ (RCP8.5)."
        ),
    },
    {
        "citation": r"\citet{schuler_reconciling_2020}",
        "summary": "",
        "rate_key": "",
        "rate_err_key": "",
        "rate_period": "",
        "period": "2000--2019",
        "method": "Upscaling of 12 in-situ records",
        "density": "n/a (not geodetic)",
        "note": (
            r"Reports both climatic ($-7\pm4$) and total ($-8\pm6$~Gt~a$^{-1}$) mass "
            "balance; only the total is comparable."
        ),
    },
    {
        "citation": r"\citet{malecki_recent_2022}",
        "summary": "",
        "rate_key": "",
        "rate_err_key": "",
        "rate_period": "",
        "period": "2011--2017",
        "method": "ArcticDEM DEM differencing",
        "density": "890$\\pm$20 / 550$\\pm$50",
        "note": (
            r"\textbf{Not archipelago-wide}: 1373~km$^2$ of small land-terminating "
            "glaciers. Use only as corroboration of the north/east anomaly."
        ),
    },
)


def _format_rate(summary: dict[str, object], source: dict[str, str]) -> str:
    """Tabulated rate for one source, as a LaTeX cell labeled with its window.

    The rate and its error are read from the fields named in ``source``, so an
    error is never attached to a rate from a different window.
    """
    if not source["rate_key"]:
        return "---"
    if "unavailable" in summary:
        return r"\emph{unavailable}"
    value = summary[source["rate_key"]]
    error = summary.get(source["rate_err_key"]) if source["rate_err_key"] else None
    number = f"${value}\\pm{error}$" if error is not None else f"${value}$"
    return f"{number}~Gt~a$^{{-1}}$ ({source['rate_period']})"


def write_comparison_table(changes: dict[str, object] | None = None) -> Path:
    """Write the Supporting Information comparison table to ``tables/comparisons.tex``.

    Every number comes from the corresponding module, so the table cannot drift from
    the analysis. The prose comparability note is the one part stated by hand, in
    :data:`TABLE_SOURCES`, because it is not derivable from the data.
    """
    summaries = get_comparisons(changes)
    lines = [
        "% Autogenerated by src.comparisons.write_comparison_table -- do not edit.",
        r"\begin{tabular}{p{2.7cm}p{1.9cm}p{3.4cm}p{2.0cm}p{2.6cm}p{4.4cm}}",
        r"\hline",
        (
            r"\textbf{Study} & \textbf{Period} & \textbf{Method} & \textbf{Rate} & "
            r"\textbf{Density (kg~m$^{-3}$)} & \textbf{Comparability} \\"
        ),
        r"\hline",
    ]
    for source in TABLE_SOURCES:
        summary = summaries.get(source["summary"], {}) if source["summary"] else {}
        rate = _format_rate(summary, source)
        lines.append(
            f"{source['citation']} & {source['period']} & {source['method']} & "
            f"{rate} & {source['density']} & {source['note']} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    path = TABLE_DIR / "comparisons.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
