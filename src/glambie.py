"""GlaMBIE-2 data submission table for GTN-G region 7 (Svalbard).

Builds the two-row CSV described in the GlaMBIE Data Submission Guide
(``resources/glambie2/Data Submission Guide - GlaMBIE.html``) from the
full-precision output of ``statistics.get_statistics()``, so the submitted
numbers cannot drift from the ones quoted in the submitted methods
document (``Mannerfelt_GlaMBIE_Description_of_Methods.docx`` in the workspace root).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from . import sampling, statistics, tools
from .config import TABLE_DIR

REGION_ID: Final = 7
UNIT: Final = "m"
REMARKS: Final = "Mean elevation change rate from ArcticDEMs for Svalbard (not Jan Mayen)."

#: Real ArcticDEM coverage begins in March 2013, and about 96% of the 2024
#: acquisitions predate May, so the record effectively ends 30 April 2024
#: (methods document, Sect. 3 "Temporal coverage"). The shared boundary between the two
#: submitted intervals is an analysis split, not an edge of the observed
#: record, and stays at the turn of the year.
RECORD_START: Final = datetime.date(2013, 3, 1)
INTERVAL_SPLIT: Final = datetime.date(2019, 1, 1)
RECORD_END: Final = datetime.date(2024, 5, 1)  # exclusive: covers through 30 April 2024

@dataclass(frozen=True)
class SubmissionRow:
    interval: tools.Interval
    start_date: datetime.date
    end_date: datetime.date

    @property
    def years(self) -> float:
        """Actual observed span, in Julian years, spanned by `start_date`-`end_date`.

        Used to convert the fitted annual rate into the submitted cumulative change, so
        the two open ends of the record (start_date short of six calendar years by
        design; see `RECORD_START`/`RECORD_END`) are reflected in the submitted value.
        """
        return (self.end_date - self.start_date).days / 365.25


ROWS: Final = (
    SubmissionRow(tools.get_interval("13_18"), RECORD_START, INTERVAL_SPLIT),
    SubmissionRow(tools.get_interval("19_24"), INTERVAL_SPLIT, RECORD_END),
)

_HEADER: Final = (
    "region_id,start_date,end_date,glacier_change_observed,glacier_change_uncertainty,"
    "unit,glacier_area_reference_start,glacier_area_reference_end,"
    "observational_coverage_percentage,remarks"
)


def _format_date(date: datetime.date) -> str:
    return date.strftime("%d/%m/%Y")


def build_submission_lines(
    stats: dict[str, object] | None = None,
    coverage: dict[str, float] | None = None,
) -> list[str]:
    """Build the GlaMBIE-2 CSV lines: a header followed by one row per submitted interval.

    Parameters
    ----------
    stats
        Full-precision output of ``statistics.get_statistics()``. Recomputed
        if not given; pass it explicitly to avoid rerunning the pipeline when
        called after ``get_statistics()`` elsewhere.
    coverage
        Observational coverage percentages keyed by ``Interval.short``, as returned
        by ``sampling.compute_coverage_fractions()``. Derived if not given, so the
        submitted coverage cannot drift from the outlines it was measured against.
    """
    if stats is None:
        stats = statistics.get_statistics()
    if coverage is None:
        coverage = sampling.compute_coverage_fractions()

    lines = [_HEADER]
    for row in ROWS:
        changes = stats["changes"][f"slope_{row.interval.stats_key}"]["all"]
        elev_rate = changes["elev_rate"]
        elev_rate_err = changes["elev_rate_err"]
        area = changes["area"]

        lines.append(
            ",".join(
                [
                    str(REGION_ID),
                    _format_date(row.start_date),
                    _format_date(row.end_date),
                    f"{elev_rate * row.years:.2f}",
                    f"{elev_rate_err * row.years:.2f}",
                    UNIT,
                    f"{area:.2f}",
                    f"{area:.2f}",
                    f"{coverage[row.interval.short]:.2f}",
                    f'"{REMARKS}"',
                ]
            )
        )
    return lines


def write_submission_csv(
    stats: dict[str, object] | None = None, date: datetime.date | None = None
) -> Path:
    """Write the GlaMBIE-2 submission CSV to ``tables/mannerfelt_glambie2_<date>.csv``.

    `date` names the output file (the submission date) and defaults to today;
    it does not affect any interval boundary.
    """
    date = date or datetime.date.today()
    out_path = TABLE_DIR / f"mannerfelt_glambie2_{date:%Y%m%d}.csv"
    out_path.write_text("\n".join(build_submission_lines(stats)) + "\n")
    return out_path


if __name__ == "__main__":
    written_path = write_submission_csv()
    print(f"Wrote {written_path}")
