# Svalbard glacier elevation change 2013–2024 from ArcticDEM: analysis

This repository turns gridded ArcticDEM elevation-change rates into per-glacier
statistics, compares them with published estimates, and produces every number, table
and figure in Mannerfelt et al. (in prep.). The methods are described in the paper;
this README covers how to reproduce its results.

## Related repositories

This repository starts from the products of two upstream repositories:

- [erikmannerfelt/ADSvalbard](https://github.com/erikmannerfelt/ADSvalbard): ArcticDEM
  strips to co-registered annual mosaics, elevation-change rate rasters and their
  uncertainties.
- [erikmannerfelt/SvalbardSubmarineRetreat](https://github.com/erikmannerfelt/SvalbardSubmarineRetreat):
  below-waterline volume change from the retreat and advance of tidewater fronts.

## Installation

Python ≥ 3.12. From the repository root:

```bash
pip install .            # installs the dependencies only; the code is not packaged
pip install --group dev  # optional: pytest, ruff, pyrefly
```

or `uv sync`. All commands below are run from the repository root.

## Input data

| Location | Contents | Source |
|---|---|---|
| `input/` | `trend_<period>_slope{,_fit_err,_baseline_err}.tif`, `trend_2019-2024_slope_uncorr.tif`, `trend_2013-2024_intercept.tif`, `dem_<year>_{count,doy,quality}.tif`, `dem_quality_mode.tif`, `aux_files.zip` | Data publication (DOI to be added), produced by ADSvalbard |
| `input/` | `submarine_volume_change.arrow`, `bed_model_validation.csv` | Data publication, produced by SvalbardSubmarineRetreat |
| `input/comparisons/hugonnet2021/` | `dh_07_rgi60_reg_cumul.csv`, `dh_07_rgi60_pergla_rates.csv.zip` | Hugonnet et al. (2021), https://doi.org/10.6096/13 (manual download) |

Everything else is downloaded automatically on first use: the Svalbard surge database
V5 with its RGI 7.0 outlines (Harcourt et al., https://doi.org/10.5281/zenodo.18033216),
the annual coastlines of the Norwegian Polar Institute
(https://doi.org/10.21334/npolar.2021.21565514), the datasets used in the comparisons
(Morris et al., 2020; Geyman et al., 2022; the WGMS annual mass-change estimates; the
RGI 7.0–6.0 links) and Natural Earth for the locator map. The URLs are recorded in the
modules that use them.

`shapes/` holds the two hand-made vector layers that are part of the method: the glacier
zones and the outline corrections.

## Reproducing the results

```bash
python main.py            # statistics, then figures
python main.py stats      # statistics only
python main.py figures    # figures only (needs the statistics caches)
```

| Output | Contents |
|---|---|
| `tables/statistics.json` | All reported statistics |
| `tables/statistics.tex` | The same as `\dataX...` LaTeX macros, which the manuscript cites |
| `tables/seasonal_coverage.tex`, `tables/classifier_importance.tex` | Tables S1 and S2 |
| `figures/` | All figures (below) |

Every number in the manuscript, including those in Tables S3–S5, comes from the
`\dataX...` macros.

| Figure | File in `figures/` |
|---|---|
| 1 | `perinterval_slope_overview.svg` |
| 2 | `elevation_change_examples.svg` |
| 3 | `surge_nosurge_bar.svg` |
| S1 | `annual_dem_count.png` |
| S2 | `rigidcoreg_histograms.svg` |
| S3 | `vertcoreg_summary.svg` |
| S4 | `coreg_source_map.png` |
| S5 | `tcorr_examples.svg` |
| S6 | `uncertainty_illustration.svg` |
| S7 | `baseline_err_variogram.svg` |
| S8 | `binned_terrain_err.svg` |
| S9 | `patch_method_vs_vgm.svg` |
| S10 | `perzone_{elevation,volume}_change{,_nonsurging}.svg` |
| S11 | `surging_examples.svg` |

Intermediate products are cached in `cache/`, most importantly the per-glacier table
`cache/outlines_sampled.arrow`. The first run builds the caches, which takes over half
an hour; later runs take about 20 s for the statistics and 5 min for the figures.
Delete `cache/` to rebuild from scratch. `tables/statistics.json` is merged into rather
than overwritten, so delete it before a clean regeneration.

## Repository layout

| Path | Contents |
|---|---|
| `src/outlines.py` | Period-specific glacier outlines and zones |
| `src/sampling.py` | Samples the rasters per glacier into `cache/outlines_sampled.arrow` |
| `src/statistics.py`, `src/reporting.py` | Aggregation and the `tables/statistics.*` output |
| `src/coreg.py`, `src/coverage.py`, `src/gap_classifier.py`, `src/positive_area.py` | Co-registration, data coverage, gap-filling and positive-change diagnostics |
| `src/submarine.py` | Reads the SvalbardSubmarineRetreat products |
| `src/comparisons/` | One module per published study compared against |
| `src/figures/` | One module per figure or figure group |

## Tests

```bash
python -m pytest
```

The tests are doctests in the modules under `src/`.

## Licence

MIT; see `LICENSE`.
