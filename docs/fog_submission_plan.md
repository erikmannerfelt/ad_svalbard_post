# FoG submission plan — above-water elevation change, Svalbard 2013–2024

**Status:** planned, not yet built. Submission is deliberately deferred until the
accompanying manuscript is published or in revision, so the reference can be a real DOI.

**Scope:** glacier-wide **above-water elevation change** for 1583 Svalbard glaciers over
three intervals, into the FoG `change` table. Nothing else. No `state` rows, no area
change, no submarine component, no `glacier` rows.

**Target contact:** Ethan Welty (WGMS). See "Lessons from the 2026 thread" at the end —
this plan is written to pre-answer the questions that generated ~10 rounds of email on the
previous submission.

---

## 1. Why this is not a drop-in

FoG's `change` table wants **the change in elevation between two dated surfaces**.

This project does not produce that. It produces a **per-pixel ordinary-least-squares rate
fitted to six annual medians**, averaged inside a period-specific outline. The mismatch is
structural and is the same one that generated the long thread on the Swiss (Mannerfelt et
al. 2022) submission.

Ethan has already stated how he resolves it, and it is the strategy adopted here:

> "as you suggested I have integrated the time-normalized elevation change rates along with
> the corresponding uncertainty and normalized dates."

So: **pick normalized dates, make every submitted number exactly self-consistent with them,
and say so in writing before he has to ask.**

---

## 2. Dates and the ×5 convention

### The decision

| Interval | `begin_date` | `end_date` | multiplier | `elevation_change` |
|---|---|---|---|---|
| 2013–2018 | `2013` | `2018` | × 5 | −3.737 m |
| 2019–2024 | `2019` | `2024` | × 5 | −5.181 m |
| 2013–2024 | `2013` | `2024` | × 11 | −9.821 m |

Year-only dates, with `begin_date_min` / `begin_date_max` and `end_date_min` /
`end_date_max` carrying the actual acquisition window (2013 is March-only; ~96 % of 2024
acquisitions predate May).

### Why ×5 and not ×6

The fit runs on **integer year indices**, so it assumes every annual median sits at the same
phase of its year. Under that assumption the separation between the 2013 and 2018 medians is
*exactly* 5.000 years — whatever that phase happens to be. `slope × 5` is therefore exact by
construction, and needs no epoch estimate (the fit deliberately discards within-year timing).

Year-only dates are the **correct precision**, not a fudge. The FoG instructions treat `2013`
as a point with unknown month and day, not as an interval, and provide `date_min`/`date_max`
for the range. So `begin_date: 2013` states "the 2013 annual median, timing within the year
unspecified" — precisely true.

### Why not inclusive end dates with ×6

`begin_date: 2013-01-01`, `end_date: 2018-12-31`, `dh = slope × 6` is **internally
consistent** — `dh/Δt` returns the rate, and the division check passes. It was seriously
considered and rejected for a different reason: it *claims* surfaces on 2013-01-01 and
2018-12-31 that were never observed. Roughly 5 of those 6 years are observed and the
remaining ~7 months are extrapolation past both ends of the record, including a Dec–Jan
window at the 2018 end with no acquisitions at all. It also inflates the reported loss by
20 % (−4.48 m vs −3.737 m), with that 20 % unobserved.

### ⚑ RAISE WITH ETHAN

**Flag this convention explicitly in the covering email and ask him to confirm before
integration.** Phrase it as: here is the quantity, here are two defensible date/multiplier
conventions, here is why ×5 was chosen, does FoG want it this way? Getting an explicit
answer costs one email round and removes the single largest source of ambiguity in the
submission. Do not let him discover the convention by dividing.

### This does NOT affect the paper

Verified against the codebase and manuscript (values refreshed 2026-09-05 after the
`outline_corrections.geojson` update):

- The manuscript reports **rates** throughout (73 uses of the `m a⁻¹` / `km³ a⁻¹` unit macros across 56 lines) and
  no cumulative elevation or volume change anywhere. Every published comparison
  (Hugonnet, AMCE, Geyman, Morris, Wang) is likewise a rate.
- The only `/ 6` in the pipeline is `src/sampling.py:154`, the acceleration term, and it is
  **correct and unrelated**: that 6 is a *midpoint* separation (2013–2018 centred on 2015.5,
  2019–2024 on 2021.5), not an endpoint separation.

The two are different quantities that sit either side of the same interval:

- **midpoint separation = 6.0 a** → divides a difference of two rates (acceleration)
- **endpoint separation = 5.0 a** → multiplies a rate into a change between two surfaces

×5 is therefore a **FoG-only convention**. Nothing propagates into the paper, the figures,
or the comparison modules.

### Relationship to the GlaMBIE-2 submission

`src/glambie.py` uses `INTERVAL_YEARS = 6` with dates 01/03/2013 → 01/01/2019. That answers a
different question (a rate integrated over a stated window, not a difference between two
surfaces) and stays as submitted. **The README must state the discrepancy**, since both
submissions become public.

Separately: the GlaMBIE row is mildly self-inconsistent on its own terms — its dates span
5.84 a but carry a ×6 value, so re-deriving a rate gives −0.767 rather than −0.747 m a⁻¹
(2.7 % off). Worth a one-line email only if the submission window is still open.

---

## 3. Outlines and glacier identity

**Submit polygons; omit `glacier_id` and `glacier_name`.**

207 glaciers carry outlines modified from published RGI 7.0 — but they are **72.9 % of the
total area** (the large tidewater basins, coastline-clipped). RGI 7.0 IDs alone would
misrepresent almost three quarters of the dataset.

The FoG extension covers this exactly:

> "columns `glacier_id` and `glacier_name` may be omitted from table `change` if columns
> `begin_outline_id` and `end_outline_id` are provided"

This is the path Ethan explicitly asked for ("we have paused trying to curate a glacier ID
for each remotely-sensed glacier observation... the outline is the only thing that can be
relied upon to explicitly describe where the glacier-wide measurement was made").

**Do not mint temporary glacier IDs this time.**

### Outline files

One GeoJSON per interval, `id` formatted `ADSV-1318-00741` (dataset, interval, RGI local
token). Attributes beyond the required `id`:

| attribute | purpose |
|---|---|
| `rgi_id` | full RGI 7.0 identifier, for Ethan's own inventory linking |
| `glac_name` | display name where one exists (939 of 1583) |
| `modified` | outline differs from published RGI 7.0 |
| `surging` | surge overlapped this interval |
| `term_type` | RGI terminus type |

### begin_outline_id == end_outline_id

Deliberate, and must be stated rather than left to inference. There is genuinely **one
outline per interval**: the RGI 7.0 outline clipped to the *union* of annual coastlines over
that interval, i.e. effectively the **maximum (≈ begin) extent**. This also tells the reader
that `area` is a begin-area, not a mean area.

---

## 4. What is deliberately NOT submitted

### No area series / no `state` rows

The outlines exist for **correct glacier attribution of elevation change**, not to measure
area change. Only the termini have variable area, so an area series built from them would be
a poor product.

**Required guardrail.** The three interval outlines have different areas and someone will
difference them:

| Interval | Area (km²) |
|---|---|
| 2013–2018 | 33 860.7 |
| 2019–2024 | 33 691.3 |
| 2013–2024 | **33 927.0** |

2013–2024 is the union of all twelve annual coastlines and is therefore the **largest** of
the three despite spanning the longest period — area appears to *grow* over the longer
interval. The README must state that these are **period-specific attribution masks** — the
extent over which each rate was averaged — not dated area measurements, and that they must
not be differenced.

### No submarine component

`change.volume_change` is **above-water only**, matching exactly what DEM differencing
observes. The submarine term (marine-front area change × bed depth) is a different
observation type; adding it into the same `volume_change` cell is structurally the same move
that produced the errata on Holmlund (2021) and Mannerfelt et al. (2024).

Out of scope for this submission. May be revisited as a separate, clearly-labelled product
once the method is settled; if so, the interval alignment question (its layers currently run
2012–2018 / 2018–2023 / 2012–2023 against the ArcticDEM 2013–2018 / 2019–2024 / 2013–2024)
must be resolved first.

### No per-row surge or temporal-correction flags

Both are already absorbed into the error model. σ_excess explicitly represents genuine
non-linearity in the elevation history — seasonal offsets, **surge pulses**, step changes —
and is summed linearly (fully systematic), so surge-affected glaciers already carry inflated
uncertainty. The temporal bias correction has its own calibrated uncertainty combined into
σ_fit in quadrature.

Putting either in `change.remarks` would read as a **data-quality warning** and invite users
to drop those rows — which would bias a regional total badly, since surge-affected glaciers
carry a large share of the loss. `surging` therefore ships as descriptive metadata in the
outline GeoJSON instead, with one README line confirming the rows need no special handling.
The temporal correction is described in the README method section only.

---

## 5. Uncertainty

### Confidence level

**1σ (68 %).** State it in the README header *and* in `change.remarks`. Ethan had to ask this
directly last time ("Do you remember – is the dh_err in the Zenodo file 1-sigma or 95 %
confidence interval...?").

### volume_change ≡ elevation_change × area, exactly

By construction, within a fixed outline. State the convention **and** state that it differs
from the (erroneous) convention in Holmlund (2021) Table 2 and Mannerfelt et al. (2024)
Table 1, where a mean elevation change was treated as a mean thickness change. Ethan found
those errors by dividing volume by area; he will do it again. Better it comes from us.

### The per-glacier errors are NOT independent

This is the most important caveat in the submission. σ_excess is treated as fully systematic,
so naive quadrature badly understates any aggregate:

| | 2013–2018 | 2019–2024 |
|---|---|---|
| Reported regional (km³ a⁻¹) | **4.11** | **3.90** |
| Quadrature of per-glacier errors | 0.78 | 0.55 |
| Linear sum of per-glacier errors | 4.18 | 4.00 |

Quadrature understates by **≈ 5.3×** — the difference between "significant" and "not" for
anyone reusing these rows.

For reference, Hugonnet et al. (2021) did **not** inflate per-glacier errors to make
quadrature work (a possibility considered and rejected here — it distorts every individual
value to make one sum come out right). Their Svalbard 2000–2020 per-glacier quadrature gives
0.277 km³ a⁻¹ against their own regional 0.476; they simply published both and let the
regional file be the authority.

### Solution: ship the components, not a warning

The exact aggregation rule (`src/statistics.py:427`) is:

```
baseline = Σ(vol_baseline_err_unscaled) / √n_eff(total area)
excess   = Σ(vol_excess_err)                    # linear
total    = hypot(baseline, excess)
```

This **cannot** be recovered from the per-glacier `_err` column alone, because baseline
de-correlates over the *aggregate* area, not per glacier. (The linear sum landing within 2 %
of the reported value is a coincidence of this dataset, not the rule.)

Therefore ship a companion `uncertainty_components.csv`, keyed by `outline_id` + interval,
carrying the baseline and excess terms separately, plus the recipe above and the published
regional and per-zone totals as a check. That makes correct aggregation **reproducible**
rather than merely warned about.

---

## 6. Metadata

### Investigators and agencies

```
investigators   Erik Schytt {Mannerfelt} (1, 2) 0000-0002-9146-557X |
                Geir Moholdt (3) 0000-0002-8883-1620 |
                Jack Kohler (3)

agencies        1. Arctic Geology, The University Centre in Svalbard, Norway |
                2. Department of Geosciences, University of Oslo, Norway |
                3. Norwegian Polar Institute, Norway
```

Notes:

- Braces are `{Mannerfelt}`, **not** `{Schytt Mannerfelt}`. This matches the surname already
  in the FoG database ("the database knows you as Erik Schytt Mannerfelt (surname:
  Mannerfelt), synonym: Erik Schytt Holmlund, orcid: 0000-0002-9146-557X"), so the
  submission attaches to the existing person record rather than minting a second one.
- **TODO: Jack Kohler's ORCID.** Optional per the spec, but it resolves ambiguity and the
  other two authors have theirs.
- **TODO: co-author order**, once decided. One-line edit.
- Optional: the spec encourages structured agency names (`UNIS > Arctic Geology`, original
  language with English in brackets). The plain strings above are used as specified; offer
  the structured form if Ethan prefers.

### References

At publication: `Mannerfelt {year} (https://doi.org/{doi})`. Until then the submission is on
hold — this is the reason for deferring.

### Remarks (per row)

Kept minimal and factual: the ×5 convention, the fixed-outline volume identity, 1σ, and the
"three intervals overlap, do not sum" note. No caveat-flavoured flags (see §4).

---

## 7. Package layout

```
tables/fog_<date>/
├── change.csv                    # 1583 glaciers × 3 intervals = 4749 rows
├── outlines/
│   ├── outlines_2013-2018.geojson
│   ├── outlines_2019-2024.geojson
│   └── outlines_2013-2024.geojson
├── uncertainty_components.csv    # baseline + excess terms, per glacier per interval
└── README.md
```

Note that `tables/` is gitignored, so the generated package is **not** version-controlled.
It is fully regenerable from `src/fog.py`, but the exact bytes actually sent to WGMS should be
archived alongside the manuscript's data publication, since FoG is a permanent archive and the
submitted version needs to stay citable.

### `change.csv` columns

`begin_outline_id`, `end_outline_id`, `begin_date`, `begin_date_min`, `begin_date_max`,
`end_date`, `end_date_min`, `end_date_max`, `area`, `elevation_change`,
`elevation_change_unc`, `volume_change`, `volume_change_unc`, `begin_platform`,
`begin_method`, `end_platform`, `end_method`, `investigators`, `agencies`, `references`,
`remarks`.

`begin_platform` / `end_platform` = `space`; `begin_method` / `end_method` = `photo`
(optical stereo photogrammetry — ArcticDEM/SETSM). All units SI per §3.2 of the
instructions: m, m², m³.

### README.md contents

Largely a re-cut of `ad_svalbard_post_text/glambie2.tex` §Method / §Coverage / §Uncertainty,
which is already written and reviewed. Plus a FoG-specific conventions section covering:
1σ; the ×5 convention and why; `volume_change ≡ elevation_change × area` in a fixed outline,
and how that differs from the 2021/2024 papers; non-independent errors and the aggregation
recipe; attribution masks are not an area product; the GlaMBIE ×6 discrepancy; and the
above-water-only scope.

---

## 8. Implementation

New `src/fog.py`, modelled on `src/glambie.py` (frozen dataclasses, `Final` constants,
numbers read from the pipeline so they cannot drift). Reads `sampling.sample_rasters()`
directly — **not** `tables/statistics.json`, whose values are pre-rounded strings.

```
build_change_rows()        -> 4749 rows
write_outline_geojsons()   -> three files
write_submission_package() -> tables/fog_<date>/
```

Entry point `python -m src.fog`, per repo convention.

**Pre-write assertions** (must pass before anything is written):

1. `volume_change == elevation_change × area` to float tolerance, on every row
2. `elevation_change == slope × n_years` with `n_years` from the interval, on every row
3. every `begin_outline_id` / `end_outline_id` resolves to exactly one polygon in the
   matching GeoJSON
4. no NaN in any submitted numeric column
5. row count == 1583 × 3

---

## 9. Pre-submission checklist

- [ ] Manuscript published or in revision; DOI available
- [ ] **Raise the ×5 vs ×6 convention with Ethan explicitly and get confirmation** (§2)
- [ ] Jack Kohler's ORCID
- [ ] Co-author order decided
- [ ] Regenerate from the current pipeline; confirm the five assertions pass
- [ ] README states: 1σ · ×5 · fixed-outline volume identity · divergence from the 2021/2024
      convention · non-independent errors + recipe · attribution masks ≠ area product ·
      GlaMBIE ×6 discrepancy · above-water only

---

## 10. Lessons from the 2026 email thread

Recorded so the next submission does not repeat them. Source:
`resources/fog/Re_ Elevation change FoG submission from 3 papers and _2000 glaciers.eml`.

| What happened | Pre-empted here by |
|---|---|
| "Is `dh_err` 1-sigma or 95 %?" | 1σ in the README header **and** in `remarks` |
| Volume ÷ area ≠ reported elevation change (Holmlund 2021 Table 2; Mannerfelt 2024 Table 1) — an error found by WGMS, not by us | `volume_change ≡ elevation_change × area` by construction, asserted before write, and the divergence from those papers stated outright |
| Submitted dates disagreed with the inventory outline dates | Dates are the fit's year labels, not acquisition dates; `_min`/`_max` carry the real acquisition window; the distinction is stated |
| Temporary glacier IDs, then told they were unnecessary | Outline-only extension; no IDs minted |
| End outlines requested after the fact | Both outline columns populated, with the begin==end reasoning stated |
| "Could you please alter your code to replace all dates by their full range?" | `_min`/`_max` used from the start |
| Analysis machine returned; code could not be re-run | Everything derives from `src/fog.py` in this repo, regenerable on demand |

The recurring theme: **every number that can be re-derived from another number will be, by
Ethan, and any inconsistency will surface.** Make the identities exact, assert them in code,
and state the conventions before being asked.
