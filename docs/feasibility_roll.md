# Experiment `roll`, Phase 0: feasibility of the prequential streams

Status: **Phase 0 done on 2026-10-30; waiting for the lead's decisions (gate ⛔, end of this document).** Branch `roll`.
Script: `scripts/feasibility_roll.py` (aggregates only; output in `work/experiments/roll/feasibility/`).

`roll` is a **pre-registered retrospective evaluation on data already examined**. The three cohorts (INSPIRE,
MIMIC-IV, eICU) were used with this outcome in `hypo`, whose per-site and per-hospital aggregates have been seen.
What has not been seen is the performance along a prequential stream with a rotating case base. Nothing here is a
prospective validation, and no data are new.

## What was checked

- **The frozen experiments.** The freezes of T2 and `hypo` were checked at the start and at the end of the script:
  - T2: tag `perisim-freeze-t2-2026-10-27-141040`, code `77b25fe8f629`, configuration `1f6d4f5b89d4`;
  - `hypo`: tag `perisim-freeze-hypo-2026-10-29-120652`, code `cf0d0b80c70a`, configuration `772940e4ed77`.

  Both were unchanged at the end. Nothing of either was modified.
- **The population.** `hypo`'s analysis landmarks: every 2 h during the first 48 h of a postoperative ICU stay, for
  a patient normotensive now, outcome known. Same inputs, cutoff and outcome (hypotension in the next 6 h).
- **The unit.** A patient. "Patients with an event" have at least one landmark followed by hypotension. Counts
  below 10 are not shown.

## The three cohorts

| cohort | patients | patients with an event | landmarks | event landmarks | prevalence (landmarks) | landmarks per patient |
|---|---|---|---|---|---|---|
| INSPIRE | 11,631 | 4,475 | 148,293 | 19,513 | 13.2 % | 12.7 |
| MIMIC-IV | 24,464 | 17,934 | 387,159 | 118,714 | 30.7 % | 15.8 |
| eICU | 32,032 | 17,589 | 466,385 | 90,214 | 19.3 % | 14.6 |

## MIMIC-IV: calendar blocks

`anchor_year_group` is the patient's 3-year window of real time.

| block | patients | patients with an event | landmarks | event landmarks | prevalence |
|---|---|---|---|---|---|
| 2008–2010 | 6,942 | 5,207 | 112,195 | 35,875 | 32.0 % |
| 2011–2013 | 5,929 | 4,409 | 92,025 | 28,835 | 31.3 % |
| 2014–2016 | 5,356 | 3,985 | 83,493 | 26,400 | 31.6 % |
| 2017–2019 | 3,686 | 2,617 | 57,452 | 16,857 | 29.3 % |
| 2020–2022 | 2,551 | 1,716 | 41,994 | 10,747 | 25.6 % |

- **There are five blocks, not four.** The MIMIC-IV version in use includes 2020–2022. That gives four boundaries
  instead of three. The last block is the smallest, and its prevalence is the lowest. Whether to include it is a
  decision for the lead (decision 1).
- **Blocks are by patient, not by stay.** A stay can fall years after its patient's anchor year. Of the 26,295
  stays, 386 (1.5 %) have no match in `icustays` and so no year. Of the rest:
  - 70.7 % are in the anchor year itself;
  - 16.3 % are 3 or more years later, and may belong to a later block;
  - 7.4 % are 6 or more years later.

  MIMIC-IV gives the real-time window per patient only, and this design uses it. The boundaries are therefore
  blurred for about one stay in six, which biases boundary effects towards zero. This is declared.
- **The eMAR → prescriptions change does not reach `hypo`'s inputs.** In MIMIC-IV those inputs come from:
  - `chartevents` (vital signs, ventilation);
  - `labevents` (haemoglobin, creatinine, lactate);
  - `inputevents` (vasopressors, sedation);
  - the common format's `admissions` (emergency) and `services` (surgical group).

  No input reads `emar` or `prescriptions`; only T2's medication variables did.

## eICU: hospitals and years

| year | patients | patients with an event | landmarks | event landmarks | prevalence |
|---|---|---|---|---|---|
| 2014 | 15,683 | 8,710 | 224,097 | 44,538 | 19.9 % |
| 2015 | 16,696 | 8,981 | 242,288 | 45,676 | 18.9 % |

- **196 hospitals** contribute landmarks. The year is `hospitaldischargeyear`, and no landmark lacks it.
- **Eligible hospitals by threshold** (patients in each year; labels H1, H2, … by size, as in `hypo`):

  | threshold, each of 2014 and 2015 | hospitals | patients |
  |---|---|---|
  | ≥ 100 patients | 49 | 24,764 |
  | ≥ 150 patients | 35 | 21,169 |
  | ≥ 200 patients | 26 | 17,767 |
  | ≥ 300 patients | 14 | 11,583 |

- **The event threshold never binds.** The same counts hold with ≥ 10, ≥ 20 or ≥ 30 patients with an event per
  year, because about half of the patients have at least one event landmark. The binding constraint for a window
  is its event landmarks (next section).
- **The 35 hospitals with ≥ 150 patients a year:**
  - median 277 patients a year (range 151–885);
  - median 14.8 landmarks per patient;
  - prevalence 7.1 % to 31.4 % across hospitals and years;
  - change in prevalence from 2014 to 2015: median −0.7 points, interquartile range −2.6 to +0.7.

  Most hospitals change little from one year to the next.
- **`patientunitstayid` does not order a hospital's patients in time.** The year predicts it with a median AUC of
  0.51 (10th percentile 0.43; 115 hospitals with ≥ 10 patients in each year; none above 0.9). The planned
  sensitivity analysis that orders patients by this id would test nothing, so it is proposed to drop it
  (decision 4).

## INSPIRE: no calendar

- **No field orders INSPIRE's patients in calendar time:**
  - `opdate` and every time column count minutes from each patient's own first admission (`admission_time` is 0
    for all 130,960 operations);
  - `opdate` has a median of 2 days and a 90th percentile of 476 days;
  - the identifiers are random: `opdate` correlates with `op_id`, `subject_id` and `hadm_id` at |r| < 0.004.
- **INSPIRE's role.** It enters as a random-order stream with no known drift, for stability and the false-alarm
  rate. It is also a starting cohort in the cross-cohort streams.

## Precision of a window, against its size

This is the AUROC of `hypo`'s transported PERISIM in 300 random windows of W patients from each external site's
test rows. The windows are random, so there is no drift by construction: this spread is the "no-drift" reference
for the excess SD.

| W (patients) | MIMIC-IV landmarks / events / ±95 % | eICU landmarks / events / ±95 % | INSPIRE landmarks / events / ±95 % |
|---|---|---|---|
| 50 | 788 / 245 / 0.063 | 729 / 138 / 0.069 | 639 / 84 / 0.093 |
| 100 | 1,572 / 484 / 0.046 | 1,453 / 282 / 0.046 | 1,272 / 167 / 0.058 |
| 150 | 2,377 / 724 / 0.034 | 2,177 / 419 / 0.039 | 1,911 / 253 / 0.050 |
| 200 | 3,171 / 974 / 0.032 | 2,913 / 562 / 0.033 | 2,561 / 334 / 0.044 |
| 300 | 4,753 / 1,458 / 0.023 | 4,359 / 844 / 0.027 | 3,816 / 499 / 0.032 |
| 500 | 7,930 / 2,429 / 0.019 | 7,287 / 1,408 / 0.019 | 6,381 / 841 / 0.027 |
| 1,000 | 15,820 / 4,848 / 0.013 | 14,571 / 2,808 / 0.016 | 12,761 / 1,680 / 0.018 |

- **For an AUROC within ±0.05**, a window needs about 100 patients in MIMIC-IV and eICU and about 150 in INSPIRE.
- **In an eICU hospital with ~280 patients a year**, windows of 100 patients give 2–3 windows a year, ±0.05. Windows
  of 50 give 5–6 windows a year, ±0.07.

These numbers set the proposals for W and E in Phase 1.

## Streams that the data allow

| stream | real changes (boundaries) | control stretches | starting case base |
|---|---|---|---|
| MIMIC-IV temporal | 4 block boundaries (3 if 2020–2022 is left out) | inside each block, random order | first N patients of 2008–2010 |
| eICU, within a hospital (a) | the change of year, in each of 35 hospitals (≥ 150 / year) | inside each year | its first N patients of 2014 |
| eICU, deployment (b) | the change from other hospitals to the hospital, then the change of year | inside each year | N patients of other hospitals |
| cross-cohort (stress) | the change of cohort | inside each cohort, random order | N patients of the first cohort |
| INSPIRE | none | the whole stream | first N patients |

**The metric of each stream** is the frozen `hypo` metric of the rotation that excluded the evaluated cohort:
ext-mimic, ext-eicu or ext-inspire.

**Cross-cohort pairs proposed** (decision 5). The case base starts in the first cohort, and the stream moves to the
second. The metric excluded the second cohort, so in the first stretch it is internal, and only in the second is it
external.

| pair | metric | first stretch | second stretch | why |
|---|---|---|---|---|
| INSPIRE → MIMIC-IV | ext-mimic | internal (INSPIRE was in its development) | external | largest rise in prevalence (13 → 31 %); hourly non-invasive → invasive arterial pressure |
| eICU → MIMIC-IV | ext-mimic | internal | external | rise in prevalence (19 → 31 %) with a similar measurement: separates case mix and practice from measurement |
| MIMIC-IV → INSPIRE | ext-inspire | internal | external | the opposite direction (31 → 13 %): a fall in base rate, where an over-predicting model fails |

INSPIRE → eICU (13 → 19 %, ext-eicu) is the alternative to the third pair.

## Other points the design must carry

- **The admission gate refuses nothing in these data.** It checks duplicates (stay, L), values outside `hypo`'s
  bounds, and impossible values (negative L, age < 18, MAP minimum above the mean). All three cohorts give 0, because
  `hypo`'s extraction already applies the bounds. The gate stays as part of the method, is applied to every arm and
  is reported.
- **Label delay does not apply.** No stream has real time inside a block, a year or a cohort: the order there is
  random. A patient enters the case base only after being evaluated, with outcomes known by construction. This is
  declared.
- **Admission rates, for translating a deployment delay into patients:**

  | where | patients with landmarks per year |
  |---|---|
  | MIMIC-IV, by block | 2,314 / 1,976 / 1,785 / 1,229 / 850 (2008–2010 … 2020–2022) |
  | eICU hospital, eligible | median 277 (151–885) |
  | INSPIRE | about 1,160 over 2011–2020 (`[VERIFY]` the period in the dataset's documentation) |

- **N against the stream sizes:**
  - MIMIC-IV: N = 2,000 is close to the whole 2020–2022 block (2,551 patients).
  - eICU: N = 250 is close to a median hospital's year (277); N = 100 is the only size that leaves most of 2014 to
    the stream.
- **T2 (secondary, descriptive).** T2's unit is the ICU discharge, not the landmark. Its counts per block and year
  are added in Phase 1.

## Where the confirmatory weight can be

- **MIMIC-IV: 3–4 temporal boundaries in a single stream.** That is enough for the level, stability and
  calibration endpoints (H1–H4), which use every window. It is not enough for an inferential test of the signal at
  the boundaries (H6), which would be descriptive here.
- **eICU: 35 hospitals with a real boundary (the change of year).** These replicates, combined by random-effects
  meta-analysis, can carry H6 and the per-hospital forms of H1–H4. The windows are small (about 2–6 a year per
  hospital).
- **Cross-cohort: 3 streams with one large change each.** These are descriptive or secondary (stress).
- **INSPIRE: no boundary.** It is a control for false alarms and stability.

**Expected:** little drift within MIMIC-IV's blocks and within eICU's hospitals, since prevalence is nearly flat
there and `hypo` found the metric transportable with a small hospital effect. So:
- a null result ("not enough drift in these data to tell the arms apart") is plausible for the primary streams;
- differences, if any, are expected in the cross-cohort streams and in calibration rather than discrimination.

## Gate ⛔: decisions for the lead

1. **MIMIC-IV blocks.** Option (a): five blocks including 2020–2022, four boundaries; the last block is small and
   COVID-era. Option (b): four blocks ending in 2019, as first planned.
2. **eICU eligibility.** Options: ≥ 150 patients in each year (35 hospitals, 21,169 patients; proposed), ≥ 100
   (49 hospitals) or ≥ 200 (26 hospitals). The per-window event minimum E is set in Phase 1 on event landmarks.
3. **INSPIRE** as a random-order control and as a starting cohort (confirmed by the data).
4. **Drop the sensitivity analysis that orders eICU by `patientunitstayid`.** The id carries no calendar order.
5. **Cross-cohort pairs.** INSPIRE → MIMIC-IV, eICU → MIMIC-IV, MIMIC-IV → INSPIRE (or INSPIRE → eICU instead of
   the last).
6. **Confirmatory weight.** H1–H4 in MIMIC-IV and in eICU's hospitals (meta-analysis); H6 confirmatory in eICU only,
   descriptive in MIMIC-IV; cross-cohort streams secondary.

**Risk.** Drift is small in the primary streams, so the experiment may end in an informative null. Within-hospital
windows are small, and their CIs will be wide.

**Points that could favour PERISIM unfairly**, to settle in Phase 1:
- **α.** `hypo` chose α by a leave-one-out over each case base. Re-choosing it at every step is a deterministic
  function of the window and uses no outcome outside it, but it is an adjustment the ML arms do not get between
  updates. The alternative is α fixed at its value on the starting window.
- **The training data of the transported ML.** The transported ML arms are trained on the same two cohorts as the
  PERISIM metric. That is symmetric, and is kept.
