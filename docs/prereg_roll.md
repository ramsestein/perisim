# Pre-registration — experiment `roll`: prequential evaluation with a rotating case base (draft)

Status: **draft by the analyst, 2026-10-30, for the lead's approval**. Written after Phases 0–2 and before any
prediction along any stream of real data. It is registered with the freeze (Phase 3), and the single run (Phase 4)
runs on the frozen version only. Design: [design_roll.md](design_roll.md); feasibility:
[feasibility_roll.md](feasibility_roll.md); dated decisions: [DECISIONS.md](DECISIONS.md); configuration:
`config/roll.yaml`; code: `src/roll/`.

**This is a pre-registered retrospective evaluation on data already examined.** It is not a prospective validation,
and it uses no new data. The three cohorts were used with this outcome in `hypo`, whose aggregates per site and per
eICU hospital have been seen (§3). What has not been seen is the performance along a prequential stream with a
rotating case base.

## 1. Question

The thesis put to the test: **"CE-kNN is not less accurate than ML managed as a real hospital manages it, and it is
more governable."** More governable means:
1. deterministic inference, reproducible by hash;
2. an update without retraining: new patients enter the case base and the oldest leave (FIFO), with no cycle of
   monitoring, decision, revalidation and deployment;
3. stable performance along the stream, without the saw-tooth of a model that degrades between updates, and a quick
   recovery after a change;
4. diagnostics (out of support, re-deformation of the shapes) that flag the real changes of the stream.

The experiment is not meant to show that PERISIM predicts better than LightGBM. It is designed so that it can fail,
and `hypo`'s numbers say that its accuracy half is expected to fail against LightGBM (§10).

## 2. Data (fixed in Phases 0–2)

- **Population and outcome**: `hypo`'s analysis landmarks, with the same definitions, inputs and cutoff.
  - Every 2 h during the first 48 h of a postoperative ICU stay, for a patient normotensive now.
  - Outcome: an hour with median MAP < 65 mmHg in the next 6 h.
  - 148,293 / 387,159 / 466,385 landmarks of 11,631 / 24,464 / 32,032 patients (INSPIRE / MIMIC-IV / eICU).
- **The unit is the patient**: all their landmarks, of all their stays. An eICU patient belongs to the hospital
  holding most of their landmarks and to their first year (142 patients have landmarks in two hospitals).
- **Calendar fields**:
  - MIMIC-IV's `anchor_year_group`: five blocks, 2008–2010 to 2020–2022;
  - eICU's `hospitaldischargeyear` (2014, 2015) and hospital;
  - INSPIRE: none.
- **eICU hospitals**: the 35 with ≥ 150 patients with landmarks in each of 2014 and 2015.
- **Admission gate**: duplicates, values outside `hypo`'s bounds, impossible values. It refuses no landmark in these
  data.
- **T2 (secondary, descriptive)**: T2's analysis stays of MIMIC-IV (24,768 stays, 23,162 patients), with T2's labels
  as its external stage made them.

## 3. Done and seen before the freeze

- **`hypo`, entirely**: its pre-registration, freeze, external results and report, and both rounds of post hoc
  analyses. Among them, the numbers this design uses:
  - the AUROC of PERISIM with case bases of n random patients (S1);
  - the transported and local comparators (S2), with their AUPRC.
- **Phase 0** (`scripts/feasibility_roll.py`): counts per block, year, hospital and cohort; the calendar fields; the
  precision of a window's AUROC against its size. The precision came from random windows of `hypo`'s transported
  predictions.
- **Phase 1**:
  - T2's counts;
  - how often each proposed ML trigger fires on noise, from random windows of `hypo`'s transported predictions.
- **Phase 2**:
  - the code, its tests and a dry run on **synthetic** data only;
  - on the real data, the streams were built and counted (sizes, windows, digests), and nothing was predicted along
    them;
  - one real-data test reproduced `hypo`'s published transported predictions (INSPIRE, fold 0) with a window equal
    to its case base;
  - two corrections came from the synthetic dry run (design §7.4–§7.5): the order of the closed test and ADWIN's cut.
    Both were chosen to make the procedures work as described, not for any effect on results.
- **Before this pre-registration, by the lead's decision**: the calibration of `hypo`'s S1 case bases (DECISIONS
  2026-10-30). This is a new aggregate of examined predictions:
  - O/E 0.94–1.06 at every size;
  - a calibration slope of about 1.4 at 100 patients;
  - a calibration in the large of ±0.4 between case bases of 100 patients.

  N was not changed after it.
- **Not seen**: the performance of any arm along any real stream. Nothing of the single run exists.

## 4. Frozen artefacts

The freeze record (`state/freeze.json`) holds these hashes. The single run checks them at its start and its end, and
stops if any differs:

- `roll`'s code hash (`src/roll` with `hypo`'s code hash, which covers the perisim code it uses) and the hash of
  `config/roll.yaml`;
- `hypo`'s freeze, by its own check:
  - code, configuration and folds;
  - the Phase 2 comparators;
  - the three transported metrics `metric_ext-<X>.pkl`;
  - the transported comparators `comp_ext-<X>_{logreg,lightgbm}.pkl`;
- T2's freeze (code and configuration hashes) and its `final_model.pkl`;
- the digest of every stream of the design. There are 1,190 stream-orders over every stream, hospital, N and order;
  the same patients will come in the same order;
- the git tag `perisim-freeze-roll-<time>`, pushed to GitHub before the run, and the SHA-256 of this document.

## 5. Streams and the loop

| stream | role | order | starting window | metric | N (primary) | R | W | M |
|---|---|---|---|---|---|---|---|---|
| MIMIC-IV temporal | confirmatory | blocks in order; random inside | first N patients of 2008–2010 | ext-mimic | 250, **500**, 1,000, 2,000 | 5 | 100 | 300 |
| eICU (a), within each of 35 hospitals | confirmatory | 2014 → 2015; random inside | its first N of 2014 | ext-eicu | **100**; 250 where 2014 has ≥ 400 | 10 | 50 | 200 |
| eICU (b), deployment to each of 35 hospitals | confirmatory | 2014 → 2015; random inside | N random patients of the other hospitals | ext-eicu | 100, **250** | 10 | 50 | 200 |
| INSPIRE → MIMIC-IV | secondary | 2,000 then 4,000 | N INSPIRE patients | ext-mimic | as MIMIC-IV | 5 | 150 / 100 | 500 / 300 |
| eICU → MIMIC-IV | secondary | 2,000 then 4,000 | N eICU patients | ext-mimic | as MIMIC-IV | 5 | 100 | 300 |
| MIMIC-IV → INSPIRE | secondary | 2,000 then 4,000 | N MIMIC-IV patients | ext-inspire | as MIMIC-IV | 5 | 100 / 150 | 300 / 500 |
| INSPIRE control | control | random | first N patients | ext-inspire | 250, **500**, 1,000, 2,000 | 5 | 150 | 500 |

- **The loop** (design §5): test-then-train, B = 10 patients per step.
  - Every arm predicts a step's patients with its state before them; only then do they enter.
  - A step never crosses an evaluation window's end.
  - Each patient is evaluated once.
  - A patient's landmarks are never neighbours of each other.
  - Label delay does not apply (no real time inside a segment).
- **Evaluation windows** (design §8.1):
  - W consecutive evaluated patients inside a segment, with no window straddling a boundary;
  - a remainder below W/2 joins the segment's last window;
  - a window counts if it holds ≥ 30 event landmarks and ≥ 10 patients with an event.
- **Boundaries** (the real changes): the four MIMIC-IV block changes, each eICU hospital's change of year, stream
  (b)'s start, the change of cohort. **Segments** (the controls): inside a block, a year or a cohort.
- **External or internal**: the metric and the transported ML are external throughout the primary streams and the
  control. In a cross-cohort stream they are internal in the first stretch and external in the second.

## 6. Arms

| # | arm | detail (design §6–§7) |
|---|---|---|
| 1 | **PERISIM-FIFO** | the frozen metric; the window of the last N patients, rotating every step; α fixed on the starting window by LOO |
| 2 | PERISIM cumulative | the case base only grows |
| 3 | PERISIM static | the starting window, fixed |
| 4a / 4b | ML frozen / frozen after go-live | `hypo`'s transported model as it is / after the go-live update only |
| 5 | **ML realistic** (primary comparator) | transported, go-live update on the starting window, monitor and primary trigger, update by closed testing, deployed after L = 3 months |
| 5′ | ML realistic, trained on the starting window | the same policy from a model trained on the starting window (reported, not tested) |
| 6 | ML ideal (ceiling, not realistic) | the update hierarchy every W patients, no trigger, no delay |
| 7 | prevalence | the window's rate (no discrimination metrics) |

- **ML families**: LightGBM (primary) and logistic regression (secondary), `hypo`'s transported comparators.
- **Update data**: the last N patients, the same patients as PERISIM-FIFO's window at that moment.
- **The update policy** (design §7):
  - Monitor: the last M patients since the deployment, extended back to E = 500 event landmarks. It is checked at
    the end of every window. The refractory period is M patients after each deployment, and there is no check during
    a delay.
  - Primary trigger: O/E outside [0.9, 1.1], or the patient-cluster robust 95 % CI of the calibration in the large
    entirely outside ±0.10.
  - Action: the hierarchy keep → recalibration in the large → logistic recalibration → refit. It is chosen by the
    closed test on the nested recalibrations (top-down), then the refit against the recalibration chosen, in a
    2 : 1 patient split of the window, with one-sided α = 0.05 on per-patient log-loss sums. No hyperparameter is
    re-tuned.
  - Delay: L = 3 months, in patients at the segment's admission rate (design §7.6); L = 0 as a scenario.
- **Sensitivity variants of arm 5**: the triggers AUROC drop ≥ 5 % or ≥ 10 %, log-loss rise ≥ 10 %, and Davis et al.
  (2020) reimplemented; and L = 0.

## 7. Hypotheses (confirmatory)

The comparator is arm 5, LightGBM, with the primary trigger and L = 3 months, at the stream's primary N. Every
contrast is PERISIM-FIFO minus arm 5, paired on the same windows.

| | estimand | rule |
|---|---|---|
| **H1** level | mean window AUROC; mean window AUPRC − prevalence | lower 95 % bound > −0.02 for the AUROC **and** > −0.03 for the AUPRC − prevalence |
| **H2** stability | excess SD (between-window variance minus mean within-window sampling variance) of the AUROC and of the AUPRC − prevalence, as a ratio FIFO ÷ arm 5, with a floor of (0.005)² added to both variances | upper 95 % bound of the ratio < 1, for both |
| **H3** low tail | 10th percentile of the window AUROC and of the window AUPRC | lower 95 % bound > 0, for both |
| **H4** calibration (co-primary) | (a) the excess calibration error, √max(0, mean CITL² − mean within-window variance of the CITL); (b) the share of checks in which the primary trigger fires, the rule computed on each arm's predictions at every check | upper 95 % bound < 0, for both |
| **H6** signal at boundaries (eICU (a) only) | (a) the out-of-support rate in the first window after the change of year minus its mean in the hospital's other windows; (b) the share of out-of-support alarms (one-sided 1 % test of the 1 % nominal rate) in those other windows | (a) pooled lower 95 % bound > 0; (b) pooled upper 95 % bound ≤ 5 % |

- **Where**: H1–H4 in the MIMIC-IV temporal stream, in eICU (a) and in eICU (b). H6 in eICU (a).
- **eICU**: each hospital's paired estimate and bootstrap SE, pooled by DerSimonian–Laird random effects (H2 on the
  log of the ratio). Reported with τ², I² and a forest plot. H6 (b) is pooled as the mean over hospitals with a
  hospital bootstrap.
- **H6 (c), reported against PERISIM**: the same quantities for the ML monitor at a matched nominal level, a test of
  CITL = 0 per window at two-sided 1 %.
- **CIs**: 95 %, from a moving-block bootstrap of the windows with 2,000 resamples:
  - stratified by segment;
  - block length ⌈N / W⌉ windows, at most half a segment's windows;
  - the same resampled positions for every arm and every order;
  - each statistic averaged over the R orders inside every resample.

  The within-window variances come from 200 patient-level resamples per window, the same for every arm.
- **Multiplicity**: none across hypotheses. Inside a hypothesis, every endpoint and every confirmatory stream must
  meet its rule for the hypothesis to hold overall, an intersection–union test that needs no adjustment (Berger,
  1982). Each stream's result is also reported on its own.
- **Reported with the same rules, not decided**:
  - against arm 6 (the ceiling), with the margins applied both ways in the discussion;
  - against logistic regression;
  - against arm 5′.

## 8. Secondary analyses (estimates with CIs, no decision rule)

- **H5, recovery**: the patients from a boundary to the first two consecutive windows within δ (AUROC 0.02; CITL
  0.10) of the arm's stable level of the next segment, which is its mean over the second half of that segment.
  FIFO against arms 5 and 6, in the cross-cohort streams, at eICU (b)'s start and at the MIMIC-IV boundaries.
- **H7, the window**: every endpoint against N (the N grid of §5); the smallest N that meets H1's and H4's rules,
  per stream type.
- **By stretch**: every endpoint in the control windows and in the post-boundary windows. These are the first
  ⌈N / W⌉ windows of a segment after a boundary, including stream (b)'s start.
- **Saw-tooth** (arms 5 and 6): per cycle between deployments, the window after the first minus the window before the
  next; the number of cycles and the mean amplitude, for AUROC and CITL.
- **The update logs**: triggers, deployments, the choices of the closed test, parameters changed, compute time.
- **PERISIM's diagnostics**:
  - D1, the out-of-support rate per window, in every stream;
  - D2, the re-deformation gain (`hypo`'s S4 "all variables" fit on the FIFO window, scored on the next window), in
    the MIMIC-IV, cross-cohort and INSPIRE streams, first order only;
  - the trigger to re-learn the metric (logged, never executed): D1 ≥ 3 % in 3 consecutive windows, or a D2 gain
    ≤ −0.005 nats with its CI below 0.
- **The cross-cohort streams and the INSPIRE control**: every endpoint and contrast of §7, as estimates.
- **The ML monitor's PSI**, as a reference. At every check, for each continuous input, it compares the window
  against the data that chose arm 5's deployed model, using that data's deciles and a bin for missing values. The
  mean and the maximum over inputs are reported.
- **Governability, measured**:
  - determinism: identical prediction hashes in two runs, with 1 and 8 threads;
  - the parameters changed and the time of every update;
  - the reconstruction of any window state from the stream log.
- **T2** (design §12): the MIMIC-IV temporal stream with T2's frozen method.
  - N = 2,000, W = 250 patients; R = 5.
  - Arms: PERISIM-FIFO, cumulative, static, the window's class shares.
  - Per window: the 4-class log-loss and the ranked probability score. Paired differences with the same bootstrap.
- **Sensitivities** (reported, never chosen):
  - H1–H4 with each sensitivity trigger, and with L = 0;
  - by stretch;
  - against logistic regression, and against arm 5′;
  - with W halved and doubled. These are the same predictions read in windows of W/2 and 2W: the arms' runs, checks
    and updates are unchanged, and H4 (b) is not re-read.

## 9. Rules for the stage

- **Running once.** The single run refuses to start without the freeze record, or when anything it covers differs
  (§4); it checks again at the end. It is resumable: a finished unit is kept, a failed one is rerun with the same
  frozen code.
- **No iteration.** Nothing is tuned on the results. No stream, hospital, boundary, N or window is added or dropped
  after the results are seen. A change to the code after results exist is a deviation: it is reported as one, and
  the original output is kept. Everything after the run that is not in this document is post hoc and labelled so.
- **Windows dropped** (fewer than 30 event landmarks or 10 patients with an event) are dropped for every arm, and
  counted.
- **Small cells.** Aggregates hold windows of ≥ 50 patients; nothing below 10 is exported. eICU hospitals are
  labelled H1, H2, … by size, never by their identifier. Patient-level predictions and logs never leave the
  workstation.

## 10. Expectations, declared before the run

- **H1 is expected to fail against LightGBM** in every confirmatory stream at the primary N. The transported LightGBM's
  AUROC is 0.788–0.810. PERISIM's with N random patients is:
  - 0.771 at 500 in MIMIC-IV;
  - 0.770 at 100 and 0.787 at 250 in eICU.

  The expected differences are −0.018 (MIMIC-IV) and −0.040 / −0.023 (eICU (a) / (b)). Against logistic regression it
  is expected to hold in MIMIC-IV and INSPIRE at N = 500. LightGBM, the margin and N are kept.
- **Little drift is expected in the primary streams.** `hypo` found the metric transportable and the hospital effect
  small. Prevalence changes little across MIMIC-IV's blocks (32.0 → 25.6 %), and by a median of −0.7 points across
  eICU's change of year. Differences, if any, are expected in the cross-cohort streams and in calibration rather
  than discrimination.
- **In eICU (a) (N = 100)**, PERISIM's calibration in the large is expected to vary from window to window with the
  window's patients (§3). This counts against PERISIM in H4 (a).
- **The predicted null is publishable.** If the ML arms neither degrade nor trigger inside the blocks and the
  hospitals, and H2–H4 show no difference, the result is "not enough drift in these data to tell the arms apart".
- **The cycle, not the drift.** If PERISIM wins H2–H4 only with L = 3 months, and loses with L = 0 or with the more
  sensitive triggers, the conclusion is "PERISIM removes the cost of the cycle of monitoring, decision and
  deployment", not "PERISIM resists drift better".

## 11. Known limitations, declared in advance

- Only three cohorts, all already examined with this outcome. No validation on new data.
- Real time order only by blocks: five in MIMIC-IV, two years per eICU hospital, none in INSPIRE. Random order inside
  them.
- Few boundaries per stream: the measurable real drift is small.
- MIMIC-IV blocks are by patient, and 16 % of stays fall ≥ 3 years after the anchor year (blurred boundaries).
  2020–2022 is the COVID era.
- eICU hospitals are small: windows of 50 patients, and a monitor that fires on noise in about 1 check in 10 at
  M = 200.
- Changes of hospital and of cohort mix population, practice and measurement.
- The deployment delay (an assumption anchored to the FDA's 90-day 510(k) review goal) and the ML triggers are
  reasoned, not observed.
- The closed test is adapted (held-out comparisons, refit tested last), and the Davis et al. detector is
  reimplemented.
- Performative prediction: the outcomes in a case base may have been treated. Label delay does not apply here.
- R orders randomise the order inside segments only. The CIs do not include the choice of the starting window
  beyond those orders.

## 12. Registration

Proposed: a local registration ID, `LOCAL-PREREG-ROLL-<date>`, as for `hypo` (D6, 2026-10-29). Filing on OSF is
the lead's action. The freeze tag and commit, pushed to GitHub before the run, are the verifiable timestamp.
