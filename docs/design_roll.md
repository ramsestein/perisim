# Experiment `roll`: prequential evaluation with a rotating case base — design

Status: **Phase 1 draft, 2026-10-30; waiting for the lead's decisions (gate ⛔, §16).** Branch `roll`. Phase 0:
[feasibility_roll.md](feasibility_roll.md), gate decided on 2026-10-30 (§3). New measurements for this design:
`python scripts/feasibility_roll.py phase1` (T2's counts and the noise of the ML monitor; aggregates only, in
`work/experiments/roll/feasibility/feasibility_roll_phase1.json`).

`roll` is a **pre-registered retrospective evaluation on data already examined**. It is not a prospective validation,
and no data are new. The three cohorts (INSPIRE, MIMIC-IV, eICU) were used with this outcome in `hypo`, whose
per-site and per-hospital aggregates have been seen. What has not been seen is the performance along a prequential
stream with a rotating case base. The design below uses `hypo`'s numbers openly, and says where.

## 1. Question

The thesis put to the test: **"CE-kNN is not less accurate than ML managed as a real hospital manages it, and it is
more governable."** More governable means:

1. deterministic inference, reproducible by hash;
2. an update without retraining: new patients enter the case base and the oldest leave (FIFO), with no cycle of
   monitoring, decision, revalidation and deployment;
3. stable performance along the stream, without the saw-tooth of a model that degrades between updates, and a quick
   recovery after a change;
4. diagnostics (out of support, re-deformation of the shapes) that flag the real changes of the stream.

It is not an attempt to show that PERISIM predicts better than LightGBM. It is designed so that it can fail, and
`hypo`'s numbers say that the accuracy half is expected to fail against LightGBM at the brief's window sizes (§9.2).

## 2. Background: kNN with a rotating memory

- **Prequential evaluation** (test-then-train; Gama et al., 2014): each example is predicted, then learned from. A
  FIFO sliding window is the simplest form of forgetting.
- **SAM-kNN** (Losing et al., 2016): a kNN whose memory is split into a short-term memory (the recent window, its size
  adapted to the interleaved test-then-train error) and a long-term memory of past concepts, kept consistent with
  the short-term one. The distance is fixed (Euclidean); the memory is the model.
- **IBLStreams** (Shaker & Hüllermeier, 2012): instance-based classification and regression on streams. The case base
  is edited as examples arrive: new ones are added, and redundant, noisy or outdated ones are removed, judged by their
  neighbourhoods.

**How PERISIM differs.**

| | SAM-kNN, IBLStreams | PERISIM in `roll` |
|---|---|---|
| distance | fixed, or tuned on the stream | learned once on other cohorts, in effect space (shapes g_j, weights w_j); frozen |
| memory | the whole model | a local case base only; the metric is separate from it |
| what adapts | memory size and content | memory content only (FIFO); no parameter is refitted |
| when the distance expires | not separable from a memory change | measured: out-of-support rate, re-deformation gain (§8.6) |

## 3. The Phase 0 gate, as decided (lead, 2026-10-30)

| # | decision |
|---|---|
| 1 | MIMIC-IV with **five** `anchor_year_group` blocks, 2008–2010 to 2020–2022: four boundaries |
| 2 | eICU hospitals with **≥ 150 patients in each of 2014 and 2015**: 35 hospitals, 21,169 patients |
| 3 | INSPIRE as a random-order control and as a starting cohort |
| 4 | the sensitivity analysis ordering eICU by `patientunitstayid` is dropped (the id carries no calendar order) |
| 5 | cross-cohort pairs INSPIRE → MIMIC-IV, eICU → MIMIC-IV, MIMIC-IV → INSPIRE |
| 6 | confirmatory: H1–H4 in MIMIC-IV and in eICU's hospitals (meta-analysis); H6 in eICU only; cross-cohort secondary |

## 4. Streams

### 4.1 The streams

| stream | role | order | starting window | evaluated | metric (frozen, `hypo`) | N grid (primary) | R |
|---|---|---|---|---|---|---|---|
| **MIMIC-IV temporal** | primary | blocks in calendar order; random within a block | first N patients of 2008–2010 | the rest, 2008 → 2022 | ext-mimic | 250, **500**, 1,000, 2,000 | 5 |
| **eICU (a), within a hospital** | primary | 2014 → 2015; random within a year | its first N patients of 2014 | its other patients | ext-eicu | **100**; 250 where 2014 has ≥ 400 patients | 10 |
| **eICU (b), deployment** | primary | 2014 → 2015; random within a year | N random patients of the other 195 hospitals | all of its patients | ext-eicu | 100, **250** | 10 |
| INSPIRE → MIMIC-IV | secondary (stress) | 2,000 INSPIRE, then 4,000 MIMIC-IV; random within each | N INSPIRE patients | 6,000 | ext-mimic | 250, **500**, 1,000, 2,000 | 5 |
| eICU → MIMIC-IV | secondary (stress) | 2,000 eICU, then 4,000 MIMIC-IV | N eICU patients | 6,000 | ext-mimic | as above | 5 |
| MIMIC-IV → INSPIRE | secondary (stress) | 2,000 MIMIC-IV, then 4,000 INSPIRE | N MIMIC-IV patients | 6,000 | ext-inspire | as above | 5 |
| **INSPIRE control** | control | random | first N patients | the rest (11,131) | ext-inspire | 250, **500**, 1,000, 2,000 | 5 |

- **R** is the number of random orders within each block, year or cohort. Each order uses all arms. The endpoints are
  averaged over the R orders (§10).
- **The 35 eICU hospitals** are those of the gate. In stream (b) the other hospitals' patients are drawn anew in each
  order. The hospital's own patients never enter its starting window.
- **The cross-cohort stretches** are drawn at random from each cohort, anew in each order. 4,000 patients is 8 times
  the primary window: enough for any FIFO window of the grid to turn over.

### 4.2 Where the metric is external

| stream | first stretch | second stretch |
|---|---|---|
| MIMIC-IV temporal, eICU (a) and (b), INSPIRE control | external throughout | — |
| INSPIRE → MIMIC-IV, eICU → MIMIC-IV | internal (INSPIRE or eICU was in the development of ext-mimic) | external |
| MIMIC-IV → INSPIRE | internal | external |

The transported ML arms were trained on the same two cohorts as each metric (`hypo`), so the same stretches are
internal and external for them.

### 4.3 Segments and boundaries

- **Segments** are the stretches with no known drift inside: a MIMIC-IV block, an eICU year (within one hospital), a
  cohort's stretch, the whole INSPIRE stream. They are the controls.
- **Boundaries** are the real changes: the four MIMIC-IV block changes, the change of year in each eICU hospital,
  the change from other hospitals to the hospital (stream (b), at its start), the change of cohort.
- **Blurred boundaries, declared.** A MIMIC-IV block is the patient's `anchor_year_group`. 16 % of stays fall 3 or
  more years after the anchor year (Phase 0). This biases boundary effects towards zero.

## 5. The prequential loop

- **Unit: the patient.** Windows, steps and delays are counted in patients. A patient enters with all their
  landmarks (all their stays) at once, and only after being evaluated.
- **Start.** The starting window is the first N patients of the stream (§4.1).
- **Step.** The next B = 10 patients are predicted by every arm, with the state each arm had before them. They then
  enter, and in the FIFO arm the 10 oldest leave. Repeat to the end. Each patient is evaluated once, before any arm
  has seen them.
- **Exclusion by patient.** A patient's landmarks are never neighbours of each other. Neighbours are distinct
  patients, as in `hypo`.
- **Label.** No stream has real time inside a segment, so a patient's outcomes are known when they enter. Label
  delay does not apply, and this is declared (Phase 0).
- **Admission gate.** Before a patient enters any arm's data (case base or update data): duplicates (stay, L),
  values outside `hypo`'s bounds, impossible values. Refused landmarks are still predicted, but they enter no arm.
  The gate is part of the method, the same for every arm, and reported. It refused nothing in Phase 0.
- **Same sequence for every arm.** One stream log (seeded) fixes the order. Every arm reads it.

## 6. Arms

| # | arm | what changes along the stream | data of each change |
|---|---|---|---|
| 1 | **PERISIM-FIFO** | the case base: the window of the last N patients, rotating every B | the window |
| 2 | PERISIM cumulative | the case base only grows | every patient so far |
| 3 | PERISIM static | nothing: the starting window | — |
| 4a | ML frozen | nothing: `hypo`'s transported model as it is | — |
| 4b | ML frozen after go-live | the go-live update only (§7.1) | the starting window |
| 5 | **ML realistic** (primary comparator) | updates when the monitor triggers, deployed L patients later (§7) | the last N patients at the trigger |
| 5′ | ML realistic, trained on the starting window | as arm 5, from a model trained on the starting window | as arm 5 |
| 6 | ML ideal (ceiling, not realistic) | the update hierarchy every W patients, no trigger, no delay | the last N patients |
| 7 | prevalence | the window's rate | the last N patients |

### 6.1 PERISIM arms

- **Metric**: `hypo`'s frozen metric of the rotation that excluded the evaluated cohort (`final/metric_ext-<X>.pkl`):
  g_j, w_j, encoder, k = 50. No metric is learned. Nothing in the metric changes during a stream.
- **α is fixed** at its leave-one-out value on the starting window, in every PERISIM arm (§14, point 1). The
  out-of-support threshold is recomputed at every evaluation window (a diagnostic; it changes no prediction).
- **The vote's prior P(c) is the case base's prevalence**, as in `hypo`. It therefore follows the window. This is
  part of the method, not a fitted parameter, and is declared.

### 6.2 ML arms

- **Families**: LightGBM and L2 logistic regression, `hypo`'s transported comparators (`final/comp_ext-<X>_<name>.pkl`,
  trained on all patients of the two other cohorts).
- **LightGBM is the primary comparator** of arms 5 and 6: it is the strongest model of `hypo` (§9.2). Logistic
  regression is reported with the same rules, as secondary.
- **Arm 5 is transported in every stream**, as the brief sets for eICU. This mirrors PERISIM: a transported model
  (metric), local data (case base). Arm 5′ starts from LightGBM / logistic regression trained on the starting window
  only (`hypo`'s settings: T2's grid, inner folds); it is reported, not tested.
- **Update window**: an ML update uses the last N patients, the same patients as PERISIM-FIFO's window at that moment.
  N is the stream's N, so each N of the grid has its own ML arms.

## 7. Update policy of the realistic ML arm ⛔

In a hospital, a model is not retrained on a fixed schedule. It is monitored, and updated when it fails. This
section models that.

### 7.1 Go-live

Before the stream starts, arm 5 runs the update hierarchy (§7.5) on the starting window. This is the local validation
a hospital does before go-live. Without it, the transported models start with calibration in the large of −0.40 to
+0.50 (`hypo` S2), which would hand PERISIM the first stretch of every stream (§14, point 2). The go-live model is live
from the first evaluated patient; L does not apply to it.

### 7.2 Monitor

- **Window**: the last M evaluated patients, extended back until it holds at least E = 500 event landmarks. Only
  patients evaluated since the current model's deployment count.
- **Cadence**: one check at the end of every evaluation window (every W patients, §8.1).
- **Refractory period**: no check until M patients have been evaluated since the last deployment. During the delay L
  (§7.6) there is no check either: an update is in progress.
- **Baseline** (for the relative triggers): the deployed model's AUROC and log-loss on the validation part of the
  window that chose it (§7.5).

**M by stream type**, from the noise of the primary trigger. Random monitoring windows of M patients from `hypo`'s
transported PERISIM predictions, which are calibrated in every site, so there is no drift by construction (300 draws
per cell; `feasibility_roll.py phase1`):

| M (patients) | MIMIC-IV: event landmarks / O/E SD / fires | eICU | INSPIRE |
|---|---|---|---|
| 100 | 485 / 0.066 / 14.7 % | 282 / 0.084 / 28.3 % | 168 / 0.119 / 38.3 % |
| 200 | 968 / 0.039 / 0.7 % | 563 / 0.060 / 9.7 % | 338 / 0.079 / 21.3 % |
| 300 | 1,457 / 0.035 / 0.7 % | 844 / 0.049 / 4.7 % | 509 / 0.066 / 17.7 % |
| 500 | 2,423 / 0.028 / 0.3 % | 1,409 / 0.038 / 1.3 % | 844 / 0.048 / 6.0 % |
| 1,000 | 4,854 / 0.018 / 0.0 % | 2,821 / 0.025 / 0.0 % | 1,675 / 0.037 / 1.0 % |

"Fires" is the share of checks in which the primary trigger (§7.3) fires on noise alone. Almost all of it comes from
the O/E rule. The CITL rule fires in at most 3.3 % of checks at M = 50, 0.3 % at M = 200, and in none from M = 300
on.

| stream type | M proposed | fires on noise, per check | in time |
|---|---|---|---|
| MIMIC-IV (temporal; the MIMIC-IV stretch of a cross-cohort stream) | **300** | 0.7 % | 1.6 months (2008–2010) to 4.2 months (2020–2022) |
| eICU hospital | **200** | 9.7 % | 8.7 months in a median hospital (277 patients a year) |
| INSPIRE (control; the INSPIRE stretch) | **500** | 6.0 % | 5.2 months (about 97 patients a month) |
| eICU pooled (the eICU stretch of eICU → MIMIC-IV) | **300** | 4.7 % | — |

**A single hospital cannot monitor calibration precisely.** In an eICU hospital, M = 200 is most of a year, and the
rule still fires on noise in 1 check in 10. A larger M would leave no check inside a two-year stream. This is the
real limit of monitoring a small hospital, and is reported as such. A false trigger leads to an update chosen by
closed testing, which keeps the current model when the new data do not improve it (§7.5). It harms neither arm.

### 7.3 Primary trigger: calibration

The trigger fires when **O/E is outside [0.9, 1.1]**, or when **the 95 % CI of the calibration in the large lies
entirely outside ±0.10**. The CI is patient-cluster robust. Calibration drifts while discrimination holds (Davis et
al., 2017), and a calibration trigger is what Davis et al. (2020) propose to inform updating.

**Reading of the CITL rule.** "The CI outside ±0.10" is read as "the whole CI beyond the band": evidence of
miscalibration beyond the tolerance. The other reading, "the CI not inside the band", fires even for a calibrated
model: in 100 % of the checks at M ≤ 300 in all three cohorts, and in 41–96 % at M = 1,000. That reading cannot be
used.

### 7.4 Sensitivity triggers

| trigger | fires on noise at the proposed M (MIMIC-IV / eICU / INSPIRE) |
|---|---|
| (a) relative AUROC drop ≥ 5 % against the baseline | 0 % / 1.3 % / 0 % |
| (a) relative AUROC drop ≥ 10 % | 0 % / 0 % / 0 % |
| (b) relative log-loss rise ≥ 10 % | 0 % / 3.3 % / 1.3 % |
| (c) Davis et al. (2020): dynamic calibration curve updated online (Adam), adaptive window (ADWIN; Bifet & Gavaldà, 2007) on the calibration error, one-sided, δ = 0.05 | measured in the Phase 2 dry run |

- (b) also fires when the prevalence changes, with no miscalibration: log-loss rises with the base rate. This is
  expected in the cross-cohort streams and declared.
- (c) is implementable. It is our reimplementation from the paper (fractional-polynomial calibration curve, the
  paper's defaults; minimum window M). No code from the authors is used.

### 7.5 Action: the update hierarchy, chosen by closed testing

The candidates (Janssen et al., 2008):

0. keep the current model;
1. recalibration in the large (intercept; logit p as offset);
2. logistic recalibration (intercept and slope);
3. full refit: logistic regression re-estimated with its C; LightGBM with the transported hyperparameters, trees by
   early stopping on the fitting part, then temperature scaling (as `hypo`).

The choice follows the closed testing procedure of Vergouwe et al. (2017), in an internal split of the update window:

- the patients of the window are split 2 : 1 (fitting : validation) by a seeded hash of the patient;
- each candidate is fitted on the fitting part and scored on the validation part (log-loss);
- the tests compare per-patient sums of log-loss differences, one-sided, α = 0.05;
- test 3 against 0: if 3 is not better, keep 0 (stop). Otherwise test 3 against 1: if not better, choose 1.
  Otherwise test 3 against 2: if not better, choose 2; else choose 3.
- The chosen candidate is refitted on the whole window.

**Adaptation, declared.** Vergouwe et al. use likelihood-ratio tests on the update data. A held-out comparison is
used here instead, for two reasons: LightGBM's refit is not nested in its recalibrations, and the landmarks of a
patient are not independent. No hyperparameter is re-tuned at an update.

### 7.6 Deployment delay L

- **L = 0** (optimistic) and **L = 3 months** (realistic, primary): the new model enters service L patients after
  the trigger, and the previous model predicts until then.
- **The source of the 3 months.** No published measurement of the delay from a monitoring alarm to an updated model
  in service is known to us (searched on 2026-10-30). Three months is a declared, reasoned assumption. Its order of
  magnitude is the FDA's 90-day review goal for a 510(k) premarket notification, for a regulated change. It also
  matches a quarterly governance cycle for a local model. It is not an observed value.
- **In patients**, by the admission rate of the segment where the trigger fires:

  | where | patients per month | L = 3 months |
  |---|---|---|
  | MIMIC-IV 2008–2010 / 2011–2013 / 2014–2016 / 2017–2019 / 2020–2022 | 193 / 165 / 149 / 102 / 71 | 579 / 494 / 446 / 307 / 213 |
  | eICU hospital | its year's patients ÷ 12 | 38–221 (median 69) |
  | INSPIRE (2011–2020, 120 months) | 97 | 291 |

### 7.7 Log

For every trigger: the step, the rule that fired, the candidate chosen, the delay applied, and the performance in the
monitoring window before and after deployment. For every update of any arm: §11.

### 7.8 Symmetry check

PERISIM has no trigger because it does not need one. At every check, the same primary rule is also computed on
PERISIM-FIFO's predictions. This shows whether its calibration holds without intervention (H4b).

## 8. Measures and endpoints

### 8.1 Evaluation windows

- **Non-overlapping windows of W consecutive evaluated patients**, formed inside each segment. A remainder below W/2
  joins the segment's last window; otherwise it is a window of its own. No window straddles a boundary. The same
  windows for every arm. Overlapping moving windows are for the figures only.
- **A window counts** if it holds at least 30 event landmarks and 10 patients with an event. Otherwise it is
  dropped for every arm, and the number dropped is reported.

| stream type | W (patients) | AUROC ±95 % in a window (Phase 0) | windows |
|---|---|---|---|
| MIMIC-IV (and its cross-cohort stretches) | **100** | ±0.046 | about 240 (temporal stream, N = 500) |
| eICU hospital | **50** | ±0.069 | about 9 per hospital in (a), 11 in (b) |
| eICU pooled stretch | 100 | ±0.046 | 20 |
| INSPIRE (and its stretches) | **150** | ±0.050 | about 74 (control stream) |

### 8.2 Per window

AUROC; AUPRC; normalised AUPRC (AUPRC − window prevalence, and AUPRC ÷ window prevalence); calibration in the large
(CITL); O/E; calibration slope; log-loss; Brier. Per landmark, as `hypo`; the patient is the resampling unit.

### 8.3 Summary endpoints, per arm and stream

- **(i) Level**: mean, median and SD over windows of the AUROC, the AUPRC and the normalised AUPRC.
- **(ii) Excess SD**: the variance between windows minus the mean sampling variance within a window (200
  patient-level bootstrap resamples per window), as a square root, floored at 0. It is the variability due to the
  model, not to the window's size.
- **(iii) Low tail**: the 10th percentile of the window AUROC and AUPRC. Also the share of windows below a clinical
  threshold (AUROC 0.70) and below a local reference minus δ_AUROC (the reference is `hypo`'s local LightGBM AUROC of
  the cohort, out of fold: INSPIRE 0.802, MIMIC-IV 0.797, eICU 0.814).
- **(iv) Calibration by window**: mean and SD of the CITL and O/E, and the share of windows with CITL outside ±0.10
  or O/E outside [0.9, 1.1].

  **This share is dominated by noise at these W.** A calibrated model has a window CITL SD of 0.118 at W = 100
  (MIMIC-IV), so about 40 % of its windows fall outside ±0.10 by chance; in an eICU hospital (W = 50, SD 0.195), about
  60 %. So the confirmatory calibration endpoint is proposed as the **excess calibration error**:
  sqrt(max(0, mean CITL² − mean within-window variance of the CITL)), with the same 200 resamples as (ii). It is 0
  for a model that is calibrated in every window, and it measures the systematic and drifting miscalibration. The
  share outside ±0.10 is reported next to it, with its noise floor.
- **Saw-tooth** (arms 5 and 6): for each cycle between two deployments, the drop from the window after the first
  deployment to the window before the next. The number of cycles and the mean amplitude, for AUROC and CITL.
- **By stretch**: every endpoint on the whole stream, inside the segments (controls), and in the windows after each
  boundary (the first ⌈N / W⌉ windows, the time the FIFO window needs to turn over).

### 8.4 Recovery

The number of evaluated patients from a boundary to the start of the first pair of consecutive windows that are both
within δ of the arm's stable level of the next segment. The stable level is the arm's mean over the second half of
that segment. δ is 0.02 for AUROC and 0.10 for CITL.

### 8.5 Predictions

Every prediction is stored locally with its step, arm and order. Never exported.

### 8.6 PERISIM's diagnostics, and the trigger to re-learn the metric

| diagnostic | definition | cadence |
|---|---|---|
| **D1, out-of-support rate** | share of the evaluated landmarks whose k-th neighbour is beyond the window's threshold (99th percentile of the LOO k-th distance in the current window); nominal 1 % | every evaluation window, every stream |
| D1 alarm | a one-sided test (patient-cluster robust, 1 %) of the window's rate against the nominal 1 % | every evaluation window |
| **D2, re-deformation gain** | `hypo`'s S4: every g_j refitted to the current window; gain in log-loss on the next W evaluated patients, with a 95 % CI | every 10 evaluation windows and at the first window after each boundary; MIMIC-IV, INSPIRE and cross-cohort streams; first order only (cost) |

- **D2 is not computed in eICU hospitals.** Their windows (100–250 patients) are far below what S4 needed in `hypo`
  (thousands of landmarks); a gain there would be noise.
- **The trigger to re-learn the metric** is logged, never executed. It fires when D1's rate is ≥ 3 % (three times the
  nominal) in 3 consecutive windows, or when D2's gain is ≤ −0.005 nats with its CI below 0. For scale: the MIMIC-IV
  transport cost was 0.002–0.003 nats in `hypo`, and a case base going from 500 to 1,000 patients gains about 0.003.

### 8.7 The ML monitor's own diagnostics

The monitor of §7.2, plus the PSI of each input between the deployed model's update window and the monitoring window,
as a reference. The PSI decides nothing.

## 9. Hypotheses and margins (proposed) ⛔

### 9.1 Hypotheses

Paired differences between arms, on the same windows. The comparator is arm 5 with LightGBM, L = 3 months and the
primary trigger, unless stated.

| | statistic | rule | where (confirmatory) |
|---|---|---|---|
| **H1 level** | mean window AUROC, FIFO − arm 5; the same for the AUPRC − prevalence | lower 95 % bound > −δ_AUROC = −0.02 and > −δ_AUPRC = −0.03 (both) | MIMIC-IV; eICU (a); eICU (b) |
| **H2 stability** | excess SD of the AUROC and of the AUPRC − prevalence, FIFO ÷ arm 5 | upper 95 % bound of the ratio < 1 (both) | as H1 |
| **H3 low tail** | 10th percentile of the window AUROC and AUPRC, FIFO − arm 5 | lower 95 % bound > 0 (both) | as H1 |
| **H4 calibration** | (a) excess calibration error, FIFO − arm 5; (b) share of checks where the primary trigger fires, FIFO − arm 5 | upper 95 % bound < 0 (both) | as H1 |
| H5 recovery | patients to recover after a boundary (§8.4), FIFO − arm 5 and FIFO − arm 6 | estimate with CI; no decision | secondary: cross-cohort, eICU (b) start, MIMIC-IV boundaries |
| **H6 signal at boundaries** | (a) D1 rate in the first window after the change of year − its mean in the hospital's within-year windows; (b) share of D1 alarms in within-year windows; (c) (a) and (b) for the ML monitor at the same nominal level | (a) pooled lower bound > 0; (b) pooled upper bound ≤ 5 %; (c) reported against PERISIM | eICU (a) |
| H7 window | every endpoint against N; the smallest N that meets H1 and H4 | descriptive | every stream type |

- **AUPRC − prevalence.** Both arms share a window's prevalence, so a paired difference of AUPRC − prevalence equals
  the paired difference of AUPRC. The normalisation matters for H2, where the prevalence's variation between
  windows would inflate both SDs. AUPRC ÷ prevalence is reported.
- **H2's floor.** If neither arm has excess variability, the ratio is 0 ÷ 0. A floor of (0.005)² is added to both
  excess variances: an excess SD of 0.005 AUROC is taken as negligible.
- **H6 in MIMIC-IV** (four boundaries in a single stream) and in the other streams is descriptive (gate, decision 6).
- **H6 (c), matched level.** The ML monitor's trigger is a tolerance rule, not a test, and fires on noise in 1 check
  in 10 in an eICU hospital (§7.2). Comparing D1's 1 % test with that rule would favour PERISIM by construction. So
  (c) compares D1 with a test of the ML's CITL = 0 at the same nominal level (two-sided 1 %).

### 9.2 What `hypo`'s numbers predict

The AUROC of PERISIM with a random case base of N patients (`hypo` S1) against the transported comparators (`hypo`
S2), whose AUROC no recalibration changes:

| stream | N | PERISIM | LightGBM transported | logistic transported | Δ against LightGBM / logistic |
|---|---|---|---|---|---|
| MIMIC-IV | 500 | 0.771 | 0.789 | 0.781 | −0.018 / −0.010 |
| MIMIC-IV | 2,000 | 0.777 | 0.789 | 0.781 | −0.012 / −0.004 |
| eICU (a) | 100 | 0.770 | 0.810 | 0.800 | −0.040 / −0.030 |
| eICU (b) | 250 | 0.787 | 0.810 | 0.800 | −0.023 / −0.013 |
| INSPIRE | 500 | 0.769 | 0.788 | 0.774 | −0.019 / −0.005 |

- **H1 is expected to fail against LightGBM in every confirmatory stream at the primary N.** In MIMIC-IV the expected
  difference (−0.018) sits at the margin, and its lower bound will be below it. In eICU (a) (N = 100) it fails by
  twice the margin. Against logistic regression it is expected to hold in MIMIC-IV and INSPIRE at N = 500, to be
  close in eICU (b) and to fail in eICU (a).
- A hospital's own patients make a slightly better case base than random eICU patients (`hypo` H4), so these
  eICU values are, if anything, a little pessimistic for PERISIM.
- **This is declared before the freeze, and it is not a reason to change the comparator or the margin.** The
  proposal keeps LightGBM, δ_AUROC = 0.02 and the brief's N. H7 says at which N the accuracy half would hold.
- The AUPRC margin rests on `hypo`'s scale: AUPRC differences between models were about twice the AUROC differences
  (INSPIRE: PERISIM 0.379, LightGBM transported 0.388, logistic transported 0.362; the other sites are being
  computed in `hypo`'s second post hoc round).

### 9.3 Readout rules, declared before running

- **The predicted null.** If the ML arms neither degrade nor trigger inside the blocks and the hospitals, and H2–H4
  show no differences, the result is reported as "not enough drift in these data to tell the arms apart". No
  stream, hospital or boundary is added after the results are seen.
- **The cycle, not the drift.** If PERISIM wins H2–H4 only with L = 3 months, and loses with L = 0 or with the more
  sensitive triggers, the conclusion is "PERISIM removes the cost of the cycle of monitoring, decision and
  deployment", not "PERISIM resists drift better".
- **Margins both ways.** In the discussion, the same margins are applied to differences in favour of either arm.
- **Sensitivities** (reported, never chosen): H1–H4 with each sensitivity trigger, with L = 0, with W halved and
  doubled, by stretch (segments against post-boundary windows), against logistic regression, against arm 5′. If a
  result depends on these choices, it is reported so.

## 10. Statistics

- **CIs**: 95 %, from a moving-block bootstrap of the evaluation windows (Künsch, 1989), 2,000 resamples, stratified
  by segment. The block length is ⌈N / W⌉ windows, the memory of the FIFO window (5 in MIMIC-IV, 2 in eICU (a), 5 in
  eICU (b), 4 in INSPIRE). The resampled windows are the same for every arm and every order: windows at the same
  position cover the same segment in all R orders.
- **Orders**: each statistic is the mean over the R orders, recomputed in every resample.
- **Within a window**: 200 patient-level bootstrap resamples, for the sampling variance of (ii) and (iv).
- **eICU**: each hospital gives its paired estimate and its bootstrap SE. They are pooled by a DerSimonian–Laird
  random-effects meta-analysis (DerSimonian & Laird, 1986): pooled effect with 95 % CI, τ², I², and a forest plot by
  hospital (labels H1, H2, … as `hypo`). H2 is pooled on the log of the ratio. With about 9–11 windows per hospital,
  each hospital's estimate is imprecise; the pooling is where the power is.
- **Multiplicity.** None across hypotheses (the brief). Inside a hypothesis, every endpoint and every confirmatory
  stream must meet its rule for the hypothesis to hold overall. This is an intersection–union test, which needs no
  adjustment (Berger, 1982). Each stream's result is also reported on its own, as `hypo` reported each site.

## 11. Governability, measured

- **Determinism**: two runs with the same metric and the same window state give bit-identical predictions (a hash of
  the float64 array). Repeated with 1 and 16 threads, and on another machine if one is available (the KHDP pod).
  The same test for the ML arms (LightGBM with a fixed seed).
- **Every update of every arm** logs: the parameters changed (PERISIM: 0; recalibration: 1 or 2; logistic refit: its
  coefficients; LightGBM refit: its trees and leaves), compute time, a hash of the window state or of the model, and
  the patients in and out.
- **Reconstruction**: the window state behind any prediction can be rebuilt from the stream log (order, step,
  patients in and out) and the metric's hash. This is tested.

## 12. T2 (secondary, descriptive)

T2's analysis stays, from its frozen targets (`feasibility_roll.py phase1`):

| | stays | patients | deaths (T2's fourth class) |
|---|---|---|---|
| INSPIRE | 12,663 | 11,808 | 116 |
| MIMIC-IV | 24,768 | 23,162 | 367 |
| … 2008–2010 / 2011–2013 / 2014–2016 / 2017–2019 / 2020–2022 | 7,980 / 5,701 / 5,141 / 3,524 / 2,422 | 7,173 / 5,302 / 4,894 / 3,433 / 2,360 | 137 / 74 / 91 / 37 / 28 |
| eICU, 2014 / 2015 | 11,684 / 12,016 | 11,408 / 11,743 | 108 / 131 |

- Of the 35 eICU hospitals of the gate, 25 have ≥ 150 T2 stays in each year (median 224 a year). T2 has about one
  death per 70 stays, so a window holds 1–4 deaths.
- **Proposal**: T2 in the MIMIC-IV temporal stream only (MIMIC-IV is external to T2's metric, which was learned on
  INSPIRE). Arms 1–3 and the window's class shares. N = 2,000 patients (T2 needed thousands: `hypo` design §2), W = 250
  patients. Labels: T2's frozen cross-fitted MIMIC-IV categories. Endpoints: the window's 4-class log-loss and
  ranked probability score. No ML arms (the update hierarchy is binary), no hypotheses. The exact frozen artefacts are
  named in Phase 2.

## 13. Cost (rough; measured in the Phase 2 dry run)

- **PERISIM**: a step is 10 patients (about 150 landmarks) queried against the window (about 8,000 landmarks at
  N = 500). Seconds per thousand steps. The cumulative arm queries up to 390,000 landmarks; minutes per order. D2 costs
  more and runs in one order only.
- **ML**: arm 5 updates a few times per stream. Arm 6 runs the hierarchy every W patients: about 240 LightGBM refits
  per MIMIC-IV order. It runs at the primary N only (the other N of the grid get arms 1–5 and 7).
- **Estimate**: 6–10 hours in all on the lead's workstation, run in the lead's terminal.

## 14. Fairness review: what could favour one side

| # | point | resolution |
|---|---|---|
| 1 | α re-chosen at every step would be an adjustment the ML arms do not get between updates | **α fixed** on the starting window; re-choosing it every W is a sensitivity |
| 2 | the transported ML starts miscalibrated (CITL −0.40 to +0.50) | **go-live update** on the starting window (§7.1), the same data PERISIM starts with |
| 3 | PERISIM's vote prior follows the window's prevalence, with no trigger | part of the method and of the thesis ("update without retraining"); declared |
| 4 | PERISIM rotates every B = 10 patients, the ideal ML every W = 100 | declared; sensitivity with the ideal arm every B in the cross-cohort streams (one order) |
| 5 | H6 against the ML's tolerance rule would compare a test with a noisy rule | H6 (c) at a matched nominal level (§9.1) |
| 6 | ML updates use the same data as PERISIM's window | symmetric by design (§6.2) |
| 7 | LightGBM, the strongest ML, is the comparator; H1 expected to fail | kept: the experiment is meant to be able to fail (§9.2) |
| 8 | the FIFO window forgets: under no drift it has less data than the cumulative arm | a real cost of FIFO; reported (arms 1 against 2) |
| 9 | the ML refit keeps the transported hyperparameters | declared; a refit on N patients rarely wins the closed test anyway |
| 10 | the 3-month delay is assumed, not observed | L = 0 reported next to it; readout rule of §9.3 |

## 15. Limitations, declared in advance

- Only three cohorts, all already examined with this outcome. No validation on new data.
- Real time order only by blocks: five blocks in MIMIC-IV, two years per eICU hospital, none in INSPIRE. Random order
  inside them.
- Few boundaries per stream, so the measurable real drift is small. The eICU change of year changes prevalence by a
  median of −0.7 points.
- MIMIC-IV blocks are by patient, and 16 % of stays fall 3 or more years after the anchor year (blurred boundaries).
  2020–2022 is the COVID era.
- eICU hospitals are small: windows of 50 patients, a monitor that cannot tell noise from drift (§7.2).
- Changes of hospital and of cohort mix population, practice and measurement (INSPIRE: non-invasive pressure only).
- The deployment delay and the ML triggers are reasoned assumptions, not observed.
- Performative prediction: the outcomes in a case base may have been treated.
- Label delay does not apply in these streams (no real time inside a segment); in a hospital it would.

## 16. Decisions for the lead (gate ⛔)

1. **Windows and monitor**: W = 100 / 50 / 150 (MIMIC-IV / eICU hospital / INSPIRE), M = 300 / 200 / 500, E = 500
   event landmarks, B = 10.
2. **Delay**: L = 0 and L = 3 months (source: an assumption anchored to the FDA's 90-day 510(k) review goal). Add
   6 months as a sensitivity?
3. **Triggers**: the primary rule with the CITL CI read as "entirely outside ±0.10" (§7.3); sensitivities (a)–(c).
4. **Closed testing** in a 2 : 1 internal split, adapted from Vergouwe et al. (§7.5).
5. **Comparator**: LightGBM primary, logistic regression secondary, both transported, with a go-live update.
6. **Margins**: δ_AUROC = 0.02, δ_AUPRC = 0.03; H2's floor (0.005)²; H6's levels (1 % test, ≤ 5 % false alarms).
7. **Calibration endpoint**: H4 (a) on the excess calibration error instead of the share outside ±0.10 (noise floor
   of 40–60 %). And H4 as **co-primary** (proposed) or key secondary?
8. **α fixed** on the starting window (proposed) or re-chosen every W.
9. **T2**: the MIMIC-IV stream only, descriptive (§12), or drop it.

**Risk.** H1 is expected to fail against LightGBM (§9.2), and drift is small in the primary streams, so H2–H4 may end
in the predicted null. The informative parts would then be the cross-cohort streams, H7 and the governability
measures. The eICU hospital estimates are imprecise one by one.

## References

DOIs and pages checked on Crossref on 2026-10-30.

- Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics, 24*(4), 295–300.
  https://doi.org/10.1080/00401706.1982.10487790
- Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing. In *Proceedings of the
  2007 SIAM International Conference on Data Mining* (pp. 443–448). SIAM. https://doi.org/10.1137/1.9781611972771.42
- Davis, S. E., Greevy, R. A., Jr., Lasko, T. A., Walsh, C. G., & Matheny, M. E. (2020). Detection of calibration
  drift in clinical prediction models to inform model updating. *Journal of Biomedical Informatics, 112*, 103611.
  https://doi.org/10.1016/j.jbi.2020.103611
- Davis, S. E., Lasko, T. A., Chen, G., Siew, E. D., & Matheny, M. E. (2017). Calibration drift in regression and
  machine learning models for acute kidney injury. *Journal of the American Medical Informatics Association, 24*(6),
  1052–1061. https://doi.org/10.1093/jamia/ocx030
- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials, 7*(3), 177–188.
  https://doi.org/10.1016/0197-2456(86)90046-2
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation.
  *ACM Computing Surveys, 46*(4), Article 44. https://doi.org/10.1145/2523813
- Janssen, K. J. M., Moons, K. G. M., Kalkman, C. J., Grobbee, D. E., & Vergouwe, Y. (2008). Updating methods improved
  the performance of a clinical prediction model in new patients. *Journal of Clinical Epidemiology, 61*(1), 76–86.
  https://doi.org/10.1016/j.jclinepi.2007.04.018
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of
  Statistics, 17*(3), 1217–1241. https://doi.org/10.1214/aos/1176347265
- Lim, L., Lee, H., Jung, C.-W., Sim, D., Borrat, X., Pollard, T. J., Celi, L. A., Mark, R. G., Vistisen, S. T., &
  Lee, H.-C. (2024). INSPIRE, a publicly available research dataset for perioperative medicine. *Scientific Data,
  11*, 655. https://doi.org/10.1038/s41597-024-03517-4
- Losing, V., Hammer, B., & Wersing, H. (2016). KNN classifier with self adjusting memory for heterogeneous concept
  drift. In *2016 IEEE 16th International Conference on Data Mining (ICDM)* (pp. 291–300). IEEE.
  https://doi.org/10.1109/ICDM.2016.0040
- Shaker, A., & Hüllermeier, E. (2012). IBLStreams: A system for instance-based classification and regression on data
  streams. *Evolving Systems, 3*(4), 235–249. https://doi.org/10.1007/s12530-012-9059-0
- U.S. Food and Drug Administration. (n.d.). *510(k) submission process*. Retrieved October 30, 2026, from
  https://www.fda.gov/medical-devices/premarket-notification-510k/510k-submission-process (the 90-FDA-day review goal)
- Vergouwe, Y., Nieboer, D., Oostenbrink, R., Debray, T. P. A., Murray, G. D., Kattan, M. W., Koffijberg, H.,
  Moons, K. G. M., & Steyerberg, E. W. (2017). A closed testing procedure to select an appropriate method for updating
  prediction models. *Statistics in Medicine, 36*(28), 4529–4539. https://doi.org/10.1002/sim.7179
