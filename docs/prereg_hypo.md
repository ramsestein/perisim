# Pre-registration — experiment `hypo`: hypotension in the next 6 h after surgery (draft)

Status: **draft by the analyst, 2026-10-29, for the lead's approval**. Written after the internal validation
(Phase 2) and before any external analysis: no metric has yet been applied to a site it was not learned on. It is
registered with the freeze (Phase 3; OSF or a local ID, **D6**) and the external stage runs once, on the frozen
version only. Design: [design_hypotension.md](design_hypotension.md); dated decisions: [DECISIONS.md](DECISIONS.md);
configuration: `config/hypo.yaml`; code: `src/hypo/`.

## 1. Question

PERISIM predicts with a learned metric (consensus shape functions g_j and weights w_j) and a case base. We
hypothesise that **the metric transports and the case base stays local**: a metric learned on two hospitals' cohorts,
frozen and applied to a third with that hospital's own patients as the case base, predicts as well as a metric
learned at the third hospital, is calibrated, and beats the honest bar of the last MAP; and that in a multi-centre
cohort the relations carried by the neighbours are hospital-specific. The same machinery is the drift instrument:
where, and with what shape, the g_j would have to deform for each site and hospital.

## 2. Data (fixed in Phase 0; design §3–§4)

- Cohorts: INSPIRE, MIMIC-IV, eICU-CRD. Postoperative adult ICU stays, the first of each hospitalization, organ
  donors excluded, deaths kept.
- Landmarks every 2 h from 2 to 48 h after the ICU admission, in the ICU and alive, with a MAP in the last hour whose
  hourly median is ≥ 65 mmHg (normotensive now), and the window observable.
- Primary outcome **y**: an hour in [L, L + 6 h) with median MAP < 65 mmHg in the ICU; death in the window counts.
  Prevalence 13.2 % (INSPIRE), 30.7 % (MIMIC-IV), 19.3 % (eICU).
- Secondary outcomes: the three bands of the lowest hourly MAP of the window (< 65 / 65–75 / ≥ 75 mmHg); y read
  through the first reading of each hour (`y_first`). Carried by the neighbours: vasopressor started in the window,
  lowest hourly MAP of the window, ICU discharge within 24 h.
- Inputs: the 25 variables of `model.inputs`. A missing value is its own value. INSPIRE cannot time infusions, so
  vasopressor or sedation "running now" and vasopressor hours are a declared source gap there (D7).
- Analysis set: 148,293 / 387,159 / 466,385 landmarks from 11,631 / 24,464 / 32,032 patients.

## 3. Done and seen before the freeze

All of it is development work on the cohorts as development sites.

- **Phase 0**: landmark builder and completeness audit, with the gate passed.
- **Phase 1**: the model (design §4, DECISIONS "`hypo` Phase 1"). There was one smoke run per rotation's development
  pair.
- **Phase 2**: internal validation by rotation. Fixed patient-level 5-fold splits per site. PERISIM was fitted on the
  training patients of each development pair, and five local comparators on each site's training patients. The
  results were seen by the lead and the analyst (`reports/cv.md`, code `11a7ad77e519`):
  - PERISIM reaches AUROC 0.78 / 0.78 / 0.80, with calibration in the large ≈ 0 and slopes 1.00–1.04.
  - Its log-loss is 0.021–0.044 lower than the last-MAP model and 0.010–0.014 lower than Euclidean kNN.
  - It equals logistic regression on all inputs (within ±0.003) and is 0.011–0.014 higher than LightGBM.
  - The partner site barely changes it (< 0.001).
- Nothing was tuned on these results. The Phase 2 folds are reused as the external stage's splits (§5).
- **Phase 3 dry run** (`reports/external_dryrun.md`): the whole external stage on eICU fold 0. eICU was served by
  the frozen metric whose development pair includes eICU, so it is in-sample, and it touches no external result. It
  checked the mechanics and timing (2 h 11 min per eICU fold with drift and shapes; about 8 h for the whole stage).
  - It found one fault: α = 0 in small case bases, which gave predictions of exactly 0 or 1 (§5). The fix removes a
    degenerate value from the grid and was chosen without regard to its effect.
  - Its in-sample figures were seen by the analyst. The fix changes two of them: H4's decision (from "does not hold"
    to "holds", in-sample) and S1's "within 0.01" (from 500 to 250 patients). Every other figure is unchanged.
  - The dry run is not evidence for any hypothesis, and no setting was chosen from its figures.

## 4. Frozen artefacts

The following are recorded at the freeze with SHA-256 hashes. The external stage checks them at its start and at its
end, and stops if any of them differs.

- **The three transported metrics** (one per rotation, named by its external site X). Each is learned once on *all*
  patients of the two development sites, with the Phase 1 procedure and seed = `seed` + 1000:
  - the encoder;
  - the consensus g_j;
  - the NCA w_j;
  - the variables left out of the metric.

  Their weights and shapes are reported at the freeze (`reports/final.md`).
- **The transported comparators** of each rotation, fitted on all landmarks of the two development sites pooled, with
  no site column and the Phase 2 procedure:
  - last MAP (logistic regression on the last MAP and its slope);
  - logistic regression on all inputs;
  - LightGBM.
- The Phase 2 folds of every site (`folds_<site>.parquet`), and the Phase 2 local comparators fitted on each site's
  training folds.
- `config/hypo.yaml`, `config/model_space.yaml`, and the code hash of `src/hypo` together with the perisim code it
  uses.
- The git tag `perisim-freeze-hypo-<time>`, pushed to GitHub before the external stage runs.

## 5. External stage

There are three rotations, each with the external site **X** (INSPIRE, MIMIC-IV, eICU) and the metric learned without
it. Every analysis runs per outer fold f of X's Phase 2 folds:

- the **case base** is every landmark of X's training-fold patients;
- the **test set** is X's fold-f landmarks.

Predictions are pooled out of fold, so every landmark of X is predicted once.

Settings shared by every kNN:
- k = 50 neighbours, distinct patients, never the query's own.
- α is chosen by the patient-level LOO of each case base, as in Phase 1, from {0.1, 0.5, 1, 2, 5, 10}. α belongs to
  the case base, not to the metric. The grid has no 0: in a small case base (a hospital, a curve draw) the LOO can
  prefer α = 0, and a unanimous neighbourhood then predicts exactly 0 or 1. The dry run found this; Phase 2 never
  chose α = 0, so its results are unchanged.
- Out of support is defined by the case base's 99th percentile.
- 95 % CIs come from a patient-level bootstrap with multinomial weights, the same resamples for every method of a
  site. There are 2,000 resamples for H1–H5 and 500 for the secondary analyses.

Methods per site:

- **PERISIM transported**: the frozen metric with X's case base.
- **PERISIM local**: a metric learned on X's training-fold patients only, with the same procedure and learning-sample
  rule (seed = `seed` + 100·f), and the same case base.
- **Local comparators**: the Phase 2 units, fitted on X's training folds.
- **Transported comparators**: the frozen ones (§4). They are used as they are, or recalibrated (logistic, intercept
  and slope on the logit) on all landmarks of n ∈ {100, 250, 500} patients drawn from X's training folds.

## 6. Hypotheses (confirmatory; H1–H3 in each external site separately, H4–H5 in eICU)

| | Hypothesis | Estimand and decision rule |
|---|---|---|
| **H1** | The transported metric discriminates as well as one learned locally | AUROC(PERISIM transported) − AUROC(PERISIM local), paired: lower 95 % CI bound > −0.02 (non-inferiority) |
| **H2** | With the local case base the predictions are calibrated | PERISIM transported: the 95 % CI of the calibration in the large (intercept with slope fixed at 1) inside ±0.10 log-odds, **and** the lower 95 % CI bound of the calibration slope ≥ 0.90 |
| **H3** | PERISIM beats the honest bar of the last MAP | log-loss(PERISIM transported) − log-loss(last MAP, local), paired: 95 % CI < 0 |
| **H4** | In eICU the local relations are hospital-specific | see below: Δ_hosp 95 % CI < 0 **and** (Δ_hosp − Δ_ctrl) 95 % CI < 0 |
| **H5** | In eICU a hospital-aware vote beats the plain pooled one | see below: for at least one γ ∈ {2, 4, 8}, the 98.33 % CI (Bonferroni over the three γ) of Δ_γ < 0 |

**H4**:
- Hospitals: eICU hospitals with ≥ 300 patients in the analysis set, after the landmark exclusions. That gives 36
  hospitals. The design's 37 counted the landmarks before the exclusions, and one hospital there had exactly 300.
- For each such hospital h and each fold f, the queries are h's fold-f landmarks. Three case bases are compared:
  - **A**: all landmarks of h's training-fold patients;
  - **B**: as many patients drawn at random from the other eICU hospitals' training folds (R = 10 draws);
  - **B′**: B with each probability moved to h's own base rate (label shift: the odds multiplied by
    [π_h / (1 − π_h)] / [π_B / (1 − π_B)], where π is the case base's landmark prevalence).
- Each case base fits its own α.
- Estimand: Δ_hosp = the mean over the eligible hospitals' test landmarks of [log-loss(A) − mean_r log-loss(B′_r)].
- Control: per fold, every eICU patient is assigned at random to groups of the eligible hospitals' sizes, and the
  same contrast is computed with the groups in place of the hospitals, giving Δ_ctrl (expected ≈ 0).
- Δ_hosp and Δ_ctrl share the bootstrap resamples.

**H5**:
- The case base is the whole eICU training fold; the queries are every eICU test landmark.
- The neighbours are unchanged. A neighbour from the query's own hospital votes with weight γ, the others with
  weight 1, and the weights are rescaled to add up to the number of valid neighbours.
- α is re-chosen by the same LOO for each γ.
- Estimand: Δ_γ = log-loss(γ) − log-loss(γ = 1).

There is no multiplicity adjustment across H1–H5, as in T2: each answers a distinct question, and estimates with CIs
are the main output. The Bonferroni correction inside H5 is part of its test.

## 7. Secondary analyses (estimates with CIs, no decision rule)

- **S1 — Local patients needed.**
  - Case bases of n ∈ {50, 100, 250, 500, 1,000, 2,000, 4,000, 8,000} patients drawn at random from X's training
    folds (R = 5 draws for n ≤ 1,000, 3 above), plus the whole training pool, all with the frozen metric and their own
    α.
  - Reported per n: log-loss and AUROC. Also the smallest n within 0.02 and within 0.01 log-loss of the whole pool,
    and the n at which the curve passes each transported comparator recalibrated on as many patients (the same
    draws).
- **S2 — Comparators.** Last MAP, logistic regression and LightGBM in three versions: transported, recalibrated
  (n = 100 / 250 / 500), and local; plus Euclidean kNN and prevalence (local). Reported against PERISIM transported
  and local.
- **S3 — In-sample control of the transport.** The Phase 2 metric that included X (X with its partner, on the same
  folds) against PERISIM transported: the cost of not having seen X.
- **S4 — Drift per site** (fold 0 only; the external site, with the two development sites as the in-sample
  control).
  - The method is T2's post hoc drift analyses, now pre-registered (scripts/posthoc_drift.py and
    posthoc_drift_missing.py, re-implemented in `src/hypo/drift.py`). Each variable becomes a lookup of points, one
    per value, and continuous variables get their own "missing" point, starting at the median bin.
  - **ICE**: 500 test landmarks moved along every value of every variable, predicted with the site's case base.
    Reported as the PDP and the 10–90 % band of the centred log-odds.
  - **Re-deformation, one variable at a time**: that variable's points are freed and fitted to the site, every other
    variable frozen. The fit uses a smoothed kNN over 150 distinct-patient candidates found with the frozen metric,
    a patient-level LOO over ≤ 20,000 anchor landmarks of the case base, and a ridge of 20 towards the frozen points.
  - **All variables at once** (candidates refreshed once), and **the missing points only**.
  - Reported per variable:
    - the change in test log-loss of the real kNN, on ≤ 20,000 test landmarks, paired with the frozen metric;
    - the effective weight, with the frozen points and with the adapted ones;
    - the shape change (weighted Procrustes residual);
    - the split-half agreement of the deformation (the anchors cut into two patient halves, each fitted alone).
  - The drift of a site is what moves beyond what moves in the development sites.
- **S5 — Shapes per site and per hospital.**
  - The consensus procedure of Phase 1 (40 accepted models) run on each site's fold-0 training patients alone, and on
    each eligible eICU hospital's.
  - Their g_j are compared with the transported metric's.
  - The **MAP threshold** is read from g_j of map_last, map_mean_6h and map_min_6h: the highest MAP below 85 mmHg at
    which the curve, linearly interpolated between bin mid-points, is ≥ ln 2 above its value at 85 mmHg (the odds
    double). It is read for every accepted model, and reported as the median and the 10–90 % range per site and per
    hospital.
- **S6 — eICU by hospital.**
  - H4's contrast per hospital, and H5's Δ_γ per hospital.
  - The enrichment of own-hospital neighbours: the share of a query's neighbours from its own hospital against the
    hospital's share of the case base.
  - The per-hospital performance of PERISIM transported.
  - The joint re-deformation (S4) with each eligible hospital's own patients as the case base.
- **S7 — Measurement and outcome definition.**
  - PERISIM transported and the local comparators within the arterial-line strata: any invasive MAP in the 6 h before
    L, for MIMIC-IV and eICU (INSPIRE has none). Also the share of neighbours in the query's own stratum.
  - The outcome read with one reading per hour (`y_first`), voted by the same neighbours.
  - The three MAP bands: multiclass log-loss and macro AUROC, PERISIM transported against local.
- **S8 — Explainability.**
  - The share of the query–neighbour squared distance carried by each variable.
  - Each variable's pull: the mean squared distance in that variable to the non-event neighbours minus that to the
    event neighbours, per site and risk decile.
  - The carried outcomes against what happened: AUROC for vasopressor start and ICU discharge in 24 h; Spearman
    correlation for the lowest MAP.
  - Out-of-support rates, transported against local.

## 8. Rules for the stage

- **Source gaps.** A frozen-metric input that X cannot measure is the same value for every landmark of X. It adds
  nothing to any distance inside X's case base: it is not imputed, re-weighted or dropped (tested).
- **Unseen surgical groups** contribute zero to the distance (perisim's rule).
- **Running once.** The external stage refuses to run without the freeze record, or when the code, the
  configuration, the folds or a frozen artefact's hash differs from it. A unit that fails is rerun with the same
  frozen code. A change to the code after the external results exist is a deviation: it is reported as one, and the
  original output is kept.
- **Small cells.** Aggregates of fewer than 10 landmarks are not exported. eICU hospitals are labelled H1, H2, … by
  size, never by their identifier.

## 9. Known limitations, declared in advance

- **The outcome is close to an input** (the MAP in the last hour). The last-MAP model is the bar, and the absolute
  levels of discrimination are bounded by it.
- **INSPIRE's measurement differs**: only non-invasive MAP, about one reading per hour, and no infusion timing. These
  are real transfer differences, part of what S4–S7 should see. They are not corrected.
- **Rotations are unequal**: each transported metric is learned on a different pair of cohorts. The three external
  sites are three different questions, not replicates.
- **Overlapping case bases**: the eICU hospitals of H4 share nothing but the frozen metric. The draws of B and of the
  control come from the same training pool, so their CIs condition on the draws.
- **Drift and shapes use fold 0 only** (cost); their CIs do not include the variation between splits.
- **Re-deformation starts from the frozen points.** A variable with (near-)zero weight has (near-)zero gradient and
  cannot gain importance there. The per-site shapes (S5) cover that case.
