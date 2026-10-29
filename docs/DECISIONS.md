# DECISIONS — dated log

Implementation decisions taken by the lead or derived from SPECS.md. Those that the pipeline takes in
the pod by declared rules (D7 from the pilot, chosen λ, Δ̃ plateaus, MLP dropped for cost,
temporal validation omitted…) are logged automatically in `<workdir>/docs/DECISIONS.md`.

## 2026-10-21 — Execution infrastructure on KHDP

- Code in a public GitHub repository, cloned in the pod through `http://github.khdp-forward-server`.
  The repository contains no data, secrets or outputs (`.gitignore` + `scripts/check_public_safety.py`).
- Single entry point `run_pod.sh`: virtual environment on the container's local disk (`~/.perisim/venv`),
  dependencies from the KHDP PyPI mirror and torch from the PyTorch mirror (CPU index, cu126 fallback).
- Read-only data in `/home/khdp-user/workspace/datasets` (automatic discovery of the three cohorts);
  work and artefacts in `updown/perisim/<user>/work`.
- Pipeline of idempotent stages with markers (hash of configuration + code): an OOM or a pod restart
  resumes from the last completed stage (or fold).
- Parallelism: at most 4 processes × 4 threads (§9.1); FAISS `IndexFlatL2` on CPU.

## 2026-10-21 — Interpretations of SPECS (see OPEN_QUESTIONS for confirmation)

- Normalisation of δ_j: each block is divided by s_j, with s_j² = E[δ_j²] over random training pairs,
  so that the mean square of each dissimilarity is 1.
- ICD-10-PCS levels: one-hot per level (prefixes of 1 to 5 characters); codes with < 10 cases in
  training → zero vector.
- `carried` is built by the targets stage from the common format (not by the extraction).
- The first postoperative stay of each hospitalisation is the unit of analysis: if it fails a later
  criterion (e.g. < 4 h), the hospitalisation is excluded.

## 2026-10-24 — Pod layout (lead)

- The datasets are the lead's own PhysioNet copies, unzipped in `<workspace>/data/{inspire,mimic4,eicu}` outside
  `updown`, for speed; `run_pod.sh` uses that folder when it exists and the KHDP mount `<workspace>/datasets`
  otherwise. Replaces the data path of 2026-10-21.
- Python environment in `<workspace>/venv` (the lead may create it beforehand; an existing environment is used as
  it is and never deleted) and run files in `<workspace>/perisim_run`, replacing `~/.perisim`: no hidden folders.
  Pip and matplotlib caches are not written to `~/.cache`; pytest writes no `.pytest_cache`.
- The repository is cloned in the workspace root (`<workspace>/perisim`) and everything is run from the root:
  `bash perisim/run_pod.sh`; the hints in messages and reports follow how the script was called.

## 2026-10-24 — Work folder on the local disk (lead)

- The first run with the three cohorts failed in `extract`: in `updown` (shared object storage) the temporary cache
  files written by the parallel extractions vanished before their rename. The work folder moves to the workspace's
  local disk (`<workspace>/perisim_work`, `PERISIM_WORKDIR`), like the data and the environment.
- After every run (also after a stop) the reports, the pod's docs and the export package — aggregates only — are
  copied to `updown/perisim/<user>/results/` (`PERISIM_PUBLISH_DIR`, empty = no copy); never predictions, extracts
  or cache. The local work folder remains the source of truth.

## 2026-10-24 — Completeness audit with the three cohorts (lead)

- MIMIC-IV `rrt_start` (prevalence < 0.5 %) accepted: RRT after the ICU exit is recorded only back in the ICU, as in
  INSPIRE; same definition, rare component.
- MIMIC-IV medication: the hospitalizations without any eMAR record (early years) use `prescriptions` (orders,
  start–stop, systemic routes only, the same name → ATC mapping), with their own D6 coverage marker; eMAR keeps
  priority where it exists (administrations rather than orders). An order running into the window counts from the
  window start (ongoing therapy, never a new course).
- The non-systemic routes of `prescriptions` are taken from its route inventory (eyes, ears, topical, inhaled,
  nasal, intradermal, catheter lock / dwell / flush, intravitreal, nerve blocks…); every configured MIMIC-IV
  item matched its d_items / d_labitems label (marked VERIFIED).
- The antibiotic-course rule now reads every record as an interval [start, end]: an order covers its days, an eMAR
  administration is one instant (identical result for point records).

## 2026-10-30 — Name of the technique: consensus effect-space kNN, CE-kNN (lead)

- The technique is named **consensus effect-space kNN (CE-kNN)**. The project and the software keep the name
  PERISIM.
- What the name describes:
  - each variable is carried to its consensus shape function g_j (centred log-odds, the consensus of a Rashomon set
    of competitive models);
  - each variable is weighted by w_j;
  - the prediction is the vote of a local case base.
- Prior work reviewed, the closest first:
  - VDM / HVDM (Stanfill & Waltz 1986; Wilson & Martinez 1997): per-value class probabilities, one variable at a
    time;
  - SHAP's explanation space (Lundberg et al. 2020): supervised clustering, with one model;
  - KAAM (Almodóvar et al. 2025, preprint): nearest patients in one additive model's logit-contribution space;
  - metric learning (NCA, LMNN, and LSML for patient similarity);
  - the prognostic score (Hansen 2008).
- None of them combines the four pieces of CE-kNN: the multi-model consensus of the curves, the SAGE + NCA weights,
  a transported metric with a local case base, and re-deforming the g_j to measure drift. The name points at the
  consensus, which the closest work lacks.
- Some names were avoided because they are taken: CEM (Coarsened Exact Matching; Iacus, King & Porro 2012), MVDM
  (Modified VDM; Cost & Salzberg 1993), EDM (Elucidated Diffusion Models).
- PERISIM alone was set aside as the technique's name for two reasons: "peri-" ties it to perioperative care, and
  PeriopSim, a perioperative training simulator, already exists.

## 2026-10-30 — `hypo` reported, followed up post hoc, merged into `main` (lead)

- Phase 5 (2026-10-29): the report, [results/hypo/REPORT.md](../results/hypo/REPORT.md), with the aggregates of every
  phase in `results/hypo/`.
- After the report the lead asked for three post hoc follow-ups. They are exploratory, not pre-registered, and are
  reported in REPORT §13:
  - the H1 arm's local weights swapped into the transported metric;
  - a threshold for the MAP block as a whole;
  - S4's drift on folds 1–4.

  They run in `scripts/posthoc_hypo.py`, outside `src/hypo`. The script only reads the frozen artefacts and checks
  the freeze record at the start and at the end.
- The branch `hypotension` is merged into `main` as a separate experiment. Its files keep their paths, because the
  freeze record, the pre-registration and the tag refer to them. README.md and results/README.md describe it next
  to T2.

## 2026-10-29 — `hypo` frozen; external stage started (lead)

- The lead approved [prereg_hypo.md](prereg_hypo.md) as drafted, including the analyst's interpretations of H1–H5
  (entry below) and the α grid without 0.
- **D6**: a local registration ID, `LOCAL-PREREG-HYPO-2026-10-29`, with no OSF filing. The pushed tag is the
  timestamp.
- Freeze: tag `perisim-freeze-hypo-2026-10-29-120652` on commit `b0c9a02`, code `cf0d0b80c70a`, configuration
  `772940e4ed77`. The branch and the tag were pushed before the external stage started (12:07:25). Record:
  [freeze_hypo.md](freeze_hypo.md).

## 2026-10-29 — `hypo` Phase 3: pre-registration draft, frozen artefacts and the external stage's code (analyst; for the lead's approval before the freeze)

- Phase 2 finished (`reports/cv.md`, code `11a7ad77e519`; summary in docs/prereg_hypo.md §3); nothing tuned on it.
- **Draft pre-registration**: [prereg_hypo.md](prereg_hypo.md). H1–H5 of the design §5 made operational, the
  secondary analyses S1–S8, rules and limitations. Analyst's choices inside the design's wording, for the lead to
  confirm or change before the freeze:
  - H1 "PERISIM refitted locally" = a metric learned on the external site's training folds only, with the same
    case base;
  - H2 "inside ±0.10" read as T2 did, the 95 % CI inside the margin;
  - H3 against the last-MAP model refitted locally (the harder bar);
  - H4's control as a decision condition (Δ_hosp − Δ_ctrl < 0);
  - H5 with a Bonferroni over the three pre-declared γ;
  - 2,000 bootstrap resamples for H1–H5 and 500 for the rest;
  - the external stage on the Phase 2 folds (case base = the training folds), so every landmark of the external
    site is predicted once;
  - drift and shapes on fold 0 only.
- **The code of the external stage is written and frozen before it runs**, as T2's was:
  - `frozen.py`: `python -m hypo final` fits the three transported metrics on all patients of each development
    pair, and the transported comparators. `python -m hypo freeze` creates the tag and the hash record and refuses
    when external results exist or the tree is dirty. `check` is the gate.
  - `external.py`: the units per site and fold.
  - `drift.py`: S4–S6, T2's post hoc drift analyses re-implemented for a binary outcome and distinct-patient
    neighbours; candidates of the one-variable fits from the frozen metric, shared by all variables.
  - `external_report.py`: H1–H5 and the secondary tables.
  - `stats.py`: perisim's calibration fits with weights.
  - The external stage checks the freeze at its start and its end.
- Small changes to Phase 1 code, behaviour unchanged (tested):
  - the distinct-patient search is a function reusable by the drift fits;
  - the case base keeps its LOO neighbours, so H5 can re-choose α;
  - the shape loop returns the accepted models' curves, so S5 can read thresholds per model;
  - `hypo`'s code hash now normalises line endings as perisim's does, so the frozen hash is the same on any checkout
    (the hash itself changes from Phase 2's `11a7ad77e519`).
- A frozen-metric input that the external site cannot measure (INSPIRE's infusion timing) is the same value for all
  of that site's landmarks, so it adds nothing to any distance in its case base (tested). It needs no rule.
- **Dry run on real data** (`python -m hypo external --dry-run`, `external_dryrun/`, `reports/external_dryrun.md`):
  - eICU fold 0 is served by the rotation whose development pair includes eICU, so it is in-sample and touches no
    external result; it checks the mechanics only;
  - every unit ran at full size in 2 h 11 min;
  - it found one fault. In small case bases (H4's hospital contrasts, the S1 curve at n ≤ 500) the LOO chose α = 0,
    and unanimous neighbourhoods then predicted exactly 0 or 1. Four hospital test rows reached the log-loss clip
    (34.5), and up to 1,909 curve predictions were 0 or 1;
  - fix: α = 0 is removed from `metric.alpha_grid` (now 0.1–10). Phase 2 never chose α = 0, so its results are
    unchanged. The grid is used only by case bases, so the frozen artefacts are unchanged too. The affected
    dry-run units (hospitals, curve, hospital drift) were rerun; no prediction is 0 or 1 any more, and the largest
    log-loss of a row is 7.1;
  - the fix changes two in-sample dry-run figures, recorded in prereg §3. H4 goes from "does not hold" to "holds":
    the own-hospital arm had carried the clipped rows. S1's "within 0.01" goes from 500 to 250 patients. The old
    outputs are kept in `external_dryrun_alpha0/`.
- Order:
  1. `final` (done; code `cf0d0b80c70a`);
  2. dry run (done);
  3. the lead approves prereg_hypo.md and decides D6;
  4. commit the code;
  5. `freeze` and push the tag;
  6. `external` (about 8 h).

## 2026-10-28 — `hypo` Phase 2: internal validation by rotation (analyst's design; Phase 1 confirmed by the lead)

- The lead confirmed Phase 1's defaults (entry below). `src/hypo/cv.py`, `compare.py`, `report.py`; `cv` in
  `config/hypo.yaml`; `python -m hypo cv` (resumable: finished units are kept) and `python -m hypo cv-report`.
- **Folds fixed per site**: 5 patient-level folds, stratified by the outcome, the same in every rotation, so a
  site's test fold is predicted by the metric learned with each of the other two sites (one line per partner).
- **PERISIM per rotation and fold**: metric learned on the learning sample of both development sites' training
  patients; each site queried with its own case base (its training landmarks); α by LOO in that case base.
- **Comparators are local** (one site's training landmarks): prevalence, last MAP + slope (logistic), logistic on
  every input (C tuned), T2's LightGBM (grid × early stopping, temperature), Euclidean kNN (neutral axes, uniform
  weights, the same case base and vote). Settings are T2's (`model_space.yaml` `baselines`), tuned in 3 inner
  patient-level folds. They do not depend on the rotation, so each runs once per site and fold, and they are the
  external stage's "refitted locally" arm too. Transported comparators (trained on the development sites, applied
  to the external one) belong to the external stage.
- Report: AUROC, log-loss, Brier, calibration in the large and slope per site and method, and PERISIM − comparator
  paired in log-loss and AUROC, with 95 % CIs from 500 patient-level bootstrap resamples (the same for every method
  of a site; weighted AUROC). Descriptive: Phase 2 has no decision rule; the hypotheses H1–H5 are for the external
  stage, after the freeze.
- Chain checked on real data in quick mode (8 models, 10 epochs; rotation external-eICU, fold 0; `cv_quick/`,
  kept apart from the run): PERISIM unit 137 s; local comparators per site and fold 35 s (INSPIRE) to 90 s
  (MIMIC-IV). The full run is estimated at 2.5–3 h.

## 2026-10-28 — `hypo` Phase 1: the metric and the case base (analyst's defaults, confirmed by the lead)

- `src/hypo/data.py` (landmarks → perisim's encoder input; learning sample), `metric.py` (shapes and weights),
  `casebase.py` (neighbours and probabilities); settings in `metric` of `config/hypo.yaml`. perisim's encoder, model
  space, surrogates, curve reading, consensus, NCA and smoothed vote are used unchanged.
- **No SAGE stage** (T2: λ ≈ 0). The weights are a free diagonal NCA; the shape loop stops at a fixed 40 accepted
  models (T2's loops accepted 31–57), with the stability trace Δ of the distance (uniform weights) recorded only.
- **Several development sites.** The site is a nuisance column of the ensemble's models, not part of the metric:
  each g_j is a within-site effect, not a base-rate difference between sites. The NCA searches each anchor's
  candidates within its own site, as the metric is used (each site queries its own case base). Each development
  site gives the learning sample the same number of landmarks (≤ 30,000; ≤ 3 random landmarks per patient).
- **Inputs constant within every development site leave the metric**: the missing indicator of a source gap is the
  site itself (INSPIRE's "vasopressor running now" and vasopressor hours when INSPIRE is a development site).
- **Neighbours are distinct patients.** A patient's consecutive landmarks are near-copies; each other patient enters
  the k = 50 once, at its most similar moment, and the query's own patient never enters. α by patient-level LOO on
  20,000 of the case base's own landmarks; out of support as T2 (99th percentile of the k-th distance).
- The secondary bands are the same neighbours' smoothed vote over three classes; carried outcomes (vasopressor
  start, ICU discharge in 24 h, MAP at 6 h) are their shares / medians.
- Smoke run (`python -m hypo smoke --external eicu`: development INSPIRE + MIMIC-IV, 80 / 20 patient split, mechanics
  and cost only): 411 s in all (shapes 174 s, NCA 187 s); 40 accepted models in 46 attempts, Δ ≈ 0.002–0.003 from
  the 20th; 18 of 27 variables shaped; no neighbourhood short of 50 distinct patients; held-out mean prediction
  equal to each site's prevalence (13 / 31 %), AUROC 0.78 at both. A quick version (8 models, 10 epochs) gave the
  same numbers. Comparators are Phase 2.

## 2026-10-28 — New experiment `hypo`: hypotension in the next 6 h in the ICU after surgery (lead)

- T2 was the deliberately hard case (little information at ICU discharge), run to learn whether the shape functions
  carry real information, where the method breaks (rare classes, small case bases) and what it is good for (transfer
  with a local case base, drift diagnosis). The next experiment applies that to a case with dense data and a frequent
  outcome, centred again on transfer and explainability. Candidates measured on the original tables; rare outcomes
  (vasopressor start, 2–9 % per landmark) discarded; the lead chose hypotension in the next 6 h.
- Design: `docs/design_hypotension.md`; feasibility: `docs/feasibility_hypotension.md`. **Approved defaults:** D1
  leave-one-site-out; D2 secondary outcome in absolute MAP bands (< 65 / 65–75 / ≥ 75); D3 arterial line as a
  stratum, not an input; D4 landmarks every 2 h; D5 MAP < 65 mmHg on the hourly median, 6-h horizon. Open: D6 OSF
  registration, D7 verify INSPIRE's ICU vasopressor infusions, D8 T1 / T3 of the original plan.
- Code: a separate package `src/hypo/` (config `config/hypo.yaml`) on the branch `hypotension`. `perisim` is used as a
  library and not changed, so T2's frozen code and configuration hashes stay valid on this branch; `hypo` has its own
  code hash (its files plus perisim's hash). Population from T2's `stays_candidates` (postoperative adult ICU stays,
  first of the hospitalization, deaths kept); vitals, labs and supports read again from the original tables, because
  T2's common format keeps only T2's index stays and, in MIMIC-IV / eICU, the vitals of a window before the ICU
  discharge.
- **Phase 0 (data), approved by the lead.** First landmark build: INSPIRE 148,293 landmarks (13.2 % hypotension),
  MIMIC-IV 387,159 (30.7 %), eICU 466,385 (19.3 %). Fixed before the model:
  - eICU invasive ventilation was 1.7 % of the landmarks (respiratoryCare intervals alone): now also the ventilator
    settings charted in respiratoryCharting (set rate / volume, plateau / peak pressure, mode, …; PEEP / CPAP / EPAP
    left out, non-invasive ventilation charts them) and `treatment` (as T2); eICU temperature also from
    nurseCharting ("Temperature (C)"), vitalPeriodic having it for 16 %.
  - Charted points of a support merge into runs while the gaps are < 6 h (T2's `support_gap_hours`); new inputs
    "vasopressor / sedation in the last 24 h", comparable across sites.
  - **D7 resolved**: INSPIRE's medication table does hold the ICU (records for 99.8 % of the stays, median 19 per
    stay); vasopressors in the ICU 8.5 % of the stays (norepinephrine 4.0 %, dopamine 4.1 %), sedation 16.3 % — a
    practice difference, not missing data. But its records are administrations, not infusion intervals, so
    "running now" (vaso_on, sedation_on) and vasopressor hours are a declared `source` missing value in INSPIRE.
  - Left out of the metric: BMI (absent in MIMIC-IV), surgery duration and time since the surgery (no surgery times
    in MIMIC-IV / eICU). Unmapped surgical group → its own category "unknown" (MIMIC-IV 25 %).
  - A completeness audit (`python -m hypo audit`) is the gate before the model: outcome prevalence ≥ 10 % per
    cohort, every input ≥ 50 % coverage unless declared a gap.
  - Ventilation checked at the first landmark: INSPIRE stayed at 61–64 % up to 24 h in every surgical group because
    its `vent` item is 0 / 1 and `hypo` counted the 0 records (39 % of them) as ventilation; now as T2's extraction
    (`value != 0`, which T2 already applied). MIMIC-IV rose from 37 % at 2 h to 52 % at 6 h (the ventilation
    procedure is charted hours after the arrival): the ventilator settings charted hourly in chartevents are added
    (mode, set volume / rate, peak / plateau / mean airway pressure; PEEP left out), as in eICU.
  - eICU lactate (23 % of the landmarks, drawn selectively) is a declared `measurement` gap: kept with its missing
    value as its own value (design §4).
  - After the fixes the audit passes with no flags. Ventilation at L = 2 / 24 h: INSPIRE 37 / 11 %, MIMIC-IV
    45 / 26 %, eICU 40 / 12 %. MIMIC-IV's rise to 60 % at 6 h is composition, not charting: in the same stays it falls
    (45 → 43 %); the stays without a MAP in the first ICU hour (8,733, 82 % ventilated, mostly cardiac) enter the
    landmarks only from L = 4–6. The earliest landmarks under-represent the busiest arrivals; a property of the
    design (a MAP in the last hour is required), reported, not corrected.

## 2026-10-28 — Post-hoc case-base analyses: patients needed, and hospital-specific relations in eICU (lead)

- Two questions of the lead after the drift analysis. (1) With the frozen INSPIRE metric (the "scaffold"), how many
  local patients, added at random, does it need to recover what the whole local pool gives? The pre-registered
  accumulation curves (analysis C) jump from 2,000 patients to the whole pool (~18,000) and are still far from it at
  2,000. (2) The lead's hypothesis: the case base carries physiology and local relations, not just a recalibration,
  and eICU transports worse because it is several hospitals; are the relations hospital-specific?
- **Exploratory, post hoc, not pre-registered, no decision rule.** Frozen metric, transport stage's split, labels, test
  sets, k and α.
- (1) Dense accumulation: n ∈ {50, …, 16,000} patients drawn at random from the pool, 20 draws per n; INSPIRE (its
  own pool) as the reference. Reported: log-loss, macro AUROC, distance to the whole pool, share of its gain over the
  local prevalence, change per doubling of n, the n that comes within 0.02 / 0.01 of the whole pool and the n that
  matches the transported, recalibrated comparators of the post-hoc comparison. A first scratch run reproduced
  analysis C (n = 500: 1.116 / 1.116; 2,000: 1.086 / 1.086, MIMIC-IV).
- (2) eICU hospitals with ≥ 300 labelled index stays (analysis F's set and labels H1, H2, …), their test stays of the
  transport test set. Same case-base size: A = the hospital's own pool; B = as many patients drawn from the other
  eICU hospitals (10 draws); B' = B with its probabilities moved to the hospital's class shares (label-shift
  correction), so that A − B' measures relations beyond base rates. Control: the same with random groups of patients
  of the same sizes. Also the enrichment of own-hospital neighbours in the whole eICU pool and the k-th neighbour
  distance. Patient bootstrap of the paired per-stay differences.
- `scripts/posthoc_case_base.py` (`run_local.ps1 case-base`), outside `src/perisim`; frozen artefacts' SHA-256 checked
  at the start and the end. Outputs: `results/posthoc_case_base/`, `reports/posthoc_case_base.html`,
  `artifacts/<variant>/posthoc_case_base.json`; hospitals only by their size rank, never by eICU identifier.

## 2026-10-28 — Post-hoc drift follow-up: only the missing values re-deformed (lead)

- The drift analysis suggested that much of eICU's drift is in what a missing value means (blood gases, GCS). The
  lead asked for the direct test: with the frozen INSPIRE metric and each site's case base, free ONLY the points of
  the missing-data values, all at once, and see how much of the drift (all points re-deformed) and of the gap to the
  local refit A(iii) that recovers.
- **Exploratory, post hoc, not pre-registered, no decision rule.** Same lookup, split, labels, k, smoothed kNN,
  ridge and evaluation as the drift analysis; INSPIRE again the (in-sample) control.
- Freed: the "missing" value of every continuous variable and missingness indicator (it starts at the median's
  point) and the "unobserved" value of every binary (at INSPIRE's prevalence); nothing else moves. A 10⁻³ jitter
  starts each freed point off the observed point it coincides with (there the gradient of those pairs is zero).
- Read in three groups, each evaluated with only its own adapted points (no refit): **measurement** (a value not
  recorded), **structural** (hours since a support that was never used: a clinical state, not a gap) and **source**
  (a binary whose source was not recorded, e.g. the medication tables). Each move is also fitted on the drift
  analysis' two patient halves (agreement in direction).
- `scripts/posthoc_drift_missing.py` (`run_local.ps1 drift-missing`), outside `src/perisim`; frozen artefacts'
  SHA-256 checked at the start and the end. Outputs: `results/posthoc_drift_missing/`,
  `reports/posthoc_drift_missing.html`, `artifacts/<variant>/posthoc_drift_missing.json`; fits cached in
  `predictions/<variant>/posthoc/` (never exported); moves of values with < 10 stays at a site are not exported.

## 2026-10-28 — Post-hoc drift analysis: ICE and re-deformation of the frozen metric (lead)

- The lead stated the purpose of the transport work: PERISIM is meant as a tool to study drift, not to beat the ML
  comparators (its loss against them was expected). With the frozen INSPIRE metric and each site's own case base
  (analysis A(i)), which variables would have to change their shape function g_j, or their importance, to fit the
  site, and how?
- **Exploratory, post hoc, not pre-registered, no decision rule.** Same split, labels and k as the transport stage.
- The metric is read as a lookup: each variable's values (INSPIRE's bins, categories, 0 / 1) are points
  √w_j·g_j/s_j; continuous variables get an extra "missing" value at the median bin's point (INSPIRE imputes the
  median), so informative missingness can appear. The lookup reproduces the frozen embedding exactly (checked per
  site; MIMIC-IV test log-loss 1.0494, as in the post-hoc comparison).
- ICE: 1000 test stays moved along every value through the frozen geometry, hard kNN with the site's case base
  (blind where g_j is flat). Re-deformation: one variable's points freed for the case base and the queries, fitted by
  L-BFGS on a smoothed version of the same kNN (votes σ((r_a − d²)/τ_a) around each anchor's k-th distance, frozen α),
  patient-level leave-one-out over the case base, ridge 20 towards INSPIRE's points; evaluated with the real kNN on
  the test set (paired patient bootstrap). Per variable: Δ log-loss, effective weight at the site with INSPIRE's and
  with the adapted points, shape change (weighted Procrustes residual), split-half reliability. Then all variables
  at once, against the local refit A(iii).
- INSPIRE is run as the control with the same split rule and fits (in-sample: the metric was read from all of it).
- Outside `src/perisim` (`scripts/posthoc_drift.py`, `run_local.ps1 drift`); frozen artefacts' SHA-256 checked at
  the start and the end. Outputs: `results/posthoc_drift/`, `reports/posthoc_drift.html`,
  `artifacts/<variant>/posthoc_drift.json`; fitted points cached in `predictions/<variant>/posthoc/` (resumable,
  never exported); adapted points of values with < 10 stays at a site are not exported.

## 2026-10-28 — Post-hoc transport comparison with the parametric comparators (lead)

- After the T2 external validation and transport analyses, the lead asked how much PERISIM loses when transported
  compared with the ML comparators of the CV, with and without recalibration, and where the sites differ (the shape
  of g_j between sites, against the distribution of each variable). The external stage had transported only the
  logistic regression, so LightGBM had never been evaluated outside INSPIRE.
- **Exploratory, post hoc: added after the external and transport results had been seen; not part of the
  pre-registration and without a decision rule** (stated in the report). Nothing confirmatory changes.
- Same split (fixed test set, nested tuning subsets of analysis D, local pool), labels (cross-fitted tertiles) and k
  as the transport stage. Each method — PERISIM, LightGBM (temperature), logistic regression, Euclidean kNN,
  prevalence — trained on INSPIRE and applied as it is; recalibrated with 100 / 250 / 500 local patients (multinomial
  p'_c ∝ exp(a_c + b·log p_c), intercepts only or intercepts + slope, penalised ML with a N(0, 2²) prior on a and
  log b); refitted on the local pool (the ceiling; PERISIM = A(iii) of the transport stage). PERISIM also with
  INSPIRE's case base plus the same n local patients (γ(n)). LightGBM and logistic regression nested as in the CV.
  Paired patient-bootstrap CIs (2000).
- Shapes: the per-site consensus of analysis E on INSPIRE's grid, divided by the site scale, drawn per variable and
  class above the share of each site's stays in INSPIRE's bins (bins with < 10 stays suppressed; the outer bin edges,
  single patients' extreme values, are not shown). Per variable: PSI over INSPIRE's bins, % outside INSPIRE's range
  and missingness (a different population) next to the shape deviation (a different relationship) and w_j.
- **Outside `src/perisim`** (`scripts/posthoc_transport.py`, `run_local.ps1 posthoc`): the frozen code hash covers
  all of `src/perisim`. It only reads the frozen artefacts and checks their SHA-256 at the start and the end. Outputs:
  `results/posthoc/`, `reports/posthoc_transport.html`, `artifacts/<variant>/posthoc_transport.json` (aggregates);
  per-patient predictions cached in `predictions/<variant>/posthoc/` (never exported).

## 2026-10-27 — T2 frozen; aggregate results versioned in the repository (lead)

- T2 frozen on the lead's workstation: tag `perisim-freeze-t2-2026-10-27-141040` (commit `c4cc8b7`, pushed to
  GitHub before the external stage ran), code hash `77b25fe8f629`, scientific configuration hash `1f6d4f5b89d4`. The
  OSF registration was not filed yet: the freeze records the local ID `LOCAL-PREREG-T2-20261027` (not an OSF ID);
  the OSF ID will be added next to it. The public commit and tag are the verifiable timestamp.
- The "no outputs in the repository" rule came from the KHDP pod. On the lead's own computer, aggregate results are
  versioned in `results/<experiment>/` (`scripts/publish_results.py`: the export package minus the zip and the bulky
  per-fold CV artifacts, scanned again for identifiers). Never in the repository: `data/` and the rest of `work/`
  (extractions, features, per-patient predictions), which are restricted patient-level data. `.gitignore` and
  `scripts/check_public_safety.py` allow CSV tables under `results/` only.

## 2026-10-25 — Secondary k sweep of the internal CV (lead)

- The lead asked how much the choice of k contributes. The CV evaluates k ∈ {10, 20, 50, 100, 200}; on the
  provisional folds the macro and death AUROC of every kNN method were still rising at k = 200, so the grid does not
  show where the curves turn.
- **Secondary and descriptive, added after the provisional results of the first CV folds had been seen** (stated in
  the report). The primary k (D11, k = 50) and the confirmatory comparisons are unchanged.
- Each outer fold's metric, saved by the CV, is queried again over k = 1…2000 (at most 20 % of the smallest training
  case base), with α refitted per k by the same patient-level LOO and nothing else refitted. Reported: the curve of
  every kNN method (log-loss, Brier, RPS, AUROC, calibration slope), the paired log-loss difference of each k against
  k = 50, every comparator against PERISIM at the same k, PERISIM against LightGBM and logistic regression, and k
  chosen inside the training fold (argmin of the LOO log-loss that fits α) against k = 50. The best k pooled on the
  test folds is shown as optimistic.
- Comparators: re-queried with the weights the CV stored (w/Σw, 6 decimals, floor 1e-9); the HVDM encoder refitted as
  in `fit_perisim`; RF-GAP refitted with the fold seed. A reproduction check compares the sweep with the CV's own
  predictions on the CV grid.
- **Not a pipeline stage:** a new stage changes `pipeline.py`, which is in the key of `harmonize` and `pilot`; through
  the upstream keys that would re-run the pilot, the CV and the sensitivity variants (about 9 h here) to reproduce the
  same results. Command `ksweep` (`run_local.ps1 ksweep`, `run_pod.sh ksweep`), keyed by the CV it was computed from;
  `experiment.k_sweep` is in no stage's key (tested). Outputs: `results/ksweep/`, `reports/ksweep.html`,
  `artifacts/<variant>/ksweep_<t>.json` (aggregates, exported); patient-level predictions in
  `predictions/<variant>/ksweep/` (never exported).

## 2026-10-25 — Local runs on the lead's workstation; deviation from the pilot's cost rule (lead)

- The experiments run on the lead's workstation (Intel Core Ultra 9 275HX, 24 cores, 64 GB RAM, NVMe) with the lead's
  own PhysioNet copies, through `run_local.ps1` and the configuration profile `config/profiles/local.yaml` (only what
  differs from the KHDP configuration; recorded by the freeze). Patient-level data stay on the workstation; only logs
  and aggregates are read back.
- **Deviation from the declared pilot rule (decided on cost only, before any cross-validation result):** the rule
  "a family costing > 3 × the median of the others is excluded" (§9.1, sized for the pod's budget) excluded random
  forest at 3.1 × (32 s vs a median of 10 s per accepted model). With the whole CV estimated at about 1 h here, the
  local profile disables the cost exclusion (`pilot.family_cost_ratio_exclude: null`) and the pilot is re-run, so the
  consensus keeps every family of the model space. The pilot's fold-0 performance was not used for this decision.

## 2026-10-25 — Transport analyses approved and implemented (lead), SPECS §6.2b

- The lead approved the map, the pre-registration amendment and the eight decisions of
  [transport_review.md](transport_review.md) §8 as proposed, margins included (H1 ±0.10, H2 0.90, H3 0.02). The stage
  `transport` replaces `transfer` (its variants are contained in A and B).
- Implementation choices (declared in the amendment): the site scale of E is the **median** over variables of the
  per-variable WLS slope (a global WLS fit was biased by a single non-transporting variable: unit test); E runs the
  reduced consensus in all three sites, INSPIRE included, so the three are estimated the same way; A, C and D use the
  primary labelling (cross-fitted tertiles), B both; in A the frozen metric keeps INSPIRE's α, the locally refitted
  metrics fit their α on the local pool; the % out of support of C is computed in the first 5 repetitions; H1's
  "beyond the margin" means a CI entirely outside ±0.10; F uses the same γ values with random order only.
- The eICU hospital is read from the raw `patient` table inside the module (not added to the extraction, which would
  re-key the extraction and the development stages).
- Also fixed: `cohorts.allow_missing_cohorts` had been left under the `transfer` block of experiment.yaml when that
  block was inserted (read only when a cohort is missing, so it had not failed).

## 2026-10-24 — Model-transfer validations (lead), SPECS §6.2b

- New stage `transfer`, after `external` (so on the frozen pipeline). In MIMIC-IV and eICU: one patient-level split,
  80 % local cases / 20 % held-out queries, the same for every variant. INSPIRE weights and shapes with the local
  cases (T_local), with INSPIRE ∪ local cases (T_pooled), against PERISIM fitted on the local cases (L_perisim),
  an unsupervised Euclidean kNN and the local prevalence, and the full transport on the same queries (A_transport).
- Multi-source development: PERISIM refitted on INSPIRE ∪ eICU, evaluated on MIMIC-IV only, T2 only (eICU has no full
  T1): with its own case base, the local cases, or both.
- Implementation choices (declared): the target cohort's cut-points come from the local cases only; each development
  cohort keeps its own cut-points; α of every case base is fitted on that case base; the local and multi-source fits
  use the same nested procedure and stopping rule as the final INSPIRE model; the final INSPIRE model is saved by the
  external stage (inside the pod, `predictions/`) and reused, not refitted.

## 2026-10-24 — Harmonization from the real INSPIRE and eICU schemas (first discover in the pod)

- INSPIRE `medications` ships one ATC column (`atc_code`); `atc_code2` / `atc_code3` (data dictionary) are optional and
  used when present. It also has `route` (po / iv / ex): `ex` (external, e.g. topical epinephrine) feeds neither the
  ATC inputs nor the vasopressor support — the INSPIRE counterpart of excluding local products in MIMIC-IV / eICU
  (AUDIT D4).
- INSPIRE item names fixed from the inventory: `lacate` (sic), `paco2`, `pao2`, `nibp_mbp`; GCS is charted by
  component (`gcs_e`, `gcs_m`, `gcs_v`) and the total is E + M + V charted at the same time, the same rule as in
  MIMIC-IV (a time point without all three gives no total). Devices `vent`, `crrt`, `ecmo`, `iabp` confirmed.
- eICU: every configured table and column exists; lab names, FiO2 labels, vasopressor names and discharge locations
  confirmed against the inventory.
- Completeness audit of the first real run (OPEN_QUESTIONS 36): red blood cell transfusion is removed from T1 in
  every cohort — INSPIRE does not record it (prevalence < 0.5 %), and development and external validation must share
  one outcome definition. Mild = AKI 1–2 or a new antibiotic course; transfusion stays a carried variable. The audit
  no longer raises issues for components listed in `t1.exclude_components`.
- Lead's answers to OPEN_QUESTIONS 32–35: department OT = ophthalmology (group of EY); CTS split by the ICD-10-PCS
  code of the operation (02 / 5A cardiac, 03–06 vascular, other codes thoracic, no code cardiac); INSPIRE MAP =
  (SBP + 2·DBP) / 3 where SBP and DBP are charted without a MAP; transfusion left to the completeness audit.

## 2026-10-22 — Scientific decisions after the full audit (lead)

Decided by the lead on the options of `docs/DECISIONS_PENDING.md` (IDs from `docs/AUDIT.md`).
Status: **decided and implemented** (2026-10-22; SPECS v0.4, tests in `tests/test_decisions.py`).
Implementation notes: the D6 coverage rule also applies to eICU medication orders (inputs only); the HCUP
procedure-class files are optional (until provided, the MIMIC-IV fallback uses a declared approximation);
the INSPIRE readmission proxy ignores ward dialysis of chronic-RRT patients; binary event inputs are now explicit
0 where the source is observed, so availability = observability (also closes AUDIT C10).
Two readings to confirm: (1) D5 "not observable" is implemented as **not applicable** — AKI and new RRT simply do
not count for chronic-RRT patients and T1 is decided by the other components; (2) D6 makes T1 (and T3)
**undefined** for stays without medication coverage, whatever their other outcomes (excluding only the "none"
ones would be an outcome-dependent selection); severe / not severe and T2 stay defined for them
(`targets.yaml: unobserved_mild_policy: reduced_definition` would instead let the other components decide).

**Cohort and targets**
- **D1 / D8 — surgery in MIMIC-IV:** timestamped operating-room evidence — a PACU/OR transfer in
  `hosp.transfers` within 24 h before the ICU admission; reoperation = a later PACU/OR transfer after t0
  (minute precision). HCUP "major therapeutic" codes only as a fallback without a PACU transfer. A surgical
  service alone no longer qualifies.
- **C5 — t0 must be a discharge to the ward:** exclude index stays followed within 6 h by OR / PACU / IR /
  another ICU (MIMIC `transfers`; INSPIRE: next operation start within 6 h of `icuout_time`); reported in the flow.
- **D10 — ICU readmission in INSPIRE (SPECS D1):** proxy = a new invasive ventilation / CRRT / ECMO / IABP run
  after t0 in `ward_vitals`, plus ICU entry after another surgery; its prevalence is validated in the audit
  before use.
- **D5 — chronic dialysis / advanced CKD:** AKI and new RRT are not observable for patients with RRT before the
  index surgery; KDIGO ≥ 4.0 mg/dL only with an acute rise and baseline < 4.0; no baseline measured during RRT.
- **D7 — KDIGO timing:** rolling windows (≥ 1.5× the minimum of the previous 7 days or ≥ 0.3 mg/dL above the
  minimum of the previous 48 h), always also relative to the SPECS baseline; stage by the maximum ratio.
- **D6 — MIMIC-IV eMAR coverage:** hospitalisations without any eMAR record are unobservable for the mild
  medication components and medication inputs; eMAR coverage by `anchor_year_group` reported in the audit.
- **D12 — limitation of life-sustaining treatment:** implemented in MIMIC-IV ("Code Status") and eICU
  (`carePlanGeneral`) with records timestamped ≤ cutoff.
- **D13 — blood gases in MIMIC-IV:** arterial samples only (specimen type "ART.").
- **D14 — INSPIRE clipping:** deferred until the first completeness audit on the real data.

**Method**
- **B2:** reparameterise the regularised NCA as log w = log c + log(imp + ε₀) + u with penalty λ‖u‖² and c free;
  λ grid re-spaced.
- **B3:** the stopping criterion Δ is measured on the metric implied by the prior (w = c·imp, within-group split
  used only for Δ), not on the NCA-converted weights. **This amends SPECS §5.3.2** ("recompute w^(n) with the
  conversion of §5.4"): the SPECS text must be updated accordingly when implemented.
- **B4:** the SAGE prior acts on group totals (λ Σ_g (log Σ_{j∈g} w_j − log c − log imp_g)²); the NCA splits within
  groups.
- **C3:** SAGE is recomputed inside each inner fold for the selection of λ (≈ 3× the loop cost; the pilot will
  quantify it before the full CV).
- **B6 / B7:** explicit "other/missing" dimension per ICD-10-PCS level (cap revised at levels 4–5); unseen VDM
  categories → P(y) (SPECS §5.2).
- **E13:** the LightGBM comparator is tuned in the inner CV (small num_leaves × learning-rate grid, early
  stopping on the inner folds).
- **C6:** α remains selected by patient-level LOO on the training fold (declared deviation from §5.5; revisit
  only if the audit shows many small nominal categories).

**External validation and reporting**
- **C4:** external T2/T3 tertiles by patient-level 2-fold cross-fitting within the external cohort, plus
  absolute categories (≤ 3 / 4–7 / > 7 d) as a co-primary external analysis.
- **E14:** the §5.6 per-patient JSON is generated inside the pod for every test-fold and external query
  (`predictions/`, never exported), with a data-free schema example in the report.
- **E10 / E11:** per-fold metrics and cut-points next to the pooled ones; SAGE dispersion reported per variable.

## 2026-10-24 — Methodology v0.5 (discussion with the lead)

Status: **decided and implemented** (SPECS v0.5, tests in `tests/test_v05.py`). Options and reasoning: section 4 of
`docs/DECISIONS_PENDING.md`.

- **Metric geometry:** every continuous / ordinal / nominal / PCS variable lives on its consensus shape function
  g_j (centred log-odds, one component per class), read from the competitive models of the SAGE loop (ALE for
  ordered variables, local effect against the reference category for nominal / PCS), fitted by least squares with a
  per-model scale, shrunk per value (DerSimonian–Laird), and replaced by a neutral axis (rank / one-hot) when it is
  not distinguishable from flat. Weights w_j (SAGE prior + NCA) unchanged. The v0.4 HVDM geometry is a comparator.
- **Flatness rule (implementation note):** "τ² = 0" alone keeps about half of the pure-noise curves (DerSimonian–
  Laird τ² > 0 whenever Q exceeds its degrees of freedom), so a curve is kept only if a homogeneity F test (between-
  model variance pooled over the values) rejects "flat" at 5 % with Bonferroni over the classes; on simulated noise
  this neutralises ≈ 93 % of the curves and keeps every weak real shape tested. The shrinkage is as decided.
- **Stopping:** Δ on the prior-implied metric over the CURRENT shapes (B3 extended); the loop is checkpointed after
  each accepted model; early accepted models outside (1+ε)·best leave the consensus (AUDIT B8).
- **Model space:** logistic regression with sampled penalty (L2 / LASSO / elastic-net), penalised spline logistic
  regression (replaces the MLP), random forest, LightGBM, XGBoost. The pilot excludes any family costing > 3× the
  median of the others.
- **T2:** short / medium / long (survivors' tertiles) / in-hospital death, ordinal with death worst.
- **Experiments:** one per target (`PERISIM_EXPERIMENT=t2|t1|t3`, default t2), each with its own freeze (git tag
  `perisim-freeze-<target>-…`) and work folder; extraction shared.
- **Comparators:** trivial prevalence and prevalence by surgical group (internal), INSPIRE prevalence and the target
  cohort's LOO prevalence (external), Euclidean kNN on neutral axes, HVDM geometry with the same prior and λ, plus
  the v0.4 list.
- **Sensitivity:** outcome-definition variants by relabelling the main metric's case base; input variants refitted
  for the experiment's target only.
- **Cost:** curve subsample 1,000 per model; pilot budget gate with the pre-declared lever order (subsample →
  shapes only for 95 % of the importance → C3 prior from the outer fold).
- **Not decided yet:** the T3 metric (learned anew, as implemented now, or combined from T1 and T2).
