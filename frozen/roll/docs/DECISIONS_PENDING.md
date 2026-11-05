# Scientific decisions (from docs/AUDIT.md)

Each item lists the options, what each implies, and the original recommendation (★). **All items except D14
were decided by the lead on 2026-10-22** (✅ = chosen option; rationale in `docs/DECISIONS.md`) and are
**implemented** (SPECS v0.4). IDs refer to docs/AUDIT.md.

## 1. Cohort and target definitions

### D1 — What counts as "surgery" in MIMIC-IV (postoperative ICU rule and reoperation) — ✅ a (+ b as fallback)
The ICD-9 chapters 01–86 rule accepts bedside procedures (catheters, thoracentesis, endoscopy…).
- **a. Timestamped OR evidence ★ ✅** — surgery = a `PACU` (or OR) transfer in `hosp.transfers` within 24 h
  before the ICU admission; reoperation = a later PACU/OR transfer after t0. Minute precision, no coding noise.
  Combine with (b) for hospitalisations without a PACU transfer.
- **b. HCUP "major therapeutic" procedure classes** for ICD-9-CM and ICD-10-PCS (day precision).
- **c. Keep the current code-range rule** with a longer exclusion list (declared limitation).

### D8 — MIMIC-IV "service only" arm of the postoperative rule — ✅ a
Admits non-operative patients under surgical services (SAH under NSURG, non-operative trauma…).
- **a. Require surgical evidence within 24 h (as INSPIRE) ★ ✅** — service rule only as a tie-breaker; report
  the number of service-only stays as a sensitivity cohort.
- **b. Keep "service OR procedure"** (SPECS v0 wording).

### C5 — t0 must be a discharge to the ward — ✅ a
An ICU exit to the OR / IR / another ICU makes the label (reoperation, readmission) known at t0.
- **a. Exclude index stays whose next location within 6 h is OR/PACU/IR/ICU ★ ✅** (MIMIC `transfers`;
  INSPIRE: next operation start within 6 h of `icuout_time`); report the count in the flow diagram.
- **b. Merge ICU → OR → ICU into one ICU episode** (t0 = final exit).
- **c. Keep** (declared limitation).

### D10 (audit) — ICU readmission in INSPIRE (SPECS decision D1) — ✅ a
Currently only readmissions linked to another surgery are visible, so severe T1 is not comparable across cohorts.
- **a. SPECS D1 proxy ★ ✅** — new invasive ventilation / CRRT / ECMO / IABP run after t0 in `ward_vitals`, or
  an ICU entry after any surgery; validate its prevalence in the audit before use.
- **b. Mark ICU readmission unobservable in INSPIRE** and declare severe T1 without it (+ sensitivity).
- **c. Keep the current definition.**

### D5 — Chronic dialysis / advanced CKD — ✅ a
Every ward haemodialysis session counts as "new RRT" and baseline 6.0 → 6.4 mg/dL counts as AKI 3.
- **a. Make AKI and RRT-start not observable for patients with RRT before the index surgery ★ ✅** (any RRT
  record before surgery, or ESRD codes when timestamped before it), and apply KDIGO ≥ 4.0 mg/dL only with an
  acute rise and a baseline < 4.0; never use a baseline measured during RRT.
- **b. Exclude ESRD / chronic dialysis patients** from the cohort (changes the population; report in the flow).
- **c. Keep** (declared).

### D7 — KDIGO timing — ✅ a
The ratio is taken over 30 days without the 7-day window; the 0.3 mg/dL rise is only searched in the first 48 h.
- **a. Rolling KDIGO ★ ✅** — ≥ 1.5× the minimum of the previous 7 days or ≥ 0.3 above the minimum of the
  previous 48 h, always also relative to the SPECS baseline; stage by the maximum ratio to baseline.
- **b. Keep the simplified rule** (declared).

### D6 — MIMIC-IV eMAR coverage — ✅ a
Early MIMIC-IV years may have no eMAR: "not recorded" becomes 0 for antibiotics, transfusion and `med_*`.
- **a. Treat hospitalisations without any eMAR record as unobservable ★ ✅** for the mild medication components
  and medication inputs; report eMAR coverage by `anchor_year_group` in the audit.
- **b. Add `prescriptions` as an order source** (orders ≠ administrations; declare).

### D12 — Limitation of life-sustaining treatment in MIMIC-IV / eICU — ✅ a
The exclusion does nothing there although the data exist (MIMIC "Code Status" 223758; eICU carePlanGeneral).
- **a. Implement it with records timestamped ≤ cutoff ★ ✅** (DNR/DNI/comfort care).
- **b. Label the flow step "not available"** for those cohorts.

### D13 — Arterial vs venous blood gases (MIMIC-IV) — ✅ a
- **a. Keep only arterial samples (specimen type 52033 = "ART.") ★ ✅**
- **b. Keep both** (declared).

### D14 — INSPIRE clipping at p2.5–p97.5 — ⏳ deferred
KDIGO ≥ 4.0 mg/dL and SpO₂ < 90 % may be unreachable after clipping. **Decide after the first audit (M1)**:
check the clip limits per item, then drop or declare impossible criteria.

## 2. Method

### B2 — Free global scale in the regularised NCA — ✅ a
With the current parameterisation the bandwidth barely moves at large λ, biasing λ selection.
- **a. Reparameterise ★ ✅** `log w = log c + log(imp + ε₀) + u`, penalty `λ‖u‖²`, `c` unpenalised; re-space the
  λ grid (e.g. {0, 10⁻³, 10⁻², 10⁻¹, 1, 10} / F).
- **b. Keep the parameterisation, but re-fit `c` in closed form (1-D line search) after every epoch.**
- **c. No free scale** (the prior fixes the bandwidth too) — not recommended.

### B3 — Stopping rule measures optimiser noise — ✅ b (amends SPECS §5.3.2)
- **a. Deterministic conversion inside the loop ★** (fixed minibatch order, frozen candidates between refreshes,
  full-batch / L-BFGS fits), plus an estimate of the noise floor (re-fit with unchanged importance) reported
  with Δ̃; `min_delta` must exceed it.
- **b. Measure Δ on the prior-implied metric `w = c·imp` ✅** (no NCA inside the loop). Simple and noise-free,
  but departs from "recompute w with the §5.4 conversion": SPECS §5.3.2 is amended accordingly.
- **c. Keep** (plateau diagnostic declared as possibly noise-driven).

### B4 — How group importance becomes per-variable priors — ✅ a
An uninformative missingness indicator gets the same prior as its strong value variable.
- **a. Prior on group totals ★ ✅** `λ Σ_g (log Σ_{j∈g} w_j − log c − log imp_g)²`; the NCA splits within groups.
- **b. Split within groups by univariate mutual information.**
- **c. Keep the equal split** (declared).

### C3 — λ selected with a SAGE prior that saw inner-validation labels — ✅ a
- **a. Recompute SAGE in each inner fold ✅** (~3× the loop cost; the pilot will say whether it fits).
- **b. Split the inner-validation fold in two: SAGE on one half, λ scoring on the other ★** (no extra models).
- **c. Keep** (declared, OQ#14).

### B6 / B7 — Rare codes in the metric — ✅ a / a
- **B6 a ★ ✅** explicit "other / missing" dimension per ICD-10-PCS level (different rare codes no longer
  "match").
- **B7 a ★ ✅** unseen VDM categories → P(y), as SPECS §5.2 states (relevant for mode A transport).

### E13 — Tuning of the LightGBM comparator — ✅ a
- **a. Small inner-CV grid (num_leaves × learning rate) with early stopping on the inner folds ★ ✅** — same
  nested fitting as PERISIM and logistic regression (§6.4).
- **b. Keep the fixed configuration** (declared; the Q2 comparison may favour PERISIM).

## 3. External validation and reporting

### C4 — T2/T3 categories in the external cohorts — ✅ a
Tertiles computed on the whole external cohort (the test set) make calibration-in-the-large optimistic.
- **a. Cross-fitted thresholds ★ ✅** — patient-level 2-fold split of the external cohort: thresholds from one
  half, evaluation on the other, then swap; report absolute categories (≤ 3 / 4–7 / > 7 d) as a co-primary
  external analysis.
- **b. Absolute categories only** for external validation.
- **c. Keep whole-cohort tertiles** (declared limitation).

### E14 — Per-patient output (§5.6) — ✅ a
- **a. Write the §5.6 JSON for every test-fold and external query inside the pod ★ ✅** (`predictions/`, never
  exported) and a de-identified example schema in the report.
- **b. Leave the function unused** until a clinical pilot.

### E10 / E11 — Reporting details — ✅ a
- **a. Per-fold metrics and cut-points next to the pooled ones; SAGE dispersion per variable (not per
  fold-specific cluster) ★ ✅**
- **b. Keep pooled-only** (declared).

### C6 — α by in-sample LOO vs inner folds — ✅ keep LOO (declared)
Measured optimism is negligible with large categories. **Recommendation: keep LOO (declared)**; revisit only
if the audit shows many small nominal categories.

## 4. Methodology v0.5 — agreed in discussion (2026-10-24), implemented (SPECS v0.5)

Agreed with the lead while discussing the heterogeneous metric; implemented on 2026-10-24 (see docs/DECISIONS.md
for the implementation notes, in particular the flatness test).

### Per-variable shape functions (amends SPECS §5.2)
`d(a,b)² = Σ_j w_j ‖g_j(a_j) − g_j(b_j)‖²`: g_j is the SHAPE (how the values of a variable relate to the outcome),
w_j the IMPORTANCE (how much it counts). Both come from the multivariable consensus models; the kNN captures the
remaining interactions by locality, and neighbours share the same reasons for their prediction (profiles).
- **Estimation (gap 1):** consensus curve fitted by least squares to the curves of all accepted models, with a
  per-model scale and offset. Stopping by Δ of the full distance (weights and shapes; consistent with B3).
  Per-variable R² = agreement between models (generality diagnostic), used to neutralise non-general shapes.
- **Scale (gap 2):** centred log-odds (transports across baseline risks, treats low risks correctly).
- **Dimension (gap 3):** one component per class (vector) for every target, since T2 now includes death; the
  share explained by a 1-D summary is reported per variable as a diagnostic.
- **Representation (gap 4):** ~20 quantile bins with light smoothing; constant extension beyond the observed range
  (the share of external values outside the INSPIRE range is reported); plain least squares.
- **Label optimism (gap 5):** estimated inside the training fold, curves read on the explanation subsample;
  ≥ 50 patients per bin / category (rare ones pooled with their neighbour or PCS parent); in-fold vs test
  optimism reported; no cross-fitting (it would break the common space the kNN needs).
- **Folding of U-shaped variables (gap 6):** accepted and declared; its cost is measured in Q4 against the current
  HVDM metric (comparator); a double axis only as a sensitivity analysis if needed.
- **Cost (gap 7):** each model's curves are read on a fresh random subsample of 1,000 patients (bootstrap-like,
  configurable). Levers pre-declared, in this order, if the pilot exceeds the budget: (1) smaller curve
  subsample, (2) shapes only for the variables covering 95 % of the SAGE importance, (3) C3 → outer-fold prior.
  The budget is set after the pilot.
- **Shape vs magnitude (gap 8):** shapes normalised like every δ_j (shape = g, magnitude = w); each value shrunk
  per its noise (DerSimonian–Laird) and a curve not distinguishable from flat keeps the neutral axis (rank / one-hot,
  the variable is kept with its weight because it may matter through interactions).
- **Neighbourhood density (gap 9):** monitored only — effective dimension of the weights, contrast (distance to the
  k-th neighbour / distance between random pairs) and out-of-support rate, each against the HVDM and Euclidean
  comparators.

### Model space of the consensus
Logistic regression with a sampled penalty (L2 / L1 / elastic-net); penalised spline logistic regression
(GAM-like, replaces the MLP: smooth non-linear shapes at low cost); random forest; LightGBM; XGBoost. No Bayesian
family: the consensus already averages models, Bayesian logistic regression ≈ L2, BART / GPs are infeasible here.

### Experiments and targets
- **Three separate experiments**, each complete (internal CV, sensitivity, external validation, its own freeze and
  pre-registration), in this order: **T2** (primary target and demonstration of the method), **T1**
  (complications, plus a comparison with T2: neighbour overlap, weight ranks, curve shapes — "does similarity
  depend on the outcome?"), **T3** (both dimensions; metric learned anew vs combined from T1 and T2, to decide).
  One work directory per experiment, extraction shared. Changes to later experiments informed by earlier
  external results are declared in DECISIONS.md.
- **T2 redefined (amends SPECS §4.2):** short / medium / long (tertiles of the survivors' post-ICU stay) /
  in-hospital death within 30 days, ordinal with death as the worst state. T2 is no longer survivors-only and is
  defined for every patient at t0.
- **Sensitivity:** outcome-definition variants by relabelling the case base with the main metric (minutes);
  input variants (4 h margin, diagnosis window) refitted for the experiment's target only.
- **Engineering:** checkpoint the consensus loop after each accepted model and each external target, so that
  stopping the pod costs minutes, not hours.
