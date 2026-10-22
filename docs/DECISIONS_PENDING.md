# Scientific decisions pending (from docs/AUDIT.md)

Each item lists the options, what each implies, and a recommendation (★). Nothing here is implemented yet:
the pipeline keeps the v0 behaviour until the lead chooses. IDs refer to docs/AUDIT.md.

## 1. Cohort and target definitions

### D1 — What counts as "surgery" in MIMIC-IV (postoperative ICU rule and reoperation)
The ICD-9 chapters 01–86 rule accepts bedside procedures (catheters, thoracentesis, endoscopy…).
- **a. Timestamped OR evidence ★** — surgery = a `PACU` (or OR) transfer in `hosp.transfers` within 24 h
  before the ICU admission; reoperation = a later PACU/OR transfer after t0. Minute precision, no coding noise.
  Combine with (b) for hospitalisations without a PACU transfer.
- **b. HCUP "major therapeutic" procedure classes** for ICD-9-CM and ICD-10-PCS (day precision).
- **c. Keep the current code-range rule** with a longer exclusion list (declared limitation).

### D8 — MIMIC-IV "service only" arm of the postoperative rule
Admits non-operative patients under surgical services (SAH under NSURG, non-operative trauma…).
- **a. Require surgical evidence within 24 h (as INSPIRE) ★** — service rule only as a tie-breaker; report the
  number of service-only stays as a sensitivity cohort.
- **b. Keep "service OR procedure"** (SPECS v0 wording).

### C5 — t0 must be a discharge to the ward
An ICU exit to the OR / IR / another ICU makes the label (reoperation, readmission) known at t0.
- **a. Exclude index stays whose next location within 6 h is OR/PACU/IR/ICU ★** (MIMIC `transfers`; INSPIRE:
  next operation start within 6 h of `icuout_time`); report the count in the flow diagram.
- **b. Merge ICU → OR → ICU into one ICU episode** (t0 = final exit).
- **c. Keep** (declared limitation).

### D10 (audit) — ICU readmission in INSPIRE (SPECS decision D1)
Currently only readmissions linked to another surgery are visible, so severe T1 is not comparable across cohorts.
- **a. SPECS D1 proxy ★** — new invasive ventilation / CRRT / ECMO / IABP run after t0 in `ward_vitals`, or an
  ICU entry after any surgery; validate its prevalence in the audit before use.
- **b. Mark ICU readmission unobservable in INSPIRE** and declare severe T1 without it (+ sensitivity).
- **c. Keep the current definition.**

### D5 — Chronic dialysis / advanced CKD
Every ward haemodialysis session counts as "new RRT" and baseline 6.0 → 6.4 mg/dL counts as AKI 3.
- **a. Make AKI and RRT-start not observable for patients with RRT before the index surgery ★** (any RRT record
  before surgery, or ESRD codes when timestamped before it), and apply KDIGO ≥ 4.0 mg/dL only with an acute rise
  and a baseline < 4.0; never use a baseline measured during RRT.
- **b. Exclude ESRD / chronic dialysis patients** from the cohort (changes the population; report in the flow).
- **c. Keep** (declared).

### D7 — KDIGO timing
The ratio is taken over 30 days without the 7-day window; the 0.3 mg/dL rise is only searched in the first 48 h.
- **a. Rolling KDIGO ★** — ≥ 1.5× the minimum of the previous 7 days or ≥ 0.3 above the minimum of the previous
  48 h, always also relative to the SPECS baseline; stage by the maximum ratio to baseline.
- **b. Keep the simplified rule** (declared).

### D6 — MIMIC-IV eMAR coverage
Early MIMIC-IV years may have no eMAR: "not recorded" becomes 0 for antibiotics, transfusion and `med_*`.
- **a. Treat hospitalisations without any eMAR record as unobservable ★** for the mild medication components and
  medication inputs; report eMAR coverage by `anchor_year_group` in the audit.
- **b. Add `prescriptions` as an order source** (orders ≠ administrations; declare).

### D12 — Limitation of life-sustaining treatment in MIMIC-IV / eICU
The exclusion does nothing there although the data exist (MIMIC "Code Status" 223758; eICU carePlanGeneral).
- **a. Implement it with records timestamped ≤ cutoff ★** (DNR/DNI/comfort care).
- **b. Label the flow step "not available"** for those cohorts.

### D13 — Arterial vs venous blood gases (MIMIC-IV)
- **a. Keep only arterial samples (specimen type 52033 = "ART.") ★**
- **b. Keep both** (declared).

### D14 — INSPIRE clipping at p2.5–p97.5
KDIGO ≥ 4.0 mg/dL and SpO₂ < 90 % may be unreachable after clipping. **Decide after the first audit (M1)**:
check the clip limits per item, then drop or declare impossible criteria.

## 2. Method

### B2 — Free global scale in the regularised NCA
With the current parameterisation the bandwidth barely moves at large λ, biasing λ selection.
- **a. Reparameterise ★** `log w = log c + log(imp + ε₀) + u`, penalty `λ‖u‖²`, `c` unpenalised; re-space the
  λ grid (e.g. {0, 10⁻³, 10⁻², 10⁻¹, 1, 10} / F).
- **b. Keep the parameterisation, but re-fit `c` in closed form (1-D line search) after every epoch.**
- **c. No free scale** (the prior fixes the bandwidth too) — not recommended.

### B3 — Stopping rule measures optimiser noise
- **a. Deterministic conversion inside the loop ★** (fixed minibatch order, frozen candidates between refreshes,
  full-batch / L-BFGS fits), plus an estimate of the noise floor (re-fit with unchanged importance) reported
  with Δ̃; `min_delta` must exceed it.
- **b. Measure Δ on the prior-implied metric `w = c·imp`** (no NCA inside the loop). Simple and noise-free, but
  departs from "recompute w with the §5.4 conversion".
- **c. Keep** (plateau diagnostic declared as possibly noise-driven).

### B4 — How group importance becomes per-variable priors
An uninformative missingness indicator gets the same prior as its strong value variable.
- **a. Prior on group totals ★** `λ Σ_g (log Σ_{j∈g} w_j − log c − log imp_g)²`; the NCA splits within groups.
- **b. Split within groups by univariate mutual information.**
- **c. Keep the equal split** (declared).

### C3 — λ selected with a SAGE prior that saw inner-validation labels
- **a. Recompute SAGE in each inner fold** (~3× the loop cost; the pilot will say whether it fits).
- **b. Split the inner-validation fold in two: SAGE on one half, λ scoring on the other ★** (no extra models).
- **c. Keep** (declared, OQ#14).

### B6 / B7 — Rare codes in the metric
- **B6 a ★** explicit "other / missing" dimension per ICD-10-PCS level (different rare codes no longer "match").
- **B7 a ★** unseen VDM categories → P(y), as SPECS §5.2 states (relevant for mode A transport).

### E13 — Tuning of the LightGBM comparator
- **a. Small inner-CV grid (num_leaves × learning rate) with early stopping on the inner folds ★** — same
  nested fitting as PERISIM and logistic regression (§6.4).
- **b. Keep the fixed configuration** (declared; the Q2 comparison may favour PERISIM).

## 3. External validation and reporting

### C4 — T2/T3 categories in the external cohorts
Tertiles computed on the whole external cohort (the test set) make calibration-in-the-large optimistic.
- **a. Cross-fitted thresholds ★** — patient-level 2-fold split of the external cohort: thresholds from one half,
  evaluation on the other, then swap; report absolute categories (≤ 3 / 4–7 / > 7 d) as a co-primary
  external analysis.
- **b. Absolute categories only** for external validation.
- **c. Keep whole-cohort tertiles** (declared limitation).

### E14 — Per-patient output (§5.6)
- **a. Write the §5.6 JSON for every test-fold and external query inside the pod ★** (`predictions/`, never
  exported) and a de-identified example schema in the report.
- **b. Leave the function unused** until a clinical pilot.

### E10 / E11 — Reporting details
- **a. Per-fold metrics and cut-points next to the pooled ones; SAGE dispersion per variable (not per
  fold-specific cluster) ★**
- **b. Keep pooled-only** (declared).

### C6 — α by in-sample LOO vs inner folds
Measured optimism is negligible with large categories. **Recommendation: keep LOO (declared)**; revisit only
if the audit shows many small nominal categories.
