# SPECS.md — PERISIM

> **For the LLM agent working in this repository:** this document is the project's source of truth. Read it in full before writing any code. If something is missing or ambiguous, **do not decide silently**: add the question to `docs/OPEN_QUESTIONS.md` and continue with whatever does not depend on it. Everything marked `[VERIFY]` is an assumption about schemas or data that must be confirmed against the official documentation before use.

- **Project:** PERISIM (*Perioperative Similarity*)
- **Lead:** Ramses Marrero García (MD, PhD), AI-Lab, Hospital Clínic de Barcelona / SIMTRI, SNUH
- **Platform:** KHDP (Korea Health Data Platform, SNUH) + PhysioNet
- **Document version:** v0.5 (2026-10-24)
- **Status:** design closed; operational definitions are a v0 proposal pending clinical validation
- **v0.4 amendments** (lead's decisions after the full audit, 2026-10-22; rationale in `docs/DECISIONS.md`, IDs from
  `docs/AUDIT.md`) are marked **[v0.4]** in the text.
- **v0.5 amendments** (methodology discussion with the lead, 2026-10-24: per-variable shape functions, T2 with death,
  one experiment per target, model space, trivial comparators, sensitivity by relabelling) are marked **[v0.5]**.

---

## 1. Objective

Develop and validate a predictive system based on **k-nearest neighbours (kNN) with a learned distance metric**. The metric weights each input variable by its predictive importance, estimated with **SAGE aggregated across many models**. It is applied to postoperative patients **at the moment of ICU discharge**.

For each new patient, the system returns:

1. The probability of a **post-ICU complication**: none / mild / severe.
2. The probability of each **post-ICU hospital length-of-stay** category.
3. The **k retrieved neighbours** and the distribution of their **carried future variables** (ICU readmission, organ support, reoperation, death, discharge destination).
4. An **out-of-support alert** when the patient lies far from the reference cases.

### 1.1 Research questions

**[v0.5] Three separate experiments, one per target**, each complete (internal CV, sensitivity, external validation,
its own freeze and pre-registration) and in this order: **T2** (primary target and demonstration of the method),
**T1** (complications; plus whether the similarity learned for T1 resembles the one learned for T2 — neighbour
overlap, weight ranks, shapes), **T3** (both dimensions). The extraction is shared. Changes to a later experiment
informed by the external results of an earlier one are declared.

- **Q1 (method):** does the metric weighted by multi-model SAGE improve the kNN's log-loss and calibration compared with the unweighted distance and with direct metric learning (diagonal NCA)?
- **Q2 (performance):** is it competitive with reference parametric models (logistic regression, gradient boosting) and with the clinical SWIFT score?
- **Q3 (transport):** does the notion of similarity transfer between health systems (Korea → USA)? Answered by **transporting the metric weights, not the cases**.
- **Q4 (carry-over):** does the neighbourhood learned for the target also group the future variables that are not used as targets?

### 1.2 Stated contribution (do not overstate)

Weighting kNN with external importances **is not new** (Addou et al., 2026; Weerts et al., 2019). What is new is the combination of:

- SAGE importance aggregated over a set of competitive models, with a convergence-based stopping rule;
- an importance → weight conversion learned with a prior (formally, a diagonal NCA regularised towards the importance);
- a heterogeneous metric (scalar weights + VDM);
- probabilistic evaluation;
- transport of the metric across cohorts.

---

## 2. Data

### 2.1 Cohorts and roles

| Role | Database | Origin | Access | What it can validate |
|---|---|---|---|---|
| Development | INSPIRE (v1.4.2 or later) | SNUH, Korea, 2011–2020 | PhysioNet credentialed / KHDP | All targets |
| Full external validation | MIMIC-IV | BIDMC, USA | PhysioNet credentialed / KHDP | All targets |
| Partial external validation | eICU-CRD | Multicentre, USA | PhysioNet credentialed / KHDP | Severe / not severe and post-ICU length of stay only (eICU does not observe the ward) |

**Closed cohort set:** the project uses these three cohorts only. No others are to be sought or added.

### 2.2 Known particularities that affect the design

**INSPIRE**

- Tables: `operations`, `diagnosis`, `vitals` (intraoperative), `ward_vitals` (ward and ICU), `labs`, `medications`. Joined on `subject_id`; one `subject_id` → several `hadm_id` → several `op_id`. The data dictionary is in `schema.csv` and `parameters.csv` `[VERIFY exact names]`.
- **Times are relative, in minutes**, from the subject's first admission (time zero).
- `icuin_time` / `icuout_time` **only record ICU admissions within 24 h after surgery**. ICU readmissions outside that window **do not appear as such**. `[VERIFY how to detect ICU readmission: possibly via location in ward_vitals, device use or ventilation]` → **critical for the severe target**; if it cannot be detected, see §10.
- Labs from 6 months before admission to 6 months after discharge. Vital signs and labs are **discretised to 5-percentile bins**, with extremes clipped (p2.5–p97.5). Age in 5-year bins.
- Diagnoses: ICD-10-CM at 3 characters, **only from admission onwards**, with relative time. **There is no pre-admission history.** The code timestamp may be the billing time at discharge, so there is a leakage risk (§4.4).
- Procedure: ICD-10-PCS at 5 characters (section, body system, root operation, body part, approach).
- Devices (mechanical ventilation, renal replacement therapy, ECMO, intra-aortic balloon pump): derived start time; end inferred from 4–8 h without records.
- Medications with ATC codes (`atc_code`, `atc_code2`, `atc_code3`). Drugs given to <100 patients, chemotherapy and investigational drugs are excluded. Ward fluids are not included.
- In-hospital and all-cause mortality in `operations` `[VERIFY columns]`.

**MIMIC-IV** `[VERIFY all names against the version used]`

- `hosp.admissions` (`admittime`, `dischtime`, `deathtime`, `discharge_location`, `hospital_expire_flag`), `icu.icustays` (`intime`, `outtime`, `first_careunit`, `last_careunit`), `hosp.transfers`, `hosp.services`, `hosp.labevents`, `hosp.emar` / `hosp.prescriptions`, `hosp.microbiologyevents`, `hosp.procedures_icd`, `hosp.diagnoses_icd`, `icu.chartevents`, `icu.inputevents`, `icu.procedureevents`.
- **No ward vital signs** (chartevents is ICU only). Mild ward complications can only be observed through labs, medications and microbiology.
- `procedures_icd` has a date only, no time.
- Population limited to patients who passed through the ICU or the emergency department.

**eICU-CRD** `[VERIFY]`

- `patient` (`patientunitstayid`, `patienthealthsystemstayid`, `unitvisitnumber`, `unitadmitsource`, `unitdischargeoffset`, `unitdischargestatus`, `hospitaldischargeoffset`, `hospitaldischargestatus`, `hospitaldischargelocation`), `apachepatientresult` (includes APACHE IV length-of-stay predictions), `apachepredvar`, `admissiondx`, `lab`, `infusiondrug`, `medication`, `treatment`, `vitalperiodic`, `vitalaperiodic`.
- **Data exist only inside the ICU.** After ICU discharge, only ICU readmission (another stay under the same `patienthealthsystemstayid`), death, and the time and destination of hospital discharge are observed.
- Times are offsets in minutes from unit admission.

---

## 3. Population and temporal design

### 3.1 Inclusion criteria (v0 proposal)

- Adults (≥18 years).
- **Postoperative** ICU admission:
  - INSPIRE: `icuin_time` within 24 h after `opend_time`.
  - MIMIC-IV **[v0.4, D1/D8]**: timestamped operating-room evidence — an OR/PACU transfer in `hosp.transfers` ending within 24 h before the ICU admission. Only for hospitalisations without any OR/PACU transfer, a "major therapeutic" procedure code (HCUP procedure classes, or a declared approximation until they are provided) on the ICU admission day or the day before. A surgical service alone does not qualify (counted as a "service only" sensitivity cohort) `[VERIFY careunit values]`.
  - eICU: `unitadmitsource` from the operating room or post-anaesthesia recovery, or a postoperative admission diagnosis `[VERIFY values]`.
- **Discharged alive from the ICU.** ICU deaths are excluded by definition and reported in the flow diagram.
- Unit of analysis: **the first postoperative ICU stay of each hospitalisation**. Patients with several hospitalisations are handled through patient-level splitting (§6).

### 3.2 Exclusion criteria (v0 proposal)

- ICU stay <4 h (likely a recording error or extended recovery) `[open decision]`.
- ICU discharge directly out of the hospital (home, another centre, palliative care), because there is no ward follow-up.
- **[v0.4, C5]** ICU exit followed within 6 h by an operating room / PACU / interventional radiology / another ICU entry (MIMIC-IV `transfers`; INSPIRE: start of the next operation; eICU: next unit stay): t0 must be a discharge to the ward, otherwise the label is known at t0.
- Patients with a limitation-of-care decision **known before the cut-off**: **[v0.4, D12]** timestamped records ≤ cut-off (MIMIC-IV Code Status, eICU `carePlanGeneral` care limitation); INSPIRE only if a timestamped field appears.
- Organ donors and related procedures (INSPIRE flags them from v1.4.2).

### 3.3 Timing

- **t0 = ICU discharge time** (physical departure).
- **Input cut-off:** data with a timestamp **≤ t0 − 2 h**. The margin avoids including data generated by the discharge decision itself. `[open decision: 2 h vs 4 h; sensitivity analysis with both]`
- **Outcome follow-up:** from t0 to hospital discharge, **truncated at 30 days** after t0.
- If the 30-day cut-off truncates the stay, post-ICU length of stay is taken as ≥30 days (top category).

---

## 4. Variables

### 4.1 Inputs (available at t0 − 2 h)

| Block | Variables | Metric type (§5.2) |
|---|---|---|
| Demographics | age, sex, BMI | continuous / binary |
| Preoperative baseline | ASA, surgical urgency, most recent preoperative labs | ordinal / binary / continuous |
| Surgery | procedure split into ICD-10-PCS levels (INSPIRE), harmonised surgical group, duration of surgery, type of anaesthesia | hierarchical / continuous / nominal |
| ICU stay | ICU length of stay; use of ventilation, vasopressors and renal replacement therapy during the stay; hours from withdrawal of each support to t0 | continuous / binary |
| Status at discharge | last labs (creatinine, Hb, white cells, platelets, lactate, bilirubin, albumin, blood gases if available), last vital signs and their trend over the 24 h before the cut-off (HR, MAP, RR, SpO₂, temperature), last FiO₂ / oxygen support, GCS if available | continuous |
| Diagnoses | one binary per 3-character ICD-10 code, **only with a timestamp before the cut-off** and prevalence ≥0.5% `[open decision: threshold]` | binary |
| Active medication | binaries per ATC level-3 group administered in the 24 h before the cut-off. **[v0.4, D6]** Where the medication source does not cover the stay (MIMIC-IV without eMAR; eICU without medication records), these inputs are unobservable (missing), never 0 | binary |

**Harmonisation across cohorts:** the final input set is the **intersection** of what is available in the three cohorts, plus an extended set for INSPIRE and MIMIC-IV only if it adds value. Document every variable in `config/harmonization.yaml`: source in each database, units, transformation and status `[VERIFIED]` / `[VERIFY]`.

**Discretised INSPIRE:** when harmonising, **do not** discretise MIMIC or eICU to mimic INSPIRE in the main analysis. As a sensitivity analysis, apply the same percentile discretisation (computed in the development cohort) to MIMIC and eICU.

### 4.2 Targets

#### T1 — Post-ICU complication (3 classes, ordinal)

Approximation of the Clavien-Dindo classification (Dindo et al., 2004) **from structured data**, never from diagnosis codes. Window: (t0, t0 + 30 d] and before hospital discharge.

- **Severe** (≈ grades III–V), if any of:
  - ICU readmission. **[v0.4, D10]** INSPIRE (ICU stays recorded only after surgery): a new ICU entry after another surgery OR a new invasive ventilation / RRT / ECMO / IABP run after t0 in `ward_vitals` (SPECS D1 proxy; ward dialysis of a chronic-RRT patient excluded); its prevalence is checked in the audit before use.
  - In-hospital death.
  - Surgical reoperation (INSPIRE: new `op_id` in the same hospitalisation; MIMIC-IV **[v0.4, D1]**: a later OR/PACU transfer after t0, minute precision; only for hospitalisations without OR/PACU transfers, a major therapeutic procedure dated after t0, day precision).
  - New mechanical ventilation, vasopressors or renal replacement therapy.
  - Stage 3 AKI by KDIGO creatinine criteria (KDIGO, 2012), using the last creatinine before the cut-off as baseline.
- **[v0.4, D7] KDIGO timing:** an AKI event at a post-t0 creatinine v(t) when v ≥ 1.5 × the minimum of the previous 7 days or v ≥ 0.3 mg/dL above the minimum of the previous 48 h; the baseline always belongs to the 7-day reference set (and to the 48-h set during the first 48 h after t0). Stage by the maximum ratio to the baseline (≥ 2 → stage 2, ≥ 3 → stage 3); ≥ 4.0 mg/dL → stage 3 only at a detected rise and with a baseline below 4.0 **[D5]**. Creatinine measured during RRT (or within 24 h after a run) is never used.
- **[v0.4, D5] Chronic dialysis:** for patients with RRT before the index surgery (device / procedure record or ESRD / dialysis-dependence code, timestamped before the surgery) AKI and new RRT are **not applicable**: they do not count, like a component that the cohort cannot observe.
- **Mild** (≈ grades I–II), if not severe and any of:
  - Stage 1–2 AKI by creatinine (KDIGO, same baseline).
  - Start of a **new systemic antibiotic course** (ATC J01): first administration after t0 + 24 h, no antibiotic in the 48 h before that administration, and at least 2 consecutive days of administration. `[VERIFY feasibility in INSPIRE and MIMIC-IV]`
  - ~~Red blood cell transfusion after t0~~ — **removed from T1 in every cohort [lead, 2026-10-24]**: INSPIRE does not record it (completeness audit: prevalence < 0.5 %); it remains a carried variable.
  - *INSPIRE sensitivity analysis only:* sustained ward hypoxaemia (SpO₂ <90% on ≥2 consecutive measurements), because MIMIC-IV has no ward vital signs.
- **None:** everything else.
- **[v0.4, D6] Unobserved mild components:** where the medication source does not cover the hospitalisation, the antibiotic-course component is unobservable and T1 (and T3) is **undefined** for the stay, whatever its other outcomes (a coverage-based exclusion, never an outcome-based one); severe / not severe remains defined. eMAR coverage by period is reported in the audit. **[2026-10-24]** MIMIC-IV hospitalisations without any eMAR record (the early years) take their medication from `prescriptions` (orders, start–stop), so only those with neither source stay undefined; an order covers its whole interval in the antibiotic-course rule.

**In eICU:** T1 reduces to **severe / not severe**. "Severe" includes only ICU readmission and in-hospital death, the observable components. It must be reported as validation of a reduced target, not of the full T1.

**Mandatory robustness:** sensitivity analyses with (a) T1 without the antibiotic component, (b) T1 without AKI, (c) alternative creatinine thresholds.

#### T2 — Post-ICU hospital length of stay

- Definition: days from t0 to hospital discharge, **in survivors only**. Death is already part of severe T1. *(v0.3)*
- **[v0.5] Death as a fourth class:** short / medium / long (tertiles of the survivors' stay) / in-hospital death within 30 days, **ordinal with death as the worst state**. At t0 it is unknown who will survive, so T2 is defined for every patient; the survivors-only selection disappears.
- **Cohort-relative** categories: tertiles computed in the training set of each cohort (short / medium / long), on the survivors' stay. This absorbs structural length-of-stay differences between health systems.
- Sensitivity: absolute categories (≤3 / 4–7 / >7 days) `[open decision]`.

#### T3 (secondary) — Composite ordinal trajectory

A single ordinal variable: **early discharge without complication** < **expected without complication** < **prolonged or with mild complication** < **severe complication or death**. Serves as an integrated kNN objective and as a secondary analysis.

### 4.3 Carried future variables (never inputs)

ICU readmission, ventilation, vasopressors, renal replacement therapy, reoperation, death, discharge destination, each individual T1 component and the continuous length of stay.

They are stored with each reference case and summarised over the k neighbours. They are used for:

- the clinical output (§5.6);
- the Q4 evaluation;
- *optionally*, as **privileged information** during training (Vapnik & Vashist, 2009): SAGE computed to predict the target **and** the future variables, combined (§5.3).

### 4.4 Anti-leakage rules (non-negotiable)

1. No input may have a timestamp > t0 − 2 h. Implement an **automated test** that checks this for every row.
2. Diagnoses in INSPIRE: if it cannot be guaranteed that the timestamp reflects when the diagnosis became known, use **only diagnoses timestamped before the start of the index surgery** (main analysis) and all those before the cut-off (sensitivity).
3. Nothing decided at the moment of discharge may be an input: discharge destination, assigned ward, limitation-of-care orders recorded at discharge.
4. Every fit that uses the target (weights, VDM, SAGE, λ, α, T2 category thresholds; k is not tuned, see §5.5) is done **inside the training fold**.
5. External validation cohorts are **not touched** until the pipeline is frozen (§6.3).

---

## 5. Method

### 5.1 Distance

For patients a and b:

```
d(a, b)² = Σ_j  w_j · δ_j(a_j, b_j)²
```

- **δ_j:** shape of the dissimilarity, by variable type (§5.2). Each δ_j is normalised to unit variance in training so that w_j keeps its meaning.
- **w_j ≥ 0:** scale (importance), derived from SAGE and adjusted as in §5.4.
- **[v0.5]** `d(a, b)² = Σ_j w_j ‖g_j(a_j) − g_j(b_j)‖²`: each variable lives on its **shape function** g_j — the consensus, across the competitive models of §5.3, of how its values move the prediction (centred log-odds, one component per class). g_j says *how* a variable relates to the outcome, w_j *how much* it counts; both come from multivariable models, and the kNN captures the remaining interactions by locality. Neighbours are therefore patients who reach the same prediction for the same reasons (profiles).

### 5.2 Variable types and δ_j

**[v0.5] Shape geometry (replaces the per-type rules below, which remain as the HVDM comparator):**

- **Continuous / ordinal:** ≤ 20 training quantile bins of ≥ 50 patients; the curve of each model is read by accumulated local effects (ALE) — only real patients, moved within their own bin — so correlated variables do not leak into it; constant extension beyond the observed range.
- **Nominal / ICD-10-PCS:** categories (codes) with ≥ 50 patients; rare PCS codes are pooled with their parent code when that pool is large enough, otherwise into "other"; the curve of each model is the local effect of each category against the most frequent one, averaged over the patients of that category. Never-seen categories → neutral (zero) contribution.
- **Binary and missingness indicators:** 0/1 (their shape is equivalent after normalisation). Missing values keep median imputation + indicator.
- **Consensus:** least-squares fit to all the models' curves with a per-model scale; agreement R² reported; each value shrunk towards the mean by τ² / (τ² + se²) (DerSimonian–Laird over the values, as the SAGE aggregation); a curve not distinguishable from flat (homogeneity F test with the between-model variance pooled over values, α = 0.05 with Bonferroni over classes) keeps its **neutral axis**: rank (empirical CDF) for continuous / ordinal, one-hot for nominal / PCS. Each model's curves are read on a fresh random subsample of 1,000 explanation cases (bootstrap-like).
- Everything is estimated inside the training fold, on the explanation subsample (cases not used to fit the models); the in-fold vs test optimism is reported; no cross-fitting (it would break the common space of the kNN).

| Type | Examples | δ_j |
|---|---|---|
| Continuous | labs, vital signs, age, duration | difference of standardised values (training mean and SD) |
| Ordinal | ASA | rank difference, scaled to [0, 1] |
| Binary | diagnoses, support use, ATC | 0/1 |
| Hierarchical | ICD-10-PCS (5 levels) | **one dimension per level**, 0/1 for a match at that level; each level has its own w_j. **[v0.4, B6]** rare / unseen codes share an explicit "other" category and missing codes a "missing" one, so a kept code never half-matches a rare one |
| Single-valued nominal | type of anaesthesia, surgical group, admission source | **smoothed VDM** (below) |

**Smoothed VDM (Stanfill & Waltz, 1986; Wilson & Martinez, 1997):**

- For category c: `p_c = (n_c · P̂(y|c) + m · P(y)) / (n_c + m)` (m-estimate towards the global prevalence).
- `δ_j(c, c') = ‖p_c − p_c'‖₂`.
- m is chosen in the inner cross-validation (grid, e.g. {1, 5, 20, 50}).
- Categories unseen in training use `P(y)`.
- If privileged information is used, `P(y|c)` concatenates the distributions of the target and of the binary future variables.
- **Always computed inside the fold.**

**Diagnoses:** treated as **independent categories** (lead's decision): one binary per code, no hierarchy.

**Missing values:**

- Continuous: training-median imputation + binary missingness indicator as an extra dimension, if missingness is >5%.
- Binary presence variables: missing = 0 **when the source is observed**. **[v0.4, D6]** Where the source does not cover the stay, the value is unobservable: training-prevalence imputation + missingness indicator if >5%.
- **[v0.4, B7]** Nominal categories seen in training but rare are pooled into one "other" category; categories never seen in training use `P(y)`.
- Document the percentage missing per variable and cohort (§7).

### 5.3 Importance: SAGE aggregated across models

**Method references:** Covert et al. (2020) for SAGE; Frye et al. (2021) for the masked surrogate; Ning et al. (2022) for aggregation over a Rashomon set.

1. **Model space**, defined before seeing results and stored in `config/model_space.yaml`: logistic regression (L2), random forest, LightGBM, XGBoost, MLP. Each family with hyperparameter ranges. **[v0.5]** The MLP is replaced by a penalised **spline logistic regression** (smooth non-linear shapes at low cost) and the logistic regression samples its penalty (**L2 / LASSO / elastic-net**). Shapes per family: straight lines, smooth curves, steps. No Bayesian family (the consensus already averages models). Accepted models that fall outside (1+ε)·best as the loop improves leave the consensus (AUDIT B8).
2. **Sampling:** at each iteration, a family and hyperparameters are sampled at random.
3. **Training with *feature dropout*:** each model is trained with random masks over **groups** of variables (§5.3.1). It thus acts as a surrogate able to predict with any subset. Models are *not* retrained per subset.
4. **Competitiveness filter:** a model is accepted only if its log-loss on the inner validation fold is ≤ (1 + ε) × the best log-loss observed so far. ε = 0.05 `[open decision]`.
5. **SAGE** on log-loss, with the permutation estimator, on an explanation subsample of 2,000–5,000 cases from the inner validation fold. Absence of a variable is simulated with the surrogate's mask, not with marginal imputation.
6. **Normalisation:** SAGE values ≥0 (negatives truncated to 0) divided by their sum.
7. **Aggregation:** inverse-variance weighted mean across models, ShapleyVIC-style. The full model × variable matrix is also stored to report heterogeneity.

#### 5.3.1 Variable groups

SAGE is computed by **groups** rather than single variables when:

- they are levels of the same hierarchical code (ICD-10-PCS levels form one group for dropout, but each level gets its own weight in §5.4);
- they are a value and its missingness indicator;
- they are the diagnosis or ATC block, if it exceeds 50 dimensions (group SAGE with internal split proportional to prevalence × univariate SAGE) `[open decision]`;
- they form a cluster of variables with |ρ| > 0.8 in training.

#### 5.3.2 Stopping rule: stability of the learned distance

Learning stops on the **stability of the metric**, not of the importances. That is what matters for the kNN, and it does not depend on any k.

At each iteration n, after adding an accepted model:

1. Update the aggregated importance `imp^(n)`.
2. **[v0.4, B3]** Recompute the weights implied by the prior, `w^(n) = c · (p^(n) + ε₀)`, where `p^(n)` is the per-variable split of `imp^(n)` (the split used as prior in §5.4) and c is any global scale (it cancels in Δ). No NCA runs inside the loop, so Δ measures the stability of the consensus rather than optimiser noise; the §5.4 conversion is applied once, after stopping. *(v0.3: "recompute w^(n) with the conversion of §5.4, warm-starting from w^(n−1)".)*
3. Measure the **change in the metric** over a fixed reference set R of patient pairs (e.g. 20,000 random pairs from the training fold, fixed with a seed at the start):

   ```
   Δ_n = median_{(a,b)∈R} | d^(n)(a,b) − d^(n−1)(a,b) | / d^(n−1)(a,b)
   ```

   This is the relative change across all distances in the reference set, with no neighbour selection, so it does not depend on k.
4. **Early stopping on the Δ curve**, just as with a loss curve:
   - Δ_n is noisy from one iteration to the next, so it is smoothed with a moving median over the last W iterations: `Δ̃_n = median(Δ_{n−W+1}, …, Δ_n)`.
   - Track the best value reached: `Δ̃_best = min(Δ̃_1, …, Δ̃_n)`.
   - Iteration n counts as an improvement if `Δ̃_n < Δ̃_best − min_delta`.
   - **Stop after P consecutive iterations without improvement (patience)**, i.e. when the Δ curve has stopped decreasing and has plateaued.
5. The final metric is `w^(n)` at the stopping point. Because the aggregated importance is a cumulative mean, the latest metric is the most stable one, so there is no rollback to the best-value iteration (unlike early stopping in a neural network).

**Plateau diagnostic (reported, does not stop):** the level of Δ̃ at stopping indicates how much the metric still moves with each new model. If the plateau is above `tol_Δ` (e.g. 1% mean relative change), log it in `DECISIONS.md` as a stabilised metric with high residual noise: this signals genuine heterogeneity across models, not a lack of iterations.

**Limits:**

- Minimum n_min = 20 accepted models before the stopping rule is evaluated.
- Maximum n_max = 150 models. If reached without stopping, log that convergence was not achieved.

**What is reported:**

- the Δ_n and Δ̃_n curves, with the stopping iteration marked;
- the stability of the importance ranking (Spearman with the vector W iterations earlier) as complementary information.

**W, P, min_delta, tol_Δ, n_min, n_max and ε are fixed before execution**, using the pilot in §11, and are not tuned after seeing results.

### 5.4 Importance → weight conversion

**Primary:** diagonal NCA regularised towards the SAGE prior.

```
L(w) = LogLoss_LOO_softkNN(w) + λ · Σ_j (log w_j − log(imp_j + ε₀))²          (v0.3)
```

**[v0.4, B2/B4]** Parameterised as `log w_j = log c + log(p_j + ε₀) + u_j`, with c a free, unpenalised global scale (the bandwidth) and the prior acting on the SAGE **group totals**:

```
L(w) = LogLoss_LOO_softkNN(w) + λ · (1/G) Σ_g (log Σ_{j∈g} w_j − log c − log(imp_g + ε₀))²
```

so the NCA itself splits the weight within a group (an uninformative missingness indicator no longer inherits the prior of its value variable), c moves freely at any λ, and λ (a mean over groups) keeps its meaning whatever the number of variables (grid {0, 10⁻³, 10⁻², 10⁻¹, 1, 10}).

- `LogLoss_LOO_softkNN`: class probability from a softmax kernel over weighted distances, excluding the case itself (Goldberger et al., 2004).
- **The metric is learned without k.** The soft-kernel objective uses all cases weighted by proximity; the bandwidth is absorbed into the scale of w. **The metric is never tuned for a particular k**: k is an execution parameter only (§5.5).
- **Scalability:** the full version is O(n²). Implement it with **anchor minibatches** and **neighbour candidates** precomputed by approximate search (FAISS, CPU) using the prior weights, refreshed every E epochs. The number of candidates is a generous computational parameter (e.g. 500), not a prediction k.
- λ is chosen in the inner cross-validation on a logarithmic grid, using the same soft-kernel log-loss (independent of k), including λ = 0 (pure diagonal NCA, which is also a comparator). **[v0.4, C3]** The SAGE prior used to score each inner validation fold is recomputed on that inner fold's training part (≈ 3× the loop cost, quantified by the pilot), so it never saw the labels it is scored on.
- Optimisation in log w space (Adam).
- *(v0.3: "within the loop of §5.3.2, each update of w warm-starts from the previous iteration's weights" — no longer applies, see §5.3.2 [v0.4, B3].)*
- Report the chosen λ: it is a result in itself (how much the data trust SAGE).

**Simple alternative (comparator):** `w_j = a · imp_j^γ`, with a and γ fitted on the same soft-kernel log-loss.

### 5.5 Inference

1. Transform the new case with the training parameters (standardisation, VDM tables).
2. Compute the weighted distance to all reference cases (exact or approximate FAISS index, with exact verification of the k candidates).
3. Take the k neighbours. **k is an execution parameter, not a learning parameter:** the metric is the same for any k.
   - Performance is evaluated over the grid k ∈ {10, 20, 50, 100, 200} and the full curve is reported.
   - The main analysis uses a **pre-specified primary k** (proposal: k = 50) `[open decision]`.
   - **[2026-10-25, secondary, added after the first CV folds had been seen]** k sweep: each outer fold's metric is queried again over k = 1…2000 (at most 20 % of the case base), with α refitted per k by the same patient-level LOO and nothing else refitted. It reports the full curve of every kNN method, each k against the primary k, the comparators against PERISIM at the same k, and k chosen by the LOO log-loss of the training fold against the primary k. Descriptive: the primary k and the confirmatory comparisons are unchanged (`python -m perisim ksweep`).
4. Class probability: `(n_c + α · P(c) · C) / (k + α · C)`, with α chosen in the inner cross-validation. **[v0.4, C6, declared deviation]** α (and the OOD threshold) are chosen by patient-level leave-one-out on the training fold; with large categories the optimism is negligible (revisit if the audit shows many small nominal categories). Kernel-weighted variant exp(−d²) as a sensitivity analysis.
5. **Out of support:** if the distance to the k-th neighbour exceeds the 99th percentile of that same distance in training (computed leave-one-out), flag `ood = True`.

### 5.6 Per-patient output

```json
{
  "patient_id": "...",
  "t0": "...",
  "p_T1": {"none": 0.0, "mild": 0.0, "severe": 0.0},
  "p_T2": {"short": 0.0, "medium": 0.0, "long": 0.0},
  "k": 0,
  "ood": false,
  "kth_distance_percentile": 0.0,
  "neighbors": [{"case_id": "...", "distance": 0.0}],
  "carried_summary": {"icu_readmission": 0.0, "ventilation": 0.0, "death": 0.0, "median_los": 0.0}
}
```

---

## 6. Validation

**[v0.5] Sensitivity analyses:** variants that change only the outcome definition (T1 without antibiotics / AKI, alternative KDIGO, ward hypoxaemia, T2 absolute categories) are evaluated by **relabelling** the case base with the main metric of each outer fold (α refitted); variants that change the inputs (4 h margin, diagnosis window) are refitted in full, for the experiment's target only.

**[v0.4, E10/E11]** Pooled out-of-fold metrics are reported next to per-fold metrics and per-fold T2/T3 cut-points, and SAGE dispersion is reported per variable (fold-specific groups mapped to their variables). **[v0.4, E14]** The §5.6 record is written inside the pod for every test-fold and external query (never exported).

### 6.1 Internal (INSPIRE)

- **Patient-level split** (`subject_id`), never by surgery or by stay.
- **Nested cross-validation:** 5 outer folds × 3 inner folds.
- Everything in §5 that learns from data (SAGE, weights, λ, α, m, T2 thresholds, standardisation) is fitted in the inner folds. **k is not tuned:** it is evaluated on the §5.5 grid with the pre-specified primary k. kNN-based comparators follow the same k protocol.
- **Additional temporal validation:** train 2011–2018, test 2019–2020 `[VERIFY that relative times allow the year to be reconstructed; otherwise omit]`.

### 6.2 External: two modes

- **Mode A, full transport:** weights, VDM tables and **INSPIRE case base**; queries from MIMIC-IV / eICU. Measures how much the whole system transfers.
- **Mode B, metric transport:** INSPIRE weights, but **case base from the target cohort**, evaluated with patient-level leave-one-out within the target cohort. VDM tables recomputed in the target cohort (with cross-validation) as a variant. Measures whether the notion of similarity transfers even when outcomes differ.
- The difference between A and B is a central result (Q3).
- **[v0.4, C4] T2/T3 categories in the external cohorts:** the target cohort's tertiles are **cross-fitted** — two patient-level halves, each labelled with the cut-points of the other half (mode B case bases carry the same labels, and the smoothing prior leaves out the query's patient). Absolute categories (≤3 / 4–7 / >7 days) are a **co-primary** external analysis, with the INSPIRE case base relabelled the same way (same learned metric).

### 6.2b Transportability, incremental case base and model-transfer validations [amendment, 2026-10-24/25]

Pre-registered as an amendment before the external validation and the freeze, approved by the lead:
[docs/prereg_amendment_transport.md](docs/prereg_amendment_transport.md) (hypotheses H1–H6 and margins) and
[docs/transport_review.md](docs/transport_review.md) (design). Stage `transport`, after `external`, on the frozen
pipeline; it contains the model-transfer validations decided on 2026-10-24. Central hypothesis: the **geometry**
(frozen g_j, w_j) transports between sites and the **calibration** comes from the local case base.

- Per external cohort, patient-level and seeded: a fixed **test** set (20 % of the patients), a **tuning** set (≤ 500
  patients) and the **local pool** for case bases, disjoint by patient. Target-cohort T2 / T3 tertiles cross-fitted
  between two patient halves (the same labels for every case base); absolute categories co-primary.
- **A.** Decomposition of the transport loss on the same case base and test: frozen INSPIRE metric; INSPIRE g_j with
  w_j refitted locally (NCA with prior at INSPIRE's w_j, INSPIRE's λ); full local PERISIM.
- **B.** Case-base composition (local; INSPIRE ∪ local; INSPIRE ∪ other external ∪ local; INSPIRE only) with a site
  weight γ ∈ {1, 0.5, 0.25, 0} on the non-local votes and prior counts and the schedule γ(n) = 500 / (500 + n);
  Euclidean kNN and prevalence of the local pool; PERISIM refitted on INSPIRE ∪ eICU for MIMIC-IV (T2).
- **C.** Accumulation curves: local patients added in batches (0 … all), starts empty / INSPIRE / INSPIRE ∪ other,
  calendar-block order (random inside) and random order, 20 repetitions.
- **D.** α (and k) tuned on 100 / 250 / 500 local patients; logistic recalibration as a sensitivity; calibration of
  every cumulative threshold of T2.
- **E.** Shapes re-estimated in each site with a reduced consensus (40 models) on INSPIRE's frozen value grid;
  between-site homogeneity (Cochran Q, I²) after one robust site-level scale; range coverage and missingness.
- **F.** eICU by hospital (≥ 300 index stays): B and C per hospital; between-hospital vs between-country spread.
- **H1–H6** (confirmatory for T2, descriptive for T1 / T3): calibration in the large of the local mode within
  ±0.10 and not of mode A; slope ≥ 0.90; AUROC of the frozen vs the local metric non-inferior (margin 0.02); more
  out-of-support queries in mode A; a crossing point of the accumulation curve; non-transporting shapes mostly
  practice / coding variables. The hashes of the frozen artefacts are checked at the start and the end.

Outputs (inside the pod): `<workdir>/experiments/<t>/results/transport/` (CSV and figures), `reports/transport.html`.

### 6.3 Freezing

Before touching MIMIC-IV and eICU:

- tag the pipeline version in git;
- store the final configuration;
- pre-register the analysis (OSF or equivalent) `[open decision]`.

Iterating on the method after looking at external results is forbidden.

### 6.4 Comparators (all with the same splits and the same nested fitting)

0. **[v0.5] Trivial references:** the training prevalence of each class, and the prevalence by surgical group (m-estimate towards the overall); externally, INSPIRE's prevalence (mode A) and the target cohort's own prevalence without the query's patient (mode B).
1. Euclidean kNN on standardised variables (no weights). **[v0.5]** On the neutral axes (rank / one-hot), i.e. without any outcome information.
1b. **[v0.5]** The v0.4 geometry (HVDM: standardised continuous + VDM nominal) with the same SAGE prior and λ: isolates the effect of the shape functions.
2. kNN with single-model weights: mean |SHAP| from LightGBM (Weerts et al., 2019).
3. kNN with random forest importance (Addou et al., 2026).
4. kNN with ReliefF weights.
5. Diagonal NCA (λ = 0).
6. kNN with RF-GAP proximities (Rhodes et al., 2023).
7. Multinomial logistic regression (L2).
8. Multiclass LightGBM, recalibrated. **[v0.4, E13]** Tuned in the inner cross-validation (small num_leaves × learning-rate grid, early stopping on the inner folds).
9. SWIFT (Gajic et al., 2008) for the severe / not-severe target, computed in all three cohorts `[VERIFY available variables]`.
10. In eICU, for length of stay: APACHE IV hospital length-of-stay prediction, as a descriptive reference (not the same estimand as T2: this must be stated).

### 6.5 Metrics

- **Primary:** multiclass log-loss and multiclass Brier score.
- **Ordinal (T1, T3):** Ranked Probability Score.
- **Calibration:** one-vs-rest curves per class, with per-class calibration intercept and slope. ECE only as a complement.
- **Discrimination (secondary):** one-vs-rest AUROC and AUPRC for the severe class.
- **Confidence intervals:** patient-level bootstrap (2,000 replicates).
- **Paired comparisons** between methods: case-level paired bootstrap of log-loss differences.
- **Q4 (carry-over):** for each future variable, log-loss and Brier score of its prediction from the neighbours, comparing the learned metric with the Euclidean one.

### 6.6 Reporting

Follow **TRIPOD+AI** (Collins et al., 2024). Include:

- flow diagram per cohort;
- characteristics table;
- missingness table;
- importances with between-model dispersion;
- SAGE / metric convergence curve;
- chosen λ;
- all sensitivity analyses.

---

## 7. Completeness audit (first technical deliverable)

Before any modelling, generate `reports/completeness_{cohort}.html` with:

- the number of patients passing each inclusion and exclusion criterion (flow diagram);
- for each input and each target: % missing, distribution and units;
- ward measurement frequency of creatinine, blood count and antibiotics (determines the feasibility of the mild category);
- the prevalence of each T1 component and of each T1 and T2 class;
- the feasibility of detecting ICU readmission in INSPIRE (§2.2).

If any T1 component has prevalence <0.5% or missingness >30% in any cohort, **stop and log it in OPEN_QUESTIONS**.

---

## 8. Implementation

### 8.1 Stack

Python ≥3.11, `polars` (or `pandas`), `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `torch`, `faiss`, `sage-importance`, `scipy`, `pyyaml`, `pytest`. Experiment tracking: `mlflow` or one JSON file per run.

### 8.2 Repository structure

```
perisim/
├── SPECS.md
├── docs/
│   ├── OPEN_QUESTIONS.md
│   └── DECISIONS.md          # dated decision log
├── config/
│   ├── cohorts.yaml           # inclusion and exclusion criteria per cohort
│   ├── harmonization.yaml     # variable map across cohorts
│   ├── targets.yaml           # operational definitions of T1, T2, T3
│   ├── model_space.yaml       # model space for SAGE
│   └── experiment.yaml        # seeds, folds, tolerances, grids
├── src/perisim/
│   ├── data/                  # per-cohort extraction → common format
│   ├── cohort/                # inclusion, exclusion, t0, temporal cut-off
│   ├── features/              # input construction, metric types
│   ├── targets/               # T1, T2, T3, carried variables
│   ├── importance/            # dropout surrogates, SAGE, aggregation, stopping
│   ├── metric/                # δ_j, VDM, learning w (regularised NCA)
│   ├── knn/                   # index, inference, out-of-support, output
│   ├── baselines/             # comparators §6.4
│   └── evaluation/            # metrics, bootstrap, calibration, reporting
├── tests/
│   ├── test_no_leakage.py     # all inputs ≤ t0 − 2 h
│   ├── test_patient_split.py  # no patient in two folds
│   └── test_vdm_in_fold.py
└── reports/
```

### 8.3 Common intermediate format

Each cohort is transformed into four common tables:

- `stays`: one row per index stay, with identifiers, t0, cohort and static variables;
- `inputs_long`: stay, variable, value, timestamp;
- `outcomes`: stay, component, event timestamp;
- `carried`.

**Nothing downstream may read the original tables.**

### 8.4 Conventions

- Fixed, logged seeds for everything random.
- No hard-coded original column names outside `src/perisim/data/` and `config/harmonization.yaml`.
- Each stage writes its artefacts tagged with the hash of the configuration used.
- Pure functions wherever possible; fits receive the fold indices explicitly.

---

## 9. Platform constraints and governance

- **KHDP:** work with platform data is done inside the secure research environment. Only **aggregates** are exported (metrics, w_j weights, VDM tables with cells of n ≥ 10, figures), after whatever review the platform requires. **Never patient-level rows.**
- **PhysioNet:** comply with each database's data use agreement. Do not upload data to external services, **including cloud LLMs**. The LLM agent works with code, schemas and aggregates, never with patient data.
- **Local development:** working locally with public INSPIRE (PhysioNet) and MIMIC-IV / eICU under their data use agreements is allowed, on machines that meet their security requirements.

### 9.1 Compute resources (fixed)

KHDP research environment: **16 vCPU, 128 GB RAM, 100 GB storage, no GPU.** The whole pipeline must run on CPU within these limits.

**Design consequences:**

- **Models:** LightGBM and XGBoost in `hist` mode on CPU. The MLP in the model space is limited to 2 small hidden layers (PyTorch on CPU). If its cost dominates in the pilot, it is dropped from the model space and logged in `DECISIONS.md`. **[v0.5]** The MLP is no longer in the model space; any family costing > 3× the median of the others in the pilot is excluded.
- **[v0.5] Budget and cost levers:** the pilot estimates the hours of the experiment (CV + sensitivity + external); if a budget is set and exceeded, the pipeline stops and lists the pre-declared levers, to apply in this order: (1) smaller curve subsample, (2) shapes only for the variables covering 95 % of the SAGE importance, (3) λ prior from the outer fold instead of per inner fold (C3). The consensus loop is checkpointed after every accepted model, so stopping the pod costs minutes.
- **Parallelism:** at most 4 concurrent jobs with 4 threads each. Do not oversubscribe cores: set `OMP_NUM_THREADS` and `n_jobs` explicitly.
- **SAGE:** explanation subsample of 2,000 cases and grouped variables (§5.3.1) by default. n_max = 150 models.
- **Metric learning:** PyTorch on CPU with minibatches. FAISS on CPU (`IndexFlatL2` over vectors scaled by √w; at this size no approximate index is needed for inference).
- **Memory:** read large tables (MIMIC-IV `chartevents`, eICU and INSPIRE vital signs) lazily (`polars.scan_*`), filtering items and columns before loading. Never load a full unfiltered table.
- **Storage (100 GB):**
  - convert extracts to compressed Parquet (zstd), with only the needed columns and items;
  - do not duplicate original tables;
  - do not store the SAGE-loop models: keep only their importance vectors, metrics and seed, enough to reproduce them;
  - check disk usage at the end of each stage.
- **Mandatory pilot** before the full pipeline: 1 outer fold and T1, to measure time per model, time per SAGE, number of models until stopping and peak memory, and to extrapolate the total cost. **[v0.5]** One pilot per experiment, on its own target.

---

## 10. Open decisions

| # | Decision | v0 proposal | Depends on |
|---|---|---|---|
| D1 | ICU readmission detection in INSPIRE | Via location or devices in `ward_vitals` | Audit (§7). If not feasible, severe T1 in INSPIRE without readmission and a declared sensitivity analysis |
| D2 | Cut-off margin | 2 h (sensitivity 4 h) | Clinical validation |
| D3 | Minimum ICU stay | ≥4 h | Audit |
| D4 | Components of the mild category | AKI 1–2, new antibiotic course (transfusion removed 2026-10-24) | Audit; clinical validation |
| D5 | T2 categories | Per-cohort tertiles | — |
| D6 | Diagnosis prevalence threshold | 0.5% | Audit |
| D7 | Stopping-rule parameters (ε, W, P, min_delta, tol_Δ, n_min, n_max) | ε = 0.05; W = 5; P = 10; n_min = 20; n_max = 150; min_delta and tol_Δ from the pilot | Pilot |
| D8 | Privileged information in SAGE | Secondary analysis | — |
| D9 | Pre-registration | OSF | — |
| D10 | Postoperative ICU rule in MIMIC-IV and eICU | §3.1 | Audit |
| D11 | Primary k for the main analysis | k = 50 (full curve over the grid) | Clinical validation |

---

## 11. Milestones

1. **M1:** extraction to the common format + completeness audit of the three cohorts (§7). Close D1, D3, D4, D6 and D10.
2. **M2:** construction of targets and carried variables + anti-leakage tests passing.
3. **M3:** SAGE and metric pilot (1 fold, T1) in the KHDP environment. Close D7. Compute estimate with the resources in §9.1.
4. **M4:** full pipeline in INSPIRE with nested cross-validation + comparators.
5. **M5:** freezing (§6.3) and external validation in MIMIC-IV and eICU (modes A and B).
6. **M6:** sensitivity analyses, TRIPOD+AI report and manuscript.
