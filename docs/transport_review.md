# Transport analysis — Phase 0 review and full experiment map

Status: **approved by the lead on 2026-10-25 as proposed (§8, margins included) and implemented in
`src/perisim/transport/` (stage `transport`).** Nothing has been run on the external cohorts yet.
Companion: [prereg_amendment_transport.md](prereg_amendment_transport.md) (hypotheses, margins, analysis plan).

Central hypothesis under test: the **geometry** (frozen consensus shapes g_j and weights w_j) transports between
sites, while the **calibration** comes from the local case base. The distance depends only on differences of
centred g_j and the kNN only on the order of the neighbours, so a shift of baseline risk or a uniform shrinkage of
all effects leaves the metric unchanged; if the metric degrades, it should show as compressed predictions
(calibration slope > 1, lower AUROC) rather than as a global calibration error.

## 1. State of the freeze and of the external validation

| Item | State | Consequence |
|---|---|---|
| Freeze (§6.3) of experiment T2 | not done | the analyses below can be **pre-registered** (amendment) and frozen with the rest |
| External validation (MIMIC-IV, eICU) | **not run** | per the rules: amendment written first, review by the lead, nothing run on external cohorts before |
| Pilot of T2 | running (60 models done) | its results fix D7 (min_delta) and the cost estimate |
| OSF registration | pending (registered at `freeze --registration`) | the amendment goes into that registration |

Everything below is therefore **confirmatory if approved before the freeze**, except where marked exploratory.

## 2. What already exists in the repository

| Piece | Where | Notes |
|---|---|---|
| Mode A (INSPIRE case base, frozen metric) | `external.py` (`A_perisim`) | per external cohort, cross-fitted tertiles + absolute categories |
| Mode B (frozen metric, target-cohort case base, patient-level LOO) | `external.py` (`B_perisim`, `_mode_b`) | uses INSPIRE's α |
| Discretized-input sensitivity | `external.py` (`discretize_like`) | |
| Cross-fitted target-cohort tertiles (C4) | `external.py` (`patient_halves`) | two patient halves, each labelled with the other's cut-points |
| Transfer validations (80 % local / 20 % held out) | `transfer.py` (stage `transfer`) | `A_transport`, `T_local`, `T_pooled`, `L_perisim`, `L_euclidean`, `L_prevalence`, multi-source `S_*` (INSPIRE ∪ eICU → MIMIC-IV) |
| Final INSPIRE model (g_j, w_j, α) | saved by `external` as `predictions/external/<t>/final_model.pkl` | the frozen artefact to hash |
| kNN, smoothing α, OOD | `knn/predict.py` (`KNNModel`: `neighbors`, `probs`, `fit_alpha_and_ood`, `kth_info`, `loo_prior`) | votes are unweighted; prior = class prevalence of the case base |
| Calibration diagnostics | `evaluation/metrics.py` (`summary`: intercept, slope, CITL, ECE, curve per class, one-vs-rest) | no cumulative-threshold calibration for ordinal T2 |
| OOD rate | `external.py` | **mode A only** |
| Patient-level bootstrap, paired contrasts | `evaluation/metrics.py`, `evaluation/report.py` | AUROC/AUPRC NaN-safe |
| Shape consensus | `importance/shapes.py` (`consensus`) | per-value mean, SE², DL τ², F flatness test; SE² not returned yet |
| Time fields | MIMIC-IV `year_group` (anchor_year_group) in `stays_index`; eICU `year` (hospitaldischargeyear) | INSPIRE: none (times are relative per patient) |
| eICU hospital | not extracted | `hospitalid` exists in `patient.csv.gz` |

## 3. What is missing, and feasibility

| Item | Feasible? | How / why not |
|---|---|---|
| A(i) frozen INSPIRE metric, local case base | yes | = `T_local` (reuse) |
| A(ii) local NCA with prior at INSPIRE w_j (g_j frozen) | yes | `DiagonalNCA.fit(prior = INSPIRE w, λ = INSPIRE's λ)` on the local case base; no new tuning |
| A(iii) full local PERISIM | yes | = `L_perisim` (exists, one full fit per cohort) |
| B compositions (local; INSPIRE ∪ local; INSPIRE ∪ other external ∪ local; INSPIRE only) | yes | case bases are concatenated embeddings (frozen encoder) |
| B site weight γ on non-local votes | yes, small extension | `KNNModel` gets an optional per-reference weight multiplying each neighbour's vote and the prior's class counts; γ = 1 must reproduce the current kNN exactly (unit test) |
| C accumulation curves | yes | only kNN queries: cheap |
| C temporal order | **partial** | MIMIC-IV: 3-year `anchor_year_group` blocks only (dates are shifted per patient), and the blocks coincide with the eMAR → prescriptions change (D6); eICU: 2014 vs 2015 only. → coarse blocks with random order inside, R = 20 repetitions, plus a fully random order; declared |
| D α (and k) on a local tuning set; logistic recalibration as sensitivity | yes | α grid already exists; recalibration on logit(p) per class |
| D cumulative-threshold calibration for ordinal T2 | yes, new metric | calibration of P(Y ≥ c) for c = medium, long, death |
| E shapes per external site, on INSPIRE's value grid | yes, **reduced ensemble** | the curves must be read on the same bins to compare sites: run the consensus loop in each external cohort with INSPIRE's frozen encoder (bins), 40 models (instead of up to 150), declared |
| E between-site homogeneity (DL / Cochran Q, I²) per variable and class | yes, small extension | `consensus` must also return per-value SE² |
| E range coverage and missingness pattern per variable | yes | % of external values outside INSPIRE's range (where g_j is constant); missingness rates per site |
| F eICU per hospital | yes, for large hospitals | `hospitalid` read from the raw `patient` table inside the new module (not added to the extraction: that would re-key and re-run the pilot and CV); threshold n ≥ 300 index stays (≥ 60 held-out stays) [to confirm with the counts] |
| OOD of the local case base (H4) | yes | `kth_info` on each case base |
| H6 variable classes | yes, needs a pre-declared table | physiological (`lab_*`, `vital_*`), practice (`support_*`, `atc:*`, ICU length before discharge), coding (`dx:*`, ICD-10-PCS, surgical group), patient (age, sex, BMI, ASA) |
| Hash of frozen artefacts at start and end | yes | SHA-256 of `final_model.pkl`, `state/freeze.json` and `state/frozen_config/*` |

Nothing in the list is infeasible; the temporal order is only approximate (declared).

## 4. Effective sizes (to be filled from the pod)

- INSPIRE, T2-evaluable index stays: about 12.7 thousand (from the pilot's fold sizes).
- MIMIC-IV and eICU index stays, and the eICU distribution by hospital: aggregate counts from the pod (commands in the
  lead's message; only counts by size class are reported, never per-hospital identifiers with small counts).

## 5. Full experiment map after the pilot (experiment T2; T1 and T3 repeat it)

| # | Step | Uses | Gate / decision | Rough time |
|---|---|---|---|---|
| 1 | Pilot report | INSPIRE fold 0 | fixes min_delta (D7), excludes over-costly families, cost estimate; cost levers only if over budget, in the declared order | done |
| 2 | Integrate this module + amendment (after approval), smoke | code | the change touches `pipeline.py` → the pilot is **re-run once** (same seeds, same result) | ~45 min |
| 3 | `cv`: nested 5 × 3 patient-level CV, PERISIM + every comparator | INSPIRE | — | overnight (pilot estimate) |
| 4 | `temporal` | INSPIRE | omitted: INSPIRE has no calendar time (declared) | — |
| 5 | `report`: internal results, shapes, density diagnostics | INSPIRE | — | minutes |
| 6 | `sensitivity`: T2 variants (`margin_4h`, `dx_before_cutoff` refitted; `t2_absolute` relabelled; …) | INSPIRE | — | hours (refits) |
| 7 | Review of internal results, then **freeze + OSF registration** (SPECS + amendment) | — | ⛔ lead | — |
| 8 | `external`: modes A / B, cross-fitted tertiles and absolute categories, discretized sensitivity, Q4; saves the final INSPIRE model | MIMIC-IV, eICU | needs the freeze | ~1–3 h |
| 9 | `transport` (this plan, A–F, H1–H6) — replaces `transfer`, whose variants it contains | MIMIC-IV, eICU (+ INSPIRE as case base) | needs `external`; hashes of the frozen artefacts checked at start and end | ~6–10 h (local fits dominate) |
| 10 | `export` + copy to updown; KHDP export review | aggregates | ⛔ admin | — |

Then **T1** (complications; compared with T2) and **T3** (both), each with its own steps 1–10. In T1 and T3 the
transport analyses are run as **descriptive** (H1–H6 are confirmatory for T2 only).

## 6. Design of the `transport` stage

**Splits (per external cohort, patient level, seeded, disjoint):** a fixed **test** set (20 % of the patients); a
**tuning** set for D (up to 500 patients, drawn from the rest); the remaining patients form the **local pool** from
which every case base is built. A unit test checks that no patient is in two of them.

**Labels:** the target cohort's T2 tertiles are **cross-fitted** over the whole cohort (two patient halves, each
labelled with the other half's cut-points, as in the external validation), so the labels are the same for every
case-base size and composition; absolute categories (≤ 3 / 4–7 / > 7 d) as the co-primary labelling. INSPIRE and the
other external cohort keep their own cut-points when they enter a case base.

**A. Decomposition of the transport loss** (same local case base = the whole pool, same test): (i) frozen INSPIRE
metric; (ii) INSPIRE g_j with w_j refitted locally by NCA with prior at INSPIRE's w_j (INSPIRE's λ); (iii) full local
PERISIM. (i) − (iii) = loss from transporting the geometry; (i) − (ii) = the part due to the weights.

**B. Case-base composition:** local only; INSPIRE ∪ local; INSPIRE ∪ other external cohort ∪ local; INSPIRE only
(mode A). **Site weight γ:** neighbours are searched in the whole composed case base (the distance is unchanged);
each non-local neighbour's vote is multiplied by γ, and so are its counts in the smoothing prior. γ = 0 is therefore
"local votes among the k nearest of the composed base", not "local only" (reported separately). γ ∈ {1, 0.5, 0.25,
0} and a declared schedule γ(n) = n₀ / (n₀ + n_local), n₀ = 500.

**C. Accumulation (deployment simulation):** local cases added in the order of §3 (time blocks, random inside;
R = 20 repetitions; and fully random) at n_local ∈ {0, 50, 100, 250, 500, 1000, 2000, all}, for each start (empty,
INSPIRE, INSPIRE ∪ other external) and each γ. k rule: k_eff = min(k, n_case_base); with an empty start and
n_local = 0 the prediction is the uniform prior (reported, not used in the crossing). α is INSPIRE's (frozen) in C.
Metrics per point: log-loss, Brier, RPS, calibration in the large, slope, ECE, AUROC/AUPRC of the worst class, %
out of support. CIs: across repetitions for the curves; patient bootstrap at the pre-declared points (n_local = 250,
1000, all and the estimated crossing).

**D. Minimal recalibration without touching the geometry:** α (and k from the existing grid) chosen on tuning sets
of 100, 250 and 500 patients; versus no tuning; logistic recalibration (intercept + slope on logit p, per class) as a
sensitivity only. Ordinal T2: calibration of every cumulative threshold P(Y ≥ c).

**E. Homogeneity of the shapes between sites:** reduced consensus (40 models, same model space and procedure) in
MIMIC-IV and eICU on INSPIRE's frozen value grid; per variable and class, Cochran Q / DL τ² and I² across the three
sites over the values; Bonferroni over variables × classes. Also per variable: range coverage (% external values
outside INSPIRE's range) and missingness difference.

**F. eICU by hospital:** B and C with per-hospital case bases for hospitals with ≥ 300 index stays (threshold
declared); between-hospital heterogeneity (variance of calibration in the large and of log-loss across hospitals)
compared with the between-country difference (INSPIRE vs MIMIC-IV / eICU).

**Outputs** (inside the pod, never in the public repository): `<workdir>/experiments/<t>/results/transport/` — CSV
tables (aggregates only, cells n ≥ 10) and figures (accumulation curves, metric-vs-case-base decomposition,
shape-homogeneity map by variable × site, calibration plots by mode), plus `transport.html`; copied to updown with
the reports (`.csv` added to the published / exportable suffixes). Code: new package `src/perisim/transport/`,
reusing `KNNModel`, `fit_perisim`, `evaluate_predictions`, `paired_between`, the metrics and the bootstrap.

## 7. Computational cost (estimate, 16 vCPU)

| Part | Cost |
|---|---|
| A(ii) local NCA × 2 cohorts | minutes |
| A(iii) local PERISIM × 2 + multi-source × 1 (already in `transfer`) | ~3 full fits, ~3–4 h in parallel |
| E reduced consensus × 2 cohorts | ~20–40 min each |
| B, C, D, F (kNN queries: ≈ 10⁴ evaluation points, metrics without bootstrap per point) | ~1–3 h |
| Bootstrap at the declared points | ~30 min |

## 8. Decisions needed from the lead before implementing

1. **Merge `transfer` into `transport`** (one module, the existing variant names kept). Recommended.
2. **Output location:** inside the work folder (the repository is public), not `results/` in the repository.
3. **Margins** of H1–H3 (proposed values in the amendment).
4. **γ semantics** as in §6 B (votes and prior weighted; γ = 0 ≠ local only) and the schedule n₀ = 500.
5. **Scope:** H1–H6 confirmatory for T2; descriptive in T1 / T3.
6. **One pilot re-run** after integration (the stage wiring changes `pipeline.py`, part of the pilot's key).
7. **E with a reduced ensemble** (40 models) per external site, on INSPIRE's grid.
8. **eICU hospital threshold** (≥ 300 index stays), to confirm with the counts.
