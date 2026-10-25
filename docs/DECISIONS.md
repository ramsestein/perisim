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
