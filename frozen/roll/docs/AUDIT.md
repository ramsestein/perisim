# PERISIM — full re-audit (2026-10-22)

Scope: the whole repository at commit `c76e296`, against SPECS.md. Five independent read-only reviews
(data leakage; scientific fidelity of the method; cohort, targets and extraction; orchestration and export
safety; evaluation and reporting), each verifying its findings with scratch experiments on synthetic data.
Findings marked ✔ were additionally re-checked by hand during consolidation. Duplicates across reviews are
merged. Nothing has been fixed yet.

Severity: **B** = blocker (the pod run fails or governance is breached) · **H** = high (biases results or
violates SPECS) · **M** = medium · **L** = low.

## A. Blockers before the first pod run

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| A1 ✔ | B | `.gitignore:14` | The unanchored rule `data/` also ignores `src/perisim/data/`: **the whole extraction package is missing from GitHub** (0 files). Cloning in the pod → `ImportError`. Local and WSL tests copied the folder, not the repo, so they could not catch it. | `/data/`; commit the package; add a test that the clone contains every module. |
| A2 ✔ | B | `data/discover.py:22` | Discovery skips any folder named `files`, so the standard PhysioNet layout (`physionet.org/files/mimiciv/3.1`, `…/eicu-crd/2.0`) is never found → stop at `discover`/`harmonize`. | Skip `files` only under waveform/cxr/ecg trees. |
| A3 | B | `data/io.py:110,124,154-159` | ID filters compare text exactly: a Parquet `hadm_id` stored as float (`20000001.0`) or CSV values like `"20000001.0"` return **0 rows silently** (eMAR, labs, supports empty). | Normalise both sides (`\.0+$`), cast to the column dtype, warn on empty filtered reads. |
| A4 | B | `pipeline.stage_audit`, `config.py:28-35`, `config.Config.hash` | The audit acknowledgement is read from `PERISIM_ACK_AUDIT` every run (not persisted), `PERISIM_ACK_AUDIT=n` counts as true, and the freeze `config_hash` includes paths, resources and gates. Following the README (`PERISIM_ACK_AUDIT=true … ; freeze`) makes the external stage stop with "iterating is forbidden". | Persist the acknowledged issue hash in `state/`; parse booleans strictly; hash only the scientific part of the config for the freeze. |
| A5 | B | `pipeline.Pipeline.key` | Stage keys hash config and code only, not upstream **outputs**: after `--stage features --force`, a re-discovery that changes a cohort, or a cohort that appears later, downstream stages (leakage, audit gate, pilot, cv) are skipped and stale results reused. Discovery never re-runs when a missing cohort is mounted later. `pipeline.py` is in no code hash. | Store an output digest in each marker and chain it into downstream keys; re-run discovery when a cohort is missing; `--force` invalidates downstream; hash `pipeline.py`, `feature_sets.json`, `d7.json`. |
| A6 | B | `evaluation/audit.py`, `export.py`, `pipeline.stage_harmonize` | Small-cell rule (n < 10) breached in exported JSON: raw flow counts (exclusions of 1–9), feature availability fractions at full precision (missing counts recoverable), audit issue text with exact prevalences, complements of `n_pos`, calibration bins with 2–7 events, VDM `p` + `n` + `m` (class counts recoverable), fold `class_counts`, rare free-text values in schema inventories. | One suppressing writer for exportable JSON (or an allow-list); suppress complements and differences; round fractions; ≥10 events and non-events per calibration bin. |
| A7 | B | repo root | A stray nested repository `r/` (created by mistake during the review) is untracked: it would be swept into the next `git add -A` and blocks `freeze` (dirty tree). | Delete `r/` (owner action). |

## B. Scientific method (§5)

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| B1 ✔ | H | `importance/sage.py:67-73` | Inverse-variance aggregation uses only the within-model Monte-Carlo SE. A model that never uses a group has SE = 0 → weight 1e8 → consensus ≈ 0 for that group (toy: 0.136 → 0.0001); with λ ≥ 0.01 the variable is removed from the metric. Such models are common (shallow XGBoost left 25/62 groups at SE 0). | Random-effects weights (se² + τ²), or plain/trimmed mean; SE = 0 means "no information". |
| B2 | H | `metric/nca.py` | With `logw` and the free scale `logc` both optimised by Adam under a stiff penalty, the global scale (bandwidth) barely moves at large λ (logc 2.298 → 2.287 in 30 epochs; Σw stuck at 9.9 vs 17.6 optimum). The profile is already pinned at λ = 0.01, so λ ∈ {1, 10, 100} differ only by this artefact → λ selection biased against SAGE. | Reparameterise `logw = logc + log(imp+ε₀) + u`, penalise only `u`; re-space the λ grid (scaled by 1/F). |
| B3 | H | `importance/loop.py:93-100`, `nca.py` | The Δ stopping rule is dominated by optimiser noise: refitting with an unchanged prior gives a moving-median Δ of 0.8–1.3 % (≈ tol_Δ), with spikes at candidate refreshes. With `lambda_loop = 1`, w ≈ c·imp, so Δ tracks the importance profile, not the final metric (§5.3.2 says the opposite). | Deterministic conversion (fixed order/seed, full batch or L-BFGS, frozen candidates between refreshes); estimate and report the noise floor. |
| B4 | H | `metric/encoding.py:322-323` | Equal split of group importance gives an uninformative (MCAR) missingness indicator the same prior as its strong value variable (≈ 19 % of the distance budget); correlated indicators chain unrelated labs into one group. | Prior on group totals (λ Σ_g (log Σ_{j∈g} w_j − log c − log imp_g)²) and let NCA split within groups. |
| B5 | M | `knn/predict.py:42-56` | The kernel variant reuses the α tuned for counts (kernel mass ≈ 0.5 vs α·C = 0.3) → sensitivity analysis dominated by shrinkage (log-loss 0.797 vs 0.679). | Tune α for the kernel variant separately. |
| B6 | M | `metric/encoding.py:131-141` | Rare, capped (> 200) and missing ICD-10-PCS codes share one zero vector: two different rare codes "match"; kept vs rare = half mismatch. | Explicit "other/missing" dimension per level; raise/remove the cap at levels 4–5. |
| B7 | M | `metric/encoding.py:115,181-187` | Unseen VDM categories map to `__other__` (pooled rare categories), not to P(y) as §5.2 states; the 50-category cap is undeclared. Matters for mode A. | Map unseen to P(y); declare pooling. |
| B8 | L | `importance/loop.py:73-74` | Models accepted early stay in the consensus even when > (1+ε)·final best (6/24 in a test). | Re-filter against the final best before the final aggregation. |
| B9 | L | `encoding.py:316-321` | Block split ∝ prevalence × MI double-counts prevalence (MI already scales with it). | Consider MI alone (declare). |
| B10 | L | `baselines/weights.py` | ReliefF uses RMS of δ over neighbours; SHAP sums |φ| across one-hot columns (inflates nominal/PCS). | Mean δ; |Σφ| per variable. |
| B11 | L | `sage.py:62-63` | All-non-positive SAGE → uniform importance with an unscaled SE (gets large weight). | Treat as uninformative. |

## C. Data leakage and temporal correctness (§4.4)

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| C1 | H | `features/build.py`, `tests/test_no_leakage.py` | The anti-leakage test is structurally blind for medications, supports and every stays-level field (provenance is computed from already-clipped values or fixed timestamps): injecting post-t0 vasopressor/antibiotic rows changed `med_*` with 0 violations reported. | Metamorphic test: X(inputs) must equal X(inputs with rows > cutoff dropped and ends clipped), on synthetic and real data; source timestamps for static fields. |
| C2 | M | `data/mimic.py:79-90` | MIMIC `pcs_code` (and the postop flag) can come from a procedure **after** t0 on the same/next day (sorted by day descending, then `seq_num`): input leak in the `extended`/`full` sets and a selection effect always. | Only procedures dated ≤ ICU admission day, prefer the day before / service-linked; never `seq_num`. |
| C3 | M | `experiment.py:147-164,212-225` | λ is selected with a SAGE prior whose surrogates saw inner-validation labels (declared OQ#14): measured bias towards large λ (2/6 vs 0/6 seeds). | Importance per inner fold, or score λ on b0 only. |
| C4 | M | `external.py:160-163` | External T2/T3 tertiles are computed on the whole external cohort (the test set) → optimistic calibration-in-the-large; mode B includes the query in its own thresholds and priors. | Patient-level calibration split in the external cohort, or absolute categories co-primary; declare. |
| C5 | M | `data/inspire.py:_common_flags`, `mimic.py` | t0 is not checked to be a discharge **to a ward** in INSPIRE/MIMIC: an ICU exit to the OR/IR closes the stay, so the label (reoperation, readmission) is known at t0. | MIMIC: next careunit in `transfers` must be a ward; INSPIRE: exclude/merge stays with an operation or ICU entry within N h of t0; audit the distribution. |
| C6 | L | `knn/predict.py:58-76`, `encoding.py` | α and the OOD threshold use in-sample LOO; VDM includes each anchor's own label (small with large categories; measured 0.03 nats with 50 tiny categories). | α on inner folds; out-of-fold VDM for anchors. |
| C7 | L | `harmonization.yaml`, `mimic.py` | Lab/chart times use `charttime`, not `storetime`: a result posted after the cutoff can be used. | Use `storetime` (or result offset). |
| C8 | L | `features/build.py:106-107`, `inspire.py:156` | `before_surgery` falls back to "≤ cutoff" when the surgery start is missing (all eICU); the dx window starts at admission − 1 day, dropping earlier-admission codes (the only ones surely known). | Exclude dx without surgery time; widen the lower bound. |
| C9 | L | `data/eicu.py:48-58` | eICU `surgical_group` (core input) from admissionDx without checking `admitdxenteredoffset ≤ cutoff`. | Read and filter the entry offset. |
| C10 | L | `pipeline.stage_harmonize` | "Availability" of binary features is their prevalence → a silent 5 % prevalence filter in every cohort (vs D6's 0.5 %), computed with external cohorts. | Availability = source observed, not prevalence. |
| C11 | L | `baselines/rfgap.py`, `weights.py` | RF-GAP OOB and ReliefF are not patient-aware (training-only effect). | Group-aware bootstrap / exclusion. |
| C12 | L | `inspire.py:79-80` | If the INSPIRE limitation column is configured, no time check (post-t0 limitation = outcome-dependent selection). | Require timestamp ≤ cutoff. |

## D. Cohort and target definitions; extraction (§2–§4)

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| D1 | H | `config/cohorts.yaml:28-30`, `mimic.py:41-45` | The ICD-9 "surgical" rule (chapters 01–86) accepts urinary catheter, diagnostic cath, LP, thoracentesis, paracentesis, bronchoscopy, EGD, colonoscopy, PEG, chest tube…: corrupts the MIMIC cohort (≈ half of admissions are ICD-9 era) and the reoperation component. | Positive OR list (AHRQ/HCUP surgery flags) and/or PACU/OR transfer in `hosp.transfers`. |
| D2 ✔ | H | `harmonization.yaml:316-322` | MIMIC °F temperatures: bounds [77, 113] (°F) are applied after the F→C transform → every °F value is discarded (most MIMIC temperatures). | Bounds in °C or before the transform; unit test. |
| D3 | H | `mimic.py:49-52,196-201` | MIMIC `icustays` includes step-down units (Neuro Intermediate/Stepdown): wrong t0, spurious readmissions, non-ICU index stays. | Rebuild ICU episodes from `transfers` with true ICU careunits. |
| D4 | H | `harmonization.yaml:359-454`, `mimic.py`, `eicu.py` | Name→ATC regex maps lidocaine-epinephrine, racepinephrine, nasal/ophthalmic phenylephrine to vasopressors and ophthalmic/topical antibiotics to J01 → spurious `vaso_start` (severe) and `antibiotic_course` (mild); support_vaso derived from eMAR/orders. | Vasopressor support only from IV infusions; filter J01 by systemic route; exclude topical/local forms. |
| D5 | H | `targets/build.py:84-108` | Chronic dialysis / advanced CKD patients are labelled severe (every ward HD = new RRT; baseline 6.0 → 6.4 = AKI 3 via the ≥ 4.0 rule). | Exclude prior/chronic RRT; ≥ 4.0 only with an acute rise and baseline < 4.0; no baseline during RRT. |
| D6 | H | `mimic.py:179-193` | MIMIC medication only from eMAR, which (to verify) does not cover the early MIMIC-IV years → "not recorded" becomes 0 for antibiotics, transfusion and every `med_*`. | Null for admissions without eMAR records, or add prescriptions; declare. |
| D7 | M | `targets/build.py:99-108` | KDIGO timing: ratio over 30 days without the 7-day window; the 0.3 rise only in (t0, t0+48 h]. | Rolling criteria (≥1.5× min within 7 d, ≥0.3 above min within 48 h), anchored on the SPECS baseline. |
| D8 | M | `mimic.py:66-90` | Service-only arm admits non-operative patients (SAH on NSURG, non-operative trauma, ward deterioration days after surgery). | Require surgical evidence within 24 h; report service-only entries. |
| D9 | M | `targets/build.py:150-153` | T2/T3 drop in-hospital deaths after day 30 (survivor = never died); §3.3 truncates follow-up at 30 d. | Survivor = no death in (t0, t0+30 d]. |
| D10 | M | `inspire.py:158-164` | INSPIRE ICU readmission is by construction a subset of reoperation (D1 proxy via ward_vitals not implemented) → severe T1 not comparable across cohorts. | Implement the D1 proxy or mark it unobservable and declare. |
| D11 | M | `eicu.py:55-58`, `harmonization.yaml` | eICU surgical group depends on row order; "vascular" matches "Cardiovascular" (CABG split between cardiac/vascular). | Use the "Operative|Diagnosis|<system>|<dx>" row, deterministic ties. |
| D12 | M | `cohort/select.py:37-38` | Limitation-of-treatment exclusion does nothing in MIMIC/eICU (data exist: chartevents 223758, carePlanGeneral) yet the flow shows "0 excluded". | Implement with records ≤ cutoff, or label the step "not available". |
| D13 | M | `harmonization.yaml:260-280` | MIMIC blood gases 50820/50818/50821 mix arterial and venous samples (venous pO2 ≈ 40 looks like severe hypoxaemia). | Join specimen type (52033 = "ART."). |
| D14 | M | `targets/build.py`, SPECS §2.2 | INSPIRE values are clipped at p2.5–p97.5: KDIGO ≥ 4.0 mg/dL and SpO₂ < 90 % may be unreachable. | Audit clip ranges; declare/drop impossible criteria. |
| D15 | L | `eicu.py:155-182` | eICU support inputs: vasopressor from orders, rate-0 infusions counted, `ventstartoffset == 0` dropped. | infusionDrug with rate > 0 + treatment only. |
| D16 | L | `data/io.py:76-78` | CSV headers read as utf-8 (a BOM breaks the first column; fails loudly). | `utf-8-sig`. |
| D17 | L | various | MIMIC: "No Response-ETT" verbal GCS = 1; `Rewritten` inputevents not excluded; newer vasopressor itemids. eICU: unit deaths mislabelled in the flow; APACHE IV vs IVa ordering; ICD-9 V/E codes parsed as ICD-10. INSPIRE: every later anaesthesia case counts as reoperation. | Small targeted fixes / declarations. |
| D18 | L | §4.1 | Partly built inputs: no O₂-device input; eICU temperature only from vitalPeriodic; no pre-op labs in MIMIC/eICU. | Add or declare. |

## E. Evaluation and reporting (§6)

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| E1 | H | `external.py:147-212` | eICU mode A (and A_euclidean, A_logreg, SWIFT) predicts INSPIRE's **full** severe definition but is scored against eICU's **reduced** label (readmission/death): the Q3 A−B comparison is dominated by the outcome definition (sign can flip). | INSPIRE case-base label restricted to readmission/death for eICU mode A; refit LR/SWIFT on it. |
| E2 | H | `experiment.py:374-376`, `report.py` | A comparator failing in one fold is pooled silently on fewer stays and its paired test disappears. | Report per-fold errors; evaluate on the common set of stays; flag in HTML. |
| E3 | H | `external.py:render_external` | The central Q3 result (A−B paired Δ with CI) is computed but not shown; no calibration, RPS, k-curve or Q4 table externally; mode A at primary k renamed inconsistently; no B_perisim vs B_euclidean / B_vdm_target tests. | Render all; explicit reference; paired tests inside mode B. |
| E4 ✔ | M | `report.py:_binary_block` | Binary Brier summed over both columns = 2 × the usual Brier; multiclass Brier convention (0–2) undeclared. | Mean (p−y)² for binary; state the convention. |
| E5 ✔ | M | `baselines/parametric.py:85` | SWIFT: GCS 14 scores 0 (Gajic 2008: 11–14 → 6). No paired PERISIM-vs-SWIFT test; in MIMIC no PERISIM binary block; SWIFT's own outcome differs from T1 severe (undeclared). | `g <= 14`; paired test; declare estimand. |
| E6 | M | `report.py`, `experiment.py:65-66` | Q4: no CIs / paired tests; discharge destination dropped (text column); external Q4 not rendered; degenerate variables (death in survivors) reported. | Paired bootstrap per variable; multinomial log-loss for destination. |
| E7 | M | reporting | TRIPOD+AI: no characteristics table; no per-target analysis flow; class counts not shown. | Add Table 1 and per-target flow. |
| E8 | M | `report.py:render_results` | Only log-loss CI shown; calibration intercept/slope/CITL not tabulated; AUROC/AUPRC CIs use an undeclared 500 replicates. | Tabulate; 2,000 replicates or declare. |
| E9 | M | `evaluation/audit.py:74-83` | §7 missingness exists only for AKI and antibiotics; all other components (incl. transfusion) hard-coded 0 % → the gate cannot fire for them; AKI missingness mixes "early discharge" with "not measured". | "Not assessable" instead of 0; observability proxies; split baseline vs follow-up missing. |
| E10 | M | `aggregate_cv` | Pooled calibration/AUROC mix fold-specific T2/T3 cut-points; no per-fold metrics; CIs ignore refit variability. | Per-fold metrics and cut-points; state it. |
| E11 | M | `report.py:_importance_dispersion` | SAGE dispersion pooled by group name across folds although groups (clusters) differ per fold. | Report per fold or map to variables. |
| E12 | M | `pipeline.stage_temporal` | Temporal split not grouped by patient (currently inactive). | Exclude test patients from training. |
| E13 | M | `experiment.py`, `model_space.yaml` | LightGBM comparator not tuned (fixed 400 trees, no early stopping) while PERISIM/LR are tuned (§6.4 "same nested fitting"). | Small inner-CV grid or early stopping. |
| E14 | M | `knn/predict.py:patient_records` | The §5.6 per-patient output is never produced. | Write it (inside the pod only) for the test folds / external queries. |
| E15 | L | `report.py`, `aggregate_cv` | Stale fold files from a previous config are pooled (no `_key` check). | Filter by key and expected folds. |
| E16 | L | various | p-values printed as 0.000 (report p < 1/B); no multiplicity plan; "euclidean" uses the supervised VDM (declare); PERISIM ≡ nca_diag when λ = 0 (flag); Q4 table header; code says P1–P4 while SPECS says Q1–Q4; D8 flag `privileged_information` unused. | Small fixes / declarations. |

## F. Orchestration and robustness

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| F1 | M | `pipeline.run_jobs`, extraction, external | One OOM-killed worker breaks the pool (`BrokenProcessPool`) and every in-flight job; no retry/back-off; external writes only at the end. | One process per job; resubmit with fewer jobs on -9; per-target checkpoints. |
| F2 | M | `run_pod.sh` | "Already running" checked after setup and smoke; no work-directory lock; stale PID → `status` says running and `stop` can kill an unrelated process. | Check first; `flock` lock with host/pid; verify the PID's command line. |
| F3 | M | `cli.cmd_freeze` | Tag failure ignored (`check=False`) yet recorded; `-f` overwrites; re-freeze allowed after external results; package versions not frozen. | Fail on tag error; refuse re-freeze after `external.json`; record versions. |
| F4 | M | `run_pod.sh test`, leakage stage | Real-data tests never run (`PERISIM_TEST_WORKDIR` not set in `test`; split/VDM tests run before folds exist). | Set the variable; run split/VDM tests as a gate after cv. |
| F5 | M | `.githooks/pre-commit`, safety script | Hook committed as 100644 (ignored on Linux/macOS); safety script misses `.tsv/.xlsx/.sav/.rds/.sqlite/.h5`, unquoted credentials, `hf_`/`AKIA`/`github_pat_` tokens. | `chmod +x` in git; extend patterns. |
| F6 | M | stage keys | Needless reruns: `experiment.evaluation` in the audit key re-runs the whole CV; the full `harmonization` section in the extract key re-extracts all cohorts; the workdir path in the discover key. | Narrow dependencies; per-cohort harmonization slices. |
| F7 | L | `run_pod.sh:37`, `experiment.yaml:10` | Default workdir differs (`…/perisim/<user>/work` vs `…/perisim/work`) when running `python -m perisim` directly. | One default. |
| F8 | L | `run_pod.sh:140-152` | Torch fallbacks (cu126 / general index) pull ~3 GB of CUDA wheels on a GPU-less pod; `--index-url` alone needs torch's deps on the torch mirror. | `--extra-index-url`; prefer CPU wheels. |
| F9 | L | `resources.py` | cgroup files read only at the root (no `/proc/self/cgroup` lookup); memory plan uses the limit, not free memory; no `max_tasks_per_child`. | Resolve the cgroup path; cap by available memory. |
| F10 | L | `utils.read_json`, status | A corrupt JSON blocks the pipeline even with `--force`; `last_error.json` never cleared; temporal/sensitivity/export never marked. | Treat unreadable as missing; clear on success; mark stages. |
| F11 | L | `config.py` | `code_hash` differs between Windows and Linux (path separators); `${VAR:-default}` ignores the default for empty values; YAML re-parsing of env values mangles paths with " #" or "null". | POSIX relative paths; treat empty as unset; parse only scalars. |
| F12 | L | smoke profile | Smoke covers 1/7 sensitivity variants and pre-acknowledges the audit (Stop path untested); the smoke marker ignores config/requirements changes. | Extend smoke; include config hash in the marker. |
| F13 | L | `pipeline.stage_sensitivity` | Variants rebuild targets/features for MIMIC/eICU needlessly. | Restrict to the development cohort. |

## Checked and found correct (summary)

SAGE permutation estimator (unbiased vs exact Shapley), surrogate masking (same at train and SAGE time),
NCA objective (matches brute-force LOO soft-kNN), VDM distance (= ‖p_c − p_c'‖₂ after scaling), δ_j
normalisation, groups as specified, inference formula, OOD rule, RF-GAP formula, stopping-rule mechanics,
patient-level splits and LOO exclusion, all target-dependent fits inside the training fold (except C3/C6),
metric formulas vs reference implementations (log-loss, multiclass Brier, RPS, calibration GLM, AUROC/AUPRC),
patient-level bootstrap, paired-difference alignment, MIMIC/eICU table/column names and itemids, eICU
hospital clock, KDIGO thresholds and baseline, antibiotic-course rule, atomic writes and per-fold resume,
`nohup setsid` process-group stop, git history free of secrets.

## Resolution log (2026-10-22)

**Fixed** (with regression tests where testable — `tests/test_audit_regressions.py`, `test_io_and_repo.py`,
`test_mimic_episodes.py`, `test_no_leakage.py`, `test_resources.py`):

- **Blockers:** A1 (`/data/` anchored; every source module must be tracked — test), A2 (`files/` walked — test),
  A3 (identifier normalisation + warning on empty filtered reads — test), A4 (audit acknowledgement persisted per
  issue set, strict booleans, freeze/external compare a *scientific* config hash without paths/resources/gates),
  A5 (stage keys chain the output digest of every upstream stage; discovery re-runs when a cohort is missing;
  `pipeline.py` hashed for harmonize/pilot; fold seed moved to `experiment.py`), A6 (`evaluation/privacy.py`:
  primary + complement + secondary suppression, flow rounding, value-free issue texts, calibration bins with
  ≥ 10 events and non-events, VDM n/p rounded, class counts suppressed, availability kept out of artifacts,
  rare inventory values not listed).
- **Method:** B1 (random-effects DerSimonian–Laird aggregation — test), B11.
- **Leakage:** C1 (metamorphic leak check in the `leakage` stage and in tests, with a positive control),
  C2 (MIMIC index procedure never from a same-day procedure that may follow t0 — test).
- **Definitions / extraction:** D2 (°F bounds — test), D3 (MIMIC ICU episodes from true ICU careunits in
  `hosp.transfers`; synthetic data now include step-down segments — test), D4 (vasopressor support only from IV
  infusion sources; local/topical/ophthalmic/otic/nasal/inhaled products and flushes excluded; eICU systemic
  routes only), D9 (survivor = no death within 30 d — test), D11 (deterministic eICU surgical group; "vascular"
  no longer matches "Cardiovascular"), D16 (CSV BOM — test).
- **Evaluation:** E1 (eICU mode A uses an INSPIRE case base labelled with the same reduced definition; LR and
  SWIFT refitted on it), E2 (stay-aligned paired differences; incomplete comparators and per-fold comparator
  errors reported), E3 (external report: shared tables, Q3 contrasts B−A, B vs Euclidean, B-VDM vs B, Q4),
  E4 (binary Brier on the usual scale — test), E5 (SWIFT GCS 11–14 → 6 — test; paired SWIFT vs PERISIM;
  estimand note), E8 (CIs for Brier/AUROC and calibration tables now shown), E9 ("not assessable" instead of
  0 %), E15 (only folds of the current run are pooled), E16 (Q1–Q4 naming; p-values shown as p < 1/B).
- **Infrastructure:** F2 (run check before setup; PID validated against the process command line; `fg` writes
  its PID), F3 (freeze fails if the tag cannot be created, never overwrites, refuses after external results,
  records package versions), F4 (`run_pod.sh test` sets `PERISIM_TEST_WORKDIR`; split/VDM tests run as a gate
  after the CV), F5 (hook and `run_pod.sh` executable in git; more file types and token patterns; console-safe),
  F6 (narrower stage dependencies, per-cohort config slices), F7, F8 (`--extra-index-url` for torch's
  dependencies), F10 (corrupt JSON treated as missing; last error cleared on success), F11 (POSIX paths in the
  code hash; empty env values take the default; only booleans/numbers re-typed).

**Scientific decisions (decided and implemented 2026-10-22, SPECS v0.4)** — see `docs/DECISIONS.md` and
`tests/test_decisions.py`: D1/D8 (OR/PACU evidence; code fallback; service-only reported), C5 (no OR/PACU/IR/ICU
entry within 6 h of t0), D10 (INSPIRE readmission proxy), D5 (chronic RRT: AKI / new RRT not applicable; ≥ 4.0
rule needs a baseline < 4.0; no creatinine during RRT), D7 (rolling KDIGO), D6 (medication coverage; T1/T3
undefined without it), D12 (timestamped limitation), D13 (arterial gases), B2/B4 (reparameterised NCA, prior on
group totals), B3 (Δ on the prior-implied metric — amends SPECS §5.3.2), C3 (SAGE per inner fold for λ), B6/B7
(PCS "other"/"missing" dimensions; unseen VDM → P(y)), E13 (LightGBM tuned), C4 (cross-fitted external tertiles
+ absolute co-primary; mode-B LOO prior), E14 (§5.6 output in the pod), E10/E11 (per-fold metrics and cut-points;
SAGE dispersion per variable), C6 (LOO kept, declared). Also closed as a consequence: C10 (availability =
observability). D14 remains deferred to the first audit.

**Pending engineering (not yet done):** F1 (adaptive retry after an OOM-killed worker), F9 (cgroup path lookup via
`/proc/self/cgroup`), F12 (smoke coverage: all variants, audit stop path), F13 (variants rebuild external cohorts),
B5 (kernel-variant α), B9, B10, C7 (storetime), C8, C9, C11, C12, D15, D17,
D18, E6 (Q4 inference, discharge destination), E7 (characteristics table, per-target flow), E12 (temporal split by
patient). A7 resolved by the owner (stray `r/` removed).

**Methodology v0.5 (2026-10-24):** B8 fixed (early accepted models outside (1+ε)·best leave the consensus); the
MLP family removed; the v0.4 HVDM geometry kept as a comparator (see SPECS v0.5 and docs/DECISIONS.md).
