# PERISIM — Perioperative Similarity

kNN with a learned distance metric for postoperative patients at ICU discharge: every variable lives on its
consensus **shape function** (how its values move the prediction, read from many competitive models) and counts
with a **weight** (SAGE aggregated across the same models + regularised diagonal NCA). Development in INSPIRE,
external validation in MIMIC-IV and eICU, with trivial, metric-learning and parametric comparators. The full
specification is in [SPECS.md](SPECS.md) (v0.5; amendments marked **[v0.4]** / **[v0.5]**, decisions in
[docs/DECISIONS.md](docs/DECISIONS.md)).

**Three experiments, one per target**, each complete with its own freeze, run in this order: **T2** (post-ICU stay:
short / medium / long / death — primary target and demonstration of the method), **T1** (complications, compared
with T2), **T3** (both). Select one with `PERISIM_EXPERIMENT` (default `t2`); the extraction is shared.

> **Public repository.** Code and configuration only: never data, outputs or secrets. Before every
> `git push`: `python scripts/check_public_safety.py --all` (the `.githooks/pre-commit` hook runs it too).

## Running in the KHDP pod

In the pod's VSCode terminal, from the workspace root (all the commands below are run from `~/workspace`):

```bash
cd ~/workspace
git clone http://github.khdp-forward-server/ramsestein/perisim.git
bash perisim/run_pod.sh
```

Pod layout (code, data, environment and work on the workspace's local disk: `updown` is shared object storage,
slow for many small files and unreliable for files being written by several processes; no hidden folders):

```
~/workspace/
├── perisim/        this repository
├── data/           the datasets, unzipped, one folder per cohort: inspire/, mimic4/, eicu/
│                   (the names are free: each cohort is found by its tables, up to 7 levels down)
├── venv/           Python environment (created by run_pod.sh, or by the user beforehand)
├── perisim_run/    logs, PID and smoke-test files of the runs
├── perisim_work/   work folder: cache, extracts, models, reports, export package
└── updown/perisim/<user>/results/   copy of the reports and export package after every run (persistent, shared)
```

`run_pod.sh` does everything on its own:

1. Uses the environment in `~/workspace/venv` if it exists (one created by the user, venv or conda, Python ≥ 3.11
   with pip, is used as it is and never deleted). Otherwise it finds Python ≥ 3.11 (or creates one with conda, which
   on KHDP already points to the mirror) and creates the virtual environment there, reusing the system packages.
2. Installs the dependencies from the KHDP PyPI mirror and torch from the PyTorch mirror
   (CPU index and, if it does not respond, cu126).
3. The first time each version of the code runs, it launches a **smoke test on synthetic data**: if it fails,
   the real data are not touched.
4. Launches the pipeline **in the background**, so that it survives closing the terminal, and shows the log.
   Ctrl-C stops following the log, but does not stop the pipeline.

It also detects the container's real CPU and memory limits (cgroup) and sizes processes × threads
automatically, at most 4 × 4 (SPECS §9.1). The consensus loop is checkpointed after every accepted model, so
`stop` (or a pod restart) costs minutes: relaunching resumes the loop where it was.

Experiments: `bash perisim/run_pod.sh` runs T2; `PERISIM_EXPERIMENT=t1 bash perisim/run_pod.sh` (later `t3`) runs the next one
on the same extraction. Only one run at a time.

| Command | Purpose |
|---|---|
| `bash perisim/run_pod.sh` | set up the environment and launch / **resume** the pipeline |
| `bash perisim/run_pod.sh status` | which stages are done and where it stopped |
| `bash perisim/run_pod.sh logs` | follow the log of the running execution |
| `bash perisim/run_pod.sh stop` | stop it |
| `bash perisim/run_pod.sh fg --until audit` | run in the foreground up to a given stage |
| `bash perisim/run_pod.sh test` | pytest, including the anti-leakage tests on the real data |
| `bash perisim/run_pod.sh freeze --registration <ID>` | §6.3 freeze of the current experiment (enables its external validation) |
| `PERISIM_EXPERIMENT=t1 bash perisim/run_pod.sh` | run another experiment (t1, t3) on the shared extraction |

To update the code: `git -C perisim pull` and `bash perisim/run_pod.sh`. Only the stages whose code or configuration have changed
are rerun, because each stage stores a marker with its hash. After an OOM or a pod restart, just relaunch:
it resumes from the last completed stage (or fold).

### Paths (configurable with environment variables)

| What | Default | Variable |
|---|---|---|
| Workspace | the first of `~/workspace`, `/home/khdp-user/workspace` or the folders above the repository that contains `data/`, `datasets/` or `updown/` (the pod user may not be `khdp-user`) | `PERISIM_WORKSPACE` |
| Data (read-only) | `<workspace>/data` if it exists (the user's own copy), otherwise the KHDP mount `<workspace>/datasets` | `PERISIM_DATASETS` |
| INSPIRE / MIMIC-IV / eICU | automatic discovery under datasets | `PERISIM_INSPIRE_DIR`, `PERISIM_MIMIC_DIR`, `PERISIM_EICU_DIR` |
| Work and results | `<workspace>/perisim_work` (shared data) and `…/perisim_work/experiments/<t1\|t2\|t3>/` | `PERISIM_WORKDIR`, `PERISIM_EXPERIMENT` |
| Copy of reports, docs and export package (after every run, also after a stop) | `updown/perisim/<user>/results/` (empty = no copy) | `PERISIM_PUBLISH_DIR` |
| Python environment | `<workspace>/venv` | `PERISIM_VENV` |
| Logs, PID and smoke-test files | `<workspace>/perisim_run` | `PERISIM_LOCAL` |

## What the pipeline does and where it stops

```
env → discover → extract → targets → features → leakage → audit ⛔ → harmonize ⛔ → pilot → cv → temporal
    → report → sensitivity → external ⛔ → transport → export
```

| Stage | Output (in the workdir) | Gate |
|---|---|---|
| `discover` | `reports/schema_<cohort>.md` (shared): actual columns and items versus `config/harmonization.yaml` | stops if INSPIRE is not found or its schema does not match |
| `extract` | `common/<cohort>/` (shared): `stays`, `inputs_long`, `outcomes` and flow diagram (§8.3) | — |
| `targets`, `features` | `derived/…`: T1/T2/T3, carried variables and inputs with the t0 − 2 h cutoff | — |
| `leakage` | row-by-row test that no input exceeds the cutoff, plus pytest | **stops** if there is leakage (§4.4) |
| `audit` | `reports/completeness_<cohort>.html` (§7), incl. INSPIRE readmission sources, chronic RRT, medication coverage by period and MIMIC-IV surgical evidence | **stops** if any T1 component has prevalence < 0.5% or missingness > 30% |
| `harmonize` | `core` / `extended` / `full` input sets | **stops** if any cohort is not mounted |
| `pilot` | `reports/pilot.html`: times, memory and estimated cost; fixes D7 with a declared rule | **stops** if `pilot.budget_hours` is set and exceeded (lists the cost levers) |
| `cv` | nested 5 × 3 patient-level CV of the experiment's target: PERISIM + trivial, kNN, metric-learning and parametric comparators, k ∈ {10…200} (shapes and λ recomputed per inner fold) | — |
| `report` | `reports/results_main.html` (TRIPOD+AI, pooled and per-fold metrics, shapes, density); §5.6 per-patient records in `predictions/`; comparison with the earlier experiments | — |
| `sensitivity` | outcome-definition variants by relabelling (minutes), input variants refitted | — |
| `external` | modes A and B in MIMIC-IV and eICU (§6.2); T2/T3 with cross-fitted tertiles and absolute categories | **waits** for `freeze` (§6.3) |
| `transport` | §6.2b pre-registered transport analyses on a fixed test set of MIMIC-IV / eICU: loss decomposition (frozen vs local metric), case-base composition and site weight γ, accumulation curves, minimal recalibration, between-site shape homogeneity, eICU by hospital, H1–H6 (includes the model-transfer validations): `results/transport/`, `reports/transport.html` | runs after `external`; checks the frozen-artefact hashes |
| `export` | `export/`: aggregates only, with a manifest and an identifier scan | — |

The three cohorts are extracted in parallel, the sensitivity variants share a single job queue, and the final
models of the external validation are fitted in parallel.

**Expected first run:** it will almost certainly stop at `discover` or at `audit`. In that case:

1. Review `reports/schema_*.md` and `reports/completeness_*.html` inside the workdir.
2. Fix `config/harmonization.yaml` on your machine, push, and then `git -C perisim pull` in the pod.
3. Record the decisions in `docs/OPEN_QUESTIONS.md` and `docs/DECISIONS.md`.
4. To continue after the audit: `PERISIM_ACK_AUDIT=true bash perisim/run_pod.sh` (once: the acknowledgement is stored for that exact set of issues; a different set stops the pipeline again).

**External validation:** once internal validation of the experiment is closed, confirm the state of the repository
and freeze it (one freeze per experiment):

```bash
bash perisim/run_pod.sh freeze --registration <ID_OSF>
```

Then relaunch `bash perisim/run_pod.sh`. From that moment on, any change to code or configuration blocks
the `external` stage, because §6.3 forbids iterating after looking at external results. Paths, compute
resources (`PERISIM_JOBS`, `PERISIM_THREADS`) and gate acknowledgements are not part of the frozen method, so
changing them does not break the freeze; a second freeze is refused once external results exist.

## Running locally on Windows

With the lead's own PhysioNet copies on a workstation, `run_local.ps1` (PowerShell) is the counterpart of
`run_pod.sh`, with the same commands (`start`, `fg`, `status`, `logs`, `stop`, `smoke`, `test`, `ksweep`, `freeze`) and the
same environment variables. Everything stays in the project folder, in folders git ignores (the repository is
public), and the script refuses a folder that git tracks or that OneDrive synchronises:

```
<repo>\data\          the datasets, one folder per cohort (inspire\, mimic4\, eicu\)
<repo>\work\          work and results
<repo>\perisim_run\   logs, PID and smoke test
```

```powershell
.\run_local.ps1            # smoke test for a new code version, then the whole pipeline in the background
.\run_local.ps1 logs       # follow it; .\run_local.ps1 status / stop
```

Default parallelism 5 processes × 4 threads (the 5 outer folds in one round; `PERISIM_JOBS`, `PERISIM_THREADS`).
The launcher passes the configuration profile `config/profiles/local.yaml` (what differs from the pod: no family
excluded for cost, a declared deviation) to `run`, `status` and `freeze`; `PERISIM_PROFILE=""` runs without it. The
pipeline keeps Windows from sleeping while it runs (keep the computer plugged in; closing the lid may still suspend
it, depending on the power settings). Only logs, reports and aggregates are ever read back: patient-level data never
leave the computer (PhysioNet data use agreement).

**k sweep (secondary):** once the CV is done, `.\run_local.ps1 ksweep` (`bash perisim/run_pod.sh ksweep` in the pod)
queries each fold's saved metric again over k = 1…2000 (`experiment.k_sweep`) and writes `reports/ksweep.html` and
`results/ksweep/`. It takes minutes and is not a stage, so it never re-runs the pipeline.

**Post-hoc transport comparison (exploratory, not pre-registered):** after the transport stage,
`.\run_local.ps1 posthoc` (`scripts/posthoc_transport.py`) compares PERISIM with LightGBM and the logistic regression
transported from INSPIRE, recalibrated with 100 / 250 / 500 local patients and refitted on the local pool, and draws
every variable's shape function by site above where each site's stays fall on INSPIRE's grid
(`reports/posthoc_transport.html`, `results/posthoc/`). It lives outside `src/perisim`, so it does not change the
frozen code hash, and only reads the frozen artefacts (checked at the start and the end).

**Post-hoc drift analysis (exploratory, not pre-registered):** `.\run_local.ps1 drift` (`scripts/posthoc_drift.py`,
about two hours, resumable) keeps the frozen INSPIRE metric, swaps its case base for each site's own patients, and
asks how each variable's shape function would have to change to fit the site: ICE curves through the frozen
geometry, then each variable's points re-fitted alone (everything else frozen) and all at once, with INSPIRE as the
control (`reports/posthoc_drift.html`, `results/posthoc_drift/`). Its follow-up `.\run_local.ps1 drift-missing`
(`scripts/posthoc_drift_missing.py`) frees only the points of the missing values (not recorded, support never used,
source not recorded), all at once, to measure how much of the drift is in what a missing value means
(`reports/posthoc_drift_missing.html`, `results/posthoc_drift_missing/`).

**Post-hoc case-base analyses (exploratory, not pre-registered):** `.\run_local.ps1 case-base`
(`scripts/posthoc_case_base.py`, minutes) keeps the frozen INSPIRE metric and asks how many local patients, added at
random, it needs to recover what the whole local pool gives (a dense accumulation curve), and whether in eICU the
neighbours of the same hospital carry relations the other hospitals' do not, at the same case-base size and with the
base rates corrected (`reports/posthoc_case_base.html`, `results/posthoc_case_base/`).

**Publishing results:** `python scripts/publish_results.py t2` copies the experiment's aggregate results (the export
package: reports, tables, figures, aggregate JSON, plus the experiment's docs) into the repository's `results/t2/` for
version control. Patient-level data (`data\` and the rest of `work\`) are never copied; see `results/README.md`.

## Export (KHDP §3.2)

`<workdir>/export/` (copied after every run to `updown/perisim/<user>/results/experiments/<t>/export/`) contains aggregates only: HTML reports, metrics, w_j weights, importances and VDM tables with
cells of n ≥ 10. It includes `EXPORT_MANIFEST.md` and a zip. To take it out of the pod, write to
`admin@khdp.net` stating the purpose and the destination, and follow their review. Per-patient predictions
(`predictions/`) and the extracts (`common/`, `derived/`, `cache/`) are **never** exported.

## Local development (no real data)

```bash
pip install -r requirements.txt torch
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m perisim smoke
```

`smoke` generates synthetic data with the schema of the three cohorts and runs the three experiments (T2, T1,
T3) end to end, including the freeze and the external validation, with the `config/profiles/smoke.yaml` profile
(`--experiments t2` for one only).

## Structure

```
config/            cohorts, harmonization (the only place with original names), targets, model_space, experiment
src/perisim/
  data/            lazy reading (Parquet zstd), discovery, INSPIRE / MIMIC-IV / eICU extractors, synthetic data
  cohort/          inclusion, exclusion, t0 and cutoff
  targets/         T1 (KDIGO, antibiotic course, readmission…), T2, T3, carried variables
  features/        inputs with temporal provenance (anti-leakage test) and metric types
  importance/      feature-dropout surrogates, permutation SAGE, shape functions (ALE + consensus), stopping (§5.3)
  metric/          shape / neutral / HVDM geometries of δ_j and regularised diagonal NCA (§5.2, §5.4)
  knn/             FAISS, smoothed probabilities, out of support and per-patient output (§5.5–5.6)
  baselines/       §6.4 comparators
  evaluation/      metrics, patient-level bootstrap, audit, reports and export
  experiment.py    nested fitting and evaluation of PERISIM and the comparators within one fold
  external.py      external validation, modes A / B (final models fitted in parallel)
  transport/       transportability and incremental case base (§6.2b, pre-registration amendment)
  sensitivity.py   outcome-definition variants by relabelling the main metric's case base
  ksweep.py        secondary k sweep of the finished CV (a command, not a stage)
  compare.py       similarity of two experiments' metrics (neighbour overlap, weight ranks, shapes)
  resources.py     real container limits (cgroup CPU quota, affinity, memory) → jobs × threads plan
  pipeline.py      stages, gates and parallelism (at most 4 processes × 4 threads)
tests/             anti-leakage, patient-level split, VDM in the fold, metric and targets
docs/              OPEN_QUESTIONS.md and DECISIONS.md
results/           published aggregate results per experiment (reports, tables, figures, aggregate JSON)
scripts/           public-safety check (pre-commit hook), publish_results.py and the post-hoc posthoc_*.py
```
