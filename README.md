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

In the pod's VSCode terminal:

```bash
mkdir -p ~/workspace/updown/perisim/$(id -un) && cd ~/workspace/updown/perisim/$(id -un)
git clone http://github.khdp-forward-server/ramsestein/perisim.git
cd perisim
bash run_pod.sh
```

`run_pod.sh` does everything on its own:

1. Finds Python ≥ 3.11 (or creates one with conda, which on KHDP already points to the mirror) and creates a
   virtual environment in `~/.perisim/venv`, reusing the system packages.
2. Installs the dependencies from the KHDP PyPI mirror and torch from the PyTorch mirror
   (CPU index and, if it does not respond, cu126).
3. The first time each version of the code runs, it launches a **smoke test on synthetic data**: if it fails,
   the real data are not touched.
4. Launches the pipeline **in the background**, so that it survives closing the terminal, and shows the log.
   Ctrl-C stops following the log, but does not stop the pipeline.

It also detects the container's real CPU and memory limits (cgroup) and sizes processes × threads
automatically, at most 4 × 4 (SPECS §9.1). The consensus loop is checkpointed after every accepted model, so
`stop` (or a pod restart) costs minutes: relaunching resumes the loop where it was.

Experiments: `bash run_pod.sh` runs T2; `PERISIM_EXPERIMENT=t1 bash run_pod.sh` (later `t3`) runs the next one
on the same extraction. Only one run at a time.

| Command | Purpose |
|---|---|
| `bash run_pod.sh` | set up the environment and launch / **resume** the pipeline |
| `bash run_pod.sh status` | which stages are done and where it stopped |
| `bash run_pod.sh logs` | follow the log of the running execution |
| `bash run_pod.sh stop` | stop it |
| `bash run_pod.sh fg --until audit` | run in the foreground up to a given stage |
| `bash run_pod.sh test` | pytest, including the anti-leakage tests on the real data |
| `bash run_pod.sh freeze --registration <ID>` | §6.3 freeze of the current experiment (enables its external validation) |
| `PERISIM_EXPERIMENT=t1 bash run_pod.sh` | run another experiment (t1, t3) on the shared extraction |

To update the code: `git pull` and `bash run_pod.sh`. Only the stages whose code or configuration have changed
are rerun, because each stage stores a marker with its hash. After an OOM or a pod restart, just relaunch:
it resumes from the last completed stage (or fold).

### Paths (configurable with environment variables)

| What | Default | Variable |
|---|---|---|
| Data (read-only) | `/home/khdp-user/workspace/datasets` | `PERISIM_DATASETS` |
| INSPIRE / MIMIC-IV / eICU | automatic discovery under datasets | `PERISIM_INSPIRE_DIR`, `PERISIM_MIMIC_DIR`, `PERISIM_EICU_DIR` |
| Work and results | `~/workspace/updown/perisim/<user>/work` (shared data) and `…/work/experiments/<t1\|t2\|t3>/` | `PERISIM_WORKDIR`, `PERISIM_EXPERIMENT` |
| Environment and terminal logs | `~/.perisim` | `PERISIM_LOCAL` |

## What the pipeline does and where it stops

```
env → discover → extract → targets → features → leakage → audit ⛔ → harmonize ⛔ → pilot → cv → temporal
    → report → sensitivity → external ⛔ → export
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
| `export` | `export/`: aggregates only, with a manifest and an identifier scan | — |

The three cohorts are extracted in parallel, the sensitivity variants share a single job queue, and the final
models of the external validation are fitted in parallel.

**Expected first run:** it will almost certainly stop at `discover` or at `audit`. In that case:

1. Review `reports/schema_*.md` and `reports/completeness_*.html` inside the workdir.
2. Fix `config/harmonization.yaml` on your machine, push, and then `git pull` in the pod.
3. Record the decisions in `docs/OPEN_QUESTIONS.md` and `docs/DECISIONS.md`.
4. To continue after the audit: `PERISIM_ACK_AUDIT=true bash run_pod.sh` (once: the acknowledgement is stored for that exact set of issues; a different set stops the pipeline again).

**External validation:** once internal validation of the experiment is closed, confirm the state of the repository
and freeze it (one freeze per experiment):

```bash
bash run_pod.sh freeze --registration <ID_OSF>
```

Then relaunch `bash run_pod.sh`. From that moment on, any change to code or configuration blocks
the `external` stage, because §6.3 forbids iterating after looking at external results. Paths, compute
resources (`PERISIM_JOBS`, `PERISIM_THREADS`) and gate acknowledgements are not part of the frozen method, so
changing them does not break the freeze; a second freeze is refused once external results exist.

## Export (KHDP §3.2)

`<workdir>/export/` contains aggregates only: HTML reports, metrics, w_j weights, importances and VDM tables with
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
  sensitivity.py   outcome-definition variants by relabelling the main metric's case base
  compare.py       similarity of two experiments' metrics (neighbour overlap, weight ranks, shapes)
  resources.py     real container limits (cgroup CPU quota, affinity, memory) → jobs × threads plan
  pipeline.py      stages, gates and parallelism (at most 4 processes × 4 threads)
tests/             anti-leakage, patient-level split, VDM in the fold, metric and targets
docs/              OPEN_QUESTIONS.md and DECISIONS.md
```
