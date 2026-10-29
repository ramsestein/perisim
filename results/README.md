# PERISIM — published results (aggregates only)

One folder per experiment (`t2/`, then `t1/`, `t3/`), copied from the pipeline's export package by
`python scripts/publish_results.py <experiment>`. The separate experiment `hypo` (hypotension in the ICU after
surgery) has its own folder, [`hypo/`](hypo/README.md), with its report and a layout of its own (described there).
What is in the pipeline's folders:

| folder | content |
|---|---|
| `reports/` | HTML reports: internal validation (`results_main.html`), sensitivity variants, k sweep (`ksweep.html`), pilot, completeness audits, cohort schemas; later the external validation and transport reports |
| `results/` | tables (CSV) and figures (PNG) of the k sweep, the transport analyses and the exploratory post-hoc comparison (`posthoc/`, with one shape figure per variable in `posthoc/shapes/`) and drift analysis (`posthoc_drift/`, one figure per shaped variable in `posthoc_drift/shapes/`; missing values only in `posthoc_drift_missing/`) and the case-base analyses (`posthoc_case_base/`: patients needed, hospital-specific relations in eICU) |
| `artifacts/` | aggregate JSON behind the reports (pooled metrics with bootstrap CIs, per-fold summaries, weights, shapes, D7 stopping parameters) |
| `docs/` | decisions logged automatically by the pipeline and the freeze record for the registration (`OSF_registration.md`) |

**Aggregates only.** Metrics, curves, weights, shape functions and tables with cells of n ≥ 10; every file is scanned
for stay/patient identifiers before it is copied. Patient-level data (the datasets in `data/` and the extractions,
features and per-patient predictions in `work/`) are restricted by the PhysioNet data use agreements and are never
copied here. The per-fold CV artifacts are left out because of their size; they are summarised in `results*.json`.

**Provenance.** T2 internal results were produced by the code of the freeze tag `perisim-freeze-t2-2026-10-27-141040`
(commit `c4cc8b7`; the k sweep by the same commit). The external validation and transport results, when added, come
from that frozen version: see `t2/docs/OSF_registration.md`. `hypo`'s results come from its own freeze tag
`perisim-freeze-hypo-2026-10-29-120652` (commit `b0c9a02`; [docs/freeze_hypo.md](../docs/freeze_hypo.md)), and its
post hoc follow-ups from `scripts/posthoc_hypo.py`, which only reads the frozen artefacts.
