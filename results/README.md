# PERISIM — published results (aggregates only)

One folder per experiment (`t2/`, then `t1/`, `t3/`), copied from the pipeline's export package by
`python scripts/publish_results.py <experiment>`. What is here:

| folder | content |
|---|---|
| `reports/` | HTML reports: internal validation (`results_main.html`), sensitivity variants, k sweep (`ksweep.html`), pilot, completeness audits, cohort schemas; later the external validation and transport reports |
| `results/` | tables (CSV) and figures (PNG) of the k sweep and, later, of the transport analyses |
| `artifacts/` | aggregate JSON behind the reports (pooled metrics with bootstrap CIs, per-fold summaries, weights, shapes, D7 stopping parameters) |
| `docs/` | decisions logged automatically by the pipeline and the freeze record for the registration (`OSF_registration.md`) |

**Aggregates only.** Metrics, curves, weights, shape functions and tables with cells of n ≥ 10; every file is scanned
for stay/patient identifiers before it is copied. Patient-level data (the datasets in `data/` and the extractions,
features and per-patient predictions in `work/`) are restricted by the PhysioNet data use agreements and are never
copied here. The per-fold CV artifacts are left out because of their size; they are summarised in `results*.json`.

**Provenance.** T2 internal results were produced by the code of the freeze tag `perisim-freeze-t2-2026-10-27-141040`
(commit `c4cc8b7`; the k sweep by the same commit). The external validation and transport results, when added, come
from that frozen version: see `t2/docs/OSF_registration.md`.
