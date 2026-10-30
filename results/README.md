# PERISIM — published results (aggregates only)

One folder per experiment:

- `t2/`, copied from the pipeline's export package by `python scripts/publish_results.py <experiment>` (`t1/` and
  `t3/` will follow the same way);
- [`hypo/`](hypo/README.md), the separate experiment `hypo` (hypotension in the ICU after surgery), with its report
  and a layout of its own, described there;
- `feasibility/`, the two feasibility studies that came before `hypo`.

`python scripts/publish_supplement.py` adds what the export package and the hand copies leave out: T2's cohort
flow, two large T2 transport tables split into parts, the feasibility studies, `hypo`'s first dry run and its second
round of post hoc analyses.

What is in T2's folder:

| folder | content |
|---|---|
| `reports/` | HTML reports: internal validation (`results_main.html`), sensitivity variants, k sweep (`ksweep.html`), pilot, completeness audits, cohort schemas, external validation (`external.html`), transport (`transport.html`) and the post hoc reports (`posthoc_*.html`) |
| `results/` | tables (CSV) and figures (PNG). They cover the k sweep and the transport analyses (`transport/`; `t2_F_hospitals` split into parts of < 2 MB). They also cover the exploratory post hoc analyses: the comparison with the parametric comparators (`posthoc/`, one shape figure per variable in `posthoc/shapes/`), drift (`posthoc_drift/`, one figure per shaped variable in `posthoc_drift/shapes/`), drift of the missing values only (`posthoc_drift_missing/`) and the case base (`posthoc_case_base/`: patients needed, hospital-specific relations in eICU) |
| `artifacts/` | aggregate JSON behind the reports (pooled metrics with bootstrap CIs, per-fold summaries, weights, shapes, D7 stopping parameters) |
| `flow/` | the cohort flow of the extraction, per cohort (steps excluding 1–9 stays merged with the next step) |
| `docs/` | decisions logged automatically by the pipeline and the freeze record for the registration (`OSF_registration.md`) |

**Aggregates only.** Metrics, curves, weights, shape functions and tables with cells of n ≥ 10. Every file is scanned
for stay/patient identifiers before it is copied. Patient-level data are never copied here: the datasets in `data/`
and, in `work/`, the extractions, features and per-patient predictions. The PhysioNet data use agreements restrict
them. The per-fold CV artifacts are left out because of their size; they are summarised in `results*.json`.

**Provenance.**
- T2's internal results come from the code of the freeze tag `perisim-freeze-t2-2026-10-27-141040` (commit `c4cc8b7`;
  the k sweep by the same commit). Its external validation and transport results come from that frozen version: see
  `t2/docs/OSF_registration.md`.
- `hypo`'s results come from its own freeze tag `perisim-freeze-hypo-2026-10-29-120652` (commit `b0c9a02`;
  [docs/freeze_hypo.md](../docs/freeze_hypo.md)).
- The post hoc analyses come from `scripts/posthoc_*.py`. Each one only reads the frozen artefacts and checks the
  freeze record at its start and end.

## Where each part of a manuscript is

| section | material |
|---|---|
| Methods: data and harmonisation | `t2/reports/schema_*.md`, `t2/reports/completeness_*.html`, `t2/flow/flow.md`; `hypo/reports/landmarks_summary.md`, `hypo/reports/audit.md`; [docs/AUDIT.md](../docs/AUDIT.md) |
| Methods: the method (CE-kNN) | [SPECS.md](../SPECS.md) §5; the library on the branch `ceknn` |
| Methods: pre-registration and freezes | [docs/prereg_hypo.md](../docs/prereg_hypo.md), [docs/freeze_hypo.md](../docs/freeze_hypo.md), [docs/prereg_amendment_transport.md](../docs/prereg_amendment_transport.md), `t2/docs/OSF_registration.md` |
| Results: T2, internal, with the comparators of SPECS §6.4 | `t2/reports/results_main.html`, `t2/artifacts/main/results.json`. The comparators are Euclidean, HVDM, mean \|SHAP\|, RF importance, ReliefF, NCA only, SAGE only, RF-GAP, logistic regression and LightGBM. Also the sensitivity reports and the k sweep |
| Results: T2, external and transport | `t2/reports/external.html`, `t2/reports/transport.html`, `t2/results/transport/` |
| Results: `hypo`, by rotation | `hypo/REPORT.md`, `hypo/reports/cv.md`, `hypo/reports/external.md` |
| Results: `hypo`, metrics and ablation | `hypo/reports/posthoc_ablation.md` and its figures |
| Results: drift and shapes | `hypo/REPORT.md` §7 and §13, `hypo/reports/posthoc.md`; `t2/reports/posthoc_drift*.html` |
| Supplement: post hoc analyses (exploratory) | T2: `posthoc_transport`, `posthoc_drift`, `posthoc_drift_missing`, `posthoc_case_base`; `hypo`: `posthoc.md`, `posthoc_ablation.md` |
| Supplement: chronology | [docs/DECISIONS.md](../docs/DECISIONS.md) (dated entries), the two freeze records and tags, `hypo/results/external_dryrun_alpha0/` |
| Background: choice of the use case | `feasibility/icu_trajectory.md`, `feasibility/hypotension.md`, [docs/design_hypotension.md](../docs/design_hypotension.md) |
