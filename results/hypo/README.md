# `hypo` results (aggregates only)

Experiment `hypo`: hypotension (MAP < 65 mmHg) in the next 6 h in the ICU after surgery, with INSPIRE, MIMIC-IV
and eICU as development and external sites by rotation. Design: [docs/design_hypotension.md](../../docs/design_hypotension.md);
pre-registration: [docs/prereg_hypo.md](../../docs/prereg_hypo.md); freeze record: [docs/freeze_hypo.md](../../docs/freeze_hypo.md)
(tag `perisim-freeze-hypo-2026-10-29-120652`, code `cf0d0b80c70a`).

Only aggregates are here. Patient-level data (landmarks, predictions, case bases, fitted models) stay on the
analysis computer under the PhysioNet data use agreements. eICU hospitals appear only as H1, H2, … by size, never by
their identifier.

| path | what |
|---|---|
| [`REPORT.md`](REPORT.md) | **Phase 5: the report** (draft for the lead) |
| `reports/landmarks_summary.*`, `reports/audit.*` | Phase 0: landmark tables and completeness audit |
| `reports/smoke_external-eicu.*` | Phase 1: smoke run of the model |
| `reports/cv.*` | Phase 2: internal validation by rotation (development results) |
| `reports/final.md` | Phase 3: the frozen artefacts (weights, MAP thresholds) |
| `reports/external_dryrun.*` | Phase 3: dry run of the external stage, in-sample, mechanics only (after the α fix; prereg §3) |
| `reports/external.*` | **Phase 4: the external stage**, H1–H5 (confirmatory) and S1–S8 (secondary) |
| `reports/posthoc.md` | post hoc follow-ups (exploratory, not pre-registered; REPORT §13): local weights swapped in, MAP-block threshold, drift over five folds |
| `reports/posthoc_ablation.md`, `reports/posthoc_ablation/*.png` | post hoc, second round (exploratory; plan in DECISIONS 2026-10-30): AUPRC, Brier, calibration curves and decision curves of the external stage, and the ablation of the metric (geometry × weights) |
| `results/freeze.json` | the freeze record with every hash the external stage checked |
| `results/cv/`, `results/final/`, `results/external/`, `results/external_dryrun/`, `results/posthoc/`, `results/posthoc_ablation/` | per-unit summaries (α, weights, shapes, drift, thresholds, timings, metrics), JSON only |
| `results/external_dryrun_alpha0/` | the first dry run of the external stage, the one that found the α = 0 fault (DECISIONS 2026-10-29); kept for the record, not evidence |
