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
| `reports/posthoc_strata.md` | post hoc, third round (exploratory; plan in DECISIONS 2026-11-04; REPORT §15.1): MIMIC-IV's transport cost by vasopressor and time-since-admission strata, and the §13.1 weight swaps by stratum |
| `reports/posthoc_map_vaso.md`, `reports/posthoc_map_vaso/*.png` | post hoc, third round (exploratory; plan in DECISIONS 2026-11-04; REPORT §15.2): the MAP–risk relation with and without a vasopressor running, MIMIC-IV and eICU |
| `../alignment/` | post hoc, third round (exploratory; plan in DECISIONS 2026-11-04; REPORT §15.3): the neighbour alignment of metrics learned apart (transport, n, the ablation's variants), beside T2's |
| `reports/posthoc_equivalence.md` | transport equivalence (exploratory; criterion fixed with the point estimates known, DECISIONS 2026-11-05; REPORT §16): R with 90 % CIs by the bioequivalence convention, all landmarks and by vasopressor |
| `reports/posthoc_fixed_metric.md` | a fixed mutual-information metric (PRAM-style) against the transported learned metric, same case base (exploratory, plan fixed before running, DECISIONS 2026-11-06; REPORT §17) |
| `results/freeze.json` | the freeze record with every hash the external stage checked |
| `results/cv/`, `results/final/`, `results/external/`, `results/external_dryrun/`, `results/posthoc/`, `results/posthoc_ablation/`, `results/posthoc_strata/`, `results/posthoc_equivalence/`, `results/posthoc_fixed_metric/`, `results/posthoc_map_vaso/` | per-unit summaries (α, weights, shapes, drift, thresholds, timings, metrics), JSON only |
| `results/external_dryrun_alpha0/` | the first dry run of the external stage, the one that found the α = 0 fault (DECISIONS 2026-10-29); kept for the record, not evidence |
