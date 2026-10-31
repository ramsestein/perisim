# `roll` results (aggregates only)

Experiment `roll`: a prequential (test-then-train) evaluation of CE-kNN with a rotating FIFO case base, against ML
managed as a hospital manages it (monitor, trigger, closed-test update, delayed deployment), on `hypo`'s outcome
(hypotension in the next 6 h in the ICU after surgery). Design: [docs/design_roll.md](../../docs/design_roll.md);
pre-registration: [docs/prereg_roll.md](../../docs/prereg_roll.md); freeze record:
[docs/freeze_roll.md](../../docs/freeze_roll.md) (tag `perisim-freeze-roll-2026-10-30-173256`, code `b630267b6518`).

Only aggregates are here. Patient-level data (streams, predictions, logs, case bases) stay on the analysis computer
under the PhysioNet data use agreements. eICU hospitals appear only as H1, H2, … by size, never by their identifier.

| path | what |
|---|---|
| [`REPORT.md`](REPORT.md) | **Phase 5: the report** (draft for the lead) |
| `reports/endpoints.md` | the tables of every endpoint the report cites |
| `results/streams/<stream>.json` | the full endpoints of MIMIC-IV temporal, the three cross-cohort streams and the INSPIRE control, every N |
| `results/eicu_pooled.json` | eICU, pooled over hospitals by the frozen code (each hospital's estimate and SE included) |
| `results/eicu_pooled_deviation.json` | the same pools under the declared deviation (DECISIONS 2026-10-31) |
| `results/eicu_<within\|deploy>_hospitals.json` | a compact table per hospital: levels, contrasts against the main comparators, updates, D1, H6 |
| `results/t2.json`, `results/governability.json`, `results/run.json` | T2 (secondary), determinism and reconstruction, the run's units and compute |
| `results/freeze.json` | the freeze record with every hash the run checked |
| `figures/` | forest plots for eICU (a) and (b), and H7 (the window N) |

The JSON comes from the frozen output, through `scripts/roll_deviation_pooling.py` and `scripts/roll_report.py`.
