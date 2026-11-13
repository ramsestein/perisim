# PERISIM — transporting a learned patient-similarity metric across hospitals

PERISIM asks whether the **similarity metric of a k-nearest-neighbour (kNN) model, learned at some hospitals, can be
frozen and used at another**, voting among that hospital's own patients, with nothing refitted.

The model is deliberately simple: a kNN whose distance gives each variable a weight learned by a diagonal
neighbourhood components analysis (NCA). The contribution is not a new algorithm. It is a careful, partly
pre-registered measurement of **what travels and what is lost** when the metric moves between health systems, on
postoperative ICU patients in four databases.

> **Branch `arc-metric`.** This branch re-runs every experiment in two arms that differ only in how a shaped value is
> placed on its axis: the published placement (`value`) and the arc length of the shape function (`arc`). Plan:
> [docs/prereg_arc.md](docs/prereg_arc.md). How to run it: [docs/RUNNING.md](docs/RUNNING.md#branch-arc-metric-every-experiment-in-two-arms).
> **Result** ([results/arc-metric/REPORT.md](results/arc-metric/REPORT.md)): the two placements predict practically
> the same (|D| ≤ 0.0014 nats in the primary units, arc never worse); formal equivalence fails in one of six units, on
> the side favourable to arc. Everything below describes the published results.

> **Start here:** [docs/FINDINGS.md](docs/FINDINGS.md) summarises every result in plain language, with the strength
> of evidence, the relation to prior work, and links to the full reports.

## Key results

The outcome is hypotension in the next 6 h after surgery in the ICU (`hypo`). The data are INSPIRE (Seoul), MIMIC-IV
(Boston), eICU (208 US hospitals) and NWICU (Chicago).

| question | answer | evidence |
|---|---|---|
| Does a metric learned at two systems keep its gain at a third? | Yes in INSPIRE and eICU (R 0.91 and 0.96, equivalent); not shown in MIMIC-IV (0.74) | pre-registered rotation ([report](results/hypo/REPORT.md)) |
| Does it hold in a new, independent system? | Yes: all three frozen metrics are equivalent to, and better than, one learned in NWICU (R 1.46–1.64) | pre-registered on OSF, [10.17605/OSF.IO/P5CA3](https://doi.org/10.17605/OSF.IO/P5CA3) ([results](results/nwicu/README.md)) |
| For a single hospital, is it better to borrow the metric or learn it locally? | Borrow: borrowed weights ≥ local ones in all 8 evaluable eICU hospitals, whether borrowed from the same system or from abroad | post hoc, pre-specified ([§20](results/hypo/REPORT.md)) |
| Does the learned metric beat a plain cosine kNN? | Yes, at all four targets; cosine is no better than Euclidean | post hoc, pre-specified ([§21](results/hypo/REPORT.md)) |
| What makes the predictions calibrated? | The local case base, with any reasonable metric; no recalibration needed | [§4, §17](results/hypo/REPORT.md) |
| Is the kNN the most accurate option? | No. A transported LightGBM with a two-parameter recalibration, or a local LightGBM, predicts better | [results/nwicu](results/nwicu/README.md), [§3](results/hypo/REPORT.md) |

R is the share of a locally learned metric's gain over a Euclidean kNN that the transported metric retains. It is
called "equivalent" when the lower limit of its 90 % CI is ≥ 0.80, by the bioequivalence convention.

## Experiments

| experiment | question | status | report |
|---|---|---|---|
| **T2** | transport at ICU discharge (length of stay, death) | done; weak signal, not equivalent | [results/t2/](results/t2/) |
| **hypo** | transport of the metric for hypotension in the next 6 h, by rotation across INSPIRE, MIMIC-IV and eICU, with post hoc follow-ups | done | [results/hypo/REPORT.md](results/hypo/REPORT.md) |
| **roll** | a rotating case base along a stream of patients, against LightGBM managed as a hospital would | done; better calibrated, non-inferior only in MIMIC-IV | [results/roll/REPORT.md](results/roll/REPORT.md) |
| **NWICU** | the frozen `hypo` metrics in a fourth system | done; pre-registered replication confirmed | [results/nwicu/README.md](results/nwicu/README.md) |

## Repository map

| path | what it holds |
|---|---|
| [docs/FINDINGS.md](docs/FINDINGS.md) | the results in plain language (read first) |
| [docs/](docs/README.md) | pre-registrations, freeze records, designs, the decision log, the audit, how to run |
| [results/](results/README.md) | published aggregate results per experiment: reports, tables, figures, JSON |
| [SPECS.md](SPECS.md) | the original specification of the method and of T2 (a frozen record) |
| `src/perisim/` | the core library and the T2 pipeline: extraction, features, metric (encoder, shapes, NCA), kNN, evaluation |
| `src/hypo/`, `src/roll/` | the `hypo` and `roll` experiments, each with its own command line (`python -m hypo`, `python -m roll`) |
| `scripts/` | post hoc analyses (`posthoc_*.py`), the NWICU build and transport (`nwicu_*.py`), publication and safety checks |
| `frozen/` | the exact v0.5 code and configuration that produced each published result, with their freeze hashes ([README](frozen/README.md)) |
| `config/` | cohorts, variable harmonisation, model space and experiment settings |
| `tests/` | unit, anti-leakage and end-to-end tests on synthetic data |

## Reproducing the results

**Data.** All four databases are on PhysioNet under credentialed access: complete the training, sign each data use
agreement, and download them yourself.
- INSPIRE: https://physionet.org/content/inspire/
- MIMIC-IV: https://physionet.org/content/mimiciv/
- eICU-CRD: https://physionet.org/content/eicu-crd/
- NWICU: https://physionet.org/content/nwicu-northwestern-icu/

No patient-level data are in this repository.

**Environment.** Python ≥ 3.11:

```bash
pip install -r requirements-lock.txt   # exact versions used; requirements.txt for ranges
pip install -e .
pytest -q                               # synthetic data only; about 260 tests
```

**Code versions.** The code in `src/` is v0.6, which fixes the defects found by the audit of 2026-11-05. Every
published result was produced by v0.5, kept file for file in `frozen/`. To re-run a published analysis against the
code that produced it, set `PERISIM_SRC=frozen/<experiment>/src`. The NWICU landmarks use v0.6's corrected
landmark code, as pre-registered.

**Commands.** [docs/RUNNING.md](docs/RUNNING.md) has the full operational guide:
- the T2 pipeline and its stages;
- the `hypo` and `roll` phases;
- the local Windows runner (`run_local.ps1`) and the pod runner (`run_pod.sh`);
- environment variables and outputs.

The post hoc and NWICU scripts document their usage in their headers.

## Privacy

This is a public repository: it holds code, configuration and aggregate results only.
- Every published table has cells of n ≥ 10, and no count between 1 and 9 is derivable from it.
- `python scripts/check_public_safety.py --all` checks this and runs as a pre-commit hook.
- Patient-level data, extractions and per-patient predictions stay on the analyst's machine.

## Project history and naming

The project began as a kNN with SAGE-weighted distances for postoperative patients (SPECS v0.5, "CE-kNN"). The
analyses showed that the learned NCA weights, not the SAGE prior or the per-variable shapes, carry the useful
structure, and the focus moved to transport. Earlier names (CE-kNN, PM-kNN) remain in the frozen records, which are
not rewritten.

Every decision, deviation and correction is dated in [docs/DECISIONS.md](docs/DECISIONS.md).

## License and citation

The code is released under the MIT License ([LICENSE](LICENSE)). To cite, see [CITATION.cff](CITATION.cff).
Author: Ramsés Marrero García (Hospital Clínic de Barcelona).
