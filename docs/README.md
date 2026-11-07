# Documents

Some documents are **records**: frozen at a date, kept as written, and never edited, because they are evidence of what
was decided and when. The others are **living** documents, kept up to date. The difference matters when reading:
for example, a pre-registration describes the plan, not the results.

## Read first

| document | what it is |
|---|---|
| [FINDINGS.md](FINDINGS.md) | every result in plain language: strength of evidence, relation to prior work, limits (living) |
| [RUNNING.md](RUNNING.md) | how to run each experiment, the pipeline stages, environment variables and outputs (living) |
| [DECISIONS.md](DECISIONS.md) | the dated log of every decision, plan, deviation, audit finding and result, newest first (living, append-only) |

## Pre-registrations and freezes (records)

| document | experiment |
|---|---|
| [prereg_nwicu.md](prereg_nwicu.md) | NWICU replication, registered on OSF ([10.17605/OSF.IO/P5CA3](https://doi.org/10.17605/OSF.IO/P5CA3)) before any outcome was built |
| [prereg_hypo.md](prereg_hypo.md), [freeze_hypo.md](freeze_hypo.md) | `hypo`: plan, hypotheses and the freeze (code and configuration hashes) |
| [prereg_roll.md](prereg_roll.md), [freeze_roll.md](freeze_roll.md) | `roll` |
| [prereg_amendment_transport.md](prereg_amendment_transport.md) | T2's amendment adding the transport analyses before the external stage |

T2's freeze record is in [results/t2/docs/](../results/t2/docs/).

## Design and feasibility (records)

| document | content |
|---|---|
| [design_hypotension.md](design_hypotension.md), [feasibility_hypotension.md](feasibility_hypotension.md) | why `hypo` and its use case were chosen |
| [design_roll.md](design_roll.md), [feasibility_roll.md](feasibility_roll.md) | the same for `roll` |
| [transport_review.md](transport_review.md) | the review behind T2's transport analyses |

## Audit (records)

| document | content |
|---|---|
| [AUDIT.md](AUDIT.md) | the pre-run audit of the T2 pipeline (2026-10-22) |
| [DECISIONS_PENDING.md](DECISIONS_PENDING.md) | the options weighed for each audit item and the choices made |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | data questions the code did not decide silently, and how each was resolved |

The audits of 2026-11-05 (software, which led to v0.6) and of 2026-11-07 (the post hoc analyses and NWICU) are
entries in [DECISIONS.md](DECISIONS.md).
