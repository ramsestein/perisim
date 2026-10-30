# PERISIM `roll` — freeze record

**Registration ID:** `LOCAL-PREREG-ROLL-2026-10-30` (local, decided by the lead on 2026-10-30; filing on OSF is the
lead's action).

The analysis plan is [prereg_roll.md](prereg_roll.md), approved by the lead before the freeze. It was fixed by the
freeze below **before any prediction along any real stream with the metrics of the run**. The single run (Phase 4)
refuses to run on anything that differs from the frozen version.

`roll` is a **pre-registered retrospective evaluation on data already examined**, not a prospective validation.

## What was frozen

| item | value |
|---|---|
| freeze time (workstation clock) | 2026-10-30 17:32:56 |
| git tag | `perisim-freeze-roll-2026-10-30-173256` (annotated tag object `d7960c9276b6`) |
| commit | `a9cc6d00f8c960dd4d01e229ab0dfc079938038f` (branch `roll`) |
| code hash (src/roll + `hypo`'s code hash, which covers src/hypo and src/perisim; line endings normalised) | `b630267b6518` |
| configuration (`config/roll.yaml`, SHA-256, line endings normalised) | `734680c6193c87c8` |
| the streams: digest of all 1,190 stream-orders (every stream, hospital, N and order; none skipped) | `98e8c48712104ce5` |

The public commit and tag are the verifiable timestamp. Anyone can check that the code and configuration of that
commit produce these hashes.

## The freezes it stands on (checked by `roll` at the start and the end of every stage)

| experiment | tag | code | configuration |
|---|---|---|---|
| T2 | `perisim-freeze-t2-2026-10-27-141040` | `77b25fe8f629` | `1f6d4f5b89d4` |
| `hypo` | `perisim-freeze-hypo-2026-10-29-120652` | `cf0d0b80c70a` | `772940e4ed77` |

`hypo`'s own check also covers its folds, its Phase 2 comparators and its frozen artefacts: the three transported
metrics `metric_ext-<X>.pkl` and comparators `comp_ext-<X>_<method>.pkl` that `roll` uses. T2's `final_model.pkl`
(its external stage's metric, used by `roll`'s T2 part): `053654c00406a05f`.

## Documents at that commit (SHA-256 of the committed file, first 16 characters)

| file | sha256 |
|---|---|
| `docs/prereg_roll.md` (pre-registration) | `c1626e504ccffc4a` |
| `docs/design_roll.md` (design) | `7556263340b75b4c` |
| `docs/feasibility_roll.md` (Phase 0) | `e4df99ca3fc74f9d` |
| `docs/DECISIONS.md` (dated decisions) | `d73a5d98ac664c53` |
| `config/roll.yaml` | `734680c6193c87c8` |

## State at the freeze

- **Phases 0–2 complete**, and the pre-registration approved by the lead. Their results are counts, the precision of
  windows and the noise of the triggers. They were read from `hypo`'s predictions or computed on synthetic data
  (prereg §3).
- **Read before the pre-registration, by the lead's decision**: the calibration of `hypo`'s S1 case bases. N was not
  changed after it.
- **The rehearsal at real size** (prereg §3; DECISIONS 2026-10-30):
  - one unit of each stream kind, with in-sample artefacts;
  - only time, determinism and reconstruction were read; no metric;
  - no fault was found;
  - the endpoints were then made faster, with results tested equal.
- **Not computed**: anything of the single run. `work/experiments/roll/runs/` holds nothing.

## The single run (Phase 4)

```
python -m roll run [--streams …] [--N …] [--threads 8]
python -m roll endpoints
python -m roll t2
```

Each command checks this freeze at its start and its end. The run is resumable, and parts may run side by side in
separate terminals.
