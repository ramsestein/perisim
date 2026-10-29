# PERISIM `hypo` — freeze record

**Registration ID:** `LOCAL-PREREG-HYPO-2026-10-29` (local; D6 decided by the lead on 2026-10-29: no OSF filing).

The analysis plan is [prereg_hypo.md](prereg_hypo.md), approved by the lead before the freeze. It was fixed by the
freeze below **before any external result existed**: the external stage started after the tag was pushed, and it
refuses to run on anything that differs from the frozen version.

## What was frozen

| item | value |
|---|---|
| freeze time (workstation clock) | 2026-10-29 12:06:52 |
| git tag | `perisim-freeze-hypo-2026-10-29-120652` (annotated tag object `faad75a564be`; pushed to GitHub on 2026-10-29, before the external stage ran) |
| commit | `b0c9a02449bc7f90f75aaee8971731ee8cda7ab9` (branch `hypotension`, pushed to github.com/ramsestein/perisim on 2026-10-29) |
| code hash (src/hypo + src/perisim, line endings normalised) | `cf0d0b80c70a` |
| configuration hash (hypo.yaml + model_space.yaml, line endings normalised) | `772940e4ed77` |
| external stage started | 2026-10-29 12:07:25, under this tag (`external/stage.json`) |

The public commit and tag are the verifiable timestamp. Anyone can check that the code and configuration of that
commit produce these hashes.

## Documents at that commit (SHA-256 of the committed file, first 16 characters)

| file | sha256 |
|---|---|
| `docs/prereg_hypo.md` (pre-registration: H1–H5, S1–S8, rules, limitations) | `2fb3525f5278b4f2` |
| `docs/design_hypotension.md` (design) | `605fa17d10478234` |
| `docs/DECISIONS.md` (dated decisions and deviations) | `6c7d3ec20940fd1f` |
| `config/hypo.yaml` | `8de1324222602b79` |
| `config/model_space.yaml` | `a811c4cc73118842` |

The configuration snapshot in `state/frozen_config/` is the same as the committed files once line endings are
normalised.

## Frozen inputs of the external stage (local, never uploaded: they hold patient-level data)

Hashes recorded in `state/freeze.json` and checked at the start and the end of the external stage:

| item | sha256 (first 16) |
|---|---|
| Phase 2 folds, INSPIRE | `68665fa041dead72` |
| Phase 2 folds, MIMIC-IV | `77428c10f55839b2` |
| Phase 2 folds, eICU | `ce7fc67b42803550` |
| Phase 2 local comparators (75 files) | `c369c32ed35b4993` |
| `metric_ext-inspire.pkl` (learned on MIMIC-IV + eICU) | `738a7989b3e0d890` |
| `metric_ext-mimic.pkl` (learned on INSPIRE + eICU) | `8dbfea6db4488981` |
| `metric_ext-eicu.pkl` (learned on INSPIRE + MIMIC-IV) | `ee43ae7b6cff5417` |
| `comp_ext-inspire_{last_map,logreg,lightgbm}.pkl` | `2af2cebdde737087`, `349a222340d08339`, `bd4348007085d77d` |
| `comp_ext-mimic_{last_map,logreg,lightgbm}.pkl` | `e2b89f21160ae0c1`, `97f29c49f4ebe7af`, `1a550d0993bf1b6b` |
| `comp_ext-eicu_{last_map,logreg,lightgbm}.pkl` | `588b3ac061dcc498`, `8d947c81cb7417a4`, `c8c6bad9e5ea612c` |

## State at the freeze

- Phases 0–2 complete. Their results were seen by the lead and the analyst: they are development results, allowed
  before the freeze (prereg §3).
- The frozen artefacts were fitted (`reports/final.md`). Their weights and MAP thresholds were seen; they are
  development results too.
- A dry run of the whole external stage ran on eICU fold 0, in-sample (served by the metric whose development pair
  includes eICU). It found one fault, α = 0 in small case bases, which was fixed before the freeze. The fix and the
  two in-sample figures it changed are recorded in prereg §3 and DECISIONS.
- External stage (H1–H5 confirmatory, S1–S8 secondary) **not run**. It runs on this frozen version only; any change to
  the code, the configuration, the folds or the artefacts stops it.
