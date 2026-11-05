# frozen/ — the code that produced the published results (v0.5)

The repository's history was condensed into a single snapshot on 2026-11-05 (v0.6). The three freezes of v0.5
(T2, `hypo`, `roll`) were git tags and commits of that history, which no longer exist. Their code and configuration are
kept here, file for file, so that every published result can still be checked against the code that made it and the
freeze records can still be verified.

| folder | experiment | original tag (commit) | code hash | configuration hash |
|---|---|---|---|---|
| `t2/` | T2 (post-ICU stay) | `perisim-freeze-t2-2026-10-27-141040` (`c4cc8b7`) | `77b25fe8f629` (src/perisim) | `1f6d4f5b89d4` (scientific, profile `local`) |
| `hypo/` | `hypo` (hypotension in the next 6 h) | `perisim-freeze-hypo-2026-10-29-120652` (`b0c9a02`) | `cf0d0b80c70a` (src/hypo + src/perisim) | `772940e4ed77` (hypo.yaml + model_space.yaml) |
| `roll/` | `roll` (rotating case base) | `perisim-freeze-roll-2026-10-30-173256` (`a9cc6d0`) | `b630267b6518` (src/roll + `hypo`'s code hash) | — |

Each folder holds the `src/`, `config/`, `SPECS.md`, `docs/` and requirements of its commit. `COMMIT` records the
original commit id. The files were extracted with `git archive` from the tags before the history was removed, and the
hashes above were recomputed from them and match the freeze records (`docs/freeze_hypo.md`, `docs/freeze_roll.md`,
`results/t2/docs/OSF_registration.md`).

## Verify a freeze

```bash
cd frozen/t2   && PYTHONPATH=src python -c "from perisim.config import code_hash, load_config; print(code_hash(), load_config(None, 'local', {}).scientific_hash())"
cd frozen/hypo && PYTHONPATH=src python -c "from hypo.config import code_hash; from hypo.frozen import config_hash; print(code_hash(), config_hash())"
cd frozen/roll && PYTHONPATH=src python -c "from roll.config import code_hash; print(code_hash())"
```

## Re-run an analysis of a published experiment

The code in the repository's `src/` is v0.6. It fixes the defects of the 2026-11-05 audit, so its code hash differs
from every v0.5 freeze, and the freeze checks refuse to run a v0.5 experiment's stages or post hoc scripts with it. To
re-run them, point the scripts at the experiment's frozen code with `PERISIM_SRC`:

```powershell
$env:PERISIM_SRC = "frozen\hypo\src"; .\run_local.ps1 equivalence
```

The published results stay those of v0.5. They were not regenerated with v0.6 (docs/DECISIONS.md, 2026-11-05).

The frozen code contains the defects the audit found. They are listed, with their effect on the published results,
in docs/DECISIONS.md (entry "v0.6"). None changes a confirmatory verdict.
