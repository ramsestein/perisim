# PERISIM T2 — freeze record for the OSF registration

**Local registration ID:** `LOCAL-PREREG-T2-20261027`

This is a local placeholder, not an OSF identifier: the OSF registration had not been filed when T2 was frozen. When
it is filed, record the OSF ID next to this one (DECISIONS) and state in the registration that the analysis plan was
fixed by the freeze below, **before any external (MIMIC-IV / eICU) result existed**.

## What was frozen (§6.3)

| item | value |
|---|---|
| freeze time (workstation clock) | 2026-10-27 14:10:40 |
| git tag | `perisim-freeze-t2-2026-10-27-141040` (annotated tag object `9aae1fc70bb5`; pushed to GitHub on 2026-10-27, before the external stage ran) |
| commit | `c4cc8b7c773686e791a4368ba67f96f1274368fe` (pushed to github.com/ramsestein/perisim on 2026-10-27, before the external stage ran) |
| code hash (src/perisim) | `77b25fe8f629` |
| scientific configuration hash | `1f6d4f5b89d4` |
| configuration profile | `local` (no family excluded for cost: declared deviation, DECISIONS 2026-10-25) |

The public commit is the verifiable timestamp: anyone can check that the code and configuration of that commit
produce these hashes (`python -m perisim code-hash`).

## Documents to attach to the registration (versions at that commit; SHA-256, first 16 characters)

| file | sha256 |
|---|---|
| `SPECS.md` (protocol; §6.2b transport amendment, §5.5 k sweep note) | `5c34fd8101d46f09` |
| `docs/prereg_amendment_transport.md` (H1–H6, margins) | `0b0da52613892f6b` |
| `docs/transport_review.md` (transport design) | `8a5aa3a8487b68bd` |
| `docs/DECISIONS.md` (dated decisions and deviations) | `7337842dc842ac79` |

Frozen configuration snapshot (`state/frozen_config/`):

| file | sha256 |
|---|---|
| `cohorts.yaml` | `af3c9558e630939c` |
| `experiment.yaml` | `847792d0f68f7e91` |
| `harmonization.yaml` | `94ba81a09374eb41` |
| `local.yaml` | `20432b74561d6cdd` |
| `model_space.yaml` | `a811c4cc73118842` |
| `targets.yaml` | `dd885066659b12f0` |

## State at the freeze

- Internal validation of T2 complete (nested 5 × 3 CV, report, sensitivity variants); results already seen by the
  lead: they are the development results, allowed before the freeze.
- Secondary k sweep added after the first CV folds had been seen (descriptive; primary k = 50 unchanged).
- External validation (§6.2) and transport analyses (§6.2b, H1–H6 confirmatory for T2) **not run**: they run on this
  frozen version only; any change to code or configuration blocks them.
