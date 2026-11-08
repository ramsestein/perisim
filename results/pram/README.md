# Why a learned metric beats cosine in PERISIM but not in PRAM (pre-registered)

Pre-registration: [docs/prereg_pram_factors.md](../../docs/prereg_pram_factors.md), OSF
[10.17605/OSF.IO/3SBFG](https://doi.org/10.17605/OSF.IO/3SBFG). Decisions and deviations, dated:
[docs/DECISIONS.md](../../docs/DECISIONS.md) (2026-11-08).

| file | what it holds |
|---|---|
| `tier2.md`, `tier2.json` | Tier 2: PRAM's setting rebuilt (MIMIC-IV → eICU-CRD; 42 static first-24 h features; acute kidney injury and 168 h mortality); replication of PRAM's Table S13 (R0), every cell of use × vote × NCA implementation, fidelity to PRAM, the cohort summary |
| `tier1.md`, `tier1.json` | Tier 1: the same factors in PERISIM's setting (INSPIRE, MIMIC-IV, eICU, NWICU) — added when the run finishes |

Aggregates only; produced by `scripts/pram_build.py`, `scripts/pram_factors_t2.py` and `scripts/pram_factors_t1.py`.
