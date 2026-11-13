"""Distillation of a PERISIM metric into a synthetic cohort built from population aggregates (docs/prereg_distill.md).

- `aggregates`: the only quantities taken from the source sites' rows (marginals, missingness, category shares and
  a rank-correlation matrix, each on >= MIN_PATIENTS patients), and the synthetic cohorts drawn from them (SYN-COP:
  Gaussian copula; SYN-IND: independent marginals).
- `scripts/distill_run.py`: the study (aggregates → cohorts → teacher labels → metric → freeze → evaluation → contrast).
"""
