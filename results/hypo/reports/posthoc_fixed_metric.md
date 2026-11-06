# A fixed mutual-information metric against the transported learned metric (`hypo`, exploratory)

Plan fixed before running: docs/DECISIONS.md, 2026-11-06. Same case base, k, neighbours and vote in every arm; only the metric changes. Δ1 = log-loss(MI-transported) − log-loss(NCA-transported): the learned metric adds at a site when the lower limit of its 95 % CI is > 0. Q = share of the transported NCA metric's gain over the Euclidean kNN already obtained by mutual-information weights. 2,000 patient resamples.

| site | landmarks | Euclidean | NCA-transported | NCA-local | MI-transported | MI-local | NCA weights, neutral |
|---|---|---|---|---|---|---|---|
| inspire | 148,293 | 0.3381 | 0.3276 | 0.3265 | 0.3324 | 0.3287 | 0.3294 |
| mimic | 387,159 | 0.5138 | 0.5059 | 0.5031 | 0.5051 | 0.5058 | 0.5044 |
| eicu | 466,385 | 0.4100 | 0.3965 | 0.3960 | 0.4005 | 0.3976 | 0.3969 |

| site | stratum | Δ1 (95 % CI) | learned adds | Q (90 % CI) | R NCA | R MI | Δ2 MI-local − NCA | Δ3 shapes |
|---|---|---|---|---|---|---|---|---|
| inspire | all | +0.0048 (+0.0036 to +0.0060) | yes | 0.55 (0.46–0.63) | 0.91 | 0.50 | +0.0011 | +0.0019 |
| mimic | all | -0.0008 (-0.0016 to -0.0001) | no | 1.11 (1.03–1.20) | 0.74 | 0.82 | -0.0001 | -0.0015 |
| mimic | vaso_on = 0 | +0.0006 (-0.0002 to +0.0015) | no | 0.93 (0.85–1.01) | 0.84 | 0.78 | +0.0016 | -0.0017 |
| mimic | vaso_on = 1 | -0.0062 (-0.0081 to -0.0045) | no | 2.14 (1.64–3.36) | 0.44 | 0.94 | -0.0064 | -0.0007 |
| eicu | all | +0.0040 (+0.0033 to +0.0047) | yes | 0.70 (0.67–0.74) | 0.96 | 0.68 | +0.0011 | +0.0004 |
| eicu | vaso_on = 0 | +0.0044 (+0.0037 to +0.0051) | yes | 0.65 (0.60–0.69) | 0.98 | 0.63 | +0.0014 | +0.0005 |
| eicu | vaso_on = 1 | -0.0002 (-0.0025 to +0.0021) | no | 1.01 (0.94–1.08) | 0.90 | 0.91 | -0.0022 | -0.0012 |

| site | arm | AUROC | CITL | slope |
|---|---|---|---|---|
| inspire | euclidean | 0.7647 | +0.062 | 1.187 |
| inspire | nca_transported | 0.7788 | +0.023 | 1.019 |
| inspire | nca_local | 0.7818 | +0.033 | 1.064 |
| inspire | mi_transported | 0.7696 | +0.021 | 1.061 |
| inspire | mi_local | 0.7770 | +0.045 | 1.083 |
| inspire | nca_w_neutral | 0.7780 | +0.027 | 1.062 |
| mimic | euclidean | 0.7734 | -0.022 | 1.112 |
| mimic | nca_transported | 0.7806 | +0.021 | 1.011 |
| mimic | nca_local | 0.7836 | -0.002 | 1.012 |
| mimic | mi_transported | 0.7815 | +0.012 | 1.005 |
| mimic | mi_local | 0.7808 | -0.010 | 1.024 |
| mimic | nca_w_neutral | 0.7823 | -0.012 | 1.017 |
| eicu | euclidean | 0.7845 | -0.003 | 1.171 |
| eicu | nca_transported | 0.7992 | +0.008 | 1.025 |
| eicu | nca_local | 0.7998 | +0.008 | 1.046 |
| eicu | mi_transported | 0.7941 | -0.006 | 1.065 |
| eicu | mi_local | 0.7974 | +0.031 | 1.043 |
| eicu | nca_w_neutral | 0.7993 | +0.009 | 1.056 |

**Rule (pre-specified):** the learned metric adds over a fixed mutual-information metric — mixed: inspire yes, mimic no, eicu yes.
