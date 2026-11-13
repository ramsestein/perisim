# Transport of the weights between eICU hospitals

Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. Each target hospital's own case base; same encoder (the frozen eICU-target metric's); R = (L_uniform − L_arm)/(L_uniform − L_local), 90 % CI. Hospitals anonymised by size rank.

| hospital | patients | landmarks | prevalence | log-loss uniform / cross-system / other hospitals / local | R other hospitals | R cross-system | cross − other (95 % CI) |
|---|---|---|---|---|---|---|---|
| h1 | 1,680 | 28,164 | 14.6% | 0.3306 / 0.3284 / 0.3259 / 0.3267 | not evaluable | not evaluable | +0.0025 (+0.0006 to +0.0043) |
| h2 | 1,102 | 13,625 | 20.0% | 0.4238 / 0.4164 / 0.4174 / 0.4205 | not evaluable | not evaluable | -0.0009 (-0.0034 to +0.0014) |
| h3 | 941 | 12,157 | 21.5% | 0.4447 / 0.4247 / 0.4225 / 0.4276 | 1.30 (1.12–1.54) | 1.17 (0.98–1.42) | +0.0022 (-0.0004 to +0.0050) |
| h4 | 818 | 13,631 | 23.0% | 0.4721 / 0.4520 / 0.4500 / 0.4547 | 1.27 (1.10–1.50) | 1.15 (0.97–1.39) | +0.0020 (-0.0007 to +0.0049) |
| h5 | 772 | 12,277 | 25.1% | 0.4655 / 0.4384 / 0.4347 / 0.4492 | 1.89 (1.57–2.49) | 1.66 (1.36–2.17) | +0.0037 (+0.0005 to +0.0069) |
| h6 | 743 | 11,629 | 13.6% | 0.3444 / 0.3345 / 0.3311 / 0.3367 | 1.72 (1.23–2.72) | 1.29 (0.91–2.05) | +0.0033 (+0.0000 to +0.0064) |
| h7 | 729 | 10,864 | 21.5% | 0.4545 / 0.4341 / 0.4320 / 0.4415 | 1.73 (1.39–2.34) | 1.57 (1.28–2.08) | +0.0020 (-0.0013 to +0.0052) |
| h8 | 711 | 11,559 | 16.8% | 0.3937 / 0.3758 / 0.3736 / 0.3753 | 1.09 (0.94–1.27) | 0.97 (0.82–1.14) | +0.0022 (-0.0003 to +0.0048) |
| h9 | 699 | 11,184 | 16.9% | 0.3880 / 0.3704 / 0.3683 / 0.3782 | 2.02 (1.56–3.05) | 1.80 (1.37–2.69) | +0.0021 (-0.0011 to +0.0053) |
| h10 | 663 | 9,323 | 22.6% | 0.4698 / 0.4469 / 0.4419 / 0.4510 | 1.49 (1.27–1.86) | 1.22 (1.02–1.51) | +0.0051 (+0.0011 to +0.0089) |
| pooled | 8,858 | 134,413 | 19.0% | 0.4081 / 0.3935 / 0.3911 / 0.3967 | 1.49 (1.39–1.61) | 1.29 (1.19–1.40) | +0.0023 (+0.0014 to +0.0032) |

AUROC and CITL by arm are in hospitals.json.

## Rule (pre-specified)

```
{
 "Q1_within_system_equivalent_pooled": "equivalent",
 "Q2_cross_vs_other_pooled": "other hospitals better",
 "Q3_hospitals_evaluable": 8,
 "Q3_hospitals_equivalent": 8,
 "Q3_cross_system_equivalent": 8
}
```
