# Transport of the weights between eICU hospitals

Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. Each target hospital's own case base; same encoder (the frozen eICU-target metric's); R = (L_uniform − L_arm)/(L_uniform − L_local), 90 % CI. Hospitals anonymised by size rank.

| hospital | patients | landmarks | prevalence | log-loss uniform / cross-system / other hospitals / local | R other hospitals | R cross-system | cross − other (95 % CI) |
|---|---|---|---|---|---|---|---|
| h1 | 1,680 | 28,164 | 14.6% | 0.3304 / 0.3263 / 0.3266 / 0.3260 | not evaluable | not evaluable | -0.0003 (-0.0020 to +0.0014) |
| h2 | 1,102 | 13,625 | 20.0% | 0.4230 / 0.4133 / 0.4188 / 0.4189 | not evaluable | not evaluable | -0.0054 (-0.0086 to -0.0022) |
| h3 | 941 | 12,157 | 21.5% | 0.4461 / 0.4251 / 0.4220 / 0.4289 | 1.40 (1.20–1.68) | 1.22 (1.02–1.48) | +0.0031 (+0.0003 to +0.0060) |
| h4 | 818 | 13,631 | 23.0% | 0.4696 / 0.4521 / 0.4512 / 0.4573 | 1.49 (1.19–1.95) | 1.42 (1.11–1.85) | +0.0009 (-0.0022 to +0.0039) |
| h5 | 772 | 12,277 | 25.1% | 0.4646 / 0.4365 / 0.4347 / 0.4416 | 1.30 (1.13–1.52) | 1.22 (1.05–1.43) | +0.0018 (-0.0017 to +0.0054) |
| h6 | 743 | 11,629 | 13.6% | 0.3453 / 0.3316 / 0.3318 / 0.3361 | 1.46 (1.13–2.00) | 1.49 (1.13–2.06) | -0.0002 (-0.0035 to +0.0033) |
| h7 | 729 | 10,864 | 21.5% | 0.4589 / 0.4341 / 0.4363 / 0.4466 | 1.84 (1.46–2.56) | 2.02 (1.61–2.80) | -0.0022 (-0.0059 to +0.0014) |
| h8 | 711 | 11,559 | 16.8% | 0.3944 / 0.3744 / 0.3728 / 0.3762 | 1.19 (1.01–1.43) | 1.10 (0.91–1.35) | +0.0017 (-0.0010 to +0.0044) |
| h9 | 699 | 11,184 | 16.9% | 0.3875 / 0.3717 / 0.3733 / 0.3760 | 1.23 (0.95–1.73) | 1.37 (1.05–1.95) | -0.0016 (-0.0048 to +0.0018) |
| h10 | 663 | 9,323 | 22.6% | 0.4680 / 0.4443 / 0.4464 / 0.4537 | 1.52 (1.17–2.17) | 1.66 (1.28–2.45) | -0.0020 (-0.0062 to +0.0024) |
| pooled | 8,858 | 134,413 | 19.0% | 0.4081 / 0.3922 / 0.3926 / 0.3966 | 1.35 (1.24–1.46) | 1.38 (1.28–1.50) | -0.0004 (-0.0013 to +0.0005) |

AUROC and CITL by arm are in hospitals.json.

## Rule (pre-specified)

```
{
 "Q1_within_system_equivalent_pooled": "equivalent",
 "Q2_cross_vs_other_pooled": "no difference",
 "Q3_hospitals_evaluable": 8,
 "Q3_hospitals_equivalent": 8,
 "Q3_cross_system_equivalent": 8
}
```
