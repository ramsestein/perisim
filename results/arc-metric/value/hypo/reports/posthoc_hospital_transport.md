# Transport of the weights between eICU hospitals

Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. Each target hospital's own case base; same encoder (the frozen eICU-target metric's); R = (L_uniform − L_arm)/(L_uniform − L_local), 90 % CI. Hospitals anonymised by size rank.

| hospital | patients | landmarks | prevalence | log-loss uniform / cross-system / other hospitals / local | R other hospitals | R cross-system | cross − other (95 % CI) |
|---|---|---|---|---|---|---|---|
| h1 | 1,680 | 28,164 | 14.6% | 0.3316 / 0.3269 / 0.3266 / 0.3290 | not evaluable | not evaluable | +0.0003 (-0.0013 to +0.0019) |
| h2 | 1,102 | 13,625 | 20.0% | 0.4217 / 0.4171 / 0.4185 / 0.4201 | not evaluable | not evaluable | -0.0014 (-0.0041 to +0.0011) |
| h3 | 941 | 12,157 | 21.5% | 0.4455 / 0.4250 / 0.4230 / 0.4290 | 1.36 (1.15–1.68) | 1.24 (1.04–1.54) | +0.0020 (-0.0007 to +0.0050) |
| h4 | 818 | 13,631 | 23.0% | 0.4725 / 0.4519 / 0.4508 / 0.4558 | 1.30 (1.12–1.52) | 1.24 (1.04–1.48) | +0.0010 (-0.0021 to +0.0043) |
| h5 | 772 | 12,277 | 25.1% | 0.4637 / 0.4380 / 0.4354 / 0.4462 | 1.61 (1.35–2.04) | 1.46 (1.22–1.84) | +0.0026 (-0.0005 to +0.0059) |
| h6 | 743 | 11,629 | 13.6% | 0.3481 / 0.3329 / 0.3311 / 0.3361 | 1.41 (1.13–1.85) | 1.26 (1.04–1.60) | +0.0018 (-0.0012 to +0.0047) |
| h7 | 729 | 10,864 | 21.5% | 0.4529 / 0.4343 / 0.4334 / 0.4414 | 1.69 (1.31–2.37) | 1.62 (1.28–2.20) | +0.0008 (-0.0030 to +0.0046) |
| h8 | 711 | 11,559 | 16.8% | 0.3945 / 0.3748 / 0.3728 / 0.3803 | 1.52 (1.28–1.87) | 1.38 (1.14–1.71) | +0.0020 (-0.0007 to +0.0048) |
| h9 | 699 | 11,184 | 16.9% | 0.3881 / 0.3709 / 0.3693 / 0.3763 | 1.59 (1.28–2.17) | 1.46 (1.14–1.98) | +0.0016 (-0.0012 to +0.0045) |
| h10 | 663 | 9,323 | 22.6% | 0.4661 / 0.4452 / 0.4419 / 0.4510 | 1.60 (1.27–2.27) | 1.39 (1.10–1.91) | +0.0032 (-0.0001 to +0.0067) |
| pooled | 8,858 | 134,413 | 19.0% | 0.4081 / 0.3929 / 0.3917 / 0.3973 | 1.52 (1.40–1.66) | 1.41 (1.31–1.54) | +0.0012 (+0.0003 to +0.0021) |

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
