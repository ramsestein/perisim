# `hypo` post hoc follow-ups (exploratory, not pre-registered)

Added on 2026-10-29 at the lead's request after the Phase 5 report (results/hypo/REPORT.md §12). The frozen artefacts are only read; the freeze record was checked at the start and the end. Log-loss differences: negative = better than the frozen transported metric; 95 % CIs by patient bootstrap on the drift test set of fold 0 (S4's case base and test sample).

## B. The weights learned locally (fold 0)

| site | frozen log-loss | local metric (stage) | all weights local, shapes transported | check: the stage's transported predictions |
|---|---|---|---|---|
| inspire | 0.3168 | -0.0016 (-0.0048 to +0.0018) | -0.0019 (-0.0048 to +0.0008) | +0.0000 (+0.0000 to +0.0000) |
| mimic | 0.5028 | -0.0022 (-0.0051 to +0.0009) | -0.0001 (-0.0026 to +0.0029) | +0.0000 (-0.0000 to +0.0000) |
| eicu | 0.3929 | -0.0012 (-0.0036 to +0.0009) | -0.0008 (-0.0028 to +0.0012) | +0.0000 (+0.0000 to +0.0000) |

One variable at a time (|w_local − w_transported| ≥ 0.02):

| site | variable | w local | w transported | Δ log-loss |
|---|---|---|---|---|
| inspire | map_mean_6h | 0.118 | 0.230 | +0.0010 (+0.0001 to +0.0019) |
| inspire | age | 0.007 | 0.110 | -0.0005 (-0.0024 to +0.0013) |
| inspire | L | 0.163 | 0.099 | -0.0004 (-0.0012 to +0.0003) |
| inspire | map_min_6h | 0.129 | 0.169 | +0.0005 (-0.0001 to +0.0011) |
| inspire | sex_male | 0.000 | 0.031 | -0.0003 (-0.0023 to +0.0016) |
| inspire | spo2_min_6h | 0.040 | 0.011 | -0.0000 (-0.0009 to +0.0009) |
| inspire | emergency | 0.031 | 0.006 | +0.0002 (-0.0008 to +0.0012) |
| inspire | lactate_last | 0.023 | 0.000 | +0.0004 (-0.0012 to +0.0020) |
| mimic | map_mean_6h | 0.297 | 0.147 | -0.0000 (-0.0009 to +0.0008) |
| mimic | L | 0.133 | 0.281 | -0.0007 (-0.0016 to +0.0004) |
| mimic | map_min_6h | 0.064 | 0.209 | -0.0010 (-0.0022 to +0.0001) |
| mimic | vaso_on | 0.133 | 0.000 | -0.0017 (-0.0031 to -0.0002) |
| mimic | map_last | 0.099 | 0.210 | +0.0004 (-0.0005 to +0.0013) |
| mimic | age | 0.061 | 0.010 | +0.0000 (-0.0014 to +0.0018) |
| mimic | map_sd_6h | 0.070 | 0.030 | +0.0005 (-0.0004 to +0.0013) |
| mimic | lactate_last__missing | 0.000 | 0.028 | -0.0005 (-0.0018 to +0.0007) |
| mimic | vaso_24h | 0.023 | 0.000 | -0.0009 (-0.0024 to +0.0007) |
| eicu | map_min_6h | 0.213 | 0.079 | +0.0013 (+0.0004 to +0.0022) |
| eicu | map_sd_6h | 0.037 | 0.118 | +0.0009 (-0.0002 to +0.0020) |
| eicu | map_mean_6h | 0.320 | 0.386 | +0.0004 (-0.0000 to +0.0008) |
| eicu | map_last | 0.178 | 0.122 | +0.0004 (-0.0002 to +0.0011) |
| eicu | L | 0.070 | 0.116 | -0.0001 (-0.0008 to +0.0007) |
| eicu | surgical_group | 0.008 | 0.038 | -0.0011 (-0.0025 to +0.0002) |
| eicu | sex_male | 0.043 | 0.014 | -0.0000 (-0.0008 to +0.0006) |
| eicu | sedation_24h | 0.000 | 0.023 | +0.0003 (-0.0009 to +0.0014) |

Re-deformation restarted from the swapped weight (effective weight in the site, units of S4):

| site | variable | eff. weight frozen → start → fitted | start − frozen | fitted − frozen | fitted − start |
|---|---|---|---|---|---|
| inspire | map_mean_6h | 2.716 → 1.392 → 1.263 | +0.0010 (+0.0001 to +0.0019) | +0.0011 (-0.0002 to +0.0022) | +0.0001 (-0.0010 to +0.0009) |
| inspire | age | 1.050 → 0.064 → 0.062 | -0.0005 (-0.0024 to +0.0013) | -0.0004 (-0.0023 to +0.0014) | +0.0000 (-0.0007 to +0.0008) |
| inspire | L | 1.322 → 2.163 → 1.990 | -0.0004 (-0.0012 to +0.0003) | -0.0005 (-0.0013 to +0.0004) | -0.0001 (-0.0010 to +0.0008) |
| inspire | temp_last__missing | 0.000 → 0.065 → 0.065 | -0.0005 (-0.0013 to +0.0002) | -0.0006 (-0.0014 to +0.0002) | -0.0000 (-0.0000 to +0.0000) |
| inspire | lactate_last | 0.003 → 0.731 → 0.626 | +0.0004 (-0.0012 to +0.0020) | +0.0004 (-0.0011 to +0.0017) | -0.0000 (-0.0008 to +0.0009) |
| inspire | sedation_24h | 0.025 → 0.141 → 0.141 | -0.0006 (-0.0018 to +0.0003) | -0.0006 (-0.0018 to +0.0003) | +0.0000 (+0.0000 to +0.0000) |
| mimic | map_mean_6h | 2.773 → 5.607 → 5.274 | -0.0000 (-0.0009 to +0.0008) | -0.0003 (-0.0014 to +0.0007) | -0.0002 (-0.0012 to +0.0008) |
| mimic | L | 4.726 → 2.232 → 2.103 | -0.0007 (-0.0016 to +0.0004) | -0.0004 (-0.0016 to +0.0009) | +0.0003 (-0.0007 to +0.0012) |
| mimic | map_min_6h | 3.812 → 1.169 → 1.094 | -0.0010 (-0.0022 to +0.0001) | -0.0008 (-0.0020 to +0.0004) | +0.0002 (-0.0006 to +0.0010) |
| mimic | hb_last | 0.091 → 0.309 → 0.232 | +0.0003 (-0.0006 to +0.0013) | +0.0006 (-0.0003 to +0.0014) | +0.0003 (-0.0005 to +0.0009) |
| mimic | vaso_on | 0.028 → 16.061 → 16.061 | -0.0017 (-0.0031 to -0.0002) | -0.0017 (-0.0031 to -0.0002) | +0.0000 (+0.0000 to +0.0000) |
| mimic | vaso_24h | 0.004 → 1.436 → 1.436 | -0.0009 (-0.0024 to +0.0007) | -0.0009 (-0.0024 to +0.0007) | +0.0000 (+0.0000 to +0.0000) |
| mimic | vent_on | 0.231 → 0.600 → 0.384 | +0.0001 (-0.0006 to +0.0008) | +0.0001 (-0.0004 to +0.0007) | +0.0001 (-0.0003 to +0.0005) |
| eicu | map_min_6h | 1.134 → 3.046 → 3.028 | +0.0013 (+0.0004 to +0.0022) | +0.0009 (-0.0001 to +0.0020) | -0.0004 (-0.0010 to +0.0003) |
| eicu | map_sd_6h | 1.310 → 0.413 → 0.368 | +0.0009 (-0.0002 to +0.0020) | +0.0001 (-0.0011 to +0.0012) | -0.0008 (-0.0016 to +0.0001) |
| eicu | map_mean_6h | 5.502 → 4.556 → 4.418 | +0.0004 (-0.0000 to +0.0008) | +0.0003 (-0.0008 to +0.0014) | -0.0001 (-0.0012 to +0.0011) |
| eicu | vent_on | 0.004 → 0.252 → 0.034 | +0.0007 (-0.0004 to +0.0018) | +0.0003 (-0.0003 to +0.0008) | -0.0005 (-0.0014 to +0.0005) |

## A. The MAP block along a steady MAP (fold-0 shapes)

Threshold: highest steady MAP below 85 mmHg where the summed log-odds of the three MAP summaries rise ln 2; odds ratios of a steady 70 and 65 mmHg against 85. Median (10–90 %) over the accepted models.

| site | shapes | block threshold | OR 70 vs 85 | OR 65 vs 85 | per-variable `map_last` |
|---|---|---|---|---|---|
| inspire | own | 79.1 (77.9–80.1) | 10.24 (4.37–14.78) | 24.05 (6.97–38.62) | 73.1 (71.2–74.6) |
| inspire | transported | 78.5 (76.7–79.7) | 6.94 (5.73–8.37) | 21.96 (12.88–30.97) | — |
| mimic | own | 77.6 (76.8–79.4) | 6.83 (5.98–8.23) | 17.57 (11.63–23.35) | 67.3 (66.2–68.1) |
| mimic | transported | 79.2 (77.4–79.9) | 9.50 (5.89–11.08) | 24.23 (10.59–29.68) | — |
| eicu | own | 79.2 (78.5–80.0) | 8.28 (7.34–10.55) | 29.07 (14.99–39.15) | 70.9 (68.2–71.5) |
| eicu | transported | 78.1 (76.5–79.0) | 8.10 (5.15–9.29) | 18.04 (9.70–22.10) | — |

Reproduction of S5 (|median per-variable threshold − S5's|, mmHg): inspire map_last 0.00, map_mean_6h 0.00, map_min_6h 0.00; mimic map_last 0.00, map_mean_6h 0.00, map_min_6h 0.00; eicu map_last 0.00, map_mean_6h 0.00, map_min_6h 0.00

eICU hospitals (36): spread of the hospital medians and median 10–90 % width inside a hospital.

| quantity | median | IQR | range | SD between hospitals | width inside |
|---|---|---|---|---|---|
| block threshold | 77.8 | 77.1–79.0 | 74.1–81.9 | 1.6 | 3.5 |
| block OR 70 vs 85 | 7.00 | 6.16–8.53 | 3.78–19.86 | 3.54 | 4.23 |
| `map_last` threshold | 70.6 | 69.3–71.8 | 67.9–76.9 | 2.1 | 3.4 |
| `map_mean_6h` threshold | 70.6 | 69.3–71.6 | 66.2–75.6 | 2.1 | 9.0 |
| `map_min_6h` threshold | 66.4 | 64.1–68.7 | 53.8–74.3 | 4.3 | 10.1 |

## C. Drift over the five folds (external role)

| site | fold | frozen log-loss | all variables re-deformed | missing only | one-variable CIs < 0 / > 0 |
|---|---|---|---|---|---|
| inspire | 0 | 0.3168 | -0.0004 (-0.0024 to +0.0015) | +0.0001 (-0.0004 to +0.0006) | 0 / 2 |
| inspire | 1 | 0.3331 | -0.0010 (-0.0030 to +0.0011) | +0.0000 (+0.0000 to +0.0000) | 1 / 0 |
| inspire | 2 | 0.3332 | -0.0003 (-0.0022 to +0.0016) | -0.0000 (-0.0005 to +0.0006) | 0 / 3 |
| inspire | 3 | 0.3219 | +0.0014 (-0.0004 to +0.0034) | +0.0001 (-0.0003 to +0.0005) | 0 / 1 |
| inspire | 4 | 0.3262 | -0.0011 (-0.0034 to +0.0009) | +0.0000 (-0.0000 to +0.0001) | 3 / 0 |
| inspire | mean (SD) | | -0.0003 (0.0010) | | |
| mimic | 0 | 0.5028 | -0.0004 (-0.0027 to +0.0018) | -0.0003 (-0.0008 to +0.0001) | 0 / 0 |
| mimic | 1 | 0.4995 | -0.0024 (-0.0042 to -0.0007) | -0.0001 (-0.0004 to +0.0003) | 2 / 0 |
| mimic | 2 | 0.5076 | -0.0013 (-0.0036 to +0.0013) | +0.0000 (-0.0000 to +0.0000) | 2 / 0 |
| mimic | 3 | 0.5030 | +0.0020 (-0.0004 to +0.0046) | -0.0004 (-0.0010 to +0.0003) | 0 / 1 |
| mimic | 4 | 0.5149 | -0.0001 (-0.0025 to +0.0021) | +0.0000 (+0.0000 to +0.0000) | 0 / 0 |
| mimic | mean (SD) | | -0.0004 (0.0017) | | |
| eicu | 0 | 0.3929 | +0.0007 (-0.0010 to +0.0021) | +0.0000 (+0.0000 to +0.0000) | 0 / 0 |
| eicu | 1 | 0.3983 | +0.0013 (-0.0012 to +0.0034) | -0.0001 (-0.0010 to +0.0010) | 0 / 1 |
| eicu | 2 | 0.4037 | +0.0015 (-0.0007 to +0.0037) | +0.0000 (+0.0000 to +0.0000) | 0 / 0 |
| eicu | 3 | 0.3924 | +0.0020 (-0.0004 to +0.0043) | +0.0002 (-0.0006 to +0.0012) | 0 / 2 |
| eicu | 4 | 0.3923 | +0.0017 (-0.0006 to +0.0042) | +0.0000 (-0.0000 to +0.0001) | 0 / 0 |
| eicu | mean (SD) | | +0.0015 (0.0005) | | |

Variables whose one-variable re-deformation has a CI below 0 in two or more folds: inspire: none; mimic: none; eicu: none

