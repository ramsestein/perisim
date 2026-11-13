# Arc vs value placement: paired contrast (docs/prereg_arc.md §6)

Patient bootstrap, 2000 resamples. Q: 90 % CI, equivalent inside [0.8, 1.25]. D, ΔR: 95 % CI. Aggregates only.

| unit | n (patients) | L value | L arc | D = L arc − L value | Q (90 % CI) | reading Q | reading D |
|---|---|---|---|---|---|---|---|
| hypo_eicu | 466,384 (32,032) | 0.3966 | 0.3960 | -0.0006 (-0.0010 to -0.0002) | 1.045 (1.022 to 1.070) | equivalent | arc better |
| hypo_inspire | 148,293 (11,631) | 0.3274 | 0.3276 | +0.0002 (-0.0005 to +0.0010) | 0.979 (0.922 to 1.041) | equivalent | no difference |
| hypo_mimic | 387,158 (24,464) | 0.5048 | 0.5040 | -0.0008 (-0.0014 to -0.0003) | 1.091 (1.040 to 1.147) | equivalent | arc better |
| nwicu_eicu | 30,833 (2,001) | 0.3183 | 0.3188 | +0.0005 (-0.0009 to +0.0018) | 0.956 (0.857 to 1.070) | equivalent | no difference |
| nwicu_inspire | 30,833 (2,001) | 0.3192 | 0.3196 | +0.0004 (-0.0010 to +0.0018) | 0.962 (0.843 to 1.089) | equivalent | no difference |
| nwicu_mimic | 30,833 (2,001) | 0.3191 | 0.3177 | -0.0014 (-0.0030 to +0.0003) | 1.138 (1.002 to 1.314) | not equivalent (arc better) | no difference |
| t2_eicu | 4,725 (4,593) | 1.0444 | 1.0474 | +0.0030 (-0.0043 to +0.0108) | 0.931 (0.792 to 1.077) | not equivalent (arc worse) | no difference |
| t2_mimic | 4,963 (4,632) | 1.0492 | 1.0467 | -0.0025 (-0.0110 to +0.0056) | 1.118 (0.808 to 1.663) | not equivalent (arc better) | no difference |

## Transport within each arm (secondary)

| unit | R value (90 % CI) | R arc (90 % CI) | ΔR = R arc − R value (95 % CI) | readings |
|---|---|---|---|---|
| hypo_eicu | 0.964 (0.930 to 0.998) | 0.970 (0.938 to 1.000) | +0.006 (-0.035 to +0.043) | value equivalent; arc equivalent; ΔR no difference |
| hypo_inspire | 1.031 (0.945 to 1.122) | 0.908 (0.837 to 0.987) | -0.122 (-0.236 to -0.017) | value equivalent; arc equivalent; ΔR lower with arc |
| hypo_mimic | 0.887 (0.833 to 0.942) | 0.898 (0.851 to 0.942) | +0.011 (-0.056 to +0.074) | value equivalent; arc equivalent; ΔR no difference |
| nwicu_eicu | 1.526 (1.164 to 2.177) | 1.528 (1.138 to 2.123) | +0.002 (-0.724 to +0.605) | value equivalent; arc equivalent; ΔR no difference |
| nwicu_inspire | 1.405 (1.062 to 1.963) | 1.416 (1.037 to 2.050) | +0.010 (-0.632 to +0.638) | value equivalent; arc equivalent; ΔR no difference |
| nwicu_mimic | 1.420 (1.075 to 2.007) | 1.691 (1.311 to 2.355) | +0.271 (-0.372 to +0.988) | value equivalent; arc equivalent; ΔR no difference |
| t2_eicu | 0.698 (0.571 to 0.817) | 0.655 (0.527 to 0.772) | -0.043 (-0.178 to +0.084) | value not equivalent; arc not equivalent; ΔR no difference |
| t2_mimic | 0.484 (0.292 to 0.665) | 0.497 (0.311 to 0.665) | +0.013 (-0.178 to +0.222) | value not equivalent; arc not equivalent; ΔR no difference |

## Locally learned metrics (secondary)

| unit | D local (95 % CI) | Q local (90 % CI) |
|---|---|---|
| hypo_eicu | -0.0005 (-0.0010 to -0.0001) | 1.039 (1.013 to 1.068) |
| hypo_inspire | -0.0012 (-0.0021 to -0.0002) | 1.111 (1.033 to 1.195) |
| hypo_mimic | -0.0008 (-0.0013 to -0.0003) | 1.078 (1.038 to 1.121) |
| nwicu_eicu | +0.0003 (-0.0020 to +0.0027) | 0.955 (0.718 to 1.308) |
| nwicu_inspire | +0.0003 (-0.0019 to +0.0027) | 0.955 (0.705 to 1.286) |
| nwicu_mimic | +0.0003 (-0.0021 to +0.0027) | 0.955 (0.710 to 1.297) |
| t2_eicu | +0.0005 (-0.0052 to +0.0062) | 0.992 (0.916 to 1.070) |
| t2_mimic | -0.0038 (-0.0100 to +0.0024) | 1.088 (0.972 to 1.241) |

## Shapes of `hypo`'s frozen metric, rotation external-eicu (descriptive)

| variable | backtracking 1 − net/arc | weight value | weight arc |
|---|---|---|---|
| temp_last | 0.95 | 0.000 | 0.003 |
| hr_last | 0.81 | 0.003 | 0.004 |
| creatinine_last | 0.76 | 0.005 | 0.001 |
| hr_slope_6h | 0.70 | 0.003 | 0.000 |
| rr_last | 0.62 | 0.000 | 0.000 |
| spo2_min_6h | 0.54 | 0.004 | 0.004 |
| spo2_last | 0.33 | 0.024 | 0.009 |
| map_min_6h | 0.26 | 0.083 | 0.080 |
| L | 0.17 | 0.090 | 0.098 |
| map_sd_6h | 0.15 | 0.125 | 0.125 |
| map_slope_6h | 0.09 | 0.003 | 0.004 |
| hb_last | 0.07 | 0.015 | 0.013 |
| lactate_last | 0.04 | 0.000 | 0.002 |
| map_last | 0.00 | 0.240 | 0.167 |
| map_mean_6h | 0.00 | 0.222 | 0.322 |
| vaso_hours | 0.00 | 0.001 | 0.000 |
| age | -0.00 | 0.052 | 0.048 |

## Shapes of `hypo`'s frozen metric, rotation external-inspire (descriptive)

| variable | backtracking 1 − net/arc | weight value | weight arc |
|---|---|---|---|
| spo2_last | 0.89 | 0.007 | 0.005 |
| spo2_min_6h | 0.85 | 0.007 | 0.000 |
| rr_last | 0.69 | 0.001 | 0.008 |
| hr_last | 0.57 | 0.009 | 0.002 |
| map_min_6h | 0.50 | 0.128 | 0.144 |
| temp_last | 0.41 | 0.001 | 0.007 |
| lactate_last | 0.33 | 0.000 | 0.000 |
| creatinine_last | 0.19 | 0.000 | 0.000 |
| age | 0.14 | 0.057 | 0.055 |
| hb_last | 0.14 | 0.008 | 0.012 |
| L | 0.01 | 0.060 | 0.112 |
| hr_slope_6h | 0.01 | 0.000 | 0.003 |
| map_mean_6h | 0.00 | 0.206 | 0.199 |
| map_sd_6h | 0.00 | 0.089 | 0.118 |
| vaso_hours | 0.00 | 0.009 | 0.015 |
| map_last | -0.00 | 0.315 | 0.204 |
| map_slope_6h | -0.00 | 0.004 | 0.028 |

## Shapes of `hypo`'s frozen metric, rotation external-mimic (descriptive)

| variable | backtracking 1 − net/arc | weight value | weight arc |
|---|---|---|---|
| temp_last | 0.97 | 0.002 | 0.006 |
| spo2_min_6h | 0.78 | 0.021 | 0.000 |
| hr_last | 0.63 | 0.007 | 0.000 |
| lactate_last | 0.57 | 0.003 | 0.000 |
| hb_last | 0.52 | 0.009 | 0.016 |
| creatinine_last | 0.50 | 0.012 | 0.012 |
| map_min_6h | 0.30 | 0.127 | 0.124 |
| hr_slope_6h | 0.23 | 0.001 | 0.006 |
| rr_last | 0.23 | 0.002 | 0.003 |
| map_sd_6h | 0.06 | 0.046 | 0.062 |
| map_slope_6h | 0.04 | 0.002 | 0.005 |
| L | 0.02 | 0.250 | 0.216 |
| age | 0.01 | 0.030 | 0.031 |
| map_last | 0.00 | 0.147 | 0.145 |
| map_mean_6h | 0.00 | 0.232 | 0.300 |
| spo2_last | 0.00 | 0.000 | 0.000 |
| vaso_hours | -0.00 | 0.000 | 0.001 |

## Euclidean check (DECISIONS 2026-11-13)

The Euclidean kNN does not depend on the arm. Units where it differed (all statistics use the value arm's):

- t2_eicu: 1 row(s), max 0.0311 nats, effect on the mean 6.6e-06 nats
- t2_mimic: 1 row(s), max 0.0343 nats, effect on the mean 6.9e-06 nats
