# `hypo` Phase 2: internal validation by rotation

Out-of-fold predictions of each development site; PERISIM is shown once per partner site (the metric learned on both sites' training patients, each site queried with its own case base); comparators are local. 95 % CIs by patient bootstrap. Code: 5c5b2fe1f006.

## inspire: 148,293 landmarks, 11,631 patients, prevalence 13.2% (folds [0, 1, 2, 3, 4])

| method | AUROC (95 % CI) | log-loss (95 % CI) | Brier | CITL | slope | mean p |
|---|---|---|---|---|---|---|
| PERISIM (with eicu) | 0.780 (0.774–0.786) | 0.3267 (0.3204–0.3328) | 0.0970 | 0.031 | 1.033 | 0.129 |
| PERISIM (with mimic) | 0.780 (0.774–0.785) | 0.3272 (0.3211–0.3332) | 0.0972 | 0.025 | 1.042 | 0.129 |
| prevalence | 0.494 (0.485–0.504) | 0.3894 (0.3826–0.3957) | 0.1143 | -0.000 | — | 0.132 |
| last MAP (logistic) | 0.743 (0.737–0.748) | 0.3489 (0.3427–0.3550) | 0.1025 | 0.000 | 0.999 | 0.132 |
| logistic, all inputs | 0.781 (0.775–0.787) | 0.3290 (0.3229–0.3352) | 0.0981 | 0.000 | 1.002 | 0.132 |
| LightGBM | 0.802 (0.797–0.807) | 0.3157 (0.3096–0.3216) | 0.0945 | 0.002 | 1.018 | 0.131 |
| Euclidean kNN | 0.765 (0.760–0.770) | 0.3382 (0.3322–0.3442) | 0.1014 | 0.066 | 1.185 | 0.125 |

| paired | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|
| PERISIM (with eicu) − prevalence | -0.0627 (-0.0659 to -0.0598) | 0.286 (0.276 to 0.297) |
| PERISIM (with eicu) − last MAP (logistic) | -0.0222 (-0.0241 to -0.0205) | 0.038 (0.034 to 0.042) |
| PERISIM (with eicu) − logistic, all inputs | -0.0023 (-0.0037 to -0.0011) | -0.001 (-0.003 to 0.002) |
| PERISIM (with eicu) − LightGBM | 0.0110 (0.0100 to 0.0118) | -0.022 (-0.023 to -0.020) |
| PERISIM (with eicu) − Euclidean kNN | -0.0115 (-0.0129 to -0.0101) | 0.015 (0.013 to 0.018) |
| PERISIM (with mimic) − prevalence | -0.0622 (-0.0653 to -0.0597) | 0.286 (0.275 to 0.295) |
| PERISIM (with mimic) − last MAP (logistic) | -0.0217 (-0.0236 to -0.0200) | 0.037 (0.033 to 0.041) |
| PERISIM (with mimic) − logistic, all inputs | -0.0018 (-0.0032 to -0.0006) | -0.002 (-0.004 to 0.001) |
| PERISIM (with mimic) − LightGBM | 0.0114 (0.0105 to 0.0124) | -0.023 (-0.025 to -0.021) |
| PERISIM (with mimic) − Euclidean kNN | -0.0110 (-0.0124 to -0.0097) | 0.015 (0.012 to 0.017) |

## mimic: 387,158 landmarks, 24,464 patients, prevalence 30.7% (folds [0, 1, 2, 3, 4])

| method | AUROC (95 % CI) | log-loss (95 % CI) | Brier | CITL | slope | mean p |
|---|---|---|---|---|---|---|
| PERISIM (with eicu) | 0.784 (0.781–0.787) | 0.5024 (0.4990–0.5062) | 0.1670 | 0.003 | 1.018 | 0.306 |
| PERISIM (with inspire) | 0.783 (0.780–0.786) | 0.5034 (0.5001–0.5070) | 0.1673 | 0.015 | 1.026 | 0.304 |
| prevalence | 0.497 (0.493–0.502) | 0.6164 (0.6136–0.6193) | 0.2126 | -0.000 | — | 0.307 |
| last MAP (logistic) | 0.736 (0.734–0.739) | 0.5476 (0.5443–0.5511) | 0.1825 | -0.000 | 1.000 | 0.307 |
| logistic, all inputs | 0.787 (0.784–0.790) | 0.5014 (0.4980–0.5051) | 0.1661 | -0.000 | 0.999 | 0.307 |
| LightGBM | 0.797 (0.793–0.799) | 0.4905 (0.4872–0.4943) | 0.1628 | 0.002 | 1.004 | 0.306 |
| Euclidean kNN | 0.773 (0.770–0.776) | 0.5140 (0.5108–0.5176) | 0.1715 | -0.022 | 1.111 | 0.311 |

| paired | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|
| PERISIM (with eicu) − prevalence | -0.1140 (-0.1166 to -0.1115) | 0.287 (0.282 to 0.292) |
| PERISIM (with eicu) − last MAP (logistic) | -0.0453 (-0.0468 to -0.0438) | 0.048 (0.046 to 0.049) |
| PERISIM (with eicu) − logistic, all inputs | 0.0010 (0.0001 to 0.0018) | -0.003 (-0.004 to -0.002) |
| PERISIM (with eicu) − LightGBM | 0.0119 (0.0112 to 0.0125) | -0.012 (-0.013 to -0.012) |
| PERISIM (with eicu) − Euclidean kNN | -0.0116 (-0.0125 to -0.0108) | 0.011 (0.010 to 0.012) |
| PERISIM (with inspire) − prevalence | -0.1130 (-0.1154 to -0.1104) | 0.286 (0.281 to 0.291) |
| PERISIM (with inspire) − last MAP (logistic) | -0.0442 (-0.0458 to -0.0428) | 0.047 (0.045 to 0.048) |
| PERISIM (with inspire) − logistic, all inputs | 0.0020 (0.0012 to 0.0028) | -0.004 (-0.005 to -0.003) |
| PERISIM (with inspire) − LightGBM | 0.0129 (0.0122 to 0.0136) | -0.013 (-0.014 to -0.013) |
| PERISIM (with inspire) − Euclidean kNN | -0.0106 (-0.0115 to -0.0098) | 0.010 (0.009 to 0.011) |

## eicu: 466,384 landmarks, 32,032 patients, prevalence 19.3% (folds [0, 1, 2, 3, 4])

| method | AUROC (95 % CI) | log-loss (95 % CI) | Brier | CITL | slope | mean p |
|---|---|---|---|---|---|---|
| PERISIM (with mimic) | 0.800 (0.798–0.803) | 0.3956 (0.3925–0.3992) | 0.1247 | 0.019 | 1.047 | 0.191 |
| PERISIM (with inspire) | 0.800 (0.798–0.803) | 0.3956 (0.3926–0.3993) | 0.1247 | 0.019 | 1.046 | 0.191 |
| prevalence | 0.495 (0.490–0.500) | 0.4911 (0.4877–0.4949) | 0.1560 | -0.000 | — | 0.193 |
| last MAP (logistic) | 0.774 (0.771–0.777) | 0.4215 (0.4178–0.4252) | 0.1313 | -0.000 | 1.003 | 0.193 |
| logistic, all inputs | 0.804 (0.802–0.807) | 0.3960 (0.3928–0.3997) | 0.1246 | 0.000 | 1.000 | 0.193 |
| LightGBM | 0.814 (0.812–0.817) | 0.3850 (0.3819–0.3886) | 0.1217 | 0.003 | 1.005 | 0.193 |
| Euclidean kNN | 0.784 (0.782–0.787) | 0.4101 (0.4067–0.4136) | 0.1302 | 0.002 | 1.156 | 0.193 |

| paired | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|
| PERISIM (with mimic) − prevalence | -0.0955 (-0.0975 to -0.0937) | 0.305 (0.300 to 0.311) |
| PERISIM (with mimic) − last MAP (logistic) | -0.0259 (-0.0272 to -0.0245) | 0.026 (0.025 to 0.028) |
| PERISIM (with mimic) − logistic, all inputs | -0.0004 (-0.0013 to 0.0005) | -0.004 (-0.005 to -0.003) |
| PERISIM (with mimic) − LightGBM | 0.0106 (0.0101 to 0.0111) | -0.014 (-0.015 to -0.013) |
| PERISIM (with mimic) − Euclidean kNN | -0.0145 (-0.0152 to -0.0136) | 0.016 (0.015 to 0.017) |
| PERISIM (with inspire) − prevalence | -0.0955 (-0.0974 to -0.0937) | 0.305 (0.300 to 0.311) |
| PERISIM (with inspire) − last MAP (logistic) | -0.0258 (-0.0272 to -0.0245) | 0.026 (0.025 to 0.028) |
| PERISIM (with inspire) − logistic, all inputs | -0.0004 (-0.0013 to 0.0005) | -0.004 (-0.005 to -0.003) |
| PERISIM (with inspire) − LightGBM | 0.0106 (0.0100 to 0.0112) | -0.014 (-0.015 to -0.013) |
| PERISIM (with inspire) − Euclidean kNN | -0.0144 (-0.0154 to -0.0136) | 0.016 (0.015 to 0.017) |

## Rotation external-inspire: metric learned on mimic + eicu (5 folds, 38 min)

Accepted models per fold [40, 40, 40, 40, 40]; α per site {'mimic': [1.0, 1.0, 1.0, 1.0, 1.0], 'eicu': [2.0, 2.0, 2.0, 2.0, 2.0]}; left out of the metric (constant within each site): none.

| variable | weight, mean (min–max over folds) | shaped in folds |
|---|---|---|
| map_mean_6h | 0.273 (0.236–0.334) | 5/5 |
| map_last | 0.178 (0.133–0.272) | 5/5 |
| L | 0.139 (0.089–0.182) | 5/5 |
| map_min_6h | 0.105 (0.068–0.151) | 5/5 |
| map_sd_6h | 0.089 (0.029–0.173) | 5/5 |
| age | 0.056 (0.035–0.087) | 5/5 |
| sex_male | 0.022 (0.017–0.025) | 0/5 |
| surgical_group | 0.018 (0.009–0.023) | 5/5 |
| map_slope_6h | 0.015 (0.000–0.023) | 5/5 |
| hb_last | 0.014 (0.009–0.024) | 5/5 |
| vaso_on | 0.014 (0.007–0.022) | 0/5 |
| spo2_last | 0.013 (0.009–0.024) | 4/5 |
| vaso_hours | 0.012 (0.002–0.018) | 5/5 |
| lactate_last | 0.006 (0.000–0.022) | 5/5 |
| vent_on | 0.006 (0.000–0.015) | 0/5 |
| rr_last | 0.005 (0.001–0.012) | 5/5 |
| emergency | 0.004 (0.000–0.012) | 0/5 |
| hr_slope_6h | 0.004 (0.000–0.010) | 5/5 |
| hr_last | 0.004 (0.000–0.013) | 5/5 |
| sedation_24h | 0.004 (0.000–0.011) | 0/5 |
| lactate_last__missing | 0.004 (0.000–0.011) | 0/5 |
| sedation_on | 0.004 (0.000–0.009) | 0/5 |
| vaso_24h | 0.003 (0.000–0.007) | 0/5 |
| temp_last__missing | 0.002 (0.000–0.005) | 0/5 |
| creatinine_last__missing | 0.002 (0.000–0.005) | 0/5 |
| hb_last__missing | 0.001 (0.000–0.005) | 0/5 |
| spo2_min_6h | 0.001 (0.000–0.003) | 5/5 |
| creatinine_last | 0.001 (0.000–0.002) | 5/5 |
| temp_last | 0.000 (0.000–0.000) | 5/5 |

## Rotation external-mimic: metric learned on inspire + eicu (5 folds, 35 min)

Accepted models per fold [40, 40, 40, 40, 40]; α per site {'inspire': [2.0, 2.0, 2.0, 2.0, 5.0], 'eicu': [2.0, 2.0, 2.0, 2.0, 2.0]}; left out of the metric (constant within each site): sedation_on__missing, vaso_hours__missing, vaso_on__missing.

| variable | weight, mean (min–max over folds) | shaped in folds |
|---|---|---|
| map_mean_6h | 0.346 (0.264–0.478) | 5/5 |
| map_last | 0.160 (0.114–0.204) | 5/5 |
| map_min_6h | 0.160 (0.042–0.265) | 5/5 |
| map_sd_6h | 0.146 (0.052–0.425) | 5/5 |
| L | 0.061 (0.047–0.068) | 5/5 |
| age | 0.028 (0.014–0.046) | 5/5 |
| surgical_group | 0.016 (0.004–0.023) | 5/5 |
| sex_male | 0.011 (0.007–0.015) | 0/5 |
| vent_on | 0.008 (0.000–0.025) | 0/5 |
| hb_last | 0.007 (0.004–0.016) | 5/5 |
| map_slope_6h | 0.007 (0.000–0.023) | 5/5 |
| creatinine_last | 0.007 (0.000–0.019) | 5/5 |
| rr_last | 0.007 (0.005–0.008) | 5/5 |
| temp_last | 0.006 (0.000–0.013) | 5/5 |
| lactate_last__missing | 0.005 (0.001–0.010) | 0/5 |
| vaso_24h | 0.005 (0.000–0.019) | 0/5 |
| spo2_last | 0.004 (0.000–0.015) | 4/5 |
| temp_last__missing | 0.004 (0.001–0.007) | 0/5 |
| vaso_hours | 0.003 (0.000–0.010) | 5/5 |
| hr_last | 0.003 (0.000–0.015) | 5/5 |
| sedation_24h | 0.001 (0.000–0.004) | 0/5 |
| sedation_on | 0.001 (0.000–0.003) | 0/5 |
| hr_slope_6h | 0.001 (0.000–0.004) | 5/5 |
| lactate_last | 0.001 (0.000–0.003) | 5/5 |
| vaso_on | 0.001 (0.000–0.003) | 0/5 |
| creatinine_last__missing | 0.001 (0.000–0.002) | 0/5 |
| emergency | 0.000 (0.000–0.001) | 0/5 |
| hb_last__missing | 0.000 (0.000–0.001) | 0/5 |
| spo2_min_6h | 0.000 (0.000–0.000) | 5/5 |

## Rotation external-eicu: metric learned on inspire + mimic (5 folds, 30 min)

Accepted models per fold [40, 40, 40, 40, 40]; α per site {'inspire': [2.0, 2.0, 2.0, 2.0, 5.0], 'mimic': [1.0, 2.0, 1.0, 1.0, 1.0]}; left out of the metric (constant within each site): sedation_on__missing, vaso_hours__missing, vaso_on__missing.

| variable | weight, mean (min–max over folds) | shaped in folds |
|---|---|---|
| map_mean_6h | 0.358 (0.299–0.431) | 5/5 |
| map_last | 0.152 (0.093–0.217) | 5/5 |
| map_min_6h | 0.107 (0.049–0.241) | 5/5 |
| map_sd_6h | 0.104 (0.044–0.159) | 5/5 |
| L | 0.095 (0.051–0.152) | 5/5 |
| age | 0.041 (0.021–0.055) | 5/5 |
| vaso_24h | 0.013 (0.002–0.035) | 0/5 |
| surgical_group | 0.013 (0.004–0.021) | 5/5 |
| map_slope_6h | 0.013 (0.000–0.048) | 5/5 |
| hb_last | 0.013 (0.002–0.021) | 5/5 |
| sex_male | 0.012 (0.005–0.018) | 0/5 |
| vent_on | 0.010 (0.000–0.025) | 0/5 |
| rr_last | 0.009 (0.000–0.015) | 5/5 |
| vaso_on | 0.008 (0.004–0.012) | 0/5 |
| creatinine_last | 0.008 (0.000–0.020) | 5/5 |
| vaso_hours | 0.008 (0.001–0.017) | 5/5 |
| spo2_last | 0.006 (0.001–0.010) | 5/5 |
| emergency | 0.005 (0.000–0.010) | 0/5 |
| lactate_last__missing | 0.005 (0.001–0.008) | 0/5 |
| hr_last | 0.004 (0.000–0.018) | 5/5 |
| spo2_min_6h | 0.004 (0.000–0.010) | 5/5 |
| sedation_24h | 0.003 (0.000–0.007) | 0/5 |
| hr_slope_6h | 0.003 (0.000–0.006) | 5/5 |
| temp_last | 0.003 (0.000–0.006) | 5/5 |
| lactate_last | 0.002 (0.000–0.005) | 5/5 |
| temp_last__missing | 0.001 (0.000–0.002) | 0/5 |
| sedation_on | 0.000 (0.000–0.001) | 0/5 |

