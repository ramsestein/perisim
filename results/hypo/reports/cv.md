# `hypo` Phase 2: internal validation by rotation

Out-of-fold predictions of each development site; PERISIM is shown once per partner site (the metric learned on both sites' training patients, each site queried with its own case base); comparators are local. 95 % CIs by patient bootstrap. Code: 11a7ad77e519.

## inspire: 148,293 landmarks, 11,631 patients, prevalence 13.2% (folds [0, 1, 2, 3, 4])

| method | AUROC (95 % CI) | log-loss (95 % CI) | Brier | CITL | slope | mean p |
|---|---|---|---|---|---|---|
| PERISIM (with eicu) | 0.778 (0.772–0.783) | 0.3281 (0.3219–0.3343) | 0.0974 | 0.025 | 1.019 | 0.129 |
| PERISIM (with mimic) | 0.779 (0.773–0.784) | 0.3277 (0.3216–0.3337) | 0.0972 | 0.021 | 1.006 | 0.130 |
| prevalence | 0.493 (0.483–0.502) | 0.3894 (0.3826–0.3957) | 0.1143 | -0.000 | — | 0.132 |
| last MAP (logistic) | 0.743 (0.737–0.749) | 0.3489 (0.3427–0.3550) | 0.1025 | 0.000 | 1.000 | 0.132 |
| logistic, all inputs | 0.781 (0.776–0.787) | 0.3288 (0.3227–0.3350) | 0.0980 | -0.000 | 0.999 | 0.132 |
| LightGBM | 0.802 (0.796–0.807) | 0.3161 (0.3100–0.3220) | 0.0946 | 0.008 | 1.017 | 0.131 |
| Euclidean kNN | 0.765 (0.759–0.770) | 0.3381 (0.3319–0.3442) | 0.1014 | 0.062 | 1.187 | 0.125 |

| paired | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|
| PERISIM (with eicu) − prevalence | -0.0613 (-0.0644 to -0.0586) | 0.284 (0.274 to 0.296) |
| PERISIM (with eicu) − last MAP (logistic) | -0.0208 (-0.0226 to -0.0191) | 0.035 (0.032 to 0.039) |
| PERISIM (with eicu) − logistic, all inputs | -0.0007 (-0.0023 to 0.0005) | -0.004 (-0.006 to -0.001) |
| PERISIM (with eicu) − LightGBM | 0.0120 (0.0109 to 0.0130) | -0.024 (-0.026 to -0.022) |
| PERISIM (with eicu) − Euclidean kNN | -0.0100 (-0.0115 to -0.0086) | 0.013 (0.010 to 0.016) |
| PERISIM (with mimic) − prevalence | -0.0617 (-0.0648 to -0.0589) | 0.285 (0.276 to 0.296) |
| PERISIM (with mimic) − last MAP (logistic) | -0.0212 (-0.0231 to -0.0194) | 0.036 (0.032 to 0.040) |
| PERISIM (with mimic) − logistic, all inputs | -0.0011 (-0.0026 to 0.0002) | -0.003 (-0.005 to 0.000) |
| PERISIM (with mimic) − LightGBM | 0.0116 (0.0107 to 0.0126) | -0.023 (-0.025 to -0.021) |
| PERISIM (with mimic) − Euclidean kNN | -0.0104 (-0.0120 to -0.0090) | 0.014 (0.011 to 0.017) |

## mimic: 387,159 landmarks, 24,464 patients, prevalence 30.7% (folds [0, 1, 2, 3, 4])

| method | AUROC (95 % CI) | log-loss (95 % CI) | Brier | CITL | slope | mean p |
|---|---|---|---|---|---|---|
| PERISIM (with eicu) | 0.783 (0.780–0.786) | 0.5034 (0.5002–0.5071) | 0.1674 | -0.004 | 1.017 | 0.307 |
| PERISIM (with inspire) | 0.783 (0.780–0.785) | 0.5041 (0.5009–0.5079) | 0.1676 | 0.003 | 1.002 | 0.306 |
| prevalence | 0.494 (0.489–0.498) | 0.6164 (0.6136–0.6194) | 0.2126 | 0.000 | — | 0.307 |
| last MAP (logistic) | 0.736 (0.733–0.739) | 0.5477 (0.5443–0.5511) | 0.1825 | -0.000 | 1.000 | 0.307 |
| logistic, all inputs | 0.787 (0.784–0.790) | 0.5015 (0.4980–0.5051) | 0.1662 | -0.000 | 0.998 | 0.307 |
| LightGBM | 0.797 (0.794–0.799) | 0.4904 (0.4872–0.4942) | 0.1628 | 0.002 | 1.006 | 0.306 |
| Euclidean kNN | 0.773 (0.770–0.776) | 0.5138 (0.5105–0.5175) | 0.1714 | -0.022 | 1.112 | 0.311 |

| paired | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|
| PERISIM (with eicu) − prevalence | -0.1130 (-0.1156 to -0.1105) | 0.289 (0.284 to 0.295) |
| PERISIM (with eicu) − last MAP (logistic) | -0.0442 (-0.0458 to -0.0428) | 0.047 (0.045 to 0.048) |
| PERISIM (with eicu) − logistic, all inputs | 0.0020 (0.0011 to 0.0029) | -0.004 (-0.005 to -0.003) |
| PERISIM (with eicu) − LightGBM | 0.0130 (0.0124 to 0.0136) | -0.014 (-0.014 to -0.013) |
| PERISIM (with eicu) − Euclidean kNN | -0.0103 (-0.0112 to -0.0094) | 0.010 (0.009 to 0.011) |
| PERISIM (with inspire) − prevalence | -0.1123 (-0.1146 to -0.1097) | 0.289 (0.283 to 0.295) |
| PERISIM (with inspire) − last MAP (logistic) | -0.0435 (-0.0450 to -0.0419) | 0.046 (0.045 to 0.048) |
| PERISIM (with inspire) − logistic, all inputs | 0.0027 (0.0018 to 0.0035) | -0.005 (-0.006 to -0.004) |
| PERISIM (with inspire) − LightGBM | 0.0137 (0.0129 to 0.0144) | -0.014 (-0.015 to -0.013) |
| PERISIM (with inspire) − Euclidean kNN | -0.0096 (-0.0104 to -0.0087) | 0.009 (0.008 to 0.010) |

## eicu: 466,385 landmarks, 32,032 patients, prevalence 19.3% (folds [0, 1, 2, 3, 4])

| method | AUROC (95 % CI) | log-loss (95 % CI) | Brier | CITL | slope | mean p |
|---|---|---|---|---|---|---|
| PERISIM (with mimic) | 0.799 (0.797–0.802) | 0.3964 (0.3933–0.4001) | 0.1249 | 0.011 | 1.037 | 0.192 |
| PERISIM (with inspire) | 0.799 (0.796–0.801) | 0.3968 (0.3937–0.4005) | 0.1251 | 0.010 | 1.032 | 0.192 |
| prevalence | 0.497 (0.492–0.502) | 0.4912 (0.4877–0.4949) | 0.1560 | -0.000 | — | 0.193 |
| last MAP (logistic) | 0.774 (0.771–0.777) | 0.4215 (0.4179–0.4253) | 0.1313 | -0.000 | 1.003 | 0.193 |
| logistic, all inputs | 0.804 (0.802–0.807) | 0.3961 (0.3929–0.3998) | 0.1246 | 0.000 | 0.999 | 0.193 |
| LightGBM | 0.814 (0.812–0.816) | 0.3851 (0.3820–0.3887) | 0.1217 | 0.002 | 1.004 | 0.193 |
| Euclidean kNN | 0.784 (0.782–0.787) | 0.4100 (0.4067–0.4135) | 0.1302 | -0.003 | 1.171 | 0.194 |

| paired | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|
| PERISIM (with mimic) − prevalence | -0.0948 (-0.0967 to -0.0929) | 0.302 (0.297 to 0.308) |
| PERISIM (with mimic) − last MAP (logistic) | -0.0251 (-0.0264 to -0.0239) | 0.025 (0.024 to 0.027) |
| PERISIM (with mimic) − logistic, all inputs | 0.0003 (-0.0006 to 0.0012) | -0.005 (-0.006 to -0.004) |
| PERISIM (with mimic) − LightGBM | 0.0113 (0.0108 to 0.0118) | -0.015 (-0.015 to -0.014) |
| PERISIM (with mimic) − Euclidean kNN | -0.0136 (-0.0144 to -0.0128) | 0.015 (0.014 to 0.016) |
| PERISIM (with inspire) − prevalence | -0.0944 (-0.0963 to -0.0926) | 0.302 (0.297 to 0.307) |
| PERISIM (with inspire) − last MAP (logistic) | -0.0247 (-0.0260 to -0.0234) | 0.025 (0.023 to 0.026) |
| PERISIM (with inspire) − logistic, all inputs | 0.0007 (-0.0002 to 0.0015) | -0.005 (-0.006 to -0.004) |
| PERISIM (with inspire) − LightGBM | 0.0117 (0.0112 to 0.0123) | -0.015 (-0.016 to -0.014) |
| PERISIM (with inspire) − Euclidean kNN | -0.0132 (-0.0141 to -0.0124) | 0.015 (0.013 to 0.016) |

## Rotation external-inspire: metric learned on mimic + eicu (5 folds, 36 min)

Accepted models per fold [40, 40, 40, 40, 40]; α per site {'mimic': [1.0, 1.0, 1.0, 1.0, 1.0], 'eicu': [2.0, 2.0, 2.0, 1.0, 2.0]}; left out of the metric (constant within each site): none.

| variable | weight, mean (min–max over folds) | shaped in folds |
|---|---|---|
| map_mean_6h | 0.299 (0.179–0.359) | 5/5 |
| map_last | 0.146 (0.067–0.185) | 5/5 |
| map_min_6h | 0.123 (0.072–0.182) | 5/5 |
| L | 0.108 (0.076–0.181) | 5/5 |
| map_sd_6h | 0.067 (0.008–0.109) | 5/5 |
| age | 0.060 (0.042–0.094) | 5/5 |
| surgical_group | 0.034 (0.013–0.074) | 5/5 |
| sex_male | 0.027 (0.018–0.040) | 0/5 |
| hb_last | 0.013 (0.008–0.018) | 5/5 |
| vaso_on | 0.013 (0.009–0.016) | 0/5 |
| lactate_last | 0.012 (0.000–0.035) | 5/5 |
| rr_last | 0.012 (0.000–0.031) | 5/5 |
| lactate_last__missing | 0.009 (0.000–0.026) | 0/5 |
| spo2_min_6h | 0.009 (0.002–0.020) | 5/5 |
| creatinine_last | 0.009 (0.000–0.018) | 5/5 |
| vaso_24h | 0.008 (0.001–0.018) | 0/5 |
| hr_slope_6h | 0.008 (0.000–0.018) | 5/5 |
| spo2_last | 0.007 (0.000–0.015) | 5/5 |
| emergency | 0.006 (0.000–0.013) | 0/5 |
| vent_on | 0.005 (0.000–0.015) | 0/5 |
| vaso_hours | 0.005 (0.002–0.012) | 5/5 |
| hr_last | 0.004 (0.000–0.014) | 5/5 |
| sedation_24h | 0.004 (0.000–0.013) | 0/5 |
| map_slope_6h | 0.003 (0.000–0.016) | 5/5 |
| hb_last__missing | 0.003 (0.000–0.013) | 0/5 |
| creatinine_last__missing | 0.002 (0.000–0.008) | 0/5 |
| temp_last | 0.001 (0.000–0.005) | 5/5 |
| temp_last__missing | 0.001 (0.000–0.004) | 0/5 |
| sedation_on | 0.000 (0.000–0.001) | 0/5 |

## Rotation external-mimic: metric learned on inspire + eicu (5 folds, 34 min)

Accepted models per fold [40, 40, 40, 40, 40]; α per site {'inspire': [2.0, 2.0, 5.0, 2.0, 2.0], 'eicu': [2.0, 2.0, 2.0, 1.0, 2.0]}; left out of the metric (constant within each site): sedation_on__missing, vaso_hours__missing, vaso_on__missing.

| variable | weight, mean (min–max over folds) | shaped in folds |
|---|---|---|
| map_last | 0.214 (0.160–0.294) | 5/5 |
| map_min_6h | 0.198 (0.116–0.286) | 5/5 |
| L | 0.192 (0.116–0.288) | 5/5 |
| map_mean_6h | 0.159 (0.091–0.201) | 5/5 |
| map_sd_6h | 0.050 (0.013–0.090) | 5/5 |
| surgical_group | 0.038 (0.006–0.108) | 5/5 |
| age | 0.034 (0.024–0.046) | 5/5 |
| sex_male | 0.015 (0.010–0.024) | 0/5 |
| hb_last | 0.015 (0.010–0.022) | 5/5 |
| lactate_last__missing | 0.010 (0.001–0.020) | 0/5 |
| rr_last | 0.009 (0.000–0.029) | 5/5 |
| sedation_24h | 0.007 (0.000–0.017) | 0/5 |
| temp_last__missing | 0.007 (0.000–0.018) | 0/5 |
| vent_on | 0.007 (0.001–0.018) | 0/5 |
| spo2_last | 0.006 (0.000–0.014) | 5/5 |
| temp_last | 0.006 (0.000–0.014) | 5/5 |
| lactate_last | 0.005 (0.000–0.019) | 5/5 |
| spo2_min_6h | 0.005 (0.000–0.017) | 5/5 |
| creatinine_last | 0.004 (0.000–0.007) | 5/5 |
| emergency | 0.004 (0.000–0.006) | 0/5 |
| vaso_24h | 0.003 (0.000–0.008) | 0/5 |
| hr_last | 0.003 (0.000–0.008) | 5/5 |
| vaso_on | 0.003 (0.000–0.007) | 0/5 |
| hb_last__missing | 0.002 (0.000–0.007) | 0/5 |
| vaso_hours | 0.002 (0.000–0.009) | 5/5 |
| sedation_on | 0.002 (0.000–0.006) | 0/5 |
| hr_slope_6h | 0.001 (0.000–0.001) | 5/5 |
| creatinine_last__missing | 0.000 (0.000–0.001) | 0/5 |
| map_slope_6h | 0.000 (0.000–0.000) | 5/5 |

## Rotation external-eicu: metric learned on inspire + mimic (5 folds, 30 min)

Accepted models per fold [40, 40, 40, 40, 40]; α per site {'inspire': [2.0, 2.0, 2.0, 2.0, 2.0], 'mimic': [1.0, 1.0, 0.5, 1.0, 1.0]}; left out of the metric (constant within each site): sedation_on__missing, vaso_hours__missing, vaso_on__missing.

| variable | weight, mean (min–max over folds) | shaped in folds |
|---|---|---|
| map_mean_6h | 0.299 (0.228–0.374) | 5/5 |
| map_last | 0.147 (0.115–0.193) | 5/5 |
| L | 0.133 (0.101–0.193) | 5/5 |
| map_min_6h | 0.095 (0.063–0.137) | 5/5 |
| map_sd_6h | 0.091 (0.045–0.125) | 5/5 |
| age | 0.060 (0.051–0.068) | 5/5 |
| surgical_group | 0.020 (0.008–0.030) | 5/5 |
| sex_male | 0.018 (0.006–0.025) | 0/5 |
| hb_last | 0.018 (0.006–0.035) | 5/5 |
| vaso_on | 0.014 (0.005–0.029) | 0/5 |
| spo2_last | 0.011 (0.000–0.026) | 5/5 |
| creatinine_last | 0.010 (0.000–0.024) | 5/5 |
| rr_last | 0.010 (0.001–0.022) | 5/5 |
| vent_on | 0.010 (0.000–0.018) | 0/5 |
| vaso_hours | 0.010 (0.000–0.027) | 5/5 |
| temp_last | 0.009 (0.000–0.020) | 5/5 |
| sedation_24h | 0.009 (0.000–0.026) | 0/5 |
| vaso_24h | 0.007 (0.004–0.011) | 0/5 |
| hr_last | 0.006 (0.000–0.008) | 5/5 |
| temp_last__missing | 0.006 (0.002–0.013) | 0/5 |
| lactate_last | 0.005 (0.000–0.010) | 5/5 |
| hr_slope_6h | 0.004 (0.000–0.016) | 5/5 |
| lactate_last__missing | 0.003 (0.001–0.006) | 0/5 |
| spo2_min_6h | 0.003 (0.000–0.011) | 5/5 |
| sedation_on | 0.002 (0.000–0.008) | 0/5 |
| emergency | 0.001 (0.000–0.003) | 0/5 |
| map_slope_6h | 0.000 (0.000–0.001) | 5/5 |

