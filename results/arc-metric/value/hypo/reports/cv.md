# `hypo` Phase 2: internal validation by rotation

Out-of-fold predictions of each development site; PERISIM is shown once per partner site (the metric learned on both sites' training patients, each site queried with its own case base); comparators are local. 95 % CIs by patient bootstrap. Code: 5c5b2fe1f006.

## inspire: 148,293 landmarks, 11,631 patients, prevalence 13.2% (folds [0, 1, 2, 3, 4])

| method | AUROC (95 % CI) | log-loss (95 % CI) | Brier | CITL | slope | mean p |
|---|---|---|---|---|---|---|
| PERISIM (with eicu) | 0.781 (0.775–0.787) | 0.3265 (0.3203–0.3325) | 0.0971 | 0.026 | 1.044 | 0.129 |
| PERISIM (with mimic) | 0.780 (0.774–0.786) | 0.3268 (0.3204–0.3328) | 0.0970 | 0.021 | 1.016 | 0.129 |
| prevalence | 0.494 (0.485–0.504) | 0.3894 (0.3826–0.3957) | 0.1143 | -0.000 | — | 0.132 |
| last MAP (logistic) | 0.743 (0.737–0.748) | 0.3489 (0.3427–0.3550) | 0.1025 | 0.000 | 0.999 | 0.132 |
| logistic, all inputs | 0.781 (0.775–0.787) | 0.3290 (0.3229–0.3352) | 0.0981 | 0.000 | 1.002 | 0.132 |
| LightGBM | 0.802 (0.797–0.807) | 0.3157 (0.3096–0.3216) | 0.0945 | 0.002 | 1.018 | 0.131 |
| Euclidean kNN | 0.765 (0.760–0.770) | 0.3382 (0.3322–0.3442) | 0.1014 | 0.066 | 1.185 | 0.125 |

| paired | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|
| PERISIM (with eicu) − prevalence | -0.0629 (-0.0660 to -0.0603) | 0.287 (0.276 to 0.297) |
| PERISIM (with eicu) − last MAP (logistic) | -0.0224 (-0.0243 to -0.0206) | 0.038 (0.035 to 0.042) |
| PERISIM (with eicu) − logistic, all inputs | -0.0025 (-0.0038 to -0.0012) | -0.000 (-0.003 to 0.002) |
| PERISIM (with eicu) − LightGBM | 0.0108 (0.0099 to 0.0116) | -0.021 (-0.023 to -0.019) |
| PERISIM (with eicu) − Euclidean kNN | -0.0117 (-0.0131 to -0.0103) | 0.016 (0.013 to 0.019) |
| PERISIM (with mimic) − prevalence | -0.0626 (-0.0658 to -0.0599) | 0.286 (0.275 to 0.296) |
| PERISIM (with mimic) − last MAP (logistic) | -0.0221 (-0.0240 to -0.0204) | 0.037 (0.034 to 0.042) |
| PERISIM (with mimic) − logistic, all inputs | -0.0022 (-0.0037 to -0.0009) | -0.001 (-0.004 to 0.002) |
| PERISIM (with mimic) − LightGBM | 0.0110 (0.0100 to 0.0120) | -0.022 (-0.024 to -0.020) |
| PERISIM (with mimic) − Euclidean kNN | -0.0114 (-0.0129 to -0.0099) | 0.015 (0.012 to 0.018) |

## mimic: 387,158 landmarks, 24,464 patients, prevalence 30.7% (folds [0, 1, 2, 3, 4])

| method | AUROC (95 % CI) | log-loss (95 % CI) | Brier | CITL | slope | mean p |
|---|---|---|---|---|---|---|
| PERISIM (with eicu) | 0.783 (0.780–0.786) | 0.5033 (0.5001–0.5068) | 0.1673 | -0.000 | 1.011 | 0.307 |
| PERISIM (with inspire) | 0.782 (0.779–0.785) | 0.5041 (0.5010–0.5077) | 0.1676 | -0.002 | 1.024 | 0.307 |
| prevalence | 0.497 (0.493–0.502) | 0.6164 (0.6136–0.6193) | 0.2126 | -0.000 | — | 0.307 |
| last MAP (logistic) | 0.736 (0.734–0.739) | 0.5476 (0.5443–0.5511) | 0.1825 | -0.000 | 1.000 | 0.307 |
| logistic, all inputs | 0.787 (0.784–0.790) | 0.5014 (0.4980–0.5051) | 0.1661 | -0.000 | 0.999 | 0.307 |
| LightGBM | 0.797 (0.793–0.799) | 0.4905 (0.4872–0.4943) | 0.1628 | 0.002 | 1.004 | 0.306 |
| Euclidean kNN | 0.773 (0.770–0.776) | 0.5140 (0.5108–0.5176) | 0.1715 | -0.022 | 1.111 | 0.311 |

| paired | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|
| PERISIM (with eicu) − prevalence | -0.1131 (-0.1155 to -0.1106) | 0.286 (0.281 to 0.291) |
| PERISIM (with eicu) − last MAP (logistic) | -0.0443 (-0.0458 to -0.0428) | 0.047 (0.045 to 0.048) |
| PERISIM (with eicu) − logistic, all inputs | 0.0019 (0.0009 to 0.0028) | -0.004 (-0.005 to -0.003) |
| PERISIM (with eicu) − LightGBM | 0.0128 (0.0121 to 0.0134) | -0.013 (-0.014 to -0.013) |
| PERISIM (with eicu) − Euclidean kNN | -0.0107 (-0.0116 to -0.0099) | 0.010 (0.009 to 0.011) |
| PERISIM (with inspire) − prevalence | -0.1123 (-0.1146 to -0.1096) | 0.285 (0.280 to 0.290) |
| PERISIM (with inspire) − last MAP (logistic) | -0.0435 (-0.0449 to -0.0421) | 0.046 (0.044 to 0.048) |
| PERISIM (with inspire) − logistic, all inputs | 0.0027 (0.0019 to 0.0036) | -0.005 (-0.006 to -0.004) |
| PERISIM (with inspire) − LightGBM | 0.0136 (0.0130 to 0.0143) | -0.014 (-0.015 to -0.013) |
| PERISIM (with inspire) − Euclidean kNN | -0.0099 (-0.0107 to -0.0090) | 0.009 (0.008 to 0.010) |

## eicu: 466,384 landmarks, 32,032 patients, prevalence 19.3% (folds [0, 1, 2, 3, 4])

| method | AUROC (95 % CI) | log-loss (95 % CI) | Brier | CITL | slope | mean p |
|---|---|---|---|---|---|---|
| PERISIM (with mimic) | 0.800 (0.797–0.802) | 0.3960 (0.3929–0.3996) | 0.1248 | 0.014 | 1.046 | 0.192 |
| PERISIM (with inspire) | 0.799 (0.797–0.802) | 0.3965 (0.3934–0.4001) | 0.1250 | 0.010 | 1.044 | 0.192 |
| prevalence | 0.495 (0.490–0.500) | 0.4911 (0.4877–0.4949) | 0.1560 | -0.000 | — | 0.193 |
| last MAP (logistic) | 0.774 (0.771–0.777) | 0.4215 (0.4178–0.4252) | 0.1313 | -0.000 | 1.003 | 0.193 |
| logistic, all inputs | 0.804 (0.802–0.807) | 0.3960 (0.3928–0.3997) | 0.1246 | 0.000 | 1.000 | 0.193 |
| LightGBM | 0.814 (0.812–0.817) | 0.3850 (0.3819–0.3886) | 0.1217 | 0.003 | 1.005 | 0.193 |
| Euclidean kNN | 0.784 (0.782–0.787) | 0.4101 (0.4067–0.4136) | 0.1302 | 0.002 | 1.156 | 0.193 |

| paired | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|
| PERISIM (with mimic) − prevalence | -0.0951 (-0.0970 to -0.0933) | 0.305 (0.299 to 0.310) |
| PERISIM (with mimic) − last MAP (logistic) | -0.0255 (-0.0268 to -0.0242) | 0.026 (0.024 to 0.027) |
| PERISIM (with mimic) − logistic, all inputs | -0.0000 (-0.0008 to 0.0009) | -0.005 (-0.006 to -0.004) |
| PERISIM (with mimic) − LightGBM | 0.0110 (0.0105 to 0.0115) | -0.014 (-0.015 to -0.014) |
| PERISIM (with mimic) − Euclidean kNN | -0.0141 (-0.0149 to -0.0132) | 0.015 (0.014 to 0.017) |
| PERISIM (with inspire) − prevalence | -0.0946 (-0.0964 to -0.0928) | 0.304 (0.299 to 0.310) |
| PERISIM (with inspire) − last MAP (logistic) | -0.0249 (-0.0262 to -0.0236) | 0.025 (0.024 to 0.027) |
| PERISIM (with inspire) − logistic, all inputs | 0.0005 (-0.0003 to 0.0015) | -0.005 (-0.006 to -0.004) |
| PERISIM (with inspire) − LightGBM | 0.0115 (0.0110 to 0.0121) | -0.015 (-0.016 to -0.014) |
| PERISIM (with inspire) − Euclidean kNN | -0.0135 (-0.0144 to -0.0127) | 0.015 (0.014 to 0.016) |

## Rotation external-inspire: metric learned on mimic + eicu (5 folds, 45 min)

Accepted models per fold [40, 40, 40, 40, 40]; α per site {'mimic': [1.0, 1.0, 1.0, 1.0, 1.0], 'eicu': [2.0, 2.0, 2.0, 2.0, 2.0]}; left out of the metric (constant within each site): none.

| variable | weight, mean (min–max over folds) | shaped in folds |
|---|---|---|
| map_mean_6h | 0.290 (0.264–0.314) | 5/5 |
| map_last | 0.188 (0.160–0.243) | 5/5 |
| map_min_6h | 0.105 (0.062–0.140) | 5/5 |
| L | 0.102 (0.081–0.123) | 5/5 |
| map_sd_6h | 0.097 (0.056–0.162) | 5/5 |
| age | 0.057 (0.036–0.095) | 5/5 |
| sex_male | 0.028 (0.018–0.042) | 0/5 |
| surgical_group | 0.017 (0.011–0.023) | 5/5 |
| hb_last | 0.015 (0.007–0.028) | 5/5 |
| vaso_on | 0.015 (0.013–0.018) | 0/5 |
| vaso_hours | 0.012 (0.005–0.019) | 5/5 |
| spo2_last | 0.012 (0.000–0.018) | 4/5 |
| sedation_on | 0.007 (0.000–0.020) | 0/5 |
| vent_on | 0.005 (0.000–0.019) | 0/5 |
| hr_last | 0.005 (0.001–0.019) | 5/5 |
| creatinine_last | 0.005 (0.000–0.012) | 5/5 |
| rr_last | 0.005 (0.000–0.019) | 5/5 |
| map_slope_6h | 0.004 (0.000–0.009) | 5/5 |
| emergency | 0.004 (0.000–0.010) | 0/5 |
| temp_last | 0.004 (0.000–0.015) | 5/5 |
| hr_slope_6h | 0.004 (0.000–0.011) | 5/5 |
| lactate_last | 0.003 (0.000–0.014) | 5/5 |
| temp_last__missing | 0.003 (0.000–0.008) | 0/5 |
| spo2_min_6h | 0.003 (0.000–0.006) | 5/5 |
| vaso_24h | 0.003 (0.000–0.011) | 0/5 |
| lactate_last__missing | 0.003 (0.000–0.007) | 0/5 |
| sedation_24h | 0.002 (0.000–0.005) | 0/5 |
| creatinine_last__missing | 0.002 (0.000–0.005) | 0/5 |
| hb_last__missing | 0.002 (0.000–0.006) | 0/5 |

## Rotation external-mimic: metric learned on inspire + eicu (5 folds, 43 min)

Accepted models per fold [40, 40, 40, 40, 40]; α per site {'inspire': [2.0, 2.0, 2.0, 2.0, 5.0], 'eicu': [2.0, 2.0, 2.0, 2.0, 2.0]}; left out of the metric (constant within each site): sedation_on__missing, vaso_hours__missing, vaso_on__missing.

| variable | weight, mean (min–max over folds) | shaped in folds |
|---|---|---|
| map_min_6h | 0.223 (0.104–0.300) | 5/5 |
| map_last | 0.199 (0.112–0.317) | 5/5 |
| map_mean_6h | 0.197 (0.114–0.280) | 5/5 |
| L | 0.122 (0.054–0.204) | 5/5 |
| map_sd_6h | 0.082 (0.013–0.200) | 5/5 |
| age | 0.035 (0.023–0.057) | 5/5 |
| surgical_group | 0.026 (0.010–0.037) | 5/5 |
| sex_male | 0.014 (0.007–0.019) | 0/5 |
| vaso_on | 0.013 (0.000–0.055) | 0/5 |
| hb_last | 0.011 (0.004–0.023) | 5/5 |
| hr_last | 0.011 (0.002–0.019) | 5/5 |
| vent_on | 0.010 (0.003–0.017) | 0/5 |
| creatinine_last | 0.008 (0.000–0.016) | 5/5 |
| lactate_last__missing | 0.008 (0.001–0.020) | 0/5 |
| temp_last | 0.007 (0.000–0.022) | 5/5 |
| temp_last__missing | 0.005 (0.001–0.011) | 0/5 |
| lactate_last | 0.004 (0.000–0.008) | 5/5 |
| rr_last | 0.003 (0.001–0.005) | 5/5 |
| emergency | 0.003 (0.000–0.013) | 0/5 |
| hb_last__missing | 0.003 (0.000–0.015) | 0/5 |
| spo2_last | 0.003 (0.000–0.006) | 4/5 |
| map_slope_6h | 0.003 (0.000–0.010) | 5/5 |
| vaso_hours | 0.002 (0.000–0.007) | 5/5 |
| sedation_24h | 0.001 (0.000–0.004) | 0/5 |
| creatinine_last__missing | 0.001 (0.000–0.002) | 0/5 |
| vaso_24h | 0.001 (0.000–0.004) | 0/5 |
| spo2_min_6h | 0.001 (0.000–0.003) | 5/5 |
| sedation_on | 0.001 (0.000–0.004) | 0/5 |
| hr_slope_6h | 0.000 (0.000–0.001) | 5/5 |

## Rotation external-eicu: metric learned on inspire + mimic (5 folds, 35 min)

Accepted models per fold [40, 40, 40, 40, 40]; α per site {'inspire': [2.0, 2.0, 2.0, 2.0, 2.0], 'mimic': [2.0, 1.0, 1.0, 1.0, 1.0]}; left out of the metric (constant within each site): sedation_on__missing, vaso_hours__missing, vaso_on__missing.

| variable | weight, mean (min–max over folds) | shaped in folds |
|---|---|---|
| map_mean_6h | 0.271 (0.165–0.414) | 5/5 |
| map_last | 0.170 (0.145–0.203) | 5/5 |
| map_min_6h | 0.152 (0.053–0.402) | 5/5 |
| L | 0.120 (0.089–0.156) | 5/5 |
| map_sd_6h | 0.081 (0.019–0.114) | 5/5 |
| age | 0.049 (0.026–0.067) | 5/5 |
| surgical_group | 0.020 (0.010–0.033) | 5/5 |
| hb_last | 0.018 (0.004–0.030) | 5/5 |
| sex_male | 0.013 (0.007–0.020) | 0/5 |
| vaso_24h | 0.012 (0.000–0.040) | 0/5 |
| vaso_hours | 0.011 (0.000–0.021) | 5/5 |
| vaso_on | 0.011 (0.007–0.019) | 0/5 |
| temp_last | 0.009 (0.001–0.020) | 5/5 |
| map_slope_6h | 0.008 (0.000–0.041) | 5/5 |
| vent_on | 0.008 (0.000–0.021) | 0/5 |
| emergency | 0.008 (0.000–0.016) | 0/5 |
| creatinine_last | 0.008 (0.000–0.018) | 5/5 |
| lactate_last | 0.006 (0.000–0.017) | 5/5 |
| sedation_24h | 0.005 (0.000–0.013) | 0/5 |
| hr_last | 0.005 (0.000–0.022) | 5/5 |
| lactate_last__missing | 0.005 (0.000–0.008) | 0/5 |
| rr_last | 0.003 (0.000–0.007) | 5/5 |
| spo2_last | 0.002 (0.000–0.007) | 5/5 |
| temp_last__missing | 0.002 (0.000–0.004) | 0/5 |
| spo2_min_6h | 0.002 (0.000–0.004) | 5/5 |
| hr_slope_6h | 0.001 (0.000–0.003) | 5/5 |
| sedation_on | 0.001 (0.000–0.003) | 0/5 |

