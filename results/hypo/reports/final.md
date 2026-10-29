# `hypo` Phase 3: the frozen artefacts

Per rotation (named by its external site), the transported metric learned on all patients of the two development sites, and the transported comparators. Weights sum to 1; MAP thresholds: the highest MAP below 85 mmHg where the odds of hypotension double (consensus g_j; median and 10–90 % over the accepted models).

## Metric for external-inspire (learned on mimic + eicu; code cf0d0b80c70a)

60,000 learning landmarks {'eicu': 30000, 'mimic': 30000}; 40 accepted models (40 competitive); left out (constant within each site): none; 5 min.

| variable | weight | shaped |
|---|---|---|
| map_mean_6h | 0.230 | yes |
| map_min_6h | 0.169 | yes |
| map_last | 0.131 | yes |
| age | 0.110 | yes |
| L | 0.099 | yes |
| map_sd_6h | 0.054 | yes |
| surgical_group | 0.032 | yes |
| sex_male | 0.031 | no |
| vaso_on | 0.020 | no |
| temp_last | 0.017 | yes |
| hb_last | 0.013 | yes |
| spo2_last | 0.013 | yes |
| spo2_min_6h | 0.011 | yes |
| lactate_last__missing | 0.011 | no |
| hr_last | 0.009 | yes |
| vaso_24h | 0.009 | no |
| vaso_hours | 0.008 | yes |
| rr_last | 0.008 | yes |
| creatinine_last__missing | 0.008 | no |
| emergency | 0.006 | no |
| vent_on | 0.005 | no |
| sedation_24h | 0.004 | no |
| hb_last__missing | 0.001 | no |
| hr_slope_6h | 0.001 | yes |
| map_slope_6h | 0.000 | yes |
| lactate_last | 0.000 | yes |
| creatinine_last | 0.000 | yes |
| sedation_on | 0.000 | no |
| temp_last__missing | 0.000 | no |

| MAP threshold | consensus | models: median (10–90 %) | models with one |
|---|---|---|---|
| map_last | 69.2 | 69.9 (68.3–70.6) | 35/40 |
| map_mean_6h | 71.2 | 72.2 (69.5–76.8) | 39/40 |
| map_min_6h | 63.0 | 64.4 (60.9–68.4) | 34/40 |

Transported comparators: last_map (853,544 landmarks, 0 s), logreg (853,544 landmarks, 26 s), lightgbm (853,544 landmarks, 187 s).

## Metric for external-mimic (learned on inspire + eicu; code cf0d0b80c70a)

60,000 learning landmarks {'eicu': 30000, 'inspire': 30000}; 40 accepted models (40 competitive); left out (constant within each site): vaso_hours__missing, vaso_on__missing, sedation_on__missing; 6 min.

| variable | weight | shaped |
|---|---|---|
| L | 0.281 | yes |
| map_last | 0.210 | yes |
| map_min_6h | 0.209 | yes |
| map_mean_6h | 0.147 | yes |
| map_sd_6h | 0.030 | yes |
| surgical_group | 0.029 | yes |
| lactate_last__missing | 0.028 | no |
| sex_male | 0.013 | no |
| age | 0.010 | yes |
| spo2_last | 0.008 | yes |
| creatinine_last | 0.006 | yes |
| sedation_24h | 0.005 | no |
| vent_on | 0.005 | no |
| hb_last__missing | 0.005 | no |
| hb_last | 0.004 | yes |
| hr_slope_6h | 0.004 | yes |
| rr_last | 0.003 | yes |
| hr_last | 0.001 | yes |
| temp_last | 0.001 | yes |
| vaso_hours | 0.001 | yes |
| emergency | 0.000 | no |
| vaso_on | 0.000 | no |
| lactate_last | 0.000 | yes |
| vaso_24h | 0.000 | no |
| creatinine_last__missing | 0.000 | no |
| sedation_on | 0.000 | no |
| temp_last__missing | 0.000 | no |
| spo2_min_6h | 0.000 | yes |
| map_slope_6h | 0.000 | yes |

| MAP threshold | consensus | models: median (10–90 %) | models with one |
|---|---|---|---|
| map_last | 71.2 | 72.9 (70.8–73.8) | 34/40 |
| map_mean_6h | 68.3 | 70.3 (68.5–74.6) | 40/40 |
| map_min_6h | 67.4 | 68.0 (62.1–70.9) | 38/40 |

Transported comparators: last_map (614,678 landmarks, 0 s), logreg (614,678 landmarks, 19 s), lightgbm (614,678 landmarks, 106 s).

## Metric for external-eicu (learned on inspire + mimic; code cf0d0b80c70a)

60,000 learning landmarks {'inspire': 30000, 'mimic': 30000}; 40 accepted models (40 competitive); left out (constant within each site): vaso_hours__missing, vaso_on__missing, sedation_on__missing; 5 min.

| variable | weight | shaped |
|---|---|---|
| map_mean_6h | 0.386 | yes |
| map_last | 0.122 | yes |
| map_sd_6h | 0.118 | yes |
| L | 0.116 | yes |
| map_min_6h | 0.079 | yes |
| surgical_group | 0.038 | yes |
| age | 0.033 | yes |
| sedation_24h | 0.023 | no |
| spo2_last | 0.023 | yes |
| sex_male | 0.014 | no |
| spo2_min_6h | 0.012 | yes |
| hb_last | 0.009 | yes |
| vaso_on | 0.008 | no |
| vaso_hours | 0.006 | yes |
| vaso_24h | 0.003 | no |
| lactate_last__missing | 0.003 | no |
| hr_last | 0.002 | yes |
| temp_last | 0.002 | yes |
| hr_slope_6h | 0.001 | yes |
| vent_on | 0.000 | no |
| emergency | 0.000 | no |
| rr_last | 0.000 | yes |
| sedation_on | 0.000 | no |
| lactate_last | 0.000 | yes |
| temp_last__missing | 0.000 | no |
| map_slope_6h | 0.000 | yes |
| creatinine_last | 0.000 | yes |

| MAP threshold | consensus | models: median (10–90 %) | models with one |
|---|---|---|---|
| map_last | 69.6 | 70.4 (69.2–71.2) | 34/40 |
| map_mean_6h | 73.2 | 73.7 (71.1–76.8) | 40/40 |
| map_min_6h | nan | 62.0 (59.4–64.6) | 31/40 |

Transported comparators: last_map (535,452 landmarks, 0 s), logreg (535,452 landmarks, 17 s), lightgbm (535,452 landmarks, 112 s).

