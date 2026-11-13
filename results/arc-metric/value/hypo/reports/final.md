# `hypo` Phase 3: the frozen artefacts

Per rotation (named by its external site), the transported metric learned on all patients of the two development sites, and the transported comparators. Weights sum to 1; MAP thresholds: the highest MAP below 85 mmHg where the odds of hypotension double (consensus g_j; median and 10–90 % over the accepted models).

## Metric for external-inspire (learned on mimic + eicu; code 5c5b2fe1f006)

60,000 learning landmarks {'eicu': 30000, 'mimic': 30000}; 40 accepted models (40 competitive); left out (constant within each site): none; 5 min.

| variable | weight | shaped |
|---|---|---|
| map_last | 0.315 | yes |
| map_mean_6h | 0.206 | yes |
| map_min_6h | 0.128 | yes |
| map_sd_6h | 0.089 | yes |
| L | 0.060 | yes |
| age | 0.057 | yes |
| sex_male | 0.026 | no |
| surgical_group | 0.021 | yes |
| vaso_on | 0.021 | no |
| vent_on | 0.016 | no |
| vaso_hours | 0.009 | yes |
| hr_last | 0.009 | yes |
| hb_last | 0.008 | yes |
| vaso_24h | 0.008 | no |
| spo2_last | 0.007 | yes |
| spo2_min_6h | 0.007 | yes |
| map_slope_6h | 0.004 | yes |
| lactate_last__missing | 0.004 | no |
| temp_last | 0.001 | yes |
| temp_last__missing | 0.001 | no |
| rr_last | 0.001 | yes |
| hb_last__missing | 0.001 | no |
| creatinine_last__missing | 0.000 | no |
| sedation_on | 0.000 | no |
| emergency | 0.000 | no |
| creatinine_last | 0.000 | yes |
| sedation_24h | 0.000 | no |
| lactate_last | 0.000 | yes |
| hr_slope_6h | 0.000 | yes |

| MAP threshold | consensus | models: median (10–90 %) | models with one |
|---|---|---|---|
| map_last | 69.3 | 70.1 (68.6–70.8) | 35/40 |
| map_mean_6h | 72.2 | 72.1 (69.8–78.3) | 39/40 |
| map_min_6h | 60.7 | 65.8 (62.3–67.4) | 30/40 |

Transported comparators: last_map (853,542 landmarks, 0 s), logreg (853,542 landmarks, 27 s), lightgbm (853,542 landmarks, 232 s).

## Metric for external-mimic (learned on inspire + eicu; code 5c5b2fe1f006)

60,000 learning landmarks {'eicu': 30000, 'inspire': 30000}; 40 accepted models (40 competitive); left out (constant within each site): vaso_hours__missing, vaso_on__missing, sedation_on__missing; 5 min.

| variable | weight | shaped |
|---|---|---|
| L | 0.250 | yes |
| map_mean_6h | 0.232 | yes |
| map_last | 0.147 | yes |
| map_min_6h | 0.127 | yes |
| vaso_on | 0.055 | no |
| map_sd_6h | 0.046 | yes |
| age | 0.030 | yes |
| surgical_group | 0.022 | yes |
| spo2_min_6h | 0.021 | yes |
| sex_male | 0.019 | no |
| creatinine_last | 0.012 | yes |
| hb_last | 0.009 | yes |
| temp_last__missing | 0.007 | no |
| hr_last | 0.007 | yes |
| sedation_24h | 0.003 | no |
| lactate_last | 0.003 | yes |
| rr_last | 0.002 | yes |
| temp_last | 0.002 | yes |
| hb_last__missing | 0.002 | no |
| creatinine_last__missing | 0.002 | no |
| map_slope_6h | 0.002 | yes |
| vent_on | 0.001 | no |
| hr_slope_6h | 0.001 | yes |
| emergency | 0.000 | no |
| lactate_last__missing | 0.000 | no |
| vaso_24h | 0.000 | no |
| spo2_last | 0.000 | yes |
| vaso_hours | 0.000 | yes |
| sedation_on | 0.000 | no |

| MAP threshold | consensus | models: median (10–90 %) | models with one |
|---|---|---|---|
| map_last | 71.0 | 72.0 (70.3–72.5) | 35/40 |
| map_mean_6h | 71.8 | 73.8 (72.2–77.3) | 39/40 |
| map_min_6h | nan | 65.2 (59.7–67.3) | 25/40 |

Transported comparators: last_map (614,677 landmarks, 0 s), logreg (614,677 landmarks, 20 s), lightgbm (614,677 landmarks, 129 s).

## Metric for external-eicu (learned on inspire + mimic; code 5c5b2fe1f006)

60,000 learning landmarks {'inspire': 30000, 'mimic': 30000}; 40 accepted models (40 competitive); left out (constant within each site): vaso_hours__missing, vaso_on__missing, sedation_on__missing; 5 min.

| variable | weight | shaped |
|---|---|---|
| map_last | 0.240 | yes |
| map_mean_6h | 0.222 | yes |
| map_sd_6h | 0.125 | yes |
| L | 0.090 | yes |
| map_min_6h | 0.083 | yes |
| age | 0.052 | yes |
| vaso_24h | 0.041 | no |
| spo2_last | 0.024 | yes |
| sedation_24h | 0.023 | no |
| surgical_group | 0.021 | yes |
| hb_last | 0.015 | yes |
| sex_male | 0.013 | no |
| temp_last__missing | 0.010 | no |
| vaso_on | 0.010 | no |
| emergency | 0.006 | no |
| sedation_on | 0.005 | no |
| creatinine_last | 0.005 | yes |
| spo2_min_6h | 0.004 | yes |
| map_slope_6h | 0.003 | yes |
| hr_slope_6h | 0.003 | yes |
| hr_last | 0.003 | yes |
| vaso_hours | 0.001 | yes |
| lactate_last__missing | 0.001 | no |
| vent_on | 0.000 | no |
| temp_last | 0.000 | yes |
| lactate_last | 0.000 | yes |
| rr_last | 0.000 | yes |

| MAP threshold | consensus | models: median (10–90 %) | models with one |
|---|---|---|---|
| map_last | 69.7 | 70.7 (68.8–71.7) | 34/40 |
| map_mean_6h | 72.5 | 72.9 (71.2–76.9) | 39/40 |
| map_min_6h | nan | 64.8 (60.5–67.7) | 29/40 |

Transported comparators: last_map (535,451 landmarks, 0 s), logreg (535,451 landmarks, 18 s), lightgbm (535,451 landmarks, 141 s).

