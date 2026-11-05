# `hypo` Phase 1 smoke run: development inspire + mimic

Code 4cfc09a35ba0. Learning sample 55,016 landmarks ({'inspire': 27508, 'mimic': 27508}); 27 metric variables, 18 shaped; 40 accepted models (40 competitive, 46 attempts). Left out (constant within every site): vaso_hours__missing, vaso_on__missing, sedation_on__missing. Time: shapes 174 s, NCA 187 s, total 411 s.

| site | case base | test | α | AUROC | log-loss | prevalence log-loss | mean p / prevalence | OOD |
|---|---|---|---|---|---|---|---|---|
| inspire | 118,832 | 29,461 | 5 | 0.777 | 0.3303 | 0.3917 | 0.131 / 0.133 | 0.8% |
| mimic | 309,804 | 77,355 | 1 | 0.784 | 0.5024 | 0.6160 | 0.306 / 0.306 | 0.9% |

| variable | weight | shaped |
|---|---|---|
| map_mean_6h | 0.372 | yes |
| map_last | 0.183 | yes |
| map_min_6h | 0.113 | yes |
| L | 0.105 | yes |
| surgical_group | 0.061 | yes |
| map_sd_6h | 0.051 | yes |
| age | 0.025 | yes |
| hb_last | 0.014 | yes |
| spo2_min_6h | 0.013 | yes |
| vaso_24h | 0.011 |  |
| sex_male | 0.010 |  |
| lactate_last__missing | 0.009 |  |
| map_slope_6h | 0.008 | yes |
| rr_last | 0.005 | yes |
| vaso_on | 0.005 |  |
| hr_slope_6h | 0.004 | yes |
| spo2_last | 0.003 | yes |
| temp_last__missing | 0.002 |  |
| vaso_hours | 0.002 | yes |
| vent_on | 0.002 |  |
| hr_last | 0.002 | yes |
| creatinine_last | 0.000 | yes |
| lactate_last | 0.000 | yes |
| temp_last | 0.000 | yes |
| sedation_24h | 0.000 |  |
| emergency | 0.000 |  |
| sedation_on | 0.000 |  |
