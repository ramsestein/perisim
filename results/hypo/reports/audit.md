# Experiment `hypo`: completeness audit of the metric's inputs

Gate: **pass**. Population: normotensive in the last hour, window observable. Missing type: `source` = the cohort cannot measure it; `measurement (declared)` = measured only for some patients by practice (both declared in config/hypo.yaml); otherwise not measured for some landmarks. Quantiles p25 / p50 / p75.

| | inspire | mimic | eicu |
|---|---|---|---|
| landmarks (patients) | 148,293 (11,631) | 387,159 (24,464) | 466,385 (32,032) |
| hypotension in the next 6 h | 13.2% | 30.7% | 19.3% |

| input | inspire | mimic | eicu |
|---|---|---|---|
| map_last | 100% · 78 / 86 / 100 | 100% · 71 / 77.5 / 86 | 100% · 73 / 80 / 90 |
| map_mean_6h | 100% · 78.69 / 87.67 / 97.67 | 100% · 71.67 / 77.33 / 84.67 | 100% · 73.17 / 80.08 / 88.88 |
| map_min_6h | 100% · 70 / 77.5 / 86 | 100% · 64 / 69 / 76 | 100% · 66.5 / 73 / 81 |
| map_sd_6h | 100% · 4.76 / 7.3 / 10.91 | 98% · 4.17 / 6.06 / 8.65 | 99% · 3.4 / 5.16 / 7.69 |
| map_slope_6h | 100% · -2.04 / -0.01 / 1.8 | 98% · -1.3 / 0.17 / 1.6 | 99% · -1.29 / 0.19 / 1.6 |
| hr_last | 99% · 70 / 82 / 92 | 100% · 73 / 83 / 95 | 100% · 72.5 / 83 / 95 |
| hr_slope_6h | 99% · -1.37 / 0 / 1.17 | 98% · -1.23 / 0 / 1.29 | 99% · -1 / 0 / 1.1 |
| spo2_last | 96% · 98 / 99 / 99 | 100% · 95 / 97 / 99 | 99% · 95 / 97 / 99 |
| spo2_min_6h | 96% · 96 / 98 / 99 | 100% · 93.5 / 96 / 98 | 99% · 94 / 96 / 98 |
| rr_last | 99% · 13 / 17 / 19 | 100% · 15 / 18 / 22 | 91% · 14.5 / 17.5 / 21 |
| temp_last | 97% · 36.5 / 37 / 37.3 | 91% · 36.61 / 36.89 / 37.22 | 89% · 36.6 / 36.9 / 37.3 |
| hb_last | 99% · 10.1 / 11.3 / 12.5 | 97% · 9.1 / 10.3 / 11.6 | 91% · 9.2 / 10.5 / 12 |
| creatinine_last | 100% · 0.61 / 0.77 / 0.98 | 97% · 0.7 / 0.9 / 1.2 | 88% · 0.71 / 0.91 / 1.24 |
| lactate_last | 96% · 0.9 / 1.5 / 2.5 | 79% · 1.2 / 1.7 / 2.3 | 23% (measurement (declared)) · 1.1 / 1.6 / 2.4 |
| age | 100% · 50 / 60 / 70 | 100% · 55 / 65 / 75 | 100% · 55 / 65 / 75 |
| L | 100% · 8 / 14 / 26 | 100% · 12 / 20 / 32 | 100% · 10 / 18 / 32 |
| vaso_hours | 0% (source) | 100% · 0 / 0 / 6 | 100% · 0 / 0 / 0 |
| vaso_on | 0% (source) | 100% · 21.1% yes | 100% · 7.6% yes |
| vaso_24h | 100% · 6.2% yes | 100% · 38.3% yes | 100% · 15.6% yes |
| sedation_on | 0% (source) | 100% · 26.2% yes | 100% · 6.5% yes |
| sedation_24h | 100% · 12.5% yes | 100% · 53.0% yes | 100% · 16.4% yes |
| vent_on | 100% · 15.4% yes | 100% · 35.3% yes | 100% · 17.0% yes |
| sex_male | 100% · 57.9% yes | 100% · 61.6% yes | 100% · 58.5% yes |
| emergency | 100% · 18.1% yes | 100% · 56.2% yes | 94% · 11.3% yes |
| surgical_group | 100% · neuro 29%, general_gi 24%, cardiac 21%, thoracic 16% | 100% · cardiac 34%, unknown 25%, general_gi 16%, neuro 11% | 100% · cardiac 32%, plastic_other 19%, general_gi 15%, vascular 11% |

No flags.
