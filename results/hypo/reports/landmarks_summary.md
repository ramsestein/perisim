# Experiment `hypo`: landmark tables

Built 2026-10-28 19:10 · code f75989425604 · config `config/hypo.yaml`. Analysis population: normotensive in the last hour, window observable. Shares; counts ≥ 10.

| | inspire | mimic | eicu |
|---|---|---|---|
| stays (patients) | 12,893 (12,002) | 28,163 (26,083) | 35,201 (33,681) |
| landmarks (patients) | 148,293 (11,631) | 387,159 (24,464) | 466,385 (32,032) |
| window not observable | 0.8% | 1.3% | 1.3% |
| **hypotension in the next 6 h** | **13.2%** | **30.7%** | **19.3%** |
| … seen through one reading per hour | 15.2% | 31.8% | 23.6% |
| lowest MAP < 65 / 65–75 / ≥ 75 | 13% / 28% / 59% | 31% / 40% / 29% | 19% / 36% / 44% |
| vasopressor started (off at L) | nan% | 2.2% | 2.1% |
| vaso running at L | nan% | 21.1% | 7.6% |
| sedation running at L | nan% | 26.2% | 6.5% |
| vent running at L | 15.4% | 35.3% | 17.0% |
| vaso in the last 24 h | 6.2% | 38.3% | 15.6% |
| sedation in the last 24 h | 12.5% | 53.0% | 16.4% |
| arterial line in the last 6 h | 0% | 71% | 61% |

Inputs available at the landmark:

| input | inspire | mimic | eicu |
|---|---|---|---|
| map_last | 100% | 100% | 100% |
| map_mean_6h | 100% | 100% | 100% |
| map_min_6h | 100% | 100% | 100% |
| map_sd_6h | 100% | 98% | 99% |
| map_slope_6h | 100% | 98% | 99% |
| hr_slope_6h | 99% | 98% | 99% |
| spo2_min_6h | 96% | 100% | 99% |
| hr_last | 99% | 100% | 100% |
| spo2_last | 96% | 100% | 99% |
| rr_last | 99% | 100% | 91% |
| temp_last | 97% | 91% | 89% |
| hb_last | 99% | 97% | 91% |
| creatinine_last | 100% | 97% | 88% |
| lactate_last | 96% | 79% | 23% |
| age | 100% | 100% | 100% |
| sex_male | 100% | 100% | 100% |
| bmi | 98% | 0% | 97% |
| emergency | 100% | 100% | 94% |
| surgical_group | 100% | 100% | 100% |
| surgery_duration_h | 100% | 0% | 0% |

eicu: 196 hospitals, 37 with ≥ 300 patients, 2 with ≥ 1,000.
