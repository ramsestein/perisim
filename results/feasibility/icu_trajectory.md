# Feasibility: ICU course of surgical patients, predicted early in the ICU stay

Aggregates of the three cohorts' common format (shares; counts ≥ 10). Population: postoperative adult ICU stays, the first of each hospitalization, ICU deaths INCLUDED. Inputs and outcomes (B, C): approximations on T2's index stays (ICU survivors with ward follow-up), the only stays the common format holds.

## A. Population

| | inspire | mimic | eicu |
|---|---|---|---|
| stays (patients) | 12,893 (12,002) | 28,163 (26,083) | 35,282 (33,759) |
| ICU stay, h, q25 | 19.7 | 28.3 | 22.6 |
| ICU stay, h, q50 | 24.1 | 47.6 | 39.1 |
| ICU stay, h, q75 | 46.5 | 84.3 | 70.9 |
| ICU stay, h, q90 | 96.3 | 170.9 | 132.2 |
| ICU stay < 6 h | 1.8% | 1.7% | 2.5% |
| ICU stay 6–24 h | 47.6% | 12.4% | 27.2% |
| ICU stay 24–48 h | 27.7% | 36.4% | 31.4% |
| ICU stay 48 h – 5 d | 16.0% | 33.3% | 27.3% |
| ICU stay > 5 d | 6.9% | 16.2% | 11.6% |
| ICU death | 0.8% | 4.3% | 1.9% |
| hospital death | 2.5% | 6.1% | 4.1% |
| ICU readmission (same hospitalization) | 3.3% | 6.8% | 6.2% |
| surgical group unseen in INSPIRE | 0.0% | 26.9% | 5.3% |

### Prediction at 6 h after ICU admission

| | inspire | mimic | eicu |
|---|---|---|---|
| still in the ICU and alive | 12,650 (98%) | 27,637 (98%) | 34,378 (97%) |
| ICU death | 0.7% | 4.0% | 1.8% |
| alive, ICU stay ≤ 48 h | 76.5% | 48.3% | 59.4% |
| alive, ICU stay 48 h – 5 d | 16.2% | 32.9% | 27.6% |
| alive, ICU stay > 5 d | 6.6% | 14.8% | 11.2% |
| remaining ICU time, tertile cuts (h) | 15.5, 29.3 | 25.8, 62.7 | 19.6, 47.6 |

### Prediction at 12 h after ICU admission

| | inspire | mimic | eicu |
|---|---|---|---|
| still in the ICU and alive | 12,299 (95%) | 27,301 (97%) | 33,707 (96%) |
| ICU death | 0.7% | 3.8% | 1.7% |
| alive, ICU stay ≤ 48 h | 75.9% | 48.0% | 58.7% |
| alive, ICU stay 48 h – 5 d | 16.7% | 33.3% | 28.1% |
| alive, ICU stay > 5 d | 6.7% | 15.0% | 11.5% |
| remaining ICU time, tertile cuts (h) | 9.8, 26.4 | 20.0, 57.3 | 14.0, 42.6 |

### Prediction at 24 h after ICU admission

| | inspire | mimic | eicu |
|---|---|---|---|
| still in the ICU and alive | 6,460 (50%) | 24,167 (86%) | 24,763 (70%) |
| ICU death | 1.3% | 3.7% | 2.0% |
| alive, ICU stay ≤ 48 h | 54.2% | 41.7% | 44.2% |
| alive, ICU stay 48 h – 5 d | 31.7% | 37.6% | 38.3% |
| alive, ICU stay > 5 d | 12.8% | 16.9% | 15.6% |
| remaining ICU time, tertile cuts (h) | 7.9, 44.7 | 13.8, 51.0 | 18.3, 48.5 |

eICU hospitals: 203 with any stay, 38 with ≥ 300, 2 with ≥ 1,000; 68% of the stays are in hospitals with ≥ 300 (median 61 stays per hospital).

Surgical groups (share of stays):

| group | inspire | mimic | eicu |
|---|---|---|---|
| (missing) | 0.0% | 26.9% | 5.3% |
| cardiac | 16.4% | 34.2% | 27.4% |
| ent_head_neck | 2.3% | 1.0% | 0.4% |
| general_gi | 23.3% | 15.3% | 15.1% |
| neuro | 30.1% | 10.5% | 4.4% |
| ortho | 4.2% | 2.6% | 4.2% |
| other | 0.3% | 0.0% | 0.0% |
| plastic_other | 1.0% | 0.4% | 18.6% |
| thoracic | 18.1% | 3.8% | 7.9% |
| uro_gyn | 2.8% | 1.6% | 3.9% |
| vascular | 1.6% | 3.7% | 12.7% |

## B. Inputs by the prediction time (T2's index stays)

T2's extraction keeps the vitals of MIMIC-IV and eICU only in a window before the ICU discharge, so in long stays the first ICU hours are missing from the common format (not from the databases): the rows 'ICU stay ≤ 26 h' are the stays where that window reaches the ICU admission.

### 6h

| variable | inspire | mimic | eicu |
|---|---|---|---|
| vital_hr | 97% | 91% | 65% |
| vital_map | 97% | 90% | 65% |
| vital_spo2 | 90% | 91% | 65% |
| vital_rr | 97% | 91% | 61% |
| vital_temp | 91% | 83% | 12% |
| vital_gcs | 44% | 87% | 58% |
| vital_fio2 | 59% | 60% | 44% |
| lab_creatinine | 100% | 96% | 70% |
| lab_hb | 99% | 97% | 80% |
| lab_plt | 97% | 97% | 75% |
| lab_wbc | 98% | 97% | 74% |
| lab_lactate | 95% | 82% | 18% |
| lab_albumin | 98% | 39% | 31% |
| lab_bilirubin | 95% | 45% | 28% |
| lab_ph | 98% | 74% | 54% |
| lab_pco2 | 98% | 74% | 55% |
| lab_po2 | 98% | 74% | 55% |
| support_vent | 37% | 48% | 40% |
| support_vaso | 23% | 34% | 18% |
| support_rrt | 1% | 0% | 1% |
| heart-rate values per stay (median) | 14 | 8 | 71 |

### 12h

| variable | inspire | mimic | eicu |
|---|---|---|---|
| vital_hr | 97% | 96% | 67% |
| vital_map | 97% | 94% | 68% |
| vital_spo2 | 96% | 95% | 67% |
| vital_rr | 97% | 95% | 64% |
| vital_temp | 96% | 88% | 13% |
| vital_gcs | 48% | 95% | 63% |
| vital_fio2 | 60% | 66% | 45% |
| lab_creatinine | 100% | 99% | 83% |
| lab_hb | 100% | 99% | 89% |
| lab_plt | 99% | 99% | 85% |
| lab_wbc | 99% | 99% | 85% |
| lab_lactate | 96% | 83% | 20% |
| lab_albumin | 98% | 41% | 36% |
| lab_bilirubin | 95% | 47% | 33% |
| lab_ph | 98% | 77% | 56% |
| lab_pco2 | 98% | 76% | 56% |
| lab_po2 | 98% | 76% | 57% |
| support_vent | 37% | 53% | 41% |
| support_vaso | 25% | 40% | 21% |
| support_rrt | 1% | 1% | 1% |
| heart-rate values per stay (median) | 21 | 14 | 142 |

### 24h

| variable | inspire | mimic | eicu |
|---|---|---|---|
| vital_hr | 97% | 96% | 67% |
| vital_map | 97% | 95% | 67% |
| vital_spo2 | 97% | 96% | 67% |
| vital_rr | 97% | 95% | 63% |
| vital_temp | 97% | 89% | 14% |
| vital_gcs | 53% | 96% | 66% |
| vital_fio2 | 72% | 72% | 52% |
| lab_creatinine | 100% | 100% | 96% |
| lab_hb | 100% | 100% | 97% |
| lab_plt | 100% | 100% | 95% |
| lab_wbc | 100% | 100% | 96% |
| lab_lactate | 97% | 86% | 23% |
| lab_albumin | 100% | 45% | 44% |
| lab_bilirubin | 97% | 51% | 40% |
| lab_ph | 99% | 81% | 63% |
| lab_pco2 | 99% | 81% | 64% |
| lab_po2 | 99% | 81% | 64% |
| support_vent | 46% | 58% | 49% |
| support_vaso | 26% | 44% | 26% |
| support_rrt | 3% | 1% | 2% |
| heart-rate values per stay (median) | 37 | 27 | 279 |

### 6h, ICU stay ≤ 26 h

| variable | inspire | mimic | eicu |
|---|---|---|---|
| vital_hr | 97% | 94% | 97% |
| vital_map | 97% | 92% | 98% |
| vital_spo2 | 91% | 94% | 97% |
| vital_rr | 97% | 94% | 92% |
| vital_temp | 92% | 91% | 13% |
| vital_gcs | 52% | 91% | 57% |
| vital_fio2 | 47% | 38% | 29% |
| lab_creatinine | 100% | 95% | 63% |
| lab_hb | 98% | 95% | 71% |
| lab_plt | 98% | 95% | 66% |
| lab_wbc | 98% | 95% | 66% |
| lab_lactate | 95% | 68% | 11% |
| lab_albumin | 99% | 30% | 25% |
| lab_bilirubin | 96% | 36% | 22% |
| lab_ph | 98% | 55% | 38% |
| lab_pco2 | 98% | 55% | 39% |
| lab_po2 | 98% | 55% | 39% |
| support_vent | 28% | 28% | 22% |
| support_vaso | 25% | 18% | 10% |
| support_rrt | 0% | 0% | 1% |
| heart-rate values per stay (median) | 13 | 8 | 71 |

### 12h, ICU stay ≤ 26 h

| variable | inspire | mimic | eicu |
|---|---|---|---|
| vital_hr | 98% | 95% | 98% |
| vital_map | 98% | 94% | 99% |
| vital_spo2 | 97% | 95% | 98% |
| vital_rr | 97% | 95% | 93% |
| vital_temp | 97% | 93% | 14% |
| vital_gcs | 55% | 95% | 61% |
| vital_fio2 | 48% | 41% | 31% |
| lab_creatinine | 100% | 99% | 80% |
| lab_hb | 99% | 99% | 85% |
| lab_plt | 99% | 99% | 80% |
| lab_wbc | 99% | 99% | 82% |
| lab_lactate | 95% | 68% | 12% |
| lab_albumin | 99% | 32% | 30% |
| lab_bilirubin | 96% | 38% | 26% |
| lab_ph | 98% | 57% | 40% |
| lab_pco2 | 98% | 57% | 40% |
| lab_po2 | 98% | 57% | 41% |
| support_vent | 28% | 30% | 23% |
| support_vaso | 27% | 20% | 11% |
| support_rrt | 0% | 1% | 1% |
| heart-rate values per stay (median) | 20 | 14 | 143 |

### 24h, ICU stay ≤ 26 h

| variable | inspire | mimic | eicu |
|---|---|---|---|
| vital_hr | 98% | 95% | 99% |
| vital_map | 98% | 95% | 99% |
| vital_spo2 | 98% | 95% | 99% |
| vital_rr | 98% | 95% | 95% |
| vital_temp | 98% | 93% | 22% |
| vital_gcs | 60% | 95% | 62% |
| vital_fio2 | 51% | 60% | 37% |
| lab_creatinine | 100% | 100% | 94% |
| lab_hb | 100% | 100% | 96% |
| lab_plt | 100% | 100% | 92% |
| lab_wbc | 100% | 100% | 95% |
| lab_lactate | 96% | 77% | 12% |
| lab_albumin | 100% | 37% | 34% |
| lab_bilirubin | 96% | 40% | 30% |
| lab_ph | 98% | 73% | 49% |
| lab_pco2 | 98% | 72% | 50% |
| lab_po2 | 98% | 72% | 50% |
| support_vent | 29% | 48% | 34% |
| support_vaso | 35% | 33% | 16% |
| support_rrt | 0% | 0% | 1% |
| heart-rate values per stay (median) | 32 | 27 | 285 |

## C. ICU-course outcomes (T2's index stays: ICU survivors, so lower bounds)

| | inspire | mimic | eicu |
|---|---|---|---|
| ventilation beyond 24 h | 5.8% | 16.9% | 6.1% |
| ventilation beyond 48 h | 3.9% | 10.1% | 3.7% |
| vasopressors beyond 24 h | 3.3% | 16.6% | 9.5% |
| RRT in the ICU | 2.1% | 2.8% | 2.0% |
| creatinine AKI in the ICU (of stays with baseline and ICU values) | 10.8% | 18.7% | 17.7% |
| stays with baseline and ICU creatinine | 84.7% | 81.0% | 45.4% |
