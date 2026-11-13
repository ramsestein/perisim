# Transport of the frozen hypo metrics to NWICU

Pre-registered: docs/prereg_nwicu.md. 2,001 patients, 30,833 landmarks, outcome 12.9%; five patient folds; 2,000 patient resamples.

| arm | refitted on NWICU | log-loss (95 % CI) | AUROC | Brier | CITL | slope |
|---|---|---|---|---|---|---|
| kNN, metric of rotation inspire | case base | 0.3192 (0.3048–0.3339) | 0.791 | 0.0962 | +0.086 | 1.04 |
| kNN, metric of rotation mimic | case base | 0.3191 (0.3045–0.3340) | 0.792 | 0.0963 | +0.107 | 1.04 |
| kNN, metric of rotation eicu | case base | 0.3183 (0.3036–0.3330) | 0.793 | 0.0960 | +0.094 | 1.06 |
| LightGBM of rotation inspire | nothing | 0.3441 (0.3330–0.3553) | 0.815 | 0.1051 | -0.815 | 1.14 |
| LightGBM of rotation mimic | nothing | 0.3188 (0.3068–0.3305) | 0.814 | 0.0964 | -0.480 | 1.09 |
| LightGBM of rotation eicu | nothing | 0.3537 (0.3429–0.3648) | 0.813 | 0.1087 | -0.899 | 1.13 |
| LightGBM of rotation inspire, recalibrated | recalibration | 0.3067 (0.2930–0.3209) | 0.814 | 0.0930 | +0.000 | 1.00 |
| LightGBM of rotation mimic, recalibrated | recalibration | 0.3067 (0.2928–0.3207) | 0.814 | 0.0930 | +0.000 | 1.00 |
| LightGBM of rotation eicu, recalibrated | recalibration | 0.3078 (0.2941–0.3222) | 0.812 | 0.0934 | +0.000 | 1.00 |
| kNN, local metric | metric + case base | 0.3222 (0.3077–0.3373) | 0.786 | 0.0971 | +0.085 | 1.07 |
| kNN, Euclidean | case base | 0.3295 (0.3148–0.3446) | 0.774 | 0.0996 | +0.108 | 1.12 |
| LightGBM, local | everything | 0.3086 (0.2948–0.3227) | 0.811 | 0.0934 | +0.014 | 1.03 |

## Primary: R of each transported metric

- rotation inspire (learned on the other two sites): 1.405 (90 % CI 1.046–1.996; equivalent)
- rotation mimic (learned on the other two sites): 1.420 (90 % CI 1.065–2.041; equivalent)
- rotation eicu (learned on the other two sites): 1.526 (90 % CI 1.176–2.201; equivalent)

## Secondary

- kNN inspire − recalibrated LightGBM inspire: +0.0126 (+0.0099 to +0.0152)
- kNN mimic − recalibrated LightGBM mimic: +0.0124 (+0.0096 to +0.0154)
- kNN eicu − recalibrated LightGBM eicu: +0.0106 (+0.0077 to +0.0135)
- local LightGBM − best transported kNN (eicu): -0.0098 (-0.0125 to -0.0070)

## Descriptive: vasopressor stratum (on in 12.8% of landmarks)

- rotation inspire: transport cost on -0.0098, off -0.0020; B0 (same-state neighbours) on the stratum -0.0006 (positive = B0 better), overall +0.0003 (B0 − kNN)
- rotation mimic: transport cost on -0.0101, off -0.0020; B0 (same-state neighbours) on the stratum +0.0013 (positive = B0 better), overall +0.0000 (B0 − kNN)
- rotation eicu: transport cost on -0.0085, off -0.0032; B0 (same-state neighbours) on the stratum +0.0005 (positive = B0 better), overall +0.0002 (B0 − kNN)

## Rule (pre-registered)

```
{
 "primary": "confirmed",
 "equivalent": [
  "inspire",
  "mimic",
  "eicu"
 ],
 "secondary_1": {
  "inspire": "kNN worse",
  "mimic": "kNN worse",
  "eicu": "kNN worse"
 },
 "secondary_2": {
  "inspire": {
   "citl": 0.08625664214503977,
   "slope": 1.0448459561698153
  },
  "mimic": {
   "citl": 0.10747928876508882,
   "slope": 1.0394282925083278
  },
  "eicu": {
   "citl": 0.09355840802631127,
   "slope": 1.05630451849627
  }
 }
}
```
