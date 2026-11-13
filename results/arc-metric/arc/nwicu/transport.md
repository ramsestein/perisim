# Transport of the frozen hypo metrics to NWICU

Pre-registered: docs/prereg_nwicu.md. 2,001 patients, 30,833 landmarks, outcome 12.9%; five patient folds; 2,000 patient resamples.

| arm | refitted on NWICU | log-loss (95 % CI) | AUROC | Brier | CITL | slope |
|---|---|---|---|---|---|---|
| kNN, metric of rotation inspire | case base | 0.3196 (0.3051–0.3348) | 0.791 | 0.0961 | +0.099 | 1.06 |
| kNN, metric of rotation mimic | case base | 0.3177 (0.3031–0.3327) | 0.794 | 0.0959 | +0.129 | 1.04 |
| kNN, metric of rotation eicu | case base | 0.3188 (0.3042–0.3340) | 0.793 | 0.0962 | +0.106 | 1.05 |
| LightGBM of rotation inspire | nothing | 0.3441 (0.3330–0.3553) | 0.815 | 0.1051 | -0.815 | 1.14 |
| LightGBM of rotation mimic | nothing | 0.3188 (0.3068–0.3305) | 0.814 | 0.0964 | -0.480 | 1.09 |
| LightGBM of rotation eicu | nothing | 0.3537 (0.3429–0.3648) | 0.813 | 0.1087 | -0.899 | 1.13 |
| LightGBM of rotation inspire, recalibrated | recalibration | 0.3067 (0.2930–0.3209) | 0.814 | 0.0930 | +0.000 | 1.00 |
| LightGBM of rotation mimic, recalibrated | recalibration | 0.3067 (0.2928–0.3207) | 0.814 | 0.0930 | +0.000 | 1.00 |
| LightGBM of rotation eicu, recalibrated | recalibration | 0.3078 (0.2941–0.3222) | 0.812 | 0.0934 | +0.000 | 1.00 |
| kNN, local metric | metric + case base | 0.3225 (0.3078–0.3376) | 0.786 | 0.0974 | +0.083 | 1.08 |
| kNN, Euclidean | case base | 0.3295 (0.3148–0.3446) | 0.774 | 0.0996 | +0.108 | 1.12 |
| LightGBM, local | everything | 0.3086 (0.2948–0.3227) | 0.811 | 0.0934 | +0.014 | 1.03 |

## Primary: R of each transported metric

- rotation inspire (learned on the other two sites): 1.416 (90 % CI 1.034–2.071; equivalent)
- rotation mimic (learned on the other two sites): 1.691 (90 % CI 1.286–2.443; equivalent)
- rotation eicu (learned on the other two sites): 1.528 (90 % CI 1.149–2.204; equivalent)

## Secondary

- kNN inspire − recalibrated LightGBM inspire: +0.0129 (+0.0102 to +0.0159)
- kNN mimic − recalibrated LightGBM mimic: +0.0109 (+0.0083 to +0.0139)
- kNN eicu − recalibrated LightGBM eicu: +0.0110 (+0.0083 to +0.0140)
- local LightGBM − best transported kNN (mimic): -0.0091 (-0.0120 to -0.0065)

## Descriptive: vasopressor stratum (on in 12.8% of landmarks)

- rotation inspire: transport cost on -0.0125, off -0.0015; B0 (same-state neighbours) on the stratum -0.0007 (positive = B0 better), overall +0.0003 (B0 − kNN)
- rotation mimic: transport cost on -0.0131, off -0.0036; B0 (same-state neighbours) on the stratum +0.0001 (positive = B0 better), overall +0.0002 (B0 − kNN)
- rotation eicu: transport cost on -0.0093, off -0.0029; B0 (same-state neighbours) on the stratum +0.0022 (positive = B0 better), overall -0.0000 (B0 − kNN)

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
   "citl": 0.0992017379869745,
   "slope": 1.055474890812884
  },
  "mimic": {
   "citl": 0.1286637641433643,
   "slope": 1.0431393192036278
  },
  "eicu": {
   "citl": 0.105518634560519,
   "slope": 1.0512681548876834
  }
 }
}
```
