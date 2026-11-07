# Transport of the frozen hypo metrics to NWICU

Pre-registered: docs/prereg_nwicu.md. 2,001 patients, 30,833 landmarks, outcome 12.9%; five patient folds; 2,000 patient resamples.

| arm | refitted on NWICU | log-loss (95 % CI) | AUROC | Brier | CITL | slope |
|---|---|---|---|---|---|---|
| kNN, metric of rotation inspire | case base | 0.3193 (0.3047–0.3344) | 0.791 | 0.0960 | +0.102 | 1.06 |
| kNN, metric of rotation mimic | case base | 0.3188 (0.3042–0.3338) | 0.792 | 0.0958 | +0.087 | 1.00 |
| kNN, metric of rotation eicu | case base | 0.3180 (0.3036–0.3328) | 0.793 | 0.0954 | +0.072 | 1.04 |
| LightGBM of rotation inspire | nothing | 0.3419 (0.3307–0.3531) | 0.815 | 0.1042 | -0.793 | 1.14 |
| LightGBM of rotation mimic | nothing | 0.3186 (0.3066–0.3305) | 0.814 | 0.0964 | -0.476 | 1.09 |
| LightGBM of rotation eicu | nothing | 0.3531 (0.3423–0.3643) | 0.813 | 0.1085 | -0.894 | 1.13 |
| LightGBM of rotation inspire, recalibrated | recalibration | 0.3066 (0.2927–0.3207) | 0.814 | 0.0930 | +0.000 | 1.00 |
| LightGBM of rotation mimic, recalibrated | recalibration | 0.3068 (0.2929–0.3208) | 0.814 | 0.0930 | +0.000 | 1.00 |
| LightGBM of rotation eicu, recalibrated | recalibration | 0.3077 (0.2941–0.3220) | 0.812 | 0.0933 | +0.000 | 1.00 |
| kNN, local metric | metric + case base | 0.3225 (0.3082–0.3377) | 0.786 | 0.0972 | +0.088 | 1.07 |
| kNN, Euclidean | case base | 0.3295 (0.3148–0.3446) | 0.774 | 0.0996 | +0.108 | 1.12 |
| LightGBM, local | everything | 0.3086 (0.2948–0.3227) | 0.811 | 0.0934 | +0.014 | 1.03 |

## Primary: R of each transported metric

- rotation inspire (learned on the other two sites): 1.461 (90 % CI 1.103–2.078; equivalent)
- rotation mimic (learned on the other two sites): 1.534 (90 % CI 1.139–2.210; equivalent)
- rotation eicu (learned on the other two sites): 1.642 (90 % CI 1.249–2.338; equivalent)

## Secondary

- kNN inspire − recalibrated LightGBM inspire: +0.0127 (+0.0100 to +0.0156)
- kNN mimic − recalibrated LightGBM mimic: +0.0119 (+0.0092 to +0.0147)
- kNN eicu − recalibrated LightGBM eicu: +0.0104 (+0.0076 to +0.0131)
- local LightGBM − best transported kNN (eicu): -0.0095 (-0.0122 to -0.0068)

## Descriptive: vasopressor stratum (on in 12.8% of landmarks)

- rotation inspire: transport cost on -0.0082, off -0.0025
- rotation mimic: transport cost on -0.0080, off -0.0031
- rotation eicu: transport cost on -0.0114, off -0.0035

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
   "citl": 0.10182894287360979,
   "slope": 1.062090225502337
  },
  "mimic": {
   "citl": 0.08653928906708067,
   "slope": 1.0009713234683801
  },
  "eicu": {
   "citl": 0.0715532522088255,
   "slope": 1.044536047332127
  }
 }
}
```
