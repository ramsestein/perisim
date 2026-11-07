# Robustness of the neighbour-alignment analysis (hypo §15.3)

Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. Fold 0; one query per test patient; 1000 query (= patient) resamples. Inputs: posthoc_alignment's caches only.

## inspire (2,288 queries)

### Case base, by k

| k | transported–local | halves | n500 pair | Euclidean–local | transported–local − halves (95 % CI) |
|---|---|---|---|---|---|
| 5 | 0.049 | 0.055 | 0.027 | 0.049 | -0.006 (-0.012 to -0.001) |
| 10 | 0.057 | 0.068 | 0.032 | 0.059 | -0.012 (-0.016 to -0.007) |
| 20 | 0.067 | 0.084 | 0.036 | 0.071 | -0.017 (-0.021 to -0.013) |
| 50 | 0.080 | 0.109 | 0.045 | 0.087 | -0.030 (-0.033 to -0.027) |

### Within a sample (Huh et al.), 1,000 items, by k (raw / chance-calibrated)

| k | chance | transported–local | halves | n500 pair | Euclidean–local |
|---|---|---|---|---|---|
| 5 | 0.005 | 0.166 / 0.162 | 0.220 / 0.216 | 0.099 / 0.094 | 0.169 / 0.165 |
| 10 | 0.010 | 0.213 / 0.205 | 0.279 / 0.271 | 0.126 / 0.117 | 0.208 / 0.200 |
| 20 | 0.020 | 0.268 / 0.253 | 0.348 / 0.335 | 0.161 / 0.144 | 0.249 / 0.234 |
| 50 | 0.050 | 0.373 / 0.340 | 0.459 / 0.431 | 0.230 / 0.189 | 0.326 / 0.291 |
| 100 | 0.100 | 0.486 / 0.429 | 0.566 / 0.518 | 0.308 / 0.231 | 0.414 / 0.349 |

### Set size (within a sample)

| items | k | transported–local | halves | n500 pair |
|---|---|---|---|---|
| 500 | 10 | 0.267 | 0.356 | 0.164 |
| 500 | 5 | 0.205 | 0.281 | 0.120 |
| 1000 | 10 | 0.213 | 0.279 | 0.126 |
| 1000 | 10 | 0.213 | 0.279 | 0.126 |
| 2000 | 10 | 0.168 | 0.227 | 0.100 |
| 2000 | 20 | 0.213 | 0.282 | 0.127 |

### Neighbours shared against agreement of the predictions (case base, k = 50)

- reference: 276 metric pairs without the transported metric; r_logit = 0.746 + 0.254 × overlap (residual SD 0.026; overlaps 0.03–0.81)
- transported–local: overlap 0.080, r_logit 0.781; predicted 0.766 (95 % PI 0.715–0.817) → **generic: within what non-transported pairs with that overlap do**
- local–local pairs: n500_a|n500_b overlap 0.045, r 0.745; n1000_a|n1000_b overlap 0.052, r 0.728; n2000_a|n2000_b overlap 0.066, r 0.809; n4000_a|n4000_b overlap 0.096, r 0.823; nhalf_a|nhalf_b overlap 0.109, r 0.793

## mimic (4,934 queries)

### Case base, by k

| k | transported–local | halves | n500 pair | Euclidean–local | transported–local − halves (95 % CI) |
|---|---|---|---|---|---|
| 5 | 0.095 | 0.097 | 0.019 | 0.028 | -0.003 (-0.007 to +0.002) |
| 10 | 0.109 | 0.114 | 0.022 | 0.034 | -0.005 (-0.009 to -0.001) |
| 20 | 0.123 | 0.132 | 0.026 | 0.039 | -0.010 (-0.013 to -0.006) |
| 50 | 0.142 | 0.166 | 0.032 | 0.045 | -0.024 (-0.027 to -0.021) |

### Within a sample (Huh et al.), 1,000 items, by k (raw / chance-calibrated)

| k | chance | transported–local | halves | n500 pair | Euclidean–local |
|---|---|---|---|---|---|
| 5 | 0.005 | 0.269 / 0.265 | 0.385 / 0.382 | 0.098 / 0.094 | 0.116 / 0.111 |
| 10 | 0.010 | 0.310 / 0.303 | 0.456 / 0.450 | 0.129 / 0.120 | 0.142 / 0.134 |
| 20 | 0.020 | 0.362 / 0.349 | 0.528 / 0.519 | 0.171 / 0.154 | 0.183 / 0.166 |
| 50 | 0.050 | 0.447 / 0.417 | 0.616 / 0.596 | 0.249 / 0.209 | 0.261 / 0.222 |
| 100 | 0.100 | 0.528 / 0.476 | 0.688 / 0.654 | 0.342 / 0.268 | 0.335 / 0.261 |

### Set size (within a sample)

| items | k | transported–local | halves | n500 pair |
|---|---|---|---|---|
| 500 | 10 | 0.358 | 0.517 | 0.166 |
| 500 | 5 | 0.302 | 0.452 | 0.131 |
| 1000 | 10 | 0.310 | 0.456 | 0.129 |
| 1000 | 10 | 0.310 | 0.456 | 0.129 |
| 2000 | 10 | 0.270 | 0.395 | 0.101 |
| 2000 | 20 | 0.314 | 0.463 | 0.129 |
| 4000 | 10 | 0.236 | 0.336 | 0.079 |
| 4000 | 40 | 0.316 | 0.464 | 0.131 |

### Neighbours shared against agreement of the predictions (case base, k = 50)

- reference: 276 metric pairs without the transported metric; r_logit = 0.846 + 0.291 × overlap (residual SD 0.040; overlaps 0.02–0.81)
- transported–local: overlap 0.142, r_logit 0.906; predicted 0.888 (95 % PI 0.809–0.966) → **generic: within what non-transported pairs with that overlap do**
- local–local pairs: n500_a|n500_b overlap 0.032, r 0.870; n1000_a|n1000_b overlap 0.082, r 0.902; n2000_a|n2000_b overlap 0.086, r 0.904; n4000_a|n4000_b overlap 0.107, r 0.915; nhalf_a|nhalf_b overlap 0.166, r 0.922

## eicu (6,431 queries)

### Case base, by k

| k | transported–local | halves | n500 pair | Euclidean–local | transported–local − halves (95 % CI) |
|---|---|---|---|---|---|
| 5 | 0.090 | 0.122 | 0.021 | 0.018 | -0.032 (-0.037 to -0.027) |
| 10 | 0.105 | 0.148 | 0.025 | 0.022 | -0.043 (-0.047 to -0.038) |
| 20 | 0.121 | 0.176 | 0.030 | 0.027 | -0.055 (-0.059 to -0.052) |
| 50 | 0.144 | 0.217 | 0.039 | 0.035 | -0.073 (-0.076 to -0.070) |

### Within a sample (Huh et al.), 1,000 items, by k (raw / chance-calibrated)

| k | chance | transported–local | halves | n500 pair | Euclidean–local |
|---|---|---|---|---|---|
| 5 | 0.005 | 0.309 / 0.306 | 0.427 / 0.424 | 0.127 / 0.123 | 0.113 / 0.109 |
| 10 | 0.010 | 0.367 / 0.361 | 0.482 / 0.477 | 0.168 / 0.159 | 0.142 / 0.134 |
| 20 | 0.020 | 0.433 / 0.422 | 0.539 / 0.530 | 0.218 / 0.202 | 0.179 / 0.163 |
| 50 | 0.050 | 0.548 / 0.524 | 0.628 / 0.608 | 0.305 / 0.269 | 0.248 / 0.208 |
| 100 | 0.100 | 0.652 / 0.614 | 0.711 / 0.679 | 0.399 / 0.332 | 0.324 / 0.248 |

### Set size (within a sample)

| items | k | transported–local | halves | n500 pair |
|---|---|---|---|---|
| 500 | 10 | 0.427 | 0.537 | 0.214 |
| 500 | 5 | 0.355 | 0.475 | 0.162 |
| 1000 | 10 | 0.367 | 0.482 | 0.168 |
| 1000 | 10 | 0.367 | 0.482 | 0.168 |
| 2000 | 10 | 0.317 | 0.438 | 0.137 |
| 2000 | 20 | 0.373 | 0.489 | 0.172 |
| 4000 | 10 | 0.272 | 0.393 | 0.110 |
| 4000 | 40 | 0.373 | 0.492 | 0.174 |

### Neighbours shared against agreement of the predictions (case base, k = 50)

- reference: 276 metric pairs without the transported metric; r_logit = 0.833 + 0.260 × overlap (residual SD 0.026; overlaps 0.02–0.71)
- transported–local: overlap 0.144, r_logit 0.895; predicted 0.871 (95 % PI 0.819–0.923) → **generic: within what non-transported pairs with that overlap do**
- local–local pairs: n500_a|n500_b overlap 0.039, r 0.843; n1000_a|n1000_b overlap 0.042, r 0.855; n2000_a|n2000_b overlap 0.077, r 0.877; n4000_a|n4000_b overlap 0.153, r 0.896; nhalf_a|nhalf_b overlap 0.217, r 0.903

## Rule (pre-specified)

```
{
 "C1_convergence_at_every_setting": true,
 "C2_transported_below_halves_case_base_every_k": false,
 "C2_transported_below_halves_within_every_setting": true,
 "C3": {
  "inspire": {
   "inspire_not_closer": false
  },
  "mimic": {
   "closer": true
  },
  "eicu": {
   "closer": true
  }
 },
 "section_15_3_robust": false,
 "R2": {
  "inspire": "generic: within what non-transported pairs with that overlap do",
  "mimic": "generic: within what non-transported pairs with that overlap do",
  "eicu": "generic: within what non-transported pairs with that overlap do"
 }
}
```
