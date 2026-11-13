# Transport equivalence — `hypo`: hypotension in the next 6 h (external stage, five folds) (exploratory)

Criterion fixed with the point estimates already known (DECISIONS 2026-11-05), by the bioequivalence convention: equivalent when the lower limit of the 90 % CI of R is ≥ 0.80. R = (L_euclidean − L_transported) / (L_euclidean − L_local). Not evaluable when the local gain over the Euclidean kNN has a 95 % CI including 0 or is below 0.005 nats. 2,000 patient-level resamples.

| site | rows | R | 90 % CI | 95 % CI | local gain (nats) | cost (nats) | verdict |
|---|---|---|---|---|---|---|---|
| inspire | 148,293 | 0.908 | 0.837–0.985 | 0.822–1.002 | 0.0116 (0.0102–0.0132) | +0.0011 (-0.0000 to +0.0021) | equivalent |
| mimic | 387,158 | 0.898 | 0.852–0.945 | 0.844–0.952 | 0.0112 (0.0103–0.0121) | +0.0011 (+0.0005 to +0.0018) | equivalent |
| mimic, vaso_on = 0 (79%) | 305,427 | 0.895 | 0.845–0.952 | 0.836–0.961 | 0.0109 (0.0099–0.0119) | +0.0011 (+0.0004 to +0.0018) | equivalent |
| mimic, vaso_on = 1 (21%) | 81,731 | 0.905 | 0.820–0.994 | 0.803–1.011 | 0.0124 (0.0104–0.0144) | +0.0012 (-0.0001 to +0.0025) | equivalent |
| eicu | 466,384 | 0.970 | 0.940–1.000 | 0.935–1.006 | 0.0145 (0.0137–0.0153) | +0.0004 (-0.0001 to +0.0010) | equivalent |
| eicu, vaso_on = 0 (93%) | 433,173 | 0.966 | 0.931–1.002 | 0.924–1.008 | 0.0134 (0.0126–0.0142) | +0.0005 (-0.0001 to +0.0010) | equivalent |
| eicu, vaso_on = 1 (7%) | 33,211 | 0.993 | 0.925–1.066 | 0.914–1.079 | 0.0289 (0.0249–0.0324) | +0.0002 (-0.0022 to +0.0025) | equivalent |
