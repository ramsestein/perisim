# Transport equivalence — `hypo`: hypotension in the next 6 h (external stage, five folds) (exploratory)

Criterion fixed with the point estimates already known (DECISIONS 2026-11-05), by the bioequivalence convention: equivalent when the lower limit of the 90 % CI of R is ≥ 0.80. R = (L_euclidean − L_transported) / (L_euclidean − L_local). Not evaluable when the local gain over the Euclidean kNN has a 95 % CI including 0 or is below 0.005 nats. 2,000 patient-level resamples.

| site | rows | R | 90 % CI | 95 % CI | local gain (nats) | cost (nats) | verdict |
|---|---|---|---|---|---|---|---|
| inspire | 148,293 | 1.031 | 0.944–1.122 | 0.926–1.138 | 0.0105 (0.0090–0.0120) | -0.0003 (-0.0014 to +0.0008) | equivalent |
| mimic | 387,158 | 0.887 | 0.834–0.943 | 0.826–0.952 | 0.0104 (0.0095–0.0113) | +0.0012 (+0.0005 to +0.0018) | equivalent |
| mimic, vaso_on = 0 (79%) | 305,427 | 0.882 | 0.823–0.944 | 0.813–0.953 | 0.0102 (0.0092–0.0112) | +0.0012 (+0.0005 to +0.0019) | equivalent |
| mimic, vaso_on = 1 (21%) | 81,731 | 0.904 | 0.802–1.010 | 0.783–1.031 | 0.0110 (0.0090–0.0130) | +0.0011 (-0.0003 to +0.0025) | equivalent |
| eicu | 466,384 | 0.964 | 0.933–0.998 | 0.925–1.006 | 0.0140 (0.0132–0.0148) | +0.0005 (-0.0001 to +0.0011) | equivalent |
| eicu, vaso_on = 0 (93%) | 433,173 | 0.966 | 0.929–1.004 | 0.925–1.013 | 0.0129 (0.0120–0.0136) | +0.0004 (-0.0002 to +0.0010) | equivalent |
| eicu, vaso_on = 1 (7%) | 33,211 | 0.951 | 0.885–1.021 | 0.873–1.034 | 0.0289 (0.0247–0.0324) | +0.0014 (-0.0009 to +0.0037) | equivalent |
