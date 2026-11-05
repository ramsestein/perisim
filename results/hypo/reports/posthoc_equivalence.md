# Transport equivalence — `hypo`: hypotension in the next 6 h (external stage, five folds) (exploratory)

Criterion fixed with the point estimates already known (DECISIONS 2026-11-05), by the bioequivalence convention: equivalent when the lower limit of the 90 % CI of R is ≥ 0.80. R = (L_euclidean − L_transported) / (L_euclidean − L_local). Not evaluable when the local gain over the Euclidean kNN has a 95 % CI including 0 or is below 0.005 nats. 2,000 patient-level resamples.

| site | rows | R | 90 % CI | 95 % CI | local gain (nats) | cost (nats) | verdict |
|---|---|---|---|---|---|---|---|
| inspire | 148,293 | 0.911 | 0.830–0.988 | 0.817–1.006 | 0.0116 (0.0102–0.0130) | +0.0010 (-0.0001 to +0.0022) | equivalent |
| mimic | 387,159 | 0.738 | 0.682–0.796 | 0.675–0.805 | 0.0107 (0.0097–0.0116) | +0.0028 (+0.0020 to +0.0035) | not equivalent |
| mimic, vaso_on = 0 (79%) | 305,428 | 0.837 | 0.773–0.903 | 0.763–0.918 | 0.0102 (0.0092–0.0112) | +0.0017 (+0.0008 to +0.0025) | not equivalent |
| mimic, vaso_on = 1 (21%) | 81,731 | 0.437 | 0.278–0.577 | 0.252–0.598 | 0.0124 (0.0103–0.0146) | +0.0070 (+0.0052 to +0.0088) | not equivalent |
| eicu | 466,385 | 0.963 | 0.931–0.998 | 0.923–1.004 | 0.0140 (0.0132–0.0148) | +0.0005 (-0.0000 to +0.0011) | equivalent |
| eicu, vaso_on = 0 (92%) | 431,041 | 0.976 | 0.935–1.016 | 0.928–1.025 | 0.0126 (0.0117–0.0134) | +0.0003 (-0.0003 to +0.0009) | equivalent |
| eicu, vaso_on = 1 (8%) | 35,344 | 0.900 | 0.839–0.961 | 0.827–0.972 | 0.0314 (0.0276–0.0353) | +0.0031 (+0.0008 to +0.0056) | equivalent |
