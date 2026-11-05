# Transport equivalence — T2: post-ICU stay (transport stage, fixed test set) (exploratory)

Criterion fixed with the point estimates already known (DECISIONS 2026-11-05), by the bioequivalence convention: equivalent when the lower limit of the 90 % CI of R is ≥ 0.80. R = (L_euclidean − L_transported) / (L_euclidean − L_local). Not evaluable when the local gain over the Euclidean kNN has a 95 % CI including 0 or is below 0.005 nats. 2,000 patient-level resamples.

| site | rows | R | 90 % CI | 95 % CI | local gain (nats) | cost (nats) | verdict |
|---|---|---|---|---|---|---|---|
| mimic | 4,963 | 0.494 | 0.288–0.681 | 0.244–0.717 | 0.0416 (0.0309–0.0529) | +0.0211 (+0.0109 to +0.0315) | not equivalent |
| eicu | 4,725 | 0.724 | 0.587–0.858 | 0.563–0.890 | 0.0582 (0.0469–0.0690) | +0.0161 (+0.0060 to +0.0260) | not equivalent |
