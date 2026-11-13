# A cosine kNN against the transported learned metric (post hoc, pre-specified)

Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-08. Same case base, k, neighbours and vote in every arm; only the metric changes. Δc = log-loss(cosine) − log-loss(NCA, transported): NCA beats cosine when its 95 % CI is above 0, cosine beats NCA when below. ΔAUROC = AUROC(NCA, transported) − AUROC(cosine). 2,000 patient resamples, shared by the arms of a target.

## Primary comparison

| target | NCA metric | Δc log-loss (95 % CI) | verdict | ΔAUROC (95 % CI) | verdict |
|---|---|---|---|---|---|
| inspire | NCA, transported | +0.0128 (+0.0112 to +0.0144) | NCA beats cosine | +0.0177 (+0.0143 to +0.0209) | NCA beats cosine |
| mimic | NCA, transported | +0.0090 (+0.0080 to +0.0100) | NCA beats cosine | +0.0079 (+0.0068 to +0.0090) | NCA beats cosine |
| eicu | NCA, transported | +0.0144 (+0.0136 to +0.0152) | NCA beats cosine | +0.0158 (+0.0146 to +0.0170) | NCA beats cosine |
| nwicu | NCA, transported (mimic+eicu) | +0.0101 (+0.0066 to +0.0136) | NCA beats cosine | +0.0173 (+0.0096 to +0.0254) | NCA beats cosine |
| nwicu | NCA, transported (inspire+eicu) | +0.0102 (+0.0067 to +0.0137) | NCA beats cosine | +0.0184 (+0.0109 to +0.0265) | NCA beats cosine |
| nwicu | NCA, transported (inspire+mimic) | +0.0110 (+0.0079 to +0.0145) | NCA beats cosine | +0.0189 (+0.0116 to +0.0265) | NCA beats cosine |

## Every arm

| target | arm | log-loss | AUROC | CITL | slope |
|---|---|---|---|---|---|
| inspire | NCA, transported | 0.3274 | 0.7791 | +0.023 | 0.996 |
| inspire | NCA, local | 0.3277 | 0.7793 | +0.032 | 1.012 |
| inspire | Euclidean | 0.3382 | 0.7650 | +0.066 | 1.185 |
| inspire | cosine (neutral axes) | 0.3402 | 0.7614 | -0.112 | 1.124 |
| inspire | cosine on the NCA embedding | 0.3291 | 0.7792 | -0.113 | 1.076 |
| mimic | NCA, transported | 0.5048 | 0.7816 | -0.002 | 1.016 |
| mimic | NCA, local | 0.5036 | 0.7830 | -0.003 | 1.027 |
| mimic | Euclidean | 0.5140 | 0.7732 | -0.022 | 1.111 |
| mimic | cosine (neutral axes) | 0.5138 | 0.7737 | -0.104 | 1.071 |
| mimic | cosine on the NCA embedding | 0.5049 | 0.7817 | -0.047 | 1.036 |
| eicu | NCA, transported | 0.3966 | 0.7992 | +0.013 | 1.047 |
| eicu | NCA, local | 0.3961 | 0.7995 | +0.015 | 1.048 |
| eicu | Euclidean | 0.4101 | 0.7844 | +0.002 | 1.156 |
| eicu | cosine (neutral axes) | 0.4110 | 0.7834 | -0.105 | 1.137 |
| eicu | cosine on the NCA embedding | 0.3978 | 0.7989 | -0.094 | 1.005 |
| nwicu | Euclidean | 0.3295 | 0.7738 | +0.108 | 1.120 |
| nwicu | NCA, local | 0.3222 | 0.7860 | +0.085 | 1.070 |
| nwicu | NCA, transported (mimic+eicu) | 0.3192 | 0.7911 | +0.086 | 1.045 |
| nwicu | cosine on the NCA embedding (mimic+eicu) | 0.3212 | 0.7885 | -0.095 | 0.995 |
| nwicu | NCA, transported (inspire+eicu) | 0.3191 | 0.7922 | +0.107 | 1.039 |
| nwicu | cosine on the NCA embedding (inspire+eicu) | 0.3214 | 0.7883 | -0.045 | 0.996 |
| nwicu | NCA, transported (inspire+mimic) | 0.3183 | 0.7926 | +0.094 | 1.056 |
| nwicu | cosine on the NCA embedding (inspire+mimic) | 0.3209 | 0.7892 | -0.094 | 1.000 |
| nwicu | cosine (neutral axes) | 0.3293 | 0.7738 | -0.076 | 1.137 |

## Secondary contrasts (descriptive)

| target | contrast | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|---|
| inspire | cos_neutral − euclidean | +0.0020 (+0.0012 to +0.0028) | -0.0035 (-0.0048 to -0.0021) |
| inspire | cos_nca − nca_transported | +0.0017 (+0.0008 to +0.0025) | +0.0001 (-0.0013 to +0.0015) |
| mimic | cos_neutral − euclidean | -0.0002 (-0.0006 to +0.0002) | +0.0005 (+0.0001 to +0.0010) |
| mimic | cos_nca − nca_transported | +0.0001 (-0.0002 to +0.0005) | +0.0001 (-0.0003 to +0.0005) |
| eicu | cos_neutral − euclidean | +0.0009 (+0.0005 to +0.0013) | -0.0009 (-0.0014 to -0.0004) |
| eicu | cos_nca − nca_transported | +0.0012 (+0.0008 to +0.0016) | -0.0003 (-0.0008 to +0.0002) |
| nwicu | cos_neutral − euclidean | -0.0002 (-0.0020 to +0.0016) | -0.0001 (-0.0032 to +0.0032) |
| nwicu | cos_nca_inspire − nca_transported_inspire | +0.0020 (+0.0000 to +0.0038) | -0.0026 (-0.0056 to +0.0004) |
| nwicu | cos_nca_mimic − nca_transported_mimic | +0.0023 (+0.0004 to +0.0040) | -0.0039 (-0.0071 to -0.0008) |
| nwicu | cos_nca_eicu − nca_transported_eicu | +0.0026 (+0.0005 to +0.0044) | -0.0034 (-0.0065 to -0.0002) |

**Rule (pre-specified), log-loss:** NCA beats cosine in 6 of 6 comparisons; cosine beats NCA in 0; no difference shown in 0.
