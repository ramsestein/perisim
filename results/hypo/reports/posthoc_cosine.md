# A cosine kNN against the transported learned metric (post hoc, pre-specified)

Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-08. Same case base, k, neighbours and vote in every arm; only the metric changes. Δc = log-loss(cosine) − log-loss(NCA, transported): NCA beats cosine when its 95 % CI is above 0, cosine beats NCA when below. ΔAUROC = AUROC(NCA, transported) − AUROC(cosine). 2,000 patient resamples, shared by the arms of a target.

## Primary comparison

| target | NCA metric | Δc log-loss (95 % CI) | verdict | ΔAUROC (95 % CI) | verdict |
|---|---|---|---|---|---|
| inspire | NCA, transported | +0.0128 (+0.0112 to +0.0144) | NCA beats cosine | +0.0177 (+0.0145 to +0.0209) | NCA beats cosine |
| mimic | NCA, transported | +0.0077 (+0.0067 to +0.0088) | NCA beats cosine | +0.0067 (+0.0056 to +0.0079) | NCA beats cosine |
| eicu | NCA, transported | +0.0146 (+0.0138 to +0.0155) | NCA beats cosine | +0.0159 (+0.0147 to +0.0171) | NCA beats cosine |
| nwicu | NCA, transported (mimic+eicu) | +0.0100 (+0.0066 to +0.0137) | NCA beats cosine | +0.0171 (+0.0093 to +0.0251) | NCA beats cosine |
| nwicu | NCA, transported (inspire+eicu) | +0.0106 (+0.0069 to +0.0144) | NCA beats cosine | +0.0179 (+0.0102 to +0.0256) | NCA beats cosine |
| nwicu | NCA, transported (inspire+mimic) | +0.0113 (+0.0079 to +0.0151) | NCA beats cosine | +0.0190 (+0.0116 to +0.0270) | NCA beats cosine |

## Every arm

| target | arm | log-loss | AUROC | CITL | slope |
|---|---|---|---|---|---|
| inspire | NCA, transported | 0.3276 | 0.7788 | +0.023 | 1.019 |
| inspire | NCA, local | 0.3265 | 0.7818 | +0.033 | 1.064 |
| inspire | Euclidean | 0.3381 | 0.7647 | +0.062 | 1.187 |
| inspire | MI-weighted, transported | 0.3324 | 0.7696 | +0.021 | 1.061 |
| inspire | cosine (neutral axes) | 0.3404 | 0.7611 | -0.115 | 1.122 |
| inspire | cosine on the NCA embedding | 0.3292 | 0.7792 | -0.120 | 1.080 |
| mimic | NCA, transported | 0.5059 | 0.7806 | +0.021 | 1.011 |
| mimic | NCA, local | 0.5031 | 0.7836 | -0.002 | 1.012 |
| mimic | Euclidean | 0.5138 | 0.7734 | -0.022 | 1.112 |
| mimic | MI-weighted, transported | 0.5051 | 0.7815 | +0.012 | 1.005 |
| mimic | cosine (neutral axes) | 0.5136 | 0.7739 | -0.104 | 1.065 |
| mimic | cosine on the NCA embedding | 0.5059 | 0.7807 | -0.049 | 1.011 |
| eicu | NCA, transported | 0.3965 | 0.7992 | +0.008 | 1.025 |
| eicu | NCA, local | 0.3960 | 0.7998 | +0.008 | 1.046 |
| eicu | Euclidean | 0.4100 | 0.7845 | -0.003 | 1.171 |
| eicu | MI-weighted, transported | 0.4005 | 0.7941 | -0.006 | 1.065 |
| eicu | cosine (neutral axes) | 0.4112 | 0.7833 | -0.112 | 1.124 |
| eicu | cosine on the NCA embedding | 0.3981 | 0.7987 | -0.110 | 0.995 |
| nwicu | Euclidean | 0.3295 | 0.7738 | +0.108 | 1.120 |
| nwicu | NCA, local | 0.3225 | 0.7857 | +0.088 | 1.066 |
| nwicu | NCA, transported (mimic+eicu) | 0.3193 | 0.7908 | +0.102 | 1.062 |
| nwicu | cosine on the NCA embedding (mimic+eicu) | 0.3213 | 0.7890 | -0.106 | 0.996 |
| nwicu | NCA, transported (inspire+eicu) | 0.3188 | 0.7916 | +0.087 | 1.001 |
| nwicu | cosine on the NCA embedding (inspire+eicu) | 0.3217 | 0.7873 | -0.102 | 0.962 |
| nwicu | NCA, transported (inspire+mimic) | 0.3180 | 0.7927 | +0.072 | 1.045 |
| nwicu | cosine on the NCA embedding (inspire+mimic) | 0.3225 | 0.7883 | -0.138 | 1.098 |
| nwicu | cosine (neutral axes) | 0.3293 | 0.7738 | -0.076 | 1.137 |

## Secondary contrasts (descriptive)

| target | contrast | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|---|
| inspire | cos_neutral − euclidean | +0.0023 (+0.0015 to +0.0030) | -0.0036 (-0.0050 to -0.0022) |
| inspire | cos_nca − nca_transported | +0.0016 (+0.0008 to +0.0024) | +0.0004 (-0.0010 to +0.0018) |
| inspire | cos_neutral − mi_transported | +0.0080 (+0.0066 to +0.0094) | -0.0085 (-0.0116 to -0.0056) |
| mimic | cos_neutral − euclidean | -0.0001 (-0.0006 to +0.0003) | +0.0005 (+0.0001 to +0.0009) |
| mimic | cos_nca − nca_transported | +0.0000 (-0.0004 to +0.0004) | +0.0001 (-0.0003 to +0.0006) |
| mimic | cos_neutral − mi_transported | +0.0085 (+0.0076 to +0.0095) | -0.0076 (-0.0086 to -0.0066) |
| eicu | cos_neutral − euclidean | +0.0011 (+0.0008 to +0.0015) | -0.0011 (-0.0016 to -0.0006) |
| eicu | cos_nca − nca_transported | +0.0016 (+0.0012 to +0.0021) | -0.0006 (-0.0011 to -0.0001) |
| eicu | cos_neutral − mi_transported | +0.0106 (+0.0098 to +0.0115) | -0.0107 (-0.0119 to -0.0095) |
| nwicu | cos_neutral − euclidean | -0.0002 (-0.0020 to +0.0016) | -0.0001 (-0.0032 to +0.0032) |
| nwicu | cos_nca_inspire − nca_transported_inspire | +0.0020 (-0.0002 to +0.0041) | -0.0018 (-0.0053 to +0.0017) |
| nwicu | cos_nca_mimic − nca_transported_mimic | +0.0029 (+0.0006 to +0.0050) | -0.0043 (-0.0079 to -0.0007) |
| nwicu | cos_nca_eicu − nca_transported_eicu | +0.0045 (+0.0023 to +0.0065) | -0.0044 (-0.0079 to -0.0011) |

**Rule (pre-specified), log-loss:** NCA beats cosine in 6 of 6 comparisons; cosine beats NCA in 0; no difference shown in 0.
