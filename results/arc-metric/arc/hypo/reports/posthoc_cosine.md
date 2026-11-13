# A cosine kNN against the transported learned metric (post hoc, pre-specified)

Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-08. Same case base, k, neighbours and vote in every arm; only the metric changes. Δc = log-loss(cosine) − log-loss(NCA, transported): NCA beats cosine when its 95 % CI is above 0, cosine beats NCA when below. ΔAUROC = AUROC(NCA, transported) − AUROC(cosine). 2,000 patient resamples, shared by the arms of a target.

## Primary comparison

| target | NCA metric | Δc log-loss (95 % CI) | verdict | ΔAUROC (95 % CI) | verdict |
|---|---|---|---|---|---|
| inspire | NCA, transported | +0.0126 (+0.0111 to +0.0141) | NCA beats cosine | +0.0169 (+0.0137 to +0.0201) | NCA beats cosine |
| mimic | NCA, transported | +0.0098 (+0.0089 to +0.0108) | NCA beats cosine | +0.0089 (+0.0079 to +0.0099) | NCA beats cosine |
| eicu | NCA, transported | +0.0150 (+0.0142 to +0.0159) | NCA beats cosine | +0.0166 (+0.0154 to +0.0179) | NCA beats cosine |
| nwicu | NCA, transported (mimic+eicu) | +0.0097 (+0.0062 to +0.0134) | NCA beats cosine | +0.0168 (+0.0090 to +0.0249) | NCA beats cosine |
| nwicu | NCA, transported (inspire+eicu) | +0.0117 (+0.0082 to +0.0152) | NCA beats cosine | +0.0203 (+0.0129 to +0.0281) | NCA beats cosine |
| nwicu | NCA, transported (inspire+mimic) | +0.0105 (+0.0071 to +0.0140) | NCA beats cosine | +0.0188 (+0.0113 to +0.0263) | NCA beats cosine |

## Every arm

| target | arm | log-loss | AUROC | CITL | slope |
|---|---|---|---|---|---|
| inspire | NCA, transported | 0.3276 | 0.7783 | +0.025 | 1.053 |
| inspire | NCA, local | 0.3265 | 0.7807 | +0.050 | 1.032 |
| inspire | Euclidean | 0.3382 | 0.7650 | +0.066 | 1.185 |
| inspire | cosine (neutral axes) | 0.3402 | 0.7614 | -0.112 | 1.124 |
| inspire | cosine on the NCA embedding | 0.3291 | 0.7785 | -0.110 | 1.100 |
| mimic | NCA, transported | 0.5040 | 0.7826 | +0.012 | 1.015 |
| mimic | NCA, local | 0.5028 | 0.7838 | +0.001 | 1.031 |
| mimic | Euclidean | 0.5140 | 0.7732 | -0.022 | 1.111 |
| mimic | cosine (neutral axes) | 0.5138 | 0.7737 | -0.104 | 1.071 |
| mimic | cosine on the NCA embedding | 0.5056 | 0.7821 | -0.087 | 1.008 |
| eicu | NCA, transported | 0.3960 | 0.8001 | +0.018 | 1.051 |
| eicu | NCA, local | 0.3955 | 0.8005 | +0.021 | 1.052 |
| eicu | Euclidean | 0.4101 | 0.7844 | +0.002 | 1.156 |
| eicu | cosine (neutral axes) | 0.4110 | 0.7834 | -0.105 | 1.137 |
| eicu | cosine on the NCA embedding | 0.3984 | 0.7987 | -0.103 | 0.998 |
| nwicu | Euclidean | 0.3295 | 0.7738 | +0.108 | 1.120 |
| nwicu | NCA, local | 0.3225 | 0.7860 | +0.083 | 1.078 |
| nwicu | NCA, transported (mimic+eicu) | 0.3196 | 0.7905 | +0.099 | 1.055 |
| nwicu | cosine on the NCA embedding (mimic+eicu) | 0.3229 | 0.7859 | -0.104 | 0.982 |
| nwicu | NCA, transported (inspire+eicu) | 0.3177 | 0.7940 | +0.129 | 1.043 |
| nwicu | cosine on the NCA embedding (inspire+eicu) | 0.3210 | 0.7889 | -0.092 | 0.983 |
| nwicu | NCA, transported (inspire+mimic) | 0.3188 | 0.7925 | +0.106 | 1.051 |
| nwicu | cosine on the NCA embedding (inspire+mimic) | 0.3214 | 0.7883 | -0.095 | 0.987 |
| nwicu | cosine (neutral axes) | 0.3293 | 0.7738 | -0.076 | 1.137 |

## Secondary contrasts (descriptive)

| target | contrast | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |
|---|---|---|---|
| inspire | cos_neutral − euclidean | +0.0020 (+0.0012 to +0.0028) | -0.0035 (-0.0048 to -0.0021) |
| inspire | cos_nca − nca_transported | +0.0015 (+0.0007 to +0.0023) | +0.0001 (-0.0013 to +0.0016) |
| mimic | cos_neutral − euclidean | -0.0002 (-0.0006 to +0.0002) | +0.0005 (+0.0001 to +0.0010) |
| mimic | cos_nca − nca_transported | +0.0016 (+0.0011 to +0.0021) | -0.0005 (-0.0010 to -0.0001) |
| eicu | cos_neutral − euclidean | +0.0009 (+0.0005 to +0.0013) | -0.0009 (-0.0014 to -0.0004) |
| eicu | cos_nca − nca_transported | +0.0024 (+0.0020 to +0.0029) | -0.0013 (-0.0019 to -0.0008) |
| nwicu | cos_neutral − euclidean | -0.0002 (-0.0020 to +0.0016) | -0.0001 (-0.0032 to +0.0032) |
| nwicu | cos_nca_inspire − nca_transported_inspire | +0.0032 (+0.0012 to +0.0052) | -0.0046 (-0.0080 to -0.0012) |
| nwicu | cos_nca_mimic − nca_transported_mimic | +0.0033 (+0.0009 to +0.0054) | -0.0052 (-0.0085 to -0.0015) |
| nwicu | cos_nca_eicu − nca_transported_eicu | +0.0026 (+0.0005 to +0.0046) | -0.0042 (-0.0074 to -0.0007) |

**Rule (pre-specified), log-loss:** NCA beats cosine in 6 of 6 comparisons; cosine beats NCA in 0; no difference shown in 0.
