# Tier 2: PRAM's setting rebuilt (MIMIC-IV → eICU-CRD)

Pre-registered: docs/prereg_pram_factors.md (OSF). Δ = AUROC(NCA arm) − AUROC(COS arm), same vote and use; 2,000 stay resamples. Home levels: mixture with the frozen logistic model, PRAM's vote 1/(d + 1e-6), NCA-full transferred.

## R0: replication of PRAM's Table S13

- aki: Δ(NCA-full transferred − cosine) = -0.0018 (-0.0033 to -0.0003); PRAM: -0.0092
- death: Δ(NCA-full transferred − cosine) = -0.0041 (-0.0057 to -0.0024); PRAM: -0.0090
- reproduced (point estimates): **yes**; with confidence: **yes**

## Fidelity (descriptive)

| outcome | quantity | ours | PRAM |
|---|---|---|---|
| aki | prevalence in eICU | 17.6% | 6.1% |
| aki | frozen logistic AUROC in eICU | 0.744 | 0.690 |
| aki | gain over frozen logistic, COS mixture | +0.0224 (+0.0204 to +0.0245) | +0.0197 |
| aki | gain over frozen logistic, MI mixture | +0.0236 (+0.0215 to +0.0257) | +0.0216 |
| aki | gain over frozen logistic, NCA-full mixture | +0.0206 (+0.0187 to +0.0225) | +0.0105 |
| aki | gain over frozen logistic, NCA-full, target bank mixture | +0.0187 (+0.0169 to +0.0206) | +0.0131 |
| death | prevalence in eICU | 6.4% | 4.6% |
| death | frozen logistic AUROC in eICU | 0.848 | 0.774 |
| death | gain over frozen logistic, COS mixture | +0.0115 (+0.0097 to +0.0134) | +0.0358 |
| death | gain over frozen logistic, MI mixture | +0.0112 (+0.0094 to +0.0131) | +0.0367 |
| death | gain over frozen logistic, NCA-full mixture | +0.0075 (+0.0061 to +0.0089) | +0.0268 |
| death | gain over frozen logistic, NCA-full, target bank mixture | +0.0079 (+0.0065 to +0.0095) | +0.0262 |

## Rule (pre-specified)

| outcome | home Δ (verdict) | switch | Δ after switch (verdict) | sufficient | effect (95 % CI) |
|---|---|---|---|---|---|
| aki | -0.0018 (-0.0033 to -0.0003) (cosine beats NCA) | U | +0.0014 (-0.0015 to +0.0043) (no difference shown) | **yes** | +0.0032 (+0.0014 to +0.0049) |
|  |  | V | -0.0017 (-0.0032 to -0.0001) (cosine beats NCA) | no | +0.0001 (-0.0001 to +0.0004) |
|  |  | I | +0.0066 (+0.0044 to +0.0087) (NCA beats cosine) | **yes** | +0.0084 (+0.0062 to +0.0104) |
|  |  | UV | +0.0014 (-0.0016 to +0.0044) (no difference shown) | **yes** |  |
|  |  | UI | +0.0343 (+0.0303 to +0.0383) (NCA beats cosine) | **yes** |  |
|  |  | VI | +0.0061 (+0.0040 to +0.0082) (NCA beats cosine) | **yes** |  |
|  |  | UVI | +0.0342 (+0.0303 to +0.0382) (NCA beats cosine) | **yes** |  |
| death | -0.0041 (-0.0057 to -0.0024) (cosine beats NCA) | U | -0.0177 (-0.0220 to -0.0136) (cosine beats NCA) | no | -0.0136 (-0.0169 to -0.0105) |
|  |  | V | -0.0039 (-0.0055 to -0.0022) (cosine beats NCA) | no | +0.0002 (+0.0000 to +0.0004) |
|  |  | I | -0.0022 (-0.0041 to -0.0003) (cosine beats NCA) | no | +0.0019 (+0.0002 to +0.0036) |
|  |  | UV | -0.0179 (-0.0223 to -0.0136) (cosine beats NCA) | no |  |
|  |  | UI | -0.0025 (-0.0069 to +0.0019) (no difference shown) | **yes** |  |
|  |  | VI | -0.0021 (-0.0041 to -0.0002) (cosine beats NCA) | no |  |
|  |  | UVI | -0.0025 (-0.0070 to +0.0020) (no difference shown) | **yes** |  |

**H-S** (PERISIM's levels in PRAM's setting: no NCA advantage for both outcomes): not supported ({'aki': 'NCA beats cosine', 'death': 'no difference shown'}).

## Every cell: Δ AUROC against COS

| outcome | metric | vote | use | Δ AUROC (95 % CI) | verdict |
|---|---|---|---|---|---|
| aki | MI | ours | alone | +0.0088 (+0.0057 to +0.0118) | NCA beats cosine |
| aki | MI | ours | mix, logistic | +0.0013 (-0.0003 to +0.0028) | no difference shown |
| aki | MI | ours | mix, LightGBM | -0.0006 (-0.0014 to +0.0003) | no difference shown |
| aki | MI | inv | alone | +0.0088 (+0.0058 to +0.0117) | NCA beats cosine |
| aki | MI | inv | mix, logistic | +0.0012 (-0.0004 to +0.0028) | no difference shown |
| aki | MI | inv | mix, LightGBM | -0.0005 (-0.0014 to +0.0004) | no difference shown |
| aki | NCA-full | ours | alone | +0.0014 (-0.0016 to +0.0044) | no difference shown |
| aki | NCA-full | ours | mix, logistic | -0.0017 (-0.0032 to -0.0001) | cosine beats NCA |
| aki | NCA-full | ours | mix, LightGBM | -0.0003 (-0.0011 to +0.0006) | no difference shown |
| aki | NCA-full | inv | alone | +0.0014 (-0.0015 to +0.0043) | no difference shown |
| aki | NCA-full | inv | mix, logistic | -0.0018 (-0.0033 to -0.0003) | cosine beats NCA |
| aki | NCA-full | inv | mix, LightGBM | -0.0003 (-0.0011 to +0.0006) | no difference shown |
| aki | NCA-full (2k) | ours | alone | -0.0032 (-0.0061 to -0.0000) | cosine beats NCA |
| aki | NCA-full (2k) | ours | mix, logistic | -0.0043 (-0.0058 to -0.0027) | cosine beats NCA |
| aki | NCA-full (2k) | ours | mix, LightGBM | -0.0011 (-0.0019 to -0.0002) | cosine beats NCA |
| aki | NCA-full (2k) | inv | alone | -0.0032 (-0.0061 to -0.0001) | cosine beats NCA |
| aki | NCA-full (2k) | inv | mix, logistic | -0.0044 (-0.0059 to -0.0029) | cosine beats NCA |
| aki | NCA-full (2k) | inv | mix, LightGBM | -0.0010 (-0.0018 to -0.0001) | cosine beats NCA |
| aki | NCA-full (10k) | ours | alone | -0.0031 (-0.0060 to +0.0000) | no difference shown |
| aki | NCA-full (10k) | ours | mix, logistic | -0.0034 (-0.0049 to -0.0019) | cosine beats NCA |
| aki | NCA-full (10k) | ours | mix, LightGBM | -0.0021 (-0.0030 to -0.0013) | cosine beats NCA |
| aki | NCA-full (10k) | inv | alone | -0.0030 (-0.0059 to +0.0001) | no difference shown |
| aki | NCA-full (10k) | inv | mix, logistic | -0.0035 (-0.0049 to -0.0020) | cosine beats NCA |
| aki | NCA-full (10k) | inv | mix, LightGBM | -0.0021 (-0.0029 to -0.0013) | cosine beats NCA |
| aki | NCA-diag | ours | alone | +0.0342 (+0.0303 to +0.0382) | NCA beats cosine |
| aki | NCA-diag | ours | mix, logistic | +0.0061 (+0.0040 to +0.0082) | NCA beats cosine |
| aki | NCA-diag | ours | mix, LightGBM | +0.0041 (+0.0027 to +0.0055) | NCA beats cosine |
| aki | NCA-diag | inv | alone | +0.0343 (+0.0303 to +0.0383) | NCA beats cosine |
| aki | NCA-diag | inv | mix, logistic | +0.0066 (+0.0044 to +0.0087) | NCA beats cosine |
| aki | NCA-diag | inv | mix, LightGBM | +0.0042 (+0.0028 to +0.0056) | NCA beats cosine |
| aki | NCA-full, target bank | ours | alone | -0.0075 (-0.0105 to -0.0044) | cosine beats NCA |
| aki | NCA-full, target bank | ours | mix, logistic | -0.0036 (-0.0051 to -0.0020) | cosine beats NCA |
| aki | NCA-full, target bank | ours | mix, LightGBM | -0.0012 (-0.0021 to -0.0003) | cosine beats NCA |
| aki | NCA-full, target bank | inv | alone | -0.0078 (-0.0108 to -0.0047) | cosine beats NCA |
| aki | NCA-full, target bank | inv | mix, logistic | -0.0037 (-0.0053 to -0.0022) | cosine beats NCA |
| aki | NCA-full, target bank | inv | mix, LightGBM | -0.0012 (-0.0021 to -0.0004) | cosine beats NCA |
| aki | NCA-diag, target bank | ours | alone | +0.0105 (+0.0068 to +0.0141) | NCA beats cosine |
| aki | NCA-diag, target bank | ours | mix, logistic | +0.0056 (+0.0034 to +0.0076) | NCA beats cosine |
| aki | NCA-diag, target bank | ours | mix, LightGBM | -0.0001 (-0.0012 to +0.0010) | no difference shown |
| aki | NCA-diag, target bank | inv | alone | +0.0109 (+0.0072 to +0.0145) | NCA beats cosine |
| aki | NCA-diag, target bank | inv | mix, logistic | +0.0055 (+0.0034 to +0.0076) | NCA beats cosine |
| aki | NCA-diag, target bank | inv | mix, LightGBM | +0.0000 (-0.0011 to +0.0011) | no difference shown |
| death | MI | ours | alone | +0.0023 (-0.0012 to +0.0059) | no difference shown |
| death | MI | ours | mix, logistic | -0.0001 (-0.0016 to +0.0014) | no difference shown |
| death | MI | ours | mix, LightGBM | +0.0013 (+0.0000 to +0.0026) | NCA beats cosine |
| death | MI | inv | alone | +0.0020 (-0.0013 to +0.0056) | no difference shown |
| death | MI | inv | mix, logistic | -0.0003 (-0.0018 to +0.0012) | no difference shown |
| death | MI | inv | mix, LightGBM | +0.0012 (+0.0001 to +0.0025) | NCA beats cosine |
| death | NCA-full | ours | alone | -0.0179 (-0.0223 to -0.0136) | cosine beats NCA |
| death | NCA-full | ours | mix, logistic | -0.0039 (-0.0055 to -0.0022) | cosine beats NCA |
| death | NCA-full | ours | mix, LightGBM | -0.0019 (-0.0032 to -0.0006) | cosine beats NCA |
| death | NCA-full | inv | alone | -0.0177 (-0.0220 to -0.0136) | cosine beats NCA |
| death | NCA-full | inv | mix, logistic | -0.0041 (-0.0057 to -0.0024) | cosine beats NCA |
| death | NCA-full | inv | mix, LightGBM | -0.0019 (-0.0032 to -0.0006) | cosine beats NCA |
| death | NCA-full (2k) | ours | alone | -0.0241 (-0.0286 to -0.0197) | cosine beats NCA |
| death | NCA-full (2k) | ours | mix, logistic | -0.0033 (-0.0049 to -0.0016) | cosine beats NCA |
| death | NCA-full (2k) | ours | mix, LightGBM | -0.0029 (-0.0041 to -0.0016) | cosine beats NCA |
| death | NCA-full (2k) | inv | alone | -0.0241 (-0.0286 to -0.0199) | cosine beats NCA |
| death | NCA-full (2k) | inv | mix, logistic | -0.0035 (-0.0051 to -0.0018) | cosine beats NCA |
| death | NCA-full (2k) | inv | mix, LightGBM | -0.0029 (-0.0041 to -0.0017) | cosine beats NCA |
| death | NCA-full (10k) | ours | alone | -0.0284 (-0.0328 to -0.0239) | cosine beats NCA |
| death | NCA-full (10k) | ours | mix, logistic | -0.0050 (-0.0065 to -0.0034) | cosine beats NCA |
| death | NCA-full (10k) | ours | mix, LightGBM | -0.0027 (-0.0039 to -0.0015) | cosine beats NCA |
| death | NCA-full (10k) | inv | alone | -0.0285 (-0.0329 to -0.0241) | cosine beats NCA |
| death | NCA-full (10k) | inv | mix, logistic | -0.0051 (-0.0066 to -0.0036) | cosine beats NCA |
| death | NCA-full (10k) | inv | mix, LightGBM | -0.0026 (-0.0038 to -0.0015) | cosine beats NCA |
| death | NCA-diag | ours | alone | -0.0025 (-0.0070 to +0.0020) | no difference shown |
| death | NCA-diag | ours | mix, logistic | -0.0021 (-0.0041 to -0.0002) | cosine beats NCA |
| death | NCA-diag | ours | mix, LightGBM | +0.0007 (-0.0009 to +0.0023) | no difference shown |
| death | NCA-diag | inv | alone | -0.0025 (-0.0069 to +0.0019) | no difference shown |
| death | NCA-diag | inv | mix, logistic | -0.0022 (-0.0041 to -0.0003) | cosine beats NCA |
| death | NCA-diag | inv | mix, LightGBM | +0.0009 (-0.0007 to +0.0025) | no difference shown |
| death | NCA-full, target bank | ours | alone | -0.0183 (-0.0225 to -0.0138) | cosine beats NCA |
| death | NCA-full, target bank | ours | mix, logistic | -0.0035 (-0.0051 to -0.0019) | cosine beats NCA |
| death | NCA-full, target bank | ours | mix, LightGBM | -0.0013 (-0.0026 to +0.0000) | no difference shown |
| death | NCA-full, target bank | inv | alone | -0.0186 (-0.0227 to -0.0143) | cosine beats NCA |
| death | NCA-full, target bank | inv | mix, logistic | -0.0036 (-0.0051 to -0.0020) | cosine beats NCA |
| death | NCA-full, target bank | inv | mix, LightGBM | -0.0012 (-0.0024 to +0.0001) | no difference shown |
| death | NCA-diag, target bank | ours | alone | +0.0018 (-0.0024 to +0.0062) | no difference shown |
| death | NCA-diag, target bank | ours | mix, logistic | -0.0008 (-0.0027 to +0.0009) | no difference shown |
| death | NCA-diag, target bank | ours | mix, LightGBM | +0.0019 (+0.0005 to +0.0034) | NCA beats cosine |
| death | NCA-diag, target bank | inv | alone | +0.0023 (-0.0019 to +0.0067) | no difference shown |
| death | NCA-diag, target bank | inv | mix, logistic | -0.0009 (-0.0027 to +0.0009) | no difference shown |
| death | NCA-diag, target bank | inv | mix, LightGBM | +0.0022 (+0.0008 to +0.0037) | NCA beats cosine |

## Every arm

| outcome | arm | AUROC | log-loss | CITL | slope |
|---|---|---|---|---|---|
| aki | cos|ours|alone | 0.7284 | 0.4148 | +0.094 | 1.055 |
| aki | cos|ours|mix_lr | 0.7666 | 0.3941 | -0.108 | 1.350 |
| aki | cos|ours|mix_gbm | 0.7762 | 0.3878 | -0.200 | 1.133 |
| aki | cos|inv|alone | 0.7291 | 0.4411 | +0.104 | 0.851 |
| aki | cos|inv|mix_lr | 0.7669 | 0.3930 | -0.122 | 1.287 |
| aki | cos|inv|mix_gbm | 0.7762 | 0.3876 | -0.209 | 1.097 |
| aki | mi|ours|alone | 0.7371 | 0.4101 | +0.185 | 1.124 |
| aki | mi|ours|mix_lr | 0.7679 | 0.3922 | -0.069 | 1.331 |
| aki | mi|ours|mix_gbm | 0.7757 | 0.3869 | -0.162 | 1.120 |
| aki | mi|inv|alone | 0.7379 | 0.4628 | +0.238 | 0.730 |
| aki | mi|inv|mix_lr | 0.7681 | 0.3899 | -0.058 | 1.215 |
| aki | mi|inv|mix_gbm | 0.7758 | 0.3864 | -0.170 | 1.044 |
| aki | ncaf|ours|alone | 0.7297 | 0.4147 | +0.181 | 1.152 |
| aki | ncaf|ours|mix_lr | 0.7650 | 0.3946 | -0.069 | 1.341 |
| aki | ncaf|ours|mix_gbm | 0.7759 | 0.3876 | -0.161 | 1.145 |
| aki | ncaf|inv|alone | 0.7305 | 0.4474 | +0.231 | 0.797 |
| aki | ncaf|inv|mix_lr | 0.7651 | 0.3925 | -0.069 | 1.226 |
| aki | ncaf|inv|mix_gbm | 0.7760 | 0.3870 | -0.181 | 1.063 |
| aki | ncaf2k|ours|alone | 0.7252 | 0.4169 | +0.166 | 1.147 |
| aki | ncaf2k|ours|mix_lr | 0.7623 | 0.3959 | -0.075 | 1.340 |
| aki | ncaf2k|ours|mix_gbm | 0.7752 | 0.3882 | -0.182 | 1.133 |
| aki | ncaf2k|inv|alone | 0.7259 | 0.4467 | +0.210 | 0.794 |
| aki | ncaf2k|inv|mix_lr | 0.7625 | 0.3938 | -0.086 | 1.221 |
| aki | ncaf2k|inv|mix_gbm | 0.7753 | 0.3876 | -0.191 | 1.065 |
| aki | ncaf10k|ours|alone | 0.7253 | 0.4163 | +0.217 | 1.115 |
| aki | ncaf10k|ours|mix_lr | 0.7632 | 0.3946 | -0.054 | 1.329 |
| aki | ncaf10k|ours|mix_gbm | 0.7741 | 0.3879 | -0.174 | 1.107 |
| aki | ncaf10k|inv|alone | 0.7261 | 0.4583 | +0.277 | 0.747 |
| aki | ncaf10k|inv|mix_lr | 0.7634 | 0.3925 | -0.063 | 1.210 |
| aki | ncaf10k|inv|mix_gbm | 0.7742 | 0.3876 | -0.195 | 1.034 |
| aki | ncad|ours|alone | 0.7626 | 0.3893 | +0.159 | 1.083 |
| aki | ncad|ours|mix_lr | 0.7727 | 0.3871 | -0.088 | 1.173 |
| aki | ncad|ours|mix_gbm | 0.7803 | 0.3801 | -0.117 | 1.117 |
| aki | ncad|inv|alone | 0.7634 | 0.4396 | +0.196 | 0.773 |
| aki | ncad|inv|mix_lr | 0.7735 | 0.3853 | -0.080 | 1.079 |
| aki | ncad|inv|mix_gbm | 0.7805 | 0.3793 | -0.126 | 1.030 |
| aki | ncaf_t|ours|alone | 0.7209 | 0.4186 | +0.201 | 1.126 |
| aki | ncaf_t|ours|mix_lr | 0.7631 | 0.3954 | -0.065 | 1.345 |
| aki | ncaf_t|ours|mix_gbm | 0.7750 | 0.3879 | -0.160 | 1.144 |
| aki | ncaf_t|inv|alone | 0.7214 | 0.4554 | +0.256 | 0.765 |
| aki | ncaf_t|inv|mix_lr | 0.7632 | 0.3933 | -0.069 | 1.231 |
| aki | ncaf_t|inv|mix_gbm | 0.7750 | 0.3873 | -0.174 | 1.067 |
| aki | ncad_t|ours|alone | 0.7388 | 0.4093 | +0.169 | 1.083 |
| aki | ncad_t|ours|mix_lr | 0.7722 | 0.3912 | -0.076 | 1.331 |
| aki | ncad_t|ours|mix_gbm | 0.7762 | 0.3866 | -0.163 | 1.119 |
| aki | ncad_t|inv|alone | 0.7400 | 0.4553 | +0.205 | 0.761 |
| aki | ncad_t|inv|mix_lr | 0.7724 | 0.3892 | -0.076 | 1.239 |
| aki | ncad_t|inv|mix_gbm | 0.7763 | 0.3860 | -0.173 | 1.057 |
| aki | base|lr | 0.7445 | 0.4137 | -0.320 | 0.816 |
| aki | base|gbm | 0.7709 | 0.3953 | -0.375 | 0.899 |
| death | cos|ours|alone | 0.8374 | 0.1873 | -0.025 | 1.087 |
| death | cos|ours|mix_lr | 0.8590 | 0.1804 | -0.024 | 1.207 |
| death | cos|ours|mix_gbm | 0.8624 | 0.1781 | +0.049 | 1.043 |
| death | cos|inv|alone | 0.8379 | 0.2524 | -0.030 | 0.604 |
| death | cos|inv|mix_lr | 0.8592 | 0.1797 | -0.026 | 1.171 |
| death | cos|inv|mix_gbm | 0.8624 | 0.1780 | +0.051 | 0.998 |
| death | mi|ours|alone | 0.8397 | 0.1860 | +0.284 | 1.040 |
| death | mi|ours|mix_lr | 0.8588 | 0.1799 | +0.106 | 1.168 |
| death | mi|ours|mix_gbm | 0.8637 | 0.1776 | +0.164 | 1.046 |
| death | mi|inv|alone | 0.8400 | 0.2986 | +0.317 | 0.408 |
| death | mi|inv|mix_lr | 0.8589 | 0.1793 | +0.114 | 1.119 |
| death | mi|inv|mix_gbm | 0.8637 | 0.1775 | +0.170 | 0.990 |
| death | ncaf|ours|alone | 0.8195 | 0.1955 | +0.360 | 1.249 |
| death | ncaf|ours|mix_lr | 0.8551 | 0.1828 | +0.103 | 1.216 |
| death | ncaf|ours|mix_gbm | 0.8605 | 0.1799 | +0.161 | 1.038 |
| death | ncaf|inv|alone | 0.8202 | 0.3331 | +0.470 | -11371901.542 |
| death | ncaf|inv|mix_lr | 0.8552 | 0.1817 | +0.107 | 1.133 |
| death | ncaf|inv|mix_gbm | 0.8606 | 0.1797 | +0.173 | 0.965 |
| death | ncaf2k|ours|alone | 0.8133 | 0.1986 | +0.371 | 1.144 |
| death | ncaf2k|ours|mix_lr | 0.8557 | 0.1829 | +0.106 | 1.216 |
| death | ncaf2k|ours|mix_gbm | 0.8595 | 0.1804 | +0.158 | 1.014 |
| death | ncaf2k|inv|alone | 0.8138 | 0.3216 | +0.453 | 5380702.388 |
| death | ncaf2k|inv|mix_lr | 0.8558 | 0.1819 | +0.104 | 1.148 |
| death | ncaf2k|inv|mix_gbm | 0.8596 | 0.1803 | +0.161 | 0.955 |
| death | ncaf10k|ours|alone | 0.8090 | 0.2001 | +0.410 | 1.265 |
| death | ncaf10k|ours|mix_lr | 0.8540 | 0.1835 | +0.119 | 1.230 |
| death | ncaf10k|ours|mix_gbm | 0.8597 | 0.1805 | +0.177 | 1.052 |
| death | ncaf10k|inv|alone | 0.8094 | 0.3362 | +0.535 | -52267486.744 |
| death | ncaf10k|inv|mix_lr | 0.8541 | 0.1823 | +0.123 | 1.146 |
| death | ncaf10k|inv|mix_gbm | 0.8598 | 0.1801 | +0.187 | 0.974 |
| death | ncad|ours|alone | 0.8349 | 0.1872 | +0.265 | 1.108 |
| death | ncad|ours|mix_lr | 0.8569 | 0.1809 | +0.097 | 1.179 |
| death | ncad|ours|mix_gbm | 0.8631 | 0.1780 | +0.155 | 1.068 |
| death | ncad|inv|alone | 0.8355 | 0.3198 | +0.325 | 0.363 |
| death | ncad|inv|mix_lr | 0.8571 | 0.1800 | +0.103 | 1.099 |
| death | ncad|inv|mix_gbm | 0.8633 | 0.1777 | +0.170 | 0.981 |
| death | ncaf_t|ours|alone | 0.8191 | 0.1958 | +0.398 | 1.163 |
| death | ncaf_t|ours|mix_lr | 0.8555 | 0.1825 | +0.126 | 1.214 |
| death | ncaf_t|ours|mix_gbm | 0.8611 | 0.1796 | +0.183 | 1.052 |
| death | ncaf_t|inv|alone | 0.8194 | 0.3333 | +0.487 | 0.328 |
| death | ncaf_t|inv|mix_lr | 0.8557 | 0.1815 | +0.126 | 1.142 |
| death | ncaf_t|inv|mix_gbm | 0.8613 | 0.1792 | +0.192 | 0.987 |
| death | ncad_t|ours|alone | 0.8392 | 0.1847 | +0.214 | 1.039 |
| death | ncad_t|ours|mix_lr | 0.8581 | 0.1797 | +0.078 | 1.159 |
| death | ncad_t|ours|mix_gbm | 0.8643 | 0.1769 | +0.136 | 1.054 |
| death | ncad_t|inv|alone | 0.8403 | 0.3046 | +0.246 | 0.400 |
| death | ncad_t|inv|mix_lr | 0.8583 | 0.1790 | +0.083 | 1.098 |
| death | ncad_t|inv|mix_gbm | 0.8647 | 0.1766 | +0.146 | 0.990 |
| death | base|lr | 0.8477 | 0.1849 | -0.025 | 1.021 |
| death | base|gbm | 0.8564 | 0.1827 | +0.088 | 0.860 |

## Sensitivity: aki_kdigo48 (KDIGO's 48 h window for the 0.3 mg/dL rise; prevalence in eICU 15.5%; deviation 2026-11-08)

- home Δ: -0.0019 (-0.0034 to -0.0003) (cosine beats NCA); frozen logistic AUROC 0.746; cosine-mixture gain +0.0225 (+0.0201 to +0.0248)
- switch U: -0.0008 (-0.0039 to +0.0024) (no difference shown) — sufficient
- switch V: -0.0017 (-0.0033 to -0.0000) (cosine beats NCA)
- switch I: +0.0078 (+0.0053 to +0.0100) (NCA beats cosine) — sufficient
- switch UV: -0.0006 (-0.0039 to +0.0026) (no difference shown) — sufficient
- switch UI: +0.0387 (+0.0343 to +0.0429) (NCA beats cosine) — sufficient
- switch VI: +0.0074 (+0.0051 to +0.0096) (NCA beats cosine) — sufficient
- switch UVI: +0.0383 (+0.0339 to +0.0425) (NCA beats cosine) — sufficient

## Sensitivity: a chosen by log-loss

| outcome | home Δ | U switch Δ (verdict) |
|---|---|---|
| aki | -0.0017 (-0.0031 to -0.0002) | +0.0014 (-0.0015 to +0.0043) (no difference shown) |
| death | -0.0037 (-0.0055 to -0.0019) | -0.0177 (-0.0220 to -0.0137) (cosine beats NCA) |

## Complementarity (descriptive)

| outcome | arm, vote | base | corr(p_kNN, p_base) | Spearman(p_kNN, y − p_base) |
|---|---|---|---|---|
| aki | COS, ours | lr | 0.560 | -0.201 |
| aki | COS, ours | gbm | 0.711 | -0.233 |
| aki | COS, inv | lr | 0.559 | -0.201 |
| aki | COS, inv | gbm | 0.711 | -0.233 |
| aki | MI, ours | lr | 0.612 | -0.204 |
| aki | MI, ours | gbm | 0.772 | -0.242 |
| aki | MI, inv | lr | 0.612 | -0.204 |
| aki | MI, inv | gbm | 0.772 | -0.242 |
| aki | NCA-full, ours | lr | 0.608 | -0.214 |
| aki | NCA-full, ours | gbm | 0.742 | -0.225 |
| aki | NCA-full, inv | lr | 0.608 | -0.214 |
| aki | NCA-full, inv | gbm | 0.743 | -0.226 |
| aki | NCA-full (2k), ours | lr | 0.605 | -0.233 |
| aki | NCA-full (2k), ours | gbm | 0.730 | -0.235 |
| aki | NCA-full (2k), inv | lr | 0.605 | -0.233 |
| aki | NCA-full (2k), inv | gbm | 0.731 | -0.235 |
| aki | NCA-full (10k), ours | lr | 0.605 | -0.203 |
| aki | NCA-full (10k), ours | gbm | 0.753 | -0.231 |
| aki | NCA-full (10k), inv | lr | 0.605 | -0.204 |
| aki | NCA-full (10k), inv | gbm | 0.754 | -0.232 |
| aki | NCA-diag, ours | lr | 0.817 | -0.170 |
| aki | NCA-diag, ours | gbm | 0.821 | -0.218 |
| aki | NCA-diag, inv | lr | 0.818 | -0.170 |
| aki | NCA-diag, inv | gbm | 0.823 | -0.218 |
| aki | NCA-full, target bank, ours | lr | 0.584 | -0.194 |
| aki | NCA-full, target bank, ours | gbm | 0.721 | -0.219 |
| aki | NCA-full, target bank, inv | lr | 0.585 | -0.194 |
| aki | NCA-full, target bank, inv | gbm | 0.722 | -0.219 |
| aki | NCA-diag, target bank, ours | lr | 0.617 | -0.179 |
| aki | NCA-diag, target bank, ours | gbm | 0.771 | -0.239 |
| aki | NCA-diag, target bank, inv | lr | 0.620 | -0.179 |
| aki | NCA-diag, target bank, inv | gbm | 0.774 | -0.239 |
| death | COS, ours | lr | 0.711 | -0.474 |
| death | COS, ours | gbm | 0.715 | -0.514 |
| death | COS, inv | lr | 0.712 | -0.481 |
| death | COS, inv | gbm | 0.716 | -0.523 |
| death | MI, ours | lr | 0.798 | -0.459 |
| death | MI, ours | gbm | 0.759 | -0.482 |
| death | MI, inv | lr | 0.798 | -0.472 |
| death | MI, inv | gbm | 0.759 | -0.496 |
| death | NCA-full, ours | lr | 0.727 | -0.433 |
| death | NCA-full, ours | gbm | 0.714 | -0.450 |
| death | NCA-full, inv | lr | 0.728 | -0.444 |
| death | NCA-full, inv | gbm | 0.714 | -0.462 |
| death | NCA-full (2k), ours | lr | 0.683 | -0.424 |
| death | NCA-full (2k), ours | gbm | 0.702 | -0.460 |
| death | NCA-full (2k), inv | lr | 0.685 | -0.438 |
| death | NCA-full (2k), inv | gbm | 0.704 | -0.475 |
| death | NCA-full (10k), ours | lr | 0.689 | -0.412 |
| death | NCA-full (10k), ours | gbm | 0.678 | -0.434 |
| death | NCA-full (10k), inv | lr | 0.690 | -0.422 |
| death | NCA-full (10k), inv | gbm | 0.679 | -0.445 |
| death | NCA-diag, ours | lr | 0.805 | -0.460 |
| death | NCA-diag, ours | gbm | 0.753 | -0.461 |
| death | NCA-diag, inv | lr | 0.806 | -0.477 |
| death | NCA-diag, inv | gbm | 0.754 | -0.479 |
| death | NCA-full, target bank, ours | lr | 0.726 | -0.413 |
| death | NCA-full, target bank, ours | gbm | 0.695 | -0.426 |
| death | NCA-full, target bank, inv | lr | 0.727 | -0.427 |
| death | NCA-full, target bank, inv | gbm | 0.696 | -0.441 |
| death | NCA-diag, target bank, ours | lr | 0.807 | -0.466 |
| death | NCA-diag, target bank, ours | gbm | 0.748 | -0.477 |
| death | NCA-diag, target bank, inv | lr | 0.808 | -0.480 |
| death | NCA-diag, target bank, inv | gbm | 0.749 | -0.492 |

## Cohort

Pre-registered: docs/prereg_pram_factors.md §4. Counts are stays.

## Flow

| step | mimic | eicu |
|---|---|---|
| first_stays | 65,366 | 139,367 |
| in_anchor_years | 56,650 | — |
| adults | 56,650 | 138,852 |
| icu_24h | 44,670 | 96,473 |
| with_creatinine_24h | 44,021 | 88,597 |

## Outcomes

| outcome | mimic | eicu |
|---|---|---|
| aki | 28.3% | 17.6% |
| death | 6.0% | 6.4% |
| aki_kdigo48 | 25.0% | 15.5% |

## Features: missing (%) and median [IQR]

| feature | mimic missing | mimic median [IQR] | eicu missing | eicu median [IQR] |
|---|---|---|---|---|
| age | 0.0 | 67.00 [55.00, 78.00] | 0.0 | 66.00 [54.00, 77.00] |
| sex_male | 0.0 | 1.00 [0.00, 1.00] | 0.0 | 1.00 [0.00, 1.00] |
| baseline_scr | 0.0 | 0.90 [0.70, 1.20] | 0.0 | 0.96 [0.71, 1.44] |
| hr_min | 0.2 | 69.00 [60.00, 79.00] | 1.3 | 68.00 [59.00, 79.00] |
| hr_max | 0.2 | 101.00 [89.00, 115.00] | 1.3 | 107.00 [94.00, 123.00] |
| hr_last | 0.2 | 82.00 [71.00, 95.00] | 1.3 | 83.00 [72.00, 96.00] |
| rr_min | 0.3 | 12.00 [10.00, 14.00] | 7.9 | 11.00 [8.00, 14.00] |
| rr_max | 0.3 | 27.00 [24.00, 31.00] | 7.9 | 30.00 [25.00, 37.00] |
| rr_last | 0.3 | 19.00 [16.00, 23.00] | 7.9 | 19.00 [16.00, 23.00] |
| spo2_min | 0.2 | 93.00 [91.00, 95.00] | 2.1 | 90.00 [84.00, 93.00] |
| spo2_max | 0.2 | 100.00 [99.00, 100.00] | 2.1 | 100.00 [100.00, 100.00] |
| spo2_last | 0.2 | 97.00 [95.00, 99.00] | 2.1 | 97.00 [95.00, 99.00] |
| temp_c_min | 3.5 | 36.44 [36.11, 36.67] | 88.3 | 35.90 [34.83, 36.60] |
| temp_c_max | 3.5 | 37.22 [36.94, 37.72] | 88.3 | 37.80 [37.30, 38.30] |
| temp_c_last | 3.5 | 36.89 [36.61, 37.22] | 88.3 | 37.20 [36.70, 37.70] |
| sbp_min | 0.5 | 91.00 [82.00, 101.00] | 1.3 | 85.00 [73.00, 100.00] |
| sbp_max | 0.5 | 146.00 [133.00, 162.00] | 1.3 | 157.00 [140.00, 177.00] |
| sbp_last | 0.5 | 117.00 [104.00, 132.00] | 1.3 | 120.00 [106.00, 137.00] |
| dbp_min | 0.5 | 46.00 [40.00, 53.00] | 1.3 | 44.00 [36.00, 52.00] |
| dbp_max | 0.5 | 86.00 [75.00, 99.00] | 1.3 | 95.00 [82.00, 112.00] |
| dbp_last | 0.5 | 62.00 [53.00, 72.00] | 1.3 | 63.00 [55.00, 74.00] |
| gcs_total_min | 0.2 | 11.00 [3.00, 15.00] | 3.6 | 14.00 [10.00, 15.00] |
| gcs_total_max | 0.2 | 15.00 [14.00, 15.00] | 3.6 | 15.00 [14.00, 15.00] |
| gcs_total_last | 0.2 | 15.00 [11.00, 15.00] | 3.6 | 15.00 [13.00, 15.00] |
| weight_kg_min | 0.5 | 78.50 [65.86, 93.12] | 1.9 | 80.20 [66.70, 97.10] |
| weight_kg_max | 0.5 | 80.40 [67.60, 96.00] | 1.9 | 80.20 [66.70, 97.10] |
| weight_kg_last | 0.5 | 79.92 [67.00, 95.30] | 1.9 | 80.20 [66.70, 97.10] |
| bun_min | 0.0 | 17.00 [12.00, 26.00] | 0.6 | 18.00 [12.00, 30.00] |
| bun_max | 0.0 | 19.00 [13.00, 30.00] | 0.6 | 19.00 [13.00, 32.00] |
| bun_last | 0.0 | 18.00 [12.00, 28.00] | 0.6 | 19.00 [12.00, 31.00] |
| calcium_min | 9.7 | 8.20 [7.70, 8.70] | 3.2 | 8.20 [7.70, 8.70] |
| calcium_max | 9.7 | 8.50 [8.00, 8.90] | 3.2 | 8.40 [7.90, 8.80] |
| calcium_last | 9.7 | 8.30 [7.90, 8.80] | 3.2 | 8.30 [7.80, 8.70] |
| inr_min | 14.1 | 1.20 [1.10, 1.40] | 59.5 | 1.20 [1.10, 1.50] |
| inr_max | 14.1 | 1.30 [1.10, 1.50] | 59.5 | 1.30 [1.10, 1.60] |
| inr_last | 14.1 | 1.20 [1.10, 1.40] | 59.5 | 1.23 [1.10, 1.50] |
| scr_feat_min | 0.0 | 0.90 [0.70, 1.20] | 0.0 | 0.96 [0.71, 1.44] |
| scr_feat_max | 0.0 | 1.00 [0.70, 1.40] | 0.0 | 1.01 [0.76, 1.58] |
| scr_feat_last | 0.0 | 0.90 [0.70, 1.30] | 0.0 | 0.99 [0.73, 1.50] |
| bun_cr_ratio_min | 0.0 | 17.27 [13.33, 22.50] | 0.6 | 17.14 [12.73, 23.08] |
| bun_cr_ratio_max | 0.0 | 20.00 [15.00, 25.45] | 0.6 | 18.33 [13.75, 24.56] |
| bun_cr_ratio_last | 0.0 | 18.18 [14.00, 23.75] | 0.6 | 17.72 [13.19, 23.81] |

