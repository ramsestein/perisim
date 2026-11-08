# Tier 1: the factors behind the learned-metric vs cosine reversal, in PERISIM's setting

Pre-registered: docs/prereg_pram_factors.md (OSF). Δ = AUROC(NCA arm) − AUROC(COS arm) with the same vote and use; 2,000 patient resamples. Home levels: alone, V-ours, NCA-diag. A switch is sufficient when it changes the verdict on Δ.

## Rule (pre-specified)

| target | home Δ (verdict) | switch | Δ after switch (verdict) | sufficient | effect (95 % CI) |
|---|---|---|---|---|---|
| inspire | +0.0177 (+0.0145 to +0.0210) (NCA beats cosine) | U | +0.0063 (+0.0051 to +0.0076) (NCA beats cosine) | no | +0.0114 (+0.0092 to +0.0136) |
|  |  | V | +0.0170 (+0.0137 to +0.0202) (NCA beats cosine) | no | +0.0007 (+0.0001 to +0.0012) |
|  |  | I | +0.0095 (+0.0065 to +0.0129) (NCA beats cosine) | no | +0.0082 (+0.0057 to +0.0107) |
|  |  | UV | +0.0063 (+0.0050 to +0.0077) (NCA beats cosine) | no |  |
|  |  | UI | +0.0011 (-0.0001 to +0.0023) (no difference shown) | **yes** |  |
|  |  | VI | +0.0087 (+0.0056 to +0.0119) (NCA beats cosine) | no |  |
|  |  | UVI | +0.0010 (-0.0003 to +0.0022) (no difference shown) | **yes** |  |
| mimic | +0.0067 (+0.0056 to +0.0079) (NCA beats cosine) | U | +0.0004 (-0.0002 to +0.0009) (no difference shown) | **yes** | +0.0064 (+0.0057 to +0.0071) |
|  |  | V | +0.0065 (+0.0053 to +0.0076) (NCA beats cosine) | no | +0.0003 (+0.0001 to +0.0004) |
|  |  | I | +0.0021 (+0.0011 to +0.0031) (NCA beats cosine) | no | +0.0046 (+0.0036 to +0.0057) |
|  |  | UV | +0.0003 (-0.0002 to +0.0008) (no difference shown) | **yes** |  |
|  |  | UI | -0.0005 (-0.0009 to -0.0001) (cosine beats NCA) | **yes** |  |
|  |  | VI | +0.0020 (+0.0010 to +0.0030) (NCA beats cosine) | no |  |
|  |  | UVI | -0.0005 (-0.0009 to -0.0001) (cosine beats NCA) | **yes** |  |
| eicu | +0.0159 (+0.0146 to +0.0171) (NCA beats cosine) | U | +0.0044 (+0.0039 to +0.0048) (NCA beats cosine) | no | +0.0115 (+0.0106 to +0.0124) |
|  |  | V | +0.0155 (+0.0143 to +0.0168) (NCA beats cosine) | no | +0.0004 (+0.0002 to +0.0006) |
|  |  | I | +0.0092 (+0.0080 to +0.0104) (NCA beats cosine) | no | +0.0067 (+0.0058 to +0.0076) |
|  |  | UV | +0.0043 (+0.0039 to +0.0048) (NCA beats cosine) | no |  |
|  |  | UI | +0.0025 (+0.0021 to +0.0029) (NCA beats cosine) | no |  |
|  |  | VI | +0.0090 (+0.0078 to +0.0103) (NCA beats cosine) | no |  |
|  |  | UVI | +0.0024 (+0.0020 to +0.0028) (NCA beats cosine) | no |  |
| NWICU (mimic+eicu) | +0.0171 (+0.0100 to +0.0248) (NCA beats cosine) | U | +0.0015 (-0.0005 to +0.0035) (no difference shown) | **yes** | +0.0156 (+0.0100 to +0.0219) |
|  |  | V | +0.0177 (+0.0105 to +0.0254) (NCA beats cosine) | no | -0.0006 (-0.0021 to +0.0008) |
|  |  | I | +0.0199 (+0.0124 to +0.0272) (NCA beats cosine) | no | -0.0028 (-0.0082 to +0.0027) |
|  |  | UV | +0.0012 (-0.0009 to +0.0033) (no difference shown) | **yes** |  |
|  |  | UI | +0.0016 (-0.0005 to +0.0036) (no difference shown) | **yes** |  |
|  |  | VI | +0.0195 (+0.0121 to +0.0268) (NCA beats cosine) | no |  |
|  |  | UVI | +0.0012 (-0.0008 to +0.0033) (no difference shown) | **yes** |  |
| NWICU (inspire+eicu) | +0.0179 (+0.0110 to +0.0257) (NCA beats cosine) | U | +0.0009 (-0.0010 to +0.0030) (no difference shown) | **yes** | +0.0169 (+0.0114 to +0.0231) |
|  |  | V | +0.0186 (+0.0114 to +0.0266) (NCA beats cosine) | no | -0.0008 (-0.0022 to +0.0007) |
|  |  | I | +0.0101 (+0.0034 to +0.0173) (NCA beats cosine) | no | +0.0078 (+0.0018 to +0.0136) |
|  |  | UV | +0.0007 (-0.0013 to +0.0030) (no difference shown) | **yes** |  |
|  |  | UI | -0.0001 (-0.0018 to +0.0018) (no difference shown) | **yes** |  |
|  |  | VI | +0.0105 (+0.0038 to +0.0178) (NCA beats cosine) | no |  |
|  |  | UVI | -0.0000 (-0.0018 to +0.0018) (no difference shown) | **yes** |  |
| NWICU (inspire+mimic) | +0.0190 (+0.0118 to +0.0268) (NCA beats cosine) | U | +0.0022 (+0.0001 to +0.0044) (NCA beats cosine) | no | +0.0167 (+0.0110 to +0.0227) |
|  |  | V | +0.0189 (+0.0118 to +0.0269) (NCA beats cosine) | no | +0.0000 (-0.0014 to +0.0014) |
|  |  | I | +0.0041 (-0.0035 to +0.0124) (no difference shown) | **yes** | +0.0149 (+0.0097 to +0.0203) |
|  |  | UV | +0.0013 (-0.0008 to +0.0037) (no difference shown) | **yes** |  |
|  |  | UI | -0.0004 (-0.0024 to +0.0017) (no difference shown) | **yes** |  |
|  |  | VI | +0.0048 (-0.0028 to +0.0127) (no difference shown) | **yes** |  |
|  |  | UVI | -0.0009 (-0.0031 to +0.0014) (no difference shown) | **yes** |  |

## Every cell: Δ AUROC against COS

| target | metric | vote | use | Δ AUROC (95 % CI) | verdict | Δ log-loss (95 % CI) |
|---|---|---|---|---|---|---|
| inspire | NCA-diag | ours | alone | +0.0177 (+0.0145 to +0.0210) | NCA beats cosine | +0.0128 (+0.0112 to +0.0143) |
| inspire | NCA-diag | ours | mix, logistic | +0.0063 (+0.0051 to +0.0076) | NCA beats cosine | +0.0056 (+0.0049 to +0.0064) |
| inspire | NCA-diag | ours | mix, LightGBM | +0.0022 (+0.0012 to +0.0031) | NCA beats cosine | +0.0041 (+0.0035 to +0.0046) |
| inspire | NCA-diag | inv | alone | +0.0170 (+0.0137 to +0.0202) | NCA beats cosine | -0.0075 (-0.0167 to +0.0018) |
| inspire | NCA-diag | inv | mix, logistic | +0.0063 (+0.0050 to +0.0077) | NCA beats cosine | +0.0055 (+0.0046 to +0.0064) |
| inspire | NCA-diag | inv | mix, LightGBM | +0.0020 (+0.0010 to +0.0030) | NCA beats cosine | +0.0039 (+0.0033 to +0.0045) |
| inspire | NCA-full | ours | alone | +0.0095 (+0.0065 to +0.0129) | NCA beats cosine | +0.0085 (+0.0070 to +0.0100) |
| inspire | NCA-full | ours | mix, logistic | +0.0011 (-0.0001 to +0.0023) | no difference shown | +0.0032 (+0.0025 to +0.0039) |
| inspire | NCA-full | ours | mix, LightGBM | -0.0004 (-0.0012 to +0.0004) | no difference shown | +0.0023 (+0.0018 to +0.0027) |
| inspire | NCA-full | inv | alone | +0.0087 (+0.0056 to +0.0119) | NCA beats cosine | -0.0171 (-0.0264 to -0.0081) |
| inspire | NCA-full | inv | mix, logistic | +0.0010 (-0.0003 to +0.0022) | no difference shown | +0.0028 (+0.0019 to +0.0036) |
| inspire | NCA-full | inv | mix, LightGBM | -0.0006 (-0.0014 to +0.0003) | no difference shown | +0.0021 (+0.0017 to +0.0026) |
| inspire | NCA-full (2k) | ours | alone | +0.0105 (+0.0072 to +0.0139) | NCA beats cosine | +0.0082 (+0.0066 to +0.0098) |
| inspire | NCA-full (2k) | ours | mix, logistic | +0.0020 (+0.0007 to +0.0032) | NCA beats cosine | +0.0031 (+0.0024 to +0.0038) |
| inspire | NCA-full (2k) | ours | mix, LightGBM | +0.0000 (-0.0008 to +0.0008) | no difference shown | +0.0021 (+0.0017 to +0.0026) |
| inspire | NCA-full (2k) | inv | alone | +0.0099 (+0.0066 to +0.0132) | NCA beats cosine | -0.0143 (-0.0230 to -0.0057) |
| inspire | NCA-full (2k) | inv | mix, logistic | +0.0018 (+0.0006 to +0.0032) | NCA beats cosine | +0.0028 (+0.0020 to +0.0036) |
| inspire | NCA-full (2k) | inv | mix, LightGBM | -0.0001 (-0.0010 to +0.0007) | no difference shown | +0.0021 (+0.0016 to +0.0026) |
| inspire | NCA-full (10k) | ours | alone | +0.0101 (+0.0068 to +0.0133) | NCA beats cosine | +0.0086 (+0.0071 to +0.0101) |
| inspire | NCA-full (10k) | ours | mix, logistic | +0.0014 (+0.0002 to +0.0026) | NCA beats cosine | +0.0033 (+0.0026 to +0.0041) |
| inspire | NCA-full (10k) | ours | mix, LightGBM | -0.0003 (-0.0011 to +0.0005) | no difference shown | +0.0025 (+0.0020 to +0.0030) |
| inspire | NCA-full (10k) | inv | alone | +0.0096 (+0.0064 to +0.0128) | NCA beats cosine | -0.0275 (-0.0376 to -0.0176) |
| inspire | NCA-full (10k) | inv | mix, logistic | +0.0014 (+0.0001 to +0.0027) | NCA beats cosine | +0.0029 (+0.0020 to +0.0037) |
| inspire | NCA-full (10k) | inv | mix, LightGBM | -0.0004 (-0.0012 to +0.0004) | no difference shown | +0.0024 (+0.0018 to +0.0029) |
| inspire | MI | ours | alone | +0.0085 (+0.0054 to +0.0115) | NCA beats cosine | +0.0080 (+0.0067 to +0.0094) |
| inspire | MI | ours | mix, logistic | +0.0019 (+0.0007 to +0.0031) | NCA beats cosine | +0.0034 (+0.0027 to +0.0040) |
| inspire | MI | ours | mix, LightGBM | -0.0003 (-0.0010 to +0.0004) | no difference shown | +0.0019 (+0.0015 to +0.0023) |
| inspire | MI | inv | alone | +0.0078 (+0.0049 to +0.0108) | NCA beats cosine | -0.0086 (-0.0171 to -0.0002) |
| inspire | MI | inv | mix, logistic | +0.0018 (+0.0006 to +0.0030) | NCA beats cosine | +0.0033 (+0.0025 to +0.0040) |
| inspire | MI | inv | mix, LightGBM | -0.0004 (-0.0011 to +0.0003) | no difference shown | +0.0018 (+0.0013 to +0.0022) |
| mimic | NCA-diag | ours | alone | +0.0067 (+0.0056 to +0.0079) | NCA beats cosine | +0.0077 (+0.0066 to +0.0088) |
| mimic | NCA-diag | ours | mix, logistic | +0.0004 (-0.0002 to +0.0009) | no difference shown | +0.0013 (+0.0009 to +0.0017) |
| mimic | NCA-diag | ours | mix, LightGBM | -0.0014 (-0.0017 to -0.0010) | cosine beats NCA | -0.0033 (-0.0037 to -0.0029) |
| mimic | NCA-diag | inv | alone | +0.0065 (+0.0053 to +0.0076) | NCA beats cosine | -0.0057 (-0.0093 to -0.0022) |
| mimic | NCA-diag | inv | mix, logistic | +0.0003 (-0.0002 to +0.0008) | no difference shown | +0.0011 (+0.0006 to +0.0015) |
| mimic | NCA-diag | inv | mix, LightGBM | -0.0013 (-0.0017 to -0.0010) | cosine beats NCA | -0.0031 (-0.0034 to -0.0027) |
| mimic | NCA-full | ours | alone | +0.0021 (+0.0011 to +0.0031) | NCA beats cosine | +0.0040 (+0.0031 to +0.0049) |
| mimic | NCA-full | ours | mix, logistic | -0.0005 (-0.0009 to -0.0001) | cosine beats NCA | -0.0008 (-0.0012 to -0.0005) |
| mimic | NCA-full | ours | mix, LightGBM | -0.0011 (-0.0014 to -0.0008) | cosine beats NCA | -0.0029 (-0.0032 to -0.0026) |
| mimic | NCA-full | inv | alone | +0.0020 (+0.0010 to +0.0030) | NCA beats cosine | -0.0025 (-0.0055 to +0.0004) |
| mimic | NCA-full | inv | mix, logistic | -0.0005 (-0.0009 to -0.0001) | cosine beats NCA | -0.0010 (-0.0014 to -0.0007) |
| mimic | NCA-full | inv | mix, LightGBM | -0.0011 (-0.0014 to -0.0008) | cosine beats NCA | -0.0026 (-0.0029 to -0.0023) |
| mimic | NCA-full (2k) | ours | alone | -0.0010 (-0.0020 to +0.0001) | no difference shown | +0.0014 (+0.0005 to +0.0024) |
| mimic | NCA-full (2k) | ours | mix, logistic | -0.0003 (-0.0007 to +0.0001) | no difference shown | -0.0010 (-0.0014 to -0.0007) |
| mimic | NCA-full (2k) | ours | mix, LightGBM | -0.0010 (-0.0014 to -0.0007) | cosine beats NCA | -0.0031 (-0.0034 to -0.0028) |
| mimic | NCA-full (2k) | inv | alone | -0.0010 (-0.0021 to +0.0000) | no difference shown | -0.0022 (-0.0050 to +0.0006) |
| mimic | NCA-full (2k) | inv | mix, logistic | -0.0004 (-0.0008 to +0.0000) | no difference shown | -0.0010 (-0.0014 to -0.0007) |
| mimic | NCA-full (2k) | inv | mix, LightGBM | -0.0010 (-0.0013 to -0.0008) | cosine beats NCA | -0.0033 (-0.0037 to -0.0030) |
| mimic | NCA-full (10k) | ours | alone | +0.0027 (+0.0017 to +0.0037) | NCA beats cosine | +0.0045 (+0.0036 to +0.0054) |
| mimic | NCA-full (10k) | ours | mix, logistic | -0.0007 (-0.0011 to -0.0003) | cosine beats NCA | -0.0011 (-0.0014 to -0.0008) |
| mimic | NCA-full (10k) | ours | mix, LightGBM | -0.0012 (-0.0015 to -0.0009) | cosine beats NCA | -0.0030 (-0.0033 to -0.0027) |
| mimic | NCA-full (10k) | inv | alone | +0.0025 (+0.0015 to +0.0036) | NCA beats cosine | +0.0002 (-0.0026 to +0.0030) |
| mimic | NCA-full (10k) | inv | mix, logistic | -0.0008 (-0.0012 to -0.0004) | cosine beats NCA | -0.0013 (-0.0017 to -0.0009) |
| mimic | NCA-full (10k) | inv | mix, LightGBM | -0.0012 (-0.0015 to -0.0009) | cosine beats NCA | -0.0027 (-0.0030 to -0.0024) |
| mimic | MI | ours | alone | +0.0076 (+0.0066 to +0.0086) | NCA beats cosine | +0.0085 (+0.0077 to +0.0095) |
| mimic | MI | ours | mix, logistic | +0.0011 (+0.0007 to +0.0016) | NCA beats cosine | +0.0020 (+0.0017 to +0.0024) |
| mimic | MI | ours | mix, LightGBM | -0.0005 (-0.0008 to -0.0002) | cosine beats NCA | -0.0016 (-0.0019 to -0.0013) |
| mimic | MI | inv | alone | +0.0075 (+0.0065 to +0.0085) | NCA beats cosine | +0.0008 (-0.0022 to +0.0038) |
| mimic | MI | inv | mix, logistic | +0.0011 (+0.0006 to +0.0015) | NCA beats cosine | +0.0020 (+0.0016 to +0.0024) |
| mimic | MI | inv | mix, LightGBM | -0.0005 (-0.0008 to -0.0002) | cosine beats NCA | -0.0013 (-0.0016 to -0.0010) |
| eicu | NCA-diag | ours | alone | +0.0159 (+0.0146 to +0.0171) | NCA beats cosine | +0.0146 (+0.0137 to +0.0155) |
| eicu | NCA-diag | ours | mix, logistic | +0.0044 (+0.0039 to +0.0048) | NCA beats cosine | +0.0082 (+0.0079 to +0.0086) |
| eicu | NCA-diag | ours | mix, LightGBM | +0.0007 (+0.0005 to +0.0009) | NCA beats cosine | +0.0034 (+0.0032 to +0.0036) |
| eicu | NCA-diag | inv | alone | +0.0155 (+0.0143 to +0.0168) | NCA beats cosine | -0.0114 (-0.0160 to -0.0067) |
| eicu | NCA-diag | inv | mix, logistic | +0.0043 (+0.0039 to +0.0048) | NCA beats cosine | +0.0083 (+0.0079 to +0.0087) |
| eicu | NCA-diag | inv | mix, LightGBM | +0.0007 (+0.0005 to +0.0009) | NCA beats cosine | +0.0033 (+0.0031 to +0.0035) |
| eicu | NCA-full | ours | alone | +0.0092 (+0.0080 to +0.0104) | NCA beats cosine | +0.0095 (+0.0086 to +0.0103) |
| eicu | NCA-full | ours | mix, logistic | +0.0025 (+0.0021 to +0.0029) | NCA beats cosine | +0.0064 (+0.0061 to +0.0067) |
| eicu | NCA-full | ours | mix, LightGBM | +0.0003 (+0.0002 to +0.0005) | NCA beats cosine | +0.0027 (+0.0025 to +0.0028) |
| eicu | NCA-full | inv | alone | +0.0090 (+0.0078 to +0.0103) | NCA beats cosine | -0.0188 (-0.0235 to -0.0143) |
| eicu | NCA-full | inv | mix, logistic | +0.0024 (+0.0020 to +0.0028) | NCA beats cosine | +0.0060 (+0.0057 to +0.0064) |
| eicu | NCA-full | inv | mix, LightGBM | +0.0003 (+0.0001 to +0.0004) | NCA beats cosine | +0.0026 (+0.0024 to +0.0027) |
| eicu | NCA-full (2k) | ours | alone | +0.0092 (+0.0079 to +0.0104) | NCA beats cosine | +0.0093 (+0.0084 to +0.0102) |
| eicu | NCA-full (2k) | ours | mix, logistic | +0.0021 (+0.0018 to +0.0025) | NCA beats cosine | +0.0055 (+0.0052 to +0.0058) |
| eicu | NCA-full (2k) | ours | mix, LightGBM | +0.0001 (-0.0000 to +0.0003) | no difference shown | +0.0017 (+0.0016 to +0.0018) |
| eicu | NCA-full (2k) | inv | alone | +0.0091 (+0.0078 to +0.0103) | NCA beats cosine | -0.0125 (-0.0169 to -0.0082) |
| eicu | NCA-full (2k) | inv | mix, logistic | +0.0021 (+0.0018 to +0.0025) | NCA beats cosine | +0.0055 (+0.0052 to +0.0058) |
| eicu | NCA-full (2k) | inv | mix, LightGBM | +0.0001 (-0.0000 to +0.0002) | no difference shown | +0.0014 (+0.0013 to +0.0015) |
| eicu | NCA-full (10k) | ours | alone | +0.0119 (+0.0106 to +0.0131) | NCA beats cosine | +0.0114 (+0.0105 to +0.0122) |
| eicu | NCA-full (10k) | ours | mix, logistic | +0.0032 (+0.0028 to +0.0036) | NCA beats cosine | +0.0068 (+0.0065 to +0.0072) |
| eicu | NCA-full (10k) | ours | mix, LightGBM | +0.0005 (+0.0003 to +0.0006) | NCA beats cosine | +0.0031 (+0.0030 to +0.0033) |
| eicu | NCA-full (10k) | inv | alone | +0.0116 (+0.0104 to +0.0129) | NCA beats cosine | -0.0167 (-0.0213 to -0.0121) |
| eicu | NCA-full (10k) | inv | mix, logistic | +0.0031 (+0.0027 to +0.0035) | NCA beats cosine | +0.0068 (+0.0064 to +0.0072) |
| eicu | NCA-full (10k) | inv | mix, LightGBM | +0.0004 (+0.0003 to +0.0006) | NCA beats cosine | +0.0027 (+0.0025 to +0.0029) |
| eicu | MI | ours | alone | +0.0107 (+0.0094 to +0.0119) | NCA beats cosine | +0.0106 (+0.0098 to +0.0114) |
| eicu | MI | ours | mix, logistic | +0.0027 (+0.0023 to +0.0031) | NCA beats cosine | +0.0059 (+0.0056 to +0.0062) |
| eicu | MI | ours | mix, LightGBM | +0.0005 (+0.0003 to +0.0006) | NCA beats cosine | +0.0025 (+0.0023 to +0.0026) |
| eicu | MI | inv | alone | +0.0105 (+0.0092 to +0.0117) | NCA beats cosine | -0.0065 (-0.0109 to -0.0025) |
| eicu | MI | inv | mix, logistic | +0.0027 (+0.0022 to +0.0031) | NCA beats cosine | +0.0059 (+0.0055 to +0.0062) |
| eicu | MI | inv | mix, LightGBM | +0.0004 (+0.0003 to +0.0006) | NCA beats cosine | +0.0022 (+0.0020 to +0.0023) |
| NWICU (mimic+eicu) | NCA-diag | ours | alone | +0.0171 (+0.0100 to +0.0248) | NCA beats cosine | +0.0100 (+0.0065 to +0.0136) |
| NWICU (mimic+eicu) | NCA-diag | ours | mix, logistic | +0.0015 (-0.0005 to +0.0035) | no difference shown | +0.0101 (+0.0088 to +0.0114) |
| NWICU (mimic+eicu) | NCA-diag | ours | mix, LightGBM | -0.0004 (-0.0014 to +0.0007) | no difference shown | +0.0008 (+0.0002 to +0.0015) |
| NWICU (mimic+eicu) | NCA-diag | inv | alone | +0.0177 (+0.0105 to +0.0254) | NCA beats cosine | -0.0229 (-0.0434 to -0.0037) |
| NWICU (mimic+eicu) | NCA-diag | inv | mix, logistic | +0.0012 (-0.0009 to +0.0033) | no difference shown | +0.0103 (+0.0089 to +0.0117) |
| NWICU (mimic+eicu) | NCA-diag | inv | mix, LightGBM | -0.0007 (-0.0018 to +0.0005) | no difference shown | -0.0000 (-0.0007 to +0.0007) |
| NWICU (mimic+eicu) | NCA-full | ours | alone | +0.0199 (+0.0124 to +0.0272) | NCA beats cosine | +0.0117 (+0.0082 to +0.0153) |
| NWICU (mimic+eicu) | NCA-full | ours | mix, logistic | +0.0016 (-0.0005 to +0.0036) | no difference shown | +0.0118 (+0.0104 to +0.0131) |
| NWICU (mimic+eicu) | NCA-full | ours | mix, LightGBM | +0.0001 (-0.0011 to +0.0014) | no difference shown | +0.0037 (+0.0030 to +0.0044) |
| NWICU (mimic+eicu) | NCA-full | inv | alone | +0.0195 (+0.0121 to +0.0268) | NCA beats cosine | -0.0124 (-0.0328 to +0.0058) |
| NWICU (mimic+eicu) | NCA-full | inv | mix, logistic | +0.0012 (-0.0008 to +0.0033) | no difference shown | +0.0115 (+0.0101 to +0.0128) |
| NWICU (mimic+eicu) | NCA-full | inv | mix, LightGBM | -0.0003 (-0.0016 to +0.0010) | no difference shown | +0.0028 (+0.0021 to +0.0036) |
| NWICU (mimic+eicu) | NCA-full (2k) | ours | alone | +0.0125 (+0.0054 to +0.0200) | NCA beats cosine | +0.0078 (+0.0047 to +0.0112) |
| NWICU (mimic+eicu) | NCA-full (2k) | ours | mix, logistic | -0.0006 (-0.0025 to +0.0015) | no difference shown | +0.0083 (+0.0072 to +0.0094) |
| NWICU (mimic+eicu) | NCA-full (2k) | ours | mix, LightGBM | -0.0002 (-0.0015 to +0.0010) | no difference shown | +0.0030 (+0.0023 to +0.0037) |
| NWICU (mimic+eicu) | NCA-full (2k) | inv | alone | +0.0125 (+0.0054 to +0.0198) | NCA beats cosine | -0.0146 (-0.0325 to +0.0029) |
| NWICU (mimic+eicu) | NCA-full (2k) | inv | mix, logistic | -0.0013 (-0.0033 to +0.0010) | no difference shown | +0.0072 (+0.0061 to +0.0085) |
| NWICU (mimic+eicu) | NCA-full (2k) | inv | mix, LightGBM | -0.0009 (-0.0022 to +0.0005) | no difference shown | +0.0005 (-0.0003 to +0.0012) |
| NWICU (mimic+eicu) | NCA-full (10k) | ours | alone | +0.0151 (+0.0080 to +0.0223) | NCA beats cosine | +0.0089 (+0.0055 to +0.0123) |
| NWICU (mimic+eicu) | NCA-full (10k) | ours | mix, logistic | +0.0009 (-0.0010 to +0.0028) | no difference shown | +0.0115 (+0.0101 to +0.0128) |
| NWICU (mimic+eicu) | NCA-full (10k) | ours | mix, LightGBM | -0.0003 (-0.0015 to +0.0008) | no difference shown | +0.0031 (+0.0024 to +0.0038) |
| NWICU (mimic+eicu) | NCA-full (10k) | inv | alone | +0.0158 (+0.0087 to +0.0233) | NCA beats cosine | -0.0194 (-0.0377 to -0.0013) |
| NWICU (mimic+eicu) | NCA-full (10k) | inv | mix, logistic | +0.0006 (-0.0013 to +0.0026) | no difference shown | +0.0107 (+0.0093 to +0.0120) |
| NWICU (mimic+eicu) | NCA-full (10k) | inv | mix, LightGBM | -0.0007 (-0.0019 to +0.0006) | no difference shown | +0.0024 (+0.0016 to +0.0031) |
| NWICU (mimic+eicu) | MI | ours | alone | +0.0092 (+0.0020 to +0.0166) | NCA beats cosine | +0.0057 (+0.0024 to +0.0090) |
| NWICU (mimic+eicu) | MI | ours | mix, logistic | -0.0011 (-0.0026 to +0.0005) | no difference shown | -0.0020 (-0.0028 to -0.0011) |
| NWICU (mimic+eicu) | MI | ours | mix, LightGBM | -0.0003 (-0.0013 to +0.0007) | no difference shown | -0.0071 (-0.0080 to -0.0062) |
| NWICU (mimic+eicu) | MI | inv | alone | +0.0109 (+0.0039 to +0.0184) | NCA beats cosine | -0.0212 (-0.0419 to -0.0025) |
| NWICU (mimic+eicu) | MI | inv | mix, logistic | -0.0005 (-0.0021 to +0.0012) | no difference shown | -0.0001 (-0.0011 to +0.0008) |
| NWICU (mimic+eicu) | MI | inv | mix, LightGBM | -0.0007 (-0.0018 to +0.0004) | no difference shown | -0.0076 (-0.0085 to -0.0066) |
| NWICU (inspire+eicu) | NCA-diag | ours | alone | +0.0179 (+0.0110 to +0.0257) | NCA beats cosine | +0.0106 (+0.0069 to +0.0142) |
| NWICU (inspire+eicu) | NCA-diag | ours | mix, logistic | +0.0009 (-0.0010 to +0.0030) | no difference shown | +0.0063 (+0.0052 to +0.0075) |
| NWICU (inspire+eicu) | NCA-diag | ours | mix, LightGBM | -0.0008 (-0.0019 to +0.0004) | no difference shown | +0.0003 (-0.0004 to +0.0010) |
| NWICU (inspire+eicu) | NCA-diag | inv | alone | +0.0186 (+0.0114 to +0.0266) | NCA beats cosine | -0.0350 (-0.0566 to -0.0153) |
| NWICU (inspire+eicu) | NCA-diag | inv | mix, logistic | +0.0007 (-0.0013 to +0.0030) | no difference shown | +0.0059 (+0.0047 to +0.0073) |
| NWICU (inspire+eicu) | NCA-diag | inv | mix, LightGBM | -0.0005 (-0.0019 to +0.0008) | no difference shown | +0.0002 (-0.0006 to +0.0010) |
| NWICU (inspire+eicu) | NCA-full | ours | alone | +0.0101 (+0.0034 to +0.0173) | NCA beats cosine | +0.0064 (+0.0031 to +0.0098) |
| NWICU (inspire+eicu) | NCA-full | ours | mix, logistic | -0.0001 (-0.0018 to +0.0018) | no difference shown | +0.0048 (+0.0038 to +0.0057) |
| NWICU (inspire+eicu) | NCA-full | ours | mix, LightGBM | -0.0006 (-0.0017 to +0.0005) | no difference shown | +0.0011 (+0.0004 to +0.0018) |
| NWICU (inspire+eicu) | NCA-full | inv | alone | +0.0105 (+0.0038 to +0.0178) | NCA beats cosine | -0.0146 (-0.0325 to +0.0038) |
| NWICU (inspire+eicu) | NCA-full | inv | mix, logistic | -0.0000 (-0.0018 to +0.0018) | no difference shown | +0.0042 (+0.0033 to +0.0052) |
| NWICU (inspire+eicu) | NCA-full | inv | mix, LightGBM | -0.0003 (-0.0014 to +0.0010) | no difference shown | +0.0008 (+0.0001 to +0.0015) |
| NWICU (inspire+eicu) | NCA-full (2k) | ours | alone | +0.0061 (-0.0006 to +0.0133) | no difference shown | +0.0043 (+0.0011 to +0.0075) |
| NWICU (inspire+eicu) | NCA-full (2k) | ours | mix, logistic | -0.0006 (-0.0023 to +0.0012) | no difference shown | +0.0027 (+0.0019 to +0.0036) |
| NWICU (inspire+eicu) | NCA-full (2k) | ours | mix, LightGBM | -0.0006 (-0.0017 to +0.0005) | no difference shown | -0.0001 (-0.0007 to +0.0006) |
| NWICU (inspire+eicu) | NCA-full (2k) | inv | alone | +0.0057 (-0.0011 to +0.0130) | no difference shown | -0.0216 (-0.0417 to -0.0032) |
| NWICU (inspire+eicu) | NCA-full (2k) | inv | mix, logistic | -0.0007 (-0.0024 to +0.0011) | no difference shown | +0.0022 (+0.0014 to +0.0031) |
| NWICU (inspire+eicu) | NCA-full (2k) | inv | mix, LightGBM | -0.0006 (-0.0018 to +0.0007) | no difference shown | -0.0002 (-0.0009 to +0.0005) |
| NWICU (inspire+eicu) | NCA-full (10k) | ours | alone | +0.0027 (-0.0048 to +0.0103) | no difference shown | +0.0025 (-0.0011 to +0.0061) |
| NWICU (inspire+eicu) | NCA-full (10k) | ours | mix, logistic | -0.0024 (-0.0040 to -0.0007) | cosine beats NCA | +0.0003 (-0.0004 to +0.0012) |
| NWICU (inspire+eicu) | NCA-full (10k) | ours | mix, LightGBM | -0.0012 (-0.0024 to +0.0000) | no difference shown | -0.0022 (-0.0031 to -0.0013) |
| NWICU (inspire+eicu) | NCA-full (10k) | inv | alone | +0.0032 (-0.0043 to +0.0106) | no difference shown | -0.0377 (-0.0598 to -0.0161) |
| NWICU (inspire+eicu) | NCA-full (10k) | inv | mix, logistic | -0.0021 (-0.0038 to -0.0004) | cosine beats NCA | +0.0000 (-0.0008 to +0.0009) |
| NWICU (inspire+eicu) | NCA-full (10k) | inv | mix, LightGBM | -0.0012 (-0.0025 to +0.0002) | no difference shown | -0.0028 (-0.0037 to -0.0018) |
| NWICU (inspire+eicu) | MI | ours | alone | +0.0116 (+0.0044 to +0.0194) | NCA beats cosine | +0.0073 (+0.0037 to +0.0109) |
| NWICU (inspire+eicu) | MI | ours | mix, logistic | -0.0021 (-0.0037 to -0.0004) | cosine beats NCA | +0.0002 (-0.0006 to +0.0011) |
| NWICU (inspire+eicu) | MI | ours | mix, LightGBM | -0.0012 (-0.0024 to -0.0000) | cosine beats NCA | -0.0032 (-0.0041 to -0.0023) |
| NWICU (inspire+eicu) | MI | inv | alone | +0.0123 (+0.0052 to +0.0203) | NCA beats cosine | -0.0231 (-0.0417 to -0.0044) |
| NWICU (inspire+eicu) | MI | inv | mix, logistic | -0.0017 (-0.0034 to +0.0002) | no difference shown | -0.0004 (-0.0013 to +0.0006) |
| NWICU (inspire+eicu) | MI | inv | mix, LightGBM | -0.0011 (-0.0024 to +0.0003) | no difference shown | -0.0035 (-0.0044 to -0.0025) |
| NWICU (inspire+mimic) | NCA-diag | ours | alone | +0.0190 (+0.0118 to +0.0268) | NCA beats cosine | +0.0113 (+0.0079 to +0.0149) |
| NWICU (inspire+mimic) | NCA-diag | ours | mix, logistic | +0.0022 (+0.0001 to +0.0044) | NCA beats cosine | +0.0140 (+0.0125 to +0.0155) |
| NWICU (inspire+mimic) | NCA-diag | ours | mix, LightGBM | -0.0003 (-0.0016 to +0.0009) | no difference shown | +0.0020 (+0.0013 to +0.0027) |
| NWICU (inspire+mimic) | NCA-diag | inv | alone | +0.0189 (+0.0118 to +0.0269) | NCA beats cosine | -0.0232 (-0.0447 to -0.0027) |
| NWICU (inspire+mimic) | NCA-diag | inv | mix, logistic | +0.0013 (-0.0008 to +0.0037) | no difference shown | +0.0114 (+0.0100 to +0.0129) |
| NWICU (inspire+mimic) | NCA-diag | inv | mix, LightGBM | -0.0009 (-0.0022 to +0.0005) | no difference shown | +0.0006 (-0.0002 to +0.0014) |
| NWICU (inspire+mimic) | NCA-full | ours | alone | +0.0041 (-0.0035 to +0.0124) | no difference shown | +0.0033 (-0.0004 to +0.0070) |
| NWICU (inspire+mimic) | NCA-full | ours | mix, logistic | -0.0004 (-0.0024 to +0.0017) | no difference shown | +0.0061 (+0.0050 to +0.0072) |
| NWICU (inspire+mimic) | NCA-full | ours | mix, LightGBM | -0.0003 (-0.0016 to +0.0011) | no difference shown | -0.0017 (-0.0025 to -0.0009) |
| NWICU (inspire+mimic) | NCA-full | inv | alone | +0.0048 (-0.0028 to +0.0127) | no difference shown | -0.0235 (-0.0440 to -0.0041) |
| NWICU (inspire+mimic) | NCA-full | inv | mix, logistic | -0.0009 (-0.0031 to +0.0014) | no difference shown | +0.0048 (+0.0037 to +0.0060) |
| NWICU (inspire+mimic) | NCA-full | inv | mix, LightGBM | -0.0005 (-0.0018 to +0.0009) | no difference shown | -0.0038 (-0.0046 to -0.0029) |
| NWICU (inspire+mimic) | NCA-full (2k) | ours | alone | +0.0035 (-0.0036 to +0.0116) | no difference shown | +0.0026 (-0.0008 to +0.0061) |
| NWICU (inspire+mimic) | NCA-full (2k) | ours | mix, logistic | -0.0011 (-0.0031 to +0.0012) | no difference shown | +0.0058 (+0.0048 to +0.0068) |
| NWICU (inspire+mimic) | NCA-full (2k) | ours | mix, LightGBM | -0.0011 (-0.0024 to +0.0004) | no difference shown | -0.0042 (-0.0050 to -0.0033) |
| NWICU (inspire+mimic) | NCA-full (2k) | inv | alone | +0.0040 (-0.0034 to +0.0120) | no difference shown | -0.0358 (-0.0569 to -0.0163) |
| NWICU (inspire+mimic) | NCA-full (2k) | inv | mix, logistic | -0.0016 (-0.0038 to +0.0007) | no difference shown | +0.0038 (+0.0028 to +0.0048) |
| NWICU (inspire+mimic) | NCA-full (2k) | inv | mix, LightGBM | -0.0013 (-0.0027 to +0.0002) | no difference shown | -0.0048 (-0.0057 to -0.0039) |
| NWICU (inspire+mimic) | NCA-full (10k) | ours | alone | +0.0100 (+0.0027 to +0.0176) | NCA beats cosine | +0.0060 (+0.0025 to +0.0095) |
| NWICU (inspire+mimic) | NCA-full (10k) | ours | mix, logistic | +0.0005 (-0.0013 to +0.0025) | no difference shown | +0.0086 (+0.0075 to +0.0097) |
| NWICU (inspire+mimic) | NCA-full (10k) | ours | mix, LightGBM | -0.0000 (-0.0012 to +0.0012) | no difference shown | -0.0002 (-0.0009 to +0.0005) |
| NWICU (inspire+mimic) | NCA-full (10k) | inv | alone | +0.0111 (+0.0038 to +0.0185) | NCA beats cosine | -0.0273 (-0.0490 to -0.0071) |
| NWICU (inspire+mimic) | NCA-full (10k) | inv | mix, logistic | +0.0004 (-0.0016 to +0.0026) | no difference shown | +0.0061 (+0.0049 to +0.0072) |
| NWICU (inspire+mimic) | NCA-full (10k) | inv | mix, LightGBM | -0.0002 (-0.0015 to +0.0012) | no difference shown | -0.0007 (-0.0014 to +0.0001) |
| NWICU (inspire+mimic) | MI | ours | alone | +0.0074 (+0.0004 to +0.0149) | NCA beats cosine | +0.0048 (+0.0015 to +0.0081) |
| NWICU (inspire+mimic) | MI | ours | mix, logistic | -0.0015 (-0.0034 to +0.0005) | no difference shown | -0.0006 (-0.0016 to +0.0004) |
| NWICU (inspire+mimic) | MI | ours | mix, LightGBM | -0.0006 (-0.0018 to +0.0006) | no difference shown | -0.0061 (-0.0070 to -0.0051) |
| NWICU (inspire+mimic) | MI | inv | alone | +0.0085 (+0.0019 to +0.0159) | NCA beats cosine | -0.0272 (-0.0480 to -0.0073) |
| NWICU (inspire+mimic) | MI | inv | mix, logistic | -0.0014 (-0.0033 to +0.0008) | no difference shown | -0.0011 (-0.0021 to +0.0000) |
| NWICU (inspire+mimic) | MI | inv | mix, LightGBM | -0.0007 (-0.0019 to +0.0006) | no difference shown | -0.0067 (-0.0076 to -0.0057) |

## Every arm

| target | arm | AUROC | log-loss | CITL | slope |
|---|---|---|---|---|---|
| inspire | cos|ours|alone | 0.7611 | 0.3404 | -0.115 | 1.122 |
| inspire | cos|ours|mix_lr | 0.7809 | 0.3298 | -0.132 | 1.030 |
| inspire | cos|ours|mix_gbm | 0.7909 | 0.3265 | -0.261 | 1.109 |
| inspire | cos|inv|alone | 0.7623 | 0.3770 | -0.123 | 0.833 |
| inspire | cos|inv|mix_lr | 0.7812 | 0.3304 | -0.136 | 0.971 |
| inspire | cos|inv|mix_gbm | 0.7911 | 0.3264 | -0.267 | 1.076 |
| inspire | mi|ours|alone | 0.7696 | 0.3324 | +0.021 | 1.061 |
| inspire | mi|ours|mix_lr | 0.7828 | 0.3264 | -0.063 | 1.014 |
| inspire | mi|ours|mix_gbm | 0.7905 | 0.3246 | -0.224 | 1.057 |
| inspire | mi|inv|alone | 0.7701 | 0.3856 | +0.023 | 0.754 |
| inspire | mi|inv|mix_lr | 0.7830 | 0.3271 | -0.063 | 0.945 |
| inspire | mi|inv|mix_gbm | 0.7907 | 0.3247 | -0.233 | 1.019 |
| inspire | ncad|ours|alone | 0.7788 | 0.3276 | +0.023 | 1.019 |
| inspire | ncad|ours|mix_lr | 0.7872 | 0.3242 | -0.063 | 0.994 |
| inspire | ncad|ours|mix_gbm | 0.7930 | 0.3225 | -0.180 | 1.063 |
| inspire | ncad|inv|alone | 0.7793 | 0.3845 | +0.025 | 0.741 |
| inspire | ncad|inv|mix_lr | 0.7875 | 0.3248 | -0.063 | 0.937 |
| inspire | ncad|inv|mix_gbm | 0.7931 | 0.3226 | -0.188 | 1.017 |
| inspire | ncaf|ours|alone | 0.7707 | 0.3318 | +0.067 | 1.043 |
| inspire | ncaf|ours|mix_lr | 0.7820 | 0.3266 | -0.041 | 0.990 |
| inspire | ncaf|ours|mix_gbm | 0.7904 | 0.3243 | -0.204 | 1.049 |
| inspire | ncaf|inv|alone | 0.7710 | 0.3941 | +0.075 | 0.719 |
| inspire | ncaf|inv|mix_lr | 0.7822 | 0.3276 | -0.039 | 0.921 |
| inspire | ncaf|inv|mix_gbm | 0.7905 | 0.3243 | -0.209 | 1.009 |
| inspire | ncaf2k|ours|alone | 0.7716 | 0.3322 | +0.058 | 1.048 |
| inspire | ncaf2k|ours|mix_lr | 0.7829 | 0.3267 | -0.045 | 0.996 |
| inspire | ncaf2k|ours|mix_gbm | 0.7909 | 0.3244 | -0.210 | 1.052 |
| inspire | ncaf2k|inv|alone | 0.7722 | 0.3912 | +0.063 | 0.742 |
| inspire | ncaf2k|inv|mix_lr | 0.7831 | 0.3275 | -0.044 | 0.937 |
| inspire | ncaf2k|inv|mix_gbm | 0.7910 | 0.3243 | -0.210 | 1.022 |
| inspire | ncaf10k|ours|alone | 0.7712 | 0.3318 | +0.085 | 1.061 |
| inspire | ncaf10k|ours|mix_lr | 0.7823 | 0.3265 | -0.032 | 0.998 |
| inspire | ncaf10k|ours|mix_gbm | 0.7906 | 0.3240 | -0.190 | 1.057 |
| inspire | ncaf10k|inv|alone | 0.7719 | 0.4045 | +0.099 | 0.681 |
| inspire | ncaf10k|inv|mix_lr | 0.7826 | 0.3275 | -0.028 | 0.915 |
| inspire | ncaf10k|inv|mix_gbm | 0.7907 | 0.3241 | -0.198 | 1.006 |
| inspire | base|lr | 0.7737 | 0.3385 | -0.153 | 0.741 |
| inspire | base|gbm | 0.7880 | 0.3296 | -0.340 | 0.941 |
| mimic | cos|ours|alone | 0.7739 | 0.5136 | -0.104 | 1.065 |
| mimic | cos|ours|mix_lr | 0.7867 | 0.5056 | +0.180 | 1.188 |
| mimic | cos|ours|mix_gbm | 0.7916 | 0.5005 | +0.233 | 1.094 |
| mimic | cos|inv|alone | 0.7743 | 0.5218 | -0.105 | 1.003 |
| mimic | cos|inv|mix_lr | 0.7868 | 0.5052 | +0.181 | 1.170 |
| mimic | cos|inv|mix_gbm | 0.7916 | 0.5007 | +0.245 | 1.075 |
| mimic | mi|ours|alone | 0.7815 | 0.5051 | +0.012 | 1.005 |
| mimic | mi|ours|mix_lr | 0.7878 | 0.5035 | +0.194 | 1.132 |
| mimic | mi|ours|mix_gbm | 0.7911 | 0.5022 | +0.278 | 1.039 |
| mimic | mi|inv|alone | 0.7817 | 0.5210 | +0.014 | 0.908 |
| mimic | mi|inv|mix_lr | 0.7879 | 0.5032 | +0.196 | 1.095 |
| mimic | mi|inv|mix_gbm | 0.7911 | 0.5021 | +0.280 | 1.018 |
| mimic | ncad|ours|alone | 0.7806 | 0.5059 | +0.021 | 1.011 |
| mimic | ncad|ours|mix_lr | 0.7871 | 0.5043 | +0.197 | 1.132 |
| mimic | ncad|ours|mix_gbm | 0.7902 | 0.5039 | +0.300 | 1.027 |
| mimic | ncad|inv|alone | 0.7807 | 0.5276 | +0.024 | 0.889 |
| mimic | ncad|inv|mix_lr | 0.7871 | 0.5041 | +0.204 | 1.088 |
| mimic | ncad|inv|mix_gbm | 0.7903 | 0.5038 | +0.303 | 1.005 |
| mimic | ncaf|ours|alone | 0.7760 | 0.5096 | +0.004 | 1.026 |
| mimic | ncaf|ours|mix_lr | 0.7862 | 0.5064 | +0.219 | 1.154 |
| mimic | ncaf|ours|mix_gbm | 0.7905 | 0.5034 | +0.294 | 1.048 |
| mimic | ncaf|inv|alone | 0.7762 | 0.5243 | +0.005 | 0.928 |
| mimic | ncaf|inv|mix_lr | 0.7863 | 0.5063 | +0.224 | 1.124 |
| mimic | ncaf|inv|mix_gbm | 0.7905 | 0.5033 | +0.296 | 1.030 |
| mimic | ncaf2k|ours|alone | 0.7729 | 0.5122 | +0.005 | 1.023 |
| mimic | ncaf2k|ours|mix_lr | 0.7864 | 0.5066 | +0.222 | 1.170 |
| mimic | ncaf2k|ours|mix_gbm | 0.7906 | 0.5036 | +0.298 | 1.053 |
| mimic | ncaf2k|inv|alone | 0.7732 | 0.5241 | +0.007 | 0.937 |
| mimic | ncaf2k|inv|mix_lr | 0.7864 | 0.5063 | +0.224 | 1.143 |
| mimic | ncaf2k|inv|mix_gbm | 0.7906 | 0.5041 | +0.312 | 1.032 |
| mimic | ncaf10k|ours|alone | 0.7765 | 0.5091 | +0.011 | 1.034 |
| mimic | ncaf10k|ours|mix_lr | 0.7860 | 0.5067 | +0.222 | 1.151 |
| mimic | ncaf10k|ours|mix_gbm | 0.7904 | 0.5036 | +0.296 | 1.047 |
| mimic | ncaf10k|inv|alone | 0.7768 | 0.5216 | +0.012 | 0.938 |
| mimic | ncaf10k|inv|mix_lr | 0.7860 | 0.5065 | +0.227 | 1.120 |
| mimic | ncaf10k|inv|mix_gbm | 0.7904 | 0.5035 | +0.298 | 1.029 |
| mimic | base|lr | 0.7814 | 0.5180 | +0.350 | 1.061 |
| mimic | base|gbm | 0.7886 | 0.5110 | +0.407 | 0.967 |
| eicu | cos|ours|alone | 0.7833 | 0.4112 | -0.112 | 1.124 |
| eicu | cos|ours|mix_lr | 0.8018 | 0.4029 | -0.290 | 1.153 |
| eicu | cos|ours|mix_gbm | 0.8104 | 0.3966 | -0.362 | 1.024 |
| eicu | cos|inv|alone | 0.7838 | 0.4328 | -0.114 | 0.975 |
| eicu | cos|inv|mix_lr | 0.8019 | 0.4028 | -0.291 | 1.133 |
| eicu | cos|inv|mix_gbm | 0.8104 | 0.3965 | -0.360 | 1.022 |
| eicu | mi|ours|alone | 0.7941 | 0.4005 | -0.006 | 1.065 |
| eicu | mi|ours|mix_lr | 0.8045 | 0.3970 | -0.208 | 1.148 |
| eicu | mi|ours|mix_gbm | 0.8108 | 0.3942 | -0.316 | 1.032 |
| eicu | mi|inv|alone | 0.7943 | 0.4394 | -0.007 | 0.864 |
| eicu | mi|inv|mix_lr | 0.8046 | 0.3969 | -0.214 | 1.096 |
| eicu | mi|inv|mix_gbm | 0.8109 | 0.3943 | -0.322 | 1.012 |
| eicu | ncad|ours|alone | 0.7992 | 0.3965 | +0.008 | 1.025 |
| eicu | ncad|ours|mix_lr | 0.8062 | 0.3947 | -0.186 | 1.123 |
| eicu | ncad|ours|mix_gbm | 0.8111 | 0.3932 | -0.299 | 1.024 |
| eicu | ncad|inv|alone | 0.7993 | 0.4442 | +0.010 | 0.811 |
| eicu | ncad|inv|mix_lr | 0.8062 | 0.3945 | -0.188 | 1.063 |
| eicu | ncad|inv|mix_gbm | 0.8111 | 0.3932 | -0.301 | 1.001 |
| eicu | ncaf|ours|alone | 0.7925 | 0.4017 | +0.061 | 1.054 |
| eicu | ncaf|ours|mix_lr | 0.8043 | 0.3965 | -0.180 | 1.147 |
| eicu | ncaf|ours|mix_gbm | 0.8107 | 0.3940 | -0.308 | 1.028 |
| eicu | ncaf|inv|alone | 0.7928 | 0.4516 | +0.069 | 0.814 |
| eicu | ncaf|inv|mix_lr | 0.8043 | 0.3967 | -0.197 | 1.091 |
| eicu | ncaf|inv|mix_gbm | 0.8107 | 0.3939 | -0.309 | 1.009 |
| eicu | ncaf2k|ours|alone | 0.7925 | 0.4018 | +0.044 | 1.052 |
| eicu | ncaf2k|ours|mix_lr | 0.8039 | 0.3974 | -0.203 | 1.139 |
| eicu | ncaf2k|ours|mix_gbm | 0.8105 | 0.3949 | -0.329 | 1.016 |
| eicu | ncaf2k|inv|alone | 0.7929 | 0.4453 | +0.049 | 0.839 |
| eicu | ncaf2k|inv|mix_lr | 0.8041 | 0.3973 | -0.208 | 1.093 |
| eicu | ncaf2k|inv|mix_gbm | 0.8105 | 0.3951 | -0.335 | 1.001 |
| eicu | ncaf10k|ours|alone | 0.7952 | 0.3998 | +0.042 | 1.057 |
| eicu | ncaf10k|ours|mix_lr | 0.8050 | 0.3961 | -0.184 | 1.145 |
| eicu | ncaf10k|ours|mix_gbm | 0.8109 | 0.3935 | -0.298 | 1.034 |
| eicu | ncaf10k|inv|alone | 0.7954 | 0.4495 | +0.047 | 0.817 |
| eicu | ncaf10k|inv|mix_lr | 0.8050 | 0.3959 | -0.189 | 1.086 |
| eicu | ncaf10k|inv|mix_gbm | 0.8108 | 0.3938 | -0.309 | 1.008 |
| eicu | base|lr | 0.7998 | 0.4079 | -0.367 | 1.036 |
| eicu | base|gbm | 0.8101 | 0.3987 | -0.402 | 0.970 |
| NWICU (mimic+eicu) | cos|ours|alone | 0.7738 | 0.3293 | -0.076 | 1.137 |
| NWICU (mimic+eicu) | cos|ours|mix_lr | 0.8070 | 0.3352 | -0.606 | 1.327 |
| NWICU (mimic+eicu) | cos|ours|mix_gbm | 0.8151 | 0.3304 | -0.629 | 1.260 |
| NWICU (mimic+eicu) | cos|inv|alone | 0.7747 | 0.3634 | -0.081 | 0.842 |
| NWICU (mimic+eicu) | cos|inv|mix_lr | 0.8072 | 0.3354 | -0.618 | 1.291 |
| NWICU (mimic+eicu) | cos|inv|mix_gbm | 0.8154 | 0.3293 | -0.619 | 1.254 |
| NWICU (mimic+eicu) | mi|ours|alone | 0.7829 | 0.3237 | +0.051 | 1.056 |
| NWICU (mimic+eicu) | mi|ours|mix_lr | 0.8060 | 0.3372 | -0.647 | 1.237 |
| NWICU (mimic+eicu) | mi|ours|mix_gbm | 0.8148 | 0.3375 | -0.739 | 1.163 |
| NWICU (mimic+eicu) | mi|inv|alone | 0.7856 | 0.3846 | +0.057 | 0.712 |
| NWICU (mimic+eicu) | mi|inv|mix_lr | 0.8067 | 0.3355 | -0.633 | 1.217 |
| NWICU (mimic+eicu) | mi|inv|mix_gbm | 0.8147 | 0.3369 | -0.731 | 1.159 |
| NWICU (mimic+eicu) | ncad|ours|alone | 0.7908 | 0.3193 | +0.102 | 1.062 |
| NWICU (mimic+eicu) | ncad|ours|mix_lr | 0.8085 | 0.3251 | -0.482 | 1.267 |
| NWICU (mimic+eicu) | ncad|ours|mix_gbm | 0.8147 | 0.3295 | -0.631 | 1.192 |
| NWICU (mimic+eicu) | ncad|inv|alone | 0.7924 | 0.3863 | +0.110 | 0.700 |
| NWICU (mimic+eicu) | ncad|inv|mix_lr | 0.8083 | 0.3251 | -0.493 | 1.215 |
| NWICU (mimic+eicu) | ncad|inv|mix_gbm | 0.8147 | 0.3293 | -0.633 | 1.175 |
| NWICU (mimic+eicu) | ncaf|ours|alone | 0.7937 | 0.3176 | +0.084 | 1.050 |
| NWICU (mimic+eicu) | ncaf|ours|mix_lr | 0.8086 | 0.3234 | -0.463 | 1.254 |
| NWICU (mimic+eicu) | ncaf|ours|mix_gbm | 0.8152 | 0.3267 | -0.596 | 1.201 |
| NWICU (mimic+eicu) | ncaf|inv|alone | 0.7942 | 0.3759 | +0.090 | 0.732 |
| NWICU (mimic+eicu) | ncaf|inv|mix_lr | 0.8084 | 0.3239 | -0.483 | 1.207 |
| NWICU (mimic+eicu) | ncaf|inv|mix_gbm | 0.8151 | 0.3265 | -0.598 | 1.180 |
| NWICU (mimic+eicu) | ncaf2k|ours|alone | 0.7863 | 0.3215 | +0.102 | 1.114 |
| NWICU (mimic+eicu) | ncaf2k|ours|mix_lr | 0.8065 | 0.3270 | -0.497 | 1.282 |
| NWICU (mimic+eicu) | ncaf2k|ours|mix_gbm | 0.8149 | 0.3274 | -0.598 | 1.221 |
| NWICU (mimic+eicu) | ncaf2k|inv|alone | 0.7872 | 0.3780 | +0.111 | 0.761 |
| NWICU (mimic+eicu) | ncaf2k|inv|mix_lr | 0.8059 | 0.3282 | -0.525 | 1.230 |
| NWICU (mimic+eicu) | ncaf2k|inv|mix_gbm | 0.8145 | 0.3288 | -0.623 | 1.191 |
| NWICU (mimic+eicu) | ncaf10k|ours|alone | 0.7889 | 0.3205 | +0.157 | 1.074 |
| NWICU (mimic+eicu) | ncaf10k|ours|mix_lr | 0.8079 | 0.3238 | -0.455 | 1.276 |
| NWICU (mimic+eicu) | ncaf10k|ours|mix_gbm | 0.8147 | 0.3273 | -0.598 | 1.205 |
| NWICU (mimic+eicu) | ncaf10k|inv|alone | 0.7906 | 0.3828 | +0.173 | 0.718 |
| NWICU (mimic+eicu) | ncaf10k|inv|mix_lr | 0.8078 | 0.3247 | -0.483 | 1.226 |
| NWICU (mimic+eicu) | ncaf10k|inv|mix_gbm | 0.8148 | 0.3269 | -0.599 | 1.185 |
| NWICU (mimic+eicu) | base|lr | 0.8050 | 0.3559 | -0.860 | 1.160 |
| NWICU (mimic+eicu) | base|gbm | 0.8147 | 0.3419 | -0.793 | 1.140 |
| NWICU (inspire+eicu) | cos|ours|alone | 0.7738 | 0.3293 | -0.076 | 1.137 |
| NWICU (inspire+eicu) | cos|ours|mix_lr | 0.8078 | 0.3242 | -0.410 | 1.387 |
| NWICU (inspire+eicu) | cos|ours|mix_gbm | 0.8153 | 0.3150 | -0.376 | 1.218 |
| NWICU (inspire+eicu) | cos|inv|alone | 0.7747 | 0.3634 | -0.081 | 0.842 |
| NWICU (inspire+eicu) | cos|inv|mix_lr | 0.8077 | 0.3237 | -0.413 | 1.351 |
| NWICU (inspire+eicu) | cos|inv|mix_gbm | 0.8151 | 0.3147 | -0.374 | 1.200 |
| NWICU (inspire+eicu) | mi|ours|alone | 0.7853 | 0.3221 | +0.085 | 1.022 |
| NWICU (inspire+eicu) | mi|ours|mix_lr | 0.8057 | 0.3239 | -0.422 | 1.278 |
| NWICU (inspire+eicu) | mi|ours|mix_gbm | 0.8141 | 0.3182 | -0.465 | 1.092 |
| NWICU (inspire+eicu) | mi|inv|alone | 0.7870 | 0.3866 | +0.094 | 0.680 |
| NWICU (inspire+eicu) | mi|inv|mix_lr | 0.8060 | 0.3241 | -0.434 | 1.256 |
| NWICU (inspire+eicu) | mi|inv|mix_gbm | 0.8141 | 0.3182 | -0.465 | 1.091 |
| NWICU (inspire+eicu) | ncad|ours|alone | 0.7916 | 0.3188 | +0.087 | 1.001 |
| NWICU (inspire+eicu) | ncad|ours|mix_lr | 0.8087 | 0.3179 | -0.319 | 1.279 |
| NWICU (inspire+eicu) | ncad|ours|mix_gbm | 0.8145 | 0.3148 | -0.388 | 1.119 |
| NWICU (inspire+eicu) | ncad|inv|alone | 0.7933 | 0.3985 | +0.098 | 0.623 |
| NWICU (inspire+eicu) | ncad|inv|mix_lr | 0.8084 | 0.3178 | -0.331 | 1.229 |
| NWICU (inspire+eicu) | ncad|inv|mix_gbm | 0.8146 | 0.3145 | -0.388 | 1.103 |
| NWICU (inspire+eicu) | ncaf|ours|alone | 0.7838 | 0.3230 | +0.116 | 1.046 |
| NWICU (inspire+eicu) | ncaf|ours|mix_lr | 0.8077 | 0.3194 | -0.331 | 1.321 |
| NWICU (inspire+eicu) | ncaf|ours|mix_gbm | 0.8147 | 0.3139 | -0.359 | 1.148 |
| NWICU (inspire+eicu) | ncaf|inv|alone | 0.7852 | 0.3780 | +0.128 | 0.713 |
| NWICU (inspire+eicu) | ncaf|inv|mix_lr | 0.8077 | 0.3195 | -0.350 | 1.278 |
| NWICU (inspire+eicu) | ncaf|inv|mix_gbm | 0.8149 | 0.3139 | -0.371 | 1.128 |
| NWICU (inspire+eicu) | ncaf2k|ours|alone | 0.7798 | 0.3251 | +0.088 | 1.045 |
| NWICU (inspire+eicu) | ncaf2k|ours|mix_lr | 0.8072 | 0.3214 | -0.369 | 1.324 |
| NWICU (inspire+eicu) | ncaf2k|ours|mix_gbm | 0.8146 | 0.3151 | -0.391 | 1.139 |
| NWICU (inspire+eicu) | ncaf2k|inv|alone | 0.7804 | 0.3850 | +0.095 | 0.690 |
| NWICU (inspire+eicu) | ncaf2k|inv|mix_lr | 0.8070 | 0.3215 | -0.381 | 1.287 |
| NWICU (inspire+eicu) | ncaf2k|inv|mix_gbm | 0.8146 | 0.3149 | -0.392 | 1.125 |
| NWICU (inspire+eicu) | ncaf10k|ours|alone | 0.7765 | 0.3268 | +0.146 | 1.014 |
| NWICU (inspire+eicu) | ncaf10k|ours|mix_lr | 0.8053 | 0.3238 | -0.411 | 1.294 |
| NWICU (inspire+eicu) | ncaf10k|ours|mix_gbm | 0.8141 | 0.3172 | -0.442 | 1.104 |
| NWICU (inspire+eicu) | ncaf10k|inv|alone | 0.7779 | 0.4012 | +0.161 | 0.615 |
| NWICU (inspire+eicu) | ncaf10k|inv|mix_lr | 0.8056 | 0.3237 | -0.417 | 1.270 |
| NWICU (inspire+eicu) | ncaf10k|inv|mix_gbm | 0.8140 | 0.3175 | -0.448 | 1.097 |
| NWICU (inspire+eicu) | base|lr | 0.8053 | 0.3311 | -0.546 | 1.254 |
| NWICU (inspire+eicu) | base|gbm | 0.8142 | 0.3186 | -0.476 | 1.088 |
| NWICU (inspire+mimic) | cos|ours|alone | 0.7738 | 0.3293 | -0.076 | 1.137 |
| NWICU (inspire+mimic) | cos|ours|mix_lr | 0.8048 | 0.3424 | -0.672 | 1.370 |
| NWICU (inspire+mimic) | cos|ours|mix_gbm | 0.8131 | 0.3360 | -0.686 | 1.270 |
| NWICU (inspire+mimic) | cos|inv|alone | 0.7747 | 0.3634 | -0.081 | 0.842 |
| NWICU (inspire+mimic) | cos|inv|mix_lr | 0.8049 | 0.3414 | -0.668 | 1.337 |
| NWICU (inspire+mimic) | cos|inv|mix_gbm | 0.8133 | 0.3353 | -0.681 | 1.256 |
| NWICU (inspire+mimic) | mi|ours|alone | 0.7811 | 0.3245 | +0.065 | 1.043 |
| NWICU (inspire+mimic) | mi|ours|mix_lr | 0.8033 | 0.3430 | -0.690 | 1.281 |
| NWICU (inspire+mimic) | mi|ours|mix_gbm | 0.8125 | 0.3421 | -0.771 | 1.179 |
| NWICU (inspire+mimic) | mi|inv|alone | 0.7832 | 0.3906 | +0.073 | 0.692 |
| NWICU (inspire+mimic) | mi|inv|mix_lr | 0.8035 | 0.3425 | -0.692 | 1.252 |
| NWICU (inspire+mimic) | mi|inv|mix_gbm | 0.8126 | 0.3419 | -0.772 | 1.169 |
| NWICU (inspire+mimic) | ncad|ours|alone | 0.7927 | 0.3180 | +0.072 | 1.045 |
| NWICU (inspire+mimic) | ncad|ours|mix_lr | 0.8070 | 0.3284 | -0.520 | 1.288 |
| NWICU (inspire+mimic) | ncad|ours|mix_gbm | 0.8128 | 0.3340 | -0.677 | 1.193 |
| NWICU (inspire+mimic) | ncad|inv|alone | 0.7936 | 0.3866 | +0.077 | 0.683 |
| NWICU (inspire+mimic) | ncad|inv|mix_lr | 0.8062 | 0.3300 | -0.553 | 1.229 |
| NWICU (inspire+mimic) | ncad|inv|mix_gbm | 0.8124 | 0.3347 | -0.689 | 1.166 |
| NWICU (inspire+mimic) | ncaf|ours|alone | 0.7778 | 0.3261 | +0.127 | 1.016 |
| NWICU (inspire+mimic) | ncaf|ours|mix_lr | 0.8044 | 0.3363 | -0.609 | 1.303 |
| NWICU (inspire+mimic) | ncaf|ours|mix_gbm | 0.8129 | 0.3378 | -0.721 | 1.197 |
| NWICU (inspire+mimic) | ncaf|inv|alone | 0.7795 | 0.3869 | +0.141 | 0.668 |
| NWICU (inspire+mimic) | ncaf|inv|mix_lr | 0.8040 | 0.3366 | -0.621 | 1.260 |
| NWICU (inspire+mimic) | ncaf|inv|mix_gbm | 0.8129 | 0.3390 | -0.740 | 1.178 |
| NWICU (inspire+mimic) | ncaf2k|ours|alone | 0.7773 | 0.3268 | +0.124 | 1.021 |
| NWICU (inspire+mimic) | ncaf2k|ours|mix_lr | 0.8037 | 0.3366 | -0.609 | 1.306 |
| NWICU (inspire+mimic) | ncaf2k|ours|mix_gbm | 0.8121 | 0.3402 | -0.749 | 1.185 |
| NWICU (inspire+mimic) | ncaf2k|inv|alone | 0.7787 | 0.3993 | +0.135 | 0.635 |
| NWICU (inspire+mimic) | ncaf2k|inv|mix_lr | 0.8032 | 0.3377 | -0.630 | 1.263 |
| NWICU (inspire+mimic) | ncaf2k|inv|mix_gbm | 0.8120 | 0.3401 | -0.750 | 1.172 |
| NWICU (inspire+mimic) | ncaf10k|ours|alone | 0.7838 | 0.3233 | +0.115 | 1.017 |
| NWICU (inspire+mimic) | ncaf10k|ours|mix_lr | 0.8053 | 0.3338 | -0.583 | 1.297 |
| NWICU (inspire+mimic) | ncaf10k|ours|mix_gbm | 0.8131 | 0.3362 | -0.704 | 1.199 |
| NWICU (inspire+mimic) | ncaf10k|inv|alone | 0.7858 | 0.3908 | +0.127 | 0.658 |
| NWICU (inspire+mimic) | ncaf10k|inv|mix_lr | 0.8053 | 0.3354 | -0.615 | 1.256 |
| NWICU (inspire+mimic) | ncaf10k|inv|mix_gbm | 0.8132 | 0.3360 | -0.706 | 1.182 |
| NWICU (inspire+mimic) | base|lr | 0.8023 | 0.3714 | -0.974 | 1.198 |
| NWICU (inspire+mimic) | base|gbm | 0.8127 | 0.3531 | -0.894 | 1.132 |

## Sensitivity: a chosen by log-loss

| target | home Δ | U switch Δ (verdict) |
|---|---|---|
| inspire | +0.0177 (+0.0144 to +0.0206) | +0.0065 (+0.0053 to +0.0076) (NCA beats cosine) |
| mimic | +0.0067 (+0.0056 to +0.0080) | +0.0008 (+0.0002 to +0.0016) (NCA beats cosine) |
| eicu | +0.0159 (+0.0146 to +0.0171) | +0.0046 (+0.0041 to +0.0051) (NCA beats cosine) |
| NWICU (mimic+eicu) | +0.0171 (+0.0094 to +0.0246) | +0.0025 (+0.0000 to +0.0052) (NCA beats cosine) |
| NWICU (inspire+eicu) | +0.0179 (+0.0108 to +0.0260) | +0.0025 (-0.0004 to +0.0058) (no difference shown) |
| NWICU (inspire+mimic) | +0.0190 (+0.0114 to +0.0270) | +0.0031 (+0.0007 to +0.0060) (NCA beats cosine) |

## Complementarity (descriptive): standalone kNN against the base model

| target | arm, vote | base | corr(p_kNN, p_base) | Spearman(p_kNN, y − p_base) |
|---|---|---|---|---|
| inspire | COS, ours | lr | 0.791 | -0.422 |
| inspire | COS, ours | gbm | 0.777 | -0.421 |
| inspire | COS, inv | lr | 0.789 | -0.420 |
| inspire | COS, inv | gbm | 0.777 | -0.420 |
| inspire | MI, ours | lr | 0.827 | -0.366 |
| inspire | MI, ours | gbm | 0.859 | -0.381 |
| inspire | MI, inv | lr | 0.827 | -0.365 |
| inspire | MI, inv | gbm | 0.860 | -0.381 |
| inspire | NCA-diag, ours | lr | 0.835 | -0.350 |
| inspire | NCA-diag, ours | gbm | 0.875 | -0.392 |
| inspire | NCA-diag, inv | lr | 0.834 | -0.351 |
| inspire | NCA-diag, inv | gbm | 0.874 | -0.393 |
| inspire | NCA-full, ours | lr | 0.858 | -0.365 |
| inspire | NCA-full, ours | gbm | 0.879 | -0.387 |
| inspire | NCA-full, inv | lr | 0.857 | -0.365 |
| inspire | NCA-full, inv | gbm | 0.879 | -0.389 |
| inspire | NCA-full (2k), ours | lr | 0.851 | -0.364 |
| inspire | NCA-full (2k), ours | gbm | 0.868 | -0.390 |
| inspire | NCA-full (2k), inv | lr | 0.851 | -0.364 |
| inspire | NCA-full (2k), inv | gbm | 0.868 | -0.391 |
| inspire | NCA-full (10k), ours | lr | 0.855 | -0.363 |
| inspire | NCA-full (10k), ours | gbm | 0.880 | -0.391 |
| inspire | NCA-full (10k), inv | lr | 0.855 | -0.364 |
| inspire | NCA-full (10k), inv | gbm | 0.880 | -0.392 |
| mimic | COS, ours | lr | 0.862 | -0.101 |
| mimic | COS, ours | gbm | 0.851 | -0.102 |
| mimic | COS, inv | lr | 0.862 | -0.100 |
| mimic | COS, inv | gbm | 0.852 | -0.102 |
| mimic | MI, ours | lr | 0.900 | -0.100 |
| mimic | MI, ours | gbm | 0.920 | -0.106 |
| mimic | MI, inv | lr | 0.900 | -0.100 |
| mimic | MI, inv | gbm | 0.920 | -0.106 |
| mimic | NCA-diag, ours | lr | 0.903 | -0.105 |
| mimic | NCA-diag, ours | gbm | 0.925 | -0.116 |
| mimic | NCA-diag, inv | lr | 0.903 | -0.105 |
| mimic | NCA-diag, inv | gbm | 0.925 | -0.116 |
| mimic | NCA-full, ours | lr | 0.884 | -0.103 |
| mimic | NCA-full, ours | gbm | 0.897 | -0.108 |
| mimic | NCA-full, inv | lr | 0.884 | -0.103 |
| mimic | NCA-full, inv | gbm | 0.897 | -0.108 |
| mimic | NCA-full (2k), ours | lr | 0.864 | -0.097 |
| mimic | NCA-full (2k), ours | gbm | 0.883 | -0.104 |
| mimic | NCA-full (2k), inv | lr | 0.864 | -0.097 |
| mimic | NCA-full (2k), inv | gbm | 0.883 | -0.104 |
| mimic | NCA-full (10k), ours | lr | 0.892 | -0.106 |
| mimic | NCA-full (10k), ours | gbm | 0.900 | -0.109 |
| mimic | NCA-full (10k), inv | lr | 0.892 | -0.106 |
| mimic | NCA-full (10k), inv | gbm | 0.901 | -0.110 |
| eicu | COS, ours | lr | 0.882 | -0.299 |
| eicu | COS, ours | gbm | 0.871 | -0.300 |
| eicu | COS, inv | lr | 0.881 | -0.298 |
| eicu | COS, inv | gbm | 0.871 | -0.300 |
| eicu | MI, ours | lr | 0.892 | -0.278 |
| eicu | MI, ours | gbm | 0.915 | -0.280 |
| eicu | MI, inv | lr | 0.891 | -0.278 |
| eicu | MI, inv | gbm | 0.915 | -0.281 |
| eicu | NCA-diag, ours | lr | 0.899 | -0.277 |
| eicu | NCA-diag, ours | gbm | 0.931 | -0.291 |
| eicu | NCA-diag, inv | lr | 0.898 | -0.277 |
| eicu | NCA-diag, inv | gbm | 0.930 | -0.291 |
| eicu | NCA-full, ours | lr | 0.885 | -0.276 |
| eicu | NCA-full, ours | gbm | 0.913 | -0.285 |
| eicu | NCA-full, inv | lr | 0.885 | -0.276 |
| eicu | NCA-full, inv | gbm | 0.913 | -0.285 |
| eicu | NCA-full (2k), ours | lr | 0.889 | -0.279 |
| eicu | NCA-full (2k), ours | gbm | 0.916 | -0.289 |
| eicu | NCA-full (2k), inv | lr | 0.889 | -0.279 |
| eicu | NCA-full (2k), inv | gbm | 0.916 | -0.289 |
| eicu | NCA-full (10k), ours | lr | 0.891 | -0.275 |
| eicu | NCA-full (10k), ours | gbm | 0.918 | -0.288 |
| eicu | NCA-full (10k), inv | lr | 0.891 | -0.275 |
| eicu | NCA-full (10k), inv | gbm | 0.918 | -0.288 |
| NWICU (mimic+eicu) | COS, ours | lr | 0.815 | -0.434 |
| NWICU (mimic+eicu) | COS, ours | gbm | 0.798 | -0.433 |
| NWICU (mimic+eicu) | COS, inv | lr | 0.814 | -0.435 |
| NWICU (mimic+eicu) | COS, inv | gbm | 0.799 | -0.434 |
| NWICU (mimic+eicu) | MI, ours | lr | 0.865 | -0.438 |
| NWICU (mimic+eicu) | MI, ours | gbm | 0.879 | -0.430 |
| NWICU (mimic+eicu) | MI, inv | lr | 0.864 | -0.438 |
| NWICU (mimic+eicu) | MI, inv | gbm | 0.879 | -0.430 |
| NWICU (mimic+eicu) | NCA-diag, ours | lr | 0.858 | -0.433 |
| NWICU (mimic+eicu) | NCA-diag, ours | gbm | 0.890 | -0.449 |
| NWICU (mimic+eicu) | NCA-diag, inv | lr | 0.858 | -0.434 |
| NWICU (mimic+eicu) | NCA-diag, inv | gbm | 0.890 | -0.450 |
| NWICU (mimic+eicu) | NCA-full, ours | lr | 0.868 | -0.428 |
| NWICU (mimic+eicu) | NCA-full, ours | gbm | 0.891 | -0.438 |
| NWICU (mimic+eicu) | NCA-full, inv | lr | 0.866 | -0.430 |
| NWICU (mimic+eicu) | NCA-full, inv | gbm | 0.890 | -0.440 |
| NWICU (mimic+eicu) | NCA-full (2k), ours | lr | 0.861 | -0.427 |
| NWICU (mimic+eicu) | NCA-full (2k), ours | gbm | 0.874 | -0.433 |
| NWICU (mimic+eicu) | NCA-full (2k), inv | lr | 0.859 | -0.429 |
| NWICU (mimic+eicu) | NCA-full (2k), inv | gbm | 0.873 | -0.434 |
| NWICU (mimic+eicu) | NCA-full (10k), ours | lr | 0.854 | -0.431 |
| NWICU (mimic+eicu) | NCA-full (10k), ours | gbm | 0.882 | -0.440 |
| NWICU (mimic+eicu) | NCA-full (10k), inv | lr | 0.853 | -0.432 |
| NWICU (mimic+eicu) | NCA-full (10k), inv | gbm | 0.881 | -0.441 |
| NWICU (inspire+eicu) | COS, ours | lr | 0.807 | -0.434 |
| NWICU (inspire+eicu) | COS, ours | gbm | 0.770 | -0.427 |
| NWICU (inspire+eicu) | COS, inv | lr | 0.807 | -0.435 |
| NWICU (inspire+eicu) | COS, inv | gbm | 0.772 | -0.428 |
| NWICU (inspire+eicu) | MI, ours | lr | 0.878 | -0.452 |
| NWICU (inspire+eicu) | MI, ours | gbm | 0.904 | -0.443 |
| NWICU (inspire+eicu) | MI, inv | lr | 0.877 | -0.453 |
| NWICU (inspire+eicu) | MI, inv | gbm | 0.904 | -0.445 |
| NWICU (inspire+eicu) | NCA-diag, ours | lr | 0.864 | -0.432 |
| NWICU (inspire+eicu) | NCA-diag, ours | gbm | 0.905 | -0.448 |
| NWICU (inspire+eicu) | NCA-diag, inv | lr | 0.863 | -0.434 |
| NWICU (inspire+eicu) | NCA-diag, inv | gbm | 0.905 | -0.450 |
| NWICU (inspire+eicu) | NCA-full, ours | lr | 0.847 | -0.429 |
| NWICU (inspire+eicu) | NCA-full, ours | gbm | 0.868 | -0.431 |
| NWICU (inspire+eicu) | NCA-full, inv | lr | 0.846 | -0.430 |
| NWICU (inspire+eicu) | NCA-full, inv | gbm | 0.868 | -0.433 |
| NWICU (inspire+eicu) | NCA-full (2k), ours | lr | 0.842 | -0.425 |
| NWICU (inspire+eicu) | NCA-full (2k), ours | gbm | 0.862 | -0.431 |
| NWICU (inspire+eicu) | NCA-full (2k), inv | lr | 0.841 | -0.426 |
| NWICU (inspire+eicu) | NCA-full (2k), inv | gbm | 0.863 | -0.433 |
| NWICU (inspire+eicu) | NCA-full (10k), ours | lr | 0.851 | -0.442 |
| NWICU (inspire+eicu) | NCA-full (10k), ours | gbm | 0.863 | -0.442 |
| NWICU (inspire+eicu) | NCA-full (10k), inv | lr | 0.850 | -0.443 |
| NWICU (inspire+eicu) | NCA-full (10k), inv | gbm | 0.864 | -0.444 |
| NWICU (inspire+mimic) | COS, ours | lr | 0.820 | -0.436 |
| NWICU (inspire+mimic) | COS, ours | gbm | 0.805 | -0.434 |
| NWICU (inspire+mimic) | COS, inv | lr | 0.818 | -0.436 |
| NWICU (inspire+mimic) | COS, inv | gbm | 0.805 | -0.435 |
| NWICU (inspire+mimic) | MI, ours | lr | 0.858 | -0.442 |
| NWICU (inspire+mimic) | MI, ours | gbm | 0.873 | -0.435 |
| NWICU (inspire+mimic) | MI, inv | lr | 0.857 | -0.442 |
| NWICU (inspire+mimic) | MI, inv | gbm | 0.873 | -0.435 |
| NWICU (inspire+mimic) | NCA-diag, ours | lr | 0.860 | -0.439 |
| NWICU (inspire+mimic) | NCA-diag, ours | gbm | 0.894 | -0.454 |
| NWICU (inspire+mimic) | NCA-diag, inv | lr | 0.858 | -0.440 |
| NWICU (inspire+mimic) | NCA-diag, inv | gbm | 0.892 | -0.456 |
| NWICU (inspire+mimic) | NCA-full, ours | lr | 0.841 | -0.436 |
| NWICU (inspire+mimic) | NCA-full, ours | gbm | 0.865 | -0.444 |
| NWICU (inspire+mimic) | NCA-full, inv | lr | 0.841 | -0.438 |
| NWICU (inspire+mimic) | NCA-full, inv | gbm | 0.865 | -0.446 |
| NWICU (inspire+mimic) | NCA-full (2k), ours | lr | 0.840 | -0.433 |
| NWICU (inspire+mimic) | NCA-full (2k), ours | gbm | 0.864 | -0.441 |
| NWICU (inspire+mimic) | NCA-full (2k), inv | lr | 0.840 | -0.435 |
| NWICU (inspire+mimic) | NCA-full (2k), inv | gbm | 0.865 | -0.444 |
| NWICU (inspire+mimic) | NCA-full (10k), ours | lr | 0.846 | -0.430 |
| NWICU (inspire+mimic) | NCA-full (10k), ours | gbm | 0.870 | -0.441 |
| NWICU (inspire+mimic) | NCA-full (10k), inv | lr | 0.845 | -0.432 |
| NWICU (inspire+mimic) | NCA-full (10k), inv | gbm | 0.870 | -0.443 |
