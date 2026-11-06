# Two repairs of the vasopressor failure (`hypo`, exploratory)

Plan fixed before running: docs/DECISIONS.md, 2026-11-06. Same encoder and shapes, case base, k and vote in every arm; only the weights or the neighbourhood change. Δ = L(NCA-refit) − L(arm): positive = the arm is better. 2,000 patient resamples.

## inspire

| arm | stratum | log-loss | Δ vs NCA-refit (95 % CI) | AUROC | CITL | slope |
|---|---|---|---|---|---|---|
| NCA transported (stored) | all | 0.3276 | +0.0000 (+0.0000 to +0.0000) | 0.779 | +0.023 | 1.02 |
| NCA local | all | 0.3265 | — | 0.782 | +0.033 | 1.06 |
| Euclidean | all | 0.3381 | — | 0.765 | +0.062 | 1.19 |
| MI-transported (§17) | all | 0.3324 | — | 0.770 | +0.021 | 1.06 |
| NCA refit (reference) | all | 0.3276 | — | 0.779 | +0.023 | 1.02 |
| A-sel: MI prior, λ chosen | all | 0.3281 | -0.0005 (-0.0008 to -0.0001) | 0.778 | +0.020 | 1.04 |
| A-strong: MI prior, λ = 1 | all | 0.3322 | -0.0046 (-0.0057 to -0.0035) | 0.769 | +0.018 | 1.10 |
| B0: frozen metric, same-state neighbours | all | 0.3276 | +0.0000 (+0.0000 to +0.0000) | 0.779 | +0.023 | 1.02 |
| B: per-state metric | all | 0.3285 | -0.0010 (-0.0016 to -0.0003) | 0.777 | +0.024 | 1.01 |

λ chosen: 0.0 (validation: {'0.0': 0.48108397325856966, '0.01': 0.48245062752520557, '0.1': 0.48320269320074377, '1.0': 0.48347396782539487, '10.0': 0.4839552718282247}); log-weight correlation NCA-refit vs frozen: 1.000; vasopressor inputs' share of the weights: frozen 3.64%, nca_refit 3.64%, a_sel 3.83%, a_strong 9.73%, b_state_0 3.08%, b_state_1 5.87%, mi_prior 9.69%

## mimic

| arm | stratum | log-loss | Δ vs NCA-refit (95 % CI) | AUROC | CITL | slope |
|---|---|---|---|---|---|---|
| NCA transported (stored) | all | 0.5059 | +0.0000 (+0.0000 to +0.0000) | 0.781 | +0.021 | 1.01 |
| NCA local | all | 0.5031 | — | 0.784 | -0.002 | 1.01 |
| Euclidean | all | 0.5138 | — | 0.773 | -0.022 | 1.11 |
| MI-transported (§17) | all | 0.5051 | — | 0.781 | +0.012 | 1.00 |
| NCA refit (reference) | all | 0.5059 | — | 0.781 | +0.021 | 1.01 |
| A-sel: MI prior, λ chosen | all | 0.5059 | +0.0000 (-0.0001 to +0.0002) | 0.781 | +0.020 | 1.01 |
| A-strong: MI prior, λ = 1 | all | 0.5068 | -0.0009 (-0.0017 to -0.0002) | 0.779 | +0.016 | 1.01 |
| B0: frozen metric, same-state neighbours | all | 0.5053 | +0.0006 (+0.0002 to +0.0010) | 0.781 | +0.013 | 1.02 |
| B: per-state metric | all | 0.5059 | +0.0000 (-0.0006 to +0.0006) | 0.780 | -0.003 | 1.02 |
| NCA transported (stored) | vaso_on = 0 | 0.4764 | +0.0000 (+0.0000 to +0.0000) | 0.783 | -0.037 | 1.01 |
| NCA local | vaso_on = 0 | 0.4747 | — | 0.785 | -0.008 | 1.01 |
| Euclidean | vaso_on = 0 | 0.4849 | — | 0.775 | -0.006 | 1.14 |
| MI-transported (§17) | vaso_on = 0 | 0.4770 | — | 0.782 | +0.006 | 1.01 |
| NCA refit (reference) | vaso_on = 0 | 0.4764 | — | 0.783 | -0.037 | 1.01 |
| A-sel: MI prior, λ chosen | vaso_on = 0 | 0.4764 | -0.0001 (-0.0002 to +0.0001) | 0.783 | -0.038 | 1.00 |
| A-strong: MI prior, λ = 1 | vaso_on = 0 | 0.4788 | -0.0024 (-0.0032 to -0.0016) | 0.780 | +0.010 | 1.01 |
| B0: frozen metric, same-state neighbours | vaso_on = 0 | 0.4767 | -0.0003 (-0.0006 to +0.0000) | 0.783 | +0.018 | 1.03 |
| B: per-state metric | vaso_on = 0 | 0.4767 | -0.0003 (-0.0007 to +0.0001) | 0.782 | +0.014 | 1.03 |
| NCA transported (stored) | vaso_on = 1 | 0.6162 | +0.0000 (+0.0000 to +0.0000) | 0.719 | +0.189 | 0.92 |
| NCA local | vaso_on = 1 | 0.6092 | — | 0.723 | +0.017 | 0.99 |
| Euclidean | vaso_on = 1 | 0.6216 | — | 0.709 | -0.066 | 1.16 |
| MI-transported (§17) | vaso_on = 1 | 0.6100 | — | 0.723 | +0.030 | 0.98 |
| NCA refit (reference) | vaso_on = 1 | 0.6162 | — | 0.719 | +0.189 | 0.92 |
| A-sel: MI prior, λ chosen | vaso_on = 1 | 0.6159 | +0.0003 (+0.0000 to +0.0006) | 0.719 | +0.187 | 0.93 |
| A-strong: MI prior, λ = 1 | vaso_on = 1 | 0.6116 | +0.0046 (+0.0029 to +0.0064) | 0.720 | +0.032 | 0.98 |
| B0: frozen metric, same-state neighbours | vaso_on = 1 | 0.6122 | +0.0040 (+0.0023 to +0.0057) | 0.719 | -0.002 | 1.00 |
| B: per-state metric | vaso_on = 1 | 0.6149 | +0.0013 (-0.0009 to +0.0037) | 0.717 | -0.048 | 1.01 |

λ chosen: 0.0 (validation: {'0.0': 0.3816314536959585, '0.01': 0.3847777073108718, '0.1': 0.3851812222217772, '1.0': 0.38470365372242055, '10.0': 0.3845210053486819}); log-weight correlation NCA-refit vs frozen: 1.000; vasopressor inputs' share of the weights: frozen 0.08%, nca_refit 0.08%, a_sel 0.08%, a_strong 3.64%, b_state_0 1.39%, b_state_1 4.09%, mi_prior 3.61%

## eicu

| arm | stratum | log-loss | Δ vs NCA-refit (95 % CI) | AUROC | CITL | slope |
|---|---|---|---|---|---|---|
| NCA transported (stored) | all | 0.3965 | +0.0000 (+0.0000 to +0.0000) | 0.799 | +0.008 | 1.03 |
| NCA local | all | 0.3960 | — | 0.800 | +0.008 | 1.05 |
| Euclidean | all | 0.4100 | — | 0.784 | -0.003 | 1.17 |
| MI-transported (§17) | all | 0.4005 | — | 0.794 | -0.006 | 1.06 |
| NCA refit (reference) | all | 0.3965 | — | 0.799 | +0.008 | 1.03 |
| A-sel: MI prior, λ chosen | all | 0.3965 | +0.0001 (-0.0001 to +0.0003) | 0.799 | +0.008 | 1.03 |
| A-strong: MI prior, λ = 1 | all | 0.4002 | -0.0036 (-0.0043 to -0.0030) | 0.794 | -0.004 | 1.05 |
| B0: frozen metric, same-state neighbours | all | 0.3964 | +0.0001 (+0.0000 to +0.0002) | 0.799 | +0.007 | 1.03 |
| B: per-state metric | all | 0.3971 | -0.0006 (-0.0009 to -0.0003) | 0.798 | +0.008 | 1.03 |
| NCA transported (stored) | vaso_on = 0 | 0.3858 | +0.0000 (+0.0000 to +0.0000) | 0.800 | +0.008 | 1.02 |
| NCA local | vaso_on = 0 | 0.3855 | — | 0.800 | +0.001 | 1.04 |
| Euclidean | vaso_on = 0 | 0.3981 | — | 0.787 | +0.010 | 1.17 |
| MI-transported (§17) | vaso_on = 0 | 0.3901 | — | 0.794 | -0.009 | 1.06 |
| NCA refit (reference) | vaso_on = 0 | 0.3858 | — | 0.800 | +0.008 | 1.02 |
| A-sel: MI prior, λ chosen | vaso_on = 0 | 0.3857 | +0.0001 (-0.0001 to +0.0003) | 0.800 | +0.008 | 1.02 |
| A-strong: MI prior, λ = 1 | vaso_on = 0 | 0.3896 | -0.0039 (-0.0045 to -0.0032) | 0.794 | -0.004 | 1.05 |
| B0: frozen metric, same-state neighbours | vaso_on = 0 | 0.3856 | +0.0001 (+0.0001 to +0.0002) | 0.800 | +0.013 | 1.03 |
| B: per-state metric | vaso_on = 0 | 0.3862 | -0.0004 (-0.0007 to -0.0001) | 0.799 | +0.013 | 1.03 |
| NCA transported (stored) | vaso_on = 1 | 0.5276 | +0.0000 (+0.0000 to +0.0000) | 0.754 | +0.008 | 1.05 |
| NCA local | vaso_on = 1 | 0.5245 | — | 0.758 | +0.064 | 1.06 |
| Euclidean | vaso_on = 1 | 0.5559 | — | 0.721 | -0.102 | 1.48 |
| MI-transported (§17) | vaso_on = 1 | 0.5275 | — | 0.754 | +0.015 | 1.10 |
| NCA refit (reference) | vaso_on = 1 | 0.5276 | — | 0.754 | +0.008 | 1.05 |
| A-sel: MI prior, λ chosen | vaso_on = 1 | 0.5278 | -0.0002 (-0.0008 to +0.0005) | 0.754 | +0.006 | 1.04 |
| A-strong: MI prior, λ = 1 | vaso_on = 1 | 0.5288 | -0.0012 (-0.0031 to +0.0010) | 0.753 | -0.001 | 1.08 |
| B0: frozen metric, same-state neighbours | vaso_on = 1 | 0.5279 | -0.0002 (-0.0011 to +0.0007) | 0.754 | -0.043 | 1.03 |
| B: per-state metric | vaso_on = 1 | 0.5304 | -0.0028 (-0.0049 to -0.0005) | 0.751 | -0.040 | 1.05 |

λ chosen: 0.0 (validation: {'0.0': 0.4391622178147774, '0.01': 0.4416340774693798, '0.1': 0.4421117512840503, '1.0': 0.44244994185390907, '10.0': 0.4426757638604336}); log-weight correlation NCA-refit vs frozen: 1.000; vasopressor inputs' share of the weights: frozen 1.75%, nca_refit 1.75%, a_sel 1.64%, a_strong 10.57%, b_state_0 2.88%, b_state_1 6.72%, mi_prior 10.64%

## Rule (pre-specified)

| arm | repairs MIMIC-IV's vasopressor stratum | safe at every site | works |
|---|---|---|---|
| A-sel: MI prior, λ chosen | yes | yes | **yes** |
| A-strong: MI prior, λ = 1 | yes | no | no |
| B0: frozen metric, same-state neighbours | yes | yes | **yes** |
| B: per-state metric | no | no | no |
