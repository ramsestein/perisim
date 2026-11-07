# §19 with the intervention strata rebuilt without the landmark leak

Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. States at L from records before L only (v0.6's rule; eICU's rate-0 stop markers not applied). Predictions and comparators unchanged.

## Reclassification of the states

| site | state | on, v0.5 | on, v0.6 rule | on → off | off → on |
|---|---|---|---|---|---|
| inspire | vent_on | 15.4% | 12.7% | 2.6% | 0.0% |
| mimic | vaso_on | 21.1% | 21.1% | 0.0% | 0.0% |
| mimic | sedation_on | 26.2% | 26.2% | 0.0% | 0.0% |
| mimic | vent_on | 35.3% | 34.0% | 1.3% | 0.0% |
| eicu | vaso_on | 7.6% | 7.1% | 0.5% | 0.0% |
| eicu | sedation_on | 6.5% | 6.1% | 0.4% | 0.0% |
| eicu | vent_on | 17.0% | 13.3% | 3.7% | 0.0% |

## Verdicts, v0.5 strata against leak-free strata

| site · state | v0.5 | leak-free | unchanged |
|---|---|---|---|
| inspire|vent_on | failure no, repairs no, safe no | failure no, repairs no, safe yes | **no** |
| mimic|sedation_on | failure no, repairs no, safe yes | failure no, repairs no, safe yes | yes |
| mimic|vent_on | failure no, repairs no, safe yes | failure no, repairs no, safe yes | yes |
| mimic|vaso_on | failure yes, repairs yes, safe yes | failure yes, repairs yes, safe yes | yes |
| eicu|sedation_on | failure no, repairs no, safe yes | failure no, repairs no, safe yes | yes |
| eicu|vent_on | failure no, repairs no, safe yes | failure no, repairs no, safe yes | yes |
| eicu|vaso_on | failure yes, repairs no, safe yes | failure no, repairs no, safe yes | **no** |

All verdicts unchanged: **no**

## Full test with the leak-free strata

# The intervention-state mechanism: sedation (test), ventilation (control), vasopressor (reference)

Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. Cost = L(transported) − L(local) (positive = transport pays); repair = L(transported) − L(B0) (positive = B0 better); 2,000 patient resamples. Minimum repair 0.001 nats; B0 safe if its overall upper 95 % limit against the transported arm is below 0.001 nats.

## sedation_on (test)

- inspire: not recorded at this site (or no stratum of 10 landmarks or more).

### mimic: on in 26.2% of landmarks; weight share in the transported metric 0.54%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 387,159 | 30.7% | 0.5059 / 0.781 / +0.021 | 0.5031 / 0.784 / -0.002 | 0.5138 / 0.773 / -0.022 | 0.5110 / 0.789 / +0.407 | 0.4904 / 0.797 / +0.002 | 0.5051 / 0.781 / +0.012 | 0.5059 / 0.780 / +0.022 |
| off | 285,690 | 29.5% | 0.4901 / 0.791 / +0.022 | 0.4875 / 0.794 / +0.004 | 0.4978 / 0.784 / -0.013 | 0.4924 / 0.800 / +0.387 | 0.4746 / 0.806 / +0.005 | 0.4894 / 0.791 / +0.016 | 0.4901 / 0.791 / +0.030 |
| on | 101,469 | 33.9% | 0.5505 / 0.748 / +0.020 | 0.5470 / 0.752 / -0.016 | 0.5587 / 0.739 / -0.043 | 0.5634 / 0.754 / +0.456 | 0.5349 / 0.766 / -0.004 | 0.5492 / 0.750 / +0.005 | 0.5503 / 0.747 / +0.003 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: +0.0035 (+0.0021 to +0.0049); off: +0.0025 (+0.0017 to +0.0034); contrast: +0.0009 (-0.0007 to +0.0026)
- B0 repair on the stratum: +0.0002 (-0.0005 to +0.0009); B0 overall loss upper 95 %: +0.0002
- R transported: 0.738 (90 % CI 0.680–0.795; not shown equivalent); R after B0: 0.740 (90 % CI 0.680–0.799; not shown equivalent)
- LightGBM transported − local, on: +0.0285 (+0.0262 to +0.0310); off: +0.0178 (+0.0167 to +0.0190)
- verdict: failure no, repairs no, safe yes

### eicu: on in 6.1% of landmarks; weight share in the transported metric 2.36%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 466,385 | 19.3% | 0.3965 / 0.799 / +0.008 | 0.3960 / 0.800 / +0.008 | 0.4100 / 0.784 / -0.003 | 0.3987 / 0.810 / -0.402 | 0.3851 / 0.814 / +0.002 | 0.4005 / 0.794 / -0.006 | 0.3966 / 0.799 / +0.007 |
| off | 437,773 | 19.0% | 0.3923 / 0.800 / +0.006 | 0.3918 / 0.801 / +0.006 | 0.4047 / 0.787 / +0.005 | 0.3940 / 0.811 / -0.398 | 0.3809 / 0.815 / +0.002 | 0.3962 / 0.795 / -0.008 | 0.3922 / 0.800 / +0.010 |
| on | 28,612 | 24.1% | 0.4605 / 0.776 / +0.036 | 0.4611 / 0.774 / +0.037 | 0.4909 / 0.736 / -0.089 | 0.4704 / 0.783 / -0.455 | 0.4496 / 0.789 / +0.000 | 0.4667 / 0.769 / +0.009 | 0.4632 / 0.772 / -0.028 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: -0.0006 (-0.0031 to +0.0018); off: +0.0006 (-0.0000 to +0.0012); contrast: -0.0012 (-0.0038 to +0.0013)
- B0 repair on the stratum: -0.0027 (-0.0047 to -0.0008); B0 overall loss upper 95 %: +0.0002
- R transported: 0.963 (90 % CI 0.928–0.998; equivalent); R after B0: 0.961 (90 % CI 0.925–0.995; equivalent)
- LightGBM transported − local, on: +0.0208 (+0.0173 to +0.0243); off: +0.0131 (+0.0123 to +0.0139)
- verdict: failure no, repairs no, safe yes

## vent_on (control)

### inspire: on in 12.7% of landmarks; weight share in the transported metric 0.52%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 148,293 | 13.2% | 0.3276 / 0.779 / +0.023 | 0.3265 / 0.782 / +0.033 | 0.3381 / 0.765 / +0.062 | 0.3296 / 0.788 / -0.340 | 0.3161 / 0.802 / +0.008 | 0.3324 / 0.770 / +0.021 | 0.3281 / 0.778 / +0.025 |
| off | 129,430 | 12.4% | 0.3165 / 0.778 / +0.003 | 0.3152 / 0.782 / +0.018 | 0.3260 / 0.765 / +0.054 | 0.3189 / 0.789 / -0.372 | 0.3054 / 0.801 / +0.005 | 0.3201 / 0.771 / +0.000 | 0.3169 / 0.777 / +0.020 |
| on | 18,863 | 18.1% | 0.4035 / 0.767 / +0.127 | 0.4044 / 0.766 / +0.117 | 0.4210 / 0.741 / +0.104 | 0.4031 / 0.770 / -0.167 | 0.3896 / 0.788 / +0.023 | 0.4161 / 0.743 / +0.126 | 0.4048 / 0.763 / +0.051 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: -0.0009 (-0.0041 to +0.0023); off: +0.0013 (+0.0001 to +0.0025); contrast: -0.0022 (-0.0055 to +0.0012)
- B0 repair on the stratum: -0.0013 (-0.0040 to +0.0015); B0 overall loss upper 95 %: +0.0010
- R transported: 0.911 (90 % CI 0.832–0.994; equivalent); R after B0: 0.865 (90 % CI 0.789–0.945; not shown equivalent)
- LightGBM transported − local, on: +0.0135 (+0.0101 to +0.0169); off: +0.0135 (+0.0122 to +0.0149)
- verdict: failure no, repairs no, safe yes

### mimic: on in 34.0% of landmarks; weight share in the transported metric 0.50%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 387,159 | 30.7% | 0.5059 / 0.781 / +0.021 | 0.5031 / 0.784 / -0.002 | 0.5138 / 0.773 / -0.022 | 0.5110 / 0.789 / +0.407 | 0.4904 / 0.797 / +0.002 | 0.5051 / 0.781 / +0.012 | 0.5061 / 0.780 / +0.021 |
| off | 255,463 | 29.2% | 0.4853 / 0.794 / +0.024 | 0.4828 / 0.796 / +0.007 | 0.4931 / 0.787 / -0.006 | 0.4871 / 0.803 / +0.383 | 0.4699 / 0.809 / +0.007 | 0.4847 / 0.794 / +0.017 | 0.4856 / 0.793 / +0.030 |
| on | 131,696 | 33.5% | 0.5457 / 0.752 / +0.017 | 0.5425 / 0.755 / -0.016 | 0.5538 / 0.743 / -0.049 | 0.5575 / 0.758 / +0.447 | 0.5304 / 0.769 / -0.006 | 0.5445 / 0.753 / +0.005 | 0.5457 / 0.751 / +0.004 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: +0.0033 (+0.0020 to +0.0046); off: +0.0025 (+0.0017 to +0.0035); contrast: +0.0007 (-0.0009 to +0.0023)
- B0 repair on the stratum: +0.0000 (-0.0004 to +0.0004); B0 overall loss upper 95 %: +0.0004
- R transported: 0.738 (90 % CI 0.680–0.795; not shown equivalent); R after B0: 0.722 (90 % CI 0.664–0.778; not shown equivalent)
- LightGBM transported − local, on: +0.0272 (+0.0251 to +0.0292); off: +0.0172 (+0.0161 to +0.0184)
- verdict: failure no, repairs no, safe yes

### eicu: on in 13.3% of landmarks; weight share in the transported metric 0.03%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 466,385 | 19.3% | 0.3965 / 0.799 / +0.008 | 0.3960 / 0.800 / +0.008 | 0.4100 / 0.784 / -0.003 | 0.3987 / 0.810 / -0.402 | 0.3851 / 0.814 / +0.002 | 0.4005 / 0.794 / -0.006 | 0.3963 / 0.800 / +0.010 |
| off | 404,336 | 18.6% | 0.3859 / 0.802 / -0.001 | 0.3855 / 0.803 / +0.001 | 0.3985 / 0.789 / +0.003 | 0.3872 / 0.813 / -0.393 | 0.3747 / 0.817 / +0.000 | 0.3898 / 0.797 / -0.015 | 0.3856 / 0.803 / +0.012 |
| on | 62,049 | 24.3% | 0.4661 / 0.771 / +0.054 | 0.4642 / 0.773 / +0.047 | 0.4851 / 0.746 / -0.029 | 0.4730 / 0.781 / -0.450 | 0.4527 / 0.787 / +0.010 | 0.4706 / 0.765 / +0.037 | 0.4656 / 0.771 / +0.003 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: +0.0019 (+0.0003 to +0.0035); off: +0.0003 (-0.0003 to +0.0009); contrast: +0.0016 (-0.0001 to +0.0033)
- B0 repair on the stratum: +0.0005 (-0.0010 to +0.0020); B0 overall loss upper 95 %: +0.0000
- R transported: 0.963 (90 % CI 0.931–0.997; equivalent); R after B0: 0.980 (90 % CI 0.948–1.014; equivalent)
- LightGBM transported − local, on: +0.0203 (+0.0181 to +0.0226); off: +0.0125 (+0.0118 to +0.0133)
- verdict: failure no, repairs no, safe yes

## vaso_on (reference)

- inspire: not recorded at this site (or no stratum of 10 landmarks or more).

### mimic: on in 21.1% of landmarks; weight share in the transported metric 0.08%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 387,159 | 30.7% | 0.5059 / 0.781 / +0.021 | 0.5031 / 0.784 / -0.002 | 0.5138 / 0.773 / -0.022 | 0.5110 / 0.789 / +0.407 | 0.4904 / 0.797 / +0.002 | 0.5051 / 0.781 / +0.012 | 0.5053 / 0.781 / +0.013 |
| off | 305,428 | 26.7% | 0.4764 / 0.783 / -0.037 | 0.4747 / 0.785 / -0.008 | 0.4849 / 0.775 / -0.006 | 0.4753 / 0.793 / +0.331 | 0.4621 / 0.799 / +0.002 | 0.4770 / 0.782 / +0.006 | 0.4767 / 0.783 / +0.018 |
| on | 81,731 | 45.6% | 0.6162 / 0.719 / +0.189 | 0.6092 / 0.723 / +0.017 | 0.6216 / 0.709 / -0.066 | 0.6446 / 0.730 / +0.624 | 0.5965 / 0.739 / +0.003 | 0.6100 / 0.723 / +0.030 | 0.6122 / 0.719 / -0.002 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: +0.0070 (+0.0053 to +0.0088); off: +0.0017 (+0.0009 to +0.0024); contrast: +0.0054 (+0.0034 to +0.0074)
- B0 repair on the stratum: +0.0040 (+0.0024 to +0.0057); B0 overall loss upper 95 %: -0.0002
- R transported: 0.738 (90 % CI 0.683–0.792; not shown equivalent); R after B0: 0.797 (90 % CI 0.747–0.848; not shown equivalent)
- LightGBM transported − local, on: +0.0481 (+0.0448 to +0.0515); off: +0.0133 (+0.0123 to +0.0142)
- verdict: failure yes, repairs yes, safe yes

### eicu: on in 7.1% of landmarks; weight share in the transported metric 1.75%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 466,385 | 19.3% | 0.3965 / 0.799 / +0.008 | 0.3960 / 0.800 / +0.008 | 0.4100 / 0.784 / -0.003 | 0.3987 / 0.810 / -0.402 | 0.3851 / 0.814 / +0.002 | 0.4005 / 0.794 / -0.006 | 0.3964 / 0.799 / +0.009 |
| off | 433,174 | 18.5% | 0.3865 / 0.800 / +0.009 | 0.3862 / 0.800 / +0.003 | 0.3989 / 0.787 / +0.010 | 0.3859 / 0.812 / -0.364 | 0.3753 / 0.815 / +0.003 | 0.3908 / 0.794 / -0.008 | 0.3864 / 0.800 / +0.014 |
| on | 33,211 | 30.6% | 0.5269 / 0.754 / -0.001 | 0.5243 / 0.757 / +0.056 | 0.5557 / 0.721 / -0.109 | 0.5651 / 0.766 / -0.724 | 0.5127 / 0.770 / -0.009 | 0.5271 / 0.754 / +0.007 | 0.5270 / 0.754 / -0.038 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: +0.0026 (+0.0003 to +0.0049); off: +0.0004 (-0.0002 to +0.0010); contrast: +0.0022 (-0.0001 to +0.0048)
- B0 repair on the stratum: -0.0002 (-0.0011 to +0.0008); B0 overall loss upper 95 %: -0.0001
- R transported: 0.963 (90 % CI 0.928–0.998; equivalent); R after B0: 0.974 (90 % CI 0.940–1.009; equivalent)
- LightGBM transported − local, on: +0.0524 (+0.0472 to +0.0574); off: +0.0106 (+0.0099 to +0.0113)
- verdict: failure no, repairs no, safe yes

## Rule (pre-specified)

```
{
 "P1_weight": false,
 "P2_sedation": {
  "failures": [],
  "repaired_where_failed": true,
  "safe_everywhere": true,
  "reading": "no failure to repair (the mechanism's weight prediction alone is tested)"
 },
 "P3_vent_control": {
  "holds": true,
  "per_site": {
   "inspire": {
    "failure": false,
    "repairs": false,
    "safe": true
   },
   "mimic": {
    "failure": false,
    "repairs": false,
    "safe": true
   },
   "eicu": {
    "failure": false,
    "repairs": false,
    "safe": true
   }
  }
 }
}
```

Weight shares by target (the metric learned on the other two sites):

- sedation_on: inspire 0.41%, mimic 0.54%, eicu 2.36%
- vent_on: inspire 0.52%, mimic 0.50%, eicu 0.03%
- vaso_on: inspire 3.64%, mimic 0.08%, eicu 1.75%

