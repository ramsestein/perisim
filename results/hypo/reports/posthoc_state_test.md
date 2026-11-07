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

### eicu: on in 6.5% of landmarks; weight share in the transported metric 2.36%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 466,385 | 19.3% | 0.3965 / 0.799 / +0.008 | 0.3960 / 0.800 / +0.008 | 0.4100 / 0.784 / -0.003 | 0.3987 / 0.810 / -0.402 | 0.3851 / 0.814 / +0.002 | 0.4005 / 0.794 / -0.006 | 0.3965 / 0.799 / +0.007 |
| off | 435,949 | 19.0% | 0.3920 / 0.800 / +0.005 | 0.3914 / 0.801 / +0.005 | 0.4043 / 0.787 / +0.005 | 0.3936 / 0.812 / -0.398 | 0.3805 / 0.815 / +0.001 | 0.3959 / 0.795 / -0.008 | 0.3917 / 0.801 / +0.010 |
| on | 30,436 | 24.1% | 0.4617 / 0.775 / +0.043 | 0.4622 / 0.773 / +0.044 | 0.4914 / 0.735 / -0.083 | 0.4706 / 0.782 / -0.446 | 0.4504 / 0.788 / +0.009 | 0.4675 / 0.768 / +0.014 | 0.4647 / 0.771 / -0.030 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: -0.0005 (-0.0028 to +0.0019); off: +0.0006 (-0.0000 to +0.0012); contrast: -0.0011 (-0.0035 to +0.0014)
- B0 repair on the stratum: -0.0029 (-0.0050 to -0.0010); B0 overall loss upper 95 %: +0.0001
- R transported: 0.963 (90 % CI 0.928–0.998; equivalent); R after B0: 0.967 (90 % CI 0.932–1.002; equivalent)
- LightGBM transported − local, on: +0.0202 (+0.0166 to +0.0236); off: +0.0131 (+0.0123 to +0.0139)
- verdict: failure no, repairs no, safe yes

## vent_on (control)

### inspire: on in 15.4% of landmarks; weight share in the transported metric 0.52%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 148,293 | 13.2% | 0.3276 / 0.779 / +0.023 | 0.3265 / 0.782 / +0.033 | 0.3381 / 0.765 / +0.062 | 0.3296 / 0.788 / -0.340 | 0.3161 / 0.802 / +0.008 | 0.3324 / 0.770 / +0.021 | 0.3281 / 0.778 / +0.022 |
| off | 125,515 | 12.2% | 0.3137 / 0.777 / -0.007 | 0.3124 / 0.780 / +0.009 | 0.3229 / 0.764 / +0.052 | 0.3163 / 0.787 / -0.376 | 0.3029 / 0.800 / +0.002 | 0.3172 / 0.769 / -0.002 | 0.3141 / 0.776 / +0.014 |
| on | 22,778 | 18.3% | 0.4042 / 0.771 / +0.145 | 0.4044 / 0.771 / +0.135 | 0.4222 / 0.745 / +0.104 | 0.4030 / 0.775 / -0.190 | 0.3887 / 0.793 / +0.033 | 0.4159 / 0.750 / +0.114 | 0.4051 / 0.768 / +0.055 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: -0.0002 (-0.0033 to +0.0029); off: +0.0013 (+0.0001 to +0.0025); contrast: -0.0015 (-0.0048 to +0.0019)
- B0 repair on the stratum: -0.0010 (-0.0034 to +0.0016); B0 overall loss upper 95 %: +0.0010
- R transported: 0.911 (90 % CI 0.832–0.994; equivalent); R after B0: 0.864 (90 % CI 0.788–0.943; not shown equivalent)
- LightGBM transported − local, on: +0.0144 (+0.0110 to +0.0177); off: +0.0133 (+0.0120 to +0.0148)
- verdict: failure no, repairs no, safe no

### mimic: on in 35.3% of landmarks; weight share in the transported metric 0.50%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 387,159 | 30.7% | 0.5059 / 0.781 / +0.021 | 0.5031 / 0.784 / -0.002 | 0.5138 / 0.773 / -0.022 | 0.5110 / 0.789 / +0.407 | 0.4904 / 0.797 / +0.002 | 0.5051 / 0.781 / +0.012 | 0.5060 / 0.780 / +0.020 |
| off | 250,503 | 29.1% | 0.4841 / 0.795 / +0.026 | 0.4813 / 0.798 / +0.007 | 0.4917 / 0.788 / -0.005 | 0.4857 / 0.804 / +0.385 | 0.4684 / 0.810 / +0.008 | 0.4833 / 0.795 / +0.017 | 0.4842 / 0.795 / +0.032 |
| on | 136,656 | 33.5% | 0.5459 / 0.751 / +0.015 | 0.5430 / 0.754 / -0.016 | 0.5542 / 0.742 / -0.049 | 0.5576 / 0.757 / +0.443 | 0.5308 / 0.768 / -0.006 | 0.5449 / 0.752 / +0.005 | 0.5460 / 0.751 / +0.002 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: +0.0029 (+0.0017 to +0.0042); off: +0.0027 (+0.0018 to +0.0037); contrast: +0.0002 (-0.0014 to +0.0018)
- B0 repair on the stratum: -0.0001 (-0.0005 to +0.0003); B0 overall loss upper 95 %: +0.0003
- R transported: 0.738 (90 % CI 0.680–0.795; not shown equivalent); R after B0: 0.729 (90 % CI 0.670–0.786; not shown equivalent)
- LightGBM transported − local, on: +0.0267 (+0.0247 to +0.0288); off: +0.0173 (+0.0161 to +0.0185)
- verdict: failure no, repairs no, safe yes

### eicu: on in 17.0% of landmarks; weight share in the transported metric 0.03%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 466,385 | 19.3% | 0.3965 / 0.799 / +0.008 | 0.3960 / 0.800 / +0.008 | 0.4100 / 0.784 / -0.003 | 0.3987 / 0.810 / -0.402 | 0.3851 / 0.814 / +0.002 | 0.4005 / 0.794 / -0.006 | 0.3963 / 0.799 / +0.006 |
| off | 387,226 | 18.4% | 0.3837 / 0.802 / -0.000 | 0.3834 / 0.803 / +0.001 | 0.3955 / 0.790 / +0.009 | 0.3848 / 0.814 / -0.389 | 0.3726 / 0.817 / +0.002 | 0.3875 / 0.797 / -0.012 | 0.3836 / 0.802 / +0.011 |
| on | 79,159 | 23.9% | 0.4593 / 0.777 / +0.041 | 0.4577 / 0.778 / +0.036 | 0.4810 / 0.749 / -0.047 | 0.4665 / 0.787 / -0.454 | 0.4461 / 0.792 / -0.000 | 0.4642 / 0.770 / +0.014 | 0.4586 / 0.777 / -0.013 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: +0.0017 (+0.0003 to +0.0032); off: +0.0003 (-0.0003 to +0.0009); contrast: +0.0014 (-0.0001 to +0.0030)
- B0 repair on the stratum: +0.0008 (-0.0005 to +0.0021); B0 overall loss upper 95 %: +0.0001
- R transported: 0.963 (90 % CI 0.931–0.997; equivalent); R after B0: 0.979 (90 % CI 0.946–1.013; equivalent)
- LightGBM transported − local, on: +0.0204 (+0.0183 to +0.0225); off: +0.0122 (+0.0114 to +0.0130)
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

### eicu: on in 7.6% of landmarks; weight share in the transported metric 1.75%

| stratum | landmarks | prevalence | NCA transported | NCA local | Euclidean | LightGBM transported | LightGBM local | MI transported | B0 (same-state neighbours) |
|---|---|---|---|---|---|---|---|---|---|
| all | 466,385 | 19.3% | 0.3965 / 0.799 / +0.008 | 0.3960 / 0.800 / +0.008 | 0.4100 / 0.784 / -0.003 | 0.3987 / 0.810 / -0.402 | 0.3851 / 0.814 / +0.002 | 0.4005 / 0.794 / -0.006 | 0.3964 / 0.799 / +0.007 |
| off | 431,041 | 18.4% | 0.3858 / 0.800 / +0.008 | 0.3855 / 0.800 / +0.001 | 0.3981 / 0.787 / +0.010 | 0.3851 / 0.812 / -0.362 | 0.3746 / 0.815 / +0.002 | 0.3901 / 0.794 / -0.009 | 0.3856 / 0.800 / +0.013 |
| on | 35,344 | 30.7% | 0.5276 / 0.754 / +0.008 | 0.5245 / 0.758 / +0.064 | 0.5559 / 0.721 / -0.102 | 0.5644 / 0.766 / -0.716 | 0.5130 / 0.770 / +0.000 | 0.5275 / 0.754 / +0.015 | 0.5279 / 0.754 / -0.043 |

(cells: log-loss / AUROC / CITL)

- transport cost, on: +0.0031 (+0.0008 to +0.0055); off: +0.0003 (-0.0003 to +0.0009); contrast: +0.0028 (+0.0005 to +0.0053)
- B0 repair on the stratum: -0.0002 (-0.0011 to +0.0006); B0 overall loss upper 95 %: -0.0000
- R transported: 0.963 (90 % CI 0.928–0.998; equivalent); R after B0: 0.971 (90 % CI 0.937–1.005; equivalent)
- LightGBM transported − local, on: +0.0515 (+0.0465 to +0.0565); off: +0.0105 (+0.0098 to +0.0112)
- verdict: failure yes, repairs no, safe yes

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
    "safe": false
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
