# PERISIM `roll`: a rotating case base against ML managed as a hospital manages it — report

Draft for the lead, 2026-10-31. Pre-registration [docs/prereg_roll.md](../../docs/prereg_roll.md), frozen on 2026-10-30
(tag `perisim-freeze-roll-2026-10-30-173256`, registration `LOCAL-PREREG-ROLL-2026-10-30`; record
[docs/freeze_roll.md](../../docs/freeze_roll.md)). Every number is from [reports/endpoints.md](reports/endpoints.md),
generated from the single run's frozen output. Aggregates only.

**This is a pre-registered retrospective evaluation on data already examined** (prereg, opening). The three cohorts
were used with this outcome in `hypo`. What had not been seen is the performance along a prequential stream.

## Summary

- **Of the confirmatory hypotheses, only H4 (calibration) holds.** H1 (level), H2 (stability), H3 (low tail) and H6
  (the out-of-support signal) do not.
- **Accuracy (H1): not inferior in MIMIC-IV only.**
  - In MIMIC-IV, PERISIM-FIFO is within the margins against arm 5 (LightGBM, managed as a hospital manages it):
    - AUROC −0.016 (−0.017 to −0.016), against a margin of −0.02;
    - AUPRC − prevalence −0.029 (−0.029 to −0.028), against a margin of −0.03. The lower bound clears it by 0.0005.
  - In eICU it fails, by the amounts the pre-registration predicted (−0.040 and −0.023):
    - eICU (a), 100 patients: −0.039 (−0.042 to −0.036);
    - eICU (b), 250 patients: −0.021 (−0.022 to −0.019).
- **Calibration (H4) is better in all three confirmatory streams.**
  - The excess calibration error:
    - MIMIC-IV −0.022 (−0.034 to −0.012);
    - eICU (a) −0.223;
    - eICU (b) −0.207.
  - The share of checks at which the primary trigger fires:
    - MIMIC-IV −0.045;
    - eICU (a) and (b) −0.54 and −0.53, after the declared deviation (§10). The frozen pooling gives degenerate values,
      which meet the rule too.
- **Where the calibration advantage comes from differs between the cohorts.**
  - In MIMIC-IV it is the cost of the update cycle:
    - PERISIM ties the ideal ML, updated every 100 patients with no delay: +0.001 (−0.008 to 0.009);
    - it wins narrowly with no delay, L = 0: −0.008 (−0.021 to −0.000);
    - it wins clearly against the four sensitivity triggers (AUROC, log-loss, Davis et al.'s detector): −0.06 to
      −0.10.
  - In eICU it is mostly the transported model's miscalibration (calibration in the large −0.44). The pre-registered
    go-live and updates leave it unfixed with 100–250 local patients: the go-live update kept the transported model
    in 267 of 350 hospital-orders in (a).
  - Against an ML trained on each hospital's starting window (arm 5′, reported only), the gap in (a) shrinks to −0.023
    (−0.038 to −0.008). This favours PERISIM, and it is flagged (§4.3).
- **Stability (H2) does not hold.**
  - The rotating case base makes PERISIM-FIFO's window AUROC vary more than arm 5's, beyond sampling noise: excess-SD
    ratio 1.23 (1.02 to 1.50) in MIMIC-IV and 1.32 (1.14 to 1.54) in eICU (a).
  - In the cross-cohort streams it is the other way round: ratios of 0.24–0.68 in eICU → MIMIC-IV and MIMIC-IV →
    INSPIRE.
- **After a change of cohort, PERISIM's calibration recovers in 480–630 patients**, against 1,120–1,680 for arm 5.
  Its AUROC stays 0.023–0.029 below.
- **The diagnostics flag little, and there was little to flag.**
  - The out-of-support rate does not rise at eICU's change of year (H6 (a) fails: −0.0015). Its false alarms are rare
    (0.23 %).
  - Re-deformation never gains more than 0.006 nats.
  - The data drift mostly in prevalence (MIMIC-IV 32.0 → 25.6 %), which the case base tracks: the static case base
    over-predicts by the end.
- **Governable, as measured.**
  - Predictions are identical with 1 and 6 threads, and the FIFO window is rebuilt from the log alone with no mismatch.
  - PERISIM updates every 10 patients and changes no parameter. Arm 5 needed about 6 trigger–decision–deployment
    cycles per MIMIC-IV order.
- **H7: the smallest N** that meets H1's and H4's rules together is 500, in MIMIC-IV temporal. No N up to 2,000 meets
  them in the other streams.
- **The thesis**, "not less accurate than ML managed as a real hospital manages it, and more governable", is only
  partly supported:
  - accuracy: not inferior in one cohort of three;
  - better calibrated without an update cycle in all three;
  - less stable within a cohort.

## 1. Question and design

- **The thesis** (prereg §1): CE-kNN is not less accurate than ML managed as a hospital manages it, and it is more
  governable. More governable means:
  - deterministic;
  - updated without retraining, by FIFO rotation of the case base;
  - stable along the stream;
  - with diagnostics that flag real changes.
- **Prequential, test-then-train** (Gama et al., 2014).
  - Patients arrive in order, in steps of B = 10.
  - Every arm predicts a step's patients with its state before them, and only then do they enter.
  - Each patient is evaluated once.
  - The case base is a rotating window of the last N patients, the short-term memory of SAM-kNN (Losing et al., 2016)
    and the case base of IBLStreams (Shaker & Hüllermeier, 2012). The metric stays `hypo`'s frozen transported metric.
- **Arms** (prereg §6):
  - PERISIM: FIFO (arm 1), cumulative (2) and static (3).
  - ML frozen (4a), and frozen after go-live (4b).
  - **ML realistic (arm 5)**, the primary comparator:
    - the transported LightGBM with a go-live update;
    - a calibration monitor and a primary trigger (O/E outside [0.9, 1.1], or a CITL CI entirely outside ±0.10);
    - the update chosen by closed testing (Vergouwe et al., 2017; adapted, design §7.5);
    - deployment after L = 3 months.
  - Its variants: L = 0, four other triggers (Davis et al., 2020 among them), and 5′, trained on the starting window.
  - ML ideal (arm 6), the hierarchy every W patients with no trigger or delay. A ceiling, not a realistic arm.
  - Prevalence (7).
  - LightGBM is primary, logistic regression secondary.
- **Endpoints** are per evaluation window of W patients. CIs are 95 % from a moving-block bootstrap of the windows
  (Künsch, 1989; 2,000 resamples, stratified by segment). eICU's hospitals are pooled by DerSimonian–Laird (1986). A
  hypothesis holds only if every endpoint meets its rule in every confirmatory stream (intersection–union; Berger,
  1982).

## 2. Data and the run

| stream | role | order | N (primary) | units |
|---|---|---|---|---|
| MIMIC-IV temporal | confirmatory | 5 blocks, 2008–2010 to 2020–2022, in order | 250, **500**, 1,000, 2,000 | 20 |
| eICU (a), within each of 35 hospitals | confirmatory | 2014 → 2015 | **100**, 250 (4 hospitals) | 390 |
| eICU (b), deployment to each of 35 hospitals | confirmatory | other hospitals → own, 2014 → 2015 | 100, **250** | 700 |
| INSPIRE → MIMIC-IV, eICU → MIMIC-IV, MIMIC-IV → INSPIRE | secondary | 2,000 then 4,000 patients | as MIMIC-IV | 20 each |
| INSPIRE control | control | random | as MIMIC-IV | 20 |

- **The single run** (Phase 4):
  - It started 2026-10-30 17:55 in four parts side by side and ended at 23:09: 1,190 units, none failed.
  - The units took 16.2 h of compute in all.
  - The freeze was checked at the start and end of each part, of `endpoints` (23:14–03:09) and of `t2` (03:12–03:20).
  - 32 of 21,930 windows were dropped (fewer than 30 event landmarks or 10 patients with an event).
- **One interim look**, by the lead's decision: MIMIC-IV temporal at N = 500, read during the run with the frozen
  endpoint code (DECISIONS 2026-10-30). Its numbers are identical to the final ones.

## 3. Confirmatory results

PERISIM-FIFO minus arm 5, paired on the same windows (negative = FIFO lower):

| | MIMIC-IV, N 500 | eICU (a), N 100 | eICU (b), N 250 | rule |
|---|---|---|---|---|
| **H1** AUROC | −0.016 (−0.017 to −0.016) ✔ | −0.039 (−0.042 to −0.036) ✘ | −0.021 (−0.022 to −0.019) ✘ | lower bound > −0.02 |
| **H1** AUPRC − prevalence | −0.029 (−0.029 to −0.028) ✔ | −0.070 (−0.075 to −0.065) ✘ | −0.034 (−0.037 to −0.031) ✘ | lower bound > −0.03 |
| **H2** excess SD ratio, AUROC | 1.23 (1.02 to 1.50) ✘ | 1.32 (1.14 to 1.54) ✘ | 1.14 (0.97 to 1.34) ✘ | upper bound < 1 |
| **H2** excess SD ratio, AUPRC − prev. | 1.33 (1.11 to 1.92) ✘ | 1.27 (0.92 to 1.74) ✘ | 1.13 (0.96 to 1.34) ✘ | upper bound < 1 |
| **H3** P10 of the window AUROC | −0.016 (−0.018 to −0.015) ✘ | −0.043 (−0.046 to −0.039) ✘ | −0.022 (−0.024 to −0.020) ✘ | lower bound > 0 |
| **H3** P10 of the window AUPRC | −0.030 (−0.034 to −0.026) ✘ | −0.067 (−0.071 to −0.062) ✘ | −0.033 (−0.037 to −0.029) ✘ | lower bound > 0 |
| **H4a** excess calibration error | −0.022 (−0.034 to −0.012) ✔ | −0.223 (−0.256 to −0.191) ✔ | −0.207 (−0.230 to −0.183) ✔ | upper bound < 0 |
| **H4b** trigger share | −0.045 (−0.065 to −0.024) ✔ | frozen −0.900 (−0.900 to −0.900) ✔; deviation −0.541 (−0.609 to −0.472) ✔ | frozen −0.700 (−0.700 to −0.700) ✔; deviation −0.526 (−0.575 to −0.477) ✔ | upper bound < 0 |

**Overall:** H1 does not hold (eICU), H2 and H3 do not hold (every stream), **H4 holds**.

Notes on the table:
- **The eICU pools** are of 35 hospitals with high heterogeneity: I² 0.94–0.97 for H1 and H4a. The forest plots are
  [figures/forest_eicu_a.png](figures/forest_eicu_a.png) and [figures/forest_eicu_b.png](figures/forest_eicu_b.png).
  - Every hospital's AUROC difference is negative.
  - In (b), 21 of 35 hospital CIs reach the −0.02 margin, but the pooled estimate is just beyond it.
  - Every hospital's calibration difference favours PERISIM.
- **H2's pools leave out hospitals where neither arm has any excess variance** (a ratio of exactly 1): 5 in (a) and
  11 in (b) for the AUROC. A tie at 1 cannot meet the rule, so this cannot change the verdict.
- **The confidence intervals are narrow** because each stream holds a whole cohort, and the metric and the
  transported models are fixed. They describe the variation between windows and orders. They leave out the
  uncertainty of having learned the metric and the models.

**H6** (eICU (a), out-of-support diagnostic D1):

| quantity | estimate (95 % CI) | rule |
|---|---|---|
| (a) out-of-support rate after the change of year − other windows | −0.0015 (−0.0026 to −0.0005) ✘ | lower bound > 0 |
| (b) out-of-support alarms in the other windows | 0.23 % (0.04 to 0.48 %) ✔ | upper bound ≤ 5 % |
| (c) ML monitor (CITL = 0, two-sided 1 %): signal / alarms in the other windows | 0.016 (−0.022 to 0.056) / 27.3 % (22.0 to 32.5 %) | reported |

H6 does not hold. D1 is quiet as designed (its false alarms are rare), but it does not rise at eICU's change of year.
There is little to rise for: the change of year moves prevalence by a median of −0.7 points (prereg §10). The ML
monitor's test fires in 27 % of the control windows. That is because the ML is miscalibrated there (§4.2), not
because of noise.

## 4. Calibration: where the advantage comes from

### 4.1 MIMIC-IV: the cost of the cycle

Levels, means over windows ([endpoints.md §3](reports/endpoints.md)):

| arm | AUROC | calibration in the large | O/E | excess calibration error | trigger share |
|---|---|---|---|---|---|
| PERISIM-FIFO | 0.771 | −0.040 | 0.979 | 0.063 | 0.022 |
| PERISIM cumulative | 0.778 | −0.067 | 0.964 | 0.097 | 0.124 |
| PERISIM static | 0.771 | −0.190 | 0.901 | 0.221 | 0.409 |
| LightGBM frozen | 0.787 | 0.403 | 1.255 | 0.412 | 0.992 |
| LightGBM after go-live (4b) | 0.787 | 0.012 | 1.015 | 0.181 | 0.334 |
| **arm 5**, LightGBM realistic | 0.787 | −0.059 | 0.970 | 0.085 | 0.067 |
| arm 5, L = 0 | 0.787 | −0.041 | 0.979 | 0.071 | 0.045 |
| arm 6, LightGBM ideal | 0.787 | −0.012 | 0.994 | 0.062 | 0.018 |

- **The drift that matters is prevalence**, from 32.0 % in 2008–2010 to 25.6 % in 2020–2022.
  - The static case base keeps the old rate and over-predicts: O/E 0.90 on average.
  - The rotating window tracks the rate.
  - The transported LightGBM under-predicts MIMIC-IV until its go-live update corrects it (calibration in the large
    0.40 → 0.01). It then drifts again between updates (saw-tooth amplitude 0.039 in calibration in the large per
    cycle).
- **PERISIM-FIFO against each version of the ML** (excess calibration error / trigger share):
  - arm 5: −0.022 / −0.045;
  - L = 0: −0.008 (−0.021 to −0.000) / −0.024;
  - the four other triggers: −0.057 to −0.095 / −0.13 to −0.20. The AUROC and log-loss triggers watch discrimination
    or overall fit, and Davis et al.'s detector reacts later, so calibration drifts further between updates;
  - **the ideal arm: +0.001 (−0.008 to 0.009) / +0.003 (−0.007 to 0.016), a tie.**
- **The reading rule declared in advance** (prereg §10, "the cycle, not the drift") does not apply as written.
  PERISIM does not lose with L = 0 or with the more sensitive triggers. But it ties the ideal arm, which has no
  cycle. So in MIMIC-IV, what PERISIM gains is the update cycle's cost:
  - the delay;
  - the trigger's lag;
  - the drift between deployments.

  PERISIM tracks drift as well as an ML refreshed every 100 patients, without the monitoring, decision and deployment
  that arm 5 needed about 6 times per order. It does not resist drift better than that ML.

### 4.2 eICU: a transported model that small-sample updating cannot fix

| arm, eICU (a) / (b) | AUROC | calibration in the large | excess calibration error |
|---|---|---|---|
| PERISIM-FIFO | 0.763 / 0.781 | −0.041 / −0.040 | 0.092 / 0.099 |
| LightGBM frozen | 0.803 / 0.802 | −0.443 / −0.443 | 0.444 / 0.448 |
| arm 5 | 0.802 / 0.802 | −0.273 / −0.214 | 0.317 / 0.317 |
| arm 6 (ideal) | 0.802 / 0.802 | −0.105 / −0.083 | 0.209 / 0.210 |
| arm 5′ (trained on the starting window) | 0.775 / 0.785 | 0.007 / −0.001 | 0.114 / 0.221 |

These are means over hospitals of each hospital's value, descriptive.

- **The transported LightGBM over-predicts eICU heavily**: calibration in the large −0.44, O/E 0.77.
- **The pre-registered go-live update rarely corrects it.** It is the closed test (Vergouwe et al., 2017, adapted) on
  the starting window, split 2 : 1 by patient, at one-sided α = 0.05. With 100 patients (67 to fit, 33 to test) the
  test seldom shows a gain:
  - go-live kept the transported model in **267 of 350** hospital-orders in (a) and **212 of 350** in (b);
  - later updates are rare: about 1.1 triggers and 0.8 deployments per hospital-order in (a), and the closed test
    chose "keep" in 229 of 293 decisions.
- **Even the ideal arm**, which runs the same test every 50 patients, ends with an excess calibration error of 0.21.
- **PERISIM's local case base is calibrated by construction**: its votes come from the hospital's own recent
  patients. In eICU, H4 therefore mostly measures a local case base against a transported model under a
  conservative updating rule on little data.

### 4.3 Fairness: what favours PERISIM here

The design's fairness review (design §14, point 2) added the go-live update so that the transported ML would not start
miscalibrated. In MIMIC-IV it worked (calibration in the large 0.40 → 0.01). In eICU, at N = 100–250, it mostly did
not. The closed test keeps the model unless the held-out data show a gain, and 33–83 held-out patients rarely do.

A hospital deploying a model that over-predicts by that much would probably recalibrate its intercept at go-live
regardless. The pre-registered policy did not.

The comparison with arm 5′ bounds the effect:
- an ML trained on the same starting window is calibrated: calibration in the large 0.007 in (a);
- PERISIM's H4 advantage over it in (a) is −0.023 (−0.038 to −0.008) in excess calibration error;
- and −0.21 (−0.31 to −0.11) in trigger share, after the deviation.

That advantage is real but an order of magnitude smaller. Arm 5′ pays for its calibration in AUROC (0.775 against
0.802). PERISIM-FIFO is 0.012 below it in (a) and 0.004 below it in (b), both within H1's margin.

### 4.4 Against logistic regression

Against logistic regression managed the same way, PERISIM-FIFO is within H1's margins in MIMIC-IV:
- AUROC −0.009 (−0.010 to −0.009);
- AUPRC − prevalence −0.010.

It is also within them in eICU (b):
- AUROC −0.010;
- AUPRC − prevalence −0.012.

It is not within them in eICU (a): AUROC −0.027.

It meets H4's rules in all three. In the INSPIRE control at N = 500, against logistic regression: AUROC −0.0071
(−0.0090 to −0.0055) and AUPRC − prevalence +0.000. As expected (prereg §10), the accuracy half holds against
logistic regression in MIMIC-IV and INSPIRE.

## 5. Stability, the low tail and the window (H2, H3, H7)

- **H2: the rotating case base adds variation from window to window.** With N patients, the neighbours of a window's
  patients come from a sample that changes every 10 patients. The excess SD of the window AUROC is 1.23 times arm 5's
  in MIMIC-IV, and 1.32 times in eICU (a).
  - Against the cumulative arm, which averages over more patients, FIFO is less discriminating:
    - −0.007 in MIMIC-IV;
    - −0.017 in eICU (a).
  - But it is better calibrated in MIMIC-IV, where prevalence drifts (−0.034), and worse in eICU (a), within one
    hospital (+0.025). That is the cost of forgetting (design §14, point 8).
- **H3** follows from the lower level. The 10th percentile of the window AUROC is lower by about the mean difference
  (−0.016 in MIMIC-IV, −0.043 in eICU (a)).
- **H7, the window N** ([figure](figures/h7_by_N.png)).
  - **The AUROC gap narrows as N grows**, and **the calibration advantage persists**. In MIMIC-IV temporal:
    - N = 250: −0.022; N = 500: −0.016; N = 1,000: −0.013; N = 2,000: −0.012;
    - excess calibration error −0.02 to −0.05 at every N.
  - **H1 and H4 are met together only in MIMIC-IV temporal, from N = 500.**
  - **Cross-cohort streams:** H1 is not met up to N = 2,000. The AUROC gap is −0.019 to −0.023 there, and the
    AUPRC − prevalence gap −0.032 to −0.033.
  - **INSPIRE control**, with no drift: H1 is met from N = 1,000, but the calibration advantage vanishes there
    (−0.009, CI includes 0). With no drift to track, the rotation has nothing to gain.

## 6. Boundaries, diagnostics and the cross-cohort streams

### 6.1 The cross-cohort streams and the control (secondary; estimates, N = 500)

| stream | AUROC | excess SD ratio, AUROC | excess calibration error | vs the ideal arm, excess calibration error |
|---|---|---|---|---|
| INSPIRE → MIMIC-IV | −0.025 (−0.025 to −0.023) | 0.43 (0.34 to 1.00) | −0.106 (−0.205 to −0.060) | −0.021 (−0.068 to 0.012) |
| eICU → MIMIC-IV | −0.023 (−0.024 to −0.022) | 0.68 (0.56 to 0.72) | −0.171 (−0.231 to −0.137) | −0.049 (−0.083 to −0.010) |
| MIMIC-IV → INSPIRE | −0.029 (−0.030 to −0.027) | 0.40 (0.36 to 0.86) | −0.084 (−0.154 to −0.032) | −0.033 (−0.067 to −0.002) |
| INSPIRE control | −0.022 (−0.023 to −0.020) | 1.28 (0.80 to 2.04) | −0.067 (−0.110 to −0.005) | −0.043 (−0.067 to 0.000) |

- **A change of cohort is the one real shift in these data.** The ML monitor's PSI is 0.11–0.29 in the cross-cohort
  streams, against 0.04 in MIMIC-IV temporal.
- **There, PERISIM-FIFO is better calibrated than arm 5 in all three streams.**
  - It is more stable in two of the three. INSPIRE → MIMIC-IV's ratio CI reaches 1.
  - It is better calibrated than the ideal arm in two of the three. INSPIRE → MIMIC-IV's CI includes 0.
  - This is the pattern the thesis predicts, in the secondary streams only.
- **H5, recovery** after the change of cohort: the patients needed to come within δ of the arm's stable level.
  - Calibration (δ = 0.10 in calibration in the large):
    - PERISIM-FIFO 480–630;
    - arm 5 1,120–1,680;
    - the ideal arm 660–1,500.
  - AUROC (δ = 0.02): FIFO 160–360, arm 5 220–510.
  - None was censored.
  - At MIMIC-IV's block changes, where little changes, arm 5 recovers calibration sooner (140 patients against 290).

### 6.2 PERISIM's diagnostics (D1, D2)

- **D1, the out-of-support rate, stays at its nominal 1 % in every stream** (0.96–1.07 %).
  - It barely moves after the change of cohort: 1.62 % into MIMIC-IV from INSPIRE, 1.08 % from eICU, and 0.70 %
    into INSPIRE.
  - Its one-sided 1 % alarm fired in 1.5 % of the INSPIRE → MIMIC-IV windows, and in at most 0.3 % elsewhere.
  - The re-learn trigger (D1 ≥ 3 % in 3 consecutive windows) never fired.
- **D2, re-deforming the shapes on the current window** (MIMIC-IV temporal, order 0; 28 refits):
  - median gain −0.0006 nats on the next window, range −0.006 to +0.010;
  - one refit passed the re-learn rule, by 0.0003 nats.

  This agrees with `hypo`: the transported shapes do not need to deform. Within these cohorts, the change is in the
  case mix and the rate, which the rotating case base absorbs.
- **Saw-tooth.** Between its deployments, arm 5 lost 0.005–0.008 AUROC per cycle. Its calibration in the large moved
  by 0.04–0.17 per cycle. PERISIM has no cycle.

## 7. Governability, measured

- **Determinism:** the first eICU (a) unit, rerun from scratch with 1 thread, gave the same predictions digest
  (`aa494d28b25944d7`) as the run with 6 threads.
- **Reconstruction:** the FIFO window at every step was rebuilt from the stream log alone, with 0 mismatches in 159
  steps.
- **Updates:** PERISIM updates every 10 patients and changes no parameter: new patients enter the case base and the
  oldest leave.
  - Arm 5 went through about 6 trigger–decision–deployment cycles per order in MIMIC-IV, and 2–3 in the cross-cohort
    streams.
  - Each of those cycles stands for monitoring, a decision, revalidation and a 3-month delay in a hospital.
  - The closed test chose "keep" in most decisions: 21 of 29 in MIMIC-IV.
  - It changed 0.41 parameters per deployment on average in MIMIC-IV, counting the deployments that kept the model.

## 8. T2 (secondary, descriptive)

T2's frozen method on MIMIC-IV temporal (N = 2,000, W = 250, 5 orders, 85 windows; 4-class outcome):

| | log-loss | ranked probability score |
|---|---|---|
| PERISIM-FIFO (level) | 1.050 | 0.134 |
| FIFO − cumulative | +0.017 (0.013 to 0.019) | +0.002 (0.002 to 0.003) |
| FIFO − static | −0.023 (−0.027 to −0.020) | −0.004 (−0.005 to −0.004) |
| FIFO − the window's class shares | −0.103 (−0.105 to −0.101) | −0.018 (−0.018 to −0.017) |

- The rotating window beats a fixed case base and the class rates.
- It loses to the growing case base: with T2's rare classes, more patients help more than recency.

## 9. What the pre-registration expected, and what happened

| expected (prereg §10) | result |
|---|---|
| H1 fails against LightGBM everywhere at the primary N: −0.018 / −0.040 / −0.023 | fails in eICU (−0.039 / −0.021); **holds narrowly in MIMIC-IV** (−0.016; the AUPRC bound −0.0295 against −0.03) |
| Against logistic regression, H1 holds in MIMIC-IV and INSPIRE at 500 | yes: −0.009 and −0.007 |
| Little drift in the primary streams; differences in calibration rather than discrimination, and in the cross-cohort streams | yes: PSI 0.04 in MIMIC-IV, D1 flat, D2 null; the differences are in calibration, largest across cohorts |
| eICU (a)'s calibration counts against PERISIM in H4 (a) | no. FIFO is under-confident there (slope 1.37), but its excess calibration error (0.092) is far below arm 5's (0.317), whose transported model stays miscalibrated (§4.2) |
| The null is publishable | not needed: the arms differ, in calibration |
| "The cycle, not the drift" | in MIMIC-IV, PERISIM ties the ideal ML, so its gain is the cycle's cost (§4.1); in eICU, the gain is mostly the go-live rule's weakness at small N (§4.3) |

## 10. Deviations and limitations

**Deviations:**
- **One declared deviation, after the run** (DECISIONS 2026-10-31).
  - The frozen DerSimonian–Laird pooling leaves out hospitals with a zero SE. Two hospitals had an SE of ~1e-16
    (floating-point noise around zero) in the trigger share. Each received an almost infinite weight and made the
    pooled value its own.
  - `scripts/roll_deviation_pooling.py` treats SEs ≤ 1e-12 as zero and calls the same frozen function.
  - It changes 30 pooled H4 (b) results and nothing else. The confirmatory verdicts do not change.
  - One reported comparison changes sign. Against arm 5′ in eICU (a), the trigger share goes from +0.20 to −0.21.
    The frozen value was one hospital's.
  - Both outputs are kept ([results/eicu_pooled.json](results/eicu_pooled.json),
    [results/eicu_pooled_deviation.json](results/eicu_pooled_deviation.json)).
- **One interim look**, by the lead's decision: MIMIC-IV temporal at N = 500, during the run, with the frozen code.
  Nothing was changed after it, and the final numbers are identical.
- **None to the code or the configuration.** The freeze check passed at every start and end.

**Limitations:**
- **Declared in advance** (prereg §11):
  - three cohorts, all examined before;
  - real time order only by blocks (MIMIC-IV) and years (eICU), random inside them;
  - few boundaries, and small real drift;
  - an adapted closed test and a reimplemented Davis et al. detector.
- **Arising from the results:**
  - **The go-live and update rule is weak at N = 100–250.** It left the transported model miscalibrated in most eICU
    hospitals, which favours PERISIM in eICU's H4 (§4.3).
  - **The CIs leave out the uncertainty of the learned metric and models** (§3).
  - **The frozen pooling has a numerical edge case** (the deviation), and it also leaves out H2's exact ties at 1.
  - **H4 (b) counts checks, and arm 5's delay keeps a miscalibrated model live for L patients after a trigger.** That
    is the realistic arm's design, but it means H4 (b) and L interact. The L = 0 variant is reported next to it
    (§4.1).

## 11. Conclusions

1. **On accuracy, the thesis does not hold.** A rotating case base of a few hundred patients is not as discriminating
   as a transported LightGBM, except in MIMIC-IV at N ≥ 500, and there by a small margin. Against logistic regression
   it is level.
2. **On calibration it holds in every confirmatory stream**, without monitoring, triggers or deployments:
   - in MIMIC-IV it matches an ML refreshed every 100 patients;
   - in eICU it beats a transported ML that the pre-registered small-sample updating cannot repair. That advantage
     is partly the updating rule's weakness (§4.3).
3. **On stability it does not hold within a cohort.** Rotating the case base costs window-to-window variation. Across
   a change of cohort, the rotation is what makes PERISIM steadier and quicker to recover its calibration.
4. **The diagnostics are quiet because these data drift little**, apart from the rate. D1 keeps its nominal rate, does
   not rise at eICU's change of year, and barely moves at a change of cohort. D2 confirms `hypo`: the shapes need no
   re-deformation.
5. **Governability holds as measured**: deterministic, reconstructible from the log, updated without changing any
   parameter. That is the practical case for CE-kNN in drifting rates: the calibration of a continuously refreshed
   model without the refresh cycle, at a cost of 0.01–0.04 AUROC.

## Reproducibility

- **Code and configuration**: tag `perisim-freeze-roll-2026-10-30-173256`, commit `a9cc6d00f8c9`, code `b630267b6518`,
  configuration `734680c6193c87c8`, streams digest `98e8c48712104ce5`. The freezes it stands on are T2's
  (`perisim-freeze-t2-2026-10-27-141040`) and `hypo`'s (`perisim-freeze-hypo-2026-10-29-120652`), checked by `roll`
  at every start and end.
- **Commands**, from the frozen checkout:

  ```
  python -m roll run [--streams …] [--N …]
  python -m roll endpoints
  python -m roll t2
  ```

  Then, outside the frozen code:

  ```
  python scripts/roll_deviation_pooling.py
  python scripts/roll_report.py
  ```

  These read the frozen output, write aggregates, and check the freeze at their start and end.
- **Compute**: one workstation.
  - The run took 5 h 14 min of wall time in four parts (16.2 h of unit compute).
  - The endpoints took 3 h 55 min, and T2 8 min.
- **Data**: patient-level data, predictions and logs stay on that computer under the PhysioNet data use agreements.
  The aggregates are in [results/](results/): every stream's endpoints, eICU's pools and per-hospital tables
  (H1, H2, … by size), T2, governability, the run's counts, and the freeze record.

## References

DOIs checked on Crossref on 2026-10-30 (design §16).

- Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics, 24*(4), 295–300.
  https://doi.org/10.1080/00401706.1982.10487790
- Davis, S. E., Greevy, R. A., Jr., Lasko, T. A., Walsh, C. G., & Matheny, M. E. (2020). Detection of calibration
  drift in clinical prediction models to inform model updating. *Journal of Biomedical Informatics, 112*, 103611.
  https://doi.org/10.1016/j.jbi.2020.103611
- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials, 7*(3), 177–188.
  https://doi.org/10.1016/0197-2456(86)90046-2
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation.
  *ACM Computing Surveys, 46*(4), Article 44. https://doi.org/10.1145/2523813
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of
  Statistics, 17*(3), 1217–1241. https://doi.org/10.1214/aos/1176347265
- Losing, V., Hammer, B., & Wersing, H. (2016). KNN classifier with self adjusting memory for heterogeneous concept
  drift. In *2016 IEEE 16th International Conference on Data Mining (ICDM)* (pp. 291–300). IEEE.
  https://doi.org/10.1109/ICDM.2016.0040
- Shaker, A., & Hüllermeier, E. (2012). IBLStreams: A system for instance-based classification and regression on data
  streams. *Evolving Systems, 3*(4), 235–249. https://doi.org/10.1007/s12530-012-9059-0
- Vergouwe, Y., Nieboer, D., Oostenbrink, R., Debray, T. P. A., Murray, G. D., Kattan, M. W., Koffijberg, H.,
  Moons, K. G. M., & Steyerberg, E. W. (2017). A closed testing procedure to select an appropriate method for updating
  prediction models. *Statistics in Medicine, 36*(28), 4529–4539. https://doi.org/10.1002/sim.7179
