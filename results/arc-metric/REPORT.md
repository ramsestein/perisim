# Arc-length placement of the shape functions: results of the two-arm re-run

Pre-registration: [docs/prereg_arc.md](../../docs/prereg_arc.md), registered on OSF as
[10.17605/OSF.IO/NAQ2T](https://doi.org/10.17605/OSF.IO/NAQ2T) on 2026-11-08, before either arm touched real data.
Code: branch `arc-metric`, commit `05ecffd` for both arms (every freeze record carries it). Arm `value` ran
2026-11-08 → 11-11, arm `arc` 2026-11-11 → 11-13, on the same machine, data and seeds. The paired contrast is
[arc_compare.md](arc_compare.md) and [compare.json](compare.json). Each arm's aggregate outputs are under
[value/](value/) and [arc/](arc/). Deviations: [§6](#6-deviations).

## 1. Summary

- **The two placements predict practically the same.** In the six primary units, the paired difference in log-loss
  of the transported metric is at most 0.0014 nats in either direction. That is the expectation declared in §6.1
  (within ±0.002).
- **Arc is never worse.** It is better in two units (`hypo` eICU −0.0006 and `hypo` MIMIC-IV −0.0008 nats, 95 % CIs
  below 0). In the other four there is no difference.
- **Overall equivalence, by the registered rule, is not established.** The rule requires Q's 90 % CI inside
  [0.80, 1.25] in every evaluable primary unit. Five units meet it. NWICU with the MIMIC-IV rotation's metric does
  not: its upper limit is 1.314, so it fails on the side favourable to arc. No unit fails on the unfavourable side.
- **"Arc predicts better" is not met either.** That reading needs D favouring arc in at least four of six units; it
  does in two.
- **Every experiment's own conclusions are the same in both arms.** The exceptions are two T2 threshold readings in
  MIMIC-IV and roll's H1 in MIMIC-IV at the AUPRC margin. None of them changes an overall verdict (§4).
- **Why so little changes.** The consensus curves g_j are identical in both arms, as predicted. The variables that
  backtrack most (temperature, SpO₂, heart rate, creatinine) carry almost no weight. The heavily weighted MAP
  variables and the landmark time L are nearly monotone, and on a monotone curve the two placements coincide (§5).

## 2. Primary contrast (prereg §6.1)

Transported metric, paired on the same rows. 2,000 patient-level bootstrap resamples per unit.

| unit | rows (patients) | L value | L arc | D = L arc − L value (95 % CI) | Q (90 % CI) | reading |
|---|---|---|---|---|---|---|
| `hypo` INSPIRE | 148,293 (11,631) | 0.3274 | 0.3276 | +0.0002 (−0.0005 to +0.0010) | 0.979 (0.922–1.041) | equivalent; D no difference |
| `hypo` MIMIC-IV | 387,158 (24,464) | 0.5048 | 0.5040 | −0.0008 (−0.0014 to −0.0003) | 1.091 (1.040–1.147) | equivalent; **D arc better** |
| `hypo` eICU | 466,384 (32,032) | 0.3966 | 0.3960 | −0.0006 (−0.0010 to −0.0002) | 1.045 (1.022–1.070) | equivalent; **D arc better** |
| NWICU, metric of rotation INSPIRE | 30,833 (2,001) | 0.3192 | 0.3196 | +0.0004 (−0.0010 to +0.0018) | 0.962 (0.843–1.089) | equivalent; D no difference |
| NWICU, metric of rotation MIMIC-IV | 30,833 (2,001) | 0.3191 | 0.3177 | −0.0014 (−0.0030 to +0.0003) | 1.138 (1.002–1.314) | **not equivalent (arc better)**; D no difference |
| NWICU, metric of rotation eICU | 30,833 (2,001) | 0.3183 | 0.3188 | +0.0005 (−0.0009 to +0.0018) | 0.956 (0.857–1.070) | equivalent; D no difference |

Q = (L_euclid − L_arc) / (L_euclid − L_value) is the share of the value placement's gain over the Euclidean kNN that
the arc placement achieves. Every unit is evaluable: the value placement's gain is ≥ 0.005 nats, with a 95 % CI
above 0.

**Overall reading (§6.1).**
- Equivalence: not established. One of six evaluable units, NWICU-MIMIC-IV, is outside the band on the
  arc-better side.
- Arc predicts better: not established (two of six units, at least four needed).
- Arc worse: in none.

The NWICU units have 2,001 patients, and their Q intervals are several times wider than those of `hypo`.

## 3. Secondary (prereg §6.2)

### 3.1 Transport within each arm

R = (L_euclid − L_transported) / (L_euclid − L_local), published rule: equivalent when the lower 90 % limit is ≥ 0.80.

| unit | R value (90 % CI) | R arc (90 % CI) | ΔR = R arc − R value (95 % CI) |
|---|---|---|---|
| `hypo` INSPIRE | 1.031 (0.945–1.122) | 0.908 (0.837–0.987) | **−0.122 (−0.236 to −0.017)** |
| `hypo` MIMIC-IV | 0.887 (0.833–0.942) | 0.898 (0.851–0.942) | +0.011 (−0.056 to +0.074) |
| `hypo` eICU | 0.964 (0.930–0.998) | 0.970 (0.938–1.000) | +0.006 (−0.035 to +0.043) |
| NWICU, rotation INSPIRE | 1.405 (1.062–1.963) | 1.416 (1.037–2.050) | +0.010 (−0.632 to +0.638) |
| NWICU, rotation MIMIC-IV | 1.420 (1.075–2.007) | 1.691 (1.311–2.355) | +0.271 (−0.372 to +0.988) |
| NWICU, rotation eICU | 1.526 (1.164–2.177) | 1.528 (1.138–2.123) | +0.002 (−0.724 to +0.605) |
| T2 MIMIC-IV | 0.484 (0.292–0.665) | 0.497 (0.311–0.665) | +0.013 (−0.178 to +0.222) |
| T2 eICU | 0.698 (0.571–0.817) | 0.655 (0.527–0.772) | −0.043 (−0.178 to +0.084) |

- **Transport is equivalent in both arms** in all six `hypo` and NWICU units, and not equivalent in both arms in
  T2, as published.
- **The one ΔR that differs, `hypo` INSPIRE,** comes from the denominator. With arc, the metric learned at INSPIRE
  improves (D_local −0.0012 nats). The transported metric does not change (D +0.0002). The ratio therefore falls,
  although transport does not get worse.

### 3.2 Locally learned metrics

| unit | D local (95 % CI) | Q local (90 % CI) |
|---|---|---|
| `hypo` INSPIRE | −0.0012 (−0.0021 to −0.0002) | 1.111 (1.033–1.195) |
| `hypo` MIMIC-IV | −0.0008 (−0.0013 to −0.0003) | 1.078 (1.038–1.121) |
| `hypo` eICU | −0.0005 (−0.0010 to −0.0001) | 1.039 (1.013–1.068) |
| NWICU (the same local metric in the three rows) | +0.0003 (−0.0021 to +0.0027) | 0.955 (0.71–1.31) |

Within its own site, the arc placement is slightly better at all three `hypo` sites, by 0.0005–0.0012 nats.

### 3.3 T2 (secondary units)

| unit | D (95 % CI) | Q (90 % CI) | reading |
|---|---|---|---|
| T2 MIMIC-IV | −0.0025 (−0.0110 to +0.0056) | 1.118 (0.808–1.663) | not equivalent (arc better); D no difference |
| T2 eICU | +0.0030 (−0.0043 to +0.0108) | 0.931 (0.792–1.077) | not equivalent (arc worse); D no difference |

T2's gain over the Euclidean kNN is small (0.02–0.04 nats), so Q is imprecise in both directions. T2's internal
cross-validation at INSPIRE (descriptive, not part of the contrast):

| method | log-loss value / arc | AUROC of the severe outcome, value / arc |
|---|---|---|
| PERISIM | 0.9417 / 0.9438 | 0.699 / 0.658 |
| NCA diagonal | 0.9407 / 0.9482 | 0.685 / 0.631 |
| HVDM | 0.9527 / 0.9533 | 0.685 / 0.676 |
| Euclidean (independent of the arm) | 0.9906 / 0.9906 | 0.635 / 0.635 |

### 3.4 Each experiment's own hypotheses in both arms

| experiment | hypothesis | value | arc | same conclusion |
|---|---|---|---|---|
| T2 transport, MIMIC-IV | H1 · H2 · H3 · H4 · H5 | ✘ ✘ ✔ ✔ ✘ | ✘ ✘ ✘ ✔ ✔ | **H3 and H5 differ** |
| T2 transport, eICU | H1 · H2 · H3 · H4 · H5 | ✘ ✘ ✘ ✔ ✘ | ✘ ✘ ✘ ✔ ✘ | yes |
| T2 transport | H6 | ✘ | ✘ | yes |
| `hypo` (each site) | H1 · H2 · H3 | ✔ ✔ ✔ at the three sites | ✔ ✔ ✔ at the three sites | yes |
| `hypo` (eICU) | H4 · H5 | ✔ ✔ | ✔ ✔ | yes |
| NWICU | R of each transported metric | equivalent ×3 | equivalent ×3 | yes |
| roll | H1 · H2 · H3 · H4, overall | ✘ ✘ ✘ ✔ | ✘ ✘ ✘ ✔ | yes |
| roll, MIMIC-IV temporal only | H1 (AUROC and AUPRC − prevalence) | ✘ (AUPRC − prev. −0.031, lower limit −0.0315) | ✔ (−0.027, lower limit −0.028) | **differs** |
| post hoc §16 (transport equivalence of `hypo`) | R per site | equivalent ×3 | equivalent ×3 | yes |
| post hoc §21 (cosine kNN) | NCA beats cosine, 4 targets | ✔ ×4 | ✔ ×4 | yes |

- **T2 MIMIC-IV.** H3 (the frozen metric discriminates almost as well as the local one) and H5 (an external start
  helps up to a crossing point) are threshold readings near their margins. They change in opposite directions, and
  neither T2's overall conclusion nor its R changes.
- **roll.** H1 does not hold in either arm, because eICU fails. In MIMIC-IV, the arc arm meets the AUPRC margin of −0.03
  that the value arm misses (lower limit −0.0315), and that the published v0.5 met.
- **Post hoc §20** (transport between eICU hospitals) is reported in each arm's
  `hypo/reports/posthoc_hospital_transport.md`, read descriptively.

## 4. Reproducibility: arm value against the published runs (v0.6 against v0.5, §6.3)

v0.6 reproduces the published conclusions everywhere. Differences in the headline figures:
- **`hypo`:** ≤ 0.001 nats.
- **NWICU:** R lower by 0.05–0.11. The numerator is unchanged (log-loss 0.319); the denominator is about 0.007 nats.
- **T2:** ≤ 0.005 nats in every method, and the severe-outcome AUROC about 0.02–0.03 lower.
- **roll:** H1–H4 verdicts unchanged.

The out-of-support rate of T2's mode A rose from 15.5 % to 17.2 % in MIMIC-IV and from 3.5 % to 4.1 % in eICU. Both
changes are consistent with the v0.6 landmark and variable corrections. `hypo`'s landmark tables were rebuilt from
the copied extracts; their row counts match the published ones within one landmark per cohort.

## 5. Why the placement changes so little (descriptive, §6.3)

- **The curves are the same.** In `hypo` the consensus curves g_j are identical in both arms (largest difference
  < 1e-6), because the ensemble is fitted on the model matrix with the same seeds.
- **The arc placement can only differ where a curve backtracks.** The backtracking index 1 − net change / arc length
  measures how much of the path along g_j the value placement cancels. In the three frozen metrics of `hypo`:
  - **High backtracking (0.5–0.97):** temperature, SpO₂ (last and 6-h minimum), heart rate, respiratory rate,
    creatinine and lactate. Each has a weight share below 0.03 in both arms.
  - **The heavy variables:** MAP last, MAP mean over 6 h, MAP SD, MAP minimum, and the landmark time L carry
    about 0.75–0.85 of the weight. Their backtracking is 0–0.3 (MAP minimum 0.26–0.50).
- **The weights barely move.** The diagonal NCA re-weights the arc axes slightly; for example, the MAP-mean share is
  0.22 → 0.32 in the eICU rotation and the MAP-last share 0.32 → 0.20 in the INSPIRE rotation. The total weight on
  MAP and L stays roughly the same (0.76–0.80 with value, 0.78–0.85 with arc).

The full per-variable tables are in [arc_compare.md](arc_compare.md).

## 6. Deviations

Both are logged in [docs/DECISIONS.md](../../docs/DECISIONS.md).

1. **2026-11-08, mechanical.** `scripts/nwicu_build.py` read duckdb results with an API that duckdb ≥ 1.2 removed.
   It now uses whichever Arrow export the installed version has. This was fixed before any real-data run; the
   script is in no code hash, and the SQL is unchanged.
2. **2026-11-13, Euclidean check in T2's units.** The contrast stops if the Euclidean kNN, which does not depend on
   the arm, differs between the arms. It differed in exactly one row of each T2 unit: 0.031 and 0.034 nats on that
   row, about 7e-6 nats on the mean. This is consistent with a tie between equidistant neighbours.
   - The lead decided, before the T2 contrast was computed, to allow at most one differing row in T2's units. The
     value arm's Euclidean kNN is used, as §6.1 specifies. The primary units keep the exact check, and passed it
     exactly.
   - The primary contrast was first computed with the registered script and is identical under the change.

**Operational events.**
- A Windows update interrupted arm arc on 2026-11-12 during `hypo external`. That stage stores each unit atomically
  and skips the stored ones, so it resumed. The launcher was changed to skip steps already logged as finished; no
  analysis step was repeated after its freeze.
- An interrupted relaunch of arm value on the same morning rewrote only T2's audit acknowledgement. The arm's frozen
  artefacts and its contrast inputs are from 2026-11-11.

## 7. Reading and limitations

- **What the result supports.** The arc-length placement can replace the value placement with no loss of
  prediction or transport: it is never worse, and slightly better at the large `hypo` sites. It has the reading that
  motivated it: a continuous, monotone distance that adds up every stretch where the prediction changes, without
  cancelling the two sides of a U.
- **What it does not support.** Formal equivalence in every unit fails because of one imprecise NWICU unit on the
  favourable side. A predictive advantage in general is not supported either.
- **Same data as the published runs.** This is a comparison of two placements on the same cohorts, not an
  independent replication. The analysts knew the published results; the readings were fixed before the runs so that
  this could not steer them.
- **Effect sizes are small throughout** (≤ 0.0014 nats in the primary units), and smaller than many method
  differences already reported in the project. The practical case for arc is interpretive.
- **The variables where the two placements differ most have little weight here.** A target in which a
  non-monotone variable (temperature, pH, sodium) carries real weight would be a sharper test of the placement.
  This could be designed as a new, pre-registered study.
