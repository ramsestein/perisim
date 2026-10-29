# PERISIM `hypo`: hypotension in the next 6 h after surgery — report

Draft for the lead, 2026-10-29. Pre-registration [docs/prereg_hypo.md](../../docs/prereg_hypo.md), frozen on
2026-10-29 (tag `perisim-freeze-hypo-2026-10-29-120652`, registration `LOCAL-PREREG-HYPO-2026-10-29`; record
[docs/freeze_hypo.md](../../docs/freeze_hypo.md)). Every figure is from the reports in [reports/](reports/), mainly
[external.md](reports/external.md). Aggregates only.

## Summary

- **All five pre-registered hypotheses hold.** H1–H3 hold in each of the three external sites, H4–H5 in eICU. H5
  holds formally, but its gain (0.0001 nats) is of no practical use.
- **The metric transports.**
  - A metric learned on two cohorts and applied to the third discriminates as well as one learned on the third:
    ΔAUROC −0.001 to −0.003.
  - With the site's own case base it is calibrated (calibration in the large 0.008–0.023, slope 1.01–1.03).
  - The transported comparators trained in the same way are not calibrated: calibration in the large −0.40 to +0.50,
    because the three sites' rates differ (13 %, 31 %, 19 %).
- **Few local patients are needed.** A case base of 250 patients is within 0.02 nats of the whole site, and 500
  patients within 0.01, in all three sites.
- **The hospital matters in eICU, but less than in T2.** A hospital's own patients make a better case base than as
  many patients from other hospitals, after correcting the base rates: −0.0043 nats, against −0.0001 for random
  groups. T2's effect was about ten times larger (−0.041). Weighting own-hospital neighbours up adds nothing.
- **Drift: the shapes do not need to deform.**
  - Refitting every g_j to the external site gains nothing (Δ −0.0004 to +0.0007; every CI includes 0). The same holds
    in the development sites used as controls.
  - Not one of the 85 one-variable refits in the external sites has a CI below 0.
  - Re-deforming only the missing-value points gains nothing either.
- **What does differ between sites** does not cost prediction:
  - the base rate and case mix, which the local case base absorbs;
  - how the credit is split among the correlated MAP summaries (last, 6-h mean, 6-h minimum), which moves the
    per-variable MAP thresholds by as much as 5–8 mmHg (MIMIC-IV's `map_last`: 67 mmHg against 73 transported);
  - the shapes of low-weight inputs (heart rate, temperature, SpO₂).
- **Against general models**: PERISIM beats the last-MAP bar by 0.021–0.042 nats (H3) and is the best of the
  transported methods. It is level with logistic regression refitted locally (−0.001 to +0.004) and below LightGBM
  refitted locally (+0.011 to +0.016). That was expected: the experiment is centred on transfer and explainability,
  not on beating general models (design, opening paragraph).
- **Post hoc (exploratory, §13):**
  - the drift null holds over all five folds;
  - a threshold for the MAP block as a whole is about 78 mmHg in every site (eICU hospitals: interquartile range
    77–79);
  - MIMIC-IV gains a little from weighting a running vasopressor infusion (−0.0017 nats). This is a lead for a
    pre-registered test.

## 1. Question and design

- **Use case** (design §1): every 2 h during the first 48 h of a postoperative ICU stay, for a patient who is
  normotensive now:
  - the probability of hypotension in the next 6 h (an hour with median MAP < 65 mmHg);
  - the k = 50 most similar moments of other patients of the same site;
  - the reasons, read from the metric.
- **Metric**: a weighted distance over shape functions g_j learned once and shared; the case base stays in each site.
- **Leave one site out** (D1): for each external site X, the metric and the transported comparators are learned on
  all patients of the other two cohorts (`reports/final.md`). X is then predicted on its Phase 2 folds: case base =
  X's training folds, test = the fold. Every landmark of X is predicted once, out of fold.
- **Comparisons**:
  - the same metric learned on X's own training folds (local);
  - the Phase 2 local comparators (last MAP, logistic, LightGBM, Euclidean kNN, prevalence);
  - the transported comparators;
  - the Phase 2 in-sample PERISIM.
- **CIs**: 95 % from patient-level bootstraps (2,000 resamples for H1–H5, 500 for the rest), the same resamples for
  every method of a site.

## 2. Data

| | INSPIRE | MIMIC-IV | eICU |
|---|---|---|---|
| landmarks (patients) | 148,293 (11,631) | 387,159 (24,464) | 466,385 (32,032) |
| hypotension in the next 6 h | 13.2 % | 30.7 % | 19.3 % |
| landmarks with an arterial line | none (non-invasive only) | 70.5 % | 61.3 % |
| hospitals with ≥ 300 patients | — | — | 36 |

Phase 0 (`reports/landmarks_summary.md`, `reports/audit.md`) built the landmarks from the original tables and passed
the completeness audit. INSPIRE measures MAP non-invasively, about once an hour, and cannot tell whether an infusion
is running (D7). Its three infusion inputs are a source gap, constant within the site, so they add nothing to any
distance there (prereg §8).

## 3. Confirmatory results

| | INSPIRE | MIMIC-IV | eICU | rule |
|---|---|---|---|---|
| **H1** ΔAUROC transported − local | −0.003 (−0.005 to −0.001) ✔ | −0.003 (−0.004 to −0.002) ✔ | −0.001 (−0.001 to 0.000) ✔ | lower bound > −0.02 |
| **H2** calibration in the large | 0.023 (−0.006 to 0.050) | 0.021 (0.007 to 0.035) | 0.008 (−0.006 to 0.022) | CI inside ±0.10 |
| **H2** calibration slope | 1.019 (0.993 to 1.045) ✔ | 1.011 (0.999 to 1.024) ✔ | 1.025 (1.013 to 1.038) ✔ | lower bound ≥ 0.90 |
| **H3** Δlog-loss vs last MAP, local | −0.0213 (−0.0232 to −0.0196) ✔ | −0.0418 (−0.0432 to −0.0403) ✔ | −0.0250 (−0.0264 to −0.0237) ✔ | CI < 0 |

**H4 (eICU)** holds:
- own-hospital case base − other hospitals' (same size, base rates corrected): Δ_hosp −0.0043 (−0.0051 to −0.0035);
- random-group control: Δ_ctrl −0.0001 (−0.0008 to 0.0006);
- difference −0.0042 (−0.0052 to −0.0031).

**H5 (eICU)** holds formally. Δlog-loss against the plain vote, 98.33 % CIs:

| γ | Δlog-loss |
|---|---|
| 2 | −0.0001 (−0.0002 to −0.0001) |
| 4 | +0.0003 (0.0000 to 0.0005) |
| 8 | +0.0029 (0.0026 to 0.0034) |

The γ = 2 gain is a ten-thousandth of a nat, and larger γ make the predictions worse. The hospital's information is
in *which* neighbours a query has (H4), not in weighting them differently.

The calibration in the large is above 0 in every site, and its CI excludes 0 in MIMIC-IV. That is a slight
under-prediction (mean predicted 30.3 % against 30.7 % observed), well inside the margin. The slopes slightly
above 1 mean the predictions are, if anything, a little under-confident.

## 4. Transport and calibration (S2, S3)

Log-loss of PERISIM transported minus each method (negative: PERISIM better), on the same rows:

| method | INSPIRE | MIMIC-IV | eICU |
|---|---|---|---|
| PERISIM local (H1's arm) | +0.0010 (0.0000 to 0.0021) | +0.0028 (0.0021 to 0.0035) | +0.0005 (−0.0000 to 0.0011) |
| PERISIM in-sample, Phase 2 (two partners) | −0.0006 / −0.0001 | +0.0025 / +0.0017 | +0.0001 / −0.0003 |
| last MAP transported | −0.0308 | −0.0651 | −0.0374 |
| logistic transported | −0.0110 | −0.0121 | −0.0113 |
| LightGBM transported | −0.0020 | −0.0052 | −0.0021 |
| last MAP local | −0.0213 | −0.0418 | −0.0250 |
| logistic local | −0.0013 | +0.0044 | +0.0004 |
| LightGBM local | +0.0115 | +0.0155 | +0.0115 |
| Euclidean kNN local (same case base) | −0.0105 | −0.0079 | −0.0135 |

- **Transport costs nothing in INSPIRE and eICU**: the transported metric equals the local one and the in-sample
  ones (the Phase 2 rows, learned with the site and one partner).
- **In MIMIC-IV it costs 0.002–0.003 nats.** Every metric that saw MIMIC-IV beats the transported one by about that
  much: the local metric and both Phase 2 pairs. Re-deforming the shapes on MIMIC-IV does not recover the cost
  (§7.1), so it is not in the shapes. The weight differences of §7.4 are the candidate.
- **Calibration comes from the case base.** The transported comparators carry their training sites' rates, so their
  calibration in the large is −0.26 / +0.50 / −0.39 for last MAP, −0.15 / +0.35 / −0.37 for logistic, and
  −0.34 / +0.41 / −0.40 for LightGBM (INSPIRE / MIMIC-IV / eICU). PERISIM's is 0.008–0.023 with nothing refitted:
  the neighbours' outcomes are local.
- **Learned against neutral axes**: the learned metric beats a Euclidean kNN with the same case base and vote by
  0.008–0.014 nats.

## 5. How many local patients (S1)

Case bases of n local patients drawn at random, with the frozen metric (mean over draws):

| n patients | INSPIRE | MIMIC-IV | eICU |
|---|---|---|---|
| 50 | +0.042 | +0.080 | +0.057 |
| 100 | +0.023 | +0.030 | +0.025 |
| 250 | +0.010 | +0.012 | +0.011 |
| 500 | +0.006 | +0.008 | +0.007 |
| 1,000 | +0.004 | +0.005 | +0.005 |
| 8,000 | +0.001 | +0.001 | +0.001 |

Each cell is the log-loss minus that of the whole training pool (9,300–25,600 patients).

- 250 patients come within 0.02 nats of the whole site, and 500 within 0.01, in every site.
- At equal local data PERISIM beats the last-MAP model recalibrated on the same patients, by 0.013–0.031 at n = 250
  and 0.017–0.035 at n = 500.
- Logistic regression recalibrated on 500 patients is level with or slightly better than PERISIM (+0.000 to +0.006).
  Recalibrated LightGBM is better (+0.010 to +0.016).
- T2 needed thousands of patients. Here a few hundred suffice: the outcome is frequent, so a neighbourhood of 50
  holds 6–15 events.

## 6. Hospitals in eICU (H4, H5, S6)

- **Own hospital is better in 16 of the 36 eligible hospitals** (CI below 0); in none is it worse (CI above 0). The
  hospital-level Δ ranges from −0.0115 to +0.0051.
- **The control is null** (−0.0001): the effect is not an artefact of drawing a smaller, more homogeneous case base.
- **Own-hospital neighbours are scarce in a pooled case base**: 2–11 % of a query's neighbours come from its own
  hospital, against 1–6 % of the case base. The pooled metric already prefers them slightly, and weighting them up
  (H5) changes nothing.
- **Reading**: hospital-specific relations exist, but they are small (0.004 nats, about a tenth of T2's).
  - The advantage of a hospital's own patients is smaller than what a larger case base gains: 0.006 nats from 500 to
    8,000 patients (S1). So for accuracy alone, a large pooled case base and a hospital's own should be close; that
    comparison was not run directly.
  - Keeping the case base in each hospital (design §1) is then a matter of governance, and of showing a clinician
    their own patients, more than of accuracy.

## 7. Drift: how the shapes deform across sites (S4, S5)

### 7.1 Re-deforming the frozen shapes to each site (S4, fold 0)

Gain from re-deforming the frozen metric to each site (negative = better than frozen). Every point of every g_j is
refitted ("all variables"), or only the missing-value points ("missing only"):

| rotation : site | role | all variables | missing only | one-variable refits with CI < 0 |
|---|---|---|---|---|
| external-INSPIRE : INSPIRE | external | −0.0004 (−0.0024 to 0.0015) | +0.0001 (−0.0004 to 0.0006) | 0 / 29 |
| external-MIMIC-IV : MIMIC-IV | external | −0.0004 (−0.0027 to 0.0018) | −0.0003 (−0.0008 to 0.0001) | 0 / 29 |
| external-eICU : eICU | external | +0.0007 (−0.0010 to 0.0021) | 0.0000 (0.0000 to 0.0000) | 0 / 27 |
| the six development-site controls | in-sample | −0.0002 to +0.0012 | −0.0002 to +0.0007 | 0–2 per site (5 in all) |

- The external sites behave like the development sites: refitting gains nothing in either. The frozen shapes are
  already as good for the external site as for the sites they were learned on.
- The split-half reproducibility of the one-variable refits is near 0 (median r 0.10–0.22). The small changes the
  fits make are noise.
- The largest one-variable gains are under 0.001 nats: MIMIC-IV `map_last` −0.0008, eICU `age` −0.0008, INSPIRE
  `hr_slope_6h` −0.0004.
- eICU's "missing only" is exactly 0 by construction. The inputs eICU misses (lactate) have zero weight in its
  frozen metric, and a zero-weight point cannot move (prereg §9).

### 7.2 The shapes learned in each site (S5)

Each site's own shapes, learned on its fold-0 training patients alone, compared with the frozen shapes that served
it. Weighted correlation and slope of the site's log-odds curve on the transported one:

| | INSPIRE r / slope | MIMIC-IV r / slope | eICU r / slope |
|---|---|---|---|
| `map_last` | 0.99 / 1.29 | 0.99 / 0.58 | 0.96 / 1.23 |
| `map_mean_6h` | 0.99 / 0.93 | 1.00 / 1.91 | 1.00 / 0.90 |
| `map_min_6h` | 0.95 / 1.07 | 0.93 / 0.40 | 0.96 / 1.35 |
| `map_sd_6h` | 0.91 / 2.06 | 0.97 / 0.92 | 0.97 / 1.07 |
| `hb_last` | 0.84 / 1.08 | 0.95 / 0.94 | 0.98 / 0.96 |
| `hr_last` | 0.69 / 1.19 | −0.02 / −0.00 | 0.57 / 0.93 |
| `temp_last` | 0.36 / 0.93 | 0.94 / 0.95 | 0.21 / 0.29 |

- **The MAP shapes have the same form everywhere** (r 0.93–1.00). What changes is how much of the risk each summary
  carries. MIMIC-IV's own models give the 6-h mean almost twice the transported amplitude (slope 1.91), and the last
  and the minimum about half (0.58, 0.40). The three are strongly correlated, so the models can trade credit among
  them, and different training data split it differently.
- **Heart rate's shape differs in every site** (r 0.69, −0.02, 0.57), and temperature's in INSPIRE and eICU (r 0.36,
  0.21). Both carry ≤ 2 % of the weight in every transported metric, and ≤ 3 % in any locally learned fold. The
  method can see these differences, and they do not matter for the prediction.

MAP threshold: the highest MAP below 85 mmHg where the odds of hypotension double (median and 10–90 % over the 40
models).

| | site's own models | transported metric |
|---|---|---|
| INSPIRE `map_last` | 73.1 (71.2–74.6) | 69.9 (68.3–70.6) |
| INSPIRE `map_mean_6h` | 71.8 (67.4–74.6) | 72.2 (69.5–76.8) |
| INSPIRE `map_min_6h` | 69.8 (65.7–72.8) | 64.4 (60.9–68.4) |
| MIMIC-IV `map_last` | 67.3 (66.2–68.1) | 72.9 (70.8–73.8) |
| MIMIC-IV `map_mean_6h` | 75.5 (73.4–78.6) | 70.3 (68.5–74.6) |
| MIMIC-IV `map_min_6h` | 60.4 (55.0–63.4); only 15 of 40 models reach a doubling | 68.0 (62.1–70.9) |
| eICU `map_last` | 70.9 (68.2–71.5) | 70.4 (69.2–71.2) |
| eICU `map_mean_6h` | 73.1 (70.6–77.9) | 73.7 (71.1–76.8) |
| eICU `map_min_6h` | 66.9 (63.0–68.8) | 62.0 (59.4–64.6) |

- **eICU's thresholds agree with the transported ones for the last and mean MAP.** eICU is the largest site.
- **INSPIRE's thresholds for the last and the minimum MAP are 3–5 mmHg higher** than the transported ones. It is
  the site with hourly non-invasive readings. eICU's minimum is also 5 mmHg higher.
- **MIMIC-IV's thresholds move in opposite directions**: last and minimum lower, mean higher. That is the credit
  trade above, not a different physiology.
- **Per-variable thresholds should therefore be read as a set.** The MAP block as a whole carries 55–73 % of the
  weight in every metric, local or transported.

### 7.3 Hospitals (S5, S6)

- **Thresholds across the 36 eICU hospitals:**
  - `map_last` median 70.6 mmHg, interquartile range 69.3–71.8, range 67.9–76.9;
  - the spread between hospitals (SD 2.1 mmHg) is the same size as the spread among the 40 models inside one
    hospital (median 10–90 % width 3.4 mmHg);
  - `map_mean_6h` and `map_min_6h` are noisier (SD 2.1 and 4.3; within-hospital widths 9.0 and 10.1).
- **Hospital shapes have the transported form** (median r 0.92 for `map_last`, 0.99 for `map_mean_6h`, 0.97 for
  `map_min_6h`), with the same trading of amplitude among the three summaries.
- **Re-deforming the shapes with each hospital's own case base gains nothing on average** (mean Δ +0.0001). 3 of 36
  hospitals gain (CI < 0) and 1 loses (CI > 0), about what chance gives with 36 tests.

### 7.4 The weights learned locally (descriptive, not pre-registered as a comparison)

The H1 arm learns a whole metric on each site's own training folds, once per fold. Its weights differ from the
transported ones well beyond the fold-to-fold range:

| site | variable | local, mean (range over 5 folds) | transported |
|---|---|---|---|
| MIMIC-IV | L (hours since ICU admission) | 0.10 (0.02–0.16) | 0.28 |
| MIMIC-IV | `map_mean_6h` | 0.37 (0.30–0.50) | 0.15 |
| MIMIC-IV | `map_min_6h` | 0.05 (0.00–0.09) | 0.21 |
| MIMIC-IV | `age` | 0.06 (0.05–0.08) | 0.01 |
| MIMIC-IV | `vaso_on` | 0.04 (0.01–0.13) | 0.00 |
| eICU | `map_min_6h` | 0.30 (0.19–0.41) | 0.08 |
| eICU | `map_mean_6h` | 0.21 (0.14–0.32) | 0.39 |
| INSPIRE | `age` | 0.02 (0.01–0.04) | 0.11 |

- The weights differ this much and yet change discrimination by ≤ 0.003 AUROC (H1) and log-loss by ≤ 0.003 nats.
  The kNN is insensitive to how credit is shared among correlated inputs. A site's weights are therefore not, by
  themselves, a measure of how that site differs.
- **Two differences are not a trade among MAP summaries:**
  - The weight of time since ICU admission: low in MIMIC-IV's own metric, high in the one learned on INSPIRE + eICU.
  - Vasopressors running now: weighted in MIMIC-IV's own metric, where 22.6 % of the landmarks are on vasopressors.
    In the transported metric it is near zero. That metric was learned on INSPIRE, which cannot tell whether an
    infusion is running, and eICU, where 7.9 % of the landmarks are on vasopressors.
- These two are the candidate site differences for a follow-up. They are not tested here; §13.1 tests them post hoc.

### 7.5 Reading

For this outcome, PERISIM's drift tools agree:
- a metric learned elsewhere needs no deformation to serve a new site;
- the null is measured with the same precision in the development sites (CIs of ±0.002 nats);
- the differences that exist live in the base rate and case mix (absorbed by the local case base), in the split of
  credit among correlated MAP summaries, and in low-weight inputs.

This contrasts with T2, where drift sat in case mix, coding and the meaning of missing values, and where
re-deformation explained the transport loss. Here there is almost no transport loss to explain: the inputs are dense
physiology measured the same way everywhere.

## 8. Measurement and outcome definition (S7)

- **Arterial line** (MIMIC-IV and eICU): transported and local PERISIM are within 0.004 nats of each other in both
  strata.
  - Discrimination is higher without an arterial line: AUROC 0.80–0.82 against 0.77–0.78.
  - Neighbours share the query's stratum 80 % / 68 % of the time with a line (MIMIC-IV / eICU), and 50–52 % without
    one.
- **One reading per hour** instead of the hourly median: the outcome becomes more frequent (15–32 %). The same
  neighbours vote it with the same AUROC (0.78 / 0.78 / 0.80) and calibration in the large 0.02–0.03.
- **Three MAP bands**: transported equals local in INSPIRE and eICU (Δ 0.0000, 0.0002) and is 0.003 nats worse in
  MIMIC-IV, as for the binary outcome.

## 9. Explainability (S8)

- **What a neighbour differs in.** The query–neighbour distance is spread over many inputs: no input takes more than
  11 % of it. Age takes the most in every site (9–11 %). The MAP summaries, haemoglobin and L take 7–9 % each.
- **What puts a query nearer its event neighbours.** The pull is the squared distance in one input to the non-event
  neighbours minus that to the event neighbours. It is small for every input, because within a neighbourhood the
  event and non-event neighbours are about equally far. The largest pulls are the 6-h minimum MAP and age in
  INSPIRE, sex and creatinine in MIMIC-IV, and sex and age in eICU (0.001–0.002 in squared-distance units).
- **What the neighbours carry beyond the label**, read from what happened to them:
  - vasopressor start in the next 6 h: AUROC 0.68 (INSPIRE), 0.67 (MIMIC-IV), 0.79 (eICU);
  - ICU discharge within 24 h: AUROC 0.73 / 0.74 / 0.65;
  - the lowest MAP of the next 6 h: Spearman 0.63 / 0.64 / 0.70.

  The neighbourhood is a usable forecast of the course, not only of the event.
- **Out of support**: 1.0–1.1 % of the external queries, the same as with the local metric (Δ ≤ 0.001). The
  external sites sit inside the case bases' support.

## 10. What T2 predicted, and what happened

| T2 finding | Set here | Result |
|---|---|---|
| The rare class fails (~0.5 events per neighbourhood) | a frequent outcome, 13–31 % | calibrated in every site (H2) |
| The case base needs thousands of patients | 12,000–32,000 patients per site | 250–500 patients suffice (S1) |
| Relations are hospital-specific (−0.041) | H4 | real but ten times smaller (−0.0043) |
| A two-site metric transports like a local one | H1, every site external once | yes, in all three (ΔAUROC ≥ −0.003) |
| Drift sits in case mix, coding and missing values | missing-only re-deformation pre-registered | missing values move nothing (Δ ≤ 0.0003) |
| Re-deformation explained the transport loss | re-deformation pre-registered with in-sample controls | no loss to explain; the controls look the same |

## 11. Deviations and limitations

**Deviations**:
- None after the freeze. The external stage ran once (2026-10-29 12:07–19:18), and the freeze gate passed at its
  start and its end.
- Before the freeze, the dry run found that the LOO could choose α = 0 in small case bases, which gives predictions
  of exactly 0 or 1. The grid lost its 0 (prereg §3, §5; DECISIONS 2026-10-29).
- Eligibility for H4 counts patients after the landmark exclusions: 36 hospitals, against the design's 37 (prereg §6).

**Limitations**:
- **Declared in advance (prereg §9):**
  - the outcome is close to an input, which bounds discrimination;
  - INSPIRE measures differently;
  - the rotations are three different questions, not replicates;
  - drift and shapes use fold 0 only, so their CIs leave out the variation between splits;
  - re-deformation cannot move zero-weight points.
- **Arising from the results:**
  - Per-variable MAP thresholds and weights are not identifiable one by one, because credit moves among the
    correlated MAP summaries (§7.2, §7.4). A threshold for the MAP block as a whole would be the robust quantity; it
    was not pre-registered.
  - The weight comparison in §7.4 is descriptive and was not planned as a test.
  - H5's formal success is not a practical one.

## 12. Conclusions

1. For postoperative hypotension, a PERISIM metric learned on two cohorts serves a third. Discrimination is level
   with a locally learned metric, and calibration is good without refitting anything, because the outcomes come
   from the local case base. A few hundred local patients are enough.
2. As a drift tool, PERISIM gives a clear and checkable answer here: the shapes do not need to deform. The CIs bound
   any gain at about 0.002 nats, and the development-site controls behave the same way.
3. What differs between sites and hospitals (base rates, time-since-admission weighting, vasopressor use) is visible
   in the metric's pieces. It is absorbed by the case base rather than requiring a new metric.
4. Three follow-ups were run post hoc, outside the frozen code (§13). They support points 2 and 3:
   - the drift null holds over all five folds;
   - a MAP-block threshold of about 78 mmHg agrees across sites (eICU hospitals: interquartile range 77–79);
   - of the two weight differences named in §7.4, only MIMIC-IV's vasopressor weight gains anything (−0.0017 nats).
     That gain is exploratory.

## 13. Post hoc follow-ups (exploratory, not pre-registered)

The lead asked for these follow-ups after the first draft of this report. They ran on 2026-10-29/30 with
`scripts/posthoc_hypo.py`, outside the frozen code. The frozen artefacts are only read, and the freeze record was
checked at the start and at the end. Nothing here changes H1–H5 or S1–S8. Full tables:
[reports/posthoc.md](reports/posthoc.md).

### 13.1 The weight differences of §7.4 (fold 0)

The weights of the H1 arm's local fold-0 metric are swapped into the transported metric. The transported shapes and
the site's own case base are kept. Δ log-loss is measured on S4's test sample (negative = better than frozen):

| site | the H1 local metric itself | all weights local, shapes transported |
|---|---|---|
| INSPIRE | −0.0016 (−0.0048 to 0.0018) | −0.0019 (−0.0048 to 0.0008) |
| MIMIC-IV | −0.0022 (−0.0051 to 0.0009) | −0.0001 (−0.0026 to 0.0029) |
| eICU | −0.0012 (−0.0036 to 0.0009) | −0.0008 (−0.0028 to 0.0012) |

- **Swapping all the weights at once gains nothing**, and neither does the whole local metric. As a whole, the weight
  differences of §7.4 are a sharing of credit that the kNN does not feel.
- **One variable at a time** (25 swaps, every variable whose weight differs by ≥ 0.02):
  - **One swap has a CI below 0: MIMIC-IV's `vaso_on`** (weight 0.000 → 0.133), −0.0017 nats (−0.0031 to −0.0002).
    It is one of the two candidates §7.4 named before this analysis. It is small (0.3 % of MIMIC-IV's log-loss of
    0.503), and it is one of 25 tests. It is a lead, not a finding.
  - The other candidate gains nothing: lowering the weight of time since admission in MIMIC-IV (L 0.281 → 0.133)
    gives −0.0007 (−0.0016 to 0.0004).
  - Two swaps have a CI above 0: INSPIRE `map_mean_6h` (+0.0010) and eICU `map_min_6h` (+0.0013). Each moves one
    MAP summary alone, which breaks the split of credit among the three (§7.2).
- **Re-deforming the shapes after a swap gains nothing further.** The swap moved some weights a lot, or started from
  a (near-)zero weight that S4 cannot move. Every "fitted − start" CI includes 0.

### 13.2 A threshold for the MAP block (fold 0)

The per-variable thresholds of §7.2 move between sites because the models split credit differently among the three
correlated MAP summaries. Adding the three log-odds curves along a steady MAP (last = 6-h mean = 6-h minimum)
removes that split. The block threshold is the highest steady MAP below 85 mmHg where the odds double relative to a
steady 85. Adding the curves is exact for the additive surrogate models and an approximation for the trees. Median
(10–90 %) over the 40 models:

| site | block threshold, own shapes | block threshold, transported | OR 70 vs 85, own / transported | per-variable `map_last`, own |
|---|---|---|---|---|
| INSPIRE | 79.1 (77.9–80.1) | 78.5 (76.7–79.7) | 10.2 / 6.9 | 73.1 |
| MIMIC-IV | 77.6 (76.8–79.4) | 79.2 (77.4–79.9) | 6.8 / 9.5 | 67.3 |
| eICU | 79.2 (78.5–80.0) | 78.1 (76.5–79.0) | 8.3 / 8.1 | 70.9 |

- **The block threshold is about 78 mmHg in every site.** It spans 1.6 mmHg across the three sites (77.6–79.2), and own
  and transported shapes differ by at most 1.6 mmHg in any site. The per-variable `map_last` thresholds of the same
  models span 5.8 mmHg (67.3–73.1).
- It sits above every per-variable threshold because a steady MAP lowers all three summaries at once.
- **Odds ratios** of a steady 70 mmHg against 85 are 6.8–10.2 (own) and 6.9–9.5 (transported). At 65 against 85 they
  are 17.6–29.1 and 18.0–24.2. The ranges over the models are wide (INSPIRE own, OR 70 vs 85: 4.4–14.8).
- **eICU hospitals (36)**: block threshold median 77.8 mmHg, interquartile range 77.1–79.0, range 74.1–81.9.
  - The spread between hospitals (SD 1.6 mmHg) is smaller than for `map_last` (2.1).
  - The spread among the models inside a hospital is the same for both (10–90 % width 3.5 and 3.4 mmHg).
- The refitted site shapes reproduce S5's nine per-variable thresholds exactly (difference 0.00 mmHg).

### 13.3 Drift over the five folds (external role)

S4's re-deformation is repeated on folds 1–4, with each fold's case base and test sample. Gain from re-deforming
every g_j (negative = better than frozen):

| site | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean (SD) |
|---|---|---|---|---|---|---|
| INSPIRE | −0.0004 | −0.0010 | −0.0003 | +0.0014 | −0.0011 | −0.0003 (0.0010) |
| MIMIC-IV | −0.0004 | −0.0024 | −0.0013 | +0.0020 | −0.0001 | −0.0004 (0.0017) |
| eICU | +0.0007 | +0.0013 | +0.0015 | +0.0020 | +0.0017 | +0.0015 (0.0005) |

- **One of the 15 refits has a CI below 0**: MIMIC-IV fold 1 (−0.0042 to −0.0007). MIMIC-IV's mean over the folds is
  −0.0004.
- **In eICU every fold is slightly worse after refitting**, and every CI includes 0. The refit adds a little variance
  and finds no signal.
- **"Missing only"** stays within ±0.0004 in every fold.
- **One-variable refits**: 8 of 425 CIs are below 0 and 10 above 0, about the 11 per side that chance gives. No
  variable has a CI below 0 in two folds. Median split-half r is 0.17–0.21, close to fold 0's 0.10–0.22.
- **Fold 0's null (§7.1) is therefore not a property of one fold.**

### 13.4 Reading

- **The drift null of §7 holds over all five folds.**
- **The one site difference with a measurable gain is a weight, not a shape.** MIMIC-IV has 22.6 % of its landmarks
  on vasopressors. It gains a little from weighting a running infusion, which the transported metric leaves out
  because it was learned where infusions are rare (eICU) or cannot be seen (INSPIRE). This is the kind of drift
  PERISIM is meant to show. Confirming it would need a pre-registered test.
- **MAP thresholds are better reported for the block than per variable.** The block threshold is about 78 mmHg in
  every site, own or transported. Across eICU's hospitals the interquartile range is 77–79 mmHg.

## Reproducibility

- **Code and configuration**: tag `perisim-freeze-hypo-2026-10-29-120652`, commit `b0c9a02`, code `cf0d0b80c70a`,
  configuration `772940e4ed77`.
- **Commands**, from the frozen checkout: `python -m hypo landmarks`, `audit`, `cv`, `final`,
  `freeze --registration <ID>`, `external`, `external-report`.
- **Post hoc** (§13): `python scripts/posthoc_hypo.py` (or `run_local.ps1 hypo-posthoc`), outside the frozen code.
  It reads the frozen artefacts and checks the freeze record at the start and at the end.
- **Compute**: one workstation, 16 threads. The external stage took 7 h 11 min, and the post hoc follow-ups 4 h 58 min.
- **Data**: patient-level data stay on that computer under the PhysioNet data use agreements.
  The per-unit summaries are in [results/](results/), and the frozen hashes in
  [results/freeze.json](results/freeze.json).
