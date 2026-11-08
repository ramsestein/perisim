# Pre-registration: why does a learned metric beat cosine in PERISIM but not in PRAM?

**Registered on OSF:** pending (to be filled in after registration).
**Author:** Ramsés Marrero García (Hospital Clínic de Barcelona).
**Repository:** https://github.com/ramsestein/perisim (this file: `docs/prereg_pram_factors.md`).

## 1. Status and prior knowledge of the data

This plan is fixed before any of the analyses below are run. What was known when it was written:
- **PRAM v2** (Jeong et al., 2026, medRxiv, posted 19 August 2026), its methods and its supplementary tables. In its
  Table S13 (eICU-CRD, full bank, change in AUROC over the frozen logistic model), cosine distance is better than a
  neighbourhood components analysis (NCA) metric, whether transferred from development or fitted on the target bank:
  - acute kidney injury: cosine +0.0197, NCA transferred +0.0105, NCA on the target bank +0.0131;
  - 168 h mortality: cosine +0.0358, NCA transferred +0.0268, NCA on the target bank +0.0262.

  A mutual-information-weighted Euclidean distance was slightly better than cosine (+0.0216, +0.0367).
- **PERISIM's own results** on its landmark data (INSPIRE, MIMIC-IV, eICU, NWICU; hypotension in the next 6 h):
  - the transported diagonal NCA metric beats a cosine kNN at all four targets, standalone, with PERISIM's vote
    (REPORT §21, pre-specified, 2026-11-08: Δ AUROC +0.007 to +0.019);
  - mutual-information weights also beat cosine there (§17, §21).
- **Not yet computed:**
  - any mixture of a kNN with a frozen model;
  - PRAM's neighbour vote on PERISIM's data;
  - a full-matrix NCA on PERISIM's data;
  - the PRAM-like cohort (Tier 2): it has not been built. Its features, outcomes and model outputs have not been
    computed.

  MIMIC-IV and eICU-CRD have been used by PERISIM for a different cohort (postoperative ICU patients), different
  features (landmarks every 2 h) and a different outcome.

So Tier 1 re-uses data whose standalone result is known; the factors it adds (mixture, vote, full NCA) have not been
looked at. Tier 2 is new.

## 2. Question

PERISIM finds that a learned metric beats cosine; PRAM finds the opposite. **Which differences between the two
studies explain the reversal?** Four candidate factors, each with a PERISIM level and a PRAM level:

| factor | PERISIM level | PRAM level |
|---|---|---|
| **U, use of the neighbours** | standalone kNN prediction | mixture with a frozen model, p = (1 − a)·p_base + a·p_kNN, a ∈ [0, 0.5] |
| **V, vote** | smoothed vote of k distinct patients, α chosen by leave-one-patient-out | weighted mean of the k neighbours' outcomes, weight 1/(d + 1e-6) |
| **I, NCA implementation** | diagonal NCA (one weight per variable; PERISIM's objective and settings) | full-matrix NCA (a linear map; scikit-learn's `NeighborhoodComponentsAnalysis`) |
| **S, setting** | landmarks every 2 h, postoperative ICU, hypotension in 6 h | 42 static first-24 h aggregates, adult ICU, acute kidney injury and 168 h mortality |

PRAM's code is not public ("available from the corresponding author on reasonable request"). The PRAM levels are
our implementation of its published description (§4.4). That PRAM used scikit-learn's full-matrix NCA is an
assumption: the paper names scikit-learn 1.3 and no other NCA software.

**Mechanism hypothesis behind U** (stated in advance; the design tests it, it is not assumed). In a mixture, the
neighbours only correct a base model. A learned metric selects neighbours by the signal the base model already
captures, so it may add less complementary information than a naive metric.

## 3. Design overview

- **Tier 1, PERISIM's setting.** The frozen `hypo` artefacts at INSPIRE, MIMIC-IV and eICU (external stage, five
  patient folds each) and at NWICU (pre-registered folds, each of the three frozen rotations). U, V and I are crossed.
- **Tier 2, PRAM's setting rebuilt.** A PRAM-like cohort from MIMIC-IV (development) and eICU-CRD (target), built
  from PRAM's published recipe. U, V and I are crossed, both outcomes. First, a **replication check** of Table S13.
- **S** is read by comparing the two tiers at matched levels of U, V and I.

Throughout, the **primary measure is AUROC** (PRAM's measure); log-loss is secondary.

## 4. Tier 2: the PRAM-like setting

### 4.1 Data and cohort
- **MIMIC-IV v3.1**, restricted to patients whose `anchor_year_group` is 2008–2010 to 2017–2019. This approximates
  PRAM's v2.2 (2008–2019); v3.1 is the version held locally.
- **eICU-CRD v2.0.**
- **Inclusion** (PRAM's):
  - adults (≥ 18 years at ICU admission);
  - the first ICU stay of each patient;
  - an ICU stay of at least 24 h.

  MIMIC-IV: the earliest `icustays.intime` per `subject_id`; age = `anchor_age` + (year of `intime` −
  `anchor_year`). eICU: per `uniquepid`, the stay of the earliest `hospitaldischargeyear`, then the lowest
  `patienthealthsystemstayid`, then the lowest `unitvisitnumber`; age "> 89" is read as 90; ICU stay length from
  `unitdischargeoffset` ≥ 1,440 min.
- **Added exclusion:** no serum creatinine in the first 24 h. It is needed for the baseline, and PRAM reports
  baseline creatinine 0 % missing, so it implies the same exclusion.
- **Anchor:** 24 h after ICU admission. Features use [admission, admission + 24 h]; outcomes use (24 h, 192 h].

### 4.2 Features (PRAM's 42)
- age; sex (one-hot, male); baseline creatinine (the minimum serum creatinine in the first 24 h);
- the minimum, maximum and last value in the first 24 h of 13 variables:
  - heart rate, respiratory rate, oxygen saturation, temperature (°C);
  - systolic and diastolic blood pressure;
  - Glasgow Coma Scale total, weight (kg);
  - blood urea nitrogen, calcium, INR, serum creatinine;
  - the BUN-to-creatinine ratio.

| variable | MIMIC-IV (itemid / table) | eICU-CRD (table, field) |
|---|---|---|
| heart rate | 220045 (chartevents) | vitalPeriodic.heartrate |
| respiratory rate | 220210, 224690 | vitalPeriodic.respiration |
| oxygen saturation | 220277 | vitalPeriodic.sao2 |
| temperature | 223762 (°C), 223761 (°F, converted) | vitalPeriodic.temperature |
| systolic BP | 220050 (arterial), 220179 (non-invasive) | vitalPeriodic.systemicsystolic, vitalAperiodic.noninvasivesystolic |
| diastolic BP | 220051, 220180 | vitalPeriodic.systemicdiastolic, vitalAperiodic.noninvasivediastolic |
| GCS total | sum of 220739 + 223900 + 223901 charted at the same time | nurseCharting, label "Glasgow coma score", name "GCS Total" |
| weight | 226512, 224639 (kg), 226531 (lb, converted) | patient.admissionweight |
| BUN | 51006 (labevents) | lab, labname "BUN" |
| calcium | 50893 | lab, "calcium" |
| INR | 51237 | lab, "PT - INR" |
| serum creatinine | 50912 | lab, "creatinine" |
| BUN : creatinine | BUN / creatinine of the same specimen (same `specimen_id`) | same `labresultoffset` |

**Plausibility ranges** (values outside become missing):

| variable | range |
|---|---|
| heart rate | 20–300 /min |
| respiratory rate | 2–80 /min |
| SpO2 | 50–100 % |
| temperature | 25–45 °C |
| systolic BP | 30–300 mmHg |
| diastolic BP | 10–200 mmHg |
| GCS | 3–15 |
| weight | 20–300 kg |
| BUN | 1–300 mg/dL |
| calcium | 2–20 mg/dL |
| INR | 0.5–20 |
| creatinine | 0.1–30 mg/dL |

The ranges are the same in both databases.

### 4.3 Outcomes (PRAM's definitions)
- **Acute kidney injury within 168 h of the anchor.** A serum creatinine in (24 h, 192 h] that is ≥ baseline +
  0.3 mg/dL or ≥ 1.5 × baseline, with the baseline being the first-24 h minimum.

  PRAM says "the rise in serum creatinine defined by the KDIGO criteria"; our reading does not apply KDIGO's 48 h
  window to the 0.3 mg/dL rise, because the baseline is a single fixed value.
- **In-hospital death within 168 h of the anchor.**
  - MIMIC-IV: `admissions.deathtime` in (24 h, 192 h] (if `deathtime` is missing and `hospital_expire_flag` = 1,
    `dischtime`).
  - eICU: `hospitaldischargestatus` = "Expired" with `hospitaldischargeoffset` in (1,440, 11,520] min.

### 4.4 PRAM's pipeline (our implementation of its Table S25)
- **Development split.** MIMIC-IV: 56 % training, 14 % validation, 30 % internal test, stratified by outcome, seed 42.
- **Preprocessing.** Fitted on the MIMIC-IV training split only and applied unchanged everywhere:
  - numeric features: multiple imputation by chained equations (scikit-learn `IterativeImputer`, defaults,
    `random_state` 42), then standardisation;
  - sex: one-hot.
- **Frozen base models**, fitted on the MIMIC-IV training split, per outcome:
  - logistic regression (L2, C = 1, at most 1,000 iterations; primary);
  - CatBoost (500 iterations, depth 6, other settings default; secondary).
- **Deployment simulation in eICU.** Five folds stratified by outcome (seed 42). In each fold, the test set is that
  fold and the bank is the other four (the full bank only; PRAM's S13 is at the full bank). Neighbours are k = 50,
  searched in the bank.
- **Mixing weight a.** Grid 0.00 to 0.50 in steps of 0.05, chosen by three-fold stratified cross-validation inside
  the bank:
  - in each inner fold, the inner-training two thirds serve as the bank for the inner-test third;
  - a maximises the AUROC of the pooled inner-test predictions;
  - ties go to the smaller a.

### 4.5 Metrics compared in Tier 2

All metrics work in the standardised 42-dimensional space (sex one-hot included):
- **COS:** cosine distance, d = 1 − cos.
- **MI:** Euclidean distance with weight w_j ∝ the mutual information of feature j with the outcome (scikit-learn
  `mutual_info_classif`, `random_state` 42, on the MIMIC-IV training split), Σ w = 1.
- **NCA-full:** scikit-learn `NeighborhoodComponentsAnalysis` (full linear map, `n_components` = all, defaults,
  `random_state` 42), fitted on a stratified random subsample of 5,000 training stays. scikit-learn's NCA is
  quadratic in the sample size; the subsample size is our choice.
- **NCA-diag:** PERISIM's diagonal NCA (`perisim.metric.nca.DiagonalNCA`) with `hypo`'s frozen settings:
  - 30 epochs, batch 256, learning rate 0.05, refresh every 5 epochs, 500 candidates, label smoothing 1e-3;
  - fitted on the whole training split;
  - input = standardised features divided by √2, so that each axis has unit mean-square random-pair distance, as in
    PERISIM.

Each NCA is fitted in two places: on the development training split and transferred (PRAM's M2; **primary**), and on
the target bank of each fold (PRAM's M3; secondary).

### 4.6 Votes and uses in Tier 2
- **V-inv (PRAM's):** weighted mean of the 50 neighbours' outcomes, weight 1/(d + 1e-6), with d the metric's
  distance (cosine distance for COS, Euclidean distance otherwise).
- **V-ours (PERISIM's):** p = (n₁ + α·π·2)/(50 + α·2). Here n₁ is the number of the 50 nearest neighbours with the
  outcome and π the bank's prevalence. α is chosen from `hypo`'s grid by leave-one-patient-out log-loss on a random
  subset of the bank (`hypo`'s procedure and settings).
- **U-alone:** the vote itself.
- **U-mix:** the PRAM mixture with the frozen logistic model (primary) or frozen CatBoost (secondary), with a chosen
  as in §4.4. The inner cross-validation uses the same vote as the arm.

## 5. Tier 1: PERISIM's setting

Targets: INSPIRE, MIMIC-IV and eICU, with the external stage's five patient folds, and NWICU, with the pre-registered
five folds, for each frozen rotation. Run with `hypo`'s frozen code (v0.5); the freeze record is checked at the start
and the end. The bank is each fold's case base; k = 50 distinct patients, as in `hypo`.

### Metrics
- **COS:** cosine on the Euclidean comparator's neutral axes, centred on the bank (REPORT §21).
- **MI:** mutual-information weights from the development sites on the neutral axes (§17's `mi_transported`).
- **NCA-diag:** the frozen transported metric (`metric_ext-<rotation>`).
- **NCA-full:** scikit-learn `NeighborhoodComponentsAnalysis` (full linear map, defaults, `random_state` 42).
  - Input: the frozen metric's unweighted embedding of the development sites' landmarks (that rotation's shapes and
    axes), on a random subsample of one landmark per patient, 5,000 patients, seed 42.
  - It is used at the target on the same embedding.

### Votes
- **V-ours:** `hypo`'s vote and α fit, unchanged.
- **V-inv:** 1/(d + 1e-6) over the same 50 neighbours (d = cosine distance for COS, the metric's Euclidean distance
  otherwise).

### Uses
- **U-alone.**
- **U-mix:** with the frozen transported logistic comparator `logreg` (primary; `comp_ext-<rotation>_logreg`, the
  analogue of PRAM's frozen logistic) or the frozen transported LightGBM (secondary, the analogue of frozen
  CatBoost).

  a is chosen as in §4.4: three patient-level folds inside the bank, seed = `hypo`'s seed + 100·fold + 7, the AUROC
  of the pooled inner predictions, ties to the smaller a. For INSPIRE, MIMIC-IV and eICU, the frozen base models'
  predictions must reproduce the stored external-stage comparator predictions (`comps` units) to 1e-9.

## 6. Estimand, primary contrasts and decision rules

**Estimand.** In each cell (setting, target, outcome, U, V, NCA variant):
- Δ = AUROC(NCA arm) − AUROC(COS arm), with the same U and V;
- 95 % CI from 2,000 bootstrap resamples (patients in Tier 1, stays in Tier 2), shared by all arms of a target and
  outcome.

**Verdict per cell:**
- "NCA beats cosine" if the lower limit is > 0;
- "cosine beats NCA" if the upper limit is < 0;
- otherwise "no difference shown".

**Factor effects.** For factor F, effect_F = Δ(F at the PERISIM level) − Δ(F at the PRAM level), with the other
factors held at the **home levels** of the tier, and a 95 % bootstrap CI:
- Tier 1 home levels: U-alone, V-ours, NCA-diag.
- Tier 2 home levels: U-mix with logistic, V-inv, NCA-full transferred.

For I, effect_I = AUROC(NCA-diag) − AUROC(NCA-full).

**R0, replication of S13 (Tier 2, home levels).** "Reproduced" if Δ(NCA-full transferred vs COS) < 0 as a point
estimate for both outcomes; "reproduced with confidence" if both upper limits are < 0. If R0 fails, Tier 2's
attributions are reported as descriptive, and the report says that S13's ordering was not reproduced with our
implementation.

**Primary hypotheses:**
- **H-U (use).** effect_U > 0 in Tier 1, at every target (each NWICU rotation counted as one), and in Tier 2, for both outcomes: the NCA advantage is
  larger standalone than in the mixture.
- **H-V (vote).** effect_V > 0 in both tiers.
- **H-I (implementation).** effect_I > 0 in both tiers: the diagonal NCA does better than the full-matrix NCA.
- **H-S (setting).** In Tier 2 at PERISIM's levels (U-alone, V-ours, NCA-diag transferred), the NCA advantage is
  not shown: the verdict is "no difference shown" or "cosine beats NCA" for both outcomes.

Each hypothesis is supported if its effect's lower 95 % limit is > 0 in every unit it names (for H-S: if the stated
verdicts hold), not supported if some upper limit is < 0, and inconclusive otherwise.

**Attribution rule:**
1. A factor **is sufficient to reverse** the ordering in a tier if switching it alone, from the home levels, changes
   the verdict on Δ from one non-null category to the other, or to "no difference shown".
2. The **main explanation** is the factor or factors whose switch is sufficient in both tiers, in opposite
   directions (Tier 1: PERISIM → PRAM level; Tier 2: PRAM → PERISIM level).
3. If no single factor is sufficient, the pairs of factors are examined the same way, and the answer is stated as a
   combination.
4. If H-S holds and no factor among U, V and I is sufficient in Tier 2, the explanation is the **setting** (the data
   representation, population and outcome together, not separated by this design).

## 7. Secondary and descriptive analyses
- **Every cell of both tiers:**
  - AUROC and log-loss of every arm;
  - calibration in the large and slope;
  - the chosen a in the mixtures, fold by fold;
  - the frozen base models' own AUROC.
- **The same contrasts for MI vs COS**, since in PRAM MI ≈ COS while in PERISIM MI > COS.
- **The NCA fitted on the target bank** (M3) in Tier 2, and the frozen CatBoost / LightGBM mixtures.
- **Fidelity of Tier 2 to PRAM** (descriptive, not criteria):
  - cohort sizes, prevalences and missingness against PRAM's Table 1 and Table S1;
  - frozen logistic AUROC in eICU against PRAM's 0.690 (acute kidney injury) and 0.774 (mortality);
  - COS-mixture gain against PRAM's +0.0197 and +0.0358.
- **Complementarity, the mechanism behind U:**
  - the correlation between the base model's prediction and each kNN's prediction on the test set;
  - the AUROC of each kNN for the base model's residual (y − p_base).
- **Sensitivity:** NCA-full fitted on 2,000 and 10,000 stays (Tier 2) or patients (Tier 1); a chosen by log-loss
  instead of AUROC.

## 8. Gates and integrity
- **Tier 1:**
  - the freeze record is checked at the start and the end;
  - the rebuilt Euclidean comparator must reproduce the stored predictions on fold 0 to 1e-9 (as in §21);
  - the frozen base models must reproduce the stored comparator predictions to 1e-9.
- **Tier 2:**
  - the cohort and features are built and checked (counts, missingness, distributions) before any outcome is
    joined to a model;
  - errors found in that check (a wrong item, a unit) are corrected and logged as deviations, with a reason, before
    any model is fitted.
- **Deviations** from this plan are dated in `docs/DECISIONS.md` and reported with the results.

## 9. Privacy

Aggregates only. No count between 1 and 9 is published or derivable. Patient-level data, cohorts, features and
predictions stay on the analyst's machine. Published files pass `scripts/check_public_safety.py`.

## 10. Limits declared in advance
- PRAM's code is not available. Tier 2 is a reconstruction from the paper; its fidelity is checked descriptively,
  not guaranteed. A failure to reproduce S13 limits what Tier 2 can attribute.
- MIMIC-IV v3.1 restricted by anchor year approximates v2.2. The eICU weight and GCS sources are our choices.
- The setting factor S bundles data representation, population and outcome. The design does not separate them.
- One target system in Tier 2 (eICU), as in S13; PRAM's MIMIC-III analyses are not repeated.
- Bootstrap CIs carry test-set sampling variability, not the variance of refitting the metrics or base models.

## 11. Implementation clarifications (fixed while writing the code, before any run)
1. **Complementarity.** An AUROC for the continuous residual y − p_base is undefined. §7's "AUROC of each kNN for the
   base model's residual" is computed as the Spearman correlation between p_kNN and y − p_base.
2. **NCA fitted on the target bank (M3) in the inner cross-validation for a.** The metric fitted on the whole bank is
   reused in the three inner folds, not refitted per inner fold. M3 is optimistic anyway, as in PRAM; it is
   secondary.
3. **Development split.** One split serves both outcomes, stratified jointly by acute kidney injury and death.
4. **eICU age.** Stays with a missing age are excluded, since adulthood cannot be verified.
5. **Seeds.**
   - The kNN α subset uses seed + fold; the inner folds use seed + fold + inner fold + 1.
   - Tier 2's seed is PRAM's 42; Tier 1's is `hypo`'s.
6. **Tier 2 cosine.** The cosine of the standardised vectors as they are, standardised on the MIMIC-IV training
   split as in PRAM, with no re-centring on the bank. Tier 1's cosine is §21's, centred on the bank.

## References

Goldberger, J., Hinton, G. E., Roweis, S. T., & Salakhutdinov, R. R. (2005). Neighbourhood components analysis. In
*Advances in Neural Information Processing Systems 17* (pp. 513–520). MIT Press.

Jeong, I., Lee, T., Kim, B., Park, J.-H., Kim, Y., & Lee, H. (2026). *PRAM: Post-hoc retrieval augmentation adapts
intensive care prediction models without retraining and can be shared across hospitals* (Version 2) [Preprint].
medRxiv. https://doi.org/10.64898/2026.04.03.26350132

Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T. J., Hao, S., Moody, B.,
Gow, B., Lehman, L. H., Celi, L. A., & Mark, R. G. (2023). MIMIC-IV, a freely accessible electronic health record
dataset. *Scientific Data, 10*, 1. https://doi.org/10.1038/s41597-022-01899-x

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,
Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011).
Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Pollard, T. J., Johnson, A. E. W., Raffa, J. D., Celi, L. A., Mark, R. G., & Badawi, O. (2018). The eICU
Collaborative Research Database, a freely available multi-center database for critical care research. *Scientific
Data, 5*, 180178. https://doi.org/10.1038/sdata.2018.178
