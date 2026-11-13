# Pre-registration — a similarity metric learned without patient rows: distillation from a frozen model into a synthetic cohort built from population aggregates (draft)

**Status.** Draft of 2026-11-13. The lead and the analyst wrote it **before any aggregate of the real sites has been
computed and before any distilled metric has been fitted on any real-derived data**. It is meant to be uploaded to OSF as written. The OSF identifier is then recorded in every freeze of this
study (`--registration <OSF ID>`).

**Registered on OSF:** _pending_ (to be filled in, together with the commit of the branch at registration).

**Author:** Ramsés Marrero García (Hospital Clínic de Barcelona; SIMTRI, Seoul National University Hospital).

**Repository:** https://github.com/ramsestein/perisim, branch `distill` (this file: `docs/prereg_distill.md`).

## 1. Question

In PERISIM, a similarity metric is learned at source sites, frozen, and used at a target site that votes among its
own patients (its case base). Across four ICU systems, that metric keeps most or all of the gain of a metric learned
at the target ([results/hypo/REPORT.md](../results/hypo/REPORT.md); NWICU, OSF 10.17605/OSF.IO/P5CA3). The metric
travels as an artefact of per-variable weights and shape tables. Those are aggregates, but they are learned from the
source sites' patient rows.

The question here is **whether a metric learned without touching any patient row keeps that performance.** The
metric is learned only from:

1. a frozen predictive model trained at the source (the *teacher*), and
2. a synthetic cohort generated from **population-level aggregates** of the source. These are marginal quantile
   tables and a rank-correlation matrix, each computed on at least 50 patients.

The teacher labels the synthetic cohort, and PERISIM's metric pipeline runs on that labelled synthetic cohort.
Nothing else changes: the metric is frozen and used at the target with the target's real case base, exactly as in
`hypo`.

If the distilled metric performs like the real one, a hospital can share a PERISIM metric whose construction never
needed patient rows. Only aggregates and a model leave the source, and the target's case base never leaves the
target.

## 2. What is known before this study

- **All results of `hypo`, NWICU and the arc-metric re-run are known to the analysts.** Those runs are documented in
  `results/hypo/`, `results/nwicu/` and `results/arc-metric/`. The most relevant facts are:
  1. What travels is mainly the weights. The per-variable shapes add little.
  2. Fixed weights from marginal mutual information come within 0.004–0.005 nats of the jointly learned NCA weights
     (REPORT §17).
  3. A transported LightGBM (the teacher's model class), recalibrated, predicts better than the transported kNN.
     Without recalibration it is miscalibrated at the target.
- **Not done before this registration:** no aggregate has been computed on these sites, no synthetic cohort has been
  generated from them, and no metric has been fitted on synthetic data. The generator and the distillation code
  (`src/distill/`, `scripts/distill_run.py`) are written and tested on the project's synthetic test data only
  (`tests/test_distill.py`). The code refuses to compute aggregates of the real sites unless the OSF ID is written in
  this file.
- **The same patients are reused.** This study uses the same source and target data as `hypo` and NWICU. It is a
  comparison of how the metric is learned, on known data. It is not an independent replication of transport.

## 3. Design

### 3.1 Units

These are the six primary units of `docs/prereg_arc.md`, with the same rows, folds, case bases, Euclidean comparator
and local metric:

- **`hypo`, three rotations.** For each target X ∈ {INSPIRE, MIMIC-IV, eICU}, the source is the two other sites.
  The target is evaluated on its external-stage test landmarks, over five patient folds, voting among its own
  training folds.
- **NWICU, three rotations.** The metric learned on each pair of sites is applied to NWICU, with NWICU's
  pre-registered folds and case base.

**Code and placement.** The code is v0.6 with the published placement of the shape functions (`shape_embedding:
value`; arm `value` of `prereg_arc.md`). The reference numbers for the real-data metric are those of arm `value`
(`work_value`, commit `05ecffd`). They are not re-run, apart from the check in §3.5.

### 3.2 Teacher

For each rotation, the teacher is the transported LightGBM comparator that arm `value` already froze
(`final/comp_ext-<target>_lightgbm.pkl`, fitted by `hypo.frozen.final` on all landmarks of the two source sites
pooled, with `hypo`'s inputs, model space, inner-fold tuning and seed). It is not re-fitted. Its predicted
probability p_T(x) labels the synthetic cohort.

### 3.3 Synthetic cohorts (the only thing the source contributes beyond the teacher)

**Rows the aggregates are computed on.** For each rotation, the real learning sample of the real-data metric
M_REAL (the rows `hypo.frozen.final` drew: at most 3 landmarks per patient, the same number per source site, same
seed). These aggregates are the only quantities taken from patient rows. Each source site's aggregates are computed
on that site's rows alone.

**Marginals.** Per input and site:

- **continuous** (more than 30 distinct observed values): the observed values clipped to their 1st and 99th
  percentiles, then their quantiles at 1/200 steps (201 values);
- **discrete** (numeric with at most 30 distinct values: the binary inputs and the landmark time L) and **nominal**
  (surgical group): the share of each value. Values held by fewer than 50 patients are pooled: they share one
  released mass, spread evenly over them when sampling (the value labels are the variable's vocabulary, not data);
- the **missingness share**. An input that a site never measures (INSPIRE's source-missing inputs) stays entirely
  missing.

Every released cell rests on at least 50 patients; the code stops otherwise.

**Joint structure (Gaussian copula).** The copula has one *value* dimension per input with observed values, and one
*missingness* dimension per input that is missing in some but not all rows:

- a value dimension is the normal score of the observed value under its own observed marginal (ties, discrete values
  and categories spread uniformly inside their segment, categories ordered by frequency); it is undefined where the
  input is missing;
- a missingness dimension places a missing row uniformly in [0, m) and an observed row in [m, 1), m being the
  missingness share, then takes its normal score.

The correlation matrix is the Pearson correlation of these normal scores on pairwise-complete rows (a pair with
fewer than 50 complete rows gets 0), made positive definite by eigenvalue clipping at 1e-6 and rescaled to a unit
diagonal. It is therefore a rank (Spearman-type) correlation, not a polychoric one.

**Generation.** Two synthetic cohorts are generated per rotation. Each source site contributes as many synthetic rows
as it has in the real learning sample (so the cohort has the learning sample's size and site mix):

- **SYN-COP (primary).** A draw from the site's Gaussian copula. Each value dimension is mapped back through its
  marginal (linear interpolation between the 201 quantiles; the category or discrete value of its segment); the
  input is set missing where its missingness dimension falls below m.
- **SYN-IND (control).** The same, with every dimension drawn independently (no joint structure; missingness
  independent of everything).

Each synthetic row is its own synthetic patient. No row of the source is copied or perturbed, and no synthetic value
lies outside the clipped 1st–99th percentile range of its input.

### 3.4 Labels and metric

- **Labels.** Each synthetic row gets y ~ Bernoulli(p_T(x)), drawn with a fixed seed per rotation and cohort.
  PERISIM's pipeline needs binary labels, and drawing them keeps the teacher's uncertainty.
- **Metric.** PERISIM's metric pipeline (`hypo.metric.fit_metric`) runs unchanged on the labelled synthetic cohort:
  - the encoder and bins;
  - the consensus shape loop, with 40 models and the flatness test;
  - the diagonal NCA, with the same settings and seed;
  - patients defined as synthetic rows.

  The result is frozen as the metric M_COP (or M_IND).
- **Use.** The frozen M_COP or M_IND is used at the target exactly as the real-data metric M_REAL of arm `value`:
  same case base, same α selection by leave-one-patient-out, same k and the same vote.

### 3.5 Integrity checks (run before any target is evaluated)

1. **Re-fit of M_REAL.** The real-data metric is re-fitted on one rotation (target eICU) with the distillation code
   path (the real learning sample through the same `fit_metric` call). It must reproduce arm `value`'s weights to
   1e-6.
2. **Case-base reproduction.** The distillation code's case-base prediction, run with arm `value`'s frozen metric,
   must reproduce arm `value`'s published transported predictions to 1e-9 for `hypo` eICU fold 0 and for NWICU
   (rotation eICU) fold 0.

   If check 1 or 2 fails, the run stops before freezing; the difference is investigated and logged.
3. **Copying (descriptive).** For each synthetic cohort, the distance from each synthetic row (up to 20,000, drawn at
   random) to its closest real learning row (distance to closest record, DCR). The space is the numeric inputs
   standardised by the real rows' median and IQR, a missing value at the centre plus a missingness indicator. It is
   reported against the same distance from one half of the real patients to the other half. Only the 1st and 5th
   percentiles, the median and the share of exact copies (distance < 1e-9) are reported.
4. **Fidelity (descriptive), per rotation:**
   - the teacher's AUROC on the real learning rows (in sample: the teacher was fitted on all source landmarks) and
     at the target;
   - the teacher's mean p_T on each synthetic cohort against the real learning sample's prevalence;
   - per site, the marginal errors of each cohort (mean and maximum error of the 5–95 % quantiles in units of the
     real IQR, maximum missingness-share error, maximum total-variation error of the category shares) and the mean
     absolute error of its copula correlations against the real ones.

## 4. Hypotheses and pre-specified readings

The patient-level bootstrap uses 2,000 resamples per unit, with one set of replicates for every statistic of a unit.
Log-loss is the primary metric. The paired contrast is computed with `scripts/arc_compare.py`'s statistics (same
seed, same replicates per unit in every contrast):

- the "value" arm is M_REAL (arm `value`'s per-row log-losses, `work_value/arc_compare/`);
- the "arc" arm is M_COP (or M_IND);
- the Euclidean kNN and the local metric are arm `value`'s, common to all;
- every unit is checked to evaluate identical rows, patients and labels.

H2 uses the same statistics with M_COP as "value" and M_IND as "arc" (its D is D_IND − D_COP).

### 4.1 Primary (H1): the copula-distilled metric is non-inferior to the real-data metric

For each of the six units:

- **Q_COP = (L_euclid − L_COP) / (L_euclid − L_REAL).** This is the share of the real metric's gain over the
  Euclidean kNN that the distilled metric keeps. It is **non-inferior** when the lower limit of its 90 % CI is
  ≥ 0.80. It is **not evaluable** when the real metric's gain has a point estimate below 0.005 nats or a 95 % CI
  including 0.
- **D_COP = L_COP − L_REAL** in nats, with its 95 % CI, is reported with every unit.

**H1 holds** when Q_COP is non-inferior in every evaluable unit (intersection–union test, no multiplicity
adjustment). It also holds if at most one of the six units is not evaluable. Each unit's reading is also reported on
its own.

### 4.2 Secondary

- **H2, joint structure matters.** D_IND − D_COP > 0, with a 95 % CI above 0, in at least four of the six units. This
  means the metric distilled from independent marginals is worse than the copula one. Q_IND is reported with the
  rule of §4.1.
- **H3, transport of the distilled metric.** R_COP = (L_euclid − L_COP) / (L_euclid − L_local), where L_local is the
  metric learned at the target as in `hypo`. It is read by the published rule: equivalent when the lower 90 % limit
  is ≥ 0.80, with the not-evaluable rule of §4.1.
- **Descriptive:**
  - calibration of the distilled metric's predictions (CITL, slope) at each target;
  - Spearman correlation between the weights of M_COP and M_REAL, and between M_IND and M_REAL;
  - each variable's weight share in the three metrics;
  - AUROC alongside log-loss.

### 4.3 Expectations, declared now

- **SYN-COP.** We expect Q_COP between 0.8 and 1.0 in most units, with H1 likely to hold. The basis is that weights
  carry the metric's gain and marginal weights already come close to it.
- **NWICU.** Its local reference is data-limited (R > 1 in NWICU with the real metric), so we expect the distilled
  metric's R to stay high there.
- **SYN-IND.** We expect it to lose more, because PERISIM's heaviest inputs (MAP last, MAP mean, MAP minimum, MAP
  SD) are strongly correlated, and independent draws put the teacher in regions that do not exist. If SYN-IND
  performs as well as SYN-COP, the joint structure is not needed. That result would make the shared artefact even
  smaller.

## 5. Rules

- §3–§4 do not change after the first synthetic cohort is generated from real aggregates.
- Mechanical deviations (a crash, a library version) are fixed without changing the method and logged with a date in
  `docs/DECISIONS.md`.
- Each rotation's teacher, aggregates, cohorts and metrics are frozen with a freeze record carrying this
  registration's OSF ID before any target is evaluated.
- Outputs are aggregates only. No synthetic or real patient row is published. Cells of n < 10 are not reported. The
  synthetic cohorts are kept only inside the work folder, like the real data, because they derive from credentialed
  data whose licences forbid redistribution of derived row-level data.
- All units are reported in full, whatever the result.

## 6. Limitations, declared in advance

- **The teacher is learned from patient rows.** Distillation removes patient rows from the metric's construction,
  not from the teacher's. The claim is about the shared artefact (aggregates plus a model), not about the source
  never touching patient data.
- **The aggregates are not formally private.** The quantile tables and the correlation matrix are computed on at
  least 50 patients, with clipped tails. That is an aggregation rule, not a differential-privacy guarantee. A formal
  guarantee (noise on the aggregates) is outside this study.
- **The Gaussian copula is simple.** It keeps pairwise rank correlations, not higher-order structure, and treats
  nominal inputs by an ordering of their categories. A better generator could do better. This study tests whether a
  simple, aggregate-only generator is enough.
- **The same data as `hypo`.** The comparison is between two ways of learning the metric on known data.
- **One teacher class.** Other teachers (logistic regression, a neural network) are not tested.
- **One outcome and one setting** (hypotension in 6 h, postoperative ICU).

## 7. Registration

The study is registered on OSF before the first synthetic cohort is generated from real aggregates. The OSF ID goes
into every freeze. The commit of branch `distill` at registration, pushed to GitHub, is the verifiable timestamp of
the code and this plan.
