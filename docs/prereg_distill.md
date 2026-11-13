# Pre-registration — a similarity metric learned without patient rows: distillation from a frozen model into a synthetic cohort built from population aggregates (draft)

**Status.** Draft of 2026-11-13. The lead and the analyst wrote it **before any distilled metric has been fitted on
any data**. It is meant to be uploaded to OSF as written. The OSF identifier is then recorded in every freeze of this
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
- **Not done before this registration:** no synthetic cohort has been generated from these sites, and no metric has
  been fitted on synthetic data. The generator and the distillation code are written and tested on the project's
  synthetic test data only (`tests/`).
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

For each rotation, the teacher is a LightGBM fitted on the two source sites pooled. It uses the inputs, model space,
inner-fold tuning and seed of `hypo`'s transported LightGBM comparator (`config/hypo.yaml`, external comparators).
The teacher is fitted once per rotation and frozen. Its predicted probability p_T(x) labels the synthetic cohort.

### 3.3 Synthetic cohorts (the only thing the source contributes beyond the teacher)

All inputs below are computed on the source sites' training rows. They are the only quantities taken from those
rows. Each is computed on at least 50 patients.

- **Marginals.** For each continuous or ordinal input:
  - the empirical quantiles at 1/200 steps, with the outer tails clipped to the 1st and 99th percentiles;
  - the missingness share, if the input can be missing.

  For each nominal input: the category shares. A category with fewer than 50 patients is merged into "other". Binary
  inputs and missingness indicators are treated as nominal.
- **Joint structure.** A Gaussian copula:
  - Each input is mapped to normal scores through its marginal. A nominal input is mapped through its categories
    ordered by frequency, with the normal-score midpoint of each category's share. A missing value is placed at a
    fixed score outside the observed range.
  - The rank correlation (Spearman) matrix of these scores is estimated on the source rows and made positive
    definite by eigenvalue clipping at 1e-6.
- **Site.** Each site's aggregates are computed separately. The synthetic cohort is a mixture of the two source
  sites, in proportion to their numbers of landmarks.

From these, two synthetic cohorts are generated per rotation, each with the same number of rows as the rotation's
real learning sample:

- **SYN-COP (primary).** Draws from each site's Gaussian copula, mapped back through the marginals.
- **SYN-IND (control).** Independent draws from each marginal, with no joint structure.

Each synthetic row is its own synthetic patient. No row of the source is copied or perturbed.

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
   path. It must reproduce arm `value`'s weights to 1e-6. If it does not, the run stops and the difference is
   investigated and logged.
2. **Copying.** For each synthetic cohort, the distance from every synthetic row to its closest real source row is
   computed in the standardised input space (distance to closest record, DCR). It is reported against the same
   distance between two disjoint halves of the real source rows. Only aggregates are reported. This shows that no
   synthetic row reproduces a real one. It is descriptive, not a decision rule.
3. **Fidelity.** These are reported per rotation and are descriptive:
   - the teacher's AUROC on held-out real source rows;
   - the agreement between the teacher's mean p_T on SYN-COP and the real source prevalence;
   - the marginal and pairwise-correlation errors of SYN-COP and SYN-IND.

## 4. Hypotheses and pre-specified readings

The patient-level bootstrap uses 2,000 resamples per unit, with one set of replicates for every statistic of a unit.
Log-loss is the primary metric. The paired contrast is computed with `scripts/arc_compare.py`'s statistics:

- the "value" arm is M_REAL;
- the "arc" arm is M_COP (or M_IND);
- the Euclidean kNN is common to all;
- every unit is checked to evaluate identical rows, patients and labels.

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
