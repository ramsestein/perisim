# Pre-registration — arc-length placement of the shape functions: every PERISIM experiment re-run in two arms

**Status.** Draft of 2026-11-08 (branch `arc-metric`), written by the lead with the analyst **before either arm has
run on any real data**. It is meant to be uploaded to OSF as written, and the OSF identifier is then passed to every
freeze of this re-run (`--registration <OSF ID>`). The code of both arms has run only on synthetic data (the test
suite and the smoke tests).

**Registered on OSF:** https://doi.org/10.17605/OSF.IO/NAQ2T (2026-11-08, before any real-data run of either arm;
code at registration: branch `arc-metric`, commit `0addc45`).

## 1. Question

In PERISIM every continuous or ordinal input lives on its consensus shape function g_j: the curve, agreed by an
ensemble of models, of how the input's values move the predicted log-odds (accumulated local effects, centred, one
component per class). The intended reading is a **deformed axis**: equal steps of a variable should count as more
distance where the risk changes fast and as less where it is flat. Temperature is the lead's example: 35.5 → 36 °C
and 38 → 38.5 °C are both 0.5 °C, but only the second crosses into fever.

The published code (v0.5–v0.6) does not place a value on its axis that way. It uses the value of g_j at the value's
bin, φ_j(x) = g_j(bin(x)), so the distance between two values is the **net change** ‖g_j(a) − g_j(b)‖:

1. it is constant within a bin and jumps at each bin edge;
2. on a non-monotone curve the two sides collapse: in `hypo`, the INSPIRE consensus of pH for death is U-shaped,
   so pH 7.32 and 7.48 sit at nearly the same height and count as neighbours on that component;
3. the bins are training quantiles, so resolution is spent where patients are, not where the curve changes. For
   example, the top temperature bin of T2 starts at 37.8 °C, so 38.0 and 38.5 °C are at distance zero.

The **arc-length placement** follows the deformed-axis reading. φ_j(x) is the length of the path walked along g_j from
its first node to x, so the distance between two values is the **path walked** between them, ∫ ‖g_j′‖. It is
continuous and monotone in x. It is zero only where the curve is flat, and every stretch where the prediction changes
adds distance, whatever its direction.

The question is whether the placement changes what PERISIM predicts and what transports. Every published experiment
is re-run in two arms that differ only in this placement:
- **T2** (ICU discharge: stay and death; transport stage);
- **`hypo`** (hypotension in the next 6 h; rotation across INSPIRE, MIMIC-IV and eICU);
- **`roll`** (rotating case base along a stream);
- **NWICU** (the frozen `hypo` metrics in a fourth system).

## 2. The change, exactly

`Encoder.shape_embedding` (`src/perisim/metric/encoding.py`):

- **value** (published): φ_j(x) = g_j[bin(x)], a vector with one component per class.
- **arc**: one dimension per variable. Node v sits at the midpoint of bin v, (e_v + e_v+1)/2, which is where the ALE
  value of a bin sits for a curve that is linear within the bin. The length of segment v is ‖g_j[v+1] − g_j[v]‖, the
  Euclidean norm over the class components. φ_j(x) is the cumulative length at x, interpolated linearly between nodes,
  and constant beyond the outer nodes. A missing value goes to the median, as in the value placement.

Everything else is identical in the two arms:
- how g_j is estimated: the ensemble, ALE, consensus, shrinkage, flatness test and bins;
- the neutral axes of variables without a shape;
- the nominal and ICD-10-PCS geometry, binary variables and missingness indicators;
- the normalisation s_j, which keeps E_pairs[δ_j²] = 1;
- the diagonal NCA, k, α, the vote and the case bases;
- every comparator;
- the cohorts, landmarks, labels, splits and seeds;
- each experiment's protocol and settings.

Code: `src/perisim/metric/encoding.py` (`_arc`, `arc_nodes`, `shape_points`, `_metric_block`). The drift lookups
quantise an arc axis to its bin nodes, as they already did for neutral rank axes (`exact = False`).

## 3. What is known before the runs

- **All published results are known to the analysts** (README, `docs/FINDINGS.md`, the reports). The most relevant
  is `hypo`'s exploratory ablation (REPORT §14): with learned weights, the shapes add nothing to prediction over
  neutral axes (within 0.001 nats). The predictive signal is in the weights.
- **The new code has run only on synthetic data.**
- **The published runs used the frozen code v0.5.** This branch starts from v0.6, which fixes the defects found by
  the audit of 2026-11-05, including the landmark code. That is why both arms are re-run (§4) instead of comparing
  the arc arm with the published numbers.

## 4. Design: two arms on the same code

- **Arms.**
  - `value`: v0.6 as on `main`.
  - `arc`: the same code with the arc placement.
- **Same code, data and seeds.** Both arms run from the same commit of this branch, on the same PhysioNet copies,
  with the same seeds and patient-level splits. The arm is set by `PERISIM_ARM`.
- **The arm enters the hashes.** It is part of the configuration (`${PERISIM_ARM}` in `config/experiment.yaml` and
  `config/hypo.yaml`), so it is part of T2's scientific hash, of `hypo`'s configuration hash and therefore of every
  freeze record.
- **Separate work folders.** Each arm has its own work folder (`work_value`, `work_arc`), bound to it by a marker
  file. A folder of the published runs, or of the other arm, is refused.
- **Shared inputs.** The metric-independent inputs may be copied from the published work folder with
  `scripts/arc_seed_inputs.py`: the T2 extraction and `hypo`'s extracts of the original tables. `hypo`'s landmark
  tables are rebuilt; they should be identical, and this is checked (§6.3).
- **Each experiment runs its own frozen protocol, once per arm, unchanged:**
  - T2: the pipeline up to `transport`, with its freeze before `external`;
  - `hypo`: `landmarks`, `audit`, `cv`, `final`, `freeze` and `external`;
  - `roll`: its freeze, `run` and `endpoints`;
  - NWICU: `nwicu_build.py` and `nwicu_transport.py`.

  Every freeze carries this registration's OSF ID. The post hoc analyses that had a pre-specified plan are re-run in
  both arms with their plans: transport equivalence (`hypo` §16, T2), weights between hospitals (§20) and cosine
  (§21). Any other post hoc script re-run on this branch is exploratory.
- **The paired contrast** (§5–§6) is computed only after both arms have finished every experiment.
  - `scripts/arc_dump.py`, run once per arm, reads the saved predictions; it never refits.
  - `scripts/arc_compare.py` stops unless both arms evaluated the same rows, patients and labels with the same
    Euclidean kNN.

## 5. Units of the paired contrast

| unit | rows | transported metric | local metric | Euclidean kNN |
|---|---|---|---|---|
| `hypo_<X>`, X ∈ {INSPIRE, MIMIC-IV, eICU} | the external stage's test landmarks of X, five folds | frozen metric learned on the other two sites, X's case base | learned on X's training folds | Phase 2's Euclidean comparator of X |
| `nwicu_<e>`, e ∈ the three `hypo` rotations | NWICU landmarks, five folds | `hypo`'s frozen metric of rotation e, NWICU case base | learned on NWICU | NWICU Euclidean comparator |
| `t2_<c>`, c ∈ {MIMIC-IV, eICU} | T2 transport stage's fixed test set | A(i) frozen INSPIRE metric, local case base | A(iii) local refit | Euclidean kNN, same case base |

**Primary units:** the six `hypo` and `nwicu` units. **Secondary units:** the two T2 units, because T2's signal was
weak and its R was not evaluable or not equivalent in the published run.

## 6. Hypotheses and pre-specified readings

There are 2,000 patient-level bootstrap resamples per unit, the same replicates for every statistic of the unit.
Log-loss is the primary metric.

### 6.1 Primary: does the placement change the transported metric's predictions?

For each primary unit:

- **Q = (L_euclid − L_arc) / (L_euclid − L_value).** This is the share of the value placement's gain over the
  Euclidean kNN that the arc placement achieves. The bioequivalence convention applies, two-sided:
  - **equivalent** if the 90 % CI of Q lies inside [0.80, 1.25];
  - **not evaluable** if the value placement's gain has a point estimate below 0.005 nats, or a 95 % CI that
    includes 0;
  - otherwise **not equivalent**, with the side named: *arc worse* if the lower limit is < 0.80, *arc better* if the
    upper limit is > 1.25.
- **D = L_arc − L_value** (nats, paired) with its 95 % CI: *arc better* if the upper limit is < 0, *arc worse* if the
  lower limit is > 0, otherwise *no difference*.

**Overall reading.** The arc placement is **equivalent to the value placement** if Q is equivalent in every
evaluable primary unit, an intersection–union rule that needs no multiplicity adjustment. It **predicts better** if,
in addition, D shows *arc better* in at least four of the six primary units and *arc worse* in none. Units that are
not evaluable are reported as such. If all six are not evaluable, there is no overall reading.

**Expectation, declared now.** Given REPORT §14, we expect equivalence and D within ±0.002 nats. A material gain in
log-loss is not expected. The case for the arc placement is interpretive: a continuous, monotone and
direction-agnostic distance. Equivalence is therefore the result that would let it replace the value placement.

### 6.2 Secondary: transport and the original hypotheses

1. **R within each arm.** R = (L_euclid − L_transported) / (L_euclid − L_local) in each arm and unit, read by the
   published rule: equivalent if the lower 90 % limit is ≥ 0.80, with the same not-evaluable rule.
   **ΔR = R_arc − R_value** with its 95 % CI: *higher with arc*, *lower with arc* or *no difference*.
2. **Each experiment's own pre-registered hypotheses**, re-evaluated in each arm with their original rules:
   - T2's transport amendment, H1–H6;
   - `hypo`'s H1–H5 (`docs/prereg_hypo.md` §6);
   - `roll`'s hypotheses (`docs/prereg_roll.md` §7);
   - NWICU's replication rule (`docs/prereg_nwicu.md`).

   A concordance table states, for each hypothesis, whether the two arms reach the same conclusion.
3. **The locally learned metrics:** Q and D between the arms (`Q_local`, `D_local`), with the same rules, read
   descriptively.
4. **T2 units:** Q, D and R as in §6.1 and §6.2.1, read descriptively.

### 6.3 Descriptive

- **Backtracking index of each shaped variable** in `hypo`'s frozen metrics: 1 − net change / arc length, where 0 is
  monotone. This measures how much of the path along g_j the value placement cancels. It is reported with each arm's
  weight for that variable, to show where the two placements can differ at all.
  - `hypo`'s g_j do not depend on the arm: the ensemble is fitted on the model matrix with the same seeds. This is
    checked to 1e-6.
  - In T2 the consensus loop stops on the stability of a distance that does depend on the placement, so the arms may
    accept different numbers of models (§8).
- **The value arm against the published results** (v0.6 against v0.5): the change in each headline figure. This is a
  reproducibility check, not a test.
- **Landmark tables rebuilt against the published ones:** identical or not, with row counts.

## 7. Rules

- Nothing in §2–§6 changes after the first real-data run of either arm. Deviations forced by mechanics (a crash, a
  resource limit) are fixed without touching the method, logged with a date in `docs/DECISIONS.md`, and reported.
- Each arm's experiments run on that arm's frozen version only. The freezes refuse a dirty repository and refuse to
  re-freeze once external results exist, as in the published runs.
- The paired contrast is computed once, after both arms have finished. Its outputs are aggregates only; cells of n
  < 10 are not reported.
- Both arms are reported in full, whatever the result.

## 8. Limitations, declared in advance

- **The same data as the published runs.** This is a comparison of two placements on the same cohorts, not an
  independent replication of transport. The analysts know the published results; the readings above are fixed so
  that knowledge cannot steer them.
- **Arc length accumulates noise.** |·| does not cancel, so an estimated curve's arc length is biased upward by its
  noise. The shrinkage τ² / (τ² + se²) and the flatness test (curves not distinguishable from flat keep the neutral
  axis) limit this, but do not remove it.
- **Approximations of the arc placement:**
  - the nodes at bin midpoints;
  - linear interpolation between them;
  - the constant extension beyond the outer nodes, which flattens the outer half-bins;
  - the bins are still training quantiles, so the resolution in the tails is unchanged.
- **Nominal variables are not placed by arc length.** Arc length is not defined without an order, so nominal and
  ICD-10-PCS variables keep the published geometry.
- **T2:** the stopping rule of the consensus loop depends on the placement (§6.3), so T2's arms can differ in the
  ensemble as well as in the placement.
- **The drift analyses** (`hypo` S4–S6 and the drift post hoc scripts) quantise an arc axis to its bin nodes, so in
  the arc arm their "frozen" lookup is an approximation of the real metric.

## 9. Registration

OSF, before the first real-data run of either arm. The OSF ID goes into every freeze of both arms. The commit of
branch `arc-metric` at registration, pushed to GitHub, is the verifiable timestamp of the code.
