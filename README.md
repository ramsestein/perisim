# pmknn — portable-metric kNN

**PM-kNN** is a k-nearest-neighbours method whose distance is learned at some sites and used, unchanged, at another:

- each input counts with a **weight** w_j learned by a **diagonal NCA** (Goldberger et al., 2004);
- each input is read on its **shape function** g_j: how its values move the prediction, the consensus of many
  competitive models (a *Rashomon set*). The g_j make the metric readable;
- the prediction is the **vote of the k nearest rows of a case base**.

The metric (g_j, w_j) is a small JSON file with aggregates only. It travels to another site, where it votes with
that site's own patients: the geometry comes from the development sites and the calibration from the local case
base, with no parameter refitted. That is what "portable" means here.

**What carries the accuracy.** In PERISIM's ablations the learned weights carry the gain over a standardised
Euclidean kNN. With learned weights, the shape functions add nothing to prediction (`hypo`, within 0.001 nats), and
the source of the weights (NCA, SAGE, SHAP, permutation) changes little. The shapes are kept as the layer for reading
the metric and for checking drift between sites.

The library is the method of the [PERISIM](https://github.com/ramsestein/perisim) project (branch `main`: SPECS §5,
experiments T2 and `hypo`), packaged on its own. See [Validation](#validation) for how it matches that implementation.

**Formerly `ceknn`** ("consensus effect-space kNN", versions 0.1–0.2). Renamed in 0.3.0 so the name says what the
method does rather than which component it is built from. Metric files written by `ceknn` load unchanged.

## Install

```bash
pip install "git+https://github.com/ramsestein/perisim.git@pmknn"
```

Optional extras:

| extra | what it adds |
|---|---|
| `boosting` | LightGBM and XGBoost in the model space |
| `fast` | exact neighbour search with FAISS |
| `plot` | `plot_shape` |
| `all` | all of the above |

For example: `pip install "pmknn[all] @ git+https://github.com/ramsestein/perisim.git@pmknn"`. The core needs only
numpy, scipy, pandas and scikit-learn.

## Quick start

```python
from pmknn import PMKNN, ModelSpace

model = PMKNN(
    inputs={"age": "continuous", "asa": ("ordinal", [1, 2, 3, 4, 5]), "surgery": "nominal",
            "emergency": "binary", "lactate": "continuous"},
    output="complication",              # binary or multiclass; classes=[...] fixes their order
    group="patient_id",                 # rows of one patient: never their own neighbours, split together
    model_space=ModelSpace.default(),   # the families of the Rashomon set
    n_models=40, epsilon=0.05, k=50,
)
model.fit(df)                           # learns g_j and w_j; df becomes the case base

model.shape("age")                      # the curve of an input, bin by bin
model.importance()                      # the weight of each input in the distance
model.rashomon()                        # every model tried, and which ones entered the consensus
model.save("metric.json")               # the metric only

there = PMKNN.load("metric.json").set_case_base(df_local)    # another site, its own patients
there.vote(df_new)                      # per class: neighbours, their share, the smoothed probability
there.neighbours(df_new, k=10)          # the neighbours themselves

model.training_report()                 # settings, stopping, stability trace, Rashomon, NCA, k and α curve
print(model.summary())
```

`examples/quickstart.py` runs the whole cycle on synthetic data.

## Inputs and output

- **`inputs`** is either a list of column names, with types inferred, or a mapping column → type. The types are:
  - `"continuous"`;
  - `"ordinal"`, or `("ordinal", levels)` with ordered numeric levels or labels;
  - `"binary"` (0/1 or boolean);
  - `"nominal"`.

  Inference reads boolean or 0/1 numeric columns as binary, other numeric columns as continuous and anything else as
  nominal. It never infers ordinal.
- **Missing values.** Missing inputs are allowed:
  - a continuous or ordinal input takes the training median, and a binary input the training prevalence;
  - either gets a missing indicator (`<input>__missing`) when more than 5 % of the rows miss it;
  - for a nominal input, missing is one more category.
- **`output`**: two or more classes. With a binary output, curves are read as effects on the log-odds of the second
  class. With several classes, they are centred log-odds per class.
- **`group`** (optional): the patient. A patient's rows never vote for each other. Each patient enters a
  neighbourhood once, at its closest row. Training and validation splits are by patient.
- **`site`** (optional), for development data from several sites:
  - the site enters the models as a nuisance column, so each g_j is a within-site effect;
  - the NCA compares rows within their site;
  - inputs constant within every site leave the metric.
- Both columns are needed to `fit`, and neither to predict or to `set_case_base`: a query without the group column
  shares no patient with the case base (its own rows can vote for it), a case base without one has one row per
  patient, and a case base without the site column is one site.

## The Rashomon set

Models are drawn at random from a `ModelSpace`, each trained with feature dropout and scored on a validation split:

1. A model is accepted while its validation log-loss is ≤ (1 + ε)·best so far.
2. The loop stops by one of two rules (`shapes={"stopping": ...}`), described below.
3. The consensus uses the accepted models still within (1 + ε) of the final best.

After every accepted model the loop records how stable the metric is:

- **Δ**: the median relative change of the distances between fixed random pairs. It is weighted by the current
  SAGE importance with `sage_in_loop`, uniformly otherwise.
- **Δ̃**: the moving median of Δ over `window` models.
- **The range of each input's curve** (log-odds) and, with `sage_in_loop`, each input's SAGE importance aggregated
  over the competitive models.
- **The Spearman correlation** of both rankings against `window` models before: the stability of the importance.

The stopping rules:

| `stopping` | stops when | settings |
|---|---|---|
| `"fixed"` (default, as in the `hypo` experiment) | `n_models` models are accepted | `n_models` |
| `"stability"` (as in T2) | Δ̃ has not improved by more than `min_delta` for `patience` accepted models | `n_min` / `n_max`; `window`, `patience`, `min_delta` (`None`: `min_delta_fraction` of Δ̃ at `n_min`); a final Δ̃ above `tol_delta` is flagged |

`sage_in_loop=True` computes SAGE for every accepted model. It is slower, and it gives the importance and its
stability during the training.

Choosing the space:

```python
ModelSpace.default()                                         # every built-in family installed
ModelSpace.default(include=["logreg", "spline_logreg"])      # some of them
ModelSpace.default(exclude=["xgboost"])
space = ModelSpace.default()
space.set_weight("random_forest", 2)                         # drawn twice as often
space.families["lightgbm"]["params"]["num_leaves"] = {"int": [8, 32]}   # a range
space.add("tree", lambda p, seed, n_jobs: DecisionTreeClassifier(max_depth=p["depth"], random_state=seed),
          params={"depth": {"int": [2, 6]}}, handles_nan=True)          # any classifier
```

The built-in families are:

| family | model | shapes it draws |
|---|---|---|
| `logreg` | logistic regression; L2, L1 or elastic-net penalty | straight lines |
| `spline_logreg` | penalised B-spline logistic regression, a GAM | smooth curves |
| `random_forest` | random forest | steps |
| `lightgbm`, `xgboost` | gradient boosting | steps |

Ranges are written `{"log": [lo, hi]}`, `{"uniform": [lo, hi]}`, `{"int": [lo, hi]}`, `{"choice": [...]}` or a fixed
value.

## Reading the metric

| call | what it gives |
|---|---|
| `shape(name)` | the curve, one row per bin (continuous / ordinal) or category (nominal): rows in it, consensus value, its standard error and the 10–90 % range of the models' curves |
| `shapes()` | every curve |
| `plot_shape(name)` | draws one (matplotlib) |
| `importance()` | per variable: `weight` (its share of the mean squared distance between random pairs, sum 1), the raw NCA weight, whether it has a shape, the range of its curve in log-odds, the models' agreement R², and `sage` after `compute_sage` |
| `compute_sage(df)` | SAGE importance of each input group with every competitive model, aggregated over the Rashomon set (DerSimonian–Laird). It needs the fitted models, so it is available only in the session that ran `fit`; its result is saved with the metric |
| `rashomon()` | every model tried: family, hyperparameters, validation log-loss, accepted, competitive |
| `effects(df)` | where each row sits on each curve (its coordinates in the effect space) |
| `transform(df)` | the embedding itself (`weighted=True` multiplies by √w) |

An input whose curve is not distinguishable from flat keeps a neutral axis: rank for continuous, one-hot for nominal.
Its weight still says how much it counts.

## Predicting

The prediction is the vote of the k nearest rows of the case base. α (and, with `select_k`, k) is chosen by a
patient-level leave-one-out over the case base.

| call | what it gives |
|---|---|
| `vote(df, k)` | per query: valid neighbours; for each class, how many neighbours (`n[c]`), their raw share (`share[c]`) and the smoothed probability (`p[c]`); the k and α used, the distance to the k-th neighbour, its percentile and the out-of-support flag |
| `predict_proba(df)` | the smoothed probability alone, p_c = (n_c + α·P(c)·C) / (k + α·C); `as_frame=True` names the class columns |
| `predict(df)` | the most probable class |
| `neighbours(df, k)` | one row per (query, neighbour): position and index in the case base, patient, distance, outcome, and whether the query is out of support (its k-th neighbour farther than the 99th percentile of the leave-one-out) |
| `out_of_support(df)` | that flag alone |
| `carry(df, columns)` | what the neighbours carry in other columns, for example later events: the mean of a numeric column, the share of each category of any other |
| `evaluate(df)` | log-loss, Brier score and AUROC |
| `set_case_base(df)` | change the rows that vote; the metric does not change, and α (and k) are chosen again |

## Seeing the training

| call | what it gives |
|---|---|
| `settings()` | every setting: section, value and what it does |
| `stopping()` | the rule, why the loop stopped, models accepted / tried / competitive, the final Δ̃ and the `min_delta` used |
| `stability()` | one row per accepted model: validation log-loss, Δ, Δ̃, the Spearman stability of the ranking, and every input's curve range (and SAGE) at that point |
| `rashomon()` | every model tried |
| `knn_curve()` | the leave-one-out log-loss of every (k, α) of the grids, with the best α of each k and the pair in use |
| `case_base_info()` | the case base: rows, class shares, k, α, leave-one-out log-loss, out-of-support threshold |
| `training_report()` | all of the above, plus the NCA loss per epoch |
| `plot_stability()`, `plot_knn()`, `plot_shape(name)` | the same, drawn |
| `summary()` | a short text |

The stability trace, the stopping, the (k, α) curve of the training case base and the NCA loss are saved with the
metric, so a loaded metric shows them too.

## The JSON file

`save` and `to_json` write:

- the schema (inputs, output, classes);
- every setting and the model space;
- once fitted, the metric: encoder, curves, weights and diagnostics;
- the training record: the Rashomon set, the stopping, the stability trace, the NCA loss, the (k, α) curve of the
  training case base and, if computed, SAGE.

An unfitted model exports its settings, so a configuration can be written once and fitted later. `PMKNN.load` gives
back a model that predicts as soon as it has a case base.

The file holds **aggregates only**:

- **Never in the file:** the case base, any data row, the fitted models.
- **Distributions:** means, SDs and medians, the inner bin edges (the outer two, the training minimum and maximum,
  are left out) and inner quantiles, each backed by ≥ `min_bin` rows (50 by default) on each side. An input with
  ≤ 2·`min_bin` observed values in the learning sample cannot be summarised this way and is left out of the metric.
  The range of an ordinal input without `levels` is exported as its outer inner edges, not its minimum and maximum.
- **Categories:** only those with ≥ `min_bin` rows. Rarer ones are pooled into `__other__`.
- **Counts:** rounded to 10; a count from 1 to 9 is suppressed (`null`), in the bins, per site and per class, and
  the class shares are those of the rounded counts. The model in memory keeps the exact values.

## Settings

Each group of settings is a dataclass, or a dict passed to `PMKNN`:

| argument | class | defaults |
|---|---|---|
| `encoding=` | `EncodingSettings` | `max_bins=20`, `min_bin=50`, `missing_threshold=0.05`, `max_categories=50`, `ecdf_knots=100` |
| `shapes=` | `ShapeSettings` | `stopping="fixed"`, `n_models=40`; `n_min=20`, `n_max=150`, `window=5`, `patience=10`, `min_delta=None`, `min_delta_fraction=0.05`, `tol_delta=0.01`; `epsilon=0.05`, `max_train=20000`, `max_val=10000`, `n_explain=2000`, `subsample=1000`, `flat_alpha=0.05`, `corr_cluster_threshold=0.8`, `sage_in_loop=False`, … |
| `nca=` | `NCASettings` | `epochs=30`, `n_candidates=500`, `lr=0.05`, `batch_size=256`, `refresh_every=5` |
| `knn=` | `KNNSettings` | `k=50`, `k_grid=[5, 10, 20, 50, 100, 200]`, `select_k=False`, `alpha_grid=[0.1, 0.5, 1, 2, 5, 10]`, `ood_percentile=99`, `distinct_groups=True` |
| `learn=` | `LearnSettings` | `max_rows=30000` per site, `rows_per_group=None` |

The learning sample (`learn`) is what the metric is fitted on. The case base always keeps every row.

## Validation

`tools/compare_with_perisim.py <PERISIM checkout>` runs this library and the PERISIM code it was ported from (the
frozen `perisim` and `hypo` packages of PERISIM v0.5) on the same synthetic data, seed and settings, in two cases:
one row per patient at one site (6000 rows), and repeated patients at two sites (6000 rows, 2000 patients with 3
rows each, each patient at one site; the second site has a higher base rate and never measures `lab`, so its
missing indicator is constant within every site). The result with 0.4.0 on 2026-11-05:

| piece | one row per patient, one site | repeated patients, two sites |
|---|---|---|
| models' matrix (with the site nuisance column) | identical (0.0) | identical (0.0) |
| inputs constant within every site | — | the same (`lab__missing`) |
| models drawn (group-level split), and their validation log-loss | the same, 1e-16 | the same, 1e-16 |
| consensus curves g_j (frozen flatness test) | ≤ 5e-16 | ≤ 3e-16 |
| NCA weights (numpy here, torch there; within-site candidates, own patient excluded) | weight share < 0.0001 apart; correlation of log w 1.000 | the same |
| case base: neighbours (own patient excluded, distinct patients), α, LOO log-loss, out-of-support threshold, probabilities | — | identical |
| neutral axis of a flat input | ≤ 0.03 (rank read from quantiles, see below) | |

Two method changes of 0.4.0 make the library differ from that frozen code on purpose, so the comparison keeps them
out and reports them apart:

- **Discrete top value:** the synthetic inputs are continuous with no ties at the maximum, so the bins are the same
  as before (the script checks it). With ≥ `min_bin` rows at the maximum the top bin now differs.
- **Flatness test:** the curves are compared with the frozen test (`consensus_shapes(..., local_test=False)`); the
  script then applies the 0.4.0 test to the same models' curves. In both cases it keeps the same inputs as the
  frozen test (all four continuous inputs, `noise` included: with 12 models its curve passes either test), so the
  metric is the same here; on other data the set of shaped inputs can differ.

## What differs from PERISIM's pipeline

- **The neutral rank and rare categories.** The rank is read from ≤ 100 inner quantiles instead of the sorted
  training values. A category not kept goes to `__other__` instead of a neutral contribution. Both keep individual
  values out of the exported metric.
- **Since 0.4.0, the discrete top value and the flatness test** follow PERISIM v0.6, not the frozen v0.5 code
  used for T2 and `hypo` (see [Changes in 0.4.0](#changes-in-040)).
- **The NCA is written in numpy**, with its gradient in closed form, so torch is not needed.
- **No SAGE prior on the weights.** In T2, the prior's λ came out ≈ 0. SAGE is an optional report, after the fit or
  inside the loop.
- **`min_delta` of the stability rule.** With `min_delta=None` it is taken within the same run, as a fraction of Δ̃
  at `n_min`. T2 took it from a separate pilot run.
- **The default stopping rule** is `"fixed"`, which is what the comparison above checks. With `"stability"` and
  `sage_in_loop`, the loop follows T2's rule.
- **Not ported:** the hierarchical (ICD-10-PCS) input type, the T1 / T2 / T3 targets and the cohort extractors.
  They belong to the PERISIM pipeline.

## Changes in 0.4.0

The library side of PERISIM v0.6 (software audit of 2026-11-05):

- **Prediction at another site.** `predict_proba`, `vote`, `neighbours` and `set_case_base` no longer need the
  group and site columns of the development data (they raised `KeyError`); see [Inputs and output](#inputs-and-output).
- **No individual value in the JSON.** Ordinal inputs without `levels` no longer export their minimum and maximum;
  the neutral rank no longer forces ≥ 3 knots (with few rows they were single observations); per-site and per-class
  counts in the training record are rounded / suppressed. See [The JSON file](#the-json-file).
- **NCA with one site.** A site column with a single level and fewer rows than `n_candidates` crashed.
- **Discrete top value** (method change). When ≥ `min_bin` training rows equal the maximum (ASA 5, GCS 15), the
  maximum gets its own top bin; before, it shared the last bin with the values below it. A lone extreme still does
  not get a bin of its own.
- **Flatness test** (method change). For continuous / ordinal inputs the test that keeps a curve reads the first
  differences of the models' curves (local effects, V − 1 values; flat = all zero) instead of the accumulated
  curve values, whose noise is correlated: in a simulation of pure-noise ALE curves (`tests/test_v040.py`) the old
  test kept 12–20 % of them, the new one about 4 % (level 0.05, Bonferroni over classes, both unchanged). Nominal inputs are unchanged.
- **Bin labels.** The outer bins read "< e₁" and "≥ e_last" (open intervals), never the training minimum / maximum,
  and `shape()` gives no `lo` for the first bin nor `hi` for the last.
- `tools/compare_with_perisim.py` now also compares repeated patients and two sites (see [Validation](#validation)).

Metric files of 0.3 load unchanged; refitting a metric with 0.4.0 can change its bins and which curves are kept.

## Development

```bash
pip install -e ".[all,test]"
pytest
```

The tests use synthetic data only. No real data enter this repository.

## Licence

MIT. Ramses Marrero García.
