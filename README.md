# ceknn — consensus effect-space kNN

**CE-kNN** is a k-nearest-neighbours method with a learned, readable distance:

- each input is carried to its **shape function** g_j: how its values move the prediction, read from many
  competitive models (a *Rashomon set*) and combined into a consensus curve. The space of the g_j is the *effect
  space*;
- each input counts with a **weight** w_j learned by a diagonal NCA;
- the prediction is the **vote of the k nearest rows of a case base**.

Neighbours are rows that reach the same prediction for the same reasons. The metric (g_j, w_j) is a small JSON file
that can travel to another site, where it votes with that site's own patients.

The library is the method of the [PERISIM](https://github.com/ramsestein/perisim) project (branch `main`: SPECS §5,
experiments T2 and `hypo`), packaged on its own. See [Validation](#validation) for how it matches that implementation.

## Install

```bash
pip install "git+https://github.com/ramsestein/perisim.git@ceknn"
```

Optional extras:

| extra | what it adds |
|---|---|
| `boosting` | LightGBM and XGBoost in the model space |
| `fast` | exact neighbour search with FAISS |
| `plot` | `plot_shape` |
| `all` | all of the above |

For example: `pip install "ceknn[all] @ git+https://github.com/ramsestein/perisim.git@ceknn"`. The core needs only
numpy, scipy, pandas and scikit-learn.

## Quick start

```python
from ceknn import CEKNN, ModelSpace

model = CEKNN(
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

there = CEKNN.load("metric.json").set_case_base(df_local)    # another site, its own patients
there.predict_proba(df_new)
there.neighbours(df_new, k=10)
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

## The Rashomon set

Models are drawn at random from a `ModelSpace`, each trained with feature dropout and scored on a validation split:

1. A model is accepted while its validation log-loss is ≤ (1 + ε)·best so far.
2. The loop stops after `n_models` accepted models.
3. The consensus uses the accepted models still within (1 + ε) of the final best.

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

| call | what it gives |
|---|---|
| `predict_proba(df)` | the smoothed vote, p_c = (n_c + α·P(c)·C) / (k + α·C). α is chosen by a patient-level leave-one-out over the case base |
| `predict(df)` | the most probable class |
| `neighbours(df, k)` | one row per (query, neighbour): position and index in the case base, patient, distance, outcome, and whether the query is out of support (its k-th neighbour farther than the 99th percentile of the leave-one-out) |
| `out_of_support(df)` | that flag alone |
| `carry(df, columns)` | the mean of other columns over the neighbours, for example later events |
| `evaluate(df)` | log-loss, Brier score and AUROC |
| `set_case_base(df)` | change the rows that vote; the metric does not change |

## The JSON file

`save` and `to_json` write:

- the schema (inputs, output, classes);
- every setting and the model space;
- once fitted, the metric: encoder, curves, weights, diagnostics, the record of the Rashomon set and, if computed,
  SAGE.

An unfitted model exports its settings, so a configuration can be written once and fitted later. `CEKNN.load` gives
back a model that predicts as soon as it has a case base.

The file holds **aggregates only**:

- **Never in the file:** the case base, any data row, the fitted models.
- **Distributions:** means, SDs and medians, the inner bin edges (the outer two, the training minimum and maximum,
  are left out) and inner quantiles, each step backed by ≥ `min_bin` rows (50 by default).
- **Categories:** only those with ≥ `min_bin` rows. Rarer ones are pooled into `__other__`.
- **Counts:** rounded to 10.

## Settings

Each group of settings is a dataclass, or a dict passed to `CEKNN`:

| argument | class | defaults |
|---|---|---|
| `encoding=` | `EncodingSettings` | `max_bins=20`, `min_bin=50`, `missing_threshold=0.05`, `max_categories=50`, `ecdf_knots=100` |
| `shapes=` | `ShapeSettings` | `n_models=40`, `epsilon=0.05`, `max_train=20000`, `max_val=10000`, `n_explain=2000`, `subsample=1000`, `flat_alpha=0.05`, `corr_cluster_threshold=0.8`, … |
| `nca=` | `NCASettings` | `epochs=30`, `n_candidates=500`, `lr=0.05`, `batch_size=256`, `refresh_every=5` |
| `knn=` | `KNNSettings` | `k=50`, `alpha_grid=[0.1, 0.5, 1, 2, 5, 10]`, `ood_percentile=99`, `distinct_groups=True` |
| `learn=` | `LearnSettings` | `max_rows=30000` per site, `rows_per_group=None` |

The learning sample (`learn`) is what the metric is fitted on. The case base always keeps every row.

## Validation

`tools/compare_with_perisim.py <PERISIM checkout>` runs this library and the PERISIM code it was ported from (the
frozen `perisim` and `hypo` packages) on the same synthetic data, seed and settings. The result on 2026-10-30:

| piece | difference |
|---|---|
| models' matrix | identical (0.0) |
| models drawn, and their validation log-loss | the same, 1e-16 |
| consensus curves g_j | ≤ 7e-16 |
| NCA weights (numpy here, torch there) | ≤ 0.0002 of the weight share; correlation of log w 1.000 |
| neutral axis of a flat input | ≤ 0.03 (rank read from quantiles, see below) |

## What differs from PERISIM's pipeline

- **The neutral rank and rare categories.** The rank is read from ≤ 100 inner quantiles instead of the sorted
  training values. A category not kept goes to `__other__` instead of a neutral contribution. Both keep individual
  values out of the exported metric.
- **The NCA is written in numpy**, with its gradient in closed form, so torch is not needed.
- **No SAGE prior on the weights.** In T2, the prior's λ came out ≈ 0. SAGE is an optional report.
- **Not ported:** the hierarchical (ICD-10-PCS) input type, the T1 / T2 / T3 targets and the cohort extractors.
  They belong to the PERISIM pipeline.

## Development

```bash
pip install -e ".[all,test]"
pytest
```

The tests use synthetic data only. No real data enter this repository.

## Licence

MIT. Ramses Marrero García.
