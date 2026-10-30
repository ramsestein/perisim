"""`CEKNN`: choose the inputs, the output and the Rashomon model space; fit on a dataset; read every input's curve
and weight; export the metric to JSON and load it elsewhere with a local case base.

    m = CEKNN(inputs={"age": "continuous", "asa": ("ordinal", [1, 2, 3, 4, 5]), "surgery": "nominal"},
              output="complication", group="patient_id")
    m.fit(df)                      # learns the metric (g_j, w_j); df becomes the case base
    m.shape("age")                 # the curve of an input
    m.importance()                 # weight of each input in the distance (+ SAGE with compute_sage)
    m.save("metric.json")          # the metric only: never the case base
    m2 = CEKNN.load("metric.json").set_case_base(df_local)
    m2.predict_proba(df_new)

The metric is what travels between sites; the case base (patient-level rows) stays where it is.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .encoding import Encoder
from .knn import CaseBase
from .models import ModelSpace
from .nca import DiagonalNCA
from .rashomon import RashomonSet, ShapeSettings, fit_rashomon, site_columns
from .sage import InputsOnly, aggregate, normalize, per_variable, sage_permutation
from .schema import Schema, Table

FORMAT = "ceknn-metric"
FORMAT_VERSION = 1


@dataclass
class EncodingSettings:
    max_bins: int = 20                 # quantile bins of a continuous / ordinal input
    min_bin: int = 50                  # rows per bin, and per kept category
    missing_threshold: float = 0.05    # share missing above which an indicator is added
    max_categories: int = 50
    ecdf_knots: int = 100              # quantiles of the neutral rank axis


@dataclass
class NCASettings:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 0.05
    refresh_every: int = 5
    n_candidates: int = 500
    label_smoothing: float = 1e-3


@dataclass
class LearnSettings:
    """The learning sample of the metric (encoder, Rashomon set and NCA); the case base keeps every row."""
    max_rows: int | None = 30000       # per site (in all without a site column); None = every row
    rows_per_group: int | None = None  # at most this many rows of each group (patient)


@dataclass
class KNNSettings:
    k: int = 50                        # neighbours that vote
    k_grid: list | None = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])   # scored by leave-one-out
    select_k: bool = False             # True: k = the best of k_grid (and k) in the leave-one-out
    alpha_grid: list = field(default_factory=lambda: [0.1, 0.5, 1, 2, 5, 10])
    alpha_queries: int = 20000         # rows of the leave-one-out that picks α (and k)
    ood_percentile: float = 99.0
    distinct_groups: bool = True       # each group (patient) enters a neighbourhood once
    fetch_factor: int = 6


#: what every setting means (shown by `CEKNN.settings()`)
MEANING = {
    ("encoding", "max_bins"): "quantile bins of a continuous / ordinal input",
    ("encoding", "min_bin"): "rows per bin and per kept category",
    ("encoding", "missing_threshold"): "share missing above which a missing indicator is added",
    ("encoding", "max_categories"): "categories kept per nominal input",
    ("encoding", "ecdf_knots"): "quantiles of the neutral rank axis",
    ("shapes", "stopping"): "'fixed': n_models accepted; 'stability': stop when Δ̃ stops improving",
    ("shapes", "n_models"): "accepted models (stopping 'fixed')",
    ("shapes", "n_min"): "at least this many accepted models (stopping 'stability')",
    ("shapes", "n_max"): "at most this many accepted models (stopping 'stability')",
    ("shapes", "window"): "W: moving median of Δ, lag of the ranking stability",
    ("shapes", "patience"): "P: accepted models without Δ̃ improving by more than min_delta",
    ("shapes", "min_delta"): "improvement of Δ̃ that resets the patience (None: fraction of Δ̃ at n_min)",
    ("shapes", "min_delta_fraction"): "that fraction",
    ("shapes", "tol_delta"): "final Δ̃ above this is flagged",
    ("shapes", "max_attempts_factor"): "models tried ≤ factor × target",
    ("shapes", "epsilon"): "competitive: validation log-loss ≤ (1 + ε)·best",
    ("shapes", "augment_copies"): "masked copies of the training rows (feature dropout)",
    ("shapes", "max_train"): "rows to train each model",
    ("shapes", "max_val"): "rows to score each model",
    ("shapes", "n_explain"): "rows the curves (and SAGE) are read on",
    ("shapes", "subsample"): "fresh subsample of them per model, for the curves",
    ("shapes", "flat_alpha"): "level of the test that keeps a curve (vs flat)",
    ("shapes", "corr_cluster_threshold"): "|ρ| above which inputs share a dropout / SAGE group",
    ("shapes", "site_adjusted"): "the site enters the models as a nuisance column",
    ("shapes", "n_reference_pairs"): "random pairs for Δ",
    ("shapes", "sage_in_loop"): "SAGE of every accepted model: importance and its stability",
    ("shapes", "sage"): "SAGE batch size, passes and convergence threshold",
    ("nca", "epochs"): "passes over the learning sample",
    ("nca", "batch_size"): "anchors per step",
    ("nca", "lr"): "Adam learning rate on log w",
    ("nca", "refresh_every"): "epochs between candidate searches",
    ("nca", "n_candidates"): "candidates per anchor (computational, not k)",
    ("nca", "label_smoothing"): "η: smoothing towards the class shares",
    ("knn", "k"): "neighbours that vote",
    ("knn", "k_grid"): "values of k scored by leave-one-out",
    ("knn", "select_k"): "use the best k of the grid",
    ("knn", "alpha_grid"): "smoothing α of the vote, chosen by leave-one-out",
    ("knn", "alpha_queries"): "rows of the leave-one-out",
    ("knn", "ood_percentile"): "percentile of the k-th distance beyond which a query is out of support",
    ("knn", "distinct_groups"): "each patient enters a neighbourhood once",
    ("knn", "fetch_factor"): "rows fetched per neighbour to find distinct patients",
    ("learn", "max_rows"): "rows of the learning sample per site (None: all)",
    ("learn", "rows_per_group"): "rows per patient in the learning sample (None: all)",
}


def _settings(cls, value):
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"expected {cls.__name__} or dict, got {type(value).__name__}")


class CEKNN:
    """Consensus effect-space kNN.

    inputs: list of columns (types inferred) or mapping column → type / (type, levels).
    output: the outcome column (binary or multiclass); `classes` fixes their order.
    group: rows of the same patient (never their own neighbours; splits are by group).
    site: development site of each row (site-adjusted shapes, within-site NCA candidates).
    model_space: `ModelSpace` of the Rashomon set (default: the built-in families installed).
    """

    def __init__(self, inputs, output: str, *, classes=None, group: str | None = None, site: str | None = None,
                 types: dict | None = None, model_space: ModelSpace | None = None, k: int | None = None,
                 n_models: int | None = None, epsilon: float | None = None, seed: int = 0, n_jobs: int = 4,
                 encoding=None, shapes=None, nca=None, knn=None, learn=None, verbose: bool = False):
        self.schema = Schema.build(inputs, output, classes=classes, group=group, site=site, types=types)
        self.model_space = model_space if model_space is not None else ModelSpace.default()
        self.encoding = _settings(EncodingSettings, encoding)
        self.shapes_settings = _settings(ShapeSettings, shapes)
        self.nca_settings = _settings(NCASettings, nca)
        self.knn_settings = _settings(KNNSettings, knn)
        self.learn = _settings(LearnSettings, learn)
        if k is not None:
            self.knn_settings.k = int(k)
        if n_models is not None:
            self.shapes_settings.n_models = int(n_models)
        if epsilon is not None:
            self.shapes_settings.epsilon = float(epsilon)
        self.seed, self.n_jobs, self.verbose = int(seed), int(n_jobs), bool(verbose)
        self.enc_: Encoder | None = None
        self.w_: np.ndarray | None = None
        self.rashomon_: RashomonSet | None = None
        self.case_base_: CaseBase | None = None
        self.info_: dict = {}
        self.sage_: dict | None = None
        self._cb: dict | None = None

    # ------------------------------------------------------------------ fitting
    @property
    def is_fitted(self) -> bool:
        return self.enc_ is not None

    def fit(self, df: pd.DataFrame) -> "CEKNN":
        """Learns the metric on `df` and makes `df` the case base."""
        sch = self.schema
        sch.resolve(df)
        y = sch.labels(df)
        t = Table.from_frame(df, sch)
        rows = self._learning_rows(t)
        tl, yl = t.take(rows), y[rows]
        enc = Encoder(**asdict(self.encoding)).fit(tl, yl, sch.n_classes)
        dropped = _site_constant(enc, tl) if tl.sites is not None else []
        if dropped:
            enc.drop(dropped)
        if enc.n_features == 0:
            raise ValueError("no input carries information (constant or entirely missing)")
        rs = fit_rashomon(enc, tl, yl, self.model_space, self.shapes_settings, seed=self.seed, n_jobs=self.n_jobs,
                          verbose=self.verbose)
        enc = enc.with_shapes(rs.shapes, tl)
        s = self.nca_settings
        nca = DiagonalNCA(enc.dim_feature, enc.n_features, sch.n_classes, n_candidates=s.n_candidates,
                          batch_size=s.batch_size, lr=s.lr, refresh_every=s.refresh_every,
                          label_smoothing=s.label_smoothing, seed=self.seed)
        nca.fit(enc.transform(tl), yl, epochs=s.epochs, groups=tl.groups, sites=tl.sites)
        self.enc_, self.w_, self.rashomon_ = enc, nca.weights, rs
        self.sage_ = rs.sage
        self.info_ = {"n_rows": int(len(t)), "n_learn": int(len(rows)), "stop": rs.stop,
                      "class_shares": (np.bincount(yl, minlength=sch.n_classes) / len(yl)).round(4).tolist(),
                      "learn_per_site": ({str(v): int((tl.sites == v).sum()) for v in np.unique(tl.sites)}
                                         if tl.sites is not None else None),
                      "dropped_site_constant": dropped, "nca_history": [round(float(x), 6) for x in nca.history],
                      "seconds_rashomon": round(rs.seconds, 2), "n_attempts": len(rs.attempts),
                      "n_accepted": len(rs.models), "n_competitive": int(rs.competitive.sum()),
                      "fitted": _dt.datetime.now().isoformat(timespec="seconds")}
        self._set_case_base(df, t, y)
        self.info_["case_base_at_fit"] = {**self.case_base_.fit_info, "loo_table": self.case_base_.loo_table}
        return self

    def _learning_rows(self, t: Table) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(t))
        r = self.learn.rows_per_group
        if r is not None:                       # at most r random rows of each group
            order = rng.permutation(idx)
            g = pd.Series(t.groups[order])
            rank = g.groupby(g).cumcount().to_numpy()
            idx = np.sort(order[rank < int(r)])
        cap = self.learn.max_rows
        if cap is None:
            return idx
        sites = t.sites[idx] if t.sites is not None else np.zeros(len(idx), dtype=object)
        out = []
        for s in np.unique(sites):
            sel = idx[sites == s]
            out.append(sel if sel.size <= cap else rng.choice(sel, size=int(cap), replace=False))
        return np.sort(np.concatenate(out))

    # ------------------------------------------------------------------ case base
    def set_case_base(self, df: pd.DataFrame) -> "CEKNN":
        """The rows that vote (any site: the metric is used as learned). α and the out-of-support threshold are
        chosen by a group-level leave-one-out over these rows."""
        self._need_metric()
        return self._set_case_base(df, Table.from_frame(df, self.schema), self.schema.labels(df))

    def _set_case_base(self, df: pd.DataFrame, t: Table, y: np.ndarray) -> "CEKNN":
        ks = self.knn_settings
        cb = CaseBase(self.enc_.transform(t), self.enc_.weights_dims(self.w_), y, self.schema.n_classes, t.groups,
                      k=ks.k, distinct=ks.distinct_groups and self.schema.group is not None, fetch=ks.fetch_factor)
        cb.fit_alpha(ks.alpha_grid, k_grid=ks.k_grid, select_k=ks.select_k, n_queries=ks.alpha_queries,
                     ood_percentile=ks.ood_percentile, seed=self.seed)
        self.case_base_ = cb
        self._cb = {"frame": df, "index": df.index.to_numpy(), "groups": t.groups}
        return self

    # ------------------------------------------------------------------ prediction
    def _need_metric(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("the metric is not fitted: call fit(df) or load a metric")

    def _need_case_base(self) -> None:
        self._need_metric()
        if self.case_base_ is None:
            raise RuntimeError("no case base: call set_case_base(df) with the rows that vote")

    def _table(self, df: pd.DataFrame) -> Table:
        return Table.from_frame(df, self.schema)

    def transform(self, df: pd.DataFrame, weighted: bool = False) -> np.ndarray:
        """Embedding of the rows in the effect space (× √w with `weighted`)."""
        self._need_metric()
        E = self.enc_.transform(self._table(df))
        return E * self.enc_.weights_dims(self.w_) if weighted else E

    def _search(self, df: pd.DataFrame, k: int | None):
        self._need_case_base()
        t = self._table(df)
        groups = t.groups if self.schema.group and self.schema.group in df.columns else None
        return self.case_base_.search(self.enc_.transform(t), groups, k=k)

    def predict_proba(self, df: pd.DataFrame, k: int | None = None, as_frame: bool = False):
        """Class probabilities (n × classes, in the order of `classes`): the smoothed vote of the k neighbours,
        p_c = (n_c + α·P(c)·C) / (k + α·C). `as_frame` returns a DataFrame with one column per class."""
        D, I, valid = self._search(df, k)
        p = self.case_base_.probs(I, valid, k)
        return pd.DataFrame(p, index=df.index, columns=list(self.schema.classes)) if as_frame else p

    def vote(self, df: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
        """The neighbours' vote for each query: how many valid neighbours, how many of them and what share are of
        each class (`n[<class>]`, `share[<class>]`), the smoothed probability (`p[<class>]`), the k and α used,
        the distance to the k-th neighbour, its percentile in the case base's leave-one-out and whether the query
        is out of support."""
        D, I, valid = self._search(df, k)
        cb = self.case_base_
        kk = cb.k if k is None else int(k)
        v = cb.vote_table(I, valid, kk)
        kth, pct, ood = cb.kth_info(D, valid)
        out = {"n_neighbours": v["n_neighbours"]}
        for c, lab in enumerate(self.schema.classes):
            out[f"n[{lab}]"] = v["counts"][:, c]
        for c, lab in enumerate(self.schema.classes):
            out[f"share[{lab}]"] = v["share"][:, c]
        for c, lab in enumerate(self.schema.classes):
            out[f"p[{lab}]"] = v["p"][:, c]
        out.update({"k": kk, "alpha": cb.alpha, "kth_distance": kth, "kth_percentile": pct, "ood": ood})
        return pd.DataFrame(out, index=df.index)

    def predict(self, df: pd.DataFrame, k: int | None = None) -> np.ndarray:
        p = self.predict_proba(df, k)
        return np.asarray(self.schema.classes, dtype=object)[p.argmax(1)]

    def neighbours(self, df: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
        """One row per (query, neighbour): its position and index in the case base, group, distance and outcome,
        plus whether the query is out of support."""
        D, I, valid = self._search(df, k)
        _, pct, ood = self.case_base_.kth_info(D, valid)
        q, r = np.nonzero(valid)
        cls = np.asarray(self.schema.classes, dtype=object)
        return pd.DataFrame({"query": q, "rank": r + 1, "case": I[q, r], "case_index": self._cb["index"][I[q, r]],
                             "case_group": self._cb["groups"][I[q, r]], "distance": np.sqrt(D[q, r]),
                             "outcome": cls[self.case_base_.y[I[q, r]]], "query_ood": ood[q],
                             "query_kth_percentile": pct[q]})

    def out_of_support(self, df: pd.DataFrame) -> np.ndarray:
        D, I, valid = self._search(df, None)
        return self.case_base_.kth_info(D, valid)[2]

    def carry(self, df: pd.DataFrame, columns: list[str], k: int | None = None) -> pd.DataFrame:
        """What the neighbours carry in other columns of the case base (later outcomes, treatments…): the mean of
        a numeric column, and the share of each category of any other column (`<column>[<category>]`)."""
        D, I, valid = self._search(df, k)
        kk = I.shape[1] if k is None else min(int(k), I.shape[1])
        frame = self._cb["frame"]
        Ik, vk = I[:, :kk], valid[:, :kk]
        out = {}
        for c in columns:
            s = frame[c]
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                v = np.where(vk, s.to_numpy(dtype=float)[Ik], np.nan)
                with np.errstate(invalid="ignore"):
                    out[c] = np.nanmean(v, axis=1)
                continue
            vals = s.to_numpy(dtype=object)[Ik]
            known = vk & pd.notna(vals)
            tot = known.sum(1)
            for cat in pd.unique(s.dropna()):
                hit = (vals == cat) & known
                out[f"{c}[{cat}]"] = np.divide(hit.sum(1), tot, out=np.full(len(tot), np.nan), where=tot > 0)
        return pd.DataFrame(out, index=df.index)

    def evaluate(self, df: pd.DataFrame, k: int | None = None) -> dict:
        """Log-loss, Brier score and (binary) AUROC of the vote on `df`. Rows of the case base are excluded from
        their own neighbourhood only through the `group` column: without one, evaluate on rows outside the case
        base."""
        y = self.schema.labels(df)
        p = self.predict_proba(df, k)
        C = self.schema.n_classes
        res = {"n": int(len(y)), "log_loss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)))),
               "brier": float(np.mean(((np.eye(C)[y] - p) ** 2).sum(1)))}
        if C == 2 and 0 < y.sum() < len(y):
            from sklearn.metrics import roc_auc_score

            res["auroc"] = float(roc_auc_score(y, p[:, 1]))
        return res

    # ------------------------------------------------------------------ reading the metric
    def _block(self, name: str):
        self._need_metric()
        fi = self.enc_.feature_index()
        if name not in fi:
            known = ", ".join(self.enc_.feature_names)
            raise KeyError(f"{name!r} is not a variable of the metric (it has: {known})")
        return fi[name], self.enc_.blocks[fi[name]]

    def _band(self, name: str):
        if self.rashomon_ is not None:
            cl = self.rashomon_.curves.get(name) or []
            if cl:
                A = np.array(cl)
                return np.quantile(A, 0.1, axis=0), np.quantile(A, 0.9, axis=0)
        b = (self._loaded_bands or {}).get(name) if hasattr(self, "_loaded_bands") else None
        if b:
            return np.asarray(b["q10"]), np.asarray(b["q90"])
        return None, None

    def shape(self, name: str) -> pd.DataFrame:
        """The curve of an input: one row per bin (continuous / ordinal) or category (nominal), with the rows in
        it (rounded to 10 after loading), the consensus value and the 10–90 % range of the models' curves.
        Binary output: `log_odds` (effect on the log-odds of the second class). Multiclass: centred log-odds per
        class (`g[<class>]`). A flat input (no shape) gives NaN values."""
        j, b = self._block(name)
        if b.kind not in ("continuous", "ordinal", "nominal"):
            raise ValueError(f"{name!r} is {b.kind}: only continuous, ordinal and nominal inputs have a curve")
        labels = self.enc_.value_labels(b)
        V = len(labels)
        out = pd.DataFrame({"value": labels, "n": np.asarray(b.params["value_counts"], dtype=float)})
        if b.kind in ("continuous", "ordinal"):
            e = b.params["edges"]
            out.insert(1, "lo", [None if not np.isfinite(x) else float(x) for x in e[:-1]])
            out.insert(2, "hi", [None if not np.isfinite(x) else float(x) for x in e[1:]])
        g = b.shape if b.shape is not None else np.full((V, self.schema.n_classes), np.nan)
        diag = self._diagnostics().get(name, {})
        se = np.asarray(diag["se"]) if diag.get("se") is not None else np.full_like(g, np.nan)
        q10, q90 = self._band(name)
        if self.schema.n_classes == 2:
            out["log_odds"] = g[:, 1] - g[:, 0]
            out["se"] = 2 * se[:, 1] if se.ndim == 2 else np.nan
            if q10 is not None:
                out["models_q10"], out["models_q90"] = q10[:, 1] - q10[:, 0], q90[:, 1] - q90[:, 0]
        else:
            for c, lab in enumerate(self.schema.classes):
                out[f"g[{lab}]"] = g[:, c]
        out.attrs.update(name=name, kind=b.kind, shaped=b.shape is not None)
        return out

    def shapes(self) -> dict[str, pd.DataFrame]:
        self._need_metric()
        return {b.name: self.shape(b.name) for b in self.enc_.blocks if b.kind in ("continuous", "ordinal", "nominal")}

    def _diagnostics(self) -> dict:
        if self.rashomon_ is not None:
            return self.rashomon_.diagnostics
        return getattr(self, "_loaded_diag", {}) or {}

    def importance(self) -> pd.DataFrame:
        """Each metric variable: its weight (share of the mean squared distance between random pairs, sum 1), the
        raw NCA weight, whether it has a shape, the range of its curve in log-odds, the models' agreement R² and,
        after `compute_sage`, its SAGE importance aggregated over the Rashomon set."""
        self._need_metric()
        w = np.asarray(self.w_, dtype=float)
        diag = self._diagnostics()
        rows = []
        for j, b in enumerate(self.enc_.blocks):
            rng_lo = np.nan
            if b.shape is not None:
                g = b.shape
                rng_lo = float(np.ptp(g[:, 1] - g[:, 0])) if self.schema.n_classes == 2 else float(np.ptp(g, 0).max())
            d = diag.get(b.name, {})
            rows.append({"variable": b.name, "input": b.source, "kind": b.kind, "weight": w[j] / w.sum(),
                         "nca_weight": w[j], "shaped": b.shape is not None, "logodds_range": rng_lo,
                         "r2": d.get("r2"), "n_models": d.get("n_models")})
        out = pd.DataFrame(rows)
        if self.sage_ is not None:
            out["sage"] = out["variable"].map(self.sage_)
        return out.sort_values("weight", ascending=False, ignore_index=True)

    def rashomon(self) -> pd.DataFrame:
        """Every model tried: family, hyperparameters, validation log-loss, accepted, competitive."""
        self._need_metric()
        rows = self.rashomon_.attempts if self.rashomon_ is not None else getattr(self, "_loaded_attempts", [])
        return pd.DataFrame(rows)

    def compute_sage(self, df: pd.DataFrame | None = None, n_rows: int = 2000, models: str = "competitive",
                     max_passes: int = 3) -> pd.DataFrame:
        """SAGE of each input group with every competitive (or accepted, `models="accepted"`) model of the
        Rashomon set, aggregated over models, split equally among the variables of a group. Needs the models in
        memory (right after `fit`; they are not saved). `df` defaults to the case base."""
        self._need_metric()
        rs = self.rashomon_
        if rs is None or not rs.models:
            raise RuntimeError("SAGE needs the models of the Rashomon set: fit the metric in this session")
        frame = self._cb["frame"] if df is None else df
        rng = np.random.default_rng(self.seed)
        rows = rng.choice(len(frame), size=min(int(n_rows), len(frame)), replace=False)
        sub = frame.iloc[rows]
        t = self._table(sub)
        y = self.schema.labels(sub)
        X, _ = self._model_matrix(t)
        G_in = len(rs.group_names)
        use = [m for m, c in zip(rs.models, rs.competitive) if c or models == "accepted"]
        vals, ses = [], []
        for m in use:
            r = sage_permutation(InputsOnly(m, G_in, rs.n_groups), X, y, G_in, rng, max_passes=max_passes)
            v, s = normalize(r["values"], r["se"])
            vals.append(v)
            ses.append(s)
        imp = per_variable(aggregate(np.array(vals), np.array(ses)), rs.groups)
        per = {b.name: float(x) for b, x in zip(self.enc_.blocks, imp)}
        self.sage_ = per
        return pd.DataFrame({"variable": list(per), "sage": list(per.values()),
                             "group": [rs.group_names[int(rs.groups[self.enc_.feature_index()[n]])] for n in per]}) \
            .sort_values("sage", ascending=False, ignore_index=True)

    def _model_matrix(self, t: Table):
        X, owner = self.enc_.transform_model(t)
        S = site_columns(t.sites, self.rashomon_.site_levels) if self.rashomon_.site_levels else None
        if self.rashomon_.site_levels:
            if S is None:
                S = np.zeros((len(t), len(self.rashomon_.site_levels)), dtype=np.float32)
            X = np.hstack([X, S])
        return X, owner

    def effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Where each row sits on each curve: g_j(x) in log-odds (binary output) or centred log-odds per class
        (multiclass, columns `<input>[<class>]`). Inputs without a shape are left out."""
        self._need_metric()
        t = self._table(df)
        out = {}
        for b in self.enc_.blocks:
            if b.shape is None:
                continue
            v = self.enc_._metric_block(b, t)
            if self.schema.n_classes == 2:
                out[b.name] = v[:, 1] - v[:, 0]
            else:
                for c, lab in enumerate(self.schema.classes):
                    out[f"{b.name}[{lab}]"] = v[:, c]
        return pd.DataFrame(out, index=df.index)

    def plot_shape(self, name: str, ax=None, cls=None):
        """Draws the curve of an input (matplotlib), with the 10–90 % range of the models' curves when known."""
        import matplotlib.pyplot as plt

        tab = self.shape(name)
        ax = ax or plt.subplots(figsize=(6, 3.5))[1]
        x = np.arange(len(tab))
        if self.schema.n_classes == 2:
            y, ylab = tab["log_odds"].to_numpy(dtype=float), "effect on the log-odds"
            if "models_q10" in tab:
                ax.fill_between(x, tab["models_q10"], tab["models_q90"], step="mid", alpha=0.2, label="models 10–90 %")
        else:
            lab = self.schema.classes[-1] if cls is None else cls
            y, ylab = tab[f"g[{lab}]"].to_numpy(dtype=float), f"centred log-odds of {lab}"
        if tab.attrs["kind"] == "nominal":
            ax.bar(x, y)
        else:
            ax.step(x, y, where="mid", label="consensus")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x, tab["value"], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(ylab)
        ax.set_title(name + ("" if tab.attrs["shaped"] else " (flat: neutral axis)"))
        return ax

    # ------------------------------------------------------------------ the training, in view
    def settings(self) -> pd.DataFrame:
        """Every setting: section, name, value and what it does."""
        rows = [("model", "seed", self.seed, "random seed of every step"),
                ("model", "model_space", ", ".join(self.model_space.families), "families of the Rashomon set")]
        for sec, obj in (("encoding", self.encoding), ("shapes", self.shapes_settings), ("nca", self.nca_settings),
                         ("knn", self.knn_settings), ("learn", self.learn)):
            for k, v in asdict(obj).items():
                rows.append((sec, k, v, MEANING.get((sec, k), "")))
        return pd.DataFrame(rows, columns=["section", "setting", "value", "meaning"])

    def stability(self) -> pd.DataFrame:
        """One row per accepted model of the Rashomon loop: its validation log-loss, how many models were
        competitive then, Δ and Δ̃ (stability of the distances), the Spearman correlation of the inputs' curve
        ranges (and of their SAGE importance, with `sage_in_loop`) against `window` models before, and every
        input's curve range (`range[<input>]`) and SAGE (`sage[<input>]`) at that point."""
        self._need_metric()
        rows = self.rashomon_.stability if self.rashomon_ is not None else getattr(self, "_loaded_stability", [])
        return pd.DataFrame(rows)

    def stopping(self) -> dict:
        """How the Rashomon loop ended: rule, reason, models accepted / tried / competitive, final Δ̃."""
        self._need_metric()
        return dict(self.rashomon_.stop if self.rashomon_ is not None else self.info_.get("stop", {}))

    def knn_curve(self) -> pd.DataFrame:
        """Leave-one-out log-loss of the vote for every (k, α) of the grids, on the current case base (or, for a
        loaded metric without one, on the case base it was fitted with). `best_alpha` marks the best α of each k
        and `used` the (k, α) in use."""
        self._need_metric()
        if self.case_base_ is not None:
            table, info = self.case_base_.loo_table, self.case_base_.fit_info
        else:
            info = self.info_.get("case_base_at_fit", {})
            table = info.get("loo_table", [])
        out = pd.DataFrame(table)
        if out.empty:
            return out
        out["best_alpha"] = out.groupby("k")["loo_logloss"].transform("min") == out["loo_logloss"]
        out["used"] = (out["k"] == info.get("k")) & (out["alpha"] == info.get("alpha"))
        return out

    def case_base_info(self) -> dict:
        """The case base in use: rows, class shares, k (chosen or given), α, leave-one-out log-loss and the
        out-of-support threshold."""
        self._need_case_base()
        return dict(self.case_base_.fit_info)

    def training_report(self) -> dict:
        """Everything about the training in one place: settings, stopping, stability trace, Rashomon record, NCA
        loss per epoch, the (k, α) leave-one-out curve and the case base."""
        self._need_metric()
        return {"settings": self.settings(), "stopping": self.stopping(), "stability": self.stability(),
                "rashomon": self.rashomon(), "nca_loss": list(self.info_.get("nca_history", [])),
                "knn": self.knn_curve(),
                "case_base": dict(self.case_base_.fit_info) if self.case_base_ is not None else
                {k: v for k, v in self.info_.get("case_base_at_fit", {}).items() if k != "loo_table"},
                "training": {k: v for k, v in self.info_.items() if k not in ("case_base_at_fit", "nca_history")}}

    def plot_stability(self, inputs: list[str] | None = None, axes=None):
        """Δ and Δ̃ over the accepted models (left) and the inputs' curve ranges (right; SAGE with
        `sage_in_loop`)."""
        import matplotlib.pyplot as plt

        st = self.stability()
        axes = axes if axes is not None else plt.subplots(1, 2, figsize=(10, 3.5))[1]
        if "delta" in st:
            axes[0].plot(st["n_accepted"], st["delta"], ".", alpha=0.5, label="Δ")
            axes[0].plot(st["n_accepted"], st["delta_smooth"], "-", label="Δ̃")
            tol = self.shapes_settings.tol_delta
            axes[0].axhline(tol, color="grey", lw=0.8, ls="--", label="tol_delta")
            axes[0].set_yscale("log")
        axes[0].set_xlabel("accepted models")
        axes[0].set_title(f"stability ({self.stopping().get('reason', '')})")
        axes[0].legend(fontsize=7)
        key = "sage" if any(c.startswith("sage[") for c in st.columns) else "range"
        cols = [c for c in st.columns if c.startswith(f"{key}[")]
        if inputs is not None:
            cols = [f"{key}[{n}]" for n in inputs]
        for c in cols:
            axes[1].plot(st["n_accepted"], st[c], label=c[len(key) + 1:-1])
        axes[1].set_xlabel("accepted models")
        axes[1].set_title("SAGE importance" if key == "sage" else "curve range (log-odds)")
        axes[1].legend(fontsize=6, ncol=2)
        return axes

    def plot_knn(self, ax=None):
        """Leave-one-out log-loss against k (best α of each k), with the k in use marked."""
        import matplotlib.pyplot as plt

        cur = self.knn_curve()
        ax = ax or plt.subplots(figsize=(5, 3.5))[1]
        best = cur[cur["best_alpha"]].drop_duplicates("k")
        ax.plot(best["k"], best["loo_logloss"], "o-")
        used = cur[cur["used"]]
        ax.plot(used["k"], used["loo_logloss"], "s", color="red", label="in use")
        ax.set_xscale("log")
        ax.set_xlabel("k")
        ax.set_ylabel("leave-one-out log-loss")
        ax.legend(fontsize=7)
        return ax

    def summary(self) -> str:
        if not self.is_fitted:
            return f"CEKNN (not fitted): {len(self.schema.inputs)} inputs → {self.schema.output}"
        imp = self.importance()
        st = self.stopping()
        lines = [f"CEKNN: {len(self.schema.inputs)} inputs → {self.schema.output} {self.schema.classes}",
                 f"metric: {self.enc_.n_features} variables, {int(imp['shaped'].sum())} with a shape",
                 f"Rashomon: stopping '{st.get('rule')}' → {st.get('reason')}; {st.get('n_accepted')} accepted of "
                 f"{st.get('n_attempts')} tried, {st.get('n_competitive')} competitive; final Δ̃ "
                 f"{_fmt(st.get('delta_smooth_final'))}" + (" (above tol_delta)" if st.get("plateau_above_tol") else "")]
        if self.case_base_ is not None:
            ci = self.case_base_.fit_info
            lines.append(f"case base: {ci['n_rows']} rows, class shares {ci['class_shares']}; k = {ci['k']}"
                         f"{' (chosen by leave-one-out)' if ci['k_selected'] else ''}, α = {ci['alpha']}, "
                         f"leave-one-out log-loss {ci['loo_logloss']:.4f}")
        else:
            lines.append("case base: none (set_case_base)")
        has_sage = "sage" in imp
        lines += [f"  {r.variable:<28} weight {r.weight:6.3f}  {'shape' if r.shaped else 'flat '}"
                  + (f"  SAGE {r.sage:6.3f}" if has_sage and pd.notna(r.sage) else "") for r in imp.itertuples()]
        return "\n".join(lines)

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict:
        """The settings and, once fitted, the metric. Never the case base nor the fitted models: only aggregates
        (see `Encoder.to_dict`), the consensus curves, the weights, the models' scores and their 10–90 % band."""
        d = {"format": FORMAT, "format_version": FORMAT_VERSION, "ceknn_version": __version__,
             "created": _dt.datetime.now().isoformat(timespec="seconds"),
             "schema": self.schema.to_dict(), "model_space": self.model_space.to_dict(),
             "settings": {"seed": self.seed, "encoding": asdict(self.encoding), "shapes": self.shapes_settings.to_dict(),
                          "nca": asdict(self.nca_settings), "knn": asdict(self.knn_settings),
                          "learn": asdict(self.learn)},
             "metric": None}
        if self.is_fitted:
            bands = {}
            for b in self.enc_.blocks:
                q10, q90 = self._band(b.name)
                if q10 is not None:
                    bands[b.name] = {"q10": np.round(q10, 6).tolist(), "q90": np.round(q90, 6).tolist()}
            rs = self.rashomon_
            d["metric"] = {
                "encoder": self.enc_.to_dict(),
                "weights": [float(x) for x in self.w_],
                "diagnostics": _clean(self._diagnostics()),
                "bands": bands,
                "rashomon": _clean(rs.attempts if rs is not None else getattr(self, "_loaded_attempts", [])),
                "groups": ({self.enc_.blocks[j].name: rs.group_names[int(g)] for j, g in enumerate(rs.groups)}
                           if rs is not None else getattr(self, "_loaded_groups", {})),
                "site_levels": rs.site_levels if rs is not None else getattr(self, "_loaded_site_levels", None),
                "sage": self.sage_,
                "stability": _clean(rs.stability if rs is not None else getattr(self, "_loaded_stability", [])),
                "training": _clean(self.info_),
            }
        return d

    def to_json(self, path: str | Path | None = None, indent: int = 1) -> str:
        s = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)
        if path is not None:
            Path(path).write_text(s, encoding="utf-8")
        return s

    def save(self, path: str | Path) -> None:
        self.to_json(path)

    @classmethod
    def from_dict(cls, d: dict, model_space: ModelSpace | None = None) -> "CEKNN":
        if d.get("format") != FORMAT:
            raise ValueError("not a CE-kNN metric file")
        if int(d.get("format_version", 0)) > FORMAT_VERSION:
            raise ValueError(f"metric format {d['format_version']} is newer than this library ({FORMAT_VERSION})")
        sch = Schema.from_dict(d["schema"])
        s = d["settings"]
        m = cls.__new__(cls)
        m.schema = sch
        m.model_space = model_space if model_space is not None else ModelSpace.from_dict(d["model_space"])
        m.encoding = EncodingSettings(**s["encoding"])
        m.shapes_settings = ShapeSettings(**s["shapes"])
        m.nca_settings = NCASettings(**s["nca"])
        m.knn_settings = KNNSettings(**s["knn"])
        m.learn = LearnSettings(**s["learn"])
        m.seed, m.n_jobs, m.verbose = int(s["seed"]), 4, False
        m.enc_ = m.w_ = m.rashomon_ = m.case_base_ = m._cb = None
        m.info_, m.sage_ = {}, None
        met = d.get("metric")
        if met is not None:
            m.enc_ = Encoder.from_dict(met["encoder"])
            m.w_ = np.asarray(met["weights"], dtype=float)
            m._loaded_diag = met.get("diagnostics") or {}
            m._loaded_bands = met.get("bands") or {}
            m._loaded_attempts = met.get("rashomon") or []
            m._loaded_groups = met.get("groups") or {}
            m._loaded_site_levels = met.get("site_levels")
            m._loaded_stability = met.get("stability") or []
            m.sage_ = met.get("sage")
            m.info_ = met.get("training") or {}
        return m

    @classmethod
    def from_json(cls, text: str, model_space: ModelSpace | None = None) -> "CEKNN":
        return cls.from_dict(json.loads(text), model_space)

    @classmethod
    def load(cls, path: str | Path, model_space: ModelSpace | None = None) -> "CEKNN":
        return cls.from_json(Path(path).read_text(encoding="utf-8"), model_space)

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "not fitted"
        return f"CEKNN({len(self.schema.inputs)} inputs → {self.schema.output!r}, {state})"


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def _site_constant(enc: Encoder, t: Table) -> list[str]:
    """Metric variables constant within every site: they carry no within-site information."""
    E = enc.transform(t)
    out = []
    for j in range(enc.n_features):
        cols = E[:, enc.dim_feature == j]
        if all(cols[t.sites == s].var(axis=0).sum() <= 1e-12 for s in np.unique(t.sites)):
            out.append(enc.blocks[j].name)
    return out


def _clean(o):
    """JSON-safe copy (numpy → Python, NaN / inf → None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o
