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
from .sage import aggregate, normalize, sage_permutation
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
    k: int = 50
    alpha_grid: list = field(default_factory=lambda: [0.1, 0.5, 1, 2, 5, 10])
    alpha_queries: int = 20000         # rows of the leave-one-out that picks α
    ood_percentile: float = 99.0
    distinct_groups: bool = True       # each group (patient) enters a neighbourhood once
    fetch_factor: int = 6


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
        self.sage_ = None
        self.info_ = {"n_rows": int(len(t)), "n_learn": int(len(rows)),
                      "class_shares": (np.bincount(yl, minlength=sch.n_classes) / len(yl)).round(4).tolist(),
                      "learn_per_site": ({str(v): int((tl.sites == v).sum()) for v in np.unique(tl.sites)}
                                         if tl.sites is not None else None),
                      "dropped_site_constant": dropped, "nca_history": [round(float(x), 6) for x in nca.history],
                      "seconds_rashomon": round(rs.seconds, 2), "n_attempts": len(rs.attempts),
                      "n_accepted": len(rs.models), "n_competitive": int(rs.competitive.sum()),
                      "fitted": _dt.datetime.now().isoformat(timespec="seconds")}
        return self._set_case_base(df, t, y)

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
        cb.fit_alpha(ks.alpha_grid, n_queries=ks.alpha_queries, ood_percentile=ks.ood_percentile, seed=self.seed)
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

    def predict_proba(self, df: pd.DataFrame, k: int | None = None) -> np.ndarray:
        """Class probabilities (n × classes, in the order of `classes`): the smoothed vote of the k neighbours."""
        D, I, valid = self._search(df, k)
        return self.case_base_.probs(I, valid, k)

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
        """Mean of other numeric columns of the case base over each query's neighbours (what the neighbours
        carry: later outcomes, treatments…)."""
        D, I, valid = self._search(df, k)
        kk = I.shape[1] if k is None else min(int(k), I.shape[1])
        frame = self._cb["frame"]
        out = {}
        for c in columns:
            v = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)[I[:, :kk]]
            v = np.where(valid[:, :kk], v, np.nan)
            out[c] = np.nanmean(v, axis=1)
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
            model = _InputsOnly(m, G_in, rs.n_groups)
            r = sage_permutation(model, X, y, G_in, rng, max_passes=max_passes)
            v, s = normalize(r["values"], r["se"])
            vals.append(v)
            ses.append(s)
        agg = aggregate(np.array(vals), np.array(ses))
        per = {}
        for g in range(G_in):
            members = [self.enc_.blocks[j].name for j in np.where(rs.groups == g)[0]]
            for n in members:
                per[n] = float(agg[g] / len(members))
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

    def summary(self) -> str:
        if not self.is_fitted:
            return f"CEKNN (not fitted): {len(self.schema.inputs)} inputs → {self.schema.output}"
        imp = self.importance()
        lines = [f"CEKNN: {len(self.schema.inputs)} inputs → {self.schema.output} {self.schema.classes}",
                 f"metric: {self.enc_.n_features} variables, {int(imp['shaped'].sum())} with a shape; "
                 f"k = {self.knn_settings.k}",
                 f"case base: {len(self.case_base_) if self.case_base_ is not None else 'none'} rows"]
        lines += [f"  {r.variable:<28} weight {r.weight:6.3f}  {'shape' if r.shaped else 'flat '}"
                  for r in imp.itertuples()]
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


class _InputsOnly:
    """A surrogate seen by SAGE over the input groups only (the site's nuisance group always present)."""

    def __init__(self, model, n_input_groups: int, n_groups: int):
        self.model, self.Gi, self.G = model, n_input_groups, n_groups

    def predict_proba(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.G > self.Gi:
            mask = np.hstack([mask, np.ones((mask.shape[0], self.G - self.Gi), dtype=bool)])
        return self.model.predict_proba(X, mask)


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
