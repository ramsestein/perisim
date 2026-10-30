"""The effect space: each input lives on its shape function g_j, ALWAYS fitted on the training data.

Each metric variable j is a block of columns φ_j with ‖φ_j(a) − φ_j(b)‖² = δ_j(a, b)². Every block is divided by
s_j, with s_j² = E_pairs[δ_j²] = 2·Σ Var(φ_j), so the mean squared dissimilarity between random training pairs is 1
and the weight w_j keeps its meaning.

- Continuous / ordinal values are binned in ≤ `max_bins` training quantile bins of ≥ `min_bin` rows, with a
  constant extension beyond the observed range. Nominal categories with < `min_bin` rows are pooled ("other").
- Until a shape is installed, and whenever the consensus finds no shape distinguishable from flat, a variable keeps
  its NEUTRAL axis: rank (empirical CDF) for continuous / ordinal, one-hot for nominal.
- Binary inputs and missing indicators keep 0/1 (their shape is equivalent after normalisation). An unobserved binary
  takes the training prevalence; a missing continuous / ordinal value takes the training median, plus an indicator
  when more than `missing_threshold` of the rows miss it.

Port of PERISIM's encoder (geometry "shape"), with two changes so that an exported metric holds no individual value:
- the neutral rank is read from ≤ `ecdf_knots` inner quantiles (each step backed by ≥ `min_bin` rows), not from the
  sorted training values;
- a category not kept (rare in training, or never seen) goes to the "other" bin when there is one, so the list of
  rare categories is not needed. Without an "other" bin it has a neutral (zero) contribution.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .schema import Table, _numeric

MISSING = "__missing__"
OTHER = "__other__"
SHAPEABLE = ("continuous", "ordinal", "nominal")


@dataclass
class Block:
    name: str            # metric variable (may be "<input>__missing")
    source: str          # input column
    kind: str            # continuous | ordinal | binary | missing | nominal
    group: str           # an input and its missing indicator share a group
    params: dict = field(default_factory=dict)
    dims: int = 1
    scale: float = 1.0
    shape: np.ndarray | None = None      # consensus values (n_values × C); None = neutral axis
    neutral_scale: float = 1.0


class Encoder:
    def __init__(self, *, max_bins: int = 20, min_bin: int = 50, missing_threshold: float = 0.05,
                 max_categories: int = 50, ecdf_knots: int = 100):
        self.max_bins = int(max_bins)
        self.min_bin = int(min_bin)
        self.missing_threshold = float(missing_threshold)
        self.max_categories = int(max_categories)
        self.ecdf_knots = int(ecdf_knots)
        self.blocks: list[Block] = []
        self.n_classes = 2

    # ------------------------------------------------------------------ fitting
    def fit(self, t: Table, y: np.ndarray, n_classes: int) -> "Encoder":
        self.n_classes = int(n_classes)
        self.blocks = []
        for v in t.variables:
            if v.type in ("continuous", "ordinal"):
                x = t.num[v.name]
                miss = ~np.isfinite(x)
                if miss.all():
                    continue
                obs = x[~miss]
                med = float(np.median(obs))
                p = {"median": med}
                if v.type == "continuous":
                    xi = np.where(miss, med, x)
                    mu, sd = float(xi.mean()), float(xi.std())
                    if sd < 1e-12:
                        continue
                    p.update(mean=mu, sd=sd)
                else:
                    if v.levels is None:
                        lo, hi = float(obs.min()), float(obs.max())
                    elif _numeric(v.levels):
                        lo, hi = float(min(float(x) for x in v.levels)), float(max(float(x) for x in v.levels))
                    else:
                        lo, hi = 0.0, float(len(v.levels) - 1)
                    if hi <= lo:
                        continue
                    p.update(lo=lo, hi=hi)
                nb = int(max(1, min(self.max_bins, obs.size // max(self.min_bin, 1))))
                edges = np.unique(np.quantile(obs, np.linspace(0, 1, nb + 1)))
                if edges.size < 2:
                    continue
                p.update(edges=edges, ecdf=_ecdf_knots(obs, self.min_bin, self.ecdf_knots))
                self.blocks.append(Block(v.name, v.name, v.type, v.name, p))
                if self.missing_threshold < miss.mean() < 1:
                    self.blocks.append(Block(f"{v.name}__missing", v.name, "missing", v.name))
            elif v.type == "binary":
                x = t.num[v.name]
                miss = ~np.isfinite(x)
                obs = int((~miss).sum())
                cnt = float((x[~miss] > 0.5).sum())
                if obs == 0 or cnt == 0 or cnt == obs:
                    continue
                self.blocks.append(Block(v.name, v.name, "binary", v.name, {"prevalence": cnt / obs}))
                if miss.mean() > self.missing_threshold:
                    self.blocks.append(Block(f"{v.name}__missing", v.name, "missing", v.name))
            elif v.type == "nominal":
                vals = _norm_cat(t.cat[v.name])
                cats, counts = np.unique(vals, return_counts=True)
                order = np.argsort(-counts, kind="stable")
                keep = [str(c) for c, k in zip(cats[order], counts[order]) if k >= self.min_bin][: self.max_categories]
                if len(keep) < 2 and not (len(keep) == 1 and (vals != keep[0]).any()):
                    continue
                vv = np.where(np.isin(vals, keep), vals, OTHER)
                values = list(keep) + ([OTHER] if (vv == OTHER).any() else [])
                vc = np.array([(vv == c).sum() for c in values], dtype=float)
                p = {"keep": keep, "onehot": sorted(set(keep) | {OTHER}), "values": values, "value_counts": vc,
                     "ref": int(np.argmax(vc))}
                self.blocks.append(Block(v.name, v.name, "nominal", v.name, p, dims=len(p["onehot"])))
        kept = []
        for b in self.blocks:
            if b.kind in ("continuous", "ordinal"):
                b.params["value_counts"] = np.bincount(self._bin_index(b, t), minlength=self.n_values(b)).astype(float)
            sj = _scale(self._metric_block(b, t))
            if sj > 1e-12:
                b.scale = b.neutral_scale = sj
                kept.append(b)
        self.blocks = kept
        self._index()
        return self

    def drop(self, names: list[str]) -> None:
        self.blocks = [b for b in self.blocks if b.name not in set(names)]
        self._index()

    def _index(self) -> None:
        self.feature_names = [b.name for b in self.blocks]
        self.feature_groups = [b.group for b in self.blocks]
        self.dim_feature = (np.concatenate([np.full(b.dims, j) for j, b in enumerate(self.blocks)]).astype(np.int64)
                            if self.blocks else np.zeros(0, dtype=np.int64))
        self.n_features = len(self.blocks)
        self.n_dims = int(self.dim_feature.size)

    def feature_index(self) -> dict[str, int]:
        return {n: j for j, n in enumerate(self.feature_names)}

    # ------------------------------------------------------------------ shape values
    def n_values(self, b: Block) -> int:
        if b.kind in ("continuous", "ordinal"):
            return int(b.params["edges"].size - 1)
        return len(b.params["values"])

    def _bin_index(self, b: Block, t: Table) -> np.ndarray:
        x = t.num[b.source]
        x = np.where(np.isfinite(x), x, b.params["median"])
        e = b.params["edges"]
        return np.clip(np.searchsorted(e, x, side="right") - 1, 0, e.size - 2)

    def value_index(self, b: Block, t: Table) -> np.ndarray:
        """Index of each row's value in the shape table of block b (−1 = no value: neutral contribution)."""
        if b.kind in ("continuous", "ordinal"):
            return self._bin_index(b, t)
        v = _norm_cat(t.cat[b.source])
        pos = {c: i for i, c in enumerate(b.params["values"])}
        oth = pos.get(OTHER, -1)
        return np.array([pos.get(x, oth) for x in v], dtype=np.int64)

    def shapeable_blocks(self) -> list[int]:
        return [j for j, b in enumerate(self.blocks) if b.kind in SHAPEABLE]

    def model_column_value(self, b: Block, raw: np.ndarray) -> np.ndarray:
        """Model-matrix value of a continuous / ordinal block for raw values (used to move a value in ALE)."""
        p = b.params
        if b.kind == "continuous":
            return (raw - p["mean"]) / p["sd"]
        return (np.clip(raw, p["lo"], p["hi"]) - p["lo"]) / (p["hi"] - p["lo"])

    def model_onehot(self, b: Block, value_idx: int) -> np.ndarray:
        """Model-matrix columns of a nominal block set to one shape value."""
        c = b.params["values"][value_idx]
        return (np.array(b.params["onehot"], dtype=object) == c).astype(float)

    def with_shapes(self, shapes: dict[int, np.ndarray | None], t: Table) -> "Encoder":
        """Copy with the consensus shapes installed (None / absent = neutral axis) and the scales recomputed on the
        training data `t`. Block indices never change, so weights and groups stay aligned."""
        enc = copy.copy(self)
        enc.blocks = [copy.copy(b) for b in self.blocks]
        for j, b in enumerate(enc.blocks):
            if b.kind not in SHAPEABLE:
                continue
            g = shapes.get(j)
            b.shape = None if g is None else np.asarray(g, dtype=np.float64)
            b.dims = b.shape.shape[1] if b.shape is not None else self._neutral_dims(b)
            sj = _scale(enc._metric_block(b, t)) if b.shape is not None else b.neutral_scale
            if b.shape is not None and sj <= 1e-12:         # constant on the training data → neutral axis
                b.shape, b.dims, sj = None, self._neutral_dims(b), b.neutral_scale
            b.scale = sj
        enc._index()
        return enc

    def _neutral_dims(self, b: Block) -> int:
        return len(b.params["onehot"]) if b.kind == "nominal" else 1

    # ------------------------------------------------------------------ transformation
    def _model_block(self, b: Block, t: Table) -> np.ndarray:
        p = b.params
        if b.kind == "continuous":
            x = t.num[b.source]
            x = np.where(np.isfinite(x), x, p["median"])
            return ((x - p["mean"]) / p["sd"])[:, None]
        if b.kind == "ordinal":
            x = t.num[b.source]
            x = np.where(np.isfinite(x), x, p["median"])
            return ((np.clip(x, p["lo"], p["hi"]) - p["lo"]) / (p["hi"] - p["lo"]))[:, None]
        if b.kind == "missing":
            return (~np.isfinite(t.num[b.source])).astype(float)[:, None]
        if b.kind == "binary":
            x = t.num[b.source]
            return np.where(np.isfinite(x), (x > 0.5).astype(float), p["prevalence"])[:, None]
        if b.kind == "nominal":
            v = _norm_cat(t.cat[b.source])
            vv = np.where(np.isin(v, p["keep"]), v, OTHER)
            return (vv[:, None] == np.array(p["onehot"], dtype=object)[None, :]).astype(float)
        raise ValueError(b.kind)

    def _neutral_block(self, b: Block, t: Table) -> np.ndarray:
        if b.kind in ("continuous", "ordinal"):            # rank: empirical CDF of the training values
            x = t.num[b.source]
            x = np.where(np.isfinite(x), x, b.params["median"])
            knots, levels = b.params["ecdf"]
            return np.interp(x, knots, levels)[:, None]
        if b.kind == "nominal":
            return self._model_block(b, t) / np.sqrt(2.0)
        return self._model_block(b, t)

    def _metric_block(self, b: Block, t: Table) -> np.ndarray:
        if b.shape is not None:
            idx = self.value_index(b, t)
            out = np.zeros((len(idx), b.shape.shape[1]))
            ok = idx >= 0
            out[ok] = b.shape[idx[ok]]
            return out
        return self._neutral_block(b, t)

    def transform(self, t: Table) -> np.ndarray:
        """Normalised embedding in the effect space (n × D, float32). Unweighted: multiply by √w per column."""
        if not self.blocks:
            return np.zeros((len(t), 0), dtype=np.float32)
        return np.hstack([self._metric_block(b, t) / b.scale for b in self.blocks]).astype(np.float32)

    def transform_model(self, t: Table) -> tuple[np.ndarray, np.ndarray]:
        """Matrix for the models (nominal as one-hot) and the metric variable owning each column."""
        cols, owner = [], []
        for j, b in enumerate(self.blocks):
            e = self._model_block(b, t)
            cols.append(e)
            owner.append(np.full(e.shape[1], j))
        return np.hstack(cols).astype(np.float32), np.concatenate(owner).astype(np.int64)

    def weights_dims(self, w: np.ndarray) -> np.ndarray:
        return np.sqrt(np.asarray(w, dtype=np.float64))[self.dim_feature].astype(np.float32)

    # ------------------------------------------------------------------ reading
    def value_labels(self, b: Block) -> list[str]:
        if b.kind in ("continuous", "ordinal"):
            e = b.params["edges"]
            out = []
            for i in range(e.size - 1):
                lo, hi = e[i], e[i + 1]
                if not np.isfinite(lo):
                    out.append(f"< {hi:.4g}")
                elif not np.isfinite(hi):
                    out.append(f"≥ {lo:.4g}")
                else:
                    out.append(f"[{lo:.4g}, {hi:.4g}{']' if i == e.size - 2 else ')'}")
            return out
        return [str(c) for c in b.params["values"]]

    # ------------------------------------------------------------------ export
    def to_dict(self) -> dict:
        """Aggregates only: medians, means, SDs, inner bin edges (the outer two are the training minimum and
        maximum, so they are left out), inner quantiles, categories with ≥ `min_bin` rows, counts rounded to 10."""
        blocks = []
        for b in self.blocks:
            p = {}
            for k, v in b.params.items():
                if k == "edges":
                    e = np.asarray(v, dtype=float).copy()
                    e[0], e[-1] = -np.inf, np.inf
                    p[k] = [None if not np.isfinite(x) else float(x) for x in e]
                elif k == "ecdf":
                    p[k] = [[float(x) for x in v[0]], [float(x) for x in v[1]]]
                elif k == "value_counts":
                    p[k] = [int(round(float(x), -1)) for x in v]
                elif isinstance(v, np.ndarray):
                    p[k] = v.tolist()
                elif isinstance(v, np.generic):
                    p[k] = v.item()
                else:
                    p[k] = v
            blocks.append({"name": b.name, "source": b.source, "kind": b.kind, "group": b.group, "params": p,
                           "dims": int(b.dims), "scale": float(b.scale), "neutral_scale": float(b.neutral_scale),
                           "shape": None if b.shape is None else np.round(b.shape, 6).tolist()})
        return {"settings": {"max_bins": self.max_bins, "min_bin": self.min_bin,
                             "missing_threshold": self.missing_threshold, "max_categories": self.max_categories,
                             "ecdf_knots": self.ecdf_knots},
                "n_classes": self.n_classes, "blocks": blocks}

    @classmethod
    def from_dict(cls, d: dict) -> "Encoder":
        enc = cls(**d["settings"])
        enc.n_classes = int(d["n_classes"])
        for bd in d["blocks"]:
            p = dict(bd["params"])
            if "edges" in p:
                p["edges"] = np.array([np.nan if x is None else x for x in p["edges"]], dtype=float)
                p["edges"][0], p["edges"][-1] = -np.inf, np.inf
            if "ecdf" in p:
                p["ecdf"] = (np.asarray(p["ecdf"][0], dtype=float), np.asarray(p["ecdf"][1], dtype=float))
            if "value_counts" in p:
                p["value_counts"] = np.asarray(p["value_counts"], dtype=float)
            enc.blocks.append(Block(bd["name"], bd["source"], bd["kind"], bd["group"], p, int(bd["dims"]),
                                    float(bd["scale"]), None if bd["shape"] is None else np.asarray(bd["shape"], float),
                                    float(bd["neutral_scale"])))
        enc._index()
        return enc


def _scale(e: np.ndarray) -> float:
    return float(np.sqrt(2.0 * e.var(axis=0).sum()))


def _ecdf_knots(obs: np.ndarray, min_bin: int, max_knots: int) -> tuple[np.ndarray, np.ndarray]:
    """Inner quantiles and their levels: the neutral rank is the ECDF interpolated between them."""
    m = int(np.clip(obs.size // max(min_bin, 1), 4, max(max_knots, 4)))
    levels = np.arange(1, m) / m
    knots = np.quantile(obs, levels)
    knots, first = np.unique(knots, return_index=True)
    return knots, levels[first]


def _norm_cat(v: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=object).copy()
    bad = np.array([x is None or (isinstance(x, float) and x != x) or x == "" for x in out], dtype=bool)
    out[bad] = MISSING
    return out.astype(str).astype(object)


# ---------------------------------------------------------------------------- groups for the feature dropout
def build_groups(enc: Encoder, E: np.ndarray, corr_threshold: float, max_rows: int = 20000,
                 seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Group of each metric variable for the models' feature dropout and for SAGE: an input with its missing
    indicator, then clusters of inputs with |ρ| > `corr_threshold` (union of groups). `E` must be the embedding
    BEFORE shapes are installed (one column per single-dimension variable)."""
    F = enc.n_features
    labels = list(enc.feature_groups)
    parent = {g: g for g in labels}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    single = [j for j in range(F) if enc.blocks[j].dims == 1]
    if len(single) > 1 and E.shape[0] > 2:
        rng = np.random.default_rng(seed)
        rows = rng.choice(E.shape[0], size=min(max_rows, E.shape[0]), replace=False)
        cols = [int(np.where(enc.dim_feature == j)[0][0]) for j in single]
        M = E[np.ix_(rows, cols)].astype(np.float64)
        ok = M.std(axis=0) > 1e-12
        C = np.zeros((len(single), len(single)))
        if ok.sum() > 1:
            C[np.ix_(ok, ok)] = np.corrcoef(M[:, ok], rowvar=False)
        for a, b in np.argwhere(np.triu(np.abs(C) > corr_threshold, k=1)):
            ga, gb = find(labels[single[a]]), find(labels[single[b]])
            if ga != gb:
                parent[gb] = ga
    final = [find(g) for g in labels]
    names = sorted(set(final))
    idx = {g: i for i, g in enumerate(names)}
    return np.array([idx[g] for g in final], dtype=np.int64), names
