"""δ_j encoding (§5.2), ALWAYS fitted on the training fold.

Each metric variable j is represented as a block of columns φ_j such that ‖φ_j(a) − φ_j(b)‖² = δ_j(a,b)².
Normalization (§5.1): each block is divided by s_j with s_j² = E_pairs[δ_j²] = 2·Σ Var(φ_j), so that the
mean squared dissimilarity between random training pairs is 1 and w_j keeps its meaning.

Two geometries:

- **shape (v0.5, decided 2026-10-24).** Each continuous, ordinal, nominal or hierarchical variable lives on its
  SHAPE FUNCTION g_j: the consensus, across the accepted models of the importance loop, of how its values move the
  prediction (centred log-odds, one component per class; importance/shapes.py). Continuous / ordinal values are
  binned in ≤ `max_bins` training quantile bins of ≥ `min_bin` patients, with a constant extension beyond the
  observed range; nominal categories and ICD-10-PCS codes with < `min_bin` patients are pooled ("other"; PCS codes
  with their parent code first). Until a shape is installed — and whenever the consensus finds no shape
  distinguishable from flat — a variable uses its NEUTRAL axis: rank (empirical CDF) for continuous / ordinal,
  one-hot for nominal / PCS. Binary variables and missingness indicators keep 0/1 (their shape is equivalent after
  normalization); an unobserved binary takes the training prevalence. Missing values keep median imputation +
  indicator.
- **hvdm (v0.4, comparator).** Continuous: standardized value; ordinal: range scaled to [0, 1]; nominal: smoothed
  VDM, φ = p_c = (n_c·P̂(y|c) + m·P(y)) / (n_c + m), rare categories pooled, unseen ones → P(y) (AUDIT B7);
  hierarchical: one-hot/√2 with explicit "other" and "missing" dimensions (AUDIT B6).

`transform_model` gives the version for the surrogate models and comparators (no target information).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from perisim.features.table import FeatureTable

MISSING = "__missing__"
OTHER = "__other__"
SHAPEABLE = ("continuous", "ordinal", "nominal", "hier")


@dataclass
class _Block:
    name: str            # metric variable (may be "<var>__missing")
    source: str          # source column
    kind: str            # continuous | ordinal | binary | missing | nominal | hier
    group: str
    block: str
    params: dict = field(default_factory=dict)
    dims: int = 1
    scale: float = 1.0
    shape: np.ndarray | None = None      # geometry "shape": consensus values (n_values × C), None = neutral axis
    neutral_scale: float = 1.0


class Encoder:
    def __init__(self, *, m: float = 5.0, missing_threshold: float = 0.05, hier_min_count: int = 10,
                 hier_max_levels: int = 200, nominal_min_count: int = 10, nominal_max_categories: int = 50,
                 dx_min_prevalence: float = 0.005, med_min_count: int = 10, geometry: str = "hvdm",
                 max_bins: int = 20, min_bin: int = 50):
        self.m = float(m)
        self.missing_threshold = missing_threshold
        self.hier_min_count = hier_min_count
        self.hier_max_levels = hier_max_levels
        self.nominal_min_count = nominal_min_count
        self.nominal_max_categories = nominal_max_categories
        self.dx_min_prevalence = dx_min_prevalence
        self.med_min_count = med_min_count
        if geometry not in ("shape", "hvdm"):
            raise ValueError(f"unknown geometry {geometry!r}")
        self.geometry = geometry
        self.max_bins = int(max_bins)
        self.min_bin = int(min_bin)
        self.blocks: list[_Block] = []

    @classmethod
    def from_config(cls, cfg, m: float | None = None, geometry: str | None = None) -> "Encoder":
        f = cfg.experiment["features"]
        c = cfg.experiment["importance"].get("curves") or {}
        return cls(m=m if m is not None else cfg.experiment["metric"]["vdm_m_grid"][1],
                   missing_threshold=f["missing_indicator_threshold"], hier_min_count=f["hier_min_count"],
                   hier_max_levels=f["hier_max_levels"], nominal_min_count=f["nominal_min_count"],
                   nominal_max_categories=f["nominal_max_categories"], dx_min_prevalence=f["dx_min_prevalence"],
                   geometry=geometry or cfg.experiment["metric"].get("geometry", "shape"),
                   max_bins=int(c.get("max_bins", 20)), min_bin=int(c.get("min_bin", 50)))

    # ------------------------------------------------------------------ fitting
    def fit(self, ft: FeatureTable, y: np.ndarray, n_classes: int, y_aux: np.ndarray | None = None) -> "Encoder":
        y = np.asarray(y, dtype=np.int64)
        self.n_classes = n_classes
        prior = np.bincount(y, minlength=n_classes).astype(float)
        prior = (prior + 1e-9) / (prior.sum() + 1e-9 * n_classes)
        aux_prior = np.nanmean(y_aux, axis=0) if y_aux is not None else None
        shape = self.geometry == "shape"
        n = len(ft)
        self.blocks = []
        for s in ft.specs:
            if s.type in ("continuous", "ordinal"):
                x = ft.num[s.name]
                miss = ~np.isfinite(x)
                if miss.all():
                    continue
                obs = x[~miss]
                med = float(np.median(obs))
                p = {"median": med}
                if s.type == "continuous":
                    xi = np.where(miss, med, x)
                    mu, sd = float(xi.mean()), float(xi.std())
                    if sd < 1e-12:
                        continue
                    p.update(mean=mu, sd=sd)
                else:
                    lv = s.levels or [float(np.nanmin(x)), float(np.nanmax(x))]
                    lo, hi = float(min(lv)), float(max(lv))
                    if hi <= lo:
                        continue
                    p.update(lo=lo, hi=hi)
                if shape:
                    nb = int(max(1, min(self.max_bins, obs.size // max(self.min_bin, 1))))
                    edges = np.unique(np.quantile(obs, np.linspace(0, 1, nb + 1)))
                    if edges.size < 2:
                        continue
                    p.update(edges=edges, sorted=np.sort(obs))
                self.blocks.append(_Block(s.name, s.name, s.type, s.group, s.block, p))
                if miss.mean() > self.missing_threshold and miss.mean() < 1:
                    self.blocks.append(_Block(f"{s.name}__missing", s.name, "missing", s.group, s.block))
            elif s.type == "binary":
                x = ft.num[s.name]
                miss = ~np.isfinite(x)
                obs = int((~miss).sum())
                cnt = float((x[~miss] > 0.5).sum())
                if obs == 0 or cnt == 0 or cnt == obs:
                    continue
                if s.block == "diagnoses" and cnt / obs < self.dx_min_prevalence:
                    continue
                if s.block == "medication" and cnt < self.med_min_count:
                    continue
                self.blocks.append(_Block(s.name, s.name, "binary", s.group, s.block, {"prevalence": cnt / obs}))
                if miss.mean() > self.missing_threshold:
                    self.blocks.append(_Block(f"{s.name}__missing", s.name, "missing", s.group, s.block))
            elif s.type == "nominal":
                v = _norm_cat(ft.cat[s.name])
                cats, counts = np.unique(v, return_counts=True)
                order = np.argsort(-counts)
                min_count = self.min_bin if shape else self.nominal_min_count
                keep = [c for c, k in zip(cats[order], counts[order]) if k >= min_count][: self.nominal_max_categories]
                if len(keep) < 2 and not (len(keep) == 1 and (v != keep[0]).any()):
                    continue
                vv = np.where(np.isin(v, keep), v, OTHER)
                seen = sorted(set(cats.tolist()))
                p = {"keep": keep, "seen": seen, "onehot": sorted(set(keep) | {OTHER})}
                if shape:
                    vals = list(keep) + ([OTHER] if (vv == OTHER).any() else [])
                    p.update(values=vals, value_counts=np.array([(vv == c).sum() for c in vals], dtype=float),
                             ref=int(np.argmax([(vv == c).sum() for c in vals])))
                    self.blocks.append(_Block(s.name, s.name, "nominal", s.group, s.block, p, dims=len(p["onehot"])))
                    continue
                table = {}
                for c in np.unique(vv):
                    sel = vv == c
                    nc = sel.sum()
                    phat = np.bincount(y[sel], minlength=n_classes) / max(nc, 1)
                    pc = (nc * phat + self.m * prior) / (nc + self.m)
                    if y_aux is not None:
                        pa = (nc * np.nanmean(y_aux[sel], axis=0) + self.m * aux_prior) / (nc + self.m)
                        pc = np.concatenate([pc, pa])
                    table[c] = (pc, int(nc))
                unseen = prior if y_aux is None else np.concatenate([prior, aux_prior])
                p.update(table=table, unseen=unseen)
                self.blocks.append(_Block(s.name, s.name, "nominal", s.group, s.block, p, dims=len(unseen)))
            elif s.type == "hier":
                v = ft.cat[s.name]
                vals = np.array([x for x in v if x is not None and x == x and x != ""], dtype=object)
                if vals.size == 0:
                    continue
                cats, counts = np.unique(vals, return_counts=True)
                order = np.argsort(-counts)
                min_count = self.min_bin if shape else self.hier_min_count
                keep = [c for c, k in zip(cats[order], counts[order]) if k >= min_count][: self.hier_max_levels]
                vmap = {c: c for c in keep}
                if shape:   # rare codes pooled with their parent code when that pool is large enough (gap 5)
                    rare = [(c, k) for c, k in zip(cats, counts) if c not in vmap]
                    pool: dict[str, int] = {}
                    for c, k in rare:
                        pool[str(c)[:-1] + "*"] = pool.get(str(c)[:-1] + "*", 0) + int(k)
                    for c, _ in rare:
                        par = str(c)[:-1] + "*"
                        if len(str(c)) > 1 and pool[par] >= min_count:
                            vmap[c] = par
                values = list(dict.fromkeys(vmap.values()))
                if not values:
                    continue
                p = {"codes": values, "vmap": vmap}
                if shape:
                    idx = self._hier_index(p, np.asarray(v, dtype=object))
                    vc = np.bincount(idx, minlength=len(values) + 2).astype(float)
                    p.update(value_counts=vc, ref=int(np.argmax(vc[:len(values)])))
                self.blocks.append(_Block(s.name, s.name, "hier", s.group, s.block, p, dims=len(values) + 2))
        # scales s_j: mean square of δ_j between random pairs = 1 (neutral axes until shapes are installed)
        kept = []
        for b in self.blocks:
            if shape and b.kind in ("continuous", "ordinal", "nominal"):
                b.params["value_counts"] = b.params.get("value_counts")
                if b.kind != "nominal":
                    b.params["value_counts"] = np.bincount(self._bin_index(b, ft), minlength=self._n_values(b)) \
                        .astype(float)
            sj = _scale(self._metric_block(b, ft))
            if sj > 1e-12:
                b.scale = sj
                b.neutral_scale = sj
                kept.append(b)
        self.blocks = kept
        self._index()
        return self

    def _index(self) -> None:
        self.feature_names = [b.name for b in self.blocks]
        self.feature_groups = [b.group for b in self.blocks]
        self.feature_blocks = [b.block for b in self.blocks]
        self.dim_feature = np.concatenate([np.full(b.dims, j) for j, b in enumerate(self.blocks)]).astype(np.int64) \
            if self.blocks else np.zeros(0, dtype=np.int64)
        self.n_features = len(self.blocks)
        self.n_dims = int(self.dim_feature.size)

    # ------------------------------------------------------------------ shape values
    @staticmethod
    def _hier_index(p: dict, v: np.ndarray) -> np.ndarray:
        """Value index of a PCS level: kept / pooled codes 0..K−1, K = other, K+1 = missing."""
        pos = {c: i for i, c in enumerate(p["codes"])}
        K = len(p["codes"])
        out = np.empty(len(v), dtype=np.int64)
        for i, x in enumerate(v):
            if x is None or x != x or x == "":
                out[i] = K + 1
            else:
                out[i] = pos.get(p["vmap"].get(x), K)
        return out

    def _n_values(self, b: _Block) -> int:
        if b.kind in ("continuous", "ordinal"):
            return int(b.params["edges"].size - 1)
        if b.kind == "nominal":
            return len(b.params["values"])
        return len(b.params["codes"]) + 2

    def _bin_index(self, b: _Block, ft: FeatureTable) -> np.ndarray:
        x = ft.num[b.source]
        x = np.where(np.isfinite(x), x, b.params["median"])
        e = b.params["edges"]
        return np.clip(np.searchsorted(e, x, side="right") - 1, 0, e.size - 2)

    def value_index(self, b: _Block, ft: FeatureTable) -> np.ndarray:
        """Index of each patient's value in the shape table of block b (−1 = never seen in training)."""
        if b.kind in ("continuous", "ordinal"):
            return self._bin_index(b, ft)
        if b.kind == "nominal":
            v = _norm_cat(ft.cat[b.source])
            pos = {c: i for i, c in enumerate(b.params["values"])}
            seen = set(b.params["seen"])
            oth = pos.get(OTHER, -1)
            return np.array([pos[x] if x in pos else (oth if x in seen else -1) for x in v], dtype=np.int64)
        return self._hier_index(b.params, np.asarray(ft.cat[b.source], dtype=object))

    def shapeable_blocks(self) -> list[int]:
        return [j for j, b in enumerate(self.blocks) if self.geometry == "shape" and b.kind in SHAPEABLE]

    def model_column_value(self, b: _Block, raw: np.ndarray) -> np.ndarray:
        """Model-matrix value of a continuous / ordinal block for raw values (used to move a value in ALE)."""
        p = b.params
        if b.kind == "continuous":
            return (raw - p["mean"]) / p["sd"]
        return (np.clip(raw, p["lo"], p["hi"]) - p["lo"]) / (p["hi"] - p["lo"])

    def model_onehot(self, b: _Block, value_idx: int) -> np.ndarray:
        """Model-matrix columns of a nominal / hierarchical block set to one shape value."""
        if b.kind == "nominal":
            c = b.params["values"][value_idx]
            return (np.array(b.params["onehot"], dtype=object) == c).astype(float)
        out = np.zeros(b.dims)
        out[value_idx] = 1.0
        return out / np.sqrt(2.0)

    def with_shapes(self, shapes: dict[int, np.ndarray | None], ft: FeatureTable) -> "Encoder":
        """Copy with the consensus shapes installed (None / absent = neutral axis) and the scales recomputed on the
        training data `ft`. Block indices never change, so weights and groups stay aligned."""
        enc = copy.copy(self)
        enc.blocks = [copy.copy(b) for b in self.blocks]
        for j, b in enumerate(enc.blocks):
            if b.kind not in SHAPEABLE or self.geometry != "shape":
                continue
            g = shapes.get(j)
            b.shape = None if g is None else np.asarray(g, dtype=np.float64)
            b.dims = b.shape.shape[1] if b.shape is not None else self._neutral_dims(b)
            sj = _scale(enc._metric_block(b, ft)) if b.shape is not None else b.neutral_scale
            if b.shape is not None and sj <= 1e-12:         # constant on the training data → neutral axis
                b.shape, b.dims, sj = None, self._neutral_dims(b), b.neutral_scale
            b.scale = sj
        enc._index()
        return enc

    def _neutral_dims(self, b: _Block) -> int:
        if b.kind == "nominal":
            return len(b.params["onehot"])
        if b.kind == "hier":
            return len(b.params["codes"]) + 2
        return 1

    # ------------------------------------------------------------------ transformation
    def _model_block(self, b: _Block, ft: FeatureTable) -> np.ndarray:
        p = b.params
        if b.kind == "continuous":
            x = ft.num[b.source]
            x = np.where(np.isfinite(x), x, p["median"])
            return ((x - p["mean"]) / p["sd"])[:, None]
        if b.kind == "ordinal":
            x = ft.num[b.source]
            x = np.where(np.isfinite(x), x, p["median"])
            return ((np.clip(x, p["lo"], p["hi"]) - p["lo"]) / (p["hi"] - p["lo"]))[:, None]
        if b.kind == "missing":
            return (~np.isfinite(ft.num[b.source])).astype(float)[:, None]
        if b.kind == "binary":
            x = ft.num[b.source]
            return np.where(np.isfinite(x), (x > 0.5).astype(float), p["prevalence"])[:, None]
        if b.kind == "nominal":
            v = _norm_cat(ft.cat[b.source])
            vv = np.where(np.isin(v, p["keep"]), v, OTHER)
            return (vv[:, None] == np.array(p["onehot"], dtype=object)[None, :]).astype(float)
        if b.kind == "hier":
            return self._hier_onehot(b, ft)
        raise ValueError(b.kind)

    def _hier_onehot(self, b: _Block, ft: FeatureTable) -> np.ndarray:
        v = np.asarray(ft.cat[b.source], dtype=object)
        if self.geometry == "shape":
            idx = self._hier_index(b.params, v)
            out = np.zeros((len(v), len(b.params["codes"]) + 2))
            out[np.arange(len(v)), idx] = 1.0
            return out / np.sqrt(2.0)
        codes = np.array(b.params["codes"], dtype=object)
        hit = (v[:, None] == codes[None, :]).astype(float)
        missing = np.array([x is None or x != x or x == "" for x in v], dtype=bool)
        other = ~missing & (hit.sum(1) == 0)
        return np.column_stack([hit, other.astype(float), missing.astype(float)]) / np.sqrt(2.0)

    def _neutral_block(self, b: _Block, ft: FeatureTable) -> np.ndarray:
        if b.kind in ("continuous", "ordinal"):           # rank: empirical CDF of the training values
            x = ft.num[b.source]
            x = np.where(np.isfinite(x), x, b.params["median"])
            srt = b.params["sorted"]
            lo = np.searchsorted(srt, x, side="left")
            hi = np.searchsorted(srt, x, side="right")
            return ((lo + hi) / (2.0 * srt.size))[:, None]
        if b.kind == "nominal":
            return self._model_block(b, ft) / np.sqrt(2.0)
        return self._model_block(b, ft)

    def _metric_block(self, b: _Block, ft: FeatureTable) -> np.ndarray:
        if self.geometry == "shape":
            if b.shape is not None:
                idx = self.value_index(b, ft)
                out = np.zeros((len(idx), b.shape.shape[1]))
                ok = idx >= 0                                   # never-seen category → neutral (zero) contribution
                out[ok] = b.shape[idx[ok]]
                return out
            return self._neutral_block(b, ft)
        p = b.params
        if b.kind == "nominal":
            v = _norm_cat(ft.cat[b.source])
            vv = np.where(np.isin(v, p["keep"]), v, OTHER)
            out = np.empty((len(v), b.dims))
            seen = set(p.get("seen", p["keep"]))
            for i, (raw, c) in enumerate(zip(v, vv)):
                # kept or pooled-rare categories use their table row; never-seen categories take P(y) (B7)
                out[i] = p["table"][c][0] if (raw in seen and c in p["table"]) else p["unseen"]
            return out
        return self._model_block(b, ft)

    def _raw_block(self, b: _Block, ft: FeatureTable, mode: str = "metric") -> np.ndarray:
        return self._model_block(b, ft) if mode == "model" else self._metric_block(b, ft)

    def transform(self, ft: FeatureTable) -> np.ndarray:
        """Normalized metric embedding (n × D, float32). Unweighted: apply √w per column afterwards."""
        if not self.blocks:
            return np.zeros((len(ft), 0), dtype=np.float32)
        return np.hstack([self._metric_block(b, ft) / b.scale for b in self.blocks]).astype(np.float32)

    def neutral_dim_feature(self) -> np.ndarray:
        return np.concatenate([np.full(self._neutral_dims(b) if self.geometry == "shape" else b.dims, j)
                               for j, b in enumerate(self.blocks)]).astype(np.int64)

    def transform_neutral(self, ft: FeatureTable) -> np.ndarray:
        """Embedding without any outcome information (unsupervised comparators). Geometry hvdm: the metric itself."""
        if self.geometry != "shape":
            return self.transform(ft)
        return np.hstack([self._neutral_block(b, ft) / b.neutral_scale for b in self.blocks]).astype(np.float32)

    def transform_model(self, ft: FeatureTable) -> tuple[np.ndarray, np.ndarray]:
        """Matrix for models (nominal as one-hot) and the metric variable owning each column."""
        cols, owner = [], []
        for j, b in enumerate(self.blocks):
            e = self._model_block(b, ft)
            cols.append(e)
            owner.append(np.full(e.shape[1], j))
        return np.hstack(cols).astype(np.float32), np.concatenate(owner).astype(np.int64)

    # ------------------------------------------------------------------ utilities
    def vdm_tables(self, min_n: int = 10) -> dict:
        """Exportable VDM tables (cells with n < min_n suppressed, §9). Geometry hvdm only."""
        out = {}
        for b in self.blocks:
            if b.kind == "nominal" and "table" in b.params:
                # n rounded to 10 and p to 2 decimals: class counts per category cannot be back-calculated (A6)
                out[b.name] = {c: {"n": int(round(n, -1)), "p": [round(float(x), 2) for x in p]}
                               for c, (p, n) in b.params["table"].items() if n >= min_n}
        return out

    def shape_tables(self, min_n: int = 10) -> dict:
        """Exportable consensus shapes (centred log-odds per value; bins / categories with n ≥ min_n, n rounded)."""
        out = {}
        for b in self.blocks:
            if b.shape is None:
                continue
            vc = b.params.get("value_counts")
            if b.kind in ("continuous", "ordinal"):
                e = b.params["edges"]
                labels = [f"[{e[i]:.4g}, {e[i + 1]:.4g}{']' if i == e.size - 2 else ')'}" for i in range(e.size - 1)]
            elif b.kind == "nominal":
                labels = [str(c) for c in b.params["values"]]
            else:
                labels = [str(c) for c in b.params["codes"]] + [OTHER, MISSING]
            out[b.name] = [{"value": lab, "n": int(round(float(vc[i]), -1)) if vc is not None else None,
                            "g": [round(float(x), 3) for x in b.shape[i]]}
                           for i, lab in enumerate(labels) if vc is None or vc[i] >= min_n]
        return out

    def weights_dims(self, w: np.ndarray) -> np.ndarray:
        return np.sqrt(np.asarray(w, dtype=np.float64))[self.dim_feature].astype(np.float32)

    def neutral_weights_dims(self, w: np.ndarray) -> np.ndarray:
        return np.sqrt(np.asarray(w, dtype=np.float64))[self.neutral_dim_feature()].astype(np.float32)

    def feature_index(self) -> dict[str, int]:
        return {n: j for j, n in enumerate(self.feature_names)}

    def mutual_information(self, ft: FeatureTable, y: np.ndarray, n_classes: int) -> np.ndarray:
        """I(x_j; y) for binary variables (= univariate SAGE with a Bayes-optimal model); 0 for the others."""
        y = np.asarray(y)
        py = np.bincount(y, minlength=n_classes) / len(y)
        hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
        mi = np.zeros(self.n_features)
        for j, b in enumerate(self.blocks):
            if b.kind not in ("binary", "missing"):
                continue
            x = self._model_block(b, ft)[:, 0] > 0.5
            h = 0.0
            for val in (True, False):
                sel = x == val
                if sel.any():
                    p = np.bincount(y[sel], minlength=n_classes) / sel.sum()
                    h += sel.mean() * -np.sum(p[p > 0] * np.log(p[p > 0]))
            mi[j] = max(hy - h, 0.0)
        return mi

    def prevalence(self, ft: FeatureTable) -> np.ndarray:
        out = np.ones(self.n_features)
        for j, b in enumerate(self.blocks):
            if b.kind in ("binary", "missing"):
                out[j] = float((self._model_block(b, ft)[:, 0] > 0.5).mean())
        return out


def _scale(e: np.ndarray) -> float:
    return float(np.sqrt(2.0 * e.var(axis=0).sum()))


def _norm_cat(v: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=object).copy()
    bad = np.array([x is None or (isinstance(x, float) and x != x) or x == "" for x in out])
    out[bad] = MISSING
    return out.astype(str).astype(object)


# ---------------------------------------------------------------------------- groups (§5.3.1)
def build_groups(enc: Encoder, E: np.ndarray, corr_threshold: float, block_threshold: int,
                 max_rows: int = 20000, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Group of each metric variable: base group (value + indicator; PCS levels; support), large diagnosis /
    ATC blocks in a single group, and clusters with |ρ| > threshold (union of groups). `E` must be the embedding
    BEFORE shapes are installed (one column per single-dimension variable)."""
    F = enc.n_features
    labels = list(enc.feature_groups)
    for blk, gname in (("diagnoses", "dx_block"), ("medication", "med_block")):
        members = [j for j in range(F) if enc.feature_blocks[j] == blk]
        if len(members) > block_threshold:
            for j in members:
                labels[j] = gname
    parent = {g: g for g in labels}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    single = [j for j in range(F) if enc.blocks[j].dims == 1 and labels[j] not in ("dx_block", "med_block")]
    if len(single) > 1:
        rng = np.random.default_rng(seed)
        rows = rng.choice(E.shape[0], size=min(max_rows, E.shape[0]), replace=False)
        cols = [int(np.where(enc.dim_feature == j)[0][0]) for j in single]
        M = E[np.ix_(rows, cols)].astype(np.float64)
        sd = M.std(axis=0)
        ok = sd > 1e-12
        C = np.zeros((len(single), len(single)))
        if ok.sum() > 1:
            C[np.ix_(ok, ok)] = np.corrcoef(M[:, ok], rowvar=False)
        iu = np.argwhere(np.triu(np.abs(C) > corr_threshold, k=1))
        for a, b in iu:
            ga, gb = find(labels[single[a]]), find(labels[single[b]])
            if ga != gb:
                parent[gb] = ga
    final = [find(g) for g in labels]
    names = sorted(set(final))
    idx = {g: i for i, g in enumerate(names)}
    return np.array([idx[g] for g in final], dtype=np.int64), names


def distribute_group_importance(imp_group: np.ndarray, group_of_feature: np.ndarray, group_names: list[str],
                                enc: Encoder, ft: FeatureTable, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Group importance → importance per metric variable (NCA prior split; the NCA itself re-splits within groups,
    B4). Large blocks: split ∝ prevalence × mutual information; others: split equally."""
    F = enc.n_features
    imp = np.zeros(F)
    mi = prev = None
    for g, gname in enumerate(group_names):
        members = np.where(group_of_feature == g)[0]
        if members.size == 0:
            continue
        if gname in ("dx_block", "med_block"):
            if mi is None:
                mi, prev = enc.mutual_information(ft, y, n_classes), enc.prevalence(ft)
            sc = prev[members] * mi[members]
            sc = sc / sc.sum() if sc.sum() > 0 else np.full(members.size, 1.0 / members.size)
            imp[members] = imp_group[g] * sc
        else:
            imp[members] = imp_group[g] / members.size
    s = imp.sum()
    return imp / s if s > 0 else np.full(F, 1.0 / max(F, 1))
