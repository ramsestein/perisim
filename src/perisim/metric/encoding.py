"""δ_j encoding (§5.2), ALWAYS fitted on the training fold.

Each metric variable j is represented as a block of columns φ_j such that ‖φ_j(a) − φ_j(b)‖² = δ_j(a,b)²:
- continuous: standardized value (median for imputation; + missingness indicator if > 5 %);
- ordinal: range scaled to [0, 1];
- binary: 0/1;
- nominal: smoothed VDM, φ = p_c = (n_c·P̂(y|c) + m·P(y)) / (n_c + m) ⇒ δ = ‖p_c − p_c'‖₂ exactly;
- hierarchical: one ICD-10-PCS level per variable, one-hot/√2 ⇒ δ ∈ {0, 1} (rare codes → zero vector).
Normalization (§5.1): each block is divided by s_j with s_j² = E_pairs[δ_j²] = 2·Σ Var(φ_j), so that the
mean squared dissimilarity between random training pairs is 1 and w_j keeps its meaning.

`transform_model` gives the version for the surrogate models and comparators (nominal as one-hot, no target).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from perisim.features.table import FeatureTable

MISSING = "__missing__"
OTHER = "__other__"


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


class Encoder:
    def __init__(self, *, m: float = 5.0, missing_threshold: float = 0.05, hier_min_count: int = 10,
                 hier_max_levels: int = 200, nominal_min_count: int = 10, nominal_max_categories: int = 50,
                 dx_min_prevalence: float = 0.005, med_min_count: int = 10):
        self.m = float(m)
        self.missing_threshold = missing_threshold
        self.hier_min_count = hier_min_count
        self.hier_max_levels = hier_max_levels
        self.nominal_min_count = nominal_min_count
        self.nominal_max_categories = nominal_max_categories
        self.dx_min_prevalence = dx_min_prevalence
        self.med_min_count = med_min_count
        self.blocks: list[_Block] = []

    @classmethod
    def from_config(cls, cfg, m: float | None = None) -> "Encoder":
        f = cfg.experiment["features"]
        return cls(m=m if m is not None else cfg.experiment["metric"]["vdm_m_grid"][1],
                   missing_threshold=f["missing_indicator_threshold"], hier_min_count=f["hier_min_count"],
                   hier_max_levels=f["hier_max_levels"], nominal_min_count=f["nominal_min_count"],
                   nominal_max_categories=f["nominal_max_categories"], dx_min_prevalence=f["dx_min_prevalence"])

    # ------------------------------------------------------------------ fitting
    def fit(self, ft: FeatureTable, y: np.ndarray, n_classes: int, y_aux: np.ndarray | None = None) -> "Encoder":
        y = np.asarray(y, dtype=np.int64)
        self.n_classes = n_classes
        prior = np.bincount(y, minlength=n_classes).astype(float)
        prior = (prior + 1e-9) / (prior.sum() + 1e-9 * n_classes)
        aux_prior = np.nanmean(y_aux, axis=0) if y_aux is not None else None
        n = len(ft)
        self.blocks = []
        for s in ft.specs:
            if s.type == "continuous":
                x = ft.num[s.name]
                miss = ~np.isfinite(x)
                if miss.all():
                    continue
                med = float(np.median(x[~miss]))
                xi = np.where(miss, med, x)
                mu, sd = float(xi.mean()), float(xi.std())
                if sd < 1e-12:
                    continue
                self.blocks.append(_Block(s.name, s.name, "continuous", s.group, s.block,
                                          {"median": med, "mean": mu, "sd": sd}))
                if miss.mean() > self.missing_threshold and miss.mean() < 1:
                    self.blocks.append(_Block(f"{s.name}__missing", s.name, "missing", s.group, s.block))
            elif s.type == "ordinal":
                x = ft.num[s.name]
                miss = ~np.isfinite(x)
                if miss.all():
                    continue
                lv = s.levels or [float(np.nanmin(x)), float(np.nanmax(x))]
                lo, hi = float(min(lv)), float(max(lv))
                if hi <= lo:
                    continue
                med = float(np.median(x[~miss]))
                self.blocks.append(_Block(s.name, s.name, "ordinal", s.group, s.block, {"median": med, "lo": lo, "hi": hi}))
                if miss.mean() > self.missing_threshold and miss.mean() < 1:
                    self.blocks.append(_Block(f"{s.name}__missing", s.name, "missing", s.group, s.block))
            elif s.type == "binary":
                x = np.nan_to_num(ft.num[s.name], nan=0.0)
                cnt = float((x > 0.5).sum())
                if cnt == 0 or cnt == n:
                    continue
                if s.block == "diagnoses" and cnt / n < self.dx_min_prevalence:
                    continue
                if s.block == "medication" and cnt < self.med_min_count:
                    continue
                self.blocks.append(_Block(s.name, s.name, "binary", s.group, s.block, {"prevalence": cnt / n}))
            elif s.type == "nominal":
                v = _norm_cat(ft.cat[s.name])
                cats, counts = np.unique(v, return_counts=True)
                order = np.argsort(-counts)
                keep = [c for c, k in zip(cats[order], counts[order]) if k >= self.nominal_min_count][: self.nominal_max_categories]
                if len(keep) < 2 and not (len(keep) == 1 and (v != keep[0]).any()):
                    continue
                vv = np.where(np.isin(v, keep), v, OTHER)
                table = {}
                for c in np.unique(vv):
                    sel = vv == c
                    nc = sel.sum()
                    phat = np.bincount(y[sel], minlength=n_classes) / max(nc, 1)
                    p = (nc * phat + self.m * prior) / (nc + self.m)
                    if y_aux is not None:
                        pa = (nc * np.nanmean(y_aux[sel], axis=0) + self.m * aux_prior) / (nc + self.m)
                        p = np.concatenate([p, pa])
                    table[c] = (p, int(nc))
                unseen = prior if y_aux is None else np.concatenate([prior, aux_prior])
                self.blocks.append(_Block(s.name, s.name, "nominal", s.group, s.block,
                                          {"keep": keep, "table": table, "unseen": unseen,
                                           "onehot": sorted(set(keep) | {OTHER})},
                                          dims=len(unseen)))
            elif s.type == "hier":
                v = ft.cat[s.name]
                vals = np.array([x for x in v if x is not None and x == x], dtype=object)
                if vals.size == 0:
                    continue
                cats, counts = np.unique(vals, return_counts=True)
                order = np.argsort(-counts)
                keep = [c for c, k in zip(cats[order], counts[order]) if k >= self.hier_min_count][: self.hier_max_levels]
                if not keep:
                    continue
                self.blocks.append(_Block(s.name, s.name, "hier", s.group, s.block, {"codes": keep}, dims=len(keep)))
        # scale s_j: mean square of δ_j between random pairs = 1
        raw = self._raw_blocks(ft)
        kept = []
        for b, e in zip(self.blocks, raw):
            sj = float(np.sqrt(2.0 * e.var(axis=0).sum()))
            if sj > 1e-12:
                b.scale = sj
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

    # ------------------------------------------------------------------ transformation
    def _raw_block(self, b: _Block, ft: FeatureTable, mode: str = "metric") -> np.ndarray:
        n = len(ft)
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
            return (np.nan_to_num(ft.num[b.source], nan=0.0) > 0.5).astype(float)[:, None]
        if b.kind == "nominal":
            v = _norm_cat(ft.cat[b.source])
            vv = np.where(np.isin(v, p["keep"]), v, OTHER)
            if mode == "model":
                cats = p["onehot"]
                return (vv[:, None] == np.array(cats, dtype=object)[None, :]).astype(float)
            out = np.empty((n, b.dims))
            for i, c in enumerate(vv):
                out[i] = p["table"][c][0] if c in p["table"] else p["unseen"]
            return out
        if b.kind == "hier":
            v = ft.cat[b.source]
            codes = np.array(p["codes"], dtype=object)
            return (np.asarray(v, dtype=object)[:, None] == codes[None, :]).astype(float) / np.sqrt(2.0)
        raise ValueError(b.kind)

    def _raw_blocks(self, ft: FeatureTable) -> list[np.ndarray]:
        return [self._raw_block(b, ft) for b in self.blocks]

    def transform(self, ft: FeatureTable) -> np.ndarray:
        """Normalized metric embedding (n × D, float32). Unweighted: apply √w per column afterwards."""
        if not self.blocks:
            return np.zeros((len(ft), 0), dtype=np.float32)
        return np.hstack([self._raw_block(b, ft) / b.scale for b in self.blocks]).astype(np.float32)

    def transform_model(self, ft: FeatureTable) -> tuple[np.ndarray, np.ndarray]:
        """Matrix for models (nominal as one-hot) and the metric variable owning each column."""
        cols, owner = [], []
        for j, b in enumerate(self.blocks):
            e = self._raw_block(b, ft, mode="model")
            cols.append(e)
            owner.append(np.full(e.shape[1], j))
        return np.hstack(cols).astype(np.float32), np.concatenate(owner).astype(np.int64)

    # ------------------------------------------------------------------ utilities
    def vdm_tables(self, min_n: int = 10) -> dict:
        """Exportable VDM tables (cells with n < min_n suppressed, §9)."""
        out = {}
        for b in self.blocks:
            if b.kind == "nominal":
                out[b.name] = {c: {"n": n, "p": [round(float(x), 4) for x in p]}
                               for c, (p, n) in b.params["table"].items() if n >= min_n}
        return out

    def weights_dims(self, w: np.ndarray) -> np.ndarray:
        return np.sqrt(np.asarray(w, dtype=np.float64))[self.dim_feature].astype(np.float32)

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
            x = self._raw_block(b, ft)[:, 0] > 0.5
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
                out[j] = float((self._raw_block(b, ft)[:, 0] > 0.5).mean())
        return out


def _norm_cat(v: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=object).copy()
    bad = np.array([x is None or (isinstance(x, float) and x != x) or x == "" for x in out])
    out[bad] = MISSING
    return out.astype(str).astype(object)


# ---------------------------------------------------------------------------- groups (§5.3.1)
def build_groups(enc: Encoder, E: np.ndarray, corr_threshold: float, block_threshold: int,
                 max_rows: int = 20000, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Group of each metric variable: base group (value + indicator; PCS levels; support), large diagnosis /
    ATC blocks in a single group, and clusters with |ρ| > threshold (union of groups)."""
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
    """Group importance → importance per metric variable (NCA prior).
    Large blocks: split ∝ prevalence × mutual information; others: split equally."""
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
