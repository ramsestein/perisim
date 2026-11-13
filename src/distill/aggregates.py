"""Population aggregates of one site's landmarks, and synthetic cohorts drawn from them (docs/prereg_distill.md §3.3).

What is kept from a site (`SiteAggregates`, JSON-serialisable, no row of the source):

- **continuous / ordinal with many distinct values**: quantiles at 1/200 steps of the observed values, clipped to the
  1st–99th percentiles, and the missingness share;
- **discrete numeric** (at most `DISCRETE_MAX` distinct observed values, e.g. binary flags, L) and **nominal**: the
  share of each value; values held by fewer than `MIN_PATIENTS` patients are pooled into one "rare" cell whose mass is
  spread evenly over those values when sampling (the value labels are vocabulary, not data); plus the missingness
  share;
- **joint structure**: a Gaussian copula over one *value* dimension per input (the normal score of the observed value
  under that input's observed marginal; ties and categories spread uniformly inside their segment) and one
  *missingness* dimension per input that is sometimes missing (missing rows uniform in [0, m), observed rows in
  [m, 1)). Its correlation matrix is the Pearson correlation of the normal scores on pairwise-complete rows (pairs
  with fewer than MIN_PATIENTS complete rows get 0), made positive definite by eigenvalue clipping at EIG_FLOOR.

A site whose input is entirely missing keeps that (missing share 1). Each aggregate is computed on the site's rows;
the patient counts behind every cell are checked against `MIN_PATIENTS`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.special import ndtr, ndtri

from perisim.features.table import FeatureTable

MIN_PATIENTS = 50
DISCRETE_MAX = 30
GRID = 201                    # quantiles at 1/200 steps
CLIP = (1.0, 99.0)
EIG_FLOOR = 1e-6
EPS = 1e-6


@dataclass
class Marginal:
    name: str
    kind: str                                   # "continuous" | "discrete" | "nominal"
    missing: float                              # share of missing rows
    quantiles: list[float] | None = None        # continuous: GRID values of the observed distribution
    values: list | None = None                  # discrete / nominal: values in decreasing frequency
    shares: list[float] | None = None           # their shares among observed rows (rare values: shared mass)
    rare_from: int | None = None                # index from which values are pooled as rare (mass spread evenly)
    min_patients: int = 0                       # smallest patient count behind any released cell


@dataclass
class SiteAggregates:
    site: str
    n_rows: int
    n_patients: int
    marginals: list[Marginal]
    corr: list[list[float]] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "SiteAggregates":
        return cls(d["site"], d["n_rows"], d["n_patients"], [Marginal(**m) for m in d["marginals"]], d["corr"])


def _patients_per_value(col: np.ndarray, subjects: np.ndarray) -> dict:
    out = {}
    for v in np.unique(col):
        out[v] = np.unique(subjects[col == v]).size
    return out


def _marginal(name: str, col: np.ndarray, subjects: np.ndarray, nominal: bool) -> Marginal:
    if nominal:
        miss = np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in col])
        obs = col[~miss].astype(str)
    else:
        col = np.asarray(col, dtype=np.float64)
        miss = ~np.isfinite(col)
        obs = col[~miss]
    m = float(miss.mean()) if col.size else 1.0
    if obs.size == 0:
        return Marginal(name, "nominal" if nominal else "continuous", 1.0)
    distinct = np.unique(obs)
    if not nominal and distinct.size > DISCRETE_MAX:
        lo, hi = np.percentile(obs, CLIP)
        q = np.quantile(np.clip(obs, lo, hi), np.linspace(0, 1, GRID))
        n_pat = np.unique(subjects[~miss]).size
        return Marginal(name, "continuous", m, quantiles=[float(x) for x in q], min_patients=int(n_pat))
    pp = _patients_per_value(obs, subjects[~miss])
    order = sorted(distinct.tolist(), key=lambda v: (-(obs == v).sum(), str(v)))
    shares = [float((obs == v).mean()) for v in order]
    big = [i for i, v in enumerate(order) if pp[v] >= MIN_PATIENTS]
    rare = [i for i in range(len(order)) if i not in big]
    vals = [order[i] for i in big] + [order[i] for i in rare]
    sh = [shares[i] for i in big]
    rare_mass = float(sum(shares[i] for i in rare))
    if rare:                                    # the rare values share one released mass, spread evenly
        sh += [rare_mass / len(rare)] * len(rare)
        rare_pat = np.unique(subjects[~miss][np.isin(obs, [order[i] for i in rare])]).size
    else:
        rare_pat = None
    released = [pp[order[i]] for i in big] + ([rare_pat] if rare else [])
    vals = [v if nominal else float(v) for v in vals]
    return Marginal(name, "nominal" if nominal else "discrete", m, values=vals, shares=sh,
                    rare_from=len(big) if rare else None, min_patients=int(min(released)))


def _missing(marg: Marginal, col: np.ndarray) -> np.ndarray:
    if marg.kind == "nominal":
        return np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in col], dtype=bool)
    return ~np.isfinite(np.asarray(col, dtype=np.float64))


def _u_value(marg: Marginal, col: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each observed row's position in (0, 1) under the marginal of the observed values (NaN where missing); ties
    (and discrete / nominal values) spread uniformly inside their segment."""
    miss = _missing(marg, col)
    u = np.full(col.size, np.nan)
    o = ~miss
    if not o.any() or marg.missing >= 1.0:
        return u
    if marg.kind == "continuous":
        x = np.asarray(col, dtype=np.float64)[o]
        r = np.argsort(np.argsort(x + rng.uniform(0, 1e-9 * (1 + np.abs(x).max()), x.size)))
        u[o] = (r + 0.5) / x.size
    else:
        vals = [str(v) for v in marg.values] if marg.kind == "nominal" else list(marg.values)
        key = col[o].astype(str) if marg.kind == "nominal" else np.asarray(col, dtype=np.float64)[o]
        cum = np.concatenate([[0.0], np.cumsum(marg.shares)])
        cum = cum / cum[-1]
        pos = {v: i for i, v in enumerate(vals)}
        idx = np.array([pos.get(v, len(vals) - 1) for v in key.tolist()])
        lo, hi = cum[idx], cum[idx + 1]
        u[o] = lo + (hi - lo) * rng.uniform(0, 1, idx.size)
    return np.clip(u, EPS, 1 - EPS)


def _u_missing(marg: Marginal, col: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The missingness dimension: a missing row uniform in [0, m), an observed row uniform in [m, 1)."""
    miss = _missing(marg, col)
    m = marg.missing
    u = np.where(miss, rng.uniform(0, 1, col.size) * m, m + rng.uniform(0, 1, col.size) * (1 - m))
    return np.clip(u, EPS, 1 - EPS)


def _inverse(marg: Marginal, u: np.ndarray) -> np.ndarray:
    """Observed values of the marginal at positions u (every row observed; missingness is applied by the caller)."""
    n = u.size
    if marg.kind == "nominal":
        out = np.full(n, None, dtype=object)
    else:
        out = np.full(n, np.nan)
    if marg.missing >= 1.0 or (marg.quantiles is None and marg.values is None):
        return out
    if marg.kind == "continuous":
        return np.interp(u, np.linspace(0, 1, GRID), np.asarray(marg.quantiles))
    cum = np.cumsum(marg.shares)
    cum = cum / cum[-1]
    idx = np.minimum(np.searchsorted(cum, u, side="right"), len(marg.values) - 1)
    vals = np.asarray(marg.values, dtype=object)
    return vals[idx] if marg.kind == "nominal" else vals[idx].astype(np.float64)


def dims(margs: list[Marginal]) -> list[tuple[str, str]]:
    """The copula's dimensions: (input, "value") for every input with observed values, (input, "missing") for every
    input that is missing in some but not all rows."""
    out = [(mg.name, "value") for mg in margs if mg.missing < 1.0]
    out += [(mg.name, "missing") for mg in margs if 0.0 < mg.missing < 1.0]
    return out


def _pairwise_corr(Z: np.ndarray) -> np.ndarray:
    """Pearson correlation of the normal scores on the pairwise-complete rows (a value dimension is NaN where its
    input is missing); a pair with fewer than MIN_PATIENTS complete rows, or a constant column, gets 0."""
    p = Z.shape[1]
    F = np.isfinite(Z)
    C = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            ok = F[:, i] & F[:, j]
            if ok.sum() < MIN_PATIENTS:
                continue
            a, b = Z[ok, i], Z[ok, j]
            sa, sb = a.std(), b.std()
            if sa > 0 and sb > 0:
                C[i, j] = C[j, i] = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
    return C


def _nearest_pd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    C = (V * np.maximum(w, EIG_FLOOR)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / d[:, None] / d[None, :]


def fit(ft: FeatureTable, site: str, seed: int) -> SiteAggregates:
    """Aggregates of one site's rows (`ft`: that site's rows only)."""
    rng = np.random.default_rng(seed)
    margs = [_marginal(s.name, ft.column(s.name), ft.subjects, nominal=s.name in ft.cat) for s in ft.specs]
    by = {mg.name: mg for mg in margs}
    Z = []
    for name, kind in dims(margs):
        mg, col = by[name], ft.column(name)
        Z.append(ndtri(_u_value(mg, col, rng) if kind == "value" else _u_missing(mg, col, rng)))
    Z = np.column_stack(Z) if Z else np.zeros((len(ft), 0))
    C = _nearest_pd(_pairwise_corr(Z)) if Z.shape[1] else np.zeros((0, 0))
    return SiteAggregates(site, len(ft), int(np.unique(ft.subjects).size), margs, C.tolist())


def sample(agg: SiteAggregates, n: int, seed: int, *, copula: bool, specs) -> FeatureTable:
    """n synthetic rows of one site: SYN-COP (copula=True) or SYN-IND (copula=False). Each row is its own patient."""
    rng = np.random.default_rng(seed)
    D = dims(agg.marginals)
    p = len(D)
    if copula and p:
        L = np.linalg.cholesky(np.asarray(agg.corr) + 1e-12 * np.eye(p))
        U = ndtr(rng.standard_normal((n, p)) @ L.T)
    else:
        U = rng.uniform(0, 1, (n, p))
    U = np.clip(U, EPS, 1 - EPS)
    col_of = {d: j for j, d in enumerate(D)}
    num, cat = {}, {}
    for mg in agg.marginals:
        if (mg.name, "value") in col_of:
            col = _inverse(mg, U[:, col_of[(mg.name, "value")]])
        else:
            col = _inverse(mg, np.zeros(n))                    # entirely missing at this site
        if (mg.name, "missing") in col_of:
            miss = U[:, col_of[(mg.name, "missing")]] < mg.missing
            col = col.astype(object) if mg.kind == "nominal" else col.astype(np.float64)
            col[miss] = None if mg.kind == "nominal" else np.nan
        if mg.kind == "nominal":
            cat[mg.name] = np.asarray(col, dtype=object)
        else:
            num[mg.name] = np.asarray(col, dtype=np.float64)
    ids = np.array([f"syn-{agg.site}-{i}" for i in range(n)], dtype=object)
    return FeatureTable(ids, ids.copy(), specs, num, cat)


def released_min_patients(agg: SiteAggregates) -> int:
    """The smallest patient count behind any released cell (must be >= MIN_PATIENTS, or the site's total)."""
    cells = [mg.min_patients for mg in agg.marginals if mg.missing < 1.0 and mg.min_patients]
    return int(min(cells)) if cells else agg.n_patients
