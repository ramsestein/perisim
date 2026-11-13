"""The distillation study (docs/prereg_distill.md): every step on explicit inputs, so that tests can run it end to end
on synthetic landmarks and `scripts/distill_run.py` on the frozen artefacts of arm `value`.

Per rotation e (the target; the two other sites are the source):

1. `rotation_aggregates`: the real learning sample of `hypo.frozen.final` (same rows, same seed); one
   `SiteAggregates` per source site, on that site's learning rows.
2. `synthetic_cohort`: per source site as many synthetic rows as the site has in the learning sample, drawn from its
   aggregates (SYN-COP or SYN-IND), labelled y ~ Bernoulli(p_T) by the frozen teacher.
3. `fit_on`: `hypo.metric.fit_metric` unchanged, with the seed of `hypo.frozen.final` (the same function re-fits
   M_REAL in integrity check 1).
4. `predict_folds`: the frozen metric at a target, with that target's case base, fold by fold, as
   `hypo.external.Stage.transported` and `scripts/nwicu_transport.run_arms` do.

Integrity and fidelity summaries (`dcr`, `fidelity`) return aggregates only.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from hypo.data import Landmarks, concat, learning_sample
from distill import aggregates as A

ARMS = ("cop", "ind")
ARM_INDEX = {"cop": 0, "ind": 1}


def learn_rows(pool: Landmarks, mcfg: dict, seed: int) -> np.ndarray:
    """The rows of the real learning sample (`hypo.frozen.final`)."""
    return learning_sample(pool, int(mcfg["learn"]["per_patient"]), int(mcfg["learn"]["per_site"]), seed)


def rotation_aggregates(learn: Landmarks, seed: int) -> dict[str, A.SiteAggregates]:
    out = {}
    for i, s in enumerate(sorted(np.unique(learn.site).tolist())):
        rows = np.where(learn.site == s)[0]
        out[s] = A.fit(learn.ft.take(rows), s, seed=int(np.random.SeedSequence([seed, 11, i]).generate_state(1)[0]))
    return out


def synthetic_cohort(aggs: dict[str, A.SiteAggregates], teacher, specs, seed: int, rotation: int, arm: str) -> tuple[Landmarks, dict]:
    """The labelled synthetic cohort of one rotation and arm (one patient per row); and its label summary."""
    parts = []
    for i, (s, agg) in enumerate(sorted(aggs.items())):
        ss = np.random.SeedSequence([seed, 23, rotation, ARM_INDEX[arm], i]).generate_state(1)[0]
        ft = A.sample(agg, agg.n_rows, int(ss), copula=(arm == "cop"), specs=specs)
        n = len(ft)
        parts.append(Landmarks(ft, np.zeros(n, dtype=np.int64), np.full(n, -1, dtype=np.int8),
                               np.full(n, s, dtype=object), np.full(n, None, dtype=object),
                               pl.DataFrame({"syn_row": np.arange(n)})))
    lm = concat(parts)
    p = np.clip(np.asarray(teacher(lm), dtype=np.float64), 0.0, 1.0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 29, rotation, ARM_INDEX[arm]]))
    lm.y = (rng.random(len(lm)) < p).astype(np.int64)
    info = {"n": len(lm), "per_site": {str(s): int((lm.site == s).sum()) for s in np.unique(lm.site)},
            "mean_p_teacher": float(p.mean()), "prevalence_drawn": float(lm.y.mean())}
    return lm, info


def teacher_labelled(learn: Landmarks, teacher, seed: int, rotation: int) -> tuple[Landmarks, dict]:
    """Post hoc (DECISIONS 2026-11-13, arm TEACH): the REAL learning rows with y ~ Bernoulli(p_T), separating the cost
    of replacing the outcome by the teacher from the cost of replacing the rows by synthetic ones."""
    import copy

    lm = copy.copy(learn)
    p = np.clip(np.asarray(teacher(learn), dtype=np.float64), 0.0, 1.0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 29, rotation, 2]))
    lm.y = (rng.random(len(lm)) < p).astype(np.int64)
    return lm, {"n": len(lm), "mean_p_teacher": float(p.mean()), "prevalence_drawn": float(lm.y.mean()),
                "agreement_with_real_y": float((lm.y == learn.y).mean())}


SOFT_K = 5


def soft_expand(lm: Landmarks, p: np.ndarray, seed: int, K: int = SOFT_K) -> tuple[Landmarks, dict]:
    """Post hoc (DECISIONS 2026-11-14, soft labels): every row K times, same patient, with labels 1{(k + U)/K < p}
    (one U ~ U(0, 1) per row, k = 0..K-1): the share of positive copies is p to within 1/K and unbiased, without the
    noise of one Bernoulli draw per row. The copies of a row never neighbour each other (the NCA and the case-base
    searches exclude the same patient; the shape models split by patient)."""
    n = len(lm)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    U = rng.random(n)
    rows = np.repeat(np.arange(n), K)
    k = np.tile(np.arange(K), n)
    y = (((k + U[rows]) / K) < p[rows]).astype(np.int64)
    out = lm.take(rows)
    out.ft.ids = np.array([f"{i}#{r}#{j}" for i, r, j in zip(lm.ft.ids[rows].astype(str), rows, k)], dtype=object)
    out.y = y
    return out, {"n_rows": n, "K": K, "n_expanded": len(out), "mean_p_teacher": float(p.mean()),
                 "share_positive": float(y.mean()), "max_abs_share_error": float(np.max(np.abs(
                     y.reshape(n, K).mean(1) - p)))}


def fit_on(learn: Landmarks, mcfg: dict, space: dict, seed: int, threads: int):
    from hypo.metric import fit_metric

    return fit_metric(learn, mcfg, space, seed=seed, threads=threads)


def predict_folds(metric, lm: Landmarks, splits, mcfg: dict, seed: int, E: np.ndarray | None = None) -> np.ndarray:
    """p of every row tested in `splits` [(f, train rows, test rows)], with the case base of the training rows
    (`E`: the rows already embedded by `metric`)."""
    from hypo.casebase import CaseBase

    E = metric.embed(lm) if E is None else E
    p = np.full(len(lm), np.nan)
    for f, tr, te in splits:
        cb = CaseBase(metric, lm.take(tr), mcfg, E=E[tr])
        cb.fit_alpha(seed + 100 * f)
        p[te] = np.asarray(cb.predict(lm.take(te), E=E[te])["p"], dtype=np.float64)
    return p


# ---------------------------------------------------------------------------- integrity and fidelity (aggregates)
def _matrix(ft, names: list[str], ref) -> np.ndarray:
    """Standardised numeric inputs (centre and scale of `ref`), missing values at the centre plus an indicator."""
    cols = []
    for n in names:
        x, r = ft.num[n], ref.num[n]
        ro = r[np.isfinite(r)]
        if ro.size == 0:
            continue
        c, s = float(np.median(ro)), float(np.subtract(*np.percentile(ro, [75, 25]))) or float(ro.std()) or 1.0
        z = (x - c) / s
        miss = ~np.isfinite(z)
        cols.append(np.where(miss, 0.0, z))
        if (~np.isfinite(r)).any():
            cols.append(miss.astype(float))
    return np.column_stack(cols) if cols else np.zeros((len(ft), 0))


def dcr(syn: Landmarks, real: Landmarks, seed: int, n_max: int = 20_000) -> dict:
    """Distance to closest record: synthetic → real source rows, against real half → the other real half (patients
    split at random). Quantiles only."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    names = [n for n in real.ft.num]
    subj = real.ft.subjects.astype(str)
    u = np.unique(subj)
    half = np.isin(subj, rng.choice(u, size=u.size // 2, replace=False))
    a, b = np.where(half)[0], np.where(~half)[0]
    Xr = _matrix(real.ft, names, real.ft)
    Xs = _matrix(syn.ft, names, real.ft)

    def q(X, Y):
        if X.shape[0] > n_max:
            X = X[rng.choice(X.shape[0], n_max, replace=False)]
        d = NearestNeighbors(n_neighbors=1).fit(Y).kneighbors(X)[0][:, 0]
        return {"p01": float(np.percentile(d, 1)), "p05": float(np.percentile(d, 5)),
                "median": float(np.median(d)), "share_exact": float((d < 1e-9).mean())}

    return {"synthetic_to_real": q(Xs, Xr[b]), "real_to_real": q(Xr[a], Xr[b]), "n_inputs": int(Xr.shape[1])}


def _corr_error(syn_ft, agg: A.SiteAggregates, seed: int) -> float | None:
    """Mean absolute difference between the copula correlations of the synthetic rows and of the real aggregates,
    over the dimensions both have."""
    b = A.fit(syn_ft, agg.site, seed)
    da, db = A.dims(agg.marginals), A.dims(b.marginals)
    common = [d for d in da if d in db]
    if len(common) < 2:
        return None
    ia, ib = [da.index(d) for d in common], [db.index(d) for d in common]
    Ca, Cb = np.asarray(agg.corr)[np.ix_(ia, ia)], np.asarray(b.corr)[np.ix_(ib, ib)]
    iu = np.triu_indices(len(common), 1)
    return float(np.mean(np.abs(Ca[iu] - Cb[iu])))


def fidelity(syn: Landmarks, learn: Landmarks, aggs: dict, seed: int) -> dict:
    """Marginal errors (quantiles 5–95 % in units of the real IQR; missing share; category shares) and the mean
    absolute error of the normal-score correlations, per source site; all on the real learning rows' aggregates."""
    res = {}
    for s, agg in aggs.items():
        rs, rr = np.where(syn.site == s)[0], np.where(learn.site == s)[0]
        fs, fr = syn.ft.take(rs), learn.ft.take(rr)
        q_err, m_err, c_err = [], [], []
        for mg in agg.marginals:
            xs, xr = fs.column(mg.name), fr.column(mg.name)
            if mg.kind == "nominal":
                ms = np.array([v is None for v in xs])
                mr = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in xr])
                m_err.append(abs(ms.mean() - mr.mean()))
                vs, vr = xs[~ms].astype(str), xr[~mr].astype(str)
                cats = np.union1d(np.unique(vs), np.unique(vr))
                if cats.size and vs.size and vr.size:
                    c_err.append(0.5 * sum(abs((vs == c).mean() - (vr == c).mean()) for c in cats))
                continue
            xs, xr = np.asarray(xs, float), np.asarray(xr, float)
            m_err.append(abs((~np.isfinite(xs)).mean() - (~np.isfinite(xr)).mean()))
            os_, or_ = xs[np.isfinite(xs)], xr[np.isfinite(xr)]
            if os_.size and or_.size:
                iqr = float(np.subtract(*np.percentile(or_, [75, 25]))) or float(or_.std()) or 1.0
                g = np.linspace(5, 95, 19)
                q_err.append(float(np.mean(np.abs(np.percentile(os_, g) - np.percentile(or_, g)))) / iqr)
        res[s] = {"quantile_error_iqr_mean": float(np.mean(q_err)) if q_err else None,
                  "quantile_error_iqr_max": float(np.max(q_err)) if q_err else None,
                  "missing_share_error_max": float(np.max(m_err)) if m_err else None,
                  "category_tv_max": float(np.max(c_err)) if c_err else None,
                  "corr_mae": _corr_error(fs, agg, seed)}
    return res


def weight_agreement(m_syn, m_real) -> dict:
    from scipy.stats import spearmanr

    a, b = m_syn.weights_table(), m_real.weights_table()
    names = sorted(set(a) | set(b))
    va, vb = np.array([a.get(n, 0.0) for n in names]), np.array([b.get(n, 0.0) for n in names])
    rho = spearmanr(va, vb).statistic if len(names) > 2 else float("nan")
    return {"spearman": float(rho), "l1_profile": float(np.abs(va - vb).sum()),
            "weights": {n: {"distilled": float(x), "real": float(y)} for n, x, y in zip(names, va, vb)}}
