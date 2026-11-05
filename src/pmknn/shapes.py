"""Shape functions g_j: read from every accepted model, then their consensus.

For every accepted model, the curve of each continuous / ordinal / nominal input is READ from the model (no
training): how its values move the prediction, in centred log-odds (one component per class), on a fresh random
subsample of the explanation rows.
- continuous / ordinal: accumulated local effects (ALE) over the training quantile bins. Only real rows are moved,
  within their own bin, so correlated inputs do not leak into the curve;
- nominal: local effect of each category against the reference (most frequent) one, averaged over its rows.

The consensus curve is fitted to all the models' curves by least squares with a per-model scale (curve_m ≈ a_m·g),
so a model that does not use the input gets a_m ≈ 0; the agreement R² is reported. The curve is then a small
meta-analysis over its values (DerSimonian–Laird), with each value's uncertainty taken from the spread of the
models' curves around their plain mean:
- it is kept only if distinguishable from flat by a homogeneity F test (between-model variance pooled over the
  values; level `flat_alpha`, Bonferroni over the classes); otherwise the input keeps its neutral axis. For a
  continuous / ordinal input the test reads the first differences of the curves along the bins (local effects,
  V − 1 values; flat = every local effect is zero), because the noise of an ALE curve is accumulated and its values
  are not independent; for a nominal input it reads the values (category effects) themselves;
- each value is shrunk towards the mean by τ² / (τ² + se²).

Port of PERISIM's `importance/shapes.py`. Additions: an approximate standard error of each value (the spread of the
models' curves, scaled by the shrinkage), kept to draw a band around the curve; and (0.4.0, PERISIM v0.6) the
flatness test on the local effects of continuous / ordinal inputs.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import f as fdist

from .encoding import Encoder
from .schema import Table


def centred_logodds(p: np.ndarray) -> np.ndarray:
    lp = np.log(np.clip(p, 1e-6, 1.0))
    return lp - lp.mean(1, keepdims=True)


def model_curves(model, Xm: np.ndarray, owner: np.ndarray, enc: Encoder, t_sub: Table, blocks: list[int]) -> dict:
    """{block: (values n_values × C, rows per value)} for one fitted model on the subsample (Xm, t_sub)."""
    f_full = centred_logodds(model.predict_proba(Xm))
    out = {}
    for j in blocks:
        b = enc.blocks[j]
        cols = np.where(owner == j)[0]
        V = enc.n_values(b)
        idx = enc.value_index(b, t_sub)
        C = f_full.shape[1]
        if b.kind in ("continuous", "ordinal"):
            obs = np.isfinite(t_sub.num[b.source])
            if not obs.any():
                continue
            bi = idx[obs]
            e = b.params["edges"]
            X_lo, X_hi = Xm[obs].copy(), Xm[obs].copy()
            X_lo[:, cols[0]] = enc.model_column_value(b, e[bi])
            X_hi[:, cols[0]] = enc.model_column_value(b, e[bi + 1])
            d = centred_logodds(model.predict_proba(X_hi)) - centred_logodds(model.predict_proba(X_lo))
            cnt = np.bincount(bi, minlength=V).astype(float)
            le = np.zeros((V, C))
            for c in range(C):
                le[:, c] = np.bincount(bi, weights=d[:, c], minlength=V) / np.maximum(cnt, 1)
            G = np.vstack([np.zeros((1, C)), np.cumsum(le, axis=0)])
            vals = 0.5 * (G[:-1] + G[1:])
        else:
            ref = int(b.params["ref"])
            X_ref = Xm.copy()
            X_ref[:, cols] = enc.model_onehot(b, ref)[None, :]
            d = f_full - centred_logodds(model.predict_proba(X_ref))
            ok = idx >= 0
            cnt = np.bincount(idx[ok], minlength=V).astype(float)
            vals = np.zeros((V, C))
            for c in range(C):
                vals[:, c] = np.bincount(idx[ok], weights=d[ok, c], minlength=V) / np.maximum(cnt, 1)
        w = np.asarray(b.params["value_counts"], dtype=float)
        vals = vals - (w[:, None] * vals).sum(0) / max(w.sum(), 1e-12)
        out[j] = (vals, cnt)
    return out


def consensus(curves: list[np.ndarray], counts: np.ndarray, n_iter: int = 10, flat_alpha: float = 0.05,
              local: bool = False) -> dict:
    """Consensus shape of one input from the models' curves (each n_values × C). Returns g (None = flat / not
    estimable), the per-model scales, the agreement R², τ², the homogeneity p-value per class and the standard
    error of each value (`se`, n_values × C, after shrinkage). `local`: the flatness test reads the first
    differences of the curves along the values (continuous / ordinal inputs)."""
    w = np.asarray(counts, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(w.size, 1.0 / max(w.size, 1))
    Cm = np.array([c - (w[:, None] * c).sum(0) for c in curves])           # M × V × C, centred per class
    M = Cm.shape[0]
    res = {"g": None, "n_models": int(M), "r2": None, "tau2": None, "flat": True, "shrinkage": None, "se": None}
    if M == 0:
        return res
    wn = w[None, :, None]
    g = Cm.mean(0)
    a = np.ones(M)
    for _ in range(n_iter):
        den = float((w[:, None] * g * g).sum())
        if den <= 1e-14:
            return res
        a = np.maximum(0.0, (wn * Cm * g[None]).sum((1, 2)) / den)
        if (a * a).sum() <= 1e-14:
            return res
        g = (a[:, None, None] * Cm).sum(0) / (a * a).sum()
    s = float(a.mean()) or 1.0
    a, g = a / s, g * s
    R = Cm - a[:, None, None] * g[None]
    tot = float((wn * Cm * Cm).sum())
    res.update(a=a.tolist(), r2=(1.0 - float((wn * R * R).sum()) / tot) if tot > 0 else None)
    if M < 3:                       # too few models for an uncertainty: provisional shape, no shrinkage
        res.update(g=g, flat=False)
        return res
    gm = Cm.mean(0)                                                    # plain mean and its standard error (V × C)
    se2 = Cm.var(0, ddof=1) / M
    use = w > 0
    C_ = g.shape[1]
    tau2, pval = np.zeros(C_), np.ones(C_)
    shrink = np.ones_like(g)
    if local:
        # v0.6: the flatness test on the local effects (first differences along the bins, V − 1 values): the ALE
        # noise is accumulated, so the curve values are not independent and the test on them declared ~35 % of
        # pure-noise curves shaped. Flat = every local effect is zero, so the differences are not centred (centring
        # would call a straight line flat); the F statistic has V − 1 and (V − 1)(M − 1) degrees of freedom.
        Dm = np.diff(Cm, axis=1)                                          # M × (V − 1) × C
        dm, dse2 = Dm.mean(0), Dm.var(0, ddof=1) / M
    for c in range(C_):
        y, s2 = gm[use, c], np.maximum(se2[use, c], 1e-12)
        wv = 1.0 / s2
        ybar = float((wv * y).sum() / wv.sum())
        q = float((wv * (y - ybar) ** 2).sum())
        df = int(y.size - 1)
        cdl = float(wv.sum() - (wv * wv).sum() / wv.sum())
        tau2[c] = max(0.0, (q - df) / cdl) if cdl > 0 else 0.0
        if local:
            ld = int(dm.shape[0])
            s2p = float(np.mean(dse2[:, c])) if ld else 0.0
            qp = float((dm[:, c] ** 2).sum() / max(s2p, 1e-12))
            pval[c] = float(fdist.sf(qp / ld, ld, ld * (M - 1))) if ld > 0 else 1.0
        else:
            s2p = float(np.mean(se2[use, c]))
            qp = float(((y - y.mean()) ** 2).sum() / max(s2p, 1e-12))
            pval[c] = float(fdist.sf(qp / df, df, y.size * (M - 1))) if df > 0 else 1.0
        k = tau2[c] / (tau2[c] + np.maximum(se2[:, c], 1e-12))
        g[:, c] = g[:, c] * k                                          # shrink towards the (zero) mean
        shrink[:, c] = k
    g = g - (w[:, None] * g).sum(0)
    res.update(tau2=tau2.tolist(), p_flat=pval.tolist(), shrinkage=float(shrink[use].mean()),
               se=np.sqrt(se2) * shrink)
    if not (pval < flat_alpha / C_).any() or (tau2 <= 0).all():
        return res                  # not distinguishable from flat → neutral axis
    res.update(g=g, flat=False)
    return res


def consensus_shapes(curves: dict[int, list[np.ndarray]], enc: Encoder, flat_alpha: float = 0.05,
                     local_test: bool = True) -> tuple[dict[int, np.ndarray | None], dict]:
    """Consensus of every shapeable input: {block: g or None} and the diagnostics per input name. `local_test`
    False restores the flatness test of pmknn ≤ 0.3 / PERISIM ≤ v0.5 (curve values), for comparisons only."""
    shapes, diag = {}, {}
    for j, cl in curves.items():
        b = enc.blocks[j]
        r = consensus(cl, b.params["value_counts"], flat_alpha=flat_alpha,
                      local=local_test and b.kind in ("continuous", "ordinal")) if cl else \
            {"g": None, "n_models": 0, "flat": True, "r2": None, "se": None}
        shapes[j] = r["g"]
        d = {k: r.get(k) for k in ("n_models", "r2", "tau2", "p_flat", "flat", "shrinkage")}
        d["shaped"] = r["g"] is not None
        d["se"] = None if r.get("se") is None else np.asarray(r["se"]).tolist()
        diag[b.name] = d
    return shapes, diag
