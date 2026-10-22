"""Permutation SAGE on log-loss (Covert et al., 2020), with absence simulated by the surrogate's mask.
For each case a random permutation of the groups is traversed and the contribution of each group is the
reduction in loss when it is added. Stops on convergence (maximum standard deviation relative to the range
of the values < threshold) or after `max_passes` passes over the explanation subsample.
"""

from __future__ import annotations

import numpy as np


def _ce(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1.0))


def sage_permutation(model, X: np.ndarray, y: np.ndarray, n_groups: int, rng: np.random.Generator,
                     batch_size: int = 512, max_passes: int = 3, threshold: float = 0.025) -> dict:
    G = int(n_groups)
    n = X.shape[0]
    s1 = np.zeros(G)
    s2 = np.zeros(G)
    cnt = np.zeros(G)
    n_eval = 0
    converged = False
    for _ in range(int(max_passes)):
        order = rng.permutation(n)
        for st in range(0, n, batch_size):
            b = order[st:st + batch_size]
            xb, yb = X[b], y[b]
            B = len(b)
            perms = np.argsort(rng.random((B, G)), axis=1)
            mask = np.zeros((B, G), dtype=bool)
            prev = _ce(model.predict_proba(xb, mask), yb)
            rows = np.arange(B)
            for t in range(G):
                g = perms[:, t]
                mask[rows, g] = True
                cur = _ce(model.predict_proba(xb, mask), yb)
                d = prev - cur
                np.add.at(s1, g, d)
                np.add.at(s2, g, d * d)
                np.add.at(cnt, g, 1)
                prev = cur
            n_eval += B * (G + 1)
        mean = s1 / np.maximum(cnt, 1)
        var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0) * cnt / np.maximum(cnt - 1, 1)
        se = np.sqrt(var / np.maximum(cnt, 1))
        rng_vals = mean.max() - mean.min()
        if rng_vals > 0 and se.max() / rng_vals < threshold:
            converged = True
            break
    mean = s1 / np.maximum(cnt, 1)
    var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0) * cnt / np.maximum(cnt - 1, 1)
    se = np.sqrt(var / np.maximum(cnt, 1))
    return {"values": mean, "se": se, "n_eval": n_eval, "converged": converged}


def normalize(values: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncates negatives to 0 and divides by the sum (§5.3.6). The standard error is scaled the same way.
    A model whose SAGE values are all ≤ 0 carries no information: uniform values with a very large SE, so it
    gets ~no weight in the aggregation (AUDIT B11)."""
    v = np.clip(values, 0, None)
    s = v.sum()
    if s <= 0:
        return np.full_like(v, 1.0 / len(v)), np.full_like(v, 1e3)
    return v / s, se / s


def aggregate(imp: np.ndarray, se: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Random-effects (DerSimonian–Laird) weighted mean per group, in the spirit of ShapleyVIC (§5.3.7).

    imp, se: (models × groups). The within-model SE is only Monte-Carlo noise: a model that never uses a group
    has SE = 0, which means "no sampling noise", not "infinite precision about the consensus". Plain
    inverse-variance weights would give such a model weight ~1e8 and drive the consensus to 0 (AUDIT B1).
    The between-model variance τ² is therefore added to every model's variance, so models are weighted by
    1 / (se² + τ²); with real heterogeneity this tends to the plain mean, without it to inverse variance.
    """
    imp = np.asarray(imp, dtype=float)
    se = np.asarray(se, dtype=float)
    if imp.shape[0] == 1:
        agg = imp[0].copy()
    else:
        M = imp.shape[0]
        v = se ** 2
        pos = v[v > 0]
        v = np.maximum(v, max(floor, 1e-3 * float(np.median(pos))) if pos.size else floor)
        w = 1.0 / v
        mu_fe = (w * imp).sum(0) / w.sum(0)
        q = (w * (imp - mu_fe) ** 2).sum(0)
        c = w.sum(0) - (w ** 2).sum(0) / w.sum(0)
        tau2 = np.maximum(0.0, (q - (M - 1)) / np.maximum(c, 1e-300))
        w_re = 1.0 / (v + tau2)
        agg = (w_re * imp).sum(0) / w_re.sum(0)
    agg = np.clip(agg, 0, None)
    s = agg.sum()
    return agg / s if s > 0 else np.full(imp.shape[1], 1.0 / imp.shape[1])
