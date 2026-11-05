"""Per-window metrics (design §8.2), point estimates and patient-bootstrap replicates at once.

Every metric takes row weights: (n,) for the point estimate, (R, n) for R bootstrap replicates (multinomial patient
weights, the same replicates for every arm of a window). AUROC counts ties as one half; AUPRC is the average
precision (steps at the distinct scores, as scikit-learn's); the calibration fits are Newton on y ~ a + b·logit(p)
(the intercept alone with the slope fixed at 1 for the calibration in the large), as `hypo.stats`.
"""

from __future__ import annotations

import numpy as np

from hypo.stats import logit


def _groups(sorted_p: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, sorted_p[1:] != sorted_p[:-1]])


def auroc(p: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    o = np.argsort(p, kind="mergesort")
    st = _groups(p[o])
    Ws, ys = W[..., o], y[o]
    pos = np.add.reduceat(Ws * ys, st, axis=-1)
    neg = np.add.reduceat(Ws * (1 - ys), st, axis=-1)
    below = np.cumsum(neg, axis=-1) - neg
    den = pos.sum(-1) * neg.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos * (below + 0.5 * neg)).sum(-1) / den


def auprc(p: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    o = np.argsort(-p, kind="mergesort")
    st = _groups(-p[o])
    Ws, ys = W[..., o], y[o]
    tp = np.add.reduceat(Ws * ys, st, axis=-1)
    fp = np.add.reduceat(Ws * (1 - ys), st, axis=-1)
    TP, FP = np.cumsum(tp, axis=-1), np.cumsum(fp, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = TP / (TP + FP)
        return (np.nan_to_num(prec) * tp).sum(-1) / tp.sum(-1)


def _sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def citl(p: np.ndarray, y: np.ndarray, W: np.ndarray, iters: int = 50) -> np.ndarray:
    off = logit(p)
    W2 = np.atleast_2d(W)
    a = np.zeros(W2.shape[0])
    for _ in range(iters):
        mu = _sig(a[:, None] + off)
        step = (W2 * (y - mu)).sum(1) / ((W2 * mu * (1 - mu)).sum(1) + 1e-9)
        a += step
        if np.abs(step).max() < 1e-8:
            break
    return a if W.ndim == 2 else a[0]


def slope(p: np.ndarray, y: np.ndarray, W: np.ndarray, iters: int = 50) -> np.ndarray:
    x = logit(p)
    W2 = np.atleast_2d(W)
    b = np.tile([0.0, 1.0], (W2.shape[0], 1))
    for _ in range(iters):
        mu = _sig(b[:, :1] + b[:, 1:] * x)
        w = W2 * (mu * (1 - mu) + 1e-9)
        r = W2 * (y - mu)
        g = np.stack([r.sum(1), (r * x).sum(1)], 1)
        H = np.stack([np.stack([w.sum(1), (w * x).sum(1)], 1), np.stack([(w * x).sum(1), (w * x * x).sum(1)], 1)], 1)
        step = np.linalg.solve(H + 1e-8 * np.eye(2), g[..., None])[..., 0]
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b[:, 1] if W.ndim == 2 else b[0, 1]


def wmean(v: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (W * v).sum(-1) / W.sum(-1)


def logloss(p: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return wmean(-(y * np.log(q) + (1 - y) * np.log(1 - q)), W)


def brier(p: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    return wmean((p - y) ** 2, W)


def oe(p: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (W * y).sum(-1) / (W * p).sum(-1)


def all_metrics(p: np.ndarray, y: np.ndarray, W: np.ndarray) -> dict:
    prev = wmean(y, W)
    ap = auprc(p, y, W)
    return {"auroc": auroc(p, y, W), "auprc": ap, "auprc_minus_prev": ap - prev, "auprc_over_prev": ap / prev,
            "citl": citl(p, y, W), "oe": oe(p, y, W), "slope": slope(p, y, W), "logloss": logloss(p, y, W),
            "brier": brier(p, y, W), "prevalence": prev}


BOOT_METRICS = ("auroc", "auprc", "auprc_minus_prev", "citl")


def boot_weights(code: np.ndarray, R: int, seed: int) -> np.ndarray:
    """(R, n) row weights from multinomial patient weights."""
    uniq, inv = np.unique(code, return_inverse=True)
    rng = np.random.default_rng(seed)
    wp = rng.multinomial(uniq.size, np.full(uniq.size, 1.0 / uniq.size), size=R).astype(np.float64)
    return wp[:, inv]


def boot_metrics(p: np.ndarray, y: np.ndarray, Wb: np.ndarray) -> dict:
    prev = wmean(y, Wb)
    ap = auprc(p, y, Wb)
    return {"auroc": auroc(p, y, Wb), "auprc": ap, "auprc_minus_prev": ap - prev, "citl": citl(p, y, Wb)}
