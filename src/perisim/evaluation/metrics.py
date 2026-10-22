"""Métricas probabilísticas (§6.5): log-loss y Brier multiclase, RPS ordinal, calibración uno-contra-resto
(intercepto y pendiente), ECE, AUROC OvR y AUPRC de la clase grave; bootstrap por paciente."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-15


def row_logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))


def row_brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1
    return ((p - oh) ** 2).sum(1)


def row_rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    C = p.shape[1]
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1
    return ((np.cumsum(p, 1) - np.cumsum(oh, 1)) ** 2).sum(1)[:] / max(C - 1, 1)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_line(p: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Regresión logística y ~ a + b·logit(p) (Newton). Devuelve (intercepto, pendiente)."""
    x = _logit(p)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        g = X.T @ (y - mu)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def calibration_in_the_large(p: np.ndarray, y: np.ndarray, iters: int = 50) -> float:
    """Intercepto con la pendiente fijada en 1 (offset logit(p))."""
    off = _logit(p)
    a = 0.0
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(a + off)))
        g = (y - mu).sum()
        h = (mu * (1 - mu)).sum() + 1e-9
        a += g / h
        if abs(g / h) < 1e-8:
            break
    return float(a)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        s = idx == b
        if s.any():
            tot += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(tot)


def calibration_curve(p: np.ndarray, y: np.ndarray, bins: int = 10, min_n: int = 10) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        s = idx == b
        if s.sum() >= min_n:
            out.append({"p_mean": float(p[s].mean()), "obs": float(y[s].mean()), "n": int(s.sum())})
    return out


def summary(p: np.ndarray, y: np.ndarray, ordinal: bool, severe_class: int | None) -> dict:
    C = p.shape[1]
    out = {"n": int(len(y)), "logloss": float(row_logloss(p, y).mean()), "brier": float(row_brier(p, y).mean())}
    if ordinal:
        out["rps"] = float(row_rps(p, y).mean())
    cal = {}
    aucs = []
    for c in range(C):
        yc = (y == c).astype(float)
        if 0 < yc.sum() < len(yc):
            a, b = calibration_line(p[:, c], yc)
            cal[str(c)] = {"intercept": a, "slope": b, "citl": calibration_in_the_large(p[:, c], yc),
                           "ece": ece(p[:, c], yc), "curve": calibration_curve(p[:, c], yc)}
            aucs.append(roc_auc_score(yc, p[:, c]))
    out["calibration"] = cal
    out["auroc_ovr_macro"] = float(np.mean(aucs)) if aucs else float("nan")
    if severe_class is not None:
        ys = (y == severe_class).astype(int)
        if 0 < ys.sum() < len(ys):
            out["auroc_severe"] = float(roc_auc_score(ys, p[:, severe_class]))
            out["auprc_severe"] = float(average_precision_score(ys, p[:, severe_class]))
            out["prevalence_severe"] = float(ys.mean())
    return out


# --------------------------------------------------------------------------- bootstrap por paciente
def _subject_index(subjects: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(subjects, return_inverse=True)
    return inv, int(inv.max()) + 1


def bootstrap_rowwise(values: dict[str, np.ndarray], subjects: np.ndarray, n_boot: int, seed: int) -> dict:
    """IC 95 % por bootstrap de pacientes para métricas promedio por fila (log-loss, Brier, RPS) y sus
    diferencias pareadas. `values`: {nombre: pérdida por fila}."""
    inv, S = _subject_index(subjects)
    rng = np.random.default_rng(seed)
    sums = {k: np.bincount(inv, weights=v, minlength=S) for k, v in values.items()}
    cnt = np.bincount(inv, minlength=S).astype(float)
    boots = {k: [] for k in sums}
    for start in range(0, n_boot, 200):  # por bloques para acotar memoria (bloque × pacientes)
        W = rng.multinomial(S, np.full(S, 1.0 / S), size=min(200, n_boot - start)).astype(float)
        den = W @ cnt
        for k, s in sums.items():
            boots[k].append((W @ s) / den)
    out = {}
    for k, s in sums.items():
        bs = np.concatenate(boots[k])
        out[k] = {"mean": float(s.sum() / cnt.sum()), "ci_low": float(np.percentile(bs, 2.5)),
                  "ci_high": float(np.percentile(bs, 97.5)), "_boot": bs}
    return out


def bootstrap_metric(fn, p: np.ndarray, y: np.ndarray, subjects: np.ndarray, n_boot: int, seed: int) -> dict:
    """IC 95 % por bootstrap de pacientes para una métrica no descomponible (AUROC, AUPRC, pendiente)."""
    inv, S = _subject_index(subjects)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(S))
    ends = np.append(starts[1:], len(inv))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, S, S)
        rows = np.concatenate([order[starts[s]:ends[s]] for s in pick])
        try:
            vals.append(fn(p[rows], y[rows]))
        except ValueError:
            continue
    vals = np.array([v for v in vals if np.isfinite(v)])
    point = fn(p, y)
    if vals.size == 0:
        return {"mean": float(point), "ci_low": float("nan"), "ci_high": float("nan")}
    return {"mean": float(point), "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5))}


def paired_difference(a: np.ndarray, b: np.ndarray, subjects: np.ndarray, n_boot: int, seed: int) -> dict:
    """Diferencia media (a − b) de pérdidas por fila con bootstrap pareado por paciente."""
    r = bootstrap_rowwise({"d": a - b}, subjects, n_boot, seed)["d"]
    bs = r.pop("_boot")
    r["p_two_sided"] = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return r
