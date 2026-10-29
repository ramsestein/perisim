"""Weighted calibration fits and the patient-level bootstrap of the external stage.

The fits are perisim's (`calibration_line`, `calibration_in_the_large`: Newton on y ~ a + b·logit(p)) with row
weights, so a bootstrap replicate is the same fit on multinomial patient weights; with unit weights they reproduce
perisim's estimates exactly (tested).
"""

from __future__ import annotations

import numpy as np

from perisim.evaluation.metrics import _logit


def logit(p: np.ndarray) -> np.ndarray:
    return _logit(np.asarray(p, dtype=np.float64))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def calibration_line_w(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, iters: int = 50) -> tuple[float, float]:
    """(intercept, slope) of y ~ a + b·x, x = logit(p), with row weights."""
    w = np.ones_like(x) if w is None else w
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = sigmoid(X @ beta)
        W = w * (mu * (1 - mu) + 1e-9)
        g = X.T @ (w * (y - mu))
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def citl_w(off: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, iters: int = 50) -> float:
    """Calibration in the large: intercept with the slope fixed at 1 (offset x = logit(p)), with row weights."""
    w = np.ones_like(off) if w is None else w
    a = 0.0
    for _ in range(iters):
        mu = sigmoid(a + off)
        g = float((w * (y - mu)).sum())
        h = float((w * mu * (1 - mu)).sum()) + 1e-9
        a += g / h
        if abs(g / h) < 1e-8:
            break
    return float(a)


def recalibrate(p_fit: np.ndarray, y_fit: np.ndarray, p_apply: np.ndarray) -> np.ndarray:
    """Logistic recalibration (intercept and slope on the logit) fitted on (p_fit, y_fit), applied to p_apply; the
    identity when the fitting set has a single class."""
    if y_fit.min() == y_fit.max():
        return np.asarray(p_apply, dtype=np.float64)
    a, b = calibration_line_w(logit(p_fit), y_fit.astype(np.float64))
    return sigmoid(a + b * logit(p_apply))


def label_shift(p: np.ndarray, pi_to: float, pi_from: float) -> np.ndarray:
    """Probabilities moved from a case base's base rate to another (odds × [π_to/(1−π_to)] / [π_from/(1−π_from)])."""
    r1, r0 = pi_to / max(pi_from, 1e-12), (1 - pi_to) / max(1 - pi_from, 1e-12)
    return p * r1 / (p * r1 + (1 - p) * r0)


def logloss_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class PatientBootstrap:
    """Multinomial patient weights, the same replicates for every statistic of a site (`seed` fixes them)."""

    def __init__(self, subjects: np.ndarray, n_boot: int, seed: int):
        self.uniq, self.inv = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
        self.n_boot, self.seed = int(n_boot), int(seed)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        S = self.uniq.size
        for _ in range(self.n_boot):
            yield rng.multinomial(S, np.full(S, 1.0 / S)).astype(np.float64)

    def rows(self, wp: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        """Row weights of the site's rows (or of the rows `idx`) from the patient weights of a replicate."""
        return wp[self.inv] if idx is None else wp[self.inv[idx]]


def ci(b: np.ndarray, level: float = 0.95) -> list[float]:
    b = np.asarray(b, dtype=np.float64)
    b = b[np.isfinite(b)]
    q = 100 * (1 - level) / 2
    return [float(np.percentile(b, q)), float(np.percentile(b, 100 - q))] if b.size else [float("nan")] * 2
