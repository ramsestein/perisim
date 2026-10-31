"""The realistic ML arm's monitor and triggers (design §7.2–§7.4).

- The monitoring window: the last M evaluated patients since the deployment, extended back until it holds E event
  landmarks; no decision without them.
- Primary trigger: O/E outside [0.9, 1.1], or the patient-cluster robust 95 % CI of the calibration in the large
  entirely outside ±0.10.
- Sensitivity triggers: relative AUROC drop ≥ 5 % or ≥ 10 % against the baseline; relative log-loss rise ≥ 10 %;
  Davis et al. (2020), reimplemented: a dynamic calibration curve logit p̂ = β · (1, √p, √p · ln p) (a second-degree
  fractional polynomial with the power 0.5 repeated), initialised by a GLM on the deployment's window and updated by
  Adam once per step on the step's landmarks; the per-patient mean of |p̂ − p| goes to a one-sided ADWIN (Bifet &
  Gavaldà, 2007, the variance form of the cut they use in practice: ε = √(2σ²/m · ln(2/δ′)) + 2/(3m) · ln(2/δ′),
  δ′ = δ / ln n, σ² the window's variance, m the harmonic mean of the two sub-windows' sizes), cut points every step,
  each sub-window at least `min_sub` patients.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from hypo.stats import citl_w, logit, logloss_rows, sigmoid


def citl_se(p: np.ndarray, y: np.ndarray, code: np.ndarray) -> tuple[float, float]:
    """Calibration in the large and its patient-cluster robust SE."""
    off = logit(p)
    a = citl_w(off, y.astype(np.float64))
    mu = sigmoid(a + off)
    _, inv = np.unique(code, return_inverse=True)
    score = np.bincount(inv, weights=y - mu)
    return a, float(np.sqrt((score ** 2).sum()) / max((mu * (1 - mu)).sum(), 1e-12))


def stats(p: np.ndarray, y: np.ndarray, code: np.ndarray) -> dict:
    y = y.astype(np.float64)
    a, se = citl_se(p, y, code)
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < y.size else float("nan")
    return {"oe": float(y.sum() / max(p.sum(), 1e-12)), "citl": a, "citl_se": se, "auroc": auc,
            "logloss": float(logloss_rows(p, y).mean()), "events": int(y.sum()), "landmarks": int(y.size),
            "patients": int(np.unique(code).size)}


def primary_fires(s: dict, t: dict) -> bool:
    lo, hi = s["citl"] - t["z"] * s["citl_se"], s["citl"] + t["z"] * s["citl_se"]
    return bool(s["oe"] < t["oe_low"] or s["oe"] > t["oe_high"] or lo > t["citl_band"] or hi < -t["citl_band"])


def sensitivity_fires(kind: str, s: dict, baseline: dict, tc: dict) -> bool:
    if kind.startswith("auroc"):
        return bool(np.isfinite(s["auroc"]) and s["auroc"] < (1 - tc["auroc_drop"]) * baseline["auroc"])
    if kind.startswith("logloss"):
        return bool(s["logloss"] > (1 + tc["logloss_rise"]) * baseline["logloss"])
    raise ValueError(kind)


def monitoring_rows(units: list[np.ndarray], M: int, E: int, y: np.ndarray) -> np.ndarray | None:
    """Rows of the monitoring window from the per-patient rows of the patients evaluated since the deployment
    (oldest first): the last M, extended back until E event landmarks; None if fewer than M patients or E events."""
    if len(units) < M:
        return None
    take, ev = [], 0
    for r in reversed(units):
        take.append(r)
        ev += int(y[r].sum())
        if len(take) >= M and ev >= E:
            return np.concatenate(take[::-1])
    return None


class Davis:
    """Dynamic calibration curve and one-sided ADWIN on the calibration error (Davis et al., 2020)."""

    def __init__(self, delta: float, lr: float, min_sub: int, step: int):
        self.delta, self.lr, self.min_sub, self.step = float(delta), float(lr), int(min_sub), int(step)
        self.beta = np.zeros(3)
        self.m = np.zeros(3)
        self.v = np.zeros(3)
        self.t = 0
        self.err: list[float] = []

    @staticmethod
    def basis(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        s = np.sqrt(p)
        return np.column_stack([np.ones_like(p), s, s * np.log(p)])

    def reset(self, p: np.ndarray, y: np.ndarray) -> None:
        """A GLM of y on the basis of p (Newton), on the window that chose the deployed model."""
        X, y = self.basis(p), y.astype(np.float64)
        b = np.zeros(3)
        for _ in range(50):
            mu = sigmoid(X @ b)
            H = X.T @ (X * (mu * (1 - mu) + 1e-9)[:, None]) + 1e-6 * np.eye(3)
            step = np.linalg.solve(H, X.T @ (y - mu))
            b += step
            if np.abs(step).max() < 1e-8:
                break
        self.beta, self.m, self.v, self.t, self.err = b, np.zeros(3), np.zeros(3), 0, []

    def observe(self, p: np.ndarray, y: np.ndarray, code: np.ndarray) -> None:
        """One step's evaluated landmarks: their errors with the current curve, then one Adam step."""
        X = self.basis(p)
        phat = sigmoid(X @ self.beta)
        _, inv = np.unique(code, return_inverse=True)
        e = np.bincount(inv, weights=np.abs(phat - p)) / np.bincount(inv)
        self.err.extend(e.tolist())
        g = X.T @ (phat - y) / len(y)
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g ** 2
        mh, vh = self.m / (1 - 0.9 ** self.t), self.v / (1 - 0.999 ** self.t)
        self.beta = self.beta - self.lr * mh / (np.sqrt(vh) + 1e-8)

    def fires(self) -> bool:
        x = np.asarray(self.err)
        n = x.size
        if n < 2 * self.min_sub:
            return False
        cs = np.r_[0.0, np.cumsum(x)]
        var = float(x.var())
        lg = np.log(2.0 * np.log(n) / self.delta)
        for t in range(self.min_sub, n - self.min_sub + 1, self.step):
            n0, n1 = t, n - t
            mu0, mu1 = cs[t] / n0, (cs[n] - cs[t]) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = np.sqrt(2.0 / m * var * lg) + 2.0 / (3.0 * m) * lg
            if mu1 - mu0 > eps:
                return True
        return False
