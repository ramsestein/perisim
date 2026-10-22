"""Parametric comparators: L2 multinomial logistic regression, recalibrated multiclass LightGBM
(temperature scaling) and SWIFT (Gajic et al., 2008) for severe / not severe."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize_scalar

from perisim.evaluation.metrics import row_logloss


def _align(p: np.ndarray, classes: np.ndarray, C: int) -> np.ndarray:
    out = np.full((p.shape[0], C), 1e-6)
    out[:, classes] = p
    return out / out.sum(1, keepdims=True)


def fit_logreg(X: np.ndarray, y: np.ndarray, C_grid: list[float], inner_folds: list[tuple[np.ndarray, np.ndarray]],
               n_classes: int):
    from sklearn.linear_model import LogisticRegression

    best = (np.inf, C_grid[0])
    for Cc in C_grid:
        ll = []
        for tr, va in inner_folds:
            m = LogisticRegression(C=Cc, max_iter=2000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X[tr], y[tr])
            ll.append(row_logloss(_align(m.predict_proba(X[va]), m.classes_, n_classes), y[va]).mean())
        if np.mean(ll) < best[0]:
            best = (float(np.mean(ll)), Cc)
    m = LogisticRegression(C=best[1], max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
    return lambda Xq: _align(m.predict_proba(Xq), m.classes_, n_classes), {"C": best[1]}


def fit_lgbm_calibrated(X: np.ndarray, y: np.ndarray, params: dict, inner_folds, n_classes: int, seed: int, threads: int):
    """LightGBM + temperature fitted on inner out-of-fold predictions."""
    import lightgbm as lgb

    oof_logit, oof_y = [], []
    for tr, va in inner_folds:
        m = lgb.LGBMClassifier(**params, n_jobs=threads, random_state=seed, verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[tr], y[tr])
        p = _align(m.predict_proba(X[va]), m.classes_, n_classes)
        oof_logit.append(np.log(np.clip(p, 1e-12, 1)))
        oof_y.append(y[va])
    L = np.vstack(oof_logit)
    yy = np.concatenate(oof_y)

    def nll(T):
        z = L / T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return row_logloss(p, yy).mean()

    T = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
    m = lgb.LGBMClassifier(**params, n_jobs=threads, random_state=seed, verbose=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)

    def predict(Xq):
        z = np.log(np.clip(_align(m.predict_proba(Xq), m.classes_, n_classes), 1e-12, 1)) / T
        z = z - z.max(1, keepdims=True)
        return np.exp(z) / np.exp(z).sum(1, keepdims=True)

    return predict, {"temperature": T}


def swift_score(icu_los_h, pf_ratio, gcs, paco2, admit_source) -> np.ndarray:
    """SWIFT (Gajic et al., 2008). Missing values → reference category (0 points) [VERIFY]."""
    los_d = np.asarray(icu_los_h, float) / 24.0
    s = np.where(los_d > 10, 14, np.where(los_d >= 2, 1, 0)).astype(float)
    pf = np.asarray(pf_ratio, float)
    s += np.where(np.isfinite(pf), np.where(pf < 100, 13, np.where(pf < 150, 10, np.where(pf < 400, 5, 0))), 0)
    g = np.asarray(gcs, float)
    s += np.where(np.isfinite(g), np.where(g < 8, 24, np.where(g <= 10, 14, np.where(g < 14, 6, 0))), 0)
    pc = np.asarray(paco2, float)
    s += np.where(np.isfinite(pc) & (pc > 45), 5, 0)
    src = np.asarray(admit_source, dtype=object)
    s += np.where(np.isin(src, ["other_hospital", "ward"]), 8, 0)
    return s


def fit_swift(score: np.ndarray, y_bin: np.ndarray):
    """1-D logistic recalibration of SWIFT on the training fold → p(severe)."""
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1e6, max_iter=1000)
    m.fit(score.reshape(-1, 1), y_bin)
    return lambda s: m.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1], \
        {"coef": float(m.coef_[0, 0]), "intercept": float(m.intercept_[0])}
