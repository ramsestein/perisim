"""The ML arms' models and their update hierarchy (design §6.2, §7.5).

A model is a base predictor (the transported comparator, or a refit) under a recalibration of its logit,
logit p = a + b · logit p₀. The hierarchy (Janssen et al., 2008): 0 keep, 1 recalibration in the large, 2 logistic
recalibration, 3 full refit. The window's patients are split 2 : 1 by a seeded draw; every candidate is fitted on
the fitting part and scored on the validation part; the tests are one-sided (α = 0.05) on per-patient sums of
log-loss differences. The choice (design §7.5, as corrected after the dry run):
- the nested recalibrations by the closed testing procedure of Vergouwe et al. (2017), top-down: 2 against 0 (if
  not better, keep), then 2 against 1 (if not better, 1; else 2);
- then the refit, which does not nest the recalibrations, against the recalibration chosen: 3 if better.
A top-down test from the refit stopped at "keep" whenever a refit on a small window lost to a badly miscalibrated
original; a bottom-up sequence missed changes of the slope alone. The chosen candidate is refitted on the whole
window. No hyperparameter is re-tuned:
logistic regression keeps its C; LightGBM its num_leaves and learning rate, its trees found by early stopping on an
inner patient split, then temperature scaling on that split (perisim's `fit_lgbm_calibrated` with one split).
"""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from hypo.stats import calibration_line_w, citl_w, logit, logloss_rows, sigmoid

CHOICES = ("keep", "intercept", "logistic", "refit")


@dataclass
class Model:
    family: str
    base: Callable[[np.ndarray], np.ndarray]      # pool rows → p₀
    a: float = 0.0
    b: float = 1.0
    origin: str = "transported"
    n_base_params: int = 0
    history: list = field(default_factory=list)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        p0 = self.base(rows)
        if self.a == 0.0 and self.b == 1.0:
            return np.asarray(p0, dtype=np.float64)
        return sigmoid(self.a + self.b * logit(p0))

    def digest(self, rows: np.ndarray) -> str:
        h = hashlib.sha256(np.asarray(self.predict(rows), dtype=np.float64).tobytes())
        h.update(f"{self.family}|{self.origin}|{self.a!r}|{self.b!r}".encode())
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- refits
def lgbm_split(groups: np.ndarray, y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """One inner patient split 3 : 1 for early stopping and the temperature (stratified by patient)."""
    from sklearn.model_selection import StratifiedGroupKFold

    sgk = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
    tr, va = next(iter(sgk.split(np.zeros(len(y)), y, groups=groups)))
    return [(tr, va)]


def refit(family: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, hyper: dict, space: dict, ucfg: dict,
          seed: int, threads: int) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
    """(predict: X → p₁, number of fitted parameters) of a full refit with the arm's hyperparameters."""
    if family == "logreg":
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=float(hyper["C"]), max_iter=2000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y)
        cls = list(m.classes_)
        if len(cls) == 1:
            const = float(cls[0])
            return (lambda Xq: np.full(len(Xq), const)), 1
        return (lambda Xq: m.predict_proba(Xq)[:, 1]), int(X.shape[1] + 1)
    if family == "lightgbm":
        from perisim.baselines.parametric import fit_lgbm_calibrated

        grid = {"num_leaves": [int(hyper["num_leaves"])], "learning_rate": [float(hyper["learning_rate"])],
                "max_estimators": int(ucfg["lgbm_max_estimators"]), "early_stopping_rounds": int(ucfg["lgbm_early_stopping"])}
        pred, info = fit_lgbm_calibrated(X, y, space["baselines"]["lgbm"], lgbm_split(groups, y, seed), 2, seed,
                                         threads, grid)
        cells = dict(zip(pred.__code__.co_freevars, [c.cell_contents for c in pred.__closure__]))
        trees = cells["m"].booster_.dump_model()["tree_info"]
        return (lambda Xq: pred(Xq)[:, 1]), int(sum(t["num_leaves"] for t in trees)) + 1
    raise ValueError(family)


# --------------------------------------------------------------------------- closed testing
def _better(ll_new: np.ndarray, ll_ref: np.ndarray, code: np.ndarray, alpha: float) -> tuple[bool, float]:
    """One-sided test that ll_new is lower: z of the mean per-patient sum of differences."""
    _, inv = np.unique(code, return_inverse=True)
    d = np.bincount(inv, weights=ll_new - ll_ref)
    if d.size < 2 or d.std(ddof=1) == 0:
        return False, 0.0
    z = float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
    return z < norm.ppf(alpha), z


def _perf(p: np.ndarray, y: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < y.size else float("nan")
    return {"auroc": auc, "logloss": float(logloss_rows(p, y).mean())}


def closed_test(model: Model, rows: np.ndarray, y_all: np.ndarray, pcode_all: np.ndarray, X: np.ndarray,
                hyper: dict, space: dict, ucfg: dict, seed: int, threads: int) -> tuple[Model, dict]:
    """The update chosen on the window `rows` (pool rows), and its record."""
    t = time.perf_counter()
    y, code = y_all[rows].astype(np.float64), pcode_all[rows]
    pats = np.unique(code)
    rng = np.random.default_rng(seed)
    nfit = int(round(float(ucfg["split_fit"]) * pats.size))
    fitp = np.isin(code, rng.permutation(pats)[:nfit])
    f, v = rows[fitp], rows[~fitp]
    yf, yv, cv = y[fitp], y[~fitp], code[~fitp]
    lp_f, lp_v = logit(model.predict(f)), logit(model.predict(v))
    a1 = citl_w(lp_f, yf)
    a2, b2 = calibration_line_w(lp_f, yf)
    p3fn, _ = refit(model.family, X[f], yf.astype(int), code[fitp], hyper, space, ucfg, seed, threads)
    pv = {"keep": sigmoid(lp_v), "intercept": sigmoid(a1 + lp_v), "logistic": sigmoid(a2 + b2 * lp_v),
          "refit": np.asarray(p3fn(X[v]), dtype=np.float64)}
    ll = {k: logloss_rows(p, yv) for k, p in pv.items()}
    alpha = float(ucfg["alpha"])
    tests = {}
    ok, tests["logistic_vs_keep"] = _better(ll["logistic"], ll["keep"], cv, alpha)     # the nested recalibrations,
    choice = "keep"                                                                     # top-down (Vergouwe et al.)
    if ok:
        ok, tests["logistic_vs_intercept"] = _better(ll["logistic"], ll["intercept"], cv, alpha)
        choice = "logistic" if ok else "intercept"
    ok, tests[f"refit_vs_{choice}"] = _better(ll["refit"], ll[choice], cv, alpha)     # then the refit, not nested
    if ok:
        choice = "refit"
    # the chosen candidate refitted on the whole window
    lp_all = logit(model.predict(rows))
    yi = y.astype(int)
    if choice == "keep":
        new, changed = model, 0
    elif choice == "intercept":
        a = citl_w(lp_all, y)
        new, changed = Model(model.family, model.base, model.a + a, model.b, model.origin, model.n_base_params), 1
    elif choice == "logistic":
        a, b = calibration_line_w(lp_all, y)
        new, changed = Model(model.family, model.base, a + b * model.a, b * model.b, model.origin,
                             model.n_base_params), 2
    else:
        fn, n_par = refit(model.family, X[rows], yi, code, hyper, space, ucfg, seed + 1, threads)
        new = Model(model.family, (lambda r, fn=fn: np.asarray(fn(X[r]), dtype=np.float64)), 0.0, 1.0,
                    "refit", n_par)
        changed = n_par
    rec = {"choice": choice, "tests_z": tests, "parameters_changed": int(changed),
           "baseline": _perf(pv[choice], yv), "n_window_patients": int(pats.size), "n_window_landmarks": int(len(rows)),
           "seconds": time.perf_counter() - t, "model_digest": new.digest(rows)}
    return new, rec
