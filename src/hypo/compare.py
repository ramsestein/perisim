"""Local comparators (design §4 "Comparators"): each fitted on one site's training landmarks only.

- prevalence: the training landmarks' rate;
- last_map: logistic regression on the last MAP and its 6-h slope (the honest bar for an outcome this close to an
  input; slope missing → training median + indicator);
- logreg: L2 logistic regression on every input (perisim's model matrix: standardized values, missing indicators,
  one-hot surgical group), C tuned in the inner folds (T2's grid);
- lightgbm: T2's LightGBM, grid × early stopping in the inner folds and temperature scaling;
- euclidean: kNN with the same case base, neighbours and vote as PERISIM, on the neutral axes (ranks, one-hot,
  0 / 1) with uniform weights: no outcome information in the metric.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from perisim.baselines.parametric import fit_lgbm_calibrated, fit_logreg
from perisim.metric.encoding import Encoder

from .casebase import CaseBase
from .data import Landmarks
from .metric import C, Metric


def inner_folds(tr: Landmarks, n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    sgk = StratifiedGroupKFold(n, shuffle=True, random_state=seed)
    return list(sgk.split(np.zeros(len(tr)), tr.y, groups=tr.ft.subjects))


def _last_map_X(lm: Landmarks, med: float) -> np.ndarray:
    s = lm.ft.num["map_slope_6h"]
    miss = ~np.isfinite(s)
    return np.column_stack([lm.ft.num["map_last"], np.where(miss, med, s), miss.astype(float)])


def fit(name: str, tr: Landmarks, mcfg: dict, baselines: dict, *, n_inner: int, seed: int, threads: int):
    """(predict: Landmarks → p, info) of one comparator fitted on the training landmarks `tr`."""
    y = tr.y
    if name == "prevalence":
        rate = float(y.mean())
        return (lambda q: np.full(len(q), rate)), {"rate": rate}
    if name == "last_map":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        med = float(np.nanmedian(tr.ft.num["map_slope_6h"]))
        X = _last_map_X(tr, med)
        sc = StandardScaler().fit(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(C=1e4, max_iter=2000).fit(sc.transform(X), y)
        return (lambda q: m.predict_proba(sc.transform(_last_map_X(q, med)))[:, 1]), \
            {"coef_per_sd": [float(c) for c in m.coef_[0]]}
    if name in ("logreg", "lightgbm"):
        enc = Encoder(geometry="shape").fit(tr.ft, y, C)
        X = enc.transform_model(tr.ft)[0]
        inner = inner_folds(tr, n_inner, seed)
        if name == "logreg":
            pred, info = fit_logreg(X, y, baselines["logreg_C_grid"], inner, C)
        else:
            pred, info = fit_lgbm_calibrated(X, y, baselines["lgbm"], inner, C, seed, threads, baselines["lgbm_grid"])
        return (lambda q: pred(enc.transform_model(q.ft)[0])[:, 1]), info
    if name == "euclidean":
        enc = Encoder(geometry="shape").fit(tr.ft, y, C)
        cb = CaseBase(Metric(enc, np.ones(enc.n_features)), tr, mcfg)
        info = cb.fit_alpha(seed)
        return (lambda q: cb.predict(q)["p"]), info
    raise ValueError(name)
