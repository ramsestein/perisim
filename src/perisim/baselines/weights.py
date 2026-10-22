"""Pesos de comparadores kNN: |SHAP| medio de LightGBM (Weerts et al., 2019), importancia de random forest
(Addou et al., 2026) y ReliefF. Todos devuelven una importancia por variable métrica (suma 1)."""

from __future__ import annotations

import warnings

import numpy as np

from perisim.knn.index import search


def _to_features(col_imp: np.ndarray, col_owner: np.ndarray, n_features: int) -> np.ndarray:
    imp = np.zeros(n_features)
    np.add.at(imp, col_owner, np.abs(col_imp))
    s = imp.sum()
    return imp / s if s > 0 else np.full(n_features, 1.0 / n_features)


def shap_lgbm(X: np.ndarray, y: np.ndarray, col_owner: np.ndarray, n_features: int, params: dict, seed: int,
              threads: int) -> np.ndarray:
    import lightgbm as lgb

    m = lgb.LGBMClassifier(**params, n_jobs=threads, random_state=seed, verbose=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
        contrib = m.predict(X, pred_contrib=True)
    contrib = np.asarray(contrib)
    P = X.shape[1]
    if contrib.ndim == 2 and contrib.shape[1] == P + 1:            # binario
        sh = np.abs(contrib[:, :P]).mean(0)
    else:                                                           # multiclase: bloques de (P+1) por clase
        k = contrib.shape[1] // (P + 1)
        sh = sum(np.abs(contrib[:, c * (P + 1): c * (P + 1) + P]).mean(0) for c in range(k))
    return _to_features(sh, col_owner, n_features)


def rf_importance(X: np.ndarray, y: np.ndarray, col_owner: np.ndarray, n_features: int, params: dict, seed: int,
                  threads: int) -> np.ndarray:
    from sklearn.ensemble import RandomForestClassifier

    m = RandomForestClassifier(**params, n_jobs=threads, random_state=seed)
    m.fit(X, y)
    return _to_features(m.feature_importances_, col_owner, n_features)


def relieff(E: np.ndarray, dim_feature: np.ndarray, n_features: int, y: np.ndarray, n_samples: int, n_neighbors: int,
            seed: int) -> np.ndarray:
    """ReliefF multiclase sobre el embedding normalizado (δ_j de §5.2): para cada muestra, k aciertos y k fallos
    por clase; w_j += (δ_j a fallos ponderados por prior − δ_j a aciertos) / k. Negativos → 0."""
    rng = np.random.default_rng(seed)
    n = E.shape[0]
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    classes = np.unique(y)
    prior = np.array([(y == c).mean() for c in classes])
    w = np.zeros(n_features)
    per_class = {c: np.where(y == c)[0] for c in classes}
    for ci, c in enumerate(classes):
        ref_idx = per_class[c]
        q = idx[y[idx] == c]
        if q.size == 0:
            continue
        for cj, c2 in enumerate(classes):
            pool = per_class[c2]
            k = min(n_neighbors + (1 if c2 == c else 0), pool.size)
            if k == 0:
                continue
            _, I = search(E[pool], E[q], k)
            nb = pool[I]
            if c2 == c:
                nb = nb[:, 1:] if k > 1 else nb
            diff2 = (E[q][:, None, :] - E[nb]) ** 2                 # q × k × D
            per_feat = np.zeros((q.size, n_features))
            np.add.at(per_feat.T, dim_feature, diff2.mean(1).T)
            delta = np.sqrt(per_feat).sum(0)
            if c2 == c:
                w -= delta
            else:
                w += delta * prior[cj] / (1 - prior[ci] + 1e-12)
    w = np.clip(w / len(idx), 0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(n_features, 1.0 / n_features)
