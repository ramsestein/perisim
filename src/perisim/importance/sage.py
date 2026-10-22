"""SAGE por permutaciones sobre log-loss (Covert et al., 2020), con ausencia simulada por la máscara del
sustituto. Para cada caso se recorre una permutación aleatoria de los grupos y la contribución de cada grupo
es la reducción de la pérdida al añadirlo. Se para por convergencia (desviación típica máxima relativa al
rango de los valores < umbral) o al agotar `max_passes` pasadas por la submuestra de explicación.
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
    """Trunca negativos a 0 y divide por la suma (§5.3.6). El error típico se escala igual."""
    v = np.clip(values, 0, None)
    s = v.sum()
    if s <= 0:
        return np.full_like(v, 1.0 / len(v)), se
    return v / s, se / s


def aggregate(imp: np.ndarray, se: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Media ponderada por la inversa de la varianza, por grupo, al estilo de ShapleyVIC (§5.3.7).
    imp, se: (modelos × grupos)."""
    w = 1.0 / np.maximum(se ** 2, floor)
    agg = (w * imp).sum(0) / w.sum(0)
    s = agg.sum()
    return agg / s if s > 0 else np.full(imp.shape[1], 1.0 / imp.shape[1])
