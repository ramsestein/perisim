"""kNN inference (§5.5) and per-patient output (§5.6).

k is a run-time parameter: the metric is the same for any k. Per-class probability
p_c = (n_c + α·P(c)·C) / (k + α·C); variant with an exp(−d²) kernel as a sensitivity analysis.
Out of support (OOD): distance to the k-th neighbour > 99th percentile of that distance in training (LOO).
"""

from __future__ import annotations

import numpy as np

from perisim.knn.index import search_excluding


class KNNModel:
    def __init__(self, E_ref: np.ndarray, sqrt_w_dims: np.ndarray, y_ref: np.ndarray, n_classes: int,
                 subjects_ref: np.ndarray, k_grid: list[int], ids_ref: np.ndarray | None = None):
        self.Ew = np.ascontiguousarray(E_ref * sqrt_w_dims, dtype=np.float32)
        self.sw = sqrt_w_dims
        self.y = np.asarray(y_ref, dtype=np.int64)
        self.C = int(n_classes)
        self.subj = np.asarray(subjects_ref)
        self.k_grid = sorted(int(k) for k in k_grid)
        self.k_max = min(self.k_grid[-1], self.Ew.shape[0] - 1)
        self.prior = np.bincount(self.y, minlength=self.C) / len(self.y)
        self.ids = ids_ref
        self.alpha: dict[int, float] = {k: 1.0 for k in self.k_grid}
        self.ood_thr: dict[int, float] = {}
        self.loo_kth: dict[int, np.ndarray] = {}

    def neighbors(self, E_query: np.ndarray | None, subjects_query: np.ndarray | None = None, loo: bool = False):
        Q = self.Ew if loo else np.ascontiguousarray(E_query * self.sw, dtype=np.float32)
        sq = self.subj if loo else subjects_query
        return search_excluding(self.Ew, Q, self.k_max, sq, self.subj if sq is not None else None, self_query=loo)

    def counts(self, I: np.ndarray, valid: np.ndarray, k: int, labels: np.ndarray | None = None,
               n_classes: int | None = None, D: np.ndarray | None = None, kernel: bool = False) -> np.ndarray:
        lab = self.y if labels is None else labels
        C = self.C if n_classes is None else n_classes
        Ik, vk = I[:, :k], valid[:, :k]
        L = lab[Ik]
        wts = np.exp(-D[:, :k].astype(np.float64)) if kernel and D is not None else np.ones(Ik.shape)
        wts = wts * vk * (L >= 0)
        out = np.zeros((I.shape[0], C))
        for c in range(C):
            out[:, c] = (wts * (L == c)).sum(1)
        return out

    def probs(self, I, valid, k, alpha=None, D=None, kernel=False) -> np.ndarray:
        a = self.alpha.get(k, 1.0) if alpha is None else alpha
        n = self.counts(I, valid, k, D=D, kernel=kernel)
        tot = n.sum(1, keepdims=True)
        p = (n + a * self.prior * self.C) / (tot + a * self.C)
        bad = ~np.isfinite(p).all(1)
        p[bad] = self.prior
        return p

    def fit_alpha_and_ood(self, alpha_grid: list[float], k_primary: int, ood_percentile: float) -> dict:
        """α per k with patient-level LOO on the reference (within the training fold) and OOD threshold."""
        D, I, valid = self.neighbors(None, loo=True)
        y = self.y
        res = {}
        for k in self.k_grid:
            kk = min(k, self.k_max)
            best = (np.inf, 1.0)
            for a in alpha_grid:
                p = self.probs(I, valid, kk, alpha=a)
                ll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))))
                if ll < best[0]:
                    best = (ll, float(a))
            self.alpha[k] = best[1]
            kth = np.sqrt(D[np.arange(len(y)), np.minimum(valid.sum(1), kk) - 1])
            self.loo_kth[k] = np.sort(kth)
            self.ood_thr[k] = float(np.percentile(kth, ood_percentile))
            res[k] = {"alpha": best[1], "loo_logloss": best[0], "ood_threshold": self.ood_thr[k]}
        return res

    def kth_info(self, D: np.ndarray, valid: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kk = min(k, self.k_max)
        kth = np.sqrt(D[np.arange(D.shape[0]), np.maximum(np.minimum(valid.sum(1), kk) - 1, 0)])
        ref = self.loo_kth.get(k)
        pct = (np.searchsorted(ref, kth, side="right") / len(ref) * 100.0) if ref is not None else np.full(len(kth), np.nan)
        ood = kth > self.ood_thr.get(k, np.inf)
        return kth, pct, ood

    def carried_summary(self, I: np.ndarray, valid: np.ndarray, k: int, carried: dict[str, np.ndarray],
                        alpha: float = 1.0) -> dict[str, np.ndarray]:
        """Summary of carried variables over the k neighbours (smoothed proportions; median for continuous)."""
        kk = min(k, self.k_max)
        out = {}
        for name, v in carried.items():
            vals = v[I[:, :kk]].astype(float)
            vals[~valid[:, :kk]] = np.nan
            if name.startswith("los"):
                out[name] = np.nanmedian(vals, axis=1)
            else:
                prev = np.nanmean(v.astype(float))
                num = np.nansum(vals, 1) + alpha * prev
                den = np.sum(np.isfinite(vals), 1) + alpha
                out[name] = num / den
        return out


def patient_records(ids, t0, p_t1, p_t2, k, kth_pct, ood, I, D, ref_ids, carried, max_neighbors: int = 20) -> list[dict]:
    """Per-patient output in the §5.6 format. CONTAINS identifiers: never exported."""
    t1_lab = ["none", "mild", "severe"]
    t2_lab = ["short", "medium", "long"]
    recs = []
    for i in range(len(ids)):
        recs.append({
            "patient_id": str(ids[i]), "t0": float(t0[i]) if t0 is not None else None,
            "p_T1": {t1_lab[c]: float(p_t1[i, c]) for c in range(p_t1.shape[1])} if p_t1 is not None else None,
            "p_T2": {t2_lab[c]: float(p_t2[i, c]) for c in range(p_t2.shape[1])} if p_t2 is not None else None,
            "k": int(k), "ood": bool(ood[i]), "kth_distance_percentile": float(kth_pct[i]),
            "neighbors": [{"case_id": str(ref_ids[j]), "distance": float(np.sqrt(d))}
                          for j, d in zip(I[i, :max_neighbors], D[i, :max_neighbors])],
            "carried_summary": {kk: (float(v[i]) if np.isfinite(v[i]) else None) for kk, v in carried.items()},
        })
    return recs
