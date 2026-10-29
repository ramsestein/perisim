"""A site's case base: every landmark of its patients, queried with the frozen metric.

Neighbours are distinct patients (`metric.distinct_patients`): each other patient enters once, at its most similar
moment, and the query's own patient never enters (patient-level exclusion). The probability is perisim's smoothed
vote, p_c = (n_c + α·P(c)·C) / (k + α·C), with P(c) the case base's prevalence and α chosen by a patient-level LOO
over a random subset of the case base's own landmarks; the out-of-support threshold is the 99th percentile of the
LOO distance to the k-th neighbour, as T2.
"""

from __future__ import annotations

import numpy as np

from perisim.knn.index import search
from perisim.knn.predict import KNNModel

from .data import Landmarks
from .metric import C, Metric


def search_distinct(Ew: np.ndarray, code: np.ndarray, Qw: np.ndarray, qc: np.ndarray, k: int, fetch: int,
                    distinct: bool = True, chunk: int = 50000):
    """(D, I, valid) of the k nearest reference rows of each query in a weighted embedding: never the query's own
    patient (`code` / `qc`: patient codes, −1 = none) and, with `distinct`, each patient once, at its closest row.
    D is the squared distance."""
    k = min(k, Ew.shape[0])
    kk = min(k * (fetch if distinct else 1) + 25, Ew.shape[0])
    out = []
    for s in range(0, Qw.shape[0], chunk):
        Q = np.ascontiguousarray(Qw[s:s + chunk], dtype=np.float32)
        D, I = search(Ew, Q, kk)
        S = code[I]
        bad = S == qc[s:s + chunk, None]
        if distinct:                           # keep each patient's first (closest) appearance only
            o = np.argsort(S, axis=1, kind="stable")
            Ss = np.take_along_axis(S, o, 1)
            first_s = np.ones_like(Ss, dtype=bool)
            first_s[:, 1:] = Ss[:, 1:] != Ss[:, :-1]
            first = np.empty_like(first_s)
            np.put_along_axis(first, o, first_s, 1)
            bad |= ~first
        o = np.argsort(bad, axis=1, kind="stable")[:, :k]
        out.append((np.take_along_axis(D, o, 1), np.take_along_axis(I, o, 1), ~np.take_along_axis(bad, o, 1)))
    return tuple(np.vstack([x[i] for x in out]) for i in range(3))


class CaseBase(KNNModel):
    def __init__(self, metric: Metric, lm: Landmarks, mcfg: dict, E: np.ndarray | None = None):
        E = metric.embed(lm) if E is None else E
        self.k = int(mcfg["k"])
        super().__init__(E, metric.sqrt_w_dims, lm.y, C, lm.ft.subjects, [self.k], ids_ref=lm.ft.ids)
        self.E, self.metric, self.lm, self.mcfg = E, metric, lm, mcfg
        self.distinct = bool(mcfg.get("distinct_patients", True))
        self.fetch = int(mcfg.get("fetch_factor", 6)) if self.distinct else 1
        self.subj_s = np.asarray(lm.ft.subjects, dtype=str)
        self.uniq, self.code = np.unique(self.subj_s, return_inverse=True)

    # ------------------------------------------------------------------ neighbours
    def _codes(self, subjects: np.ndarray | None, n: int) -> np.ndarray:
        if subjects is None:
            return np.full(n, -1)
        s = np.asarray(subjects, dtype=str)
        pos = np.minimum(np.searchsorted(self.uniq, s), self.uniq.size - 1)
        return np.where(self.uniq[pos] == s, pos, -1)

    def search(self, Eq: np.ndarray, subjects_q: np.ndarray | None, chunk: int = 50000, k: int | None = None):
        """(D, I, valid) of the k neighbours of the queries (embedded, unweighted), with the exclusions above."""
        return search_distinct(self.Ew, self.code, np.asarray(Eq) * self.sw, self._codes(subjects_q, Eq.shape[0]),
                               self.k if k is None else k, self.fetch, self.distinct, chunk)

    def neighbors(self, E_query, subjects_query=None, loo: bool = False):
        if loo:
            raise NotImplementedError("LOO over the whole case base: use fit_alpha (a random subset of queries)")
        return self.search(E_query, subjects_query)

    # ------------------------------------------------------------------ fitting α and the OOD threshold
    def fit_alpha(self, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        n = self.Ew.shape[0]
        rows = rng.choice(n, size=min(int(self.mcfg["alpha_queries"]), n), replace=False)
        D, I, valid = self.search(self.E[rows], self.subj_s[rows])
        self.loo = {"rows": rows, "D": D, "I": I, "valid": valid}       # kept for votes re-weighted later (H5)
        y = self.y[rows]
        best = (np.inf, 1.0)
        for a in self.mcfg["alpha_grid"]:
            p = self.probs(I, valid, self.k, alpha=float(a))
            ll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))))
            best = min(best, (ll, float(a)))
        self.alpha[self.k] = best[1]
        nv = valid.sum(1)
        kth = np.sqrt(D[np.arange(len(y)), np.maximum(nv - 1, 0)])
        self.loo_kth[self.k] = np.sort(kth)
        self.ood_thr[self.k] = float(np.percentile(kth, float(self.mcfg["ood_percentile"])))
        return {"alpha": best[1], "loo_logloss": best[0], "ood_threshold": self.ood_thr[self.k],
                "short_neighbourhoods": float((nv < self.k).mean()), "n_queries": int(len(rows))}

    # ------------------------------------------------------------------ prediction
    def predict(self, q: Landmarks, E: np.ndarray | None = None) -> dict:
        """Probability of hypotension, the neighbours and what they carry, for each query landmark."""
        E = self.metric.embed(q) if E is None else E
        D, I, valid = self.search(E, q.ft.subjects)
        p = self.probs(I, valid, self.k)[:, 1]
        kth, pct, ood = self.kth_info(D, valid, self.k)
        return {"p": p, "D": D, "I": I, "valid": valid, "ood": ood, "kth_percentile": pct,
                "p_band": self.vote(I, valid, self.lm.band, 3)}

    def vote(self, I: np.ndarray, valid: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
        """Smoothed vote of the same neighbours for another outcome (labels −1 = unknown, left out)."""
        lab = np.asarray(labels)
        ok = lab >= 0
        prior = np.bincount(lab[ok], minlength=n_classes) / max(int(ok.sum()), 1)
        n = self.counts(I, valid, self.k, labels=lab, n_classes=n_classes)
        a = self.alpha.get(self.k, 1.0)
        return (n + a * prior * n_classes) / (n.sum(1, keepdims=True) + a * n_classes)

    def carried(self, I: np.ndarray, valid: np.ndarray, column: str) -> np.ndarray:
        """Share (binary) or median (continuous) of a carried column over each query's valid neighbours."""
        v = self.lm.frame[column].cast(float).to_numpy()[I]
        v = np.where(valid, v, np.nan)
        return np.nanmedian(v, axis=1) if self.lm.frame[column].dtype.is_float() else np.nanmean(v, axis=1)
