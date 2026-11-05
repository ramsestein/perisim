"""The case base of a PERISIM arm: `hypo`'s kNN on a set of rows of a stream's pool.

The search, the exclusions and the vote are `hypo`'s (`hypo.casebase`): distinct patients, never the query's own
patient, p = (n_1 + α·P(1)·2) / (n_valid + α·2) with P(1) the case base's prevalence, α chosen by a patient-level LOO
over a random subset of the case base, the out-of-support threshold the 99th percentile of the LOO distance to the
k-th neighbour. Here the embeddings are computed once for the whole pool and the case base is a sorted set of rows,
so a window can rotate every step without rebuilding anything. With the rows of a `hypo` case base, in the same
order, the predictions are `hypo`'s (tested).
"""

from __future__ import annotations

import hashlib

import numpy as np

from hypo.casebase import search_distinct

C = 2


class CaseBase:
    def __init__(self, Ew: np.ndarray, y: np.ndarray, pcode: np.ndarray, mcfg: dict):
        self.Ew_all, self.y_all, self.pcode_all = Ew, np.asarray(y, dtype=np.int64), np.asarray(pcode, dtype=np.int64)
        self.k = int(mcfg["k"])
        self.distinct = bool(mcfg.get("distinct_patients", True))
        self.fetch = int(mcfg.get("fetch_factor", 6)) if self.distinct else 1
        self.alpha_grid = [float(a) for a in mcfg["alpha_grid"]]
        self.alpha_queries = int(mcfg["alpha_queries"])
        self.ood_pct = float(mcfg["ood_percentile"])
        self.alpha = 1.0
        self.ood_thr = np.inf
        self.rows = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------ state
    def set_rows(self, rows: np.ndarray) -> None:
        self.rows = np.sort(np.asarray(rows, dtype=np.int64))
        self.Ew = self.Ew_all[self.rows]
        self.y = self.y_all[self.rows]
        self.code = self.pcode_all[self.rows]
        self.prior = np.bincount(self.y, minlength=C) / max(len(self.y), 1)

    def digest(self) -> str:
        return hashlib.sha256(self.rows.tobytes()).hexdigest()[:16]

    # ------------------------------------------------------------------ search and vote
    def search(self, qrows: np.ndarray, qcode: np.ndarray | None = None):
        qc = self.pcode_all[qrows] if qcode is None else qcode
        return search_distinct(self.Ew, self.code, self.Ew_all[qrows], qc, self.k, self.fetch, self.distinct)

    def vote(self, I: np.ndarray, valid: np.ndarray, alpha: float | None = None) -> np.ndarray:
        a = self.alpha if alpha is None else alpha
        L = self.y[I]
        n1 = ((L == 1) & valid).sum(1).astype(np.float64)
        n0 = ((L == 0) & valid).sum(1).astype(np.float64)
        return (n1 + a * self.prior[1] * C) / (n0 + n1 + a * C)

    def _loo(self, seed: int):
        rng = np.random.default_rng(seed)
        n = len(self.rows)
        pos = rng.choice(n, size=min(self.alpha_queries, n), replace=False)
        D, I, valid = search_distinct(self.Ew, self.code, self.Ew[pos], self.code[pos], self.k, self.fetch,
                                      self.distinct)
        return pos, D, I, valid

    def _threshold(self, D: np.ndarray, valid: np.ndarray) -> None:
        nv = valid.sum(1)
        kth = np.sqrt(D[np.arange(len(nv)), np.maximum(nv - 1, 0)])
        self.ood_thr = float(np.percentile(kth, self.ood_pct))

    def fit_alpha(self, seed: int) -> dict:
        """α by LOO over the case base, and the out-of-support threshold (as `hypo.casebase.CaseBase.fit_alpha`)."""
        pos, D, I, valid = self._loo(seed)
        y = self.y[pos]
        best = (np.inf, 1.0)
        for a in self.alpha_grid:
            p1 = self.vote(I, valid, alpha=a)
            p = np.where(y == 1, p1, 1 - p1)
            best = min(best, (float(-np.mean(np.log(np.clip(p, 1e-15, 1)))), a))
        self.alpha = best[1]
        self._threshold(D, valid)
        return {"alpha": self.alpha, "loo_logloss": best[0], "ood_threshold": self.ood_thr}

    def refresh_threshold(self, seed: int) -> float:
        """The out-of-support threshold of the current case base; α is left as it is."""
        _, D, _, valid = self._loo(seed)
        self._threshold(D, valid)
        return self.ood_thr

    def predict(self, qrows: np.ndarray) -> dict:
        D, I, valid = self.search(qrows)
        kk = min(self.k, len(self.rows) - 1)
        nv = np.minimum(valid.sum(1), kk)
        kth = np.sqrt(D[np.arange(len(qrows)), np.maximum(nv - 1, 0)])
        return {"p": self.vote(I, valid), "ood": kth > self.ood_thr, "kth": kth, "I": I, "valid": valid}
