"""Neighbour search and the vote of the case base.

Distances are squared Euclidean distances on embeddings already scaled by √w (d² = Σ_j w_j ‖g_j(a) − g_j(b)‖²).
Exact search with FAISS when it is installed, numpy otherwise.

The case base votes: p_c = (n_c + α·P(c)·C) / (k + α·C), with P(c) the case base's class shares and α chosen by a
group-level leave-one-out over (a random subset of) the case base itself. Neighbours are never from the query's own
group (patient) and, with `distinct`, each group enters once, at its most similar row. Out of support: the distance
to the k-th neighbour is above the `ood_percentile` of that distance in the leave-one-out.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on the environment
    import faiss

    HAVE_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    HAVE_FAISS = False


def search(ref: np.ndarray, query: np.ndarray, k: int, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    ref = np.ascontiguousarray(ref, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    k = int(min(k, ref.shape[0]))
    if ref.shape[1] == 0:
        return np.zeros((query.shape[0], k), np.float32), np.tile(np.arange(k), (query.shape[0], 1))
    if HAVE_FAISS:
        index = faiss.IndexFlatL2(ref.shape[1])
        index.add(ref)
        D, I = index.search(query, k)
        return np.maximum(D, 0.0), I.astype(np.int64)
    r2 = (ref.astype(np.float64) ** 2).sum(1)
    Ds, Is = [], []
    for s in range(0, query.shape[0], chunk):
        q = query[s:s + chunk].astype(np.float64)
        d = (q ** 2).sum(1)[:, None] + r2[None, :] - 2 * q @ ref.T.astype(np.float64)
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        dp = np.take_along_axis(d, part, 1)
        o = np.argsort(dp, axis=1, kind="stable")
        Is.append(np.take_along_axis(part, o, 1))
        Ds.append(np.maximum(np.take_along_axis(dp, o, 1), 0.0))
    return np.vstack(Ds).astype(np.float32), np.vstack(Is).astype(np.int64)


def search_excluding(ref: np.ndarray, query: np.ndarray, k: int, query_groups: np.ndarray | None = None,
                     ref_groups: np.ndarray | None = None, self_query: bool = False, extra: int = 25):
    """k neighbours excluding the row itself (`self_query`) and rows of the same group. Returns (D, I, valid)."""
    kk = int(min(k + extra + (1 if self_query else 0), ref.shape[0]))
    D, I = search(ref, query, kk)
    bad = np.zeros_like(I, dtype=bool)
    if self_query:
        bad |= I == np.arange(query.shape[0])[:, None]
    if query_groups is not None and ref_groups is not None:
        bad |= ref_groups[I] == query_groups[:, None]
    order = np.argsort(bad, axis=1, kind="stable")[:, :k]
    return (np.take_along_axis(D, order, 1), np.take_along_axis(I, order, 1), ~np.take_along_axis(bad, order, 1))


def search_distinct(Ew: np.ndarray, code: np.ndarray, Qw: np.ndarray, qc: np.ndarray, k: int, fetch: int,
                    distinct: bool = True, chunk: int = 50000):
    """(D, I, valid) of the k nearest reference rows of each query in a weighted embedding: never the query's own
    group (`code` / `qc`: group codes, −1 = none) and, with `distinct`, each group once, at its closest row."""
    k = min(k, Ew.shape[0])
    kk = min(k * (fetch if distinct else 1) + 25, Ew.shape[0])
    out = []
    for s in range(0, Qw.shape[0], chunk):
        Q = np.ascontiguousarray(Qw[s:s + chunk], dtype=np.float32)
        D, I = search(Ew, Q, kk)
        S = code[I]
        bad = S == qc[s:s + chunk, None]
        if distinct:
            o = np.argsort(S, axis=1, kind="stable")
            Ss = np.take_along_axis(S, o, 1)
            first_s = np.ones_like(Ss, dtype=bool)
            first_s[:, 1:] = Ss[:, 1:] != Ss[:, :-1]
            first = np.empty_like(first_s)
            np.put_along_axis(first, o, first_s, 1)
            bad |= ~first
        o = np.argsort(bad, axis=1, kind="stable")[:, :k]
        out.append((np.take_along_axis(D, o, 1), np.take_along_axis(I, o, 1), ~np.take_along_axis(bad, o, 1)))
    if not out:
        z = np.zeros((0, k))
        return z.astype(np.float32), z.astype(np.int64), z.astype(bool)
    return tuple(np.vstack([x[i] for x in out]) for i in range(3))


class CaseBase:
    """The rows that vote: their embedding (weighted), class labels and groups."""

    def __init__(self, E: np.ndarray, sqrt_w_dims: np.ndarray, y: np.ndarray, n_classes: int, groups: np.ndarray,
                 *, k: int = 50, distinct: bool = True, fetch: int = 6):
        self.sw = np.asarray(sqrt_w_dims, dtype=np.float32)
        self.Ew = np.ascontiguousarray(E * self.sw, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.C = int(n_classes)
        self.groups = np.asarray(groups, dtype=str)
        self.uniq, self.code = np.unique(self.groups, return_inverse=True)
        self.k = int(min(k, max(len(self.y) - 1, 1)))
        self.distinct = bool(distinct)
        self.fetch = int(fetch) if self.distinct else 1
        self.prior = np.bincount(self.y, minlength=self.C) / max(len(self.y), 1)
        self.alpha = 1.0
        self.ood_threshold = np.inf
        self.loo_kth = np.zeros(0)
        self.fit_info: dict = {}

    def __len__(self) -> int:
        return len(self.y)

    def _codes(self, groups: np.ndarray | None, n: int) -> np.ndarray:
        if groups is None:
            return np.full(n, -1)
        s = np.asarray(groups, dtype=str)
        pos = np.minimum(np.searchsorted(self.uniq, s), self.uniq.size - 1)
        return np.where(self.uniq[pos] == s, pos, -1)

    def search(self, E_query: np.ndarray, groups_query: np.ndarray | None = None, k: int | None = None):
        """(D, I, valid) of the neighbours of unweighted query embeddings."""
        return search_distinct(self.Ew, self.code, np.asarray(E_query) * self.sw,
                               self._codes(groups_query, E_query.shape[0]), self.k if k is None else int(k),
                               self.fetch, self.distinct)

    def counts(self, I: np.ndarray, valid: np.ndarray, k: int, labels: np.ndarray | None = None,
               n_classes: int | None = None) -> np.ndarray:
        lab = self.y if labels is None else np.asarray(labels)
        C = self.C if n_classes is None else int(n_classes)
        Ik, vk = I[:, :k], valid[:, :k]
        L = lab[Ik]
        wts = vk * (L >= 0)
        out = np.zeros((I.shape[0], C))
        for c in range(C):
            out[:, c] = (wts * (L == c)).sum(1)
        return out

    def probs(self, I: np.ndarray, valid: np.ndarray, k: int | None = None, alpha: float | None = None) -> np.ndarray:
        k = self.k if k is None else int(k)
        a = self.alpha if alpha is None else float(alpha)
        n = self.counts(I, valid, k)
        prior = np.broadcast_to(self.prior, n.shape)
        p = (n + a * prior * self.C) / (n.sum(1, keepdims=True) + a * self.C)
        bad = ~np.isfinite(p).all(1)
        p[bad] = prior[bad]
        return p

    def fit_alpha(self, alpha_grid: list[float], *, n_queries: int = 20000, ood_percentile: float = 99.0,
                  seed: int = 0) -> dict:
        """α and the out-of-support threshold by a group-level leave-one-out over a random subset of the rows."""
        rng = np.random.default_rng(seed)
        n = len(self.y)
        rows = rng.choice(n, size=min(int(n_queries), n), replace=False)
        D, I, valid = search_distinct(self.Ew, self.code, self.Ew[rows], self.code[rows], self.k, self.fetch,
                                      self.distinct)
        y = self.y[rows]
        best = (np.inf, 1.0)
        for a in alpha_grid:
            p = self.probs(I, valid, self.k, alpha=float(a))
            ll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))))
            best = min(best, (ll, float(a)))
        self.alpha = best[1]
        nv = valid.sum(1)
        kth = np.sqrt(D[np.arange(len(y)), np.maximum(nv - 1, 0)])
        self.loo_kth = np.sort(kth)
        self.ood_threshold = float(np.percentile(kth, float(ood_percentile)))
        self.fit_info = {"alpha": best[1], "loo_logloss": best[0], "ood_threshold": self.ood_threshold,
                         "short_neighbourhoods": float((nv < self.k).mean()), "n_queries": int(len(rows))}
        return self.fit_info

    def kth_info(self, D: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Distance to the k-th valid neighbour, its percentile in the leave-one-out and the out-of-support flag."""
        nv = np.maximum(valid.sum(1), 1)
        kth = np.sqrt(D[np.arange(D.shape[0]), nv - 1])
        ref = self.loo_kth
        pct = (np.searchsorted(ref, kth, side="right") / len(ref) * 100.0) if len(ref) else np.full(len(kth), np.nan)
        return kth, pct, kth > self.ood_threshold
