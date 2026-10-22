"""Exact neighbour search (FAISS IndexFlatL2 on CPU; numpy fallback if FAISS is not available).

The returned distances are squared Euclidean distances on vectors already scaled by √w (d² of §5.1).
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
        o = np.argsort(dp, axis=1)
        Is.append(np.take_along_axis(part, o, 1))
        Ds.append(np.maximum(np.take_along_axis(dp, o, 1), 0.0))
    return np.vstack(Ds).astype(np.float32), np.vstack(Is).astype(np.int64)


def search_excluding(ref: np.ndarray, query: np.ndarray, k: int, query_subjects: np.ndarray | None = None,
                     ref_subjects: np.ndarray | None = None, self_query: bool = False, extra: int = 25
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k neighbours excluding the case itself (self_query) and cases from the same patient (patient-level LOO).

    Returns (D, I, valid): `valid` marks the valid positions (there may be fewer than k if candidates run out).
    """
    kk = int(min(k + extra + (1 if self_query else 0), ref.shape[0]))
    D, I = search(ref, query, kk)
    bad = np.zeros_like(I, dtype=bool)
    if self_query:
        bad |= I == np.arange(query.shape[0])[:, None]
    if query_subjects is not None and ref_subjects is not None:
        bad |= ref_subjects[I] == query_subjects[:, None]
    order = np.argsort(bad, axis=1, kind="stable")[:, :k]
    D = np.take_along_axis(D, order, 1)
    I = np.take_along_axis(I, order, 1)
    valid = ~np.take_along_axis(bad, order, 1)
    return D, I, valid
