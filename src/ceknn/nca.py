"""Diagonal NCA: the weight w_j of each variable in the distance (Goldberger et al., 2004, diagonal form).

L(w) = mean over anchors a of −log((1 − η)·p_a + η·P(y_a)),  p_a = Σ_{c same class} softmax_c(−d²(a, c)),
with d²(a, c) = Σ_j w_j ‖φ_j(a) − φ_j(c)‖² over each anchor's K nearest candidates (never the row itself nor its
group, and only from its own site when sites are given). The metric is learned without k: the bandwidth is absorbed
into the scale of w. Minibatches of anchors, Adam on log w, candidates recomputed with the current weights every
`refresh_every` epochs.

Port of PERISIM's `metric/nca.py` (mode "nca", no prior) and of `hypo`'s within-site candidates, written in numpy
with the gradient in closed form, so the library does not need torch:
∂L_a/∂w_j = (1 − η)/q_a · Σ_c s_c (1[c same] − p_a) Δ_acj, with s the softmax over the valid candidates.
"""

from __future__ import annotations

import numpy as np

from .knn import search_excluding

INIT_TOTAL_WEIGHT = 10.0          # initial Σ w ⇒ expected d² between random pairs ≈ 10


class DiagonalNCA:
    def __init__(self, dim_feature: np.ndarray, n_features: int, n_classes: int, *, n_candidates: int = 500,
                 batch_size: int = 256, lr: float = 0.05, refresh_every: int = 5, label_smoothing: float = 1e-3,
                 seed: int = 0):
        self.dim_feature = np.asarray(dim_feature, dtype=np.int64)
        self.F = int(n_features)
        self.C = int(n_classes)
        self.K = int(n_candidates)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.refresh_every = max(int(refresh_every), 1)
        self.eta = float(label_smoothing)
        self.rng = np.random.default_rng(seed)
        self.logw = np.log(np.full(self.F, INIT_TOTAL_WEIGHT / max(self.F, 1)))
        self.history: list[float] = []
        # the blocks of a variable are contiguous: Δ_j sums the squared differences over [start_j, start_{j+1})
        self._starts = np.searchsorted(self.dim_feature, np.arange(self.F))

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.logw)

    def _candidates(self, E: np.ndarray, groups, sites, w: np.ndarray):
        Ew = (E * np.sqrt(w)[self.dim_feature]).astype(np.float32)
        if sites is None:
            _, I, valid = search_excluding(Ew, Ew, self.K, groups, groups, self_query=True)
            return I, valid
        K = min(self.K, E.shape[0] - 1)
        cand = np.zeros((E.shape[0], K), dtype=np.int64)
        valid = np.zeros((E.shape[0], K), dtype=bool)
        for s in np.unique(sites):
            idx = np.where(sites == s)[0]
            g = None if groups is None else groups[idx]
            _, I, v = search_excluding(Ew[idx], Ew[idx], self.K, g, g, self_query=True)
            cand[idx, :I.shape[1]] = idx[I]
            valid[idx, :I.shape[1]] = v
        return cand, valid

    def _delta(self, E: np.ndarray, anchors: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """Δ_acj = ‖φ_j(a) − φ_j(c)‖² (B × K × F)."""
        diff = E[anchors][:, None, :] - E[cand]
        return np.add.reduceat(diff * diff, self._starts, axis=2)

    def _loss_grad(self, Delta: np.ndarray, same: np.ndarray, valid: np.ndarray, prior_a: np.ndarray, w: np.ndarray):
        logits = np.where(valid, -(Delta @ w), -np.inf)
        mx = np.max(np.where(valid, logits, -np.inf), axis=1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        ex = np.where(valid, np.exp(logits - mx), 0.0)
        tot = ex.sum(1, keepdims=True)
        s = np.divide(ex, tot, out=np.zeros_like(ex), where=tot > 0)
        p = (s * same).sum(1)
        q = (1.0 - self.eta) * p + self.eta * prior_a
        loss = float(-np.mean(np.log(q)))
        coef = (1.0 - self.eta) / q                                          # B
        r = s * (same.astype(float) - p[:, None])                            # B × K
        grad_w = (coef[:, None] * np.einsum("bk,bkf->bf", r, Delta)).mean(0)
        return loss, grad_w

    def fit(self, E: np.ndarray, y: np.ndarray, *, epochs: int = 30, groups: np.ndarray | None = None,
            sites: np.ndarray | None = None) -> "DiagonalNCA":
        E = np.ascontiguousarray(E, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n = E.shape[0]
        prior_y = np.bincount(y, minlength=self.C) / n + 1e-6
        groups = None if groups is None else np.asarray(groups)
        sites = None if sites is None else np.asarray(sites)
        theta = self.logw.astype(np.float64).copy()
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        cand = valid = None
        for ep in range(int(epochs)):
            if cand is None or ep % self.refresh_every == 0:
                cand, valid = self._candidates(E, groups, sites, np.exp(theta))
            order = self.rng.permutation(n)
            tot = 0.0
            for st in range(0, n, self.batch_size):
                a = order[st:st + self.batch_size]
                ca, va = cand[a], valid[a]
                Delta = self._delta(E, a, ca).astype(np.float64)
                same = (y[ca] == y[a][:, None]) & va
                w = np.exp(theta)
                loss, gw = self._loss_grad(Delta, same, va, prior_y[y[a]], w)
                g = gw * w                                                    # gradient on log w
                t += 1
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                theta -= self.lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
                np.clip(theta, -20.0, 12.0, out=theta)
                tot += loss * len(a)
            self.history.append(tot / n)
        self.logw = theta
        return self
