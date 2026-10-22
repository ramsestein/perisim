"""Diagonal NCA regularized towards the SAGE prior (§5.4), in PyTorch on CPU.

L(w) = LogLoss_LOO_softkNN(w) + λ · Σ_j (log w_j − log c − log(imp_j + ε₀))²

- Softmax kernel over −d² between each anchor and its candidates (excluding the case itself and its patient).
- The metric is learned WITHOUT k: the bandwidth is absorbed into the scale of w.
- c is a free (unpenalized) global scale: the prior regularizes the importance PROFILE, not the bandwidth
  (interpretation recorded in DECISIONS.md / OPEN_QUESTIONS.md).
- Minibatches of anchors with candidates precomputed by exact search (FAISS) with the current weights,
  recomputed every `refresh_every` epochs. The number of candidates is computational, not a prediction k.
- Modes: "nca" (free w), "power" (w = a·imp^γ, comparator §5.4), "scale" (w = a·w_base, scale only).
"""

from __future__ import annotations

import numpy as np
import torch

from perisim.knn.index import search_excluding

INIT_TOTAL_WEIGHT = 10.0  # initial Σ w ⇒ expected d² between random pairs ≈ 10


class DiagonalNCA:
    def __init__(self, dim_feature: np.ndarray, n_features: int, n_classes: int, *, n_candidates: int = 500,
                 batch_size: int = 256, lr: float = 0.05, refresh_every: int = 5, eps0: float = 1e-4,
                 label_smoothing: float = 1e-3, seed: int = 0):
        self.dim_feature = torch.as_tensor(dim_feature, dtype=torch.long)
        self.F = int(n_features)
        self.C = int(n_classes)
        self.K = int(n_candidates)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.refresh_every = max(int(refresh_every), 1)
        self.eps0 = float(eps0)
        self.eta = float(label_smoothing)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.logw = np.log(np.full(self.F, INIT_TOTAL_WEIGHT / max(self.F, 1)))
        self.cand: np.ndarray | None = None
        self.valid: np.ndarray | None = None
        self.history: list[float] = []

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.logw)

    # ------------------------------------------------------------------ candidates
    def refresh_candidates(self, E: np.ndarray, subjects: np.ndarray | None, w: np.ndarray | None = None) -> None:
        w = self.weights if w is None else w
        sw = np.sqrt(w)[self.dim_feature.numpy()].astype(np.float32)
        Ew = E * sw
        _, I, valid = search_excluding(Ew, Ew, self.K, subjects, subjects, self_query=True)
        self.cand, self.valid = I, valid

    # ------------------------------------------------------------------ loss
    def _batch_loss(self, Et, E2, yt, prior_y, anchors, cand, valid, logw_t, y_ref=None, E_ref=None, E2_ref=None):
        wd = torch.exp(logw_t)[self.dim_feature]
        Er = Et if E_ref is None else E_ref
        E2r = E2 if E2_ref is None else E2_ref
        yr = yt if y_ref is None else y_ref
        Ea = Et[anchors]
        Ec = Er[cand]                                        # B × K × D
        e2w_a = E2[anchors] @ wd
        e2w_c = (E2r[cand] * wd).sum(-1)
        cross = torch.bmm(Ec, (Ea * wd).unsqueeze(-1)).squeeze(-1)
        d2 = torch.clamp(e2w_a.unsqueeze(1) + e2w_c - 2.0 * cross, min=0.0)
        logits = (-d2).masked_fill(~valid, float("-inf"))
        ya = yt[anchors]
        same = (yr[cand] == ya.unsqueeze(1)) & valid
        lse_all = torch.logsumexp(logits, dim=1)
        lse_same = torch.logsumexp(logits.masked_fill(~same, float("-inf")), dim=1)
        p = torch.nan_to_num(torch.exp(lse_same - lse_all), nan=0.0)
        p = (1.0 - self.eta) * p + self.eta * prior_y[ya]
        return -torch.log(p).mean()

    def _logw(self, params: dict, mode: str, log_prior: torch.Tensor | None, log_base: torch.Tensor | None):
        if mode == "nca":
            return params["logw"]
        if mode == "power":
            return params["loga"] + params["gamma"] * log_prior
        if mode == "scale":
            return params["loga"] + log_base
        raise ValueError(mode)

    # ------------------------------------------------------------------ fitting
    def fit(self, E: np.ndarray, y: np.ndarray, *, prior: np.ndarray | None = None, lam: float = 0.0,
            epochs: int = 30, init_logw: np.ndarray | None = None, subjects: np.ndarray | None = None,
            mode: str = "nca", base_w: np.ndarray | None = None) -> "DiagonalNCA":
        n = E.shape[0]
        Et = torch.from_numpy(np.ascontiguousarray(E, dtype=np.float32))
        E2 = Et * Et
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
        prior_y = torch.as_tensor(np.bincount(y, minlength=self.C) / len(y) + 1e-6, dtype=torch.float32)
        log_prior = torch.as_tensor(np.log(np.asarray(prior) + self.eps0), dtype=torch.float32) \
            if prior is not None else None
        log_base = torch.as_tensor(np.log(np.asarray(base_w) + 1e-12), dtype=torch.float32) if base_w is not None else None

        if init_logw is not None:
            start = np.asarray(init_logw, dtype=np.float64)
        elif prior is not None:
            p = (np.asarray(prior) + self.eps0)
            start = np.log(p / p.sum() * INIT_TOTAL_WEIGHT)
        else:
            start = self.logw
        if mode == "nca":
            params = {"logw": torch.tensor(start, dtype=torch.float32, requires_grad=True)}
            if log_prior is not None and lam > 0:
                params["logc"] = torch.tensor(float(np.mean(start - log_prior.numpy())), requires_grad=True)
        elif mode == "power":
            params = {"loga": torch.tensor(float(np.log(INIT_TOTAL_WEIGHT)), requires_grad=True),
                      "gamma": torch.tensor(1.0, requires_grad=True)}
        else:
            params = {"loga": torch.tensor(float(np.log(INIT_TOTAL_WEIGHT / max(np.sum(base_w), 1e-12))),
                                           requires_grad=True)}
        opt = torch.optim.Adam(list(params.values()), lr=self.lr)
        for ep in range(int(epochs)):
            if self.cand is None or self.cand.shape[0] != n or (ep > 0 and ep % self.refresh_every == 0):
                with torch.no_grad():
                    cur = self._logw(params, mode, log_prior, log_base).numpy()
                self.refresh_candidates(E, subjects, np.exp(cur))
            cand_t = torch.as_tensor(self.cand)
            valid_t = torch.as_tensor(self.valid)
            order = self.rng.permutation(n)
            tot = 0.0
            for s in range(0, n, self.batch_size):
                a = torch.as_tensor(order[s:s + self.batch_size])
                logw_t = self._logw(params, mode, log_prior, log_base)
                loss = self._batch_loss(Et, E2, yt, prior_y, a, cand_t[a], valid_t[a], logw_t)
                if mode == "nca" and log_prior is not None and lam > 0:
                    loss = loss + lam * torch.sum((params["logw"] - params["logc"] - log_prior) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():  # numerical stability
                    for k in ("logw", "loga"):
                        if k in params:
                            params[k].clamp_(-20.0, 12.0)
                tot += float(loss.detach()) * len(a)
            self.history.append(tot / n)
        with torch.no_grad():
            self.logw = self._logw(params, mode, log_prior, log_base).numpy().astype(np.float64)
        self.params_ = {k: v.detach().numpy().tolist() for k, v in params.items() if v.numel() == 1}
        return self

    # ------------------------------------------------------------------ evaluation (independent of k)
    def soft_loss(self, E_ref: np.ndarray, y_ref: np.ndarray, E_q: np.ndarray, y_q: np.ndarray,
                  w: np.ndarray | None = None, subj_ref: np.ndarray | None = None,
                  subj_q: np.ndarray | None = None, self_query: bool = False) -> float:
        """Soft-kernel kNN log-loss of the queries against the reference (no gradient)."""
        w = self.weights if w is None else w
        sw = np.sqrt(w)[self.dim_feature.numpy()].astype(np.float32)
        _, I, valid = search_excluding(E_ref * sw, E_q * sw, self.K, subj_q, subj_ref, self_query=self_query)
        Er = torch.from_numpy(np.ascontiguousarray(E_ref, dtype=np.float32))
        Eq = torch.from_numpy(np.ascontiguousarray(E_q, dtype=np.float32))
        yr = torch.as_tensor(np.asarray(y_ref), dtype=torch.long)
        yq = torch.as_tensor(np.asarray(y_q), dtype=torch.long)
        prior_y = torch.as_tensor(np.bincount(y_ref, minlength=self.C) / len(y_ref) + 1e-6, dtype=torch.float32)
        logw_t = torch.as_tensor(np.log(w), dtype=torch.float32)
        tot = 0.0
        with torch.no_grad():
            for s in range(0, Eq.shape[0], self.batch_size):
                a = torch.arange(s, min(s + self.batch_size, Eq.shape[0]))
                l = self._batch_loss(Eq, Eq * Eq, yq, prior_y, a, torch.as_tensor(I[a.numpy()]),
                                     torch.as_tensor(valid[a.numpy()]), logw_t, y_ref=yr, E_ref=Er, E2_ref=Er * Er)
                tot += float(l) * len(a)
        return tot / Eq.shape[0]


def pair_sq_diffs(E: np.ndarray, dim_feature: np.ndarray, n_features: int, pairs: np.ndarray) -> np.ndarray:
    """δ_j(a,b)² per pair and variable (R × F): d(a,b) = sqrt(Dsq @ w). Basis of the stopping criterion §5.3.2."""
    diff2 = (E[pairs[:, 0]].astype(np.float64) - E[pairs[:, 1]].astype(np.float64)) ** 2
    out = np.zeros((pairs.shape[0], n_features))
    np.add.at(out.T, dim_feature, diff2.T)
    return out
