"""Surrogate models trained with group-wise *feature dropout* (Frye et al., 2021).

Each model learns to predict with any subset of groups present: this way SAGE simulates the absence of a
group with the model's own mask, without marginal imputation or retraining per subset.
- Trees (RF, LightGBM, XGBoost): absent group → NaN (all three support NaN natively).
- Linear and MLP: absent group → 0 (the mean after standardization) + a per-group mask indicator.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

TREE_FAMILIES = {"random_forest", "lightgbm", "xgboost"}


def sample_params(space: dict, rng: np.random.Generator, exclude: set[str] | None = None) -> tuple[str, dict]:
    fams = {k: v for k, v in space["families"].items() if not exclude or k not in exclude}
    names = list(fams)
    wts = np.array([float(fams[k].get("weight", 1)) for k in names])
    fam = names[rng.choice(len(names), p=wts / wts.sum())]
    params = {}
    for k, spec in fams[fam]["params"].items():
        if isinstance(spec, dict) and "log" in spec:
            lo, hi = spec["log"]
            params[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif isinstance(spec, dict) and "uniform" in spec:
            params[k] = float(rng.uniform(*spec["uniform"]))
        elif isinstance(spec, dict) and "int" in spec:
            lo, hi = spec["int"]
            params[k] = int(rng.integers(lo, hi + 1))
        elif isinstance(spec, dict) and "choice" in spec:
            params[k] = spec["choice"][rng.integers(len(spec["choice"]))]
        else:
            params[k] = spec
    return fam, params


def random_masks(n: int, G: int, rng: np.random.Generator) -> np.ndarray:
    """Mask of present groups: per row p ~ U(0,1) and each group present with prob. 1 − p."""
    p = rng.uniform(0, 1, size=(n, 1))
    return rng.uniform(0, 1, size=(n, G)) >= p


class Surrogate:
    def __init__(self, family: str, params: dict, col_group: np.ndarray, n_groups: int, n_classes: int,
                 seed: int = 0, threads: int = 4):
        self.family = family
        self.params = dict(params)
        self.col_group = np.asarray(col_group)
        self.G = int(n_groups)
        self.C = int(n_classes)
        self.seed = int(seed)
        self.threads = int(threads)
        self.fit_seconds = float("nan")

    # ------------------------------------------------------------------ mask
    def _apply_mask(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        absent_cols = ~mask[:, self.col_group]
        if self.family in TREE_FAMILIES:
            Xm = X.astype(np.float32, copy=True)
            Xm[absent_cols] = np.nan
            return Xm
        Xm = np.where(absent_cols, 0.0, X).astype(np.float32)
        return np.hstack([Xm, (~mask).astype(np.float32)])

    # ------------------------------------------------------------------ fitting
    def fit(self, X: np.ndarray, y: np.ndarray, augment_copies: int = 2) -> "Surrogate":
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        masks = [np.ones((n, self.G), dtype=bool)] + [random_masks(n, self.G, rng) for _ in range(augment_copies)]
        Xa = np.vstack([self._apply_mask(X, m) for m in masks])
        ya = np.tile(np.asarray(y), len(masks))
        self.classes_ = np.unique(ya)
        lut = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([lut[v] for v in ya])
        self.model = self._make(len(self.classes_))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.family == "mlp":
                self._fit_mlp(Xa, yi, len(self.classes_))
            else:
                self.model.fit(Xa, yi)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def _make(self, k: int):
        p = self.params
        if self.family == "logreg":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(C=p["C"], max_iter=int(p.get("max_iter", 1000)))
        if self.family == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            mf = p["max_features"]
            return RandomForestClassifier(n_estimators=int(p["n_estimators"]), max_depth=p["max_depth"],
                                          min_samples_leaf=int(p["min_samples_leaf"]),
                                          max_features=mf if isinstance(mf, str) else float(mf),
                                          n_jobs=self.threads, random_state=self.seed)
        if self.family == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(n_estimators=int(p["n_estimators"]), learning_rate=p["learning_rate"],
                                      num_leaves=int(p["num_leaves"]), min_child_samples=int(p["min_child_samples"]),
                                      subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
                                      reg_lambda=p["reg_lambda"], n_jobs=self.threads, random_state=self.seed,
                                      verbose=-1)
        if self.family == "xgboost":
            import xgboost as xgb

            return xgb.XGBClassifier(n_estimators=int(p["n_estimators"]), learning_rate=p["learning_rate"],
                                     max_depth=int(p["max_depth"]), min_child_weight=p["min_child_weight"],
                                     subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                                     reg_lambda=p["reg_lambda"], tree_method="hist", n_jobs=self.threads,
                                     random_state=self.seed, verbosity=0)
        if self.family == "mlp":
            return None
        raise ValueError(self.family)

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray, k: int) -> None:
        import torch
        from torch import nn

        p = self.params
        torch.manual_seed(self.seed)
        net = nn.Sequential(nn.Linear(X.shape[1], int(p["hidden1"])), nn.ReLU(), nn.Dropout(p["dropout"]),
                            nn.Linear(int(p["hidden1"]), int(p["hidden2"])), nn.ReLU(), nn.Dropout(p["dropout"]),
                            nn.Linear(int(p["hidden2"]), k))
        opt = torch.optim.Adam(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        Xt = torch.from_numpy(X.astype(np.float32))
        yt = torch.from_numpy(y.astype(np.int64))
        bs = int(p.get("batch_size", 256))
        g = torch.Generator().manual_seed(self.seed)
        lossf = nn.CrossEntropyLoss()
        net.train()
        for _ in range(int(p["epochs"])):
            perm = torch.randperm(Xt.shape[0], generator=g)
            for s in range(0, Xt.shape[0], bs):
                b = perm[s:s + bs]
                opt.zero_grad()
                loss = lossf(net(Xt[b]), yt[b])
                loss.backward()
                opt.step()
        net.eval()
        self.model = net

    # ------------------------------------------------------------------ prediction
    def predict_proba(self, X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.ones((X.shape[0], self.G), dtype=bool)
        Xm = self._apply_mask(X, mask)
        if self.family == "mlp":
            import torch

            with torch.no_grad():
                pr = torch.softmax(self.model(torch.from_numpy(Xm)), dim=1).numpy()
        else:
            pr = self.model.predict_proba(Xm)
        out = np.full((X.shape[0], self.C), 1e-6)
        out[:, self.classes_] = pr
        return out / out.sum(1, keepdims=True)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))))
