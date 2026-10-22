"""kNN con proximidades RF-GAP (Rhodes et al., 2023).

p(i, j) = (1/|S_i|) Σ_{t∈S_i} c_j(t)·1[j ∈ J_i(t)] / |M_i(t)|, con S_i los árboles en los que i es OOB (para
consultas nuevas, todos), c_j(t) la multiplicidad bootstrap de j en t y |M_i(t)| la multiplicidad total en la
hoja de i. Se entrena cada árbol con su propia muestra bootstrap para conocer las multiplicidades.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier


class RFGAP:
    def __init__(self, n_estimators: int = 200, min_samples_leaf: int = 5, max_features="sqrt", seed: int = 0):
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFGAP":
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        Xf = np.nan_to_num(X.astype(np.float32))
        self.trees, self.leaf_mats, self.oob = [], [], []
        for t in range(self.n_estimators):
            boot = rng.integers(0, n, n)
            c = np.bincount(boot, minlength=n).astype(float)
            tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                          random_state=int(rng.integers(0, 2 ** 31 - 1)))
            tree.fit(Xf[boot], y[boot])
            leaves = tree.apply(Xf)
            nl = tree.tree_.node_count
            # matriz hoja × caso con c_j / |M_hoja|
            M = np.bincount(leaves, weights=c, minlength=nl)
            vals = np.where(c > 0, c / np.maximum(M[leaves], 1e-12), 0.0)
            L = sparse.csr_matrix((vals, (leaves, np.arange(n))), shape=(nl, n))
            self.trees.append(tree)
            self.leaf_mats.append(L)
            self.oob.append(c == 0)
        self.n_train = n
        return self

    def proximities(self, Xq: np.ndarray, train_idx: np.ndarray | None = None, chunk: int = 2000
                    ) -> list[sparse.csr_matrix]:
        """Proximidades de las consultas a los casos de entrenamiento. Si las consultas son casos de
        entrenamiento (`train_idx`), solo cuentan los árboles en los que el caso es OOB."""
        Xf = np.nan_to_num(Xq.astype(np.float32))
        out = []
        for s in range(0, Xf.shape[0], chunk):
            q = Xf[s:s + chunk]
            acc = sparse.csr_matrix((q.shape[0], self.n_train))
            cnt = np.zeros(q.shape[0])
            for t, (tree, L) in enumerate(zip(self.trees, self.leaf_mats)):
                leaves = tree.apply(q)
                use = self.oob[t][train_idx[s:s + chunk]] if train_idx is not None else np.ones(q.shape[0], dtype=bool)
                rows = np.where(use)[0]
                if rows.size == 0:
                    continue
                Q = sparse.csr_matrix((np.ones(rows.size), (rows, leaves[rows])), shape=(q.shape[0], L.shape[0]))
                acc = acc + Q @ L
                cnt += use
            acc = sparse.diags(1.0 / np.maximum(cnt, 1)) @ acc
            out.append(acc.tocsr())
        return out

    def topk(self, Xq: np.ndarray, k_max: int, train_idx: np.ndarray | None = None, subjects_q=None,
             subjects_train=None) -> np.ndarray:
        """Índices de los k_max casos de mayor proximidad (−1 si no hay suficientes). Con `train_idx`
        (consultas del propio entrenamiento) se excluyen el caso y su paciente. Se calcula una sola vez."""
        out = np.full((Xq.shape[0], k_max), -1, dtype=np.int64)
        offset = 0
        for P in self.proximities(Xq, train_idx=train_idx):
            P = P.tolil()
            for r in range(P.shape[0]):
                cols = np.array(P.rows[r], dtype=np.int64)
                vals = np.array(P.data[r])
                i = offset + r
                keep = np.ones(cols.size, dtype=bool)
                if train_idx is not None:
                    keep &= cols != train_idx[i]
                if subjects_q is not None and subjects_train is not None:
                    keep &= subjects_train[cols] != subjects_q[i]
                cols, vals = cols[keep], vals[keep]
                nb = cols[np.argsort(-vals)[:k_max]]
                out[i, :nb.size] = nb
            offset += P.shape[0]
        return out

    @staticmethod
    def probs_from_topk(NB: np.ndarray, y_train: np.ndarray, n_classes: int, k: int, alpha: float) -> np.ndarray:
        """Probabilidad con el mismo suavizado α que el kNN métrico."""
        prior = np.bincount(y_train, minlength=n_classes) / len(y_train)
        nb = NB[:, :k]
        valid = nb >= 0
        lab = np.where(valid, y_train[np.maximum(nb, 0)], -1)
        cnt = np.stack([(lab == c).sum(1) for c in range(n_classes)], 1).astype(float)
        tot = valid.sum(1, keepdims=True)
        p = (cnt + alpha * prior * n_classes) / (tot + alpha * n_classes)
        p[(tot[:, 0] == 0) | ~np.isfinite(p).all(1)] = prior
        return p
