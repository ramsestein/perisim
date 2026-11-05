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
                 subjects_ref: np.ndarray, k_grid: list[int], ids_ref: np.ndarray | None = None,
                 ref_weight: np.ndarray | None = None):
        self.Ew = np.ascontiguousarray(E_ref * sqrt_w_dims, dtype=np.float32)
        self.sw = sqrt_w_dims
        self.y = np.asarray(y_ref, dtype=np.int64)
        self.C = int(n_classes)
        self.subj = np.asarray(subjects_ref)
        self.k_grid = sorted(int(k) for k in k_grid)
        self.k_max = min(self.k_grid[-1], self.Ew.shape[0] - 1)
        self.ids = ids_ref
        self.alpha: dict[int, float] = {k: 1.0 for k in self.k_grid}
        self.ood_thr: dict[int, float] = {}
        self.loo_kth: dict[int, np.ndarray] = {}
        self.set_ref_weight(ref_weight)

    def set_ref_weight(self, ref_weight: np.ndarray | None) -> None:
        """Weight of each reference case in the votes and in the smoothing prior (transport §6.2c: site weight γ of
        the non-local cases). None = every case counts 1, exactly the unweighted kNN. The neighbours do not change."""
        self.rw = None if ref_weight is None else np.asarray(ref_weight, dtype=np.float64)
        self.prior = self._class_share(self.y)

    def _class_share(self, lab: np.ndarray) -> np.ndarray:
        ok = lab >= 0
        if self.rw is None:
            return np.bincount(lab[ok], minlength=self.C) / max(int(ok.sum()), 1)
        cnt = np.bincount(lab[ok], weights=self.rw[ok], minlength=self.C)
        return cnt / cnt.sum() if cnt.sum() > 0 else np.bincount(lab[ok], minlength=self.C) / max(int(ok.sum()), 1)

    def neighbors(self, E_query: np.ndarray | None, subjects_query: np.ndarray | None = None, loo: bool = False):
        Q = self.Ew if loo else np.ascontiguousarray(E_query * self.sw, dtype=np.float32)
        sq = self.subj if loo else subjects_query
        return search_excluding(self.Ew, Q, self.k_max, sq, self.subj if sq is not None else None, self_query=loo)

    def loo_prior(self, labels: np.ndarray | None = None) -> np.ndarray:
        """Per-query class prior of a patient-level LOO over the reference itself: P(c) computed without the
        query's own patient, so a query never contributes to its own smoothing prior (AUDIT C4, mode B)."""
        lab = self.y if labels is None else np.asarray(labels)
        ok = lab >= 0
        tot = np.bincount(lab[ok], minlength=self.C).astype(float)
        _, inv = np.unique(self.subj, return_inverse=True)
        per = np.zeros((inv.max() + 1, self.C))
        np.add.at(per, (inv[ok], lab[ok]), 1.0)
        own = per[inv]
        rest = tot[None, :] - own
        return rest / np.maximum(rest.sum(1, keepdims=True), 1.0)

    def counts(self, I: np.ndarray, valid: np.ndarray, k: int, labels: np.ndarray | None = None,
               n_classes: int | None = None, D: np.ndarray | None = None, kernel: bool = False) -> np.ndarray:
        lab = self.y if labels is None else labels
        C = self.C if n_classes is None else n_classes
        Ik, vk = I[:, :k], valid[:, :k]
        L = lab[Ik]
        wts = np.exp(-D[:, :k].astype(np.float64)) if kernel and D is not None else np.ones(Ik.shape)
        wts = wts * vk * (L >= 0)
        if self.rw is not None:
            wts = wts * self.rw[Ik]
        out = np.zeros((I.shape[0], C))
        for c in range(C):
            out[:, c] = (wts * (L == c)).sum(1)
        return out

    def probs(self, I, valid, k, alpha=None, D=None, kernel=False, labels: np.ndarray | None = None,
              prior: np.ndarray | None = None) -> np.ndarray:
        """Smoothed class probabilities. `labels` relabels the reference (same neighbours); `prior` may be one
        vector or one row per query (patient-level LOO prior)."""
        a = self.alpha.get(k, 1.0) if alpha is None else alpha
        n = self.counts(I, valid, k, labels=labels, D=D, kernel=kernel)
        if prior is None and labels is not None:
            prior = self._class_share(np.asarray(labels))
        prior = np.broadcast_to(self.prior if prior is None else prior, n.shape)
        tot = n.sum(1, keepdims=True)
        p = (n + a * prior * self.C) / (tot + a * self.C)
        bad = ~np.isfinite(p).all(1)
        p[bad] = prior[bad]
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


CLASS_LABELS = {"t1": ["none", "mild", "severe"], "t2": ["short", "medium", "long", "death"],
                "t3": ["early_no_complication", "expected_no_complication", "prolonged_or_mild", "severe_or_death"],
                "t1_bin": ["not_severe", "severe"]}
OUTPUT_KEY = {"t1": "p_T1", "t2": "p_T2", "t3": "p_T3", "t1_bin": "p_T1_severe"}


def patient_pieces(ids, t0, target: str, P: np.ndarray, k: int, kth_pct=None, ood=None, I=None, D=None, valid=None,
                   ref_ids=None, carried: dict | None = None, max_neighbors: int = 20) -> list[dict]:
    """One target's part of the §5.6 per-patient output (AUDIT E14). The neighbours, out-of-support flag and
    carried summary are added only for the target that feeds the clinical output (experiment.patient_output_target).
    CONTAINS identifiers: written under predictions/ only, never exported."""
    lab = CLASS_LABELS[target]
    recs = []
    for i in range(len(ids)):
        r = {"patient_id": str(ids[i]), OUTPUT_KEY[target]: {lab[c]: float(P[i, c]) for c in range(P.shape[1])}}
        if I is not None:
            nb = [(j, d) for j, d, v in zip(I[i, :max_neighbors], D[i, :max_neighbors], valid[i, :max_neighbors]) if v]
            r.update({"t0": float(t0[i]) if t0 is not None and np.isfinite(t0[i]) else None, "k": int(k),
                      "ood": bool(ood[i]), "kth_distance_percentile": float(kth_pct[i]),
                      "neighbors": [{"case_id": str(ref_ids[j]), "distance": float(np.sqrt(max(d, 0.0)))} for j, d in nb],
                      "carried_summary": {("median_los" if kk.startswith("los") else kk):
                                          (float(v[i]) if np.isfinite(v[i]) else None)
                                          for kk, v in (carried or {}).items()}})
        recs.append(r)
    return recs


def merge_patient_outputs(pieces: list[list[dict]]) -> list[dict]:
    """§5.6 records: one per patient, with p_T1 / p_T2 / p_T3 from each target's model and the neighbours of the
    clinical-output target. A target that excludes the patient (e.g. T2 for non-survivors) leaves its key null."""
    keys = [OUTPUT_KEY[t] for t in ("t1", "t2", "t3", "t1_bin")]
    out: dict[str, dict] = {}
    for part in pieces:
        for r in part:
            rec = out.setdefault(r["patient_id"], {"patient_id": r["patient_id"]})
            rec.update({k: v for k, v in r.items() if k != "patient_id"})
    for rec in out.values():
        for k in keys[:3]:
            rec.setdefault(k, None)
    return list(out.values())
