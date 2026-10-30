"""PERISIM's diagnostics along a stream (design §8.6).

- D1, the out-of-support rate, comes from the FIFO arm's predictions (the threshold is refreshed at every window);
  its per-window rate and alarm are computed with the endpoints.
- D2, the re-deformation gain: `hypo`'s S4 "all variables" fit (`hypo.drift.fit_joint`, its settings) on the current
  FIFO window, at the end of every `every_windows`-th window and of the first window after each boundary. The gain is
  the log-loss of the real kNN with the adapted points minus with the frozen points, on the next window's
  patients (evaluated after the fit, so never in the fitted window), with a patient-bootstrap 95 % CI. Anchors: at
  most 3 random landmarks per patient of the window (`hypo`'s per-patient cap; the window may mix cohorts).
"""

from __future__ import annotations

import time

import numpy as np

from hypo.drift import fit_joint, knn_p1, lookup, value_codes
from hypo.stats import PatientBootstrap, ci, logloss_rows


class Redeform:
    def __init__(self, run, metric, hcfg: dict):
        self.look = lookup(metric.enc, metric.w)
        self.frozen = [lk["points"] for lk in self.look]
        self.codes = value_codes(metric.enc, run.stream.pool.lm.ft, self.look)
        self.xd = hcfg["external"]["drift"]
        dcfg = run.cfg["diagnostics"]["redeform"]
        self.every, self.n_boot = int(dcfg["every_windows"]), int(dcfg["boot"])
        self.per_patient = int(hcfg["metric"]["learn"]["per_patient"])
        self.k, self.fetch = int(run.mcfg["k"]), int(run.mcfg.get("fetch_factor", 6))
        first = {}
        for w in run.stream.windows:
            first.setdefault(w["seg"], w["index"])
        self.after_boundary = {i for s, i in first.items() if i != 0}
        self.pending = None
        self.results: list[dict] = []

    def _anchors(self, run, rows: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(run.seed("anchors", len(self.results)))
        key = rng.random(rows.size)
        code = run.pcode[rows]
        order = np.lexsort((key, code))
        first = np.r_[True, code[order][1:] != code[order][:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(rows.size), 0))
        rank = np.empty(rows.size, dtype=np.int64)
        rank[order] = np.arange(rows.size) - start
        a = np.where(rank < self.per_patient)[0]
        n = min(a.size, int(self.xd["anchors"]))
        return np.sort(rng.choice(a, size=n, replace=False))

    def _fit(self, run, w: dict, pos: int) -> None:
        t = time.perf_counter()
        rows = run.fifo_rows()
        codes_ref = self.codes[rows]
        _, code_ref = np.unique(run.pcode[rows], return_inverse=True)
        y_ref = run.y[rows]
        fifo = next(a for a in run.arms if getattr(a, "mode", None) == "fifo")
        alpha, prior1 = float(fifo.cb.alpha), float(y_ref.mean())
        a_rows = self._anchors(run, rows)
        free = [np.arange(p.shape[0]) for p in self.frozen]
        P = fit_joint(self.frozen, free, codes_ref, code_ref, a_rows, y_ref, self.k, alpha, prior1, self.fetch, self.xd)
        self.pending = {"window": w["index"], "pos": pos, "P": P, "codes_ref": codes_ref, "code_ref": code_ref,
                        "y_ref": y_ref, "alpha": alpha, "prior1": prior1, "n_anchors": int(a_rows.size),
                        "fit_seconds": time.perf_counter() - t}

    def _evaluate(self, run, w: dict) -> None:
        p = self.pending
        q = np.where(run.window_of == w["index"])[0]
        args = (p["codes_ref"], p["code_ref"], p["y_ref"], self.codes[q], self.k, p["alpha"], p["prior1"], self.fetch)
        ll0 = logloss_rows(knn_p1(self.frozen, *args), run.y[q])
        ll1 = logloss_rows(knn_p1(p["P"], *args), run.y[q])
        d = ll1 - ll0
        boot = PatientBootstrap(run.pcode[q], self.n_boot, run.seed("redeform_boot", w["index"]))
        bs = [float((boot.rows(wp) * d).sum() / boot.rows(wp).sum()) for wp in boot]
        self.results.append({"fit_window": p["window"], "fit_pos": p["pos"], "eval_window": w["index"],
                             "segment": w["seg"], "gain": float(d.mean()), "ci": ci(np.asarray(bs)),
                             "n_anchors": p["n_anchors"], "n_eval_landmarks": int(q.size),
                             "fit_seconds": p["fit_seconds"]})
        self.pending = None

    def window_end(self, run, w: dict, pos: int) -> None:
        if self.pending is not None:
            self._evaluate(run, w)
        last = w["index"] == len(run.stream.windows) - 1
        if not last and (w["index"] % self.every == 0 or w["index"] in self.after_boundary):
            self._fit(run, w, pos)
