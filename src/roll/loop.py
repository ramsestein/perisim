"""The prequential loop (design §5): every arm over one stream (one order, one N), test-then-train.

At each step the next patients are predicted by every arm with the state it had before them; only then do they
enter the arms' data (the FIFO window keeps the last N patients). At the end of each evaluation window the arms run
their checks, and the primary trigger is also evaluated on every arm's predictions (the symmetry check, H4b).
Predictions and logs are patient-level: they stay in <workdir>/experiments/roll (never exported).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import polars as pl

from .arms import Perisim, make_arms
from .data import unwrap
from .ml import Model
from .monitor import monitoring_rows, primary_fires, stats

LOG = logging.getLogger("roll")


class Run:
    def __init__(self, stream, frozen, cfg: dict, mcfg: dict, space: dict, *, threads: int = 8, arms=None,
                 redeform: bool = False, hcfg: dict | None = None):
        self.stream, self.frozen, self.cfg, self.mcfg, self.space, self.threads = stream, frozen, cfg, mcfg, space, threads
        pool = stream.pool
        self.y, self.pcode, self.gate = pool.y, pool.pcode, pool.gate_ok
        Es = [frozen.Ew(stream.metric, c) for c in pool.cohorts]
        self.Ew = Es[0] if len(Es) == 1 else np.ascontiguousarray(np.vstack(Es))
        self.n = len(self.y)
        self.arms = make_arms(cfg) if arms is None else arms
        self.pred = {a.name: np.full(self.n, np.nan) for a in self.arms}
        self.ood = np.zeros(self.n, dtype=bool)
        self.kth = np.full(self.n, np.nan)
        self.pos_of = np.full(self.n, -1, dtype=np.int64)
        self.step_of = np.full(self.n, -1, dtype=np.int64)
        self.window_of = np.full(self.n, -1, dtype=np.int64)
        self._rows, self._p0, self._X = {}, {}, {}
        self.fifo = deque(int(u) for u in stream.start)
        self.checks: list[dict] = []
        self.redeform = None
        if redeform:
            from .diagnostics import Redeform

            self.redeform = Redeform(self, frozen.metrics[stream.metric], hcfg)

    # ------------------------------------------------------------------ helpers for the arms
    def seed(self, *parts) -> int:
        s = self.stream
        key = zlib.crc32("|".join([s.name, s.hospital, *map(str, parts)]).encode())
        return int(np.random.SeedSequence([int(self.cfg["seed"]), key, s.order, s.N]).generate_state(1)[0])

    def rows_of(self, u: int) -> np.ndarray:
        u = int(u)
        if u not in self._rows:
            self._rows[u] = self.stream.pool.rows([u])
        return self._rows[u]

    def admitted(self, u: int) -> np.ndarray:
        r = self.rows_of(u)
        return r[self.gate[r]]

    def _concat(self, units) -> np.ndarray:
        parts = [self.admitted(u) for u in units]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def start_rows(self) -> np.ndarray:
        return self._concat(self.stream.start)

    def fifo_rows(self) -> np.ndarray:
        return self._concat(self.fifo)

    def fifo_rate(self) -> float:
        r = self.fifo_rows()
        return float(self.y[r].mean()) if r.size else 0.0

    def _stack(self, f) -> np.ndarray:
        parts = [f(c) for c in self.stream.pool.cohorts]
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def p0(self, family: str) -> np.ndarray:
        if family not in self._p0:
            self._p0[family] = self._stack(lambda c: self.frozen.p0(self.stream.metric, family, c))
        return self._p0[family]

    def X(self, family: str) -> np.ndarray:
        if family not in self._X:
            self._X[family] = self._stack(lambda c: self.frozen.X(self.stream.metric, family, c))
        return self._X[family]

    def hyper(self, family: str) -> dict:
        info = self.frozen.infos[(self.stream.metric, family)]
        return {k: info[k] for k in ("C", "num_leaves", "learning_rate") if k in info}

    def M(self, seg: str) -> int:
        return int(self.cfg["monitor"]["M"][self.stream.pop[seg]])

    def train_on_window(self, family: str):
        """Arm 5′: the comparator trained on the starting window only, with `hypo`'s settings (tuned in inner
        folds); its encoder and hyperparameters are kept for its later refits."""
        from hypo.compare import fit

        t = time.perf_counter()
        rows = self.start_rows()
        lm = self.stream.pool.lm.take(rows)
        fn, info = fit(family, lm, self.mcfg, self.space["baselines"], n_inner=int(self.cfg["ml"]["inner_folds"]),
                       seed=self.seed("window", family), threads=self.threads)
        enc, pred = unwrap(fn)
        Xw = enc.transform_model(self.stream.pool.lm.ft)[0]
        model = Model(family, lambda r, Xw=Xw, pred=pred: pred(Xw[r])[:, 1], origin="window")
        hyper = {k: info[k] for k in ("C", "num_leaves", "learning_rate") if k in info}
        return model, Xw, hyper, {"hyper": hyper, "seconds": time.perf_counter() - t}

    # ------------------------------------------------------------------ the loop
    def _psi(self, ref: np.ndarray, cur: np.ndarray) -> dict:
        """The population stability index of each continuous input, the window against the primary realistic arm's
        reference (deciles of the reference, missing values as their own bin): a reference only, deciding nothing."""
        ft = self.stream.pool.lm.ft
        vals = []
        for sp in ft.specs:
            if sp.type != "continuous":
                continue
            x = ft.num[sp.name]
            r, c = x[ref], x[cur]
            fin = r[np.isfinite(r)]
            edges = np.unique(np.quantile(fin, np.linspace(0.1, 0.9, 9))) if fin.size else np.zeros(0)

            def share(v):
                b = np.where(np.isfinite(v), np.searchsorted(edges, v, side="right"), edges.size + 1)
                return np.bincount(b, minlength=edges.size + 2) / max(v.size, 1) + 1e-4

            pr, pc = share(r), share(c)
            vals.append(float(((pc - pr) * np.log(pc / pr)).sum()))
        return {"psi_mean": float(np.mean(vals)) if vals else np.nan, "psi_max": float(np.max(vals)) if vals else np.nan}

    def _check_all(self, w: dict, pos: int) -> None:
        """The primary rule on every arm's predictions over the last M evaluated patients (H4b), and the PSI."""
        M, E = self.M(w["seg"]), int(self.cfg["monitor"]["E"])
        units = [self.rows_of(u) for u in self.stream.seq[max(0, pos - 20 * M):pos]]
        mon = monitoring_rows(units, M, E, self.y)
        rec = {"window": w["index"], "pos": pos}
        if mon is not None:
            for a in self.arms:
                s = stats(self.pred[a.name][mon], self.y[mon], self.pcode[mon])
                rec[a.name] = primary_fires(s, self.cfg["triggers"]["primary"])
        ref = next((a for a in self.arms if a.name == f"{self.cfg['ml']['primary_family']}_realistic"), None)
        if ref is not None:
            rec.update(self._psi(ref.ref_rows, np.where(self.window_of == w["index"])[0]))
        self.checks.append(rec)

    def execute(self) -> dict:
        t0 = time.perf_counter()
        s, B = self.stream, int(self.cfg["loop"]["step"])
        timing = {a.name: 0.0 for a in self.arms}
        for a in self.arms:
            t = time.perf_counter()
            a.begin(self)
            timing[a.name] += time.perf_counter() - t
        for k, (a0, a1, wi) in enumerate(s.steps(B)):
            for a in self.arms:
                a.step_start(self, a0)
            units = s.seq[a0:a1]
            rows = np.concatenate([self.rows_of(u) for u in units])
            for a in self.arms:
                t = time.perf_counter()
                self.pred[a.name][rows] = a.predict(self, rows)
                timing[a.name] += time.perf_counter() - t
            self.pos_of[rows] = np.repeat(np.arange(a0, a1), [len(self.rows_of(u)) for u in units])
            self.step_of[rows] = k
            self.window_of[rows] = wi
            for u in units:
                self.fifo.append(int(u))
                if len(self.fifo) > s.N:
                    self.fifo.popleft()
            for a in self.arms:
                t = time.perf_counter()
                a.enter(self, units, a1)
                timing[a.name] += time.perf_counter() - t
            w = s.windows[wi]
            if a1 == w["end"]:
                for a in self.arms:
                    t = time.perf_counter()
                    a.window_end(self, w, a1)
                    timing[a.name] += time.perf_counter() - t
                self._check_all(w, a1)
                if self.redeform is not None:
                    self.redeform.window_end(self, w, a1)
        self.seconds = time.perf_counter() - t0
        self.timing = timing
        return self.result()

    # ------------------------------------------------------------------ outputs
    def frame(self) -> pl.DataFrame:
        ev = np.where(self.pos_of >= 0)[0]
        ev = ev[np.argsort(self.pos_of[ev], kind="stable")]
        seg = np.asarray(self.stream.seg, dtype=object)
        cols = {"row": ev, "pos": self.pos_of[ev], "step": self.step_of[ev], "window": self.window_of[ev],
                "segment": seg[self.pos_of[ev]].astype(str), "y": self.y[ev], "patient": self.pcode[ev],
                "ood": self.ood[ev], "kth": self.kth[ev]}
        cols.update({f"p_{a.name}": self.pred[a.name][ev] for a in self.arms})
        return pl.DataFrame(cols)

    def digest(self) -> str:
        ev = np.where(self.pos_of >= 0)[0]
        h = hashlib.sha256()
        for a in self.arms:
            h.update(a.name.encode())
            h.update(np.ascontiguousarray(self.pred[a.name][ev]).tobytes())
        return h.hexdigest()[:16]

    def result(self) -> dict:
        fifo = next(a for a in self.arms if isinstance(a, Perisim) and a.mode == "fifo")
        return {"stream": self.stream.describe(), "windows": self.stream.windows, "alpha": fifo.alpha_info,
                "arms": {a.name: a.log for a in self.arms}, "checks": self.checks,
                "redeform": self.redeform.results if self.redeform is not None else [],
                "seconds": self.seconds, "timing": self.timing, "predictions_digest": self.digest()}


def replay(run: Run, fifo_log: list[dict]) -> dict:
    """Governability (design §11): rebuild PERISIM-FIFO's case base at every step from the stream log alone (the
    starting window, then each step's patients in and out) and compare it with the state hash the arm logged."""
    units = deque(int(u) for u in run.stream.start)
    steps = bad = 0
    for e in fifo_log:
        if e["event"] != "rotate":
            continue
        for u in e["in"]:
            units.append(int(u))
        for u in e["out"]:
            if units.popleft() != int(u):
                bad += 1
        rows = np.sort(np.concatenate([run.admitted(u) for u in units]))
        steps += 1
        bad += hashlib.sha256(rows.tobytes()).hexdigest()[:16] != e["state"]
    return {"steps": steps, "mismatches": int(bad)}


def save(run: Run, res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    run.frame().write_parquet(out / "predictions.parquet")
    (out / "log.json").write_text(json.dumps(res, indent=1, default=_json), encoding="utf-8")


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
