"""The arms (design §6). Every arm sees the same sequence, through the same calls of the loop:

- `begin`: the starting window (and, for the ML arms, the go-live update);
- `step_start`: a pending deployment that is due enters service;
- `predict`: the next step's landmarks, with the state the arm had before them;
- `enter`: the step's patients, now evaluated, enter the arm's data;
- `window_end`: the end of an evaluation window (monitor checks, the ideal arm's updates, the out-of-support
  threshold).

Every change of state is logged: what changed, how many parameters, the time it took, a hash of the new state.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from .knn import CaseBase
from .ml import Model, closed_test
from .monitor import Davis, monitoring_rows, primary_fires, sensitivity_fires, stats


class Arm:
    family = None

    def __init__(self, name: str):
        self.name = name
        self.log: list[dict] = []

    def begin(self, run) -> None: ...

    def step_start(self, run, pos: int) -> None: ...

    def predict(self, run, rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def enter(self, run, units: np.ndarray, pos: int) -> None: ...

    def window_end(self, run, w: dict, pos: int) -> None: ...


# --------------------------------------------------------------------------- PERISIM
class Perisim(Arm):
    """PERISIM with the frozen metric: `fifo` (the last N patients), `cumulative` (every patient so far) or `static`
    (the starting window). α is fixed on the starting window (design §6.1)."""

    def __init__(self, name: str, mode: str):
        super().__init__(name)
        self.mode = mode

    def _rows(self, run) -> np.ndarray:
        return np.concatenate([run.admitted(u) for u in self.units]) if self.units else np.zeros(0, dtype=np.int64)

    def begin(self, run) -> None:
        self.units = deque(int(u) for u in run.stream.start)
        self.cb = CaseBase(run.Ew, run.y, run.pcode, run.mcfg)
        self.cb.set_rows(self._rows(run))
        t = time.perf_counter()
        a = self.cb.fit_alpha(run.seed("alpha"))
        self.alpha_info = a
        self.log.append({"event": "start", "pos": 0, "patients": len(self.units), "landmarks": len(self.cb.rows),
                         "alpha": a["alpha"], "ood_threshold": a["ood_threshold"], "parameters_changed": 1,
                         "seconds": time.perf_counter() - t, "state": self.cb.digest()})

    def predict(self, run, rows: np.ndarray) -> np.ndarray:
        out = self.cb.predict(rows)
        if self.mode == "fifo":
            run.ood[rows] = out["ood"]
            run.kth[rows] = out["kth"]
        return out["p"]

    def enter(self, run, units: np.ndarray, pos: int) -> None:
        if self.mode == "static":
            return
        t = time.perf_counter()
        left = []
        for u in units:
            self.units.append(int(u))
            if self.mode == "fifo" and len(self.units) > run.stream.N:
                left.append(self.units.popleft())
        self.cb.set_rows(self._rows(run))
        self.log.append({"event": "rotate", "pos": pos, "in": [int(u) for u in units], "out": left,
                         "parameters_changed": 0, "seconds": time.perf_counter() - t, "state": self.cb.digest()})

    def window_end(self, run, w: dict, pos: int) -> None:
        if self.mode != "fifo":
            return
        thr = self.cb.refresh_threshold(run.seed("ood", w["index"]))
        self.log.append({"event": "ood_threshold", "pos": pos, "window": w["index"], "threshold": thr})


class Prevalence(Arm):
    """The rate of the last N patients' landmarks (the FIFO window)."""

    def begin(self, run) -> None:
        self.rate = run.fifo_rate()

    def predict(self, run, rows: np.ndarray) -> np.ndarray:
        return np.full(len(rows), self.rate)

    def enter(self, run, units: np.ndarray, pos: int) -> None:
        self.rate = run.fifo_rate()


# --------------------------------------------------------------------------- ML
class ML(Arm):
    """`frozen` (4a), `golive` (4b), `realistic` (5, 5′) or `ideal` (6); `origin` "transported" or "window" (5′)."""

    def __init__(self, name: str, family: str, policy: str, trigger: str = "primary", months: float = 3.0,
                 origin: str = "transported"):
        super().__init__(name)
        self.family, self.policy, self.trigger, self.months, self.origin = family, policy, trigger, months, origin

    # ------------------------------------------------------------------ helpers
    def _deploy(self, run, model: Model, rec: dict, pos: int, rows: np.ndarray) -> None:
        self.model = model
        self.since: list[np.ndarray] = []
        self.baseline = rec.get("baseline")
        if self.trigger == "davis":
            self.davis.reset(model.predict(rows), run.y[rows])
        self.log.append({"event": "deploy", "pos": pos, **rec})

    def _update(self, run, rows: np.ndarray, tag: str) -> tuple[Model, dict]:
        return closed_test(self.model, rows, run.y, run.pcode, self.X, self.hyper, run.space, run.cfg["update"],
                           run.seed("update", self.name, tag), run.threads)

    def begin(self, run) -> None:
        start = run.start_rows()
        if self.origin == "transported":
            P0 = run.p0(self.family)
            self.model = Model(self.family, lambda r, P0=P0: P0[r])
            self.X, self.hyper = run.X(self.family), run.hyper(self.family)
        else:
            self.model, self.X, self.hyper, info = run.train_on_window(self.family)
            self.log.append({"event": "trained_on_window", "pos": 0, **info})
        self.pending = None
        self.since = []
        self.baseline = None
        if self.trigger == "davis":
            c = run.cfg["triggers"]["sensitivity"]["davis"]
            self.davis = Davis(c["delta"], c["lr"], c["min_sub"], int(run.cfg["loop"]["step"]))
        if self.policy == "frozen" or self.origin == "window":
            if self.trigger == "davis":
                self.davis.reset(self.model.predict(start), run.y[start])
            self.log.append({"event": "start", "pos": 0, "parameters_changed": 0, "model_digest": self.model.digest(start)})
            return
        model, rec = self._update(run, start, "golive")
        self._deploy(run, model, {"phase": "go-live", **rec}, 0, start)

    def step_start(self, run, pos: int) -> None:
        if self.pending is not None and pos >= self.pending["at"]:
            p = self.pending
            self.pending = None
            self._deploy(run, p["model"], {"phase": "update", "trigger_pos": p["trigger_pos"], "delay": p["delay"],
                                           **p["rec"]}, pos, p["rows"])

    def predict(self, run, rows: np.ndarray) -> np.ndarray:
        return self.model.predict(rows)

    def enter(self, run, units: np.ndarray, pos: int) -> None:
        if self.policy != "realistic":
            return
        for u in units:
            self.since.append(run.rows_of(u))
        if self.trigger == "davis" and self.pending is None:
            r = np.concatenate([run.rows_of(u) for u in units])
            self.davis.observe(run.pred[self.name][r], run.y[r], run.pcode[r])

    def window_end(self, run, w: dict, pos: int) -> None:
        if self.policy == "ideal":
            rows = run.fifo_rows()
            model, rec = self._update(run, rows, f"w{w['index']}")
            self._deploy(run, model, {"phase": "ideal", "window": w["index"], **rec}, pos, rows)
            return
        if self.policy != "realistic" or self.pending is not None:
            return
        M, E = run.M(w["seg"]), int(run.cfg["monitor"]["E"])
        mon = monitoring_rows(self.since, M, E, run.y)
        if mon is None:
            self.log.append({"event": "check", "pos": pos, "window": w["index"], "decided": False,
                             "since_deploy": len(self.since)})
            return
        s = stats(run.pred[self.name][mon], run.y[mon], run.pcode[mon])
        if self.trigger == "primary":
            fires = primary_fires(s, run.cfg["triggers"]["primary"])
        elif self.trigger == "davis":
            fires = self.davis.fires()
        else:
            fires = self.baseline is not None and sensitivity_fires(
                self.trigger, s, self.baseline, run.cfg["triggers"]["sensitivity"][self.trigger])
        self.log.append({"event": "check", "pos": pos, "window": w["index"], "decided": True, "fires": bool(fires),
                         **{k: s[k] for k in ("oe", "citl", "citl_se", "auroc", "logloss", "events", "patients")}})
        if not fires:
            return
        rows = run.fifo_rows()
        model, rec = self._update(run, rows, f"t{pos}")
        L = run.stream.delay(w["seg"], self.months)
        self.pending = {"at": pos + L, "model": model, "rec": rec, "trigger_pos": pos, "delay": L, "rows": rows}
        self.log.append({"event": "trigger", "pos": pos, "window": w["index"], "rule": self.trigger, "delay": L,
                         "choice": rec["choice"]})
        if L == 0:
            self.step_start(run, pos)


def make_arms(cfg: dict) -> list[Arm]:
    """Every arm of design §6, with the sensitivity variants of the realistic arm (design §9.3)."""
    arms: list[Arm] = [Perisim("perisim_fifo", "fifo"), Perisim("perisim_cumulative", "cumulative"),
                       Perisim("perisim_static", "static"), Prevalence("prevalence")]
    m = float(cfg["delay"]["months"][0])
    for f in cfg["ml"]["families"]:
        arms += [ML(f"{f}_frozen", f, "frozen"), ML(f"{f}_golive", f, "golive"),
                 ML(f"{f}_realistic", f, "realistic", months=m),
                 ML(f"{f}_realistic_L0", f, "realistic", months=0.0),
                 *[ML(f"{f}_realistic_{t}", f, "realistic", trigger=t, months=m) for t in cfg["triggers"]["sensitivity"]],
                 ML(f"{f}_realistic_window", f, "realistic", months=m, origin="window"),
                 ML(f"{f}_ideal", f, "ideal")]
    return arms
