"""Phase 2: internal validation by rotation (D1), resumable.

Units, each written once to <workdir>/cv[_quick]/ (patient level, git-ignored) and skipped when present:
- folds_<site>.parquet: the patient-level outer fold of every landmark (fixed per site, stratified by the outcome);
- comp_<site>_f<fold>_<method>.parquet: a local comparator's predictions for the site's test fold;
- perisim_ext-<external>_f<fold>.parquet: PERISIM's predictions for both development sites' test fold, the metric
  learned on their training patients; with an aggregate .json beside it (weights, shapes diagnostics, α, times).
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from . import compare
from .casebase import CaseBase
from .config import REPO, code_hash, load, workdir
from .data import Landmarks, concat, learning_sample
from .data import load as load_lm
from .metric import fit_metric

LOG = logging.getLogger("hypo.cv")
QUICK = {"shapes": {"n_models": 8}, "nca": {"epochs": 10}}


def cv_dir(quick: bool) -> Path:
    return workdir() / ("cv_quick" if quick else "cv")


def settings(quick: bool) -> tuple[dict, dict, dict]:
    cfg = load()
    mcfg = copy.deepcopy(cfg["metric"])
    if quick:
        for k, v in QUICK.items():
            mcfg[k].update(v)
    with open(REPO / "config" / "model_space.yaml", encoding="utf-8") as fh:
        space = yaml.safe_load(fh)
    return cfg, mcfg, space


def outer_folds(lm: Landmarks, n: int, seed: int) -> np.ndarray:
    fold = np.full(len(lm), -1)
    sgk = StratifiedGroupKFold(n, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(sgk.split(np.zeros(len(lm)), lm.y, groups=lm.ft.subjects)):
        fold[te] = f
    return fold


class Sites:
    """Landmarks and folds of each cohort, loaded once."""

    def __init__(self, cfg: dict, out: Path):
        self.cfg, self.out, self._lm, self._fold = cfg, out, {}, {}

    def lm(self, c: str) -> Landmarks:
        if c not in self._lm:
            self._lm[c] = load_lm(c, self.cfg)
        return self._lm[c]

    def fold(self, c: str) -> np.ndarray:
        if c not in self._fold:
            p = self.out / f"folds_{c}.parquet"
            lm = self.lm(c)
            if p.exists():
                d = pl.read_parquet(p)
                if d.height != len(lm) or (d["id"].to_numpy().astype(object) != lm.ft.ids).any():
                    raise RuntimeError(f"{p} belongs to other landmarks: remove the cv folder to start again")
                self._fold[c] = d["fold"].to_numpy()
            else:
                seed = int(self.cfg["seed"]) + self.cfg["cohorts"].index(c)
                self._fold[c] = outer_folds(lm, int(self.cfg["cv"]["outer_folds"]), seed)
                pl.DataFrame({"id": lm.ft.ids.astype(str), "fold": self._fold[c]}).write_parquet(p)
        return self._fold[c]

    def split(self, c: str, f: int) -> tuple[np.ndarray, np.ndarray]:
        fold = self.fold(c)
        return np.where(fold != f)[0], np.where(fold == f)[0]


def _write(path: Path, df: pl.DataFrame, info: dict) -> None:
    df.write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    path.with_suffix(".tmp").replace(path)


def run_comparator(S: Sites, c: str, f: int, name: str, mcfg: dict, space: dict, threads: int) -> None:
    path = S.out / f"comp_{c}_f{f}_{name}.parquet"
    if path.exists():
        return
    t = time.perf_counter()
    tr, te = S.split(c, f)
    lm = S.lm(c)
    seed = int(S.cfg["seed"]) + 100 * f
    pred, info = compare.fit(name, lm.take(tr), mcfg, space["baselines"], n_inner=int(S.cfg["cv"]["inner_folds"]),
                             seed=seed, threads=threads)
    p = np.asarray(pred(lm.take(te)), dtype=np.float64)
    info = {"site": c, "fold": f, "method": name, "n_train": int(tr.size), "n_test": int(te.size),
            "seconds": time.perf_counter() - t, "code_hash": code_hash(), **info}
    _write(path, pl.DataFrame({"row": te, "p": p}), info)
    LOG.info("%s fold %d %s: %.0f s", c, f, name, info["seconds"])


def run_perisim(S: Sites, external: str, f: int, mcfg: dict, space: dict, threads: int) -> None:
    path = S.out / f"perisim_ext-{external}_f{f}.parquet"
    if path.exists():
        return
    t = time.perf_counter()
    dev = [c for c in S.cfg["cohorts"] if c != external]
    seed = int(S.cfg["seed"]) + 100 * f
    parts = {c: S.split(c, f) for c in dev}
    pool = concat([S.lm(c).take(parts[c][0]) for c in dev])
    rows = learning_sample(pool, int(mcfg["learn"]["per_patient"]), int(mcfg["learn"]["per_site"]), seed)
    metric = fit_metric(pool.take(rows), mcfg, space, seed=seed, threads=threads)
    del pool
    out, sites = [], {}
    for c in dev:
        tr, te = parts[c]
        cb = CaseBase(metric, S.lm(c).take(tr), mcfg)
        a = cb.fit_alpha(seed)
        pr = cb.predict(S.lm(c).take(te))
        out.append(pl.DataFrame({"site": c, "row": te, "p": pr["p"], "ood": pr["ood"],
                                 **{f"p_band{b}": pr["p_band"][:, b] for b in range(3)}}))
        sites[c] = {"n_case_base": int(tr.size), "n_test": int(te.size), **a}
        del cb
    m = metric.info
    info = {"external": external, "development": dev, "fold": f, "code_hash": code_hash(),
            "seconds": time.perf_counter() - t, "weights": metric.weights_table(),
            "shaped": {n: d["shaped"] for n, d in m["shape_diag"].items()}, "sites": sites,
            **{k: m[k] for k in ("n_learn", "learn_per_site", "dropped_site_constant", "n_features", "n_shaped",
                                 "n_accepted", "n_competitive", "n_attempts", "delta", "nca_history")},
            "metric_seconds": m["seconds"]}
    _write(path, pl.concat(out), info)
    LOG.info("rotation external-%s fold %d: %.0f s", external, f, info["seconds"])


def run(externals: list[str], methods: list[str], folds: list[int] | None, *, quick: bool, threads: int) -> Path:
    cfg, mcfg, space = settings(quick)
    out = cv_dir(quick)
    out.mkdir(parents=True, exist_ok=True)
    S = Sites(cfg, out)
    folds = list(range(int(cfg["cv"]["outer_folds"]))) if folds is None else folds
    comps = [m for m in methods if m != "perisim"]
    for f in folds:
        if "perisim" in methods:
            for ext in externals:
                run_perisim(S, ext, f, mcfg, space, threads)
        dev_sites = sorted({c for ext in externals for c in cfg["cohorts"] if c != ext}, key=cfg["cohorts"].index)
        for c in dev_sites:
            for name in comps:
                run_comparator(S, c, f, name, mcfg, space, threads)
    return out
