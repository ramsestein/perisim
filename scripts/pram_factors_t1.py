"""Pre-registered (OSF; docs/prereg_pram_factors.md), Tier 1: the factors behind the learned-metric vs cosine
reversal, crossed in PERISIM's setting.

Targets: INSPIRE, MIMIC-IV and eICU (external stage, five patient folds) and NWICU (pre-registered folds, each frozen
rotation). Run with `hypo`'s frozen code (PERISIM_SRC=frozen/hypo/src, or a v0.5 checkout); the freeze record is
checked at the start and the end.

Crossed per target: metric (COS, MI, NCA-diag, NCA-full; sensitivity NCA-full on 2,000 and 10,000 patients) × vote
(V-ours: `hypo`'s smoothed vote; V-inv: PRAM's 1/(d + 1e-6)) × use (U-alone; U-mix with the frozen transported
logistic or LightGBM comparator, p = (1 − a)·p_base + a·p_kNN, a ∈ {0, 0.05, …, 0.5} chosen by AUROC over three
patient-level folds inside the bank).

Gates (prereg §8): the rebuilt Euclidean comparator, the MI arm and the NCA-diag arm reproduce the stored predictions
on fold 0, and the frozen base models reproduce the stored comparator predictions, all to 1e-9.

Usage:  python scripts/pram_factors_t1.py [--sites inspire mimic eicu nwicu] [--threads 8] [--gate-only]
Outputs (aggregates only): work/experiments/hypo/pram_factors_t1/t1.json, reports/pram_factors_t1.md;
row-level units in pram_factors_t1/units/ (git-ignored work folder, never published).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:                                    # v0.6 repository: $PERISIM_SRC (frozen code) first on sys.path
    import _frozen  # noqa: E402
    _frozen.use_src()
except ImportError:                     # a v0.5 checkout: its own src is the frozen code
    sys.path.insert(0, str(ROOT / "src"))

import posthoc_cosine as PC  # noqa: E402
import posthoc_fixed_metric as FM  # noqa: E402

LOG = logging.getLogger("pram.t1")
N_BOOT = 2000
SEED = 20261109
GATE_TOL = 1e-9
A_GRID = np.round(np.arange(0.0, 0.501, 0.05), 2)
INNER_FOLDS = 3
NCA_SEED = 42
NCA_N = {"ncaf": 5000, "ncaf2k": 2000, "ncaf10k": 10000}
ROTATIONS = ("inspire", "mimic", "eicu")
HYPO_SITES = ("inspire", "mimic", "eicu")
VOTES = ("ours", "inv")
BASES = ("lr", "gbm")
USES = ("alone", "mix_lr", "mix_gbm")


# --------------------------------------------------------------------------- metrics
class LinearMetric:
    """A full-matrix NCA on a base metric's unweighted embedding: embed(lm) = E·Lᵀ, weights 1."""

    def __init__(self, base, L: np.ndarray):
        self.base, self.L = base, np.asarray(L, dtype=np.float64)

    def map(self, E: np.ndarray) -> np.ndarray:
        return (np.asarray(E, dtype=np.float64) @ self.L.T).astype(np.float32)

    def embed(self, lm) -> np.ndarray:
        return self.map(self.base.embed(lm))

    @property
    def sqrt_w_dims(self) -> np.ndarray:
        return np.ones(self.L.shape[0])


def fit_full_nca(base, dev_lms: list, n_patients: int, seed: int = NCA_SEED) -> tuple[np.ndarray, dict]:
    """scikit-learn's NCA (full linear map, defaults) on the base metric's unweighted embedding of the development
    landmarks: one random landmark per patient, `n_patients` random patients."""
    from sklearn.neighbors import NeighborhoodComponentsAnalysis

    rng = np.random.default_rng(seed)
    X, y = [], []
    for d in dev_lms:
        subj = np.asarray(d.ft.subjects).astype(str)
        perm = rng.permutation(subj.size)
        _, first = np.unique(subj[perm], return_index=True)
        pick = perm[first]                                       # one random landmark per patient
        X.append(np.asarray(base.embed(d.take(np.sort(pick))), dtype=np.float64))
        y.append(np.asarray(d.y)[np.sort(pick)])
    X, y = np.vstack(X), np.concatenate(y)
    n = min(int(n_patients), X.shape[0])
    sel = rng.choice(X.shape[0], size=n, replace=False)
    t = time.perf_counter()
    nca = NeighborhoodComponentsAnalysis(random_state=seed).fit(X[sel], y[sel])
    return nca.components_, {"n": int(n), "dims": int(X.shape[1]), "seconds": time.perf_counter() - t,
                             "n_iter": int(getattr(nca, "n_iter_", -1))}


def mi_dev(target_lm, dev_lms: list) -> dict[str, float]:
    """§17's development MI by variable name (pooled development landmarks; the target's encoder only names them)."""
    from hypo.metric import C
    from perisim.metric.encoding import Encoder

    enc = Encoder(geometry="shape").fit(target_lm.ft, target_lm.y, C)
    pft, py = FM._pool(dev_lms)
    out = {}
    for b in enc.blocks:
        c = FM.block_codes(b, pft)
        out[b.name] = 0.0 if c is None or not c[1].any() else FM.plugin_mi(c[0], py[c[1]])
    return out


def mi_metric(bank, w_by_name: dict):
    from hypo.metric import C, Metric
    from perisim.metric.encoding import Encoder

    enc = Encoder(geometry="shape").fit(bank.ft, bank.y, C)
    return Metric(enc, FM.weight_vector(enc, w_by_name))


class Arm:
    """One metric arm: `make(bank_lm)` returns the metric for a bank; `fixed` metrics do not depend on the bank and
    their embedding of the whole target is cached."""

    def __init__(self, name: str, make, fixed: bool, cosine: bool):
        self.name, self.make, self.fixed, self.cosine = name, make, fixed, cosine
        self._E = None

    def embedding(self, metric, lm, rows):
        if self.fixed:
            if self._E is None:
                self._E = metric.embed(lm)
            return self._E[rows]
        return metric.embed(lm.take(rows))


# --------------------------------------------------------------------------- one kNN run, both votes
def inv_vote(D: np.ndarray, I: np.ndarray, valid: np.ndarray, y_bank: np.ndarray, cosine: bool,
             prior: float) -> np.ndarray:
    """PRAM's vote: weighted mean of the neighbours' outcomes, weight 1/(d + 1e-6). D is the squared distance in the
    searched space; for the cosine arms that space holds unit vectors, so the cosine distance is D/2."""
    d = np.maximum(D, 0.0) / 2.0 if cosine else np.sqrt(np.maximum(D, 0.0))
    w = np.where(valid, 1.0 / (d + 1e-6), 0.0)
    s = w.sum(1)
    p = (w * y_bank[I]).sum(1) / np.where(s > 0, s, 1.0)
    return np.where(s > 0, p, prior)


def run_knn(arm: Arm, lm, bank: np.ndarray, query: np.ndarray, mcfg: dict, seed: int) -> dict:
    from hypo.casebase import CaseBase

    blm, qlm = lm.take(bank), lm.take(query)
    metric = arm.make(blm)
    cb = CaseBase(metric, blm, mcfg, E=arm.embedding(metric, lm, bank))
    info = cb.fit_alpha(seed)
    out = cb.predict(qlm, E=arm.embedding(metric, lm, query))
    y_bank = np.asarray(blm.y, dtype=np.float64)
    return {"ours": np.asarray(out["p"], dtype=np.float64),
            "inv": inv_vote(out["D"], out["I"], out["valid"], y_bank, arm.cosine, float(y_bank.mean())),
            "alpha": info.get("alpha")}


def inner_groups(subjects: np.ndarray, seed: int, k: int = INNER_FOLDS) -> np.ndarray:
    u, inv = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
    g = np.random.default_rng(seed).permutation(u.size) % k
    return g[inv]


def choose_a(p_base: np.ndarray, p_knn: np.ndarray, y: np.ndarray) -> dict:
    """a on PRAM's grid maximising the pooled AUROC (ties: the smaller a); also the log-loss choice (sensitivity)."""
    from hypo.stats import logloss_rows

    ones = np.ones(y.size)
    auc = [PC.wauc((1 - a) * p_base + a * p_knn, y, ones) for a in A_GRID]
    ll = [float(logloss_rows(np.clip((1 - a) * p_base + a * p_knn, 1e-6, 1 - 1e-6), y).mean()) for a in A_GRID]
    best = max(auc)
    return {"a": float(A_GRID[next(i for i, v in enumerate(auc) if v >= best - 1e-12)]),
            "a_logloss": float(A_GRID[int(np.argmin(ll))]), "auroc_curve": [float(v) for v in auc]}


# --------------------------------------------------------------------------- one target unit
def run_unit(name: str, lm, fold: np.ndarray, arms: list[Arm], base: dict[str, np.ndarray], mcfg: dict, seed0: int,
             units: Path) -> None:
    import polars as pl

    units.mkdir(parents=True, exist_ok=True)
    y_all = np.asarray(lm.y)
    for f in sorted(set(fold.tolist())):
        bank, test = np.where(fold != f)[0], np.where(fold == f)[0]
        seed = seed0 + 100 * f
        grp = inner_groups(lm.ft.subjects[bank], seed0 + 100 * f + 7)
        for arm in arms:
            path = units / f"{name}_{arm.name}_f{f}.parquet"
            if path.exists():
                continue
            t = time.perf_counter()
            outer = run_knn(arm, lm, bank, test, mcfg, seed)
            oof = {v: np.full(bank.size, np.nan) for v in VOTES}
            for j in range(INNER_FOLDS):
                ib, iq = bank[grp != j], bank[grp == j]
                r = run_knn(arm, lm, ib, iq, mcfg, seed + j + 1)
                for v in VOTES:
                    oof[v][grp == j] = r[v]
            yb = y_all[bank].astype(np.float64)
            choice = {v: {b: choose_a(base[b][bank], oof[v], yb) for b in BASES} for v in VOTES}
            pl.DataFrame({"row": test, "ours": outer["ours"], "inv": outer["inv"]}).write_parquet(path)
            path.with_suffix(".json").write_text(json.dumps(
                {"alpha": outer["alpha"], "choice": choice, "seconds": time.perf_counter() - t}, indent=1,
                default=float))
            LOG.info("t1 %s fold %d %s: α %s, a %s (%.0f s)", name, f, arm.name, outer["alpha"],
                     {v: {b: choice[v][b]["a"] for b in BASES} for v in VOTES}, time.perf_counter() - t)


def collect(name: str, lm, fold: np.ndarray, arms: list[str], base: dict[str, np.ndarray], units: Path,
            a_key: str = "a") -> dict[str, np.ndarray]:
    """Test predictions of every (arm, vote, use), all folds, in row order 0..n−1."""
    import polars as pl

    n = len(lm)
    out = {}
    for arm in arms:
        P = {v: np.full(n, np.nan) for v in VOTES}
        A = {(v, b): np.full(n, np.nan) for v in VOTES for b in BASES}
        for f in sorted(set(fold.tolist())):
            path = units / f"{name}_{arm}_f{f}.parquet"
            d = pl.read_parquet(path)
            ch = json.loads(path.with_suffix(".json").read_text())["choice"]
            r = d["row"].to_numpy()
            for v in VOTES:
                P[v][r] = d[v].to_numpy()
                for b in BASES:
                    A[(v, b)][r] = ch[v][b][a_key]
        for v in VOTES:
            out[f"{arm}|{v}|alone"] = P[v]
            for b in BASES:
                out[f"{arm}|{v}|mix_{b}"] = (1 - A[(v, b)]) * base[b] + A[(v, b)] * P[v]
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise RuntimeError(f"t1 {name}: {k} has rows without a prediction")
    return out


# --------------------------------------------------------------------------- statistics and rule
def verdict(ci: list[float]) -> str:
    return PC.verdict(ci, "NCA beats cosine", "cosine beats NCA")


def contrasts(p: dict[str, np.ndarray], base: dict[str, np.ndarray], y: np.ndarray, subjects: np.ndarray,
              ncas: list[str], seed: int, n_boot: int) -> dict:
    """Δ per cell, factor effects at the home levels, sufficiency of each switch (prereg §6)."""
    from hypo.stats import calibration_line_w, citl_w, logit

    keys = list(p) + [f"base|{b}" for b in BASES]
    allp = {**p, **{f"base|{b}": base[b] for b in BASES}}
    B = PC.boot(allp, y, subjects, n_boot, seed)
    yf = y.astype(np.float64)

    def d(a, b, what="A"):
        est = (B["est_a"][a] - B["est_a"][b]) if what == "A" else (B["est_l"][b] - B["est_l"][a])
        arr = (B["A"][a] - B["A"][b]) if what == "A" else (B["L"][b] - B["L"][a])
        return est, arr

    def pack(est, arr):
        return {"est": float(est), "ci95": FM._ci(arr)}

    arms = {}
    for k in keys:
        x = logit(np.clip(allp[k], 1e-6, 1 - 1e-6))
        arms[k] = {"auroc": B["est_a"][k], "auroc_ci95": FM._ci(B["A"][k]), "logloss": B["est_l"][k],
                   "citl": citl_w(x, yf), "slope": calibration_line_w(x, yf)[1]}
    cells = {}
    for m in ncas + [a for a in {k.split("|")[0] for k in p} if a.startswith("mi")]:
        for v in VOTES:
            for u in USES:
                a, c = f"{m}|{v}|{u}", f"cos|{v}|{u}"
                if a in p and c in p:
                    est, arr = d(a, c)
                    estl, arrl = d(a, c, "L")
                    cells[f"{m}|{v}|{u}"] = {"delta_auroc": pack(est, arr), "verdict": verdict(FM._ci(arr)),
                                             "delta_logloss": pack(estl, arrl)}
    rule = {}
    for m in ncas:
        if not m.startswith("ncad"):
            continue
        full = m.replace("ncad", "ncaf")
        home = f"{m}|ours|alone"
        switches = {"U": f"{m}|ours|mix_lr", "V": f"{m}|inv|alone", "I": f"{full}|ours|alone",
                    "UV": f"{m}|inv|mix_lr", "UI": f"{full}|ours|mix_lr", "VI": f"{full}|inv|alone",
                    "UVI": f"{full}|inv|mix_lr"}

        def delta_arr(cell):
            arm, v, u = cell.split("|")
            return d(cell, f"cos|{v}|{u}")

        h_est, h_arr = delta_arr(home)
        r = {"home": {"cell": home, "delta": pack(h_est, h_arr), "verdict": verdict(FM._ci(h_arr))}, "effects": {},
             "switch": {}}
        for F, cell in switches.items():
            if cell not in p:
                continue
            s_est, s_arr = delta_arr(cell)
            r["switch"][F] = {"cell": cell, "delta": pack(s_est, s_arr), "verdict": verdict(FM._ci(s_arr)),
                              "sufficient": verdict(FM._ci(s_arr)) != r["home"]["verdict"]}
            if len(F) == 1:
                r["effects"][F] = pack(h_est - s_est, h_arr - s_arr)
        rule[m] = r
    comp = {}
    for k, v in p.items():
        if k.endswith("|alone"):
            for b in BASES:
                res = yf - base[b]
                comp[f"{k}|{b}"] = {"corr_base": float(np.corrcoef(v, base[b])[0, 1]),
                                    "spearman_residual": _spearman(v, res)}
    return {"arms": arms, "cells": cells, "rule": rule, "complementarity": comp, "n": int(y.size),
            "patients": int(np.unique(np.asarray(subjects).astype(str)).size)}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


# --------------------------------------------------------------------------- targets
def gate_eq(p_new: np.ndarray, p_old: np.ndarray, what: str) -> None:
    PC._gate(np.asarray(p_new, dtype=np.float64), None if p_old is None else np.asarray(p_old, dtype=np.float64), what)


def hypo_target(st, X: str, units: Path, gate_only: bool, sens: bool) -> dict:
    import polars as pl
    from hypo.external_report import _on, _phase2, _units

    S, mcfg = st.S, st.mcfg
    lm = S.lm(X)
    fold = np.asarray(S.fold(X))
    seed0 = int(S.cfg["seed"])
    dev = [c for c in S.cfg["cohorts"] if c != X]
    t0 = time.perf_counter()
    # gates on fold 0
    b0, q0 = np.where(fold != 0)[0], np.where(fold == 0)[0]
    p_e, _ = PC.knn_predict(PC.neutral_metric(lm.take(b0)), lm.take(b0), lm.take(q0), mcfg, seed0)
    gate_eq(p_e, _on(_phase2(X, "euclidean"), q0, "p"), f"{X} euclidean")
    base = {"lr": np.asarray(st.comp(X, "logreg")(lm), dtype=np.float64),
            "gbm": np.asarray(st.comp(X, "lightgbm")(lm), dtype=np.float64)}
    C = _units(f"comps_{X}_f*.parquet")
    for b, col in (("lr", "p_logreg"), ("gbm", "p_lightgbm")):
        gate_eq(base[b][C["row"].to_numpy()], C[col].to_numpy(), f"{X} base {b}")
    nca = st.metric(X)
    p_n, _ = PC.knn_predict(nca, lm.take(b0), lm.take(q0), mcfg, seed0)
    gate_eq(p_n, _on(_units(f"transported_{X}_f*.parquet"), q0, "p"), f"{X} nca-diag")
    w_mi = mi_dev(lm, [S.lm(c) for c in dev])
    p_m, _ = PC.knn_predict(mi_metric(lm.take(b0), w_mi), lm.take(b0), lm.take(q0), mcfg, seed0)
    fm = sorted((PC.st_dir(st) / "posthoc_fixed_metric" / "units").glob(f"mi_transported_{X}_f0.parquet"))
    gate_eq(p_m, _on(pl.read_parquet(fm[0]).sort("row"), q0, "p") if fm else None, f"{X} mi")
    if gate_only:
        return {}
    arms = [Arm("cos", lambda b: PC.CosineMetric(PC.neutral_metric(b), b), False, True),
            Arm("mi", lambda b: mi_metric(b, w_mi), False, False),
            Arm("ncad", lambda b: nca, True, False)]
    nca_info = {}
    for key, n in NCA_N.items():
        if key != "ncaf" and not sens:
            continue
        L, info = fit_full_nca(nca, [S.lm(c) for c in dev], n)
        nca_info[key] = info
        arms.append(Arm(key, lambda b, L=L: LinearMetric(nca, L), True, False))
        LOG.info("t1 %s: %s fitted (%s)", X, key, info)
    run_unit(X, lm, fold, arms, base, mcfg, seed0, units)
    names = [a.name for a in arms]
    res = {}
    for a_key in ("a", "a_logloss"):
        p = collect(X, lm, fold, names, base, units, a_key)
        res[a_key] = contrasts(p, base, np.asarray(lm.y), lm.ft.subjects, [n for n in names if n.startswith("nca")],
                               SEED, N_BOOT if a_key == "a" else max(N_BOOT // 4, 100))
    res["nca_full"] = nca_info
    res["seconds"] = time.perf_counter() - t0
    return res


def nwicu_target(st, units: Path, gate_only: bool, sens: bool, quick: bool) -> dict:
    import nwicu_transport as NT
    from hypo.cv import outer_folds, settings
    from hypo.frozen import final_dir, load_pickle

    cfg, mcfg, _ = settings(quick)
    lm = NT.landmarks(cfg)
    fold = outer_folds(lm, int(cfg["cv"]["outer_folds"]), int(cfg["seed"]))
    seed0 = int(cfg["seed"])
    nu = NT.work() / "units"
    t0 = time.perf_counter()
    b0, q0 = np.where(fold != 0)[0], np.where(fold == 0)[0]

    def stored(fname):
        r_, v_ = NT.read(nu / fname)
        out = np.full(q0.size, np.nan)
        pos = np.searchsorted(q0, r_)
        ok = (pos < q0.size) & (q0[np.minimum(pos, q0.size - 1)] == r_)
        out[pos[ok]] = v_[ok]
        return out

    p_e, _ = PC.knn_predict(PC.neutral_metric(lm.take(b0)), lm.take(b0), lm.take(q0), mcfg, seed0)
    gate_eq(p_e, stored("euclidean_f0.parquet"), "nwicu euclidean")
    metrics = {e: load_pickle(final_dir() / f"metric_ext-{e}.pkl") for e in ROTATIONS}
    res = {"rotations": {}}
    for e in ROTATIONS:
        base = {"lr": np.asarray(load_pickle(final_dir() / f"comp_ext-{e}_logreg.pkl")(lm), dtype=np.float64),
                "gbm": np.asarray(load_pickle(final_dir() / f"comp_ext-{e}_lightgbm.pkl")(lm), dtype=np.float64)}
        gate_eq(base["gbm"][q0], stored(f"lgbm_{e}_f0.parquet"), f"nwicu base gbm {e}")
        from hypo.casebase import CaseBase
        cb = CaseBase(metrics[e], lm.take(b0), mcfg)
        cb.fit_alpha(seed0)
        gate_eq(np.asarray(cb.predict(lm.take(q0))["p"], dtype=np.float64), stored(f"knn_{e}_f0.parquet"),
                f"nwicu nca-diag {e}")
        if gate_only:
            continue
        dev = [c for c in ROTATIONS if c != e]
        w_mi = mi_dev(lm, [st.S.lm(c) for c in dev])
        arms = [Arm("cos", lambda b: PC.CosineMetric(PC.neutral_metric(b), b), False, True),
                Arm("mi", lambda b, w=w_mi: mi_metric(b, w), False, False),
                Arm("ncad", lambda b, m=metrics[e]: m, True, False)]
        nca_info = {}
        for key, n in NCA_N.items():
            if key != "ncaf" and not sens:
                continue
            L, info = fit_full_nca(metrics[e], [st.S.lm(c) for c in dev], n)
            nca_info[key] = info
            arms.append(Arm(key, lambda b, L=L, m=metrics[e]: LinearMetric(m, L), True, False))
        name = f"nwicu-{e}"
        run_unit(name, lm, fold, arms, base, mcfg, seed0, units)
        names = [a.name for a in arms]
        r = {}
        for a_key in ("a", "a_logloss"):
            p = collect(name, lm, fold, names, base, units, a_key)
            r[a_key] = contrasts(p, base, np.asarray(lm.y), lm.ft.subjects,
                                 [n for n in names if n.startswith("nca")], SEED + 1,
                                 N_BOOT if a_key == "a" else max(N_BOOT // 4, 100))
        r["nca_full"] = nca_info
        res["rotations"][e] = r
    res["seconds"] = time.perf_counter() - t0
    return res


# --------------------------------------------------------------------------- report
LABEL = {"cos": "COS", "mi": "MI", "ncad": "NCA-diag", "ncaf": "NCA-full", "ncaf2k": "NCA-full (2k)",
         "ncaf10k": "NCA-full (10k)"}
USE_L = {"alone": "alone", "mix_lr": "mix, logistic", "mix_gbm": "mix, LightGBM"}


def _units_of(res: dict):
    for X, r in res.items():
        if X == "nwicu":
            for e, rr in r.get("rotations", {}).items():
                yield f"NWICU ({'+'.join(c for c in ROTATIONS if c != e)})", rr
        elif r:
            yield X, r


def _f(d: dict, f: str = "+.4f") -> str:
    return f"{d['est']:{f}} ({d['ci95'][0]:{f}} to {d['ci95'][1]:{f}})"


def render(res: dict) -> str:
    L = ["# Tier 1: the factors behind the learned-metric vs cosine reversal, in PERISIM's setting", "",
         "Pre-registered: docs/prereg_pram_factors.md (OSF). Δ = AUROC(NCA arm) − AUROC(COS arm) with the same vote "
         f"and use; {N_BOOT:,} patient resamples. Home levels: alone, V-ours, NCA-diag. A switch is sufficient when "
         "it changes the verdict on Δ.", "", "## Rule (pre-specified)", "",
         "| target | home Δ (verdict) | switch | Δ after switch (verdict) | sufficient | effect (95 % CI) |",
         "|---|---|---|---|---|---|"]
    for name, r in _units_of(res):
        rule = r["a"]["rule"]["ncad"]
        h = rule["home"]
        first = True
        for F, s in rule["switch"].items():
            eff = _f(rule["effects"][F]) if F in rule["effects"] else ""
            L.append(f"| {name if first else ''} | {_f(h['delta']) + ' (' + h['verdict'] + ')' if first else ''} | "
                     f"{F} | {_f(s['delta'])} ({s['verdict']}) | {'**yes**' if s['sufficient'] else 'no'} | {eff} |")
            first = False
    L += ["", "## Every cell: Δ AUROC against COS", "", "| target | metric | vote | use | Δ AUROC (95 % CI) | verdict | "
          "Δ log-loss (95 % CI) |", "|---|---|---|---|---|---|---|"]
    for name, r in _units_of(res):
        for k, c in r["a"]["cells"].items():
            m, v, u = k.split("|")
            L.append(f"| {name} | {LABEL.get(m, m)} | {v} | {USE_L[u]} | {_f(c['delta_auroc'])} | {c['verdict']} | "
                     f"{_f(c['delta_logloss'])} |")
    L += ["", "## Every arm", "", "| target | arm | AUROC | log-loss | CITL | slope |", "|---|---|---|---|---|---|"]
    for name, r in _units_of(res):
        for k, a in r["a"]["arms"].items():
            L.append(f"| {name} | {k} | {a['auroc']:.4f} | {a['logloss']:.4f} | {a['citl']:+.3f} | {a['slope']:.3f} |")
    L += ["", "## Sensitivity: a chosen by log-loss", "", "| target | home Δ | U switch Δ (verdict) |", "|---|---|---|"]
    for name, r in _units_of(res):
        rule = r["a_logloss"]["rule"]["ncad"]
        s = rule["switch"].get("U")
        if s:
            L.append(f"| {name} | {_f(rule['home']['delta'])} | {_f(s['delta'])} ({s['verdict']}) |")
    L += ["", "## Complementarity (descriptive): standalone kNN against the base model", "",
          "| target | arm, vote | base | corr(p_kNN, p_base) | Spearman(p_kNN, y − p_base) |", "|---|---|---|---|---|"]
    for name, r in _units_of(res):
        for k, c in r["a"]["complementarity"].items():
            arm, v, _, b = k.split("|")
            L.append(f"| {name} | {LABEL.get(arm, arm)}, {v} | {b} | {c['corr_base']:.3f} | {c['spearman_residual']:.3f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="*", default=list(HYPO_SITES) + ["nwicu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="quick frozen artefacts (tests)")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the 2k / 10k NCA-full arms (tests)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.external import Stage
    from hypo.frozen import check, external_dir

    rec = check()
    od = external_dir().with_name("pram_factors_t1")
    st = Stage(quick=a.quick, threads=a.threads)
    res = {"freeze": rec.get("tag"), "sites": {}}
    for X in a.sites:
        if X == "nwicu":
            res["sites"][X] = nwicu_target(st, od / "units", a.gate_only, not a.no_sensitivity, a.quick)
        else:
            res["sites"][X] = hypo_target(st, X, od / "units", a.gate_only, not a.no_sensitivity)
    check()
    if a.gate_only:
        LOG.info("t1: gates passed for %s", ", ".join(a.sites))
        return 0
    (od / "t1.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    rep = external_dir().parent / "reports" / "pram_factors_t1.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(render(res["sites"]), encoding="utf-8")
    LOG.info("t1: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
