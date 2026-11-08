"""Pre-registered (OSF; docs/prereg_pram_factors.md §4, §6), Tier 2: PRAM's setting rebuilt from its published recipe
(MIMIC-IV → eICU-CRD, 42 static first-24 h features, acute kidney injury and 168 h mortality). The script first
checks the replication of PRAM's Table S13 (R0), then crosses use × vote × NCA implementation.

Run after scripts/pram_build.py, with `hypo`'s frozen code for the kNN search, the vote settings and the diagonal NCA
(PERISIM_SRC=frozen/hypo/src, or a v0.5 checkout).

Usage:  python scripts/pram_factors_t2.py [--threads 8] [--no-sensitivity]
Outputs (aggregates only): work/experiments/pram/t2.json, t2.md; row-level units in pram/units/ (never published).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import _frozen  # noqa: E402
    _frozen.use_src()
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

import posthoc_cosine as PC  # noqa: E402
import posthoc_fixed_metric as FM  # noqa: E402
import pram_build as PB  # noqa: E402
import pram_factors_t1 as T1  # noqa: E402

LOG = logging.getLogger("pram.t2")
SEED = 42                                   # PRAM's seed (Table S25)
N_BOOT = 2000
BOOT_SEED = 20261110
NCA_N = {"ncaf": 5000, "ncaf2k": 2000, "ncaf10k": 10000}
VOTES, BASES, USES = T1.VOTES, T1.BASES, T1.USES
PRAM_S13 = {"aki": {"cos": 0.0197, "ncaf": 0.0105, "ncaf_t": 0.0131, "mi": 0.0216},
            "death": {"cos": 0.0358, "ncaf": 0.0268, "ncaf_t": 0.0262, "mi": 0.0367}}
PRAM_FROZEN_LR = {"aki": 0.690, "death": 0.774}


def work() -> Path:
    return Path(os.environ.get("PERISIM_WORKDIR", "work")) / "experiments" / "pram"


# --------------------------------------------------------------------------- data and PRAM's pipeline
def load_sites():
    import polars as pl
    return {s: pl.read_parquet(work() / f"{s}.parquet") for s in ("mimic", "eicu")}


def dev_split(df) -> np.ndarray:
    """0 = training (56 %), 1 = validation (14 %), 2 = internal test (30 %), stratified by both outcomes, seed 42."""
    from sklearn.model_selection import train_test_split

    strata = df["aki"].cast(int).to_numpy() * 2 + df["death"].cast(int).to_numpy()
    idx = np.arange(df.height)
    tr, rest = train_test_split(idx, train_size=0.56, stratify=strata, random_state=SEED)
    va, _ = train_test_split(rest, train_size=0.14 / 0.44, stratify=strata[rest], random_state=SEED)
    out = np.full(df.height, 2)
    out[tr], out[va] = 0, 1
    return out


class Prep:
    """Imputation (chained equations) and standardisation of the numeric features, mode imputation of sex; fitted on
    the MIMIC-IV training split only."""

    def __init__(self, df_train):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.preprocessing import StandardScaler

        self.num = [c for c in PB.FEATURES if c != "sex_male"]
        Xn = df_train.select(self.num).to_numpy().astype(np.float64)
        self.imp = IterativeImputer(random_state=SEED).fit(Xn)
        self.sc = StandardScaler().fit(self.imp.transform(Xn))
        s = df_train["sex_male"].drop_nulls().to_numpy()
        self.sex_mode = float(np.bincount(s.astype(int)).argmax()) if s.size else 0.0

    def __call__(self, df) -> np.ndarray:
        Xn = self.sc.transform(self.imp.transform(df.select(self.num).to_numpy().astype(np.float64)))
        sex = df["sex_male"].fill_null(self.sex_mode).to_numpy().astype(np.float64)
        return np.column_stack([Xn, sex])


def base_models(X: np.ndarray, y: np.ndarray, threads: int):
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    gbm = CatBoostClassifier(iterations=500, depth=6, random_seed=SEED, verbose=0, thread_count=threads).fit(X, y)
    return {"lr": lambda Z: lr.predict_proba(Z)[:, 1], "gbm": lambda Z: gbm.predict_proba(Z)[:, 1]}


# --------------------------------------------------------------------------- metrics (as embeddings)
def mi_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_classif
    w = np.maximum(mutual_info_classif(X, y, random_state=SEED), 0.0)
    return w / w.sum() if w.sum() > 0 else np.full(X.shape[1], 1.0 / X.shape[1])


def stratified_sub(y: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    from sklearn.model_selection import train_test_split
    if n >= y.size:
        return np.arange(y.size)
    sel, _ = train_test_split(np.arange(y.size), train_size=n, stratify=y, random_state=seed)
    return np.sort(sel)


def full_nca(X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    s = stratified_sub(y, n)
    return NeighborhoodComponentsAnalysis(random_state=SEED).fit(X[s], y[s]).components_


def diag_nca(X: np.ndarray, y: np.ndarray, ncfg: dict) -> np.ndarray:
    from perisim.metric.nca import DiagonalNCA
    d = X.shape[1]
    nca = DiagonalNCA(np.arange(d), d, 2, n_candidates=int(ncfg["n_candidates"]), batch_size=int(ncfg["batch_size"]),
                      lr=float(ncfg["lr"]), refresh_every=int(ncfg["refresh_every"]), eps0=float(ncfg["eps0"]),
                      label_smoothing=float(ncfg["label_smoothing"]), seed=SEED)
    nca.fit(np.ascontiguousarray(X / np.sqrt(2.0), dtype=np.float32), y.astype(int), epochs=int(ncfg["epochs"]),
            subjects=np.arange(X.shape[0]).astype(str))
    return nca.weights


def embed(kind: str, X: np.ndarray, par) -> np.ndarray:
    if kind == "cos":
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return (X / np.where(n > 1e-12, n, 1.0)).astype(np.float32)
    if kind == "mi":
        return (X * np.sqrt(par)).astype(np.float32)
    if kind.startswith("ncaf"):
        return (X @ par.T).astype(np.float32)
    if kind.startswith("ncad"):
        return (X / np.sqrt(2.0) * np.sqrt(par)).astype(np.float32)
    raise ValueError(kind)


# --------------------------------------------------------------------------- kNN with both votes
def knn_votes(Eb: np.ndarray, yb: np.ndarray, Eq: np.ndarray, cosine: bool, mcfg: dict, seed: int) -> dict:
    """k nearest bank stays of each query (one row per patient, so neighbours are distinct patients); V-ours with α by
    leave-one-out on a random subset of the bank, V-inv with weight 1/(d + 1e-6)."""
    from perisim.knn.index import search

    k = min(int(mcfg["k"]), Eb.shape[0] - 1)
    yb = yb.astype(np.float64)
    prior = float(yb.mean())
    rng = np.random.default_rng(seed)
    rows = rng.choice(Eb.shape[0], size=min(int(mcfg["alpha_queries"]), Eb.shape[0]), replace=False)
    D, I = search(Eb, Eb[rows], k + 1)
    keep = I != rows[:, None]                          # drop the query itself (first occurrence kept otherwise)
    drop_last = keep.all(1)
    keep[drop_last, -1] = False
    Il = I[keep].reshape(rows.size, k)
    n1 = yb[Il].sum(1)
    best = (np.inf, 1.0)
    for a in mcfg["alpha_grid"]:
        p = (n1 + float(a) * prior * 2) / (k + 2 * float(a))
        ll = float(-np.mean(np.where(yb[rows] == 1, np.log(np.clip(p, 1e-15, 1)), np.log(np.clip(1 - p, 1e-15, 1)))))
        best = min(best, (ll, float(a)))
    alpha = best[1]
    D, I = search(Eb, Eq, k)
    ours = (yb[I].sum(1) + alpha * prior * 2) / (k + 2 * alpha)
    inv = T1.inv_vote(D, I, np.ones_like(I, dtype=bool), yb, cosine, prior)
    return {"ours": ours, "inv": inv, "alpha": alpha}


# --------------------------------------------------------------------------- one outcome
def run_outcome(o: str, S: dict, split: np.ndarray, prep: Prep, mcfg: dict, ncfg: dict, sens: bool, threads: int,
                units: Path) -> dict:
    from sklearn.model_selection import StratifiedKFold

    dev, tgt = S["mimic"], S["eicu"]
    Xd = prep(dev)[split == 0]
    yd = dev[o].cast(int).to_numpy()[split == 0]
    Xt, yt = prep(tgt), tgt[o].cast(int).to_numpy()
    t0 = time.perf_counter()
    bases = base_models(Xd, yd, threads)
    base = {b: np.asarray(f(Xt), dtype=np.float64) for b, f in bases.items()}
    par = {"cos": None, "mi": mi_weights(Xd, yd)}
    info = {"fit_seconds": {}}
    for key, n in NCA_N.items():
        if key != "ncaf" and not sens:
            continue
        t = time.perf_counter()
        par[key] = full_nca(Xd, yd, n)
        info["fit_seconds"][key] = time.perf_counter() - t
    t = time.perf_counter()
    par["ncad"] = diag_nca(Xd, yd, ncfg)
    info["fit_seconds"]["ncad"] = time.perf_counter() - t
    info["weights_ncad"] = dict(zip(PB.FEATURES, np.round(par["ncad"] / par["ncad"].sum(), 4).tolist()))
    info["weights_mi"] = dict(zip(PB.FEATURES, np.round(par["mi"], 4).tolist()))
    LOG.info("t2 %s: development fits done (%.0f s)", o, time.perf_counter() - t0)
    arms = list(par) + ["ncaf_t", "ncad_t"]
    units.mkdir(parents=True, exist_ok=True)
    pred = {f"{a}|{v}": np.full(yt.size, np.nan) for a in arms for v in VOTES}
    A = {f"{a}|{v}|{b}": np.full(yt.size, np.nan) for a in arms for v in VOTES for b in BASES}
    A_ll = {k: v.copy() for k, v in A.items()}
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for f, (bank, test) in enumerate(skf.split(Xt, yt)):
        path = units / f"{o}_f{f}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=True)
            res = z["res"].item()
        else:
            t = time.perf_counter()
            fp = dict(par)
            fp["ncaf_t"] = full_nca(Xt[bank], yt[bank], NCA_N["ncaf"])          # PRAM's M3: fitted on the bank
            fp["ncad_t"] = diag_nca(Xt[bank], yt[bank], ncfg)
            inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(bank, yt[bank]))
            res = {}
            for a in arms:
                kind = a.replace("_t", "")
                E = embed(kind, Xt, fp[a])
                cos = kind == "cos"
                out = knn_votes(E[bank], yt[bank], E[test], cos, mcfg, SEED + f)
                oof = {v: np.full(bank.size, np.nan) for v in VOTES}
                for j, (ib, iq) in enumerate(inner):
                    r = knn_votes(E[bank[ib]], yt[bank[ib]], E[bank[iq]], cos, mcfg, SEED + f + j + 1)
                    for v in VOTES:
                        oof[v][iq] = r[v]
                ch = {v: {b: T1.choose_a(base[b][bank], oof[v], yt[bank].astype(float)) for b in BASES} for v in VOTES}
                res[a] = {"ours": out["ours"], "inv": out["inv"], "alpha": out["alpha"], "choice": ch}
            np.savez(path, res=np.array(res, dtype=object), test=test)
            LOG.info("t2 %s fold %d done (%.0f s)", o, f, time.perf_counter() - t)
        for a in arms:
            for v in VOTES:
                pred[f"{a}|{v}"][test] = res[a][v]
                for b in BASES:
                    A[f"{a}|{v}|{b}"][test] = res[a]["choice"][v][b]["a"]
                    A_ll[f"{a}|{v}|{b}"][test] = res[a]["choice"][v][b]["a_logloss"]
        info.setdefault("alpha", {})[f"f{f}"] = {a: res[a]["alpha"] for a in arms}
        info.setdefault("a_chosen", {})[f"f{f}"] = {a: {v: {b: res[a]["choice"][v][b]["a"] for b in BASES}
                                                        for v in VOTES} for a in arms}
    out = {"n_target": int(yt.size), "prevalence_target": float(yt.mean()), "n_dev_train": int(yd.size),
           "prevalence_dev_train": float(yd.mean()), "info": info}
    for key, AA, nb in (("a", A, N_BOOT), ("a_logloss", A_ll, max(N_BOOT // 4, 100))):
        p = {}
        for a in arms:
            for v in VOTES:
                p[f"{a}|{v}|alone"] = pred[f"{a}|{v}"]
                for b in BASES:
                    p[f"{a}|{v}|mix_{b}"] = (1 - AA[f"{a}|{v}|{b}"]) * base[b] + AA[f"{a}|{v}|{b}"] * pred[f"{a}|{v}"]
        out[key] = contrasts(p, base, yt, nb)
    out["seconds"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------- contrasts and the rule (Tier 2 home)
HOME = "ncaf|inv|mix_lr"
SWITCH = {"U": "ncaf|inv|alone", "V": "ncaf|ours|mix_lr", "I": "ncad|inv|mix_lr", "UV": "ncaf|ours|alone",
          "UI": "ncad|inv|alone", "VI": "ncad|ours|mix_lr", "UVI": "ncad|ours|alone"}


def contrasts(p: dict, base: dict, y: np.ndarray, n_boot: int) -> dict:
    from hypo.stats import calibration_line_w, citl_w, logit

    allp = {**p, **{f"base|{b}": base[b] for b in BASES}}
    ids = np.arange(y.size).astype(str)
    B = PC.boot(allp, y, ids, n_boot, BOOT_SEED)
    yf = y.astype(np.float64)

    def delta(cell):
        arm, v, u = cell.split("|")
        c = f"cos|{v}|{u}"
        return B["est_a"][cell] - B["est_a"][c], B["A"][cell] - B["A"][c]

    def pk(est, arr):
        return {"est": float(est), "ci95": FM._ci(arr)}

    arms = {}
    for k, v in allp.items():
        x = logit(np.clip(v, 1e-6, 1 - 1e-6))
        arms[k] = {"auroc": B["est_a"][k], "auroc_ci95": FM._ci(B["A"][k]), "logloss": B["est_l"][k],
                   "citl": citl_w(x, yf), "slope": calibration_line_w(x, yf)[1]}
    cells = {}
    for k in p:
        if not k.startswith("cos|"):
            e, a = delta(k)
            cells[k] = {"delta_auroc": pk(e, a), "verdict": T1.verdict(FM._ci(a))}
    h_e, h_a = delta(HOME)
    rule = {"home": {"cell": HOME, "delta": pk(h_e, h_a), "verdict": T1.verdict(FM._ci(h_a))}, "switch": {},
            "effects": {}}
    for F, cell in SWITCH.items():
        e, a = delta(cell)
        rule["switch"][F] = {"cell": cell, "delta": pk(e, a), "verdict": T1.verdict(FM._ci(a)),
                             "sufficient": T1.verdict(FM._ci(a)) != rule["home"]["verdict"]}
        if len(F) == 1:     # effect_F = Δ(PERISIM level) − Δ(PRAM level)
            rule["effects"][F] = pk(e - h_e, a - h_a)
    gains = {}
    for arm in ("cos", "mi", "ncaf", "ncaf_t"):
        c = f"{arm}|inv|mix_lr"
        gains[arm] = pk(B["est_a"][c] - B["est_a"]["base|lr"], B["A"][c] - B["A"]["base|lr"])
    comp = {}
    for k, v in p.items():
        if k.endswith("|alone"):
            for b in BASES:
                comp[f"{k}|{b}"] = {"corr_base": float(np.corrcoef(v, base[b])[0, 1]),
                                    "spearman_residual": T1._spearman(v, yf - base[b])}
    return {"arms": arms, "cells": cells, "rule": rule, "gain_over_frozen_lr": gains, "complementarity": comp}


def r0(res: dict) -> dict:
    """Replication of S13 at the home levels: Δ(NCA-full transferred vs COS) < 0 for both outcomes."""
    d = {o: res[o]["a"]["rule"]["home"]["delta"] for o in res}
    point = all(v["est"] < 0 for v in d.values())
    conf = all(v["ci95"][1] < 0 for v in d.values())
    return {"delta": d, "reproduced": point, "reproduced_with_confidence": conf}


def h_s(res: dict) -> dict:
    """H-S: at PERISIM's levels (alone, V-ours, NCA-diag transferred) the NCA advantage is not shown, both outcomes."""
    v = {o: res[o]["a"]["rule"]["switch"]["UVI"]["verdict"] for o in res}
    return {"verdicts": v, "supported": all(x != "NCA beats cosine" for x in v.values())}


# --------------------------------------------------------------------------- report
LABEL = {"cos": "COS", "mi": "MI", "ncad": "NCA-diag", "ncaf": "NCA-full", "ncaf2k": "NCA-full (2k)",
         "ncaf10k": "NCA-full (10k)", "ncaf_t": "NCA-full, target bank", "ncad_t": "NCA-diag, target bank"}


def render(out: dict, build_md: str | None) -> str:
    f = T1._f
    L = ["# Tier 2: PRAM's setting rebuilt (MIMIC-IV → eICU-CRD)", "",
         "Pre-registered: docs/prereg_pram_factors.md (OSF). Δ = AUROC(NCA arm) − AUROC(COS arm), same vote and use; "
         f"{N_BOOT:,} stay resamples. Home levels: mixture with the frozen logistic model, PRAM's vote 1/(d + 1e-6), "
         "NCA-full transferred.", "", "## R0: replication of PRAM's Table S13", ""]
    R = out["r0"]
    for o, d in R["delta"].items():
        L.append(f"- {o}: Δ(NCA-full transferred − cosine) = {f(d)}; PRAM: "
                 f"{PRAM_S13[o]['ncaf'] - PRAM_S13[o]['cos']:+.4f}")
    L += [f"- reproduced (point estimates): **{'yes' if R['reproduced'] else 'no'}**; with confidence: "
          f"**{'yes' if R['reproduced_with_confidence'] else 'no'}**", "",
          "## Fidelity (descriptive)", "", "| outcome | quantity | ours | PRAM |", "|---|---|---|---|"]
    for o in ("aki", "death"):
        r = out["outcomes"][o]
        L.append(f"| {o} | prevalence in eICU | {r['prevalence_target']:.1%} | {'6.1%' if o == 'aki' else '4.6%'} |")
        L.append(f"| {o} | frozen logistic AUROC in eICU | {r['a']['arms']['base|lr']['auroc']:.3f} | "
                 f"{PRAM_FROZEN_LR[o]:.3f} |")
        for arm in ("cos", "mi", "ncaf", "ncaf_t"):
            g = r["a"]["gain_over_frozen_lr"][arm]
            L.append(f"| {o} | gain over frozen logistic, {LABEL[arm]} mixture | {f(g)} | {PRAM_S13[o][arm]:+.4f} |")
    L += ["", "## Rule (pre-specified)", "", "| outcome | home Δ (verdict) | switch | Δ after switch (verdict) | sufficient | "
          "effect (95 % CI) |", "|---|---|---|---|---|---|"]
    for o in ("aki", "death"):
        rule = out["outcomes"][o]["a"]["rule"]
        h = rule["home"]
        for i, (F, s) in enumerate(rule["switch"].items()):
            L.append(f"| {o if i == 0 else ''} | {f(h['delta']) + ' (' + h['verdict'] + ')' if i == 0 else ''} | {F} | "
                     f"{f(s['delta'])} ({s['verdict']}) | {'**yes**' if s['sufficient'] else 'no'} | "
                     f"{f(rule['effects'][F]) if F in rule['effects'] else ''} |")
    hs = out["h_s"]
    L += ["", f"**H-S** (PERISIM's levels in PRAM's setting: no NCA advantage for both outcomes): "
          f"{'supported' if hs['supported'] else 'not supported'} ({hs['verdicts']}).", "",
          "## Every cell: Δ AUROC against COS", "", "| outcome | metric | vote | use | Δ AUROC (95 % CI) | verdict |",
          "|---|---|---|---|---|---|"]
    for o in ("aki", "death"):
        for k, c in out["outcomes"][o]["a"]["cells"].items():
            m, v, u = k.split("|")
            L.append(f"| {o} | {LABEL.get(m, m)} | {v} | {T1.USE_L[u]} | {f(c['delta_auroc'])} | {c['verdict']} |")
    L += ["", "## Every arm", "", "| outcome | arm | AUROC | log-loss | CITL | slope |", "|---|---|---|---|---|---|"]
    for o in ("aki", "death"):
        for k, a in out["outcomes"][o]["a"]["arms"].items():
            L.append(f"| {o} | {k} | {a['auroc']:.4f} | {a['logloss']:.4f} | {a['citl']:+.3f} | {a['slope']:.3f} |")
    for o in PB.SENSITIVITY:
        if o in out["outcomes"]:
            rule = out["outcomes"][o]["a"]["rule"]
            r = out["outcomes"][o]
            L += ["", f"## Sensitivity: {o} (KDIGO's 48 h window for the 0.3 mg/dL rise; prevalence in eICU "
                  f"{r['prevalence_target']:.1%}; deviation 2026-11-08)", "",
                  f"- home Δ: {f(rule['home']['delta'])} ({rule['home']['verdict']}); frozen logistic AUROC "
                  f"{r['a']['arms']['base|lr']['auroc']:.3f}; cosine-mixture gain {f(r['a']['gain_over_frozen_lr']['cos'])}"]
            for F, s_ in rule["switch"].items():
                L.append(f"- switch {F}: {f(s_['delta'])} ({s_['verdict']}){' — sufficient' if s_['sufficient'] else ''}")
    L += ["", "## Sensitivity: a chosen by log-loss", "", "| outcome | home Δ | U switch Δ (verdict) |", "|---|---|---|"]
    for o in ("aki", "death"):
        rule = out["outcomes"][o]["a_logloss"]["rule"]
        L.append(f"| {o} | {f(rule['home']['delta'])} | {f(rule['switch']['U']['delta'])} ({rule['switch']['U']['verdict']}) |")
    L += ["", "## Complementarity (descriptive)", "", "| outcome | arm, vote | base | corr(p_kNN, p_base) | "
          "Spearman(p_kNN, y − p_base) |", "|---|---|---|---|---|"]
    for o in ("aki", "death"):
        for k, c in out["outcomes"][o]["a"]["complementarity"].items():
            arm, v, _, b = k.split("|")
            L.append(f"| {o} | {LABEL.get(arm, arm)}, {v} | {b} | {c['corr_base']:.3f} | {c['spearman_residual']:.3f} |")
    if build_md:
        L += ["", "## Cohort", "", build_md.split("\n", 2)[2] if build_md.count("\n") > 2 else build_md]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tests: small α subset")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.config import load

    cfg = load()
    mcfg, ncfg = dict(cfg["metric"]), dict(cfg["metric"]["nca"])
    if a.quick:
        mcfg["alpha_queries"], ncfg["epochs"] = 500, 2
    S = load_sites()
    split = dev_split(S["mimic"])
    import polars as pl
    prep = Prep(S["mimic"].filter(pl.Series(split == 0)))
    out = {"outcomes": {}, "n_dev": int(S["mimic"].height), "split": {k: int((split == i).sum()) for i, k in
                                                                       enumerate(("train", "validation", "test"))}}
    for o in PB.OUTCOMES + PB.SENSITIVITY:       # aki_kdigo48: deviation 2026-11-08, sensitivity only
        out["outcomes"][o] = run_outcome(o, S, split, prep, mcfg, ncfg, not a.no_sensitivity, a.threads,
                                         work() / "units")
    primary = {o: out["outcomes"][o] for o in PB.OUTCOMES}
    out["r0"] = r0(primary)
    out["h_s"] = h_s(primary)
    (work() / "t2.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    bm = work() / "build.md"
    (work() / "t2.md").write_text(render(out, bm.read_text(encoding="utf-8") if bm.exists() else None),
                                  encoding="utf-8")
    LOG.info("t2: %s", work() / "t2.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
