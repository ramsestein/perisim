"""Transport INSPIRE → iPROVE: a kNN with NCA weights learned in INSPIRE, used in iPROVE with iPROVE's own patients as
the case base, against logistic regression and LightGBM trained in INSPIRE and only recalibrated in iPROVE (plan:
docs/DECISIONS.md 2026-11-07).

Target: intraoperative vasoactive drugs (`usoFarmacosVasoactivos`) from preoperative variables only
(scripts/iprove_build_inspire.py). Nothing is refitted on iPROVE except what each arm declares:
  knn_uniform        all weights 1 (R's reference)
  knn_mi             weights ∝ mutual information with the target in INSPIRE (a PRAM-like fixed metric)
  knn_nca            NCA weights learned in INSPIRE (transported)
  knn_nca_local      NCA weights learned on iPROVE's case-base patients of each fold (reference, refitted)
  lr_raw, lgbm_raw   INSPIRE models applied as they are
  lr_recal, lgbm_recal   the same, logistic recalibration (intercept and slope) fitted on iPROVE's other folds
  lr_local, lgbm_local   trained on iPROVE's other folds (ceiling, refitted)
Every kNN arm uses INSPIRE's axes (each variable's mid-rank in INSPIRE, scaled to unit mean square pair distance),
k = 50 distinct patients, the smoothed vote p = (n1 + α·P·2)/(k + 2α) with P the case base's prevalence and α chosen
by leave-one-out on the case base, as in `hypo`. iPROVE folds: five, stratified, fixed seed.

Usage:  python scripts/iprove_transport.py [--threads 8]
Outputs (aggregates only): <workdir>/experiments/iprove/transport.json, transport.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from iprove_build_inspire import PREDICTORS, TARGET  # noqa: E402

WORK = Path(os.environ.get("PERISIM_WORKDIR", "work")) / "experiments" / "iprove"
SEED = 20261107
K = 50
ALPHAS = (0.1, 0.5, 1, 2, 5, 10)
N_BOOT = 2000
N_FOLDS = 5
NCA = dict(n_candidates=500, batch_size=256, lr=0.05, refresh_every=5, eps0=1e-4, label_smoothing=1e-3, epochs=30)
R_MARGIN, MIN_DEN = 0.80, 0.005


# --------------------------------------------------------------------------- geometry
class Axes:
    """Each variable's mid-rank in the source (ties share their mean rank), scaled to unit mean square distance."""

    def fit(self, X: np.ndarray) -> "Axes":
        self.grid, self.val = [], []
        for j in range(X.shape[1]):
            u, c = np.unique(X[:, j], return_counts=True)
            mid = (np.cumsum(c) - c / 2) / c.sum()
            self.grid.append(u)
            self.val.append(mid)
        Z = self.raw(X)
        self.scale = np.sqrt(2 * Z.var(0) + 1e-12)          # E(z_a − z_b)² = 2·Var(z)
        return self

    def raw(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.interp(X[:, j], self.grid[j], self.val[j]) for j in range(X.shape[1])])

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self.raw(X) / self.scale).astype(np.float32)


def nca_weights(E: np.ndarray, y: np.ndarray, seed: int, threads: int) -> np.ndarray:
    import torch
    from perisim.metric.nca import DiagonalNCA

    torch.set_num_threads(max(1, threads))
    m = DiagonalNCA(np.arange(E.shape[1]), E.shape[1], 2, seed=seed,
                    **{k: v for k, v in NCA.items() if k != "epochs"})
    m.fit(E, y.astype(np.int64), epochs=NCA["epochs"], subjects=np.arange(E.shape[0]))
    return np.asarray(m.weights, dtype=np.float64)


def mi_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.metrics import mutual_info_score
    w = np.array([mutual_info_score(X[:, j], y) for j in range(X.shape[1])])
    w = np.maximum(w, 0)
    return w / w.sum() * X.shape[1]


class KNN:
    """k nearest case-base patients in the weighted axes; α by leave-one-out on the case base."""

    def __init__(self, w: np.ndarray, k: int = K):
        self.sw, self.k = np.sqrt(np.asarray(w, dtype=np.float64)), k

    def fit(self, E: np.ndarray, y: np.ndarray) -> "KNN":
        from sklearn.neighbors import NearestNeighbors
        self.y, self.P = y.astype(np.float64), float(y.mean())
        self.nn = NearestNeighbors(n_neighbors=min(self.k + 1, len(y)), algorithm="brute").fit(E * self.sw)
        _, I = self.nn.kneighbors(E * self.sw)
        n1 = self.y[I[:, 1:self.k + 1]].sum(1)                    # leave the patient itself out
        best = None
        for a in ALPHAS:
            p = (n1 + a * self.P * 2) / (self.k + 2 * a)
            ll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
            best = min(best, (ll, a)) if best else (ll, a)
        self.alpha = best[1]
        return self

    def predict(self, Q: np.ndarray) -> np.ndarray:
        _, I = self.nn.kneighbors(Q * self.sw, n_neighbors=self.k)
        return (self.y[I].sum(1) + self.alpha * self.P * 2) / (self.k + 2 * self.alpha)


# --------------------------------------------------------------------------- models
def fit_lr(X, y, seed):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=5, scoring="neg_log_loss",
                                                                max_iter=2000, random_state=seed)).fit(X, y)


def fit_lgbm(X, y, seed, threads):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    best = None
    for nl in (7, 15, 31):
        for mcs in (20, 100):
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=nl, min_child_samples=mcs,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                                   n_jobs=threads, deterministic=True, verbose=-1)
            s = cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                scoring="neg_log_loss").mean()
            best = max(best, (s, nl, mcs)) if best else (s, nl, mcs)
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=best[1], min_child_samples=best[2],
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                              n_jobs=threads, deterministic=True, verbose=-1).fit(X, y)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recal(p_tr, y_tr, p_te):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_tr)[:, None], y_tr)
    return m.predict_proba(logit(p_te)[:, None])[:, 1]


# --------------------------------------------------------------------------- statistics
def loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calib(p, y):
    from sklearn.linear_model import LogisticRegression
    x = logit(p)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(x[:, None], y).coef_[0, 0]
    # CITL: the intercept with the logit as offset (Newton steps)
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(x + a)))
        a += (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
    return float(a), float(slope)


def summarise(P: dict, y: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    n = y.size
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(np.float64)
    L = {a: loss(p, y) for a, p in P.items()}
    B = {a: W @ v / n for a, v in L.items()}
    ci = lambda b, lv=0.95: [float(np.quantile(b, (1 - lv) / 2)), float(np.quantile(b, 1 - (1 - lv) / 2))]  # noqa
    arms = {}
    for a, p in P.items():
        c, s = calib(p, y)
        arms[a] = {"logloss": float(L[a].mean()), "logloss_ci95": ci(B[a]), "auroc": float(roc_auc_score(y, p)),
                   "auprc": float(average_precision_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
                   "citl": c, "slope": s}

    def R(arm):
        den_b, den = B["knn_uniform"] - B["knn_nca_local"], arms["knn_uniform"]["logloss"] - arms["knn_nca_local"]["logloss"]
        lo, hi = ci(den_b)
        if den < MIN_DEN or lo <= 0:
            return {"est": None, "verdict": "not evaluable", "denominator": {"est": den, "ci95": [lo, hi]}}
        r = (B["knn_uniform"] - B[arm]) / den_b
        c90 = ci(r, 0.90)
        return {"est": float((arms["knn_uniform"]["logloss"] - arms[arm]["logloss"]) / den), "ci90": c90,
                "verdict": "equivalent" if c90[0] >= R_MARGIN else "not shown equivalent",
                "denominator": {"est": den, "ci95": [lo, hi]}}

    def D(a, b):
        return {"est": arms[a]["logloss"] - arms[b]["logloss"], "ci95": ci(B[a] - B[b])}

    return {"n": int(n), "prevalence": float(y.mean()), "arms": arms,
            "R": {"knn_nca": R("knn_nca"), "knn_mi": R("knn_mi")},
            "diff": {"knn_nca − lr_recal": D("knn_nca", "lr_recal"), "knn_nca − lgbm_recal": D("knn_nca", "lgbm_recal"),
                     "knn_nca − knn_mi": D("knn_nca", "knn_mi"), "knn_nca − lgbm_local": D("knn_nca", "lgbm_local")}}


def rule(s: dict) -> dict:
    def verdict(d):
        return "kNN better" if d["ci95"][1] < 0 else "kNN worse" if d["ci95"][0] > 0 else "no difference"
    k = s["arms"]["knn_nca"]
    return {"Q1_R_knn_nca": s["R"]["knn_nca"]["verdict"],
            "Q2_vs_lr_recal": verdict(s["diff"]["knn_nca − lr_recal"]),
            "Q2_vs_lgbm_recal": verdict(s["diff"]["knn_nca − lgbm_recal"]),
            "Q3_knn_nca_calibration": {"citl": k["citl"], "slope": k["slope"]}}


# --------------------------------------------------------------------------- main
def run(threads: int) -> dict:
    import polars as pl
    from sklearn.model_selection import StratifiedKFold

    S = pl.read_parquet(WORK / "inspire.parquet")
    T = pl.read_parquet(WORK / "iprove.parquet")
    Xs, ys = S.select(PREDICTORS).to_numpy().astype(np.float64), S[TARGET].to_numpy().astype(np.int64)
    Xt, yt = T.select(PREDICTORS).to_numpy().astype(np.float64), T[TARGET].to_numpy().astype(np.int64)
    ax = Axes().fit(Xs)
    Es, Et = ax.transform(Xs), ax.transform(Xt)
    w = {"knn_uniform": np.ones(len(PREDICTORS)), "knn_mi": mi_weights(Xs, ys),
         "knn_nca": nca_weights(Es, ys, SEED, threads)}
    lr, lg = fit_lr(Xs, ys, SEED), fit_lgbm(Xs, ys, SEED, threads)
    raw = {"lr_raw": lr.predict_proba(Xt)[:, 1], "lgbm_raw": lg.predict_proba(Xt)[:, 1]}
    P = {a: np.full(yt.size, np.nan) for a in ["knn_uniform", "knn_mi", "knn_nca", "knn_nca_local", "lr_raw",
                                              "lgbm_raw", "lr_recal", "lgbm_recal", "lr_local", "lgbm_local"]}
    P["lr_raw"], P["lgbm_raw"] = raw["lr_raw"], raw["lgbm_raw"]
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xt, yt))
    for f, (tr, te) in enumerate(folds):
        wl = nca_weights(Et[tr], yt[tr], SEED + 1 + f, threads)
        for a, ww in {**w, "knn_nca_local": wl}.items():
            P[a][te] = KNN(ww).fit(Et[tr], yt[tr]).predict(Et[te])
        for a in ("lr", "lgbm"):
            P[f"{a}_recal"][te] = recal(raw[f"{a}_raw"][tr], yt[tr], raw[f"{a}_raw"][te])
        P["lr_local"][te] = fit_lr(Xt[tr], yt[tr], SEED).predict_proba(Xt[te])[:, 1]
        P["lgbm_local"][te] = fit_lgbm(Xt[tr], yt[tr], SEED, threads).predict_proba(Xt[te])[:, 1]
        print(f"iprove transport: fold {f} done", flush=True)
    s = summarise(P, yt.astype(np.float64))
    s["n_source"], s["prevalence_source"] = int(ys.size), float(ys.mean())
    s["weights"] = {a: dict(zip(PREDICTORS, (v / v.sum()).round(4).tolist())) for a, v in w.items() if a != "knn_uniform"}
    s["rule"] = rule(s)
    return s


def render(s: dict) -> str:
    def r(d):
        return "not evaluable" if d.get("est") is None else f"{d['est']:.2f} (90 % CI {d['ci90'][0]:.2f}–{d['ci90'][1]:.2f}; {d['verdict']})"
    L = ["# INSPIRE → iPROVE: intraoperative vasoactive drugs from preoperative variables", "",
         "Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. Source INSPIRE "
         f"({s['n_source']:,} patients, {s['prevalence_source']:.1%} with vasoactives); target iPROVE ({s['n']:,} "
         f"patients, {s['prevalence']:.1%}). Five folds in iPROVE; {N_BOOT:,} patient resamples.", "",
         "| arm | refitted on iPROVE | log-loss (95 % CI) | AUROC | AUPRC | Brier | CITL | slope |",
         "|---|---|---|---|---|---|---|---|"]
    refit = {"knn_uniform": "case base only", "knn_mi": "case base only", "knn_nca": "case base only",
             "knn_nca_local": "weights + case base", "lr_raw": "nothing", "lgbm_raw": "nothing",
             "lr_recal": "recalibration", "lgbm_recal": "recalibration", "lr_local": "everything", "lgbm_local": "everything"}
    for a, m in s["arms"].items():
        L.append(f"| {a} | {refit[a]} | {m['logloss']:.4f} ({m['logloss_ci95'][0]:.4f}–{m['logloss_ci95'][1]:.4f}) | "
                 f"{m['auroc']:.3f} | {m['auprc']:.3f} | {m['brier']:.4f} | {m['citl']:+.3f} | {m['slope']:.2f} |")
    L += ["", f"- R, NCA transported: {r(s['R']['knn_nca'])}", f"- R, MI weights: {r(s['R']['knn_mi'])}"]
    for k, d in s["diff"].items():
        L.append(f"- {k}: {d['est']:+.4f} ({d['ci95'][0]:+.4f} to {d['ci95'][1]:+.4f})")
    L += ["", "## Weights learned in INSPIRE (share)", "", "| variable | NCA | MI |", "|---|---|---|"]
    for v in PREDICTORS:
        L.append(f"| {v} | {s['weights']['knn_nca'][v]:.3f} | {s['weights']['knn_mi'][v]:.3f} |")
    L += ["", "## Rule (pre-specified)", "", "```", json.dumps(s["rule"], indent=1), "```", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    s = run(a.threads)
    (WORK / "transport.json").write_text(json.dumps(s, indent=1, default=float), encoding="utf-8")
    (WORK / "transport.md").write_text(render(s), encoding="utf-8")
    print(f"iprove transport: {WORK / 'transport.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
