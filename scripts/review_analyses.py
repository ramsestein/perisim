"""Post-hoc, EXPLORATORY review analyses for the article (docs/DECISIONS.md, 2026-11-15, "Review analyses").

Blocks (each resumable; patient-level units in <experiments>/hypo/review/units, never published):
  est  primary estimands: D (90 % CI), ΔNB transported − local, R with the count of G_local ≤ 0, eICU cluster bootstrap
  A    learning curves in local labelled patients (PERISIM vs LightGBM recalibrated: intercept, logistic, isotonic)
  B    comparators of the geometry: raw, shapes + uniform weights, shapes + NCA, a 1-D score kNN, an additive model,
       persistence
  C    uncertainty of learning the metric: B metrics re-learnt on patient bootstrap resamples of the development pool
  D    cohort: dropped and partly observed windows, sustained hypotension, strata, ties, MAP weights
  E    deployment and use: artefact without exact ranks, episode level, neighbour stability
  G    distillation cells (aggregates file)
  H    the autoregressive outcome: a blanking hour [L + 1, L + 6), the stratum last MAP ≥ 75 mmHg, gains over persistence

Usage:  python scripts/review_analyses.py [--blocks est A B C D E G H] [--units inspire mimic eicu nwicu] [--threads 16]
                                          [--n-src 10] [--distill-dir work_distill/distill]
Outputs (aggregates only): <experiments>/hypo/review/review.json and <experiments>/hypo/reports/review.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:                                    # v0.6 repository: $PERISIM_SRC (frozen code) first on sys.path
    import _frozen  # noqa: E402
    _frozen.use_src()
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

import posthoc_paper as PP  # noqa: E402

LOG = logging.getLogger("review")
SEED = 20261115
ROTATIONS = ("inspire", "mimic", "eicu")
CURVE_N = (50, 100, 250, 500, 1000, 2000, 4000, 8000)
N_BOOT, N_BOOT_AUC = 2000, 500
NB_T = (0.10, 0.20, 0.30)
NB_MARGIN = 0.002
MARGIN = 0.005
THR = 65.0
GATE_TOL = 1e-9
EPS = 1e-6
QUICK = False                 # tests: quick settings and frozen artefacts


# =========================================================================== small statistics (numpy, tested)
def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def recal(method: str, p_fit, y_fit, p_apply) -> np.ndarray:
    """Recalibrate p_apply with (p_fit, y_fit): 'intercept' (logit offset), 'logistic' (intercept and slope) or
    'isotonic'; the identity when the fitting set has one class."""
    p_fit, y_fit = np.asarray(p_fit, dtype=np.float64), np.asarray(y_fit, dtype=np.float64)
    if y_fit.size == 0 or y_fit.min() == y_fit.max():
        return np.asarray(p_apply, dtype=np.float64)
    if method == "intercept":
        a, _ = PP._logistic(logit(p_fit), y_fit, np.ones_like(y_fit), slope=False)
        return sigmoid(a + logit(p_apply))
    if method == "logistic":
        a, b = PP._logistic(logit(p_fit), y_fit, np.ones_like(y_fit), slope=True)
        return sigmoid(a + b * logit(p_apply))
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p_fit, y_fit)
        return np.clip(iso.predict(np.asarray(p_apply, dtype=np.float64)), EPS, 1 - EPS)
    raise ValueError(method)


def jaccard_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard overlap of two integer sets (rows of A and B; −1 = empty slot)."""
    out = np.empty(A.shape[0])
    for i in range(A.shape[0]):
        a, b = set(A[i][A[i] >= 0].tolist()), set(B[i][B[i] >= 0].tolist())
        u = len(a | b)
        out[i] = len(a & b) / u if u else 1.0
    return out


def rank_nodes_approx(x: np.ndarray, srt: np.ndarray, median: float, nodes: int = 101) -> np.ndarray:
    """The neutral rank axis as the artefact without exact ranks writes it: the rank (#{t < x} + #{t ≤ x}) / 2n at
    `nodes` quantiles of the training values, interpolated linearly; missing → the median's rank (exact)."""
    srt = np.asarray(srt, dtype=np.float64)
    n = srt.size
    r = lambda v: (np.searchsorted(srt, v, "left") + np.searchsorted(srt, v, "right")) / (2.0 * n)
    xs = np.unique(np.quantile(srt, np.linspace(0.0, 1.0, nodes)))
    ys = r(xs)
    x = np.asarray(x, dtype=np.float64)
    out = np.interp(np.where(np.isfinite(x), x, median), xs, ys)
    return np.where(np.isfinite(x), out, r(np.array([median]))[0])


def episodes(stay: np.ndarray, L: np.ndarray, y: np.ndarray, p: np.ndarray, onset: dict, t: float) -> dict:
    """Episode-level reading of landmark predictions at threshold t. `onset`: stay → first hour with hourly median
    MAP < 65 at or after its first landmark (stays without one are absent). An episode is detected when a landmark
    L with L ≤ T ≤ L + 5 (its window holds the onset T) has p ≥ t; lead = T − the earliest such L. False alerts:
    alerts at landmarks with y = 0; one patient-day = 12 two-hourly predictions."""
    alert = p >= t
    n_ep, det, leads = 0, 0, []
    order = np.lexsort((L, stay))
    s, l, a = stay[order], L[order], alert[order]
    bounds = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1, s.size]
    for i in range(bounds.size - 1):
        lo, hi = bounds[i], bounds[i + 1]
        T = onset.get(s[lo])
        if T is None:
            continue
        n_ep += 1
        win = (l[lo:hi] <= T) & (l[lo:hi] >= T - 5)
        hit = win & a[lo:hi]
        if hit.any():
            det += 1
            leads.append(float(T - l[lo:hi][hit].min()))
    days = y.size / 12.0
    return {"t": t, "episodes": n_ep, "detected_share": det / n_ep if n_ep else float("nan"),
            "lead_median_h": float(np.median(leads)) if leads else float("nan"),
            "lead_iqr_h": [float(np.percentile(leads, 25)), float(np.percentile(leads, 75))] if leads else None,
            "alerts_per_patient_day": float(alert.sum() / days),
            "false_alerts_per_patient_day": float((alert & (y == 0)).sum() / days),
            "ppv": float(y[alert].mean()) if alert.any() else float("nan")}


def onsets(stay_h: np.ndarray, hour_h: np.ndarray, map_h: np.ndarray, first_L: dict) -> dict:
    """stay → first hour ≥ its first landmark with hourly median MAP < 65."""
    ok = np.isfinite(map_h) & (map_h < THR)
    out = {}
    for s, h in zip(stay_h[ok], hour_h[ok]):
        L0 = first_L.get(s)
        if L0 is None or h < L0:
            continue
        if s not in out or h < out[s]:
            out[s] = int(h)
    return out


def sustained(stay_lm: np.ndarray, L: np.ndarray, died: np.ndarray, stay_h: np.ndarray, hour_h: np.ndarray,
              map_h: np.ndarray, min_hours: int = 2, horizon: int = 6) -> np.ndarray:
    """Sustained hypotension: ≥ min_hours hours with median MAP < 65 in [L, L + horizon), or death in the window."""
    low = np.isfinite(map_h) & (map_h < THR)
    by = {}
    for s, h in zip(stay_h[low], hour_h[low]):
        by.setdefault(s, []).append(int(h))
    by = {s: np.sort(np.array(v)) for s, v in by.items()}
    out = np.zeros(L.size, dtype=np.int64)
    for i, (s, l0) in enumerate(zip(stay_lm, L)):
        h = by.get(s)
        if h is not None:
            out[i] = int(np.searchsorted(h, l0 + horizon, "left") - np.searchsorted(h, l0, "left") >= min_hours)
    return np.where(died, 1, out)


def gap_outcome(stay_lm: np.ndarray, L: np.ndarray, died: np.ndarray, stay_h: np.ndarray, hour_h: np.ndarray,
                map_h: np.ndarray, gap: int = 1, horizon: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Hypotension with a blanking hour: any hour with median MAP < 65 in [L + gap, L + horizon), or death in the
    window. Known when that span has an hourly MAP or the patient died in the window."""
    fin = np.isfinite(map_h)
    seen, low = {}, {}
    for s, h, m in zip(stay_h[fin], hour_h[fin], map_h[fin]):
        seen.setdefault(s, []).append(int(h))
        if m < THR:
            low.setdefault(s, []).append(int(h))
    seen = {k: np.sort(np.array(v)) for k, v in seen.items()}
    low = {k: np.sort(np.array(v)) for k, v in low.items()}

    def count(d, s, a, b):
        h = d.get(s)
        return 0 if h is None else int(np.searchsorted(h, b, "left") - np.searchsorted(h, a, "left"))

    y = np.zeros(L.size, dtype=np.int64)
    k = np.zeros(L.size, dtype=bool)
    for i, (s, l0) in enumerate(zip(stay_lm, L)):
        y[i] = int(count(low, s, l0 + gap, l0 + horizon) > 0)
        k[i] = count(seen, s, l0 + gap, l0 + horizon) > 0
    return np.where(died, 1, y), k | died


# =========================================================================== shared summaries
def arm_stats(P: dict, y: np.ndarray, ici: bool = True) -> dict:
    out = {}
    for a, p in P.items():
        d = {"logloss": float(PP.ll_rows(p, y).mean()), "auroc": PP.wauc(p, y, np.ones_like(y)),
             "auprc": PP.wauprc(p, y, np.ones_like(y))}
        if ici:
            c = PP.calib(p, y)
            d.update(ici=c["ici"], citl=c["citl"], slope=c["slope"])
        out[a] = d
    return out


def contrasts(P: dict, y: np.ndarray, subj: np.ndarray, ref: str, others: list[str], seed: int,
              n_boot: int = N_BOOT, n_auc: int = N_BOOT_AUC, weights=None) -> dict:
    """Δ log-loss (arm − ref; 95 % and 90 % CI) and ΔAUROC (arm − ref; 95 % CI) with shared patient weights."""
    y = y.astype(np.float64)
    L = {a: PP.ll_rows(P[a], y) for a in [ref] + others}
    D = {a: [] for a in others}
    A = {a: [] for a in others}
    gen = weights if weights is not None else PP.boot_weights(subj, n_boot, seed)
    for b, w in enumerate(gen):
        sw = w.sum()
        lr = float(w @ L[ref]) / sw
        ar = PP.wauc(P[ref], y, w) if b < n_auc else None
        for a in others:
            D[a].append(float(w @ L[a]) / sw - lr)
            if ar is not None:
                A[a].append(PP.wauc(P[a], y, w) - ar)
    one = np.ones_like(y)
    out = {}
    for a in others:
        out[a] = {"dlogloss": {"est": float(L[a].mean() - L[ref].mean()), "ci95": PP.ci(D[a]), "ci90": PP.ci(D[a], 0.90)},
                  "dauroc": {"est": PP.wauc(P[a], y, one) - PP.wauc(P[ref], y, one), "ci95": PP.ci(A[a])}}
    return out


def cluster_weights(subj: np.ndarray, hosp: np.ndarray, n_boot: int, seed: int):
    """Hospital-then-patient bootstrap weights per row."""
    s = np.asarray(subj).astype(str)
    h = np.asarray(hosp).astype(str)
    _, hinv = np.unique(h, return_inverse=True)
    su, sinv = np.unique(s, return_inverse=True)
    # a patient seen in several hospitals is assigned to the hospital of its first row; hospitals left without
    # patients drop out of the resampling
    first = np.unique(sinv, return_index=True)[1]
    pat_h = hinv[first]
    groups = [np.flatnonzero(pat_h == g) for g in np.unique(pat_h)]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        wh = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups))).astype(np.float64)
        wp = np.zeros(su.size)
        for g in np.flatnonzero(wh):
            idx = groups[g]
            wp[idx] = wh[g] * rng.multinomial(idx.size, np.full(idx.size, 1.0 / idx.size))
        yield wp[sinv]


def estimands(P: dict, y, subj, seed: int, weights=None, n_boot: int = N_BOOT) -> dict:
    """D (90 % CI) against ±0.005, ΔNB transported − local at 0.10/0.20/0.30 against ±0.002, R (90 %) with the count of
    replicates with G_local ≤ 0, ΔAUROC transported − local."""
    y = y.astype(np.float64)
    L = {a: PP.ll_rows(P[a], y) for a in ("transported", "local", "euclidean")}
    B = {"D": [], "R": [], "G": [], "dauc": []}
    nb = {t: [] for t in NB_T}
    gen = weights if weights is not None else PP.boot_weights(subj, n_boot, seed)
    for b, w in enumerate(gen):
        sw = w.sum()
        m = {a: float(w @ L[a]) / sw for a in L}
        g = m["euclidean"] - m["local"]
        B["G"].append(g)
        B["D"].append(m["transported"] - m["local"])
        B["R"].append((m["euclidean"] - m["transported"]) / g if g > 0 else np.nan)
        if b < N_BOOT_AUC:
            B["dauc"].append(PP.wauc(P["transported"], y, w) - PP.wauc(P["local"], y, w))
            for t in NB_T:
                nb[t].append(PP.net_benefit(P["transported"], y, t, w) - PP.net_benefit(P["local"], y, t, w))
    est = {a: float(v.mean()) for a, v in L.items()}
    g = est["euclidean"] - est["local"]
    D90 = PP.ci(B["D"], 0.90)
    G = np.asarray(B["G"])
    out = {"D": {"est": est["transported"] - est["local"], "ci90": D90, "ci95": PP.ci(B["D"]),
                 "equivalent": bool(D90[0] >= -MARGIN and D90[1] <= MARGIN),
                 "noninferior": bool(D90[1] <= MARGIN), "superior": bool(D90[1] < 0)},
           "R": {"est": (est["euclidean"] - est["transported"]) / g if g > 0 else float("nan"),
                 "ci90": PP.ci(B["R"], 0.90), "replicates_G_local_le_0": int((G <= 0).sum()),
                 "replicates_G_local_below_margin": int((G < MARGIN).sum()), "n_boot": int(G.size)},
           "dauroc": {"est": PP.wauc(P["transported"], y, np.ones_like(y)) - PP.wauc(P["local"], y, np.ones_like(y)),
                      "ci95": PP.ci(B["dauc"])},
           "dnb": {}}
    for t in NB_T:
        c, c90 = PP.ci(nb[t]), PP.ci(nb[t], 0.90)
        out["dnb"][f"{t:.2f}"] = {"est": PP.net_benefit(P["transported"], y, t) - PP.net_benefit(P["local"], y, t),
                                  "ci95": c, "ci90": c90, "within_margin": bool(c[0] >= -NB_MARGIN and c[1] <= NB_MARGIN),
                                  "noninferior": bool(c90[0] >= -NB_MARGIN)}
    return out


# =========================================================================== units: data of one evaluation unit
class Unit:
    """Everything one evaluation unit needs: landmarks, folds, the frozen metric of its rotation, the stored
    predictions on every row, and the case-base machinery."""

    def __init__(self, name, lm, fold, metric, E, mcfg, P, comps, seed, dev_sites, st, raw_path, vitals_path, ci):
        self.name, self.lm, self.fold, self.metric, self.E, self.mcfg = name, lm, fold, metric, E, mcfg
        self.P, self.comps, self.seed, self.dev_sites, self.st = P, comps, seed, dev_sites, st
        self.raw_path, self.vitals_path, self.ci = raw_path, vitals_path, ci
        self.y = lm.y.astype(np.int64)
        self.subj = np.asarray(lm.ft.subjects).astype(str)
        ids = np.asarray(lm.ft.ids).astype(str)
        self.stay = np.array([i.split(":")[0] for i in ids])
        self.L = np.array([int(float(i.split(":")[1])) for i in ids])

    @property
    def folds(self):
        return sorted(set(self.fold.tolist()))

    def split(self, f):
        return np.where(self.fold != f)[0], np.where(self.fold == f)[0]

    def draw(self, f: int, tr: np.ndarray, n: int, r: int) -> np.ndarray:
        """Positions in tr of the landmarks of n random patients (the external stage's draw for `hypo`)."""
        s = self.subj[tr]
        uniq = np.unique(s)
        rng = np.random.default_rng([int(self.seed), 7, self.ci, f, n, r])
        pick = rng.choice(uniq, size=min(n, uniq.size), replace=False)
        return np.where(np.isin(s, pick))[0]

    def case_base(self, metric, rows, E, seed):
        from hypo.casebase import CaseBase
        cb = CaseBase(metric, self.lm.take(rows), self.mcfg, E=E[rows])
        cb.fit_alpha(seed)
        return cb

    def knn(self, metric, E, tr, te, seed) -> tuple[np.ndarray, dict]:
        cb = self.case_base(metric, tr, E, seed)
        pr = cb.predict(self.lm.take(te), E=E[te])
        return np.asarray(pr["p"], dtype=np.float64), {"cb": cb, "I": pr["I"], "valid": pr["valid"]}


def _cfg():
    from hypo.cv import settings
    return settings(QUICK)


def _recal_col(C, name: str) -> str:
    """The stored recalibration on 500 local patients (first draw); the largest n stored otherwise (quick tests)."""
    ns = sorted(int(c.split("_n")[1].split("_")[0]) for c in C.columns if c.startswith(f"p_{name}_n") and c.endswith("_r0"))
    n = 500 if 500 in ns else ns[-1]
    return f"p_{name}_n{n}_r0"


def unit_hypo(st, X: str) -> Unit:
    from hypo.config import workdir
    from hypo.external_report import _on, _phase2, _units

    S = st.S
    lm = S.lm(X)
    n = len(lm)
    rows = np.arange(n)
    T = _units(f"transported_{X}_f*.parquet")
    if T is None or T.height != n:
        raise RuntimeError(f"review {X}: transported units do not cover every landmark")
    C = _units(f"comps_{X}_f*.parquet")
    P = {"transported": _on(T, rows, "p"), "local": _on(_units(f"local_{X}_f*.parquet"), rows, "p"),
         "euclidean": _on(_phase2(X, "euclidean"), rows, "p"), "lgbm_transported": _on(C, rows, "p_lightgbm"),
         "lgbm_recal": _on(C, rows, _recal_col(C, "lightgbm")), "last_map_transported": _on(C, rows, "p_last_map"),
         "last_map_recal": _on(C, rows, _recal_col(C, "last_map"))}
    for a, v in P.items():
        if v is None or not np.all(np.isfinite(v)):
            raise RuntimeError(f"review {X}: arm {a} has rows without a prediction")
    comps = {"lightgbm": st.comp(X, "lightgbm"), "last_map": st.comp(X, "last_map")}
    wd = workdir()
    return Unit(f"hypo_{X}", lm, S.fold(X), st.metric(X), st.embed(X, X), st.mcfg, P, comps, st.seed,
                st.development(X), st, wd / "landmarks" / f"{X}.parquet", wd / "extract" / f"{X}_vitals.parquet",
                st.ci(X))


def unit_nwicu(st, e: str, cache: dict) -> Unit:
    import nwicu_transport as NT
    from hypo.cv import outer_folds
    from hypo.frozen import final_dir, load_pickle

    cfg, mcfg, _ = _cfg()
    if "lm" not in cache:
        cache["lm"] = NT.landmarks(cfg)
        cache["fold"] = outer_folds(cache["lm"], int(cfg["cv"]["outer_folds"]), int(cfg["seed"]))
        cache["G"] = NT.gather(NT.work() / "units", len(cache["lm"]))
    lm, fold, G = cache["lm"], cache["fold"], cache["G"]
    metric = load_pickle(final_dir() / f"metric_ext-{e}.pkl")
    lmap = load_pickle(final_dir() / f"comp_ext-{e}_last_map.pkl")
    P = {"transported": G[f"knn_{e}"], "local": G["local"], "euclidean": G["euclidean"],
         "lgbm_transported": G[f"lgbm_{e}"], "lgbm_recal": G[f"lgbmrecal_{e}"]}
    p_lm = np.asarray(lmap(lm), dtype=np.float64)
    P["last_map_transported"] = p_lm
    rec = np.full(len(lm), np.nan)
    for f in sorted(set(fold.tolist())):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        rec[te] = recal("logistic", p_lm[tr], lm.y[tr], p_lm[te])
    P["last_map_recal"] = rec
    comps = {"lightgbm": load_pickle(final_dir() / f"comp_ext-{e}_lightgbm.pkl"), "last_map": lmap}
    w = NT.work()
    return Unit(f"nwicu_{e}", lm, fold, metric, metric.embed(lm), mcfg, P, comps, int(cfg["seed"]),
                [c for c in ROTATIONS if c != e], st, w / "landmarks" / "nwicu.parquet", w / "vitals.parquet",
                10 + ROTATIONS.index(e))


# =========================================================================== block est
def block_est(u: Unit) -> dict:
    out = estimands(u.P, u.y, u.subj, SEED + zlib.crc32(u.name.encode()) % 1000)
    if u.lm.hospital[0] is not None:
        hosp = np.asarray(u.lm.hospital).astype(str)
        out["cluster_hospital"] = estimands(u.P, u.y, u.subj, 0,
                                            weights=cluster_weights(u.subj, hosp, N_BOOT, SEED + 5),
                                            n_boot=N_BOOT)
        out["cluster_hospital"]["n_hospitals"] = int(np.unique(hosp).size)
    return out


# =========================================================================== block A
def block_A(u: Unit, units: Path) -> dict:
    from hypo.external_report import _units

    y = u.y.astype(np.float64)
    p_lgbm = u.P["lgbm_transported"]
    methods = ("intercept", "logistic", "isotonic")
    stored_curve = _units(f"curve_{u.name.split('_')[1]}_f*.parquet") if u.name.startswith("hypo_") else None
    res = {}
    n_pat_min = min(np.unique(u.subj[u.split(f)[0]]).size for f in u.folds)
    grid = [n for n in CURVE_N if n < n_pat_min] + ["all"]
    for n in grid:
        reps = 1 if n == "all" else (3 if n >= 2000 else 5)
        rows = {"perisim": [], **{f"lgbm_{m}": [] for m in methods}}
        for r in range(reps):
            P = {k: np.full(y.size, np.nan) for k in rows}
            for f in u.folds:
                tr, te = u.split(f)
                pos = np.arange(tr.size) if n == "all" else u.draw(f, tr, int(n), r)
                fit = tr[pos]
                for m in methods:
                    P[f"lgbm_{m}"][te] = recal(m, p_lgbm[fit], y[fit], p_lgbm[te])
                if n == "all":
                    P["perisim"][te] = u.P["transported"][te]
                elif stored_curve is not None and f"p_n{n}_r{r}" in stored_curve.columns:
                    sc = stored_curve.filter(stored_curve["row"].is_in(te.tolist()))
                    pos_te = np.searchsorted(te, sc["row"].to_numpy())
                    P["perisim"][te[pos_te]] = sc[f"p_n{n}_r{r}"].to_numpy()
                else:
                    path = units / f"curve_{u.name}_n{n}_r{r}_f{f}.npy"
                    if path.exists():
                        P["perisim"][te] = np.load(path)
                    else:
                        p, _ = u.knn(u.metric, u.E, fit, te, u.seed + 100 * f + r)
                        np.save(path, p)
                        P["perisim"][te] = p
            for k in rows:
                if np.all(np.isfinite(P[k])):
                    rows[k].append(arm_stats({k: P[k]}, y)[k])
        res[str(n)] = {k: {s: {"mean": float(np.mean([d[s] for d in v])), "min": float(np.min([d[s] for d in v])),
                               "max": float(np.max([d[s] for d in v]))} for s in ("logloss", "auroc", "ici")}
                       for k, v in rows.items() if v}
        res[str(n)]["draws"] = reps
        LOG.info("review A %s n=%s done", u.name, n)
    return {"grid": [str(n) for n in grid], "by_n": res, "reading": curve_reading(res, grid)}


def curve_reading(res: dict, grid: list) -> dict:
    out = {}
    for arm in ("perisim", "lgbm_intercept", "lgbm_logistic", "lgbm_isotonic"):
        ok = [n for n in grid if arm in res[str(n)] and res[str(n)][arm]["ici"]["mean"] <= 0.01]
        out[f"first_n_ici_le_0.01_{arm}"] = str(ok[0]) if ok else None
    for m in ("lgbm_logistic", "lgbm_isotonic", "lgbm_intercept"):
        beat = [n for n in grid if m in res[str(n)] and "perisim" in res[str(n)]
                and res[str(n)][m]["logloss"]["mean"] < res[str(n)]["perisim"]["logloss"]["mean"]]
        out[f"first_n_{m}_beats_perisim_logloss"] = str(beat[0]) if beat else None
    return out


# =========================================================================== block B
class _Score:
    """A one-axis 'metric' (the embedding is passed in)."""
    sqrt_w_dims = np.ones(1, dtype=np.float32)

    def embed(self, lm):
        raise RuntimeError("pass E")


def additive_model(u: Unit):
    """Logistic regression (L2, C = 1) on the frozen encoder's shape embedding, fitted on the development pool's
    learning sample (the frozen settings)."""
    from sklearn.linear_model import LogisticRegression
    from hypo.data import concat, learning_sample

    st = u.st
    dev = concat([st.S.lm(c) for c in u.dev_sites])
    rows = learning_sample(dev, int(u.mcfg["learn"]["per_patient"]), int(u.mcfg["learn"]["per_site"]), SEED)
    d = dev.take(rows)
    X = u.metric.embed(d)
    m = LogisticRegression(C=1.0, max_iter=3000).fit(X, d.y)
    return np.asarray(m.predict_proba(u.E)[:, 1], dtype=np.float64)


def block_B(u: Unit, units: Path) -> dict:
    from hypo.metric import Metric

    y = u.y.astype(np.float64)
    w = np.asarray(u.metric.w, dtype=np.float64)
    m_unif = Metric(u.metric.enc, np.where(w > 0, 1.0, 0.0))
    E1 = logit(u.P["lgbm_transported"])[:, None].astype(np.float32)
    p_add = additive_model(u)
    P = {k: np.full(y.size, np.nan) for k in ("shapes_uniform", "score_1d", "additive_n500", "additive_all",
                                                "gate_transported")}
    for f in u.folds:
        tr, te = u.split(f)
        seed = u.seed + 100 * f
        path = units / f"B_{u.name}_f{f}.npz"
        if path.exists():
            z = np.load(path)
            for k in ("shapes_uniform", "score_1d"):
                P[k][te] = z[k]
        else:
            P["shapes_uniform"][te], _ = u.knn(m_unif, u.E, tr, te, seed)
            P["score_1d"][te], _ = u.knn(_Score(), E1, tr, te, seed)
            np.savez(path, shapes_uniform=P["shapes_uniform"][te], score_1d=P["score_1d"][te])
        pos = u.draw(f, tr, 500, 0)
        P["additive_n500"][te] = recal("logistic", p_add[tr[pos]], y[tr[pos]], p_add[te])
        P["additive_all"][te] = recal("logistic", p_add[tr], y[tr], p_add[te])
        if f == u.folds[0]:                                     # gate: the stored transported kNN is rebuilt
            P["gate_transported"][te], _ = u.knn(u.metric, u.E, tr, te, seed)
            gap = float(np.max(np.abs(P["gate_transported"][te] - u.P["transported"][te])))
            if gap > GATE_TOL:
                raise RuntimeError(f"review B {u.name}: the transported kNN is not rebuilt (max |Δp| = {gap})")
    del P["gate_transported"]
    P["additive_transported"] = p_add
    arms = {"raw_euclidean": u.P["euclidean"], "shapes_uniform": P["shapes_uniform"],
            "shapes_nca": u.P["transported"], "score_1d": P["score_1d"],
            "additive_transported": p_add, "additive_n500": P["additive_n500"], "additive_all": P["additive_all"],
            "persistence_transported": u.P["last_map_transported"], "persistence_recal": u.P["last_map_recal"],
            "lgbm_recal": u.P["lgbm_recal"], "local": u.P["local"]}
    stats = arm_stats(arms, y)
    con = contrasts(arms, y, u.subj, "shapes_nca", [a for a in arms if a != "shapes_nca"], SEED + 11)
    gl = stats["raw_euclidean"]["logloss"] - stats["shapes_nca"]["logloss"]
    share = {"shapes": (stats["raw_euclidean"]["logloss"] - stats["shapes_uniform"]["logloss"]) / gl if gl > 0 else None,
             "weights": (stats["shapes_uniform"]["logloss"] - stats["shapes_nca"]["logloss"]) / gl if gl > 0 else None}
    s1 = con["score_1d"]["dlogloss"]
    return {"arms": stats, "vs_shapes_nca": con, "gain_share": share,
            "score_1d_not_worse": bool(s1["ci90"][1] <= MARGIN) if s1 else None}


# =========================================================================== block C
def resample_metric(st, dev_sites, b: int, mcfg, space, threads: int):
    from hypo.data import concat, learning_sample
    from hypo.metric import fit_metric

    pool = concat([st.S.lm(c) for c in dev_sites])
    s = np.asarray(pool.ft.subjects).astype(str)
    uniq, inv = np.unique(s, return_inverse=True)
    rng = np.random.default_rng([SEED, 31, b])
    pick = rng.integers(0, uniq.size, uniq.size)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(uniq.size))
    ends = np.searchsorted(inv[order], np.arange(uniq.size), "right")
    idx, new_s = [], []
    copy = {}
    for i in pick:
        c = copy.get(i, 0)
        copy[i] = c + 1
        rr = order[starts[i]:ends[i]]
        idx.append(rr)
        new_s.append(np.full(rr.size, f"{uniq[i]}#{c}", dtype=object))
    idx = np.concatenate(idx)
    sub = pool.take(idx)
    sub.ft.subjects = np.concatenate(new_s)
    rows = learning_sample(sub, int(mcfg["learn"]["per_patient"]), int(mcfg["learn"]["per_site"]), SEED + b)
    return fit_metric(sub.take(rows), mcfg, space, seed=SEED + 1000 + b, threads=threads)


def block_C(st, units: Path, n_src: int, threads: int, targets: dict) -> dict:
    """targets: rotation → list of Units that use that rotation's metric (the hypo target and NWICU)."""
    from hypo.frozen import dump, load_pickle

    _, mcfg, space = _cfg()
    out = {}
    for X, us in targets.items():
        dev = [c for c in ROTATIONS if c != X]
        for b in range(n_src):
            mp = units / f"C_metric_{X}_b{b}.pkl"
            if not mp.exists():
                t = time.perf_counter()
                dump(resample_metric(st, dev, b, mcfg, space, threads), mp)
                LOG.info("review C metric %s b=%d: %.0f s", X, b, time.perf_counter() - t)
            m = load_pickle(mp)
            for u in us:
                path = units / f"C_{u.name}_b{b}.npy"
                if path.exists():
                    continue
                E = m.embed(u.lm)
                p = np.full(len(u.lm), np.nan)
                for f in u.folds:
                    tr, te = u.split(f)
                    p[te], _ = u.knn(m, E, tr, te, u.seed + 100 * f)
                np.save(path, p)
        for u in us:
            y = u.y.astype(np.float64)
            Ls = {a: float(PP.ll_rows(u.P[a], y).mean()) for a in ("local", "euclidean", "transported")}
            Db, Rb, Ab = [], [], []
            for b in range(n_src):
                p = np.load(units / f"C_{u.name}_b{b}.npy")
                lb = float(PP.ll_rows(p, y).mean())
                Db.append(lb - Ls["local"])
                g = Ls["euclidean"] - Ls["local"]
                Rb.append((Ls["euclidean"] - lb) / g if g > 0 else np.nan)
                Ab.append(PP.wauc(p, y, np.ones_like(y)))
            est = estimands(u.P, u.y, u.subj, SEED + 3, n_boot=1000)
            var_t = ((est["D"]["ci90"][1] - est["D"]["ci90"][0]) / (2 * 1.645)) ** 2
            sd_s = float(np.std(Db, ddof=1)) if len(Db) > 1 else float("nan")
            half = 1.645 * np.sqrt(var_t + sd_s ** 2)
            out[u.name] = {"n_src": n_src, "D_resamples": {"mean": float(np.mean(Db)), "sd": sd_s,
                                                           "min": float(np.min(Db)), "max": float(np.max(Db))},
                           "R_resamples": {"mean": float(np.nanmean(Rb)), "sd": float(np.nanstd(Rb, ddof=1)),
                                           "min": float(np.nanmin(Rb)), "max": float(np.nanmax(Rb))},
                           "auroc_resamples": {"mean": float(np.mean(Ab)), "sd": float(np.std(Ab, ddof=1)),
                                               "min": float(np.min(Ab)), "max": float(np.max(Ab))},
                           "D_frozen": est["D"]["est"],
                           "D_total_ci90": [est["D"]["est"] - half, est["D"]["est"] + half],
                           "equivalent_total": bool(est["D"]["est"] - half >= -MARGIN and est["D"]["est"] + half <= MARGIN)}
    return out


# =========================================================================== block D
def _raw(u: Unit):
    import polars as pl
    df = pl.read_parquet(u.raw_path)
    return df


def hourly_map(u: Unit):
    import polars as pl
    from hypo.landmarks import hourly
    H = hourly(pl.read_parquet(u.vitals_path))
    return (H["stay_id"].cast(pl.Utf8).to_numpy(), H["hour"].to_numpy().astype(np.int64),
            H["map"].cast(pl.Float64).to_numpy())


def block_D(u: Unit, units: Path, Hcache: dict) -> dict:
    import polars as pl
    from hypo.data import load as load_lm
    from posthoc_cosine import neutral_metric

    cfg, _, _ = _cfg()
    y = u.y.astype(np.float64)
    out = {}
    # (1) windows with no MAP hour, and partly observed windows
    df = _raw(u)
    site = u.raw_path.stem
    norm = df.filter(pl.col("normotensive"))
    unk = norm.filter(~pl.col("known"))
    inputs = [c for k in ("continuous", "binary") for c in cfg["model"]["inputs"][k] if c in norm.columns]
    known = norm.filter(pl.col("known"))
    desc = {}
    for c in inputs:
        a, b = known[c].cast(pl.Float64), unk[c].cast(pl.Float64)
        desc[c] = {"included_mean": a.mean(), "dropped_mean": b.mean() if unk.height >= 10 else None,
                   "included_missing": float(a.is_null().mean()),
                   "dropped_missing": float(b.is_null().mean()) if unk.height >= 10 else None}
    o1 = {"normotensive_landmarks": int(norm.height), "dropped_no_map_hour": int(unk.height),
          "dropped_share": float(unk.height / max(norm.height, 1)), "inputs": desc}
    if unk.height >= 10:
        lu = load_lm(site, cfg, frame=unk.with_columns(pl.lit(True).alias("known")))
        Eu = u.metric.embed(lu)
        cb = u.case_base(u.metric, np.arange(len(u.lm)), u.E, u.seed)
        pu = np.asarray(cb.predict(lu, E=Eu)["p"], dtype=np.float64)
        base = arm_stats({"t": u.P["transported"]}, y, ici=False)["t"]
        for yv in (0, 1):
            pp = np.r_[u.P["transported"], pu]
            yy = np.r_[y, np.full(pu.size, float(yv))]
            s = arm_stats({"t": pp}, yy, ici=False)["t"]
            o1[f"all_dropped_y{yv}"] = {"logloss": s["logloss"], "auroc": s["auroc"],
                                        "dlogloss": s["logloss"] - base["logloss"], "dauroc": s["auroc"] - base["auroc"]}
        o1["dropped_mean_p"] = float(pu.mean())
    ids = np.char.add(np.char.add(known["stay_id"].cast(pl.Utf8).to_numpy().astype(str), ":"),
                      known["L"].cast(pl.Utf8).to_numpy().astype(str))
    if {"win_hours", "los_h"} <= set(known.columns):
        Lk = known["L"].to_numpy().astype(np.float64)
        poss = np.minimum(Lk + 6, np.ceil(known["los_h"].to_numpy().astype(np.float64))) - Lk
        full = known["win_hours"].fill_null(0).to_numpy() >= poss
        fmap = dict(zip(ids, full))
        lm_ids = np.char.add(np.char.add(u.stay, ":"), u.L.astype(str))
        fu = np.array([bool(fmap.get(i, False)) for i in lm_ids])
        o1["fully_observed_share"] = float(fu.mean())
        if fu.sum() > 100:
            Pf = {a: u.P[a][fu] for a in ("transported", "local", "euclidean")}
            o1["fully_observed"] = estimands(Pf, u.y[fu], u.subj[fu], SEED + 21, n_boot=1000)
    out["windows"] = o1
    # (2) sustained hypotension
    if u.raw_path not in Hcache:
        Hcache[u.raw_path] = hourly_map(u)
    sh, hh, mh = Hcache[u.raw_path]
    died = u.lm.frame["died_in_window"].cast(pl.Boolean).to_numpy() if "died_in_window" in u.lm.frame.columns \
        else np.zeros(len(u.lm), bool)
    y2 = sustained(u.stay, u.L, died, sh, hh, mh)
    P2 = {k: np.full(y.size, np.nan) for k in ("transported", "euclidean", "lgbm_n500", "lgbm_all")}
    pl_ = u.P["lgbm_transported"]
    for f in u.folds:
        tr, te = u.split(f)
        path = units / f"D2_{u.name}_f{f}.npz"
        if path.exists():
            z = np.load(path)
            P2["transported"][te], P2["euclidean"][te] = z["t"], z["e"]
        else:
            _, info = u.knn(u.metric, u.E, tr, te, u.seed + 100 * f)
            P2["transported"][te] = info["cb"].vote(info["I"], info["valid"], y2[tr], 2)[:, 1]
            nm = neutral_metric(u.lm.take(tr))
            En = nm.embed(u.lm)
            _, info = u.knn(nm, En, tr, te, u.seed + 100 * f)
            P2["euclidean"][te] = info["cb"].vote(info["I"], info["valid"], y2[tr], 2)[:, 1]
            np.savez(path, t=P2["transported"][te], e=P2["euclidean"][te])
        pos = u.draw(f, tr, 500, 0)
        P2["lgbm_n500"][te] = recal("logistic", pl_[tr[pos]], y2[tr[pos]], pl_[te])
        P2["lgbm_all"][te] = recal("logistic", pl_[tr], y2[tr], pl_[te])
    y2f = y2.astype(np.float64)
    out["sustained"] = {"prevalence": float(y2.mean()), "prevalence_any": float(y.mean()),
                        "arms": arm_stats(P2, y2f),
                        "vs_transported": contrasts(P2, y2f, u.subj, "transported",
                                                    ["euclidean", "lgbm_n500", "lgbm_all"], SEED + 23, n_boot=1000)}
    # (3) strata
    v = np.asarray(u.lm.ft.num.get("vaso_on", np.full(len(u.lm), np.nan)), dtype=np.float64)
    vaso_var = "vaso_on"
    if not np.isfinite(v).any():
        v, vaso_var = np.asarray(u.lm.ft.num["vaso_24h"], dtype=np.float64), "vaso_24h"
    art = u.lm.frame["art_line"].cast(pl.Float64).fill_null(0).to_numpy() if "art_line" in u.lm.frame.columns else None
    strata = {f"{vaso_var}=1": np.nan_to_num(v) > 0.5, f"{vaso_var}=0": ~(np.nan_to_num(v) > 0.5)}
    if art is not None:
        strata.update({"art_line=1": art > 0.5, "art_line=0": art <= 0.5})
    so = {}
    for k, msk in strata.items():
        if msk.sum() < 200 or np.unique(y[msk]).size < 2:
            so[k] = {"n": int(msk.sum()), "note": "too few"}
            continue
        Pm = {a: u.P[a][msk] for a in ("transported", "local", "euclidean", "lgbm_recal")}
        so[k] = {"n": int(msk.sum()), "patients": int(np.unique(u.subj[msk]).size), "prevalence": float(y[msk].mean()),
                 "arms": arm_stats(Pm, y[msk]),
                 "vs_local": contrasts(Pm, y[msk], u.subj[msk], "local", ["transported", "lgbm_recal"], SEED + 25,
                                       n_boot=1000)}
    out["strata"] = so
    # (4) ties (fold 0)
    f0 = u.folds[0]
    tr, te = u.split(f0)
    rng = np.random.default_rng(SEED)
    q = np.sort(rng.choice(te, size=min(20000, te.size), replace=False))
    cb = u.case_base(u.metric, tr, u.E, u.seed + 100 * f0)
    k = int(u.mcfg["k"])
    D, I, valid = cb.search(u.E[q], u.lm.ft.subjects[q], k=k + 1)
    ok = valid[:, k]
    tie_k = (D[:, k - 1] == D[:, k]) & ok
    tie_any = np.any(D[:, :k - 1] == D[:, 1:k], axis=1)
    out["ties"] = {"queries": int(q.size), "share_kth_tied_with_next": float(tie_k.mean()),
                   "share_any_tie_in_top_k": float(tie_any.mean()),
                   "rule": "exact search (FAISS IndexFlatL2); among equal distances the index's return order, then "
                           "the patient's closest landmark kept (stable sort): deterministic for a given case base"}
    # (5) MAP weights
    wt = u.metric.weights_table()
    out["map_weight_share"] = {n: v for n, v in wt.items() if n.startswith("map_")}
    out["map_weight_share_total"] = float(sum(out["map_weight_share"].values()))
    return out


# =========================================================================== block E
# =========================================================================== block H
def _local_metric(u: Unit, f: int, tr: np.ndarray, units: Path, threads: int):
    """The fold's local metric, refitted as the external stage / NWICU script fitted it (same rows, seed and space),
    cached as a pickle (the stored units keep only its predictions)."""
    import pickle
    from hypo.data import learning_sample
    from hypo.metric import fit_metric

    path = units / f"H_local_{u.raw_path.stem}_f{f}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    _, mcfg, space = _cfg()
    seed = u.seed + 100 * f
    pool = u.lm.take(tr)
    rows = learning_sample(pool, int(mcfg["learn"]["per_patient"]), int(mcfg["learn"]["per_site"]), seed)
    m = fit_metric(pool.take(rows), mcfg, space, seed=seed, threads=threads)
    with open(path, "wb") as fh:
        pickle.dump(m, fh)
    return m


def block_H(u: Unit, units: Path, Hcache: dict, threads: int) -> dict:
    """The autoregressive outcome: (1) a blanking hour, [L + 1, L + 6); (2) the stratum far from the threshold, last
    MAP ≥ 75 mmHg; (3) every arm's gain over persistence (the last-MAP model recalibrated on 500 local patients)."""
    import polars as pl
    from hypo.casebase import CaseBase
    from posthoc_cosine import neutral_metric

    out = {}
    y = u.y.astype(np.float64)
    # (1) blanking hour
    if u.raw_path not in Hcache:
        Hcache[u.raw_path] = hourly_map(u)
    sh, hh, mh = Hcache[u.raw_path]
    died = u.lm.frame["died_in_window"].cast(pl.Boolean).to_numpy() if "died_in_window" in u.lm.frame.columns \
        else np.zeros(len(u.lm), bool)
    yg, kg = gap_outcome(u.stay, u.L, died, sh, hh, mh)
    arms = ("transported", "local", "euclidean", "lgbm_n500", "persistence_n500")
    Pg = {a: np.full(y.size, np.nan) for a in arms}
    gate = None
    for f in u.folds:
        tr, te = u.split(f)
        trk, tek = tr[kg[tr]], te[kg[te]]
        seed = u.seed + 100 * f
        path = units / f"H1_{u.name}_f{f}.npz"
        if path.exists():
            z = np.load(path)
            for a in ("transported", "local", "euclidean"):
                Pg[a][tek] = z[a]
        else:
            m_loc = _local_metric(u, f, tr, units, threads)
            E_loc = m_loc.embed(u.lm)
            if f == u.folds[0]:                  # gate: the refitted local metric rebuilds the stored local predictions
                cb = CaseBase(m_loc, u.lm.take(tr), u.mcfg, E=E_loc[tr])
                cb.fit_alpha(seed)
                p0 = np.asarray(cb.predict(u.lm.take(te), E=E_loc[te])["p"], dtype=np.float64)
                gate = float(np.max(np.abs(p0 - u.P["local"][te])))
            nm = neutral_metric(u.lm.take(tr))
            for a, (m, E) in {"transported": (u.metric, u.E), "local": (m_loc, E_loc),
                              "euclidean": (nm, nm.embed(u.lm))}.items():
                _, info = u.knn(m, E, trk, tek, seed)
                Pg[a][tek] = info["cb"].vote(info["I"], info["valid"], yg[trk], 2)[:, 1]
            np.savez(path, **{a: Pg[a][tek] for a in ("transported", "local", "euclidean")})
        pos = u.draw(f, trk, 500, 0)
        Pg["lgbm_n500"][tek] = recal("logistic", u.P["lgbm_transported"][trk[pos]], yg[trk[pos]],
                                     u.P["lgbm_transported"][tek])
        Pg["persistence_n500"][tek] = recal("logistic", u.P["last_map_transported"][trk[pos]], yg[trk[pos]],
                                            u.P["last_map_transported"][tek])
    m = kg
    ygf = yg[m].astype(np.float64)
    Pm = {a: v[m] for a, v in Pg.items()}
    out["blanking_hour"] = {
        "known_share": float(kg.mean()), "prevalence": float(ygf.mean()), "prevalence_primary": float(y.mean()),
        "local_rebuild_gap": gate, "arms": arm_stats(Pm, ygf),
        "estimands": estimands(Pm, ygf, u.subj[m], SEED + 31, n_boot=1000),
        "vs_persistence": contrasts(Pm, ygf, u.subj[m], "persistence_n500",
                                    ["transported", "local", "lgbm_n500"], SEED + 32, n_boot=1000)}
    # (2) far from the threshold, and (3) gain over persistence, on the primary outcome
    ml = np.asarray(u.lm.ft.num["map_last"], dtype=np.float64)
    P = {"transported": u.P["transported"], "local": u.P["local"], "euclidean": u.P["euclidean"],
         "lgbm_recal": u.P["lgbm_recal"], "persistence_recal": u.P["last_map_recal"]}
    others = ["transported", "local", "lgbm_recal"]
    out["vs_persistence"] = contrasts(P, y, u.subj, "persistence_recal", others, SEED + 33, n_boot=1000)
    out["arms_all"] = arm_stats(P, y, ici=False)
    for name, s in {"map_last_ge_75": np.nan_to_num(ml, nan=0.0) >= 75.0,
                    "map_last_65_75": (ml >= 65.0) & (ml < 75.0)}.items():
        n_pat = int(np.unique(u.subj[s]).size)
        o = {"landmarks": int(s.sum()), "patients": n_pat, "share": float(s.mean()), "prevalence": float(y[s].mean())}
        if n_pat >= 100 and 10 <= y[s].sum() <= s.sum() - 10:
            Ps = {a: v[s] for a, v in P.items()}
            o.update(arms=arm_stats(Ps, y[s], ici=False),
                     estimands=estimands(Ps, y[s], u.subj[s], SEED + 34, n_boot=1000),
                     vs_persistence=contrasts(Ps, y[s], u.subj[s], "persistence_recal", others, SEED + 35, n_boot=1000))
        out[name] = o
    return out


def approx_embedding(metric, ft, nodes: int = 101) -> tuple[np.ndarray, dict]:
    enc = metric.enc
    E = np.array(enc.transform(ft), dtype=np.float32)
    info = {"rank_axes": 0, "approximated": 0}
    for j, b in enumerate(enc.blocks):
        if b.kind not in ("continuous", "ordinal") or b.shape is not None:
            continue
        info["rank_axes"] += 1
        srt = np.asarray(b.params["sorted"], dtype=np.float64)
        if np.unique(srt).size <= nodes:
            continue
        info["approximated"] += 1
        cols = np.flatnonzero(enc.dim_feature == j)
        E[:, cols[0]] = (rank_nodes_approx(ft.num[b.source], srt, float(b.params["median"]), nodes)
                         / float(b.scale)).astype(np.float32)
    return E, info


def block_E(u: Unit, units: Path, Hcache: dict) -> dict:
    from perisim.features.table import FeatureTable
    from posthoc_cosine import neutral_metric

    y = u.y.astype(np.float64)
    out = {}
    # (1) artefact without exact ranks
    Ea, info = approx_embedding(u.metric, u.lm.ft)
    gap0 = float(np.max(np.abs(np.asarray(u.metric.enc.transform(u.lm.ft), dtype=np.float32) - u.E)))
    pa = np.full(y.size, np.nan)
    path = units / f"E1_{u.name}.npy"
    if path.exists():
        pa = np.load(path)
    else:
        for f in u.folds:
            tr, te = u.split(f)
            pa[te], _ = u.knn(u.metric, Ea, tr, te, u.seed + 100 * f)
        np.save(path, pa)
    d = np.abs(pa - u.P["transported"])
    s0 = arm_stats({"exact": u.P["transported"], "quantiles": pa}, y)
    out["artefact_quantiles"] = {**info, "embedding_rebuild_gap": gap0, "max_abs_dp": float(d.max()),
                                 "mean_abs_dp": float(d.mean()), "share_dp_le_1e-9": float((d <= 1e-9).mean()),
                                 "arms": s0, "dauroc": s0["quantiles"]["auroc"] - s0["exact"]["auroc"],
                                 "dlogloss": s0["quantiles"]["logloss"] - s0["exact"]["logloss"]}
    # (2) episodes
    if u.raw_path not in Hcache:
        Hcache[u.raw_path] = hourly_map(u)
    sh, hh, mh = Hcache[u.raw_path]
    first_L = {}
    for s, l in zip(u.stay, u.L):
        if s not in first_L or l < first_L[s]:
            first_L[s] = int(l)
    last_L = {}
    for s, l in zip(u.stay, u.L):
        last_L[s] = max(last_L.get(s, l), int(l))
    on = {s: T for s, T in onsets(sh, hh, mh, first_L).items() if T <= last_L[s] + 5}
    ep = {}
    for arm in ("transported", "local", "lgbm_recal"):
        ep[arm] = [episodes(u.stay, u.L, u.y, u.P[arm], on, t) for t in NB_T]
    out["episodes"] = {"stays": int(len(first_L)), "stays_with_onset": int(len(on)), "by_arm": ep}
    # (3) neighbour stability (fold 0)
    f0 = u.folds[0]
    tr, te = u.split(f0)
    rng = np.random.default_rng(SEED + 7)
    q = np.sort(rng.choice(te, size=min(5000, te.size), replace=False))
    ftq = u.lm.ft.take(q)
    iqr = {}
    for s in u.lm.ft.specs:
        if s.type == "continuous":
            x = u.lm.ft.num[s.name][tr]
            x = x[np.isfinite(x)]
            iqr[s.name] = float(np.subtract(*np.percentile(x, [75, 25]))) if x.size else 0.0
    res = {}
    for label, metric, E_all in (("transported", u.metric, u.E), ("euclidean", None, None)):
        if metric is None:
            metric = neutral_metric(u.lm.take(tr))
            E_all = metric.embed(u.lm)
        cb = u.case_base(metric, tr, E_all, u.seed + 100 * f0)
        _, I0, v0 = cb.search(E_all[q], ftq.subjects)
        P0 = np.where(v0, cb.code[I0], -1)
        p0 = cb.probs(I0, v0, cb.k)[:, 1]
        J, dp = [], []
        for r in range(10):
            num = {}
            for k_, x in ftq.num.items():
                if k_ in iqr:
                    noise = rng.uniform(-0.05, 0.05, x.size) * iqr[k_]
                    num[k_] = np.where(np.isfinite(x), x + noise, x)
                else:
                    num[k_] = x
            ftp = FeatureTable(ftq.ids, ftq.subjects, ftq.specs, num, ftq.cat)
            Ep = np.asarray(metric.enc.transform(ftp), dtype=np.float32)
            _, I1, v1 = cb.search(Ep, ftq.subjects)
            J.append(jaccard_rows(P0, np.where(v1, cb.code[I1], -1)))
            dp.append(np.abs(cb.probs(I1, v1, cb.k)[:, 1] - p0))
        J, dp = np.concatenate(J), np.concatenate(dp)
        res[label] = {"jaccard_mean": float(J.mean()), "jaccard_median": float(np.median(J)),
                      "jaccard_p10": float(np.percentile(J, 10)), "abs_dp_mean": float(dp.mean()),
                      "abs_dp_p90": float(np.percentile(dp, 90))}
    out["neighbour_stability"] = {"queries": int(q.size), "perturbations": 10, "noise": "±5 % of the IQR", **res}
    return out


# =========================================================================== block G
def block_G(distill_dir: Path | None) -> dict:
    if distill_dir is None or not distill_dir.exists():
        return {"note": "distillation folder not given or absent"}
    mins = {}

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "min_patients_cell" and isinstance(v, (int, float)):
                    mins[path + "/" + k] = min(mins.get(path + "/" + k, np.inf), float(v))
                walk(v, path + "/" + str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x[:100000]):
                walk(v, path)

    files = sorted(distill_dir.rglob("aggregates*.json"))
    for f in files:
        walk(json.loads(f.read_text(encoding="utf-8")), f.stem)
    allmin = min(mins.values()) if mins else None
    return {"files": [f.name for f in files], "min_patients_any_cell": allmin,
            "all_cells_ge_50": bool(allmin is not None and allmin >= 50)}


# =========================================================================== report
def _f(x, s="+.4f"):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, s)


def render(res: dict) -> str:
    L = ["# Review analyses for the article (post hoc, exploratory)", "",
         "Plan: docs/DECISIONS.md, 2026-11-15 (written before they ran, not registered). v0.6, arm value. Log-loss in nats; "
         "net benefit per prediction (landmark). Aggregates only.", ""]
    E = res.get("est", {})
    if E:
        L += ["## Primary estimands", "", "| unit | D (90 % CI) | equivalent ±0.005 | ΔAUROC tr − local (95 % CI) | "
              "ΔNB 0.10 | ΔNB 0.20 | ΔNB 0.30 | R (90 % CI) | replicates G_local ≤ 0 |", "|---|---|---|---|---|---|---|---|---|"]
        for u, e in E.items():
            nb = e["dnb"]
            L.append(f"| {u} | {_f(e['D']['est'])} ({_f(e['D']['ci90'][0])} to {_f(e['D']['ci90'][1])}) | "
                     f"{'yes' if e['D']['equivalent'] else 'no'} | {_f(e['dauroc']['est'])} ({_f(e['dauroc']['ci95'][0])} to "
                     f"{_f(e['dauroc']['ci95'][1])}) | " + " | ".join(
                         f"{_f(nb[t]['est'])} ({_f(nb[t]['ci95'][0])} to {_f(nb[t]['ci95'][1])})" for t in nb) +
                     f" | {_f(e['R']['est'], '.2f')} ({_f(e['R']['ci90'][0], '.2f')}–{_f(e['R']['ci90'][1], '.2f')}) | "
                     f"{e['R']['replicates_G_local_le_0']} of {e['R']['n_boot']} |")
            if "cluster_hospital" in e:
                c = e["cluster_hospital"]
                L.append(f"| {u}, hospital cluster bootstrap ({c['n_hospitals']} hospitals) | {_f(c['D']['est'])} "
                         f"({_f(c['D']['ci90'][0])} to {_f(c['D']['ci90'][1])}) | {'yes' if c['D']['equivalent'] else 'no'} | "
                         f"{_f(c['dauroc']['est'])} ({_f(c['dauroc']['ci95'][0])} to {_f(c['dauroc']['ci95'][1])}) | | | | "
                         f"{_f(c['R']['est'], '.2f')} ({_f(c['R']['ci90'][0], '.2f')}–{_f(c['R']['ci90'][1], '.2f')}) | |")
        L.append("")
    if res.get("A"):
        L += ["## A. Learning curves", ""]
        for u, a in res["A"].items():
            L += [f"**{u}** — reading: " + "; ".join(f"{k}: {v}" for k, v in a["reading"].items()), "",
                  "| n patients | arm | log-loss | AUROC | ICI |", "|---|---|---|---|---|"]
            for n in a["grid"]:
                for arm, s in a["by_n"][n].items():
                    if arm == "draws":
                        continue
                    L.append(f"| {n} | {arm} | {s['logloss']['mean']:.4f} | {s['auroc']['mean']:.3f} | {s['ici']['mean']:.4f} |")
            L.append("")
    if res.get("B"):
        L += ["## B. Comparators of the geometry", ""]
        for u, b in res["B"].items():
            L += [f"**{u}** — share of the gain: shapes {_f(b['gain_share']['shapes'], '.2f')}, weights "
                  f"{_f(b['gain_share']['weights'], '.2f')}; 1-D score not worse than shapes + NCA: {b['score_1d_not_worse']}",
                  "", "| arm | log-loss | AUROC | AUPRC | ICI | Δ log-loss vs shapes + NCA (90 % CI) | ΔAUROC vs shapes + NCA (95 % CI) |",
                  "|---|---|---|---|---|---|---|"]
            for a, s in b["arms"].items():
                c = b["vs_shapes_nca"].get(a)
                cc = (f"{_f(c['dlogloss']['est'])} ({_f(c['dlogloss']['ci90'][0])} to {_f(c['dlogloss']['ci90'][1])}) | "
                      f"{_f(c['dauroc']['est'])} ({_f(c['dauroc']['ci95'][0])} to {_f(c['dauroc']['ci95'][1])})") if c else "ref | ref"
                L.append(f"| {a} | {s['logloss']:.4f} | {s['auroc']:.3f} | {s['auprc']:.3f} | {s['ici']:.4f} | {cc} |")
            L.append("")
    if res.get("C"):
        L += ["## C. Uncertainty of learning the metric", "", "| unit | resamples | D frozen | D resamples mean (SD) | "
              "D total 90 % CI | equivalent | R resamples mean (range) | AUROC resamples mean (SD) |",
              "|---|---|---|---|---|---|---|---|"]
        for u, c in res["C"].items():
            L.append(f"| {u} | {c['n_src']} | {_f(c['D_frozen'])} | {_f(c['D_resamples']['mean'])} ({c['D_resamples']['sd']:.4f}) | "
                     f"{_f(c['D_total_ci90'][0])} to {_f(c['D_total_ci90'][1])} | {'yes' if c['equivalent_total'] else 'no'} | "
                     f"{c['R_resamples']['mean']:.2f} ({c['R_resamples']['min']:.2f}–{c['R_resamples']['max']:.2f}) | "
                     f"{c['auroc_resamples']['mean']:.3f} ({c['auroc_resamples']['sd']:.4f}) |")
        L.append("")
    if res.get("D"):
        L += ["## D. Cohort", ""]
        for u, d in res["D"].items():
            w = d["windows"]
            L.append(f"**{u}** — dropped (no MAP hour in the window): {w['dropped_no_map_hour']:,} of "
                     f"{w['normotensive_landmarks']:,} ({100 * w['dropped_share']:.1f} %); fully observed windows "
                     f"{100 * w.get('fully_observed_share', float('nan')):.1f} %.")
            for yv in (0, 1):
                if f"all_dropped_y{yv}" in w:
                    x = w[f"all_dropped_y{yv}"]
                    L.append(f"- all dropped y = {yv}: Δ log-loss {_f(x['dlogloss'])}, ΔAUROC {_f(x['dauroc'])}")
            if "fully_observed" in w:
                fo = w["fully_observed"]
                L.append(f"- fully observed windows only: D {_f(fo['D']['est'])} ({_f(fo['D']['ci90'][0])} to "
                         f"{_f(fo['D']['ci90'][1])}), ΔAUROC {_f(fo['dauroc']['est'])}")
            s = d["sustained"]
            L.append(f"- sustained hypotension (≥ 2 h): prevalence {100 * s['prevalence']:.1f} % (any: "
                     f"{100 * s['prevalence_any']:.1f} %); AUROC " + ", ".join(
                         f"{a} {v['auroc']:.3f}" for a, v in s["arms"].items()))
            for k, v in d["strata"].items():
                if "arms" in v:
                    c = v["vs_local"]["transported"]
                    L.append(f"- stratum {k} (n = {v['n']:,}, prevalence {100 * v['prevalence']:.1f} %): D tr − local "
                             f"{_f(c['dlogloss']['est'])} ({_f(c['dlogloss']['ci90'][0])} to {_f(c['dlogloss']['ci90'][1])}), "
                             f"ΔAUROC {_f(c['dauroc']['est'])}; ICI transported {v['arms']['transported']['ici']:.4f}")
            t = d["ties"]
            L.append(f"- ties (fold 0, {t['queries']:,} queries): k-th tied with the next {100 * t['share_kth_tied_with_next']:.1f} %, "
                     f"any tie in the top k {100 * t['share_any_tie_in_top_k']:.1f} %")
            L.append(f"- MAP inputs' weight share: {d['map_weight_share_total']:.2f}")
            L.append("")
    if res.get("E"):
        L += ["## E. Deployment and use", ""]
        for u, e in res["E"].items():
            a = e["artefact_quantiles"]
            L.append(f"**{u}** — artefact without exact ranks: {a['approximated']} of {a['rank_axes']} rank axes "
                     f"approximated; max |Δp| {a['max_abs_dp']:.2e}, ΔAUROC {_f(a['dauroc'])}, Δ log-loss {_f(a['dlogloss'])}.")
            for arm, rows in e["episodes"]["by_arm"].items():
                L.append(f"- episodes ({e['episodes']['stays_with_onset']:,} first episodes), {arm}: " + "; ".join(
                    f"t {r['t']:.2f}: detected {100 * r['detected_share']:.0f} %, lead {r['lead_median_h']:.0f} h, "
                    f"false alerts/patient-day {r['false_alerts_per_patient_day']:.2f}" for r in rows))
            ns = e["neighbour_stability"]
            L.append(f"- neighbour stability: Jaccard transported {ns['transported']['jaccard_mean']:.2f}, Euclidean "
                     f"{ns['euclidean']['jaccard_mean']:.2f}; |Δp| transported {ns['transported']['abs_dp_mean']:.4f}")
            L.append("")
    if res.get("H"):
        L += ["## H. The autoregressive outcome", "", "Non-inferiority: upper limit of D's 90 % CI ≤ +0.005. Gains over "
              "persistence (last-MAP model recalibrated on 500 local patients): arm − persistence, negative = better.", ""]
        for u, h in res["H"].items():
            b = h["blanking_hour"]
            e = b["estimands"]["D"]
            L.append(f"### {u}")
            L.append(f"- blanking hour [L + 1, L + 6): known {100 * b['known_share']:.1f} %, prevalence "
                     f"{100 * b['prevalence']:.1f} % (primary {100 * b['prevalence_primary']:.1f} %); local rebuild gap "
                     f"{_f(b['local_rebuild_gap'], '.1e')}; D {_f(e['est'])} ({_f(e['ci90'][0])} to {_f(e['ci90'][1])}), "
                     f"non-inferior {e['noninferior']}; AUROC transported {b['arms']['transported']['auroc']:.3f}, local "
                     f"{b['arms']['local']['auroc']:.3f}, persistence {b['arms']['persistence_n500']['auroc']:.3f}")
            for name in ("map_last_ge_75", "map_last_65_75"):
                o = h[name]
                if "estimands" in o:
                    d = o["estimands"]["D"]
                    L.append(f"- {name}: {o['landmarks']} landmarks ({100 * o['share']:.0f} %), prevalence "
                             f"{100 * o['prevalence']:.1f} %; D {_f(d['est'])} ({_f(d['ci90'][0])} to {_f(d['ci90'][1])}), "
                             f"non-inferior {d['noninferior']}; AUROC transported {o['arms']['transported']['auroc']:.3f}, "
                             f"persistence {o['arms']['persistence_recal']['auroc']:.3f}")
            for a, c in h["vs_persistence"].items():
                L.append(f"- {a} − persistence (all): Δ log-loss {_f(c['dlogloss']['est'])} ({_f(c['dlogloss']['ci95'][0])} "
                         f"to {_f(c['dlogloss']['ci95'][1])}), ΔAUROC {_f(c['dauroc']['est'])} ({_f(c['dauroc']['ci95'][0])} "
                         f"to {_f(c['dauroc']['ci95'][1])})")
            L.append("")
    if res.get("G"):
        g = res["G"]
        L += ["## G. Distillation cells", "", f"Smallest patient count in any released cell: {g.get('min_patients_any_cell')}; "
              f"all ≥ 50: {g.get('all_cells_ge_50')}", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", nargs="*", default=["est", "A", "B", "C", "D", "E", "G", "H"])
    ap.add_argument("--units", nargs="*", default=list(ROTATIONS) + ["nwicu"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--n-src", type=int, default=10)
    ap.add_argument("--distill-dir", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="quick frozen artefacts (tests)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.external import Stage
    from hypo.frozen import check, external_dir

    global QUICK
    QUICK = a.quick
    rec = check()
    st = Stage(quick=a.quick, threads=a.threads)
    od = external_dir().with_name("review")
    units = od / "units"
    units.mkdir(parents=True, exist_ok=True)
    jf = od / "review.json"
    res = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else {}
    res.update({"freeze": rec.get("tag"), "plan": "docs/DECISIONS.md 2026-11-15"})
    U, cache = [], {}
    for X in a.units:
        if X == "nwicu":
            U += [unit_nwicu(st, e, cache) for e in ROTATIONS]
        else:
            U.append(unit_hypo(st, X))
    Hc = {}

    def save():
        jf.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        rep = external_dir().parent / "reports" / "review.md"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(render(res), encoding="utf-8")

    t0 = time.perf_counter()
    for blk, fn in (("est", lambda u: block_est(u)), ("A", lambda u: block_A(u, units)),
                    ("B", lambda u: block_B(u, units)), ("D", lambda u: block_D(u, units, Hc)),
                    ("E", lambda u: block_E(u, units, Hc)), ("H", lambda u: block_H(u, units, Hc, a.threads))):
        if blk not in a.blocks:
            continue
        res.setdefault(blk, {})
        for u in U:
            res[blk][u.name] = fn(u)
            LOG.info("review %s %s done (%.0f s)", blk, u.name, time.perf_counter() - t0)
            save()
    if "C" in a.blocks:
        targets = {}
        for u in U:
            X = u.name.split("_")[1]
            targets.setdefault(X, []).append(u)
        res["C"] = block_C(st, units, a.n_src, a.threads, targets)
        save()
    if "G" in a.blocks:
        res["G"] = block_G(a.distill_dir)
        save()
    check()
    LOG.info("review: %s", jf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
