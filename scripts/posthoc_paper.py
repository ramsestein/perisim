"""Post-hoc, EXPLORATORY analyses for the article (docs/DECISIONS.md, 2026-11-14, "Post hoc analyses for the article").

On stored predictions only (nothing refitted). Units: the three `hypo` targets (the rotation's transported metric) and
NWICU with each frozen metric. Runs on whichever code version and work folder the environment selects (as the other
post hoc scripts): the published v0.5 results (PERISIM_SRC=frozen/hypo/src, work) or the v0.6 re-run (work_value).

1. absolute equivalence and the R denominator;  2. calibration (CITL, slope, ICI, E50, E90);  3. decision curves;
4. a PRAM-style linear mixture (LightGBM transported + kNN transported, λ cross-fitted over folds);
5. the contingency tables behind §17's mutual-information weights: smallest patient count per cell.

Usage:  python scripts/posthoc_paper.py [--units inspire mimic eicu nwicu] [--tag v05] [--quick]
Outputs (aggregates only): <experiments>/hypo/posthoc_paper/paper_<tag>.json and reports/posthoc_paper_<tag>.md.
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
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

LOG = logging.getLogger("posthoc.paper")
N_BOOT, N_BOOT_SMOOTH, SEED = 2000, 500, 20261114
ROTATIONS = ("inspire", "mimic", "eicu")
MARGIN = 0.005
THRESH = np.round(np.arange(0.05, 0.501, 0.05), 2)
LAMBDAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
MIN_CELL = 50
EPS = 1e-6


# --------------------------------------------------------------------------- statistics (numpy only, tested)
def ll_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_weights(subjects: np.ndarray, n_boot: int, seed: int):
    """Generator of per-row multinomial patient weights (shared by every statistic of a unit)."""
    _, inv = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
    P = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.multinomial(P, np.full(P, 1.0 / P)).astype(np.float64)[inv]


def ci(b, level=0.95) -> list[float]:
    b = np.asarray(b, dtype=np.float64)
    b = b[np.isfinite(b)]
    q = 100 * (1 - level) / 2
    return [float(np.percentile(b, q)), float(np.percentile(b, 100 - q))] if b.size else [float("nan")] * 2


def wauc(score, y, w) -> float:
    u, g = np.unique(score, return_inverse=True)
    pos = np.bincount(g, weights=w * y, minlength=u.size)
    neg = np.bincount(g, weights=w * (1 - y), minlength=u.size)
    den = pos.sum() * neg.sum()
    return float((pos * (np.cumsum(neg) - 0.5 * neg)).sum() / den) if den > 0 else float("nan")


def wauprc(score, y, w) -> float:
    """Average precision with weights (ties grouped)."""
    o = np.argsort(-score, kind="stable")
    s, yy, ww = score[o], y[o], w[o]
    tp = np.cumsum(ww * yy)
    fp = np.cumsum(ww * (1 - yy))
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = tp[last], fp[last]
    P = tp[-1]
    if P <= 0:
        return float("nan")
    rec = tp / P
    prec = tp / np.maximum(tp + fp, 1e-300)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def smooth_observed(p, y, w=None, span=0.75, grid=200, bw=None, return_bw=False):
    """Local-linear tricube smoother of y on p (window: the nearest `span` share of the weight around each grid point,
    found by bisection on the sorted p), evaluated on a quantile grid and interpolated to every row."""
    w = np.ones_like(p) if w is None else w
    o = np.argsort(p)
    ps, ys, ws = p[o], y[o], w[o]
    keep = ws > 0
    ps, ys, ws = ps[keep], ys[keep], ws[keep]
    cw = np.r_[0.0, np.cumsum(ws)]
    tot = cw[-1]
    g = np.unique(np.quantile(p, np.linspace(0, 1, grid))) if bw is None else bw[0]
    out = np.empty(g.size)
    hs = np.empty(g.size)
    span_mass = span * tot
    hi_h = max(ps[-1] - ps[0], 1e-12)
    for i, x0 in enumerate(g):
        if bw is not None:                               # bootstrap: the point estimate's grid and bandwidths
            h = bw[1][i]
        else:
            lo, hi = 0.0, hi_h
            for _ in range(60):                          # smallest h with mass(|p − x0| <= h) >= span · total
                h = 0.5 * (lo + hi)
                m = cw[np.searchsorted(ps, x0 + h, "right")] - cw[np.searchsorted(ps, x0 - h, "left")]
                lo, hi = (lo, h) if m >= span_mass else (h, hi)
            h = max(hi, 1e-12)
        hs[i] = h
        a_, b_ = np.searchsorted(ps, x0 - h, "left"), np.searchsorted(ps, x0 + h, "right")
        x, yy, ww = ps[a_:b_], ys[a_:b_], ws[a_:b_]
        k = (1 - np.clip(np.abs(x - x0) / h, 0, 1) ** 3) ** 3 * ww
        sw = k.sum()
        if sw <= 0:
            out[i] = np.nan
            continue
        xm, ym = (k * x).sum() / sw, (k * yy).sum() / sw
        sxx = (k * (x - xm) ** 2).sum()
        b = (k * (x - xm) * (yy - ym)).sum() / sxx if sxx > 1e-15 else 0.0
        out[i] = ym + b * (x0 - xm)
    ok = np.isfinite(out)
    sm = np.interp(p, g[ok], np.clip(out[ok], 0.0, 1.0))
    return (sm, (g, hs)) if return_bw else sm


def calib(p, y, w=None) -> dict:
    """CITL and slope (logistic recalibration), ICI, E50, E90."""
    w = np.ones_like(p) if w is None else w
    x = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    a, b = _logistic(x, y, w, slope=True)
    a0, _ = _logistic(x, y, w, slope=False)
    e = np.abs(p - smooth_observed(p, y, w))
    q = np.repeat(e, np.round(w).astype(int)) if not np.allclose(w, 1) else e
    return {"citl": float(a0), "slope": float(b), "ici": float(np.average(e, weights=w)),
            "e50": float(np.median(q)), "e90": float(np.percentile(q, 90))}


def _logistic(x, y, w, slope=True, iters=50):
    """Weighted logistic fit of y on (offset x) [+ slope]: returns (intercept, slope) or (CITL, 1)."""
    a, b = 0.0, 1.0
    for _ in range(iters):
        eta = a + b * x if slope else a + x
        mu = 1 / (1 + np.exp(-eta))
        r = w * (y - mu)
        v = w * mu * (1 - mu)
        if slope:
            g = np.array([r.sum(), (r * x).sum()])
            H = np.array([[v.sum(), (v * x).sum()], [(v * x).sum(), (v * x * x).sum()]])
            step = np.linalg.solve(H + 1e-12 * np.eye(2), g)
            a, b = a + step[0], b + step[1]
        else:
            step = r.sum() / max(v.sum(), 1e-12)
            a += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return a, b


def net_benefit(p, y, t, w=None) -> float:
    w = np.ones_like(p) if w is None else w
    n = w.sum()
    pos = p >= t
    tp = (w * pos * y).sum() / n
    fp = (w * pos * (1 - y)).sum() / n
    return float(tp - fp * t / (1 - t))


def crossfit_lambda(p_a, p_b, y, fold) -> tuple[np.ndarray, dict]:
    """p_mix = λ p_a + (1 − λ) p_b, λ of each fold chosen on the other folds' log-loss."""
    out = np.empty_like(p_a)
    chosen = {}
    for f in np.unique(fold):
        oth = fold != f
        losses = [ll_rows(l * p_a[oth] + (1 - l) * p_b[oth], y[oth]).mean() for l in LAMBDAS]
        lam = float(LAMBDAS[int(np.argmin(losses))])
        chosen[int(f)] = lam
        te = fold == f
        out[te] = lam * p_a[te] + (1 - lam) * p_b[te]
    return out, chosen


# --------------------------------------------------------------------------- analyses of one unit
def analyse(P: dict, y: np.ndarray, subj: np.ndarray, fold: np.ndarray, tr: str, lgbm_t: str, lgbm_r: str,
            seed: int, n_boot: int = N_BOOT, n_boot_s: int = N_BOOT_SMOOTH) -> dict:
    y = y.astype(np.float64)
    P = dict(P)
    P["mixture"], lam = crossfit_lambda(P[lgbm_t], P[tr], y, fold)
    L = {a: ll_rows(v, y) for a, v in P.items()}
    est = {a: float(v.mean()) for a, v in L.items()}
    auc = {a: wauc(v, y, np.ones_like(y)) for a, v in P.items()}
    ap = {a: wauprc(v, y, np.ones_like(y)) for a, v in P.items()}
    keysB = ["G_local", "G_transp", "D", "R", "mix_vs_tr", "mix_vs_lgbmr", "auc_mix_vs_tr", "auc_mix_vs_lgbmr",
             "ap_mix_vs_tr", "ap_mix_vs_lgbmr"] + [f"dnb_{t}" for t in (0.1, 0.2, 0.3)]
    B = {k: [] for k in keysB}
    for b, w in enumerate(boot_weights(subj, n_boot, seed)):
        sw = w.sum()
        m = {a: float(w @ L[a]) / sw for a in ("euclidean", "local", tr, "mixture", lgbm_r)}
        gl, gt = m["euclidean"] - m["local"], m["euclidean"] - m[tr]
        B["G_local"].append(gl)
        B["G_transp"].append(gt)
        B["D"].append(m[tr] - m["local"])
        B["R"].append(gt / gl if gl > 0 else np.nan)
        B["mix_vs_tr"].append(m["mixture"] - m[tr])
        B["mix_vs_lgbmr"].append(m["mixture"] - m[lgbm_r])
        if b < n_boot_s:
            B["auc_mix_vs_tr"].append(wauc(P["mixture"], y, w) - wauc(P[tr], y, w))
            B["auc_mix_vs_lgbmr"].append(wauc(P["mixture"], y, w) - wauc(P[lgbm_r], y, w))
            B["ap_mix_vs_tr"].append(wauprc(P["mixture"], y, w) - wauprc(P[tr], y, w))
            B["ap_mix_vs_lgbmr"].append(wauprc(P["mixture"], y, w) - wauprc(P[lgbm_r], y, w))
            for t in (0.1, 0.2, 0.3):
                B[f"dnb_{t}"].append(net_benefit(P[tr], y, t, w) - net_benefit(P[lgbm_r], y, t, w))
    B = {k: np.asarray(v) for k, v in B.items()}
    gl, gt = est["euclidean"] - est["local"], est["euclidean"] - est[tr]
    D90 = ci(B["D"], 0.90)
    out = {"n": int(y.size), "patients": int(np.unique(np.asarray(subj).astype(str)).size),
           "prevalence": float(y.mean()), "logloss": est, "auroc": auc, "auprc": ap,
           "abs_equivalence": {"G_local": {"est": gl, "ci95": ci(B["G_local"])},
                               "G_transp": {"est": gt, "ci95": ci(B["G_transp"])},
                               "D": {"est": est[tr] - est["local"], "ci95": ci(B["D"]), "ci90": D90},
                               "R": {"est": gt / gl if gl > 0 else float("nan"), "ci90": ci(B["R"], 0.90)},
                               "share_G_local_below_margin": float(np.mean(B["G_local"] < MARGIN)),
                               "equivalent_abs": bool(D90[0] >= -MARGIN and D90[1] <= MARGIN)},
           "mixture": {"lambda_by_fold": lam,
                       "logloss_vs_knn": {"est": est["mixture"] - est[tr], "ci95": ci(B["mix_vs_tr"])},
                       "logloss_vs_lgbm_recal": {"est": est["mixture"] - est[lgbm_r], "ci95": ci(B["mix_vs_lgbmr"])},
                       "auroc_vs_knn": {"est": auc["mixture"] - auc[tr], "ci95": ci(B["auc_mix_vs_tr"])},
                       "auroc_vs_lgbm_recal": {"est": auc["mixture"] - auc[lgbm_r], "ci95": ci(B["auc_mix_vs_lgbmr"])},
                       "auprc_vs_knn": {"est": ap["mixture"] - ap[tr], "ci95": ci(B["ap_mix_vs_tr"])},
                       "auprc_vs_lgbm_recal": {"est": ap["mixture"] - ap[lgbm_r], "ci95": ci(B["ap_mix_vs_lgbmr"])}},
           "calibration": {}, "dca": {}, "dnb_knn_minus_lgbm_recal": {}}
    for a in (tr, "local", "euclidean", lgbm_t, lgbm_r, "mixture"):
        out["calibration"][a] = calib(P[a], y)
    ici = []
    _, bw = smooth_observed(P[tr], y, return_bw=True)
    for b, w in enumerate(boot_weights(subj, min(n_boot_s, 200), seed + 7)):
        e = np.abs(P[tr] - smooth_observed(P[tr], y, w, bw=bw))
        ici.append(float(np.average(e, weights=w)))
    out["calibration"][tr]["ici_ci95"] = ci(ici)
    for a in (tr, lgbm_t, lgbm_r):
        out["dca"][a] = {f"{t:.2f}": net_benefit(P[a], y, t) for t in THRESH}
    out["dca"]["treat_all"] = {f"{t:.2f}": float(y.mean() - (1 - y.mean()) * t / (1 - t)) for t in THRESH}
    for t in (0.1, 0.2, 0.3):
        out["dnb_knn_minus_lgbm_recal"][f"{t:.2f}"] = {
            "est": net_benefit(P[tr], y, t) - net_benefit(P[lgbm_r], y, t), "ci95": ci(B[f"dnb_{t}"])}
    return out


# --------------------------------------------------------------------------- loaders
def load_hypo(st, X: str):
    import polars as pl
    from hypo.external_report import _on, _phase2, _units

    S = st.S
    lm = S.lm(X)
    T = _units(f"transported_{X}_f*.parquet")
    rows = T["row"].to_numpy()
    C = _units(f"comps_{X}_f*.parquet")
    P = {"transported": T["p"].to_numpy(), "local": _on(_units(f"local_{X}_f*.parquet"), rows, "p"),
         "euclidean": _on(_phase2(X, "euclidean"), rows, "p"),
         "lgbm_transported": _on(C, rows, "p_lightgbm")}
    rec = [c for c in C.columns if c.startswith("p_lightgbm_n500_r")]
    P["lgbm_recal"] = _on(C, rows, rec[0])
    for a, v in P.items():
        if v is None or not np.all(np.isfinite(v)):
            raise RuntimeError(f"paper {X}: arm {a} has rows without a prediction")
    fold = S.fold(X)[rows]
    extra = {"lgbm_recal_draws": [float(ll_rows(_on(C, rows, c), lm.y[rows]).mean()) for c in rec]}
    return P, lm.y[rows], lm.ft.subjects[rows], fold, extra


def load_nwicu(quick: bool):
    import nwicu_transport as NT
    from hypo.cv import outer_folds, settings

    cfg, _, _ = settings(quick)
    lm = NT.landmarks(cfg)
    fold = outer_folds(lm, int(cfg["cv"]["outer_folds"]), int(cfg["seed"]))
    G = NT.gather(NT.work() / "units", len(lm))
    return G, lm.y, lm.ft.subjects, fold


def mi_audit(st, X: str) -> dict:
    """Smallest patient count behind any (code × outcome) cell of §17's MI tables on the development pool."""
    import posthoc_fixed_metric as FM
    from hypo.metric import C
    from perisim.metric.encoding import Encoder

    S = st.S
    lm = S.lm(X)
    dev = [S.lm(c) for c in S.cfg["cohorts"] if c != X]
    pft, py = FM._pool(dev)
    subj = np.concatenate([np.asarray(d.ft.subjects).astype(str) for d in dev])
    enc = Encoder(geometry="shape").fit(lm.ft, lm.y, C)
    mins, n_cells, n_small, per = [], 0, 0, {}
    for b in enc.blocks:
        c = FM.block_codes(b, pft)
        if c is None or not c[1].any():
            continue
        code, obs = c
        yy, ss = np.asarray(py)[obs].astype(int), subj[obs]
        cell = code * 2 + yy
        pats = {}
        for k in np.unique(cell):
            pats[int(k)] = int(np.unique(ss[cell == k]).size)
        m = min(pats.values())
        mins.append(m)
        n_cells += len(pats)
        small = sum(v < MIN_CELL for v in pats.values())
        n_small += small
        per[b.name] = {"cells": len(pats), "cells_below_50": int(small), "min_ge_50": bool(m >= MIN_CELL)}
    return {"blocks": len(per), "cells": n_cells, "cells_below_50": n_small,
            "all_cells_ge_50": bool(min(mins) >= MIN_CELL) if mins else None,
            "blocks_with_small_cells": sorted(k for k, v in per.items() if not v["min_ge_50"])}


# --------------------------------------------------------------------------- report
def _f(x, s="+.4f"):
    return format(x, s)


def _c(d, k="ci95", s="+.4f"):
    return f"{_f(d['est'], s)} ({_f(d[k][0], s)} to {_f(d[k][1], s)})"


def render(res: dict) -> str:
    L = [f"# Post hoc analyses for the article ({res['tag']})", "",
         "Exploratory, not pre-registered; plan in docs/DECISIONS.md (2026-11-14). Log-loss in nats; patient bootstrap.",
         "", "## 1. Absolute equivalence and the R denominator", "",
         "| unit | G_local (95 % CI) | G_transported (95 % CI) | D = L_tr − L_local (90 % CI) | equivalent within ±0.005 | "
         "share of replicates with G_local < 0.005 | R (90 % CI) |", "|---|---|---|---|---|---|---|"]
    U = res["units"]
    for u, v in U.items():
        a = v["abs_equivalence"]
        L.append(f"| {u} | {_c(a['G_local'])} | {_c(a['G_transp'])} | {_c(a['D'], 'ci90')} | "
                 f"{'yes' if a['equivalent_abs'] else 'no'} | {a['share_G_local_below_margin']:.3f} | "
                 f"{_c(a['R'], 'ci90', '.2f')} |")
    L += ["", "## 2. Calibration", "", "| unit | arm | CITL | slope | ICI | E50 | E90 |", "|---|---|---|---|---|---|---|"]
    for u, v in U.items():
        for arm, c in v["calibration"].items():
            ic = f"{c['ici']:.4f}" + (f" ({c['ici_ci95'][0]:.4f}–{c['ici_ci95'][1]:.4f})" if "ici_ci95" in c else "")
            L.append(f"| {u} | {arm} | {c['citl']:+.3f} | {c['slope']:.3f} | {ic} | {c['e50']:.4f} | {c['e90']:.4f} |")
    L += ["", "## 3. Decision curves (net benefit)", ""]
    for u, v in U.items():
        ts = list(next(iter(v["dca"].values())).keys())
        L += [f"**{u}**", "", "| arm | " + " | ".join(ts) + " |", "|---|" + "---|" * len(ts)]
        for arm, d in v["dca"].items():
            L.append(f"| {arm} | " + " | ".join(f"{d[t]:.4f}" for t in ts) + " |")
        L.append("")
        L.append("ΔNB kNN − LightGBM recalibrated: " + "; ".join(
            f"{t}: {_c(x)}" for t, x in v["dnb_knn_minus_lgbm_recal"].items()))
        L.append("")
    L += ["## 4. PRAM-style linear mixture (LightGBM transported + kNN transported)", "",
          "| unit | λ by fold | Δ log-loss vs kNN | Δ log-loss vs LightGBM recal. | Δ AUROC vs kNN | Δ AUROC vs LightGBM recal. "
          "| Δ AUPRC vs kNN | Δ AUPRC vs LightGBM recal. |", "|---|---|---|---|---|---|---|---|"]
    for u, v in U.items():
        m = v["mixture"]
        L.append(f"| {u} | {', '.join(f'{x:.2f}' for x in m['lambda_by_fold'].values())} | {_c(m['logloss_vs_knn'])} | "
                 f"{_c(m['logloss_vs_lgbm_recal'])} | {_c(m['auroc_vs_knn'])} | {_c(m['auroc_vs_lgbm_recal'])} | "
                 f"{_c(m['auprc_vs_knn'])} | {_c(m['auprc_vs_lgbm_recal'])} |")
    if res.get("mi_audit"):
        L += ["", "## 5. Mutual-information weights from aggregates: cell audit", "",
              "| rotation | blocks | cells | every cell ≥ 50 patients | blocks with a cell under 50 patients |",
              "|---|---|---|---|---|"]
        for X, a in res["mi_audit"].items():
            L.append(f"| {X} | {a['blocks']} | {a['cells']} | {'yes' if a['all_cells_ge_50'] else 'no'} | "
                     f"{', '.join(a['blocks_with_small_cells']) or '—'} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units", nargs="*", default=list(ROTATIONS) + ["nwicu"])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-mi-audit", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.external import Stage
    from hypo.frozen import check, external_dir

    rec = check()
    st = Stage(quick=a.quick, threads=a.threads)
    res = {"tag": a.tag, "freeze": rec.get("tag"), "units": {}, "mi_audit": {}}
    t0 = time.perf_counter()
    for i, X in enumerate(u for u in a.units if u != "nwicu"):
        P, y, s, fold, extra = load_hypo(st, X)
        res["units"][f"hypo_{X}"] = analyse(P, y, s, fold, "transported", "lgbm_transported", "lgbm_recal", SEED + 10 * i)
        res["units"][f"hypo_{X}"]["lgbm_recal_draws_logloss"] = extra["lgbm_recal_draws"]
        LOG.info("paper hypo %s done (%.0f s)", X, time.perf_counter() - t0)
        if not a.no_mi_audit:
            res["mi_audit"][X] = mi_audit(st, X)
    if "nwicu" in a.units:
        G, y, s, fold = load_nwicu(a.quick)
        for i, e in enumerate(ROTATIONS):
            P = {"transported": G[f"knn_{e}"], "local": G["local"], "euclidean": G["euclidean"],
                 "lgbm_transported": G[f"lgbm_{e}"], "lgbm_recal": G[f"lgbmrecal_{e}"]}
            res["units"][f"nwicu_{e}"] = analyse(P, y, s, fold, "transported", "lgbm_transported", "lgbm_recal",
                                                 SEED + 100 + 10 * i)
            LOG.info("paper nwicu %s done (%.0f s)", e, time.perf_counter() - t0)
    check()
    od = external_dir().with_name("posthoc_paper")
    od.mkdir(parents=True, exist_ok=True)
    (od / f"paper_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    rep = external_dir().parent / "reports" / f"posthoc_paper_{a.tag}.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(render(res), encoding="utf-8")
    LOG.info("paper: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
