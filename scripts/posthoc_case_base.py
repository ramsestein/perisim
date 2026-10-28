"""Post-hoc, EXPLORATORY analyses of the case base (not pre-registered: added on 2026-10-28 at the lead's request, after
the transport and drift analyses). The frozen INSPIRE metric (the "scaffold": g_j, w_j, α, k) is kept as it is; only
its case base changes. Two questions:

1. How many local patients does the scaffold need? Dense accumulation curve: n patients drawn at random from the
   site's pool (R repetitions per n, a finer grid than the pre-registered analysis C, which jumps from 2,000 to the
   whole pool), predicted on the transport stage's test set. Reported per n: log-loss, macro AUROC, distance to the
   whole pool, share of the whole pool's gain over the local prevalence, and the n at which the curve reaches the
   whole pool within 0.02 / 0.01 and the transported, recalibrated comparators of posthoc_transport.py.
2. Are the local relations hospital-specific? (eICU, the multi-centre cohort.) For the test stays of each eICU
   hospital with ≥ `min_index_stays` (as analysis F), at the SAME case-base size: A = the hospital's own pool patients;
   B = as many patients drawn at random from the other eICU hospitals (R repetitions); B' = B with its class
   probabilities moved to the hospital's own class shares (label-shift correction, p_c ∝ p_c · π_h,c / π_B,c), which
   removes a difference of base rates. A < B' = the neighbours of the same hospital carry relations that the other
   hospitals' do not. Control: the same contrasts with the hospitals replaced by random groups of patients of the same
   sizes (expected ≈ 0). Also: the share of a stay's k neighbours, in the whole eICU pool, that come from its own
   hospital, against the hospital's share of the pool (enrichment).

Usage:  python scripts/posthoc_case_base.py [--profile local] [--workdir <work>] [--threads N]
        (run_local.ps1 case-base)

Outside src/perisim on purpose (the frozen code hash covers all of it); uses the transport stage's split, labels, k and
α, and reads the frozen artefacts only (hashes checked at the start and the end). Outputs (aggregates only; hospitals
labelled H1, H2, … by size as in analysis F, never by their eICU identifier): results/posthoc_case_base/,
artifacts/<variant>/posthoc_case_base.json, reports/posthoc_case_base.html.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import posthoc_transport as PT  # noqa: E402
from perisim.evaluation import metrics as M  # noqa: E402
from perisim.evaluation.html import box, fig_img, page, table  # noqa: E402
from perisim.experiment import load_cohort_data  # noqa: E402
from perisim.external import _eval_target_for  # noqa: E402
from perisim.knn.index import search  # noqa: E402
from perisim.transport.analyses import context, hospital_map  # noqa: E402
from perisim.transport.core import SEED_OFFSET, check_hashes, frozen_hashes, patient_split, source_labels  # noqa: E402
from perisim.transport.run import _csv  # noqa: E402
from perisim.utils import write_json  # noqa: E402

LOG = logging.getLogger("perisim.casebase")
GRID = [50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 14000, 16000]
R_ACC = 20                    # repetitions per n of the accumulation curve
R_HOSP = 10                   # draws of the other hospitals' patients per hospital
MIN_TEST = 20                 # test stays a hospital needs to enter the comparison
MARGINS = (0.02, 0.01)        # "the whole pool within …"
SITES, S_COL, INK, MUTED, GRID_C, MIN_N = PT.SITES, PT.S_COL, PT.INK, PT.MUTED, PT.GRID, PT.MIN_N


# --------------------------------------------------------------------------- pure helpers
def knn(E_ref: np.ndarray, y_ref: np.ndarray, E_q: np.ndarray, k: int, alpha: float, C: int):
    """The hard kNN of the transport stage (k_eff = min(k, case base)); also the neighbours and the k-th distance."""
    k = min(k, len(y_ref))
    Dd, I = search(E_ref, E_q, k)
    L = y_ref[I]
    n = np.stack([(L == c).sum(1) for c in range(C)], 1).astype(float)
    prior = np.bincount(y_ref, minlength=C) / len(y_ref)
    return (n + alpha * prior * C) / (n.sum(1, keepdims=True) + alpha * C), I, Dd[:, -1]


def class_shares(y: np.ndarray, C: int) -> np.ndarray:
    return (np.bincount(y, minlength=C) + 1.0) / (len(y) + C)


def prior_correct(p: np.ndarray, to: np.ndarray, frm: np.ndarray) -> np.ndarray:
    """Label-shift correction: the class probabilities moved from base rates `frm` to `to` (Saerens et al., 2002)."""
    q = p * (to / frm)[None, :]
    return q / q.sum(1, keepdims=True)


def macro_auroc(p: np.ndarray, y: np.ndarray) -> float:
    a = [M.auroc_binary((y == c).astype(float), p[:, c]) for c in range(p.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(a)) if a else float("nan")


def n_to_reach(ns: np.ndarray, ll: np.ndarray, target: float) -> float:
    """Smallest n at which the mean curve reaches `target`, interpolated in log n between grid points; NaN if never."""
    ok = np.where(ll <= target)[0]
    if ok.size == 0:
        return float("nan")
    i = int(ok[0])
    if i == 0:
        return float(ns[0])
    x0, x1, y0, y1 = np.log(ns[i - 1]), np.log(ns[i]), ll[i - 1], ll[i]
    return float(np.exp(x0 + (target - y0) * (x1 - x0) / (y1 - y0))) if y1 != y0 else float(ns[i])


def per_doubling(ns: np.ndarray, ll: np.ndarray, n_min: int = 1000) -> float:
    """Slope of the mean curve per doubling of n (least squares on log₂ n, n ≥ n_min)."""
    ok = ns >= n_min
    return float(np.polyfit(np.log2(ns[ok]), ll[ok], 1)[0]) if ok.sum() >= 2 else float("nan")


def permuted_groups(subjects: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Random groups with the same patient counts as `groups` (each patient keeps one group for all its stays)."""
    pats, first = np.unique(subjects, return_index=True)
    g = groups[first].copy()
    np.random.default_rng(seed).shuffle(g)
    return g[np.searchsorted(pats, subjects)]


# --------------------------------------------------------------------------- data
def site_arrays(cfg, ws, t: str, s: str, args: dict, enc, w) -> dict:
    """Frozen embedding, labels, subjects and (eICU) hospital of the site's pool and test set (transport split)."""
    tc = cfg.experiment["transport"]
    if s == args["dev"]:
        X = load_cohort_data(ws.common(s), ws.derived(s), args["names"])
        y = source_labels(t, t, X, cfg)["own"]
        seed = int(cfg.seed) + 101 * SEED_OFFSET.get(s, 9)
        split = patient_split(X.ft.subjects, float(tc["test_fraction"]), int(tc["tuning_max_patients"]), seed)
    else:
        ctx = context(cfg, ws, t, s, args)
        X, y, split, seed = ctx["X"], ctx["lab"]["own"], ctx["split"], ctx["seed"]
    E = (enc.transform(X.ft) * enc.weights_dims(w)).astype(np.float32)
    subj = X.ft.subjects.astype(str)
    hosp = hospital_map(args["eicu_src"], X.ft.ids, ws.cache(s)) if s == "eicu" and args.get("eicu_src") else None
    out = {"seed": seed}
    lab = y >= 0
    for r in ("pool", "test"):
        idx = split[r][lab[split[r]]]
        out.update({f"E_{r}": E[idx], f"y_{r}": y[idx], f"s_{r}": subj[idx],
                    f"h_{r}": hosp[idx] if hosp is not None else None})
    if hosp is not None:                      # analysis F's hospitals and labels (H1 = most labelled index stays)
        vals, counts = np.unique(hosp[lab & (hosp != None)].astype(str), return_counts=True)  # noqa: E711
        big = [v for v, n in sorted(zip(vals, counts), key=lambda q: -q[1])
               if n >= int(tc["hospitals"]["min_index_stays"])]
        out["hospitals"] = {h: f"H{i}" for i, h in enumerate(big, start=1)}
    return out


# --------------------------------------------------------------------------- 1. accumulation
def accumulation(d: dict, k: int, alpha: float, C: int, seed: int) -> dict:
    Ep, yp, sp, Eq, yq = d["E_pool"], d["y_pool"], d["s_pool"], d["E_test"], d["y_test"]
    pats = np.unique(sp)
    p_full = knn(Ep, yp, Eq, k, alpha, C)[0]
    full = {"n": int(pats.size), "logloss": float(M.row_logloss(p_full, yq).mean()), "auroc": macro_auroc(p_full, yq)}
    prev = float(M.row_logloss(np.tile(np.bincount(yp, minlength=C) / len(yp), (len(yq), 1)), yq).mean())
    reps = []
    for n in [g for g in GRID if g < pats.size]:
        for r in range(R_ACC):
            sel = np.isin(sp, np.random.default_rng(seed + 1000 * r + n).choice(pats, size=n, replace=False))
            p = knn(Ep[sel], yp[sel], Eq, k, alpha, C)[0]
            reps.append({"n_patients": n, "rep": r, "logloss": float(M.row_logloss(p, yq).mean()),
                         "auroc": macro_auroc(p, yq)})
    return {"full": full, "prevalence_logloss": prev, "reps": reps}


def accumulation_summary(s: str, acc: dict, refs: dict) -> tuple[list[dict], dict]:
    ns = sorted({r["n_patients"] for r in acc["reps"]})
    full, prev = acc["full"], acc["prevalence_logloss"]
    rows = []
    for n in ns:
        ll = np.array([r["logloss"] for r in acc["reps"] if r["n_patients"] == n])
        au = np.array([r["auroc"] for r in acc["reps"] if r["n_patients"] == n])
        m = float(ll.mean())
        rows.append({"site": s, "n_patients": n, "logloss": m, "logloss_p10": float(np.percentile(ll, 10)),
                     "logloss_p90": float(np.percentile(ll, 90)), "auroc": float(np.nanmean(au)),
                     "gap_to_full": m - full["logloss"], "share_of_full_gain": (prev - m) / (prev - full["logloss"])})
    rows.append({"site": s, "n_patients": full["n"], "logloss": full["logloss"], "logloss_p10": full["logloss"],
                 "logloss_p90": full["logloss"], "auroc": full["auroc"], "gap_to_full": 0.0, "share_of_full_gain": 1.0})
    x = np.array([r["n_patients"] for r in rows], float)
    y = np.array([r["logloss"] for r in rows])
    summ = {"site": s, "pool_patients": full["n"], "full_logloss": full["logloss"], "prevalence_logloss": prev,
            "per_doubling": per_doubling(x, y)}
    for mg in MARGINS:
        summ[f"n_full_within_{mg}"] = n_to_reach(x, y, full["logloss"] + mg)
    for name, v in refs.items():
        summ[f"ref_{name}"] = v
        summ[f"n_to_match_{name}"] = n_to_reach(x, y, v)
    return rows, summ


def comparator_refs(ws, t: str, s: str) -> dict:
    """The transported comparators of posthoc_transport.py at this site (when that analysis has run)."""
    p = ws.root / "results" / "posthoc" / f"{t}_comparators.csv"
    if not p.exists():
        return {}
    import pandas as pd

    c = pd.read_csv(p)
    out = {}
    for method in ("lgbm", "logreg"):
        for n in (100, 500):
            v = c[(c.cohort == s) & (c.method == method) & (c.situation == f"recal. intercept+slope (n={n})")]
            if len(v):
                out[f"{method}_recal_{n}"] = float(v.logloss.iloc[0])
    for method in ("lgbm", "perisim"):
        v = c[(c.cohort == s) & (c.method == method) & (c.situation == "local refit")]
        if len(v):
            out[f"{method}_local_refit"] = float(v.logloss.iloc[0])
    return out


# --------------------------------------------------------------------------- 2. hospitals
def hospital_contrasts(d: dict, groups_pool: np.ndarray, groups_test: np.ndarray, labels: dict, k: int, alpha: float,
                       C: int, seed: int) -> dict:
    """Per hospital (group): A own pool, B the same number of other groups' patients (R_HOSP draws, per-stay losses
    averaged), B' = B moved to the hospital's class shares; the whole pool and the pool without the hospital; the
    enrichment of own-hospital neighbours in the whole pool."""
    Ep, yp, sp, Eq, yq, sq = d["E_pool"], d["y_pool"], d["s_pool"], d["E_test"], d["y_test"], d["s_test"]
    known = groups_pool != None  # noqa: E711
    p_all, I_all, _ = knn(Ep[known], yp[known], Eq, k, alpha, C)
    g_known = groups_pool[known]
    loss = {x: [] for x in ("A", "B", "Bc", "pooled", "others")}
    subj, rows = [], []
    for h, lab in labels.items():
        te = np.where(groups_test == h)[0]
        own = np.where(groups_pool == h)[0]
        oth = np.where(known & (groups_pool != h))[0]
        if te.size < MIN_TEST or own.size < MIN_TEST:
            continue
        n_h = np.unique(sp[own]).size
        pa, _, da = knn(Ep[own], yp[own], Eq[te], k, alpha, C)
        to = class_shares(yp[own], C)
        lb, lbc, db = np.zeros(te.size), np.zeros(te.size), []
        opats = np.unique(sp[oth])
        for r in range(R_HOSP):
            pick = oth[np.isin(sp[oth], np.random.default_rng(seed + 7919 * r).choice(opats, n_h, replace=False))]
            pb, _, dbr = knn(Ep[pick], yp[pick], Eq[te], k, alpha, C)
            lb += M.row_logloss(pb, yq[te]) / R_HOSP
            lbc += M.row_logloss(prior_correct(pb, to, class_shares(yp[pick], C)), yq[te]) / R_HOSP
            db.append(dbr.mean())
        po = knn(Ep[oth], yp[oth], Eq[te], k, alpha, C)[0]
        la, lp, lo = M.row_logloss(pa, yq[te]), M.row_logloss(p_all[te], yq[te]), M.row_logloss(po, yq[te])
        for x, v in (("A", la), ("B", lb), ("Bc", lbc), ("pooled", lp), ("others", lo)):
            loss[x].append(v)
        subj.append(sq[te])
        share_pool = float((g_known == h).mean())
        share_nn = float((g_known[I_all[te]] == h).mean())
        rows.append({"hospital": lab, "test_stays": int(te.size), "pool_patients": int(n_h),
                     "own": float(la.mean()), "other": float(lb.mean()), "other_corrected": float(lbc.mean()),
                     "pooled": float(lp.mean()), "pooled_without_own": float(lo.mean()),
                     "own_minus_other_corrected": float((la - lbc).mean()),
                     "kth_distance_own": float(da.mean()), "kth_distance_other": float(np.mean(db)),
                     "share_of_pool": share_pool, "share_of_neighbours": share_nn,
                     "enrichment": share_nn / share_pool if share_pool > 0 else float("nan")})
    return {"rows": rows, "loss": {x: np.concatenate(v) if v else np.zeros(0) for x, v in loss.items()},
            "subjects": np.concatenate(subj) if subj else np.zeros(0, dtype=str)}


def hospital_summary(res: dict, nb: int, seed: int) -> dict:
    L, sq = res["loss"], res["subjects"]
    if not sq.size:
        return {}
    B = M.bootstrap_rowwise(L, sq, nb, seed)

    def diff(a, b):
        return PT._pv(B[a]["_boot"] - B[b]["_boot"], B[a]["mean"] - B[b]["mean"])

    out = {x: {"logloss": B[x]["mean"], "ci_low": B[x]["ci_low"], "ci_high": B[x]["ci_high"]} for x in L}
    out.update({"own_minus_other": diff("A", "B"), "own_minus_other_corrected": diff("A", "Bc"),
                "other_corrected_minus_other": diff("Bc", "B"), "pooled_minus_without_own": diff("pooled", "others"),
                "test_stays": int(sq.size), "hospitals": len(res["rows"])})
    rows = res["rows"]
    if rows:
        w = np.array([r["test_stays"] for r in rows], float)
        out["enrichment_weighted"] = float(np.average([r["enrichment"] for r in rows], weights=w))
        out["kth_distance_own"] = float(np.average([r["kth_distance_own"] for r in rows], weights=w))
        out["kth_distance_other"] = float(np.average([r["kth_distance_other"] for r in rows], weights=w))
    return out


# --------------------------------------------------------------------------- figures
def accumulation_figure(summ_rows: dict, summ: dict, sites: list[str], dev: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sites), figsize=(4.6 * len(sites), 4.0), squeeze=False)
    for ax, s in zip(axes[0], sites):
        R = summ_rows[s]
        x = np.array([r["n_patients"] for r in R])
        m, lo, hi = (np.array([r[c] for r in R]) for c in ("logloss", "logloss_p10", "logloss_p90"))
        ax.fill_between(x, lo, hi, color=S_COL[s], alpha=0.15, lw=0)
        ax.plot(x, m, color=S_COL[s], marker="o", ms=3, lw=1.6, label="frozen metric + n local patients")
        S = summ[s]
        ax.axhline(S["prevalence_logloss"], color=MUTED, lw=0.8, ls=":", label="local prevalence")
        for key, lab, ls in (("ref_lgbm_recal_500", "LightGBM transported, recalibrated (500)", (0, (4, 2))),
                             ("ref_lgbm_local_refit", "LightGBM refitted locally", "-."),
                             ("ref_perisim_local_refit", "PERISIM refitted locally", (0, (1, 1)))):
            if key in S:
                ax.axhline(S[key], color=INK, lw=0.8, ls=ls, label=lab)
        ax.set_xscale("log")
        lo_y = min(np.nanmin(m), *(S.get(k_, np.inf) for k_ in ("ref_lgbm_local_refit", "ref_perisim_local_refit")))
        ax.set_ylim(lo_y - 0.01, S["prevalence_logloss"] + 0.06)
        PT._style(ax, SITES.get(s, s) + (" (case base from INSPIRE's own pool)" if s == dev else ""),
                  "local patients in the case base", "log-loss (test)" if s == sites[0] else "")
    axes[0][-1].legend(fontsize=6.5, frameon=False, labelcolor=INK, loc="upper right")
    fig.text(0.01, -0.04, "Mean of 20 random draws per n; band: 10–90 %. Below ~250 patients many draws contain no death "
             "(the rare class) and the curve lies off the top of the panel.", fontsize=7, color=MUTED)
    return fig


def hospital_figure(rows: list[dict], summ: dict, perm: dict):
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["own_minus_other_corrected"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 0.24 * len(rows) + 2.0), sharey=True,
                                 gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.06})
    yy = np.arange(len(rows))
    a1.scatter([r["own_minus_other_corrected"] for r in rows], yy, s=[4 + r["test_stays"] / 4 for r in rows],
               color=S_COL["eicu"], alpha=0.85)
    a1.axvline(0, color=MUTED, lw=0.6)
    for d, lab, col in ((summ.get("own_minus_other_corrected"), "all hospitals", INK),
                        (perm.get("own_minus_other_corrected"), "control: random groups", MUTED)):
        if d:
            a1.axvspan(d["ci_low"], d["ci_high"], color=col, alpha=0.12, lw=0)
            a1.axvline(d["mean"], color=col, lw=1.2, label=f"{lab}: {d['mean']:+.3f} [{d['ci_low']:+.3f}, "
                                                            f"{d['ci_high']:+.3f}]")
    a1.set_yticks(yy, [f"{r['hospital']} ({r['test_stays']})" for r in rows], fontsize=6.5)
    PT._style(a1, "Own hospital − other hospitals (same size, base rates corrected)",
              "Δ log-loss per test stay (negative = own hospital better)", "")
    a1.grid(axis="x", color=GRID_C, lw=0.8)
    a1.legend(fontsize=6.5, frameon=False, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    a2.barh(yy, [r["enrichment"] for r in rows], color=S_COL["eicu"], alpha=0.85)
    a2.axvline(1, color=MUTED, lw=0.6)
    PT._style(a2, "Own-hospital neighbours (whole eICU pool)", "share of neighbours / share of the pool", "")
    a2.grid(axis="x", color=GRID_C, lw=0.8)
    return fig


# --------------------------------------------------------------------------- report
def _ll(x: dict | None) -> str:
    return "—" if not x else f"{x['logloss']:.4f} [{x['ci_low']:.4f}, {x['ci_high']:.4f}]"


def _d(x: dict | None) -> str:
    return "—" if not x else f"{x['mean']:+.4f} [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]"


def _n(v) -> str:
    return "not reached" if v is None or not np.isfinite(v) else f"{v:,.0f}"


def render(acc_rows: dict, acc_summ: dict, hosp: dict, sites: list[str], dev: str, t: str) -> str:
    sec = [box("<b>Exploratory, post hoc.</b> Added on 2026-10-28 at the lead's request, after the transport and drift "
               "analyses; not pre-registered, no decision rule. The frozen INSPIRE metric (shapes, weights, α, k) is "
               "kept; only the case base changes. Same split, labels and test sets as the transport stage.")]
    rows = []
    for s in sites:
        S = acc_summ[s]
        rows.append([SITES.get(s, s) + (" (control)" if s == dev else ""), f"{S['pool_patients']:,}",
                     f"{S['full_logloss']:.4f}", f"{S['prevalence_logloss']:.4f}", f"{S['per_doubling']:+.4f}",
                     _n(S.get("n_full_within_0.02")), _n(S.get("n_full_within_0.01")),
                     _n(S.get("n_to_match_lgbm_recal_500")), _n(S.get("n_to_match_logreg_recal_500"))])
    sec.append("<h2>1. How many local patients the scaffold needs</h2>" + box(
        "n patients drawn at random from the site's pool (20 draws per n), the frozen metric, the transport stage's "
        "test set. <b>Per doubling</b>: change of log-loss each time the case base doubles (n ≥ 1,000). <b>Within "
        "0.02 / 0.01</b>: n at which the mean curve comes within that distance of the whole pool. <b>To match</b>: n at "
        "which it reaches LightGBM / the logistic regression transported from INSPIRE and recalibrated (intercept and "
        "slope) on 500 local patients (posthoc_transport.py).") + table(
        ["Site", "pool patients", "whole pool", "local prevalence", "per doubling", "within 0.02", "within 0.01",
         "= LightGBM recal. 500", "= logistic recal. 500"], rows)
        + fig_img(accumulation_figure(acc_rows, acc_summ, sites, dev)))
    body = [[SITES.get(r["site"], r["site"]), f"{r['n_patients']:,}", r["logloss"], r["logloss_p10"],
             r["logloss_p90"], r["auroc"], r["gap_to_full"], f"{r['share_of_full_gain']:.0%}"]
            for s in sites for r in acc_rows[s]]
    sec.append(table(["Site", "patients", "log-loss", "p10", "p90", "macro AUROC", "− whole pool",
                      "share of the whole pool's gain"], body, nd=4))
    if hosp:
        H, P = hosp["summary"], hosp["control"]
        rows = [[lab, _ll(H.get(x)), _ll(P.get(x))] for x, lab in
                (("A", "own hospital's pool (A)"), ("B", "as many patients of other hospitals (B)"),
                 ("Bc", "B moved to the hospital's base rates (B')"), ("pooled", "whole eICU pool"),
                 ("others", "whole pool without the hospital"))]
        rows += [[lab, _d(H.get(x)), _d(P.get(x))] for x, lab in
                 (("own_minus_other", "A − B"), ("own_minus_other_corrected", "A − B'  (relations)"),
                  ("other_corrected_minus_other", "B' − B  (base rates)"),
                  ("pooled_minus_without_own", "whole pool − without the hospital"))]
        sec.append("<h2>2. Are the local relations hospital-specific? (eICU)</h2>" + box(
            f"{H.get('hospitals', 0)} eICU hospitals with ≥ 300 labelled index stays (analysis F), {H.get('test_stays', 0)} "
            "test stays of the transport stage. For each hospital, at the same case-base size: A = its own pool; B = as "
            f"many patients of the other eICU hospitals ({R_HOSP} draws); B' = B with its probabilities moved to the "
            "hospital's own class shares (removes a difference of base rates). A − B' &lt; 0: the hospital's own "
            "neighbours carry relations the others do not. The control repeats everything with the hospitals replaced "
            "by random groups of patients of the same sizes.") + table(
            ["", "hospitals", "control: random groups"], rows))
        sec.append(box(f"Own-hospital share of the {hosp['k']} neighbours in the whole eICU pool / the hospital's share of "
                       f"the pool, weighted by test stays: <b>{H.get('enrichment_weighted', float('nan')):.2f}</b> "
                       f"(control {P.get('enrichment_weighted', float('nan')):.2f}). Mean distance to the k-th "
                       f"neighbour: own hospital {H.get('kth_distance_own', float('nan')):.3f}, other hospitals "
                       f"{H.get('kth_distance_other', float('nan')):.3f}.")
                   + fig_img(hospital_figure(hosp["rows"], H, P)))
        cols = ["hospital", "test_stays", "pool_patients", "own", "other", "other_corrected", "pooled",
                "pooled_without_own", "own_minus_other_corrected", "enrichment"]
        sec.append(table(cols, [[r[c] for c in cols] for r in hosp["rows"]], nd=4))
    return page("PERISIM post-hoc case base", sec, "Exploratory · not pre-registered · aggregates only")


# --------------------------------------------------------------------------- driver
def run(pipe) -> int:
    cfg, ws = pipe.cfg, pipe.ws
    dev = cfg.experiment["cohorts"]["development"]
    externals = [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()]
    targets = list(cfg.targets_to_model)
    for t in targets:
        if not (ws.predictions("external", t) / "final_model.pkl").exists():
            raise SystemExit("case-base: run the external stage first (no final model)")
    before = frozen_hashes(ws, targets)
    args = {"names": pipe.feature_names(), "dev": dev, "externals": externals,
            "eicu_src": (pipe.disc.get("eicu") or {}).get("path")}
    out: dict = {"note": "Exploratory, post hoc (2026-10-28): not pre-registered.", "targets": {}}
    rd = ws.root / "results" / "posthoc_case_base"
    rd.mkdir(parents=True, exist_ok=True)
    nb, seed0 = int(cfg.experiment["evaluation"]["n_bootstrap"]), int(cfg.seed)
    for t in targets:
        fm = PT._load(ws.predictions("external", t) / "final_model.pkl")
        enc, w = fm["enc"], np.asarray(fm["w"], float)
        k = int(cfg.experiment["knn"]["k_primary"])
        alpha = float(dict(fm["alpha"]).get(k, 1.0))
        sites = [dev] + [c for c in externals if _eval_target_for(t, c, cfg) == t]
        acc_rows, acc_summ, hosp = {}, {}, {}
        for s in sites:
            LOG.info("case-base: %s/%s", t, s)
            d = site_arrays(cfg, ws, t, s, args, enc, w)
            C = int(1 + max(d["y_pool"].max(), d["y_test"].max()))
            acc = accumulation(d, k, alpha, C, d["seed"])
            acc_rows[s], acc_summ[s] = accumulation_summary(s, acc, comparator_refs(ws, t, s))
            LOG.info("case-base: %s whole pool %.4f, per doubling %+.4f, within 0.01 at n = %s", s,
                     acc_summ[s]["full_logloss"], acc_summ[s]["per_doubling"], _n(acc_summ[s]["n_full_within_0.01"]))
            if d.get("hospitals"):
                labels = d["hospitals"]
                LOG.info("case-base: %s, %d hospitals: own vs other hospitals", s, len(labels))
                main = hospital_contrasts(d, d["h_pool"], d["h_test"], labels, k, alpha, C, d["seed"])
                gp = permuted_groups(np.r_[d["s_pool"], d["s_test"]], np.r_[d["h_pool"], d["h_test"]], d["seed"] + 5)
                n_pool = len(d["y_pool"])
                ctrl = hospital_contrasts(d, gp[:n_pool], gp[n_pool:], labels, k, alpha, C, d["seed"])
                hosp = {"rows": main["rows"], "summary": hospital_summary(main, nb, seed0),
                        "control": hospital_summary(ctrl, nb, seed0), "k": k}
                LOG.info("case-base: own − other (base rates corrected) %s; control %s",
                         _d(hosp["summary"].get("own_minus_other_corrected")),
                         _d(hosp["control"].get("own_minus_other_corrected")))
        _csv([r for s in sites for r in acc_rows[s]], rd / f"{t}_case_base_accumulation.csv")
        _csv([acc_summ[s] for s in sites], rd / f"{t}_case_base_accumulation_summary.csv")
        PT._save(accumulation_figure(acc_rows, acc_summ, sites, dev), rd / f"{t}_fig_case_base_accumulation.png")
        if hosp:
            _csv(hosp["rows"], rd / f"{t}_case_base_hospitals.csv")
            _csv([{"set": x, **{f"{a}_{b}": v for a, y in hosp[x].items() if isinstance(y, dict) for b, v in y.items()},
                   **{a: y for a, y in hosp[x].items() if not isinstance(y, dict)}} for x in ("summary", "control")],
                 rd / f"{t}_case_base_hospitals_summary.csv")
            PT._save(hospital_figure(hosp["rows"], hosp["summary"], hosp["control"]),
                     rd / f"{t}_fig_case_base_hospitals.png")
        out["targets"][t] = {"accumulation": acc_summ, "hospitals": {x: hosp.get(x) for x in ("summary", "control")}
                             if hosp else None}
        (ws.reports / "posthoc_case_base.html").write_text(render(acc_rows, acc_summ, hosp, sites, dev, t),
                                                           encoding="utf-8")
    after = frozen_hashes(ws, targets)
    check_hashes(before, after)
    out["frozen_hashes"] = {"start": before, "end": after, "unchanged": True}
    write_json(ws.artifacts() / "posthoc_case_base.json", out)
    LOG.info("case-base: results in %s; report %s", rd, ws.reports / "posthoc_case_base.html")
    return 0


def main(argv=None) -> int:
    import matplotlib

    from perisim.config import load_config
    from perisim.pipeline import Pipeline
    from perisim.utils import now_iso, set_threads, setup_logging

    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--threads", type=int, default=None, help="FAISS threads (default: all CPUs)")
    a = ap.parse_args(argv)
    over = {"experiment": {"paths": {"workdir": str(Path(a.workdir).resolve())}}} if a.workdir else {}
    pipe = Pipeline(load_config(None, a.profile, over))
    setup_logging(pipe.ws.logs / f"case_base_{now_iso().replace(':', '')}.log")
    set_threads(a.threads or os.cpu_count() or 4)
    return run(pipe)


if __name__ == "__main__":
    raise SystemExit(main())
