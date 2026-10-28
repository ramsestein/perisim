"""Post-hoc, EXPLORATORY follow-up of the drift analysis (not pre-registered: added on 2026-10-28 at the lead's
request, after scripts/posthoc_drift.py). The question: how much of a site's drift is in what a MISSING value means.

The frozen INSPIRE metric is kept as it is, with each site's own patients as the case base (as in posthoc_drift.py),
and only the points of the missing-data values are freed and fitted to the site, all at once: the "missing" value of
every continuous variable (it starts at the median bin's point, as INSPIRE imputes the median), the "missing" value of
every missingness indicator and the "unobserved" value of every binary (it starts at INSPIRE's prevalence). Every other
point (bins, categories, 0 / 1) stays frozen. Same smoothed kNN, patient-level leave-one-out, ridge and evaluation as
the drift analysis, so the result reads against its other fits on the same test set:

    frozen metric  →  missing points adapted  →  all points adapted (drift, joint)  →  PERISIM refitted locally

The missing values are read in three groups, reported separately by taking only one group's adapted points (no refit):
  measurement  a laboratory value, vital sign, GCS, FiO₂ or trend that was not recorded;
  structural   hours since a support ended, missing because the support was never used (never ventilated, no
               vasopressors): a clinical state, not a gap in the record;
  source       a binary whose source was not recorded for the stay (medication tables, emergency status).
Each missing point's move is also fitted on two patient halves of the case base (agreement of the two moves).

Usage:  python scripts/posthoc_drift_missing.py [--profile local] [--workdir <work>] [--force]
        (run_local.ps1 drift-missing)

Outside src/perisim on purpose (the frozen code hash covers all of it); reuses posthoc_drift.py and reads the frozen
artefacts only (hashes checked at the start and the end). Fits cached in
predictions/<variant>/posthoc/<t>/drift_missing_<site>.pkl (never exported). Outputs (aggregates only; the moves of
values with < 10 stays at a site are not exported): results/posthoc_drift_missing/,
artifacts/<variant>/posthoc_drift_missing.json, reports/posthoc_drift_missing.html.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import posthoc_drift as D  # noqa: E402
import posthoc_transport as PT  # noqa: E402
from perisim.evaluation import metrics as M  # noqa: E402
from perisim.evaluation.html import box, fig_img, page, table  # noqa: E402
from perisim.external import _eval_target_for  # noqa: E402
from perisim.transport.core import check_hashes, frozen_hashes, variable_class  # noqa: E402
from perisim.transport.run import _csv  # noqa: E402
from perisim.utils import write_json  # noqa: E402

LOG = logging.getLogger("perisim.drift")
PHASE = "drift-missing-1"     # change when the cached fits change meaning
MISSING_LABELS = ("missing", "unobserved")
GROUPS = ("measurement", "structural", "source")
G_COL = {"measurement": "#2a78d6", "structural": "#eda100", "source": "#8b6fd0"}
MIN_N, SITES, S_COL, INK, MUTED, GRID = D.MIN_N, D.SITES, D.S_COL, D.INK, D.MUTED, D.GRID


# --------------------------------------------------------------------------- which points move (pure helpers)
def missing_group(name: str, label: str) -> str:
    if label == "unobserved":
        return "source"
    if name.startswith("hours_since_"):
        return "structural"
    return "measurement"


def missing_rows(look: list[dict], groups: tuple[str, ...] = GROUPS) -> list[np.ndarray]:
    """Per variable, which of its values are missing-data values of the given groups."""
    return [np.array([lab in MISSING_LABELS and missing_group(lj["name"], lab) in groups for lab in lj["labels"]])
            for lj in look]


def with_rows(base: list[np.ndarray], adapted: list[np.ndarray], free: list[np.ndarray]) -> list[np.ndarray]:
    """`base` with the rows `free` taken from `adapted`."""
    return [np.where(f[:, None], a, b) if f.any() else b for b, a, f in zip(base, adapted, free)]


def fit_rows(points: list[np.ndarray], free: list[np.ndarray], codes: np.ndarray, y: np.ndarray, subj: np.ndarray,
             C: int, k: int, alpha: float, rounds: int = D.ROUNDS, iters: int = D.ITERS,
             ridge: float = D.RIDGE, seed: int = 0) -> dict:
    """Only the rows `free[j]` of the points move, all variables at once (patient-level LOO, ridge towards the frozen
    points); every other point is frozen. The variables with a free row enter each anchor–candidate distance through
    their own block, the rest of the distance is fixed within a round (float32, anchors in chunks)."""
    J = [j for j in range(len(points)) if free[j].any()]
    n = len(y)
    P0 = {j: torch.tensor(points[j], dtype=torch.float32) for j in J}
    idx = {j: torch.as_tensor(np.where(free[j])[0]) for j in J}
    # a missing value starts ON an observed value's point (the median's), where the pull of those pairs is exactly
    # zero: a tiny fixed jitter (10⁻³ of the variable's scale) breaks the tie
    rng = np.random.default_rng(seed)
    delta = {j: torch.tensor(rng.normal(0.0, 1e-3 * max(float(np.abs(points[j]).max()), 1e-6),
                                        (len(idx[j]), points[j].shape[1])), dtype=torch.float32, requires_grad=True)
             for j in J}
    prior = torch.as_tensor(np.bincount(y, minlength=C) / n, dtype=torch.float32)
    ct, y_t = torch.as_tensor(codes), torch.as_tensor(y)

    def current() -> dict:
        return {j: torch.index_put(P0[j], (idx[j],), P0[j][idx[j]] + delta[j]) for j in J}

    trace = []
    for _ in range(rounds):
        with torch.no_grad():
            cur = current()
        pts = list(points)
        for j in J:
            pts[j] = cur[j].numpy().astype(float)
        Dd, I, valid = D._candidates(D.embed(pts, codes), subj)
        It, vt, yc = torch.as_tensor(I), torch.as_tensor(valid), torch.as_tensor(y[I])
        rest = torch.as_tensor(Dd, dtype=torch.float32).clone()
        for s in range(0, n, D.J_CHUNK):                       # the free variables' part, taken out once per round
            a = slice(s, min(s + D.J_CHUNK, n))
            Ea = torch.cat([cur[j][ct[a, j]] for j in J], 1)
            Ec = torch.cat([cur[j][ct[It[a], j]] for j in J], 2)
            rest[a] -= ((Ea[:, None, :] - Ec) ** 2).sum(-1)
        rest.clamp_(min=0.0)
        opt = torch.optim.LBFGS([delta[j] for j in J], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def closure(grad: bool = True):
            if grad:
                opt.zero_grad()
            tot = 0.0
            for s in range(0, n, D.J_CHUNK):
                a = slice(s, min(s + D.J_CHUNK, n))
                Pc = current()                                   # small: rebuilt per chunk, one graph each
                Ea = torch.cat([Pc[j][ct[a, j]] for j in J], 1)
                Ec = torch.cat([Pc[j][ct[It[a], j]] for j in J], 2)
                p = D.smoothed_probs(rest[a] + ((Ea[:, None, :] - Ec) ** 2).sum(-1), vt[a], yc[a], C, k, alpha, prior)
                part = -torch.log(p[torch.arange(p.shape[0]), y_t[a]]).sum() / n
                if grad:
                    part.backward()
                tot += float(part.detach())
            reg = ridge / n * sum((delta[j] ** 2).sum() for j in J)
            if grad:
                reg.backward()
            return torch.tensor(tot + float(reg.detach()))

        with torch.no_grad():
            start = float(closure(False))
        opt.step(closure)
        with torch.no_grad():
            trace.append((start, float(closure(False))))
    with torch.no_grad():
        cur = current()
    out = list(points)
    for j in J:
        out[j] = cur[j].numpy().astype(float)
    return {"points": out, "trace": trace}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


# --------------------------------------------------------------------------- fits per site (cached)
def _cache_key(ws, t: str) -> str:
    return hashlib.sha256(f"{PHASE}|{D._cache_key(ws, t)}".encode()).hexdigest()[:16]


def site_fits(path: Path, key: str, look: list[dict], free: list[np.ndarray], d: dict, C: int, k: int, alpha: float,
              force: bool) -> dict:
    fits = D._load(path) if path.exists() and not force else {}
    if fits.get("key") != key:
        fits = {"key": key, "fit": None, "halves": []}
    P0 = [lj["points"] for lj in look]
    cp, yp, sp = d["codes_pool"], d["y_pool"], d["s_pool"]
    pats = np.random.default_rng(d["seed"] + 23).permutation(np.unique(sp))       # the halves of posthoc_drift.py
    half = np.isin(sp, pats[: pats.size // 2])
    if fits["fit"] is None:
        t0 = time.time()
        fits["fit"] = fit_rows(P0, free, cp, yp, sp, C, k, alpha)
        LOG.info("drift-missing: missing points at once %.0fs  smoothed LOO %.4f → %.4f", time.time() - t0,
                 fits["fit"]["trace"][0][0], fits["fit"]["trace"][-1][1])
        D._dump(fits, path)
    while len(fits["halves"]) < 2:
        m = half if not fits["halves"] else ~half
        t0 = time.time()
        fits["halves"].append(fit_rows(P0, free, cp[m], yp[m], sp[m], C, k, alpha)["points"])
        LOG.info("drift-missing: patient half %d %.0fs", len(fits["halves"]), time.time() - t0)
        D._dump(fits, path)
    fits["half_counts"] = [[np.bincount(cp[m][:, j], minlength=P0[j].shape[0]) for j in range(len(P0))]
                           for m in (half, ~half)]
    return fits


# --------------------------------------------------------------------------- evaluation per site
def evaluate_site(cfg, s: str, d: dict, fits: dict, look: list[dict], free: list[np.ndarray], w_raw: np.ndarray,
                  C: int, k: int, alpha: float, joint_pts: list[np.ndarray] | None,
                  local_rows: np.ndarray | None) -> dict:
    P0 = [lj["points"] for lj in look]
    Pm = fits["fit"]["points"]
    cp, yp, cq, yq, sq = d["codes_pool"], d["y_pool"], d["codes_test"], d["y_test"], d["s_test"]
    nb, seed = int(cfg.experiment["evaluation"]["n_bootstrap"]), int(cfg.seed)

    def rows(pts):
        return M.row_logloss(D.knn_probs(D.embed(pts, cp), yp, D.embed(pts, cq), k, alpha, C), yq)

    vals = {"frozen": rows(P0), "missing": rows(Pm)}
    counts = [np.bincount(cp[:, j], minlength=P0[j].shape[0]) for j in range(len(look))]
    for g in GROUPS:                                     # a group the site's case base has (≥ MIN_N stays somewhere)
        fg = missing_rows(look, (g,))
        if max(int(counts[j][fg[j]].max()) if fg[j].any() else 0 for j in range(len(look))) >= MIN_N:
            vals[f"g:{g}"] = rows(with_rows(P0, Pm, fg))
    if joint_pts is not None:
        vals["joint"] = rows(joint_pts)
    if local_rows is not None:
        vals["local"] = local_rows
    B = M.bootstrap_rowwise(vals, sq, nb, seed)

    def diff(a, b):
        return PT._pv(B[a]["_boot"] - B[b]["_boot"], B[a]["mean"] - B[b]["mean"])

    summary = {x: {"logloss": B[x]["mean"], "ci_low": B[x]["ci_low"], "ci_high": B[x]["ci_high"]} for x in vals}
    for x in vals:
        if x != "frozen":
            summary[f"{x}_minus_frozen"] = diff(x, "frozen")
    for ref in ("joint", "local"):
        if ref in vals:
            summary[f"missing_minus_{ref}"] = diff("missing", ref)
    gain = B["frozen"]["mean"] - B["missing"]["mean"]
    for ref in ("joint", "local"):
        if ref in vals:
            gap = B["frozen"]["mean"] - B[ref]["mean"]
            summary[f"share_of_{ref}_gain"] = float(gain / gap) if abs(gap) > 1e-9 else float("nan")

    W_tot = float(np.sum(w_raw))
    HA, HB = fits["halves"]
    ca, cb = fits["half_counts"]
    out_rows, ha_all, hb_all = [], [], []
    for j, lj in enumerate(look):
        if not free[j].any():
            continue
        pi = np.bincount(cp[:, j], minlength=P0[j].shape[0]).astype(float)
        share = PT.suppressed_shares(pi)
        obs = ~np.isin(np.array(lj["labels"]), MISSING_LABELS)
        mu = (pi[obs, None] * P0[j][obs]).sum(0) / pi[obs].sum() if pi[obs].sum() > 0 else P0[j][obs].mean(0)
        for v in np.where(free[j])[0]:
            keep = pi[v] >= MIN_N
            mv, ma, mb = Pm[j][v] - P0[j][v], HA[j][v] - P0[j][v], HB[j][v] - P0[j][v]
            both = ca[j][v] >= MIN_N and cb[j][v] >= MIN_N
            if keep and both:
                ha_all.append(ma)
                hb_all.append(mb)
            r = {"site": s, "variable": lj["name"], "label": lj["labels"][v], "kind": lj["kind"],
                 "group": missing_group(lj["name"], lj["labels"][v]), "variable_class": variable_class(lj["name"]),
                 "share": float(share[v]), "w_inspire": float(w_raw[j] / W_tot),
                 "dist_observed_frozen": float(np.linalg.norm(P0[j][v] - mu)),
                 "dist_observed_adapted": float(np.linalg.norm(Pm[j][v] - mu)) if keep else np.nan,
                 "move": float(np.linalg.norm(mv)) if keep else np.nan,
                 "half_cosine": cosine(ma, mb) if keep and both else np.nan,
                 "w_site_frozen": D.effective_weight(P0[j], pi) / W_tot,
                 "w_site_adapted": D.effective_weight(Pm[j], pi) / W_tot if keep else np.nan}
            for c in range(P0[j].shape[1]):
                r[f"move_c{c}"] = float(mv[c]) if keep else np.nan
            out_rows.append(r)
    a, b = (np.concatenate(x) if x else np.zeros(0) for x in (ha_all, hb_all))
    summary["split_half_r"] = float(np.corrcoef(a, b)[0, 1]) if a.size >= 3 and a.std() > 1e-9 and b.std() > 1e-9 \
        else float("nan")
    return {"site": s, "n_test": int(len(yq)), "test_patients": int(np.unique(sq).size), "summary": summary,
            "points": out_rows, "trace": fits["fit"]["trace"]}


# --------------------------------------------------------------------------- figures
CONDS = [("missing", "missing points, all"), ("g:measurement", "  only measurement gaps"),
         ("g:structural", "  only supports never used"), ("g:source", "  only source not recorded"),
         ("joint", "all points (drift, joint)"), ("local", "PERISIM refitted locally")]


def gain_figure(res: dict, sites: list[str], dev: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    h = 0.8 / len(sites)
    for i, s in enumerate(sites):
        S = res[s]["summary"]
        ys, g, lo, hi = [], [], [], []
        for c, (key, _) in enumerate(CONDS):
            dd = S.get(f"{key}_minus_frozen")
            if dd is None:
                continue
            ys.append(c + (i - (len(sites) - 1) / 2) * h)
            g.append(-dd["mean"])
            lo.append(-dd["ci_high"])
            hi.append(-dd["ci_low"])
        g, lo, hi = map(np.array, (g, lo, hi))
        ax.barh(ys, g, h, color=S_COL[s], alpha=0.55 if s == dev else 0.9,
                label=SITES.get(s, s) + (" (control)" if s == dev else ""))
        ax.errorbar(g, ys, xerr=[g - lo, hi - g], fmt="none", ecolor=INK, elinewidth=0.6, capsize=0)
    ax.set_yticks(np.arange(len(CONDS)), [lab for _, lab in CONDS], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.6)
    PT._style(ax, "Log-loss recovered with the frozen metric and the site's case base (test set)",
              "log-loss gain over the frozen metric, 95 % CI", "")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(fontsize=7, frameon=False, ncol=3, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.4, -0.15))
    return fig


def points_figure(res: dict, sites: list[str], dev: str, top: int = 18):
    import matplotlib.pyplot as plt

    ext = [s for s in sites if s != dev] or sites
    R = {(r["site"], r["variable"], r["label"]): r for s in sites for r in res[s]["points"]}
    keys = sorted({(v, lab) for (_, v, lab) in R if any(np.isfinite(R.get((s, v, lab), {}).get("move", np.nan))
                                                        for s in ext)},
                  key=lambda x: -max(np.nan_to_num(R.get((s,) + x, {}).get("move", 0.0)) for s in ext))
    med = [x for x in keys if x[0].startswith("med_")]
    keys = [x for x in keys if not x[0].startswith("med_")][:top] + med[:1]   # medication flags: one block, one row
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 0.3 * len(keys) + 1.8), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 2.2], "wspace": 0.05})
    yy = np.arange(len(keys))
    hgt = 0.8 / len(sites)
    for i, s in enumerate(sites):
        off = (i - (len(sites) - 1) / 2) * hgt
        sh = np.array([R.get((s,) + x, {}).get("share", np.nan) for x in keys]) * 100
        a1.barh(yy + off, np.nan_to_num(sh), hgt, color=S_COL[s], label=SITES.get(s, s))
        f = np.array([R.get((s,) + x, {}).get("dist_observed_frozen", np.nan) for x in keys])
        ad = np.array([R.get((s,) + x, {}).get("dist_observed_adapted", np.nan) for x in keys])
        ok = np.isfinite(ad)
        a2.hlines(yy[ok] + off, f[ok], ad[ok], color=S_COL[s], lw=1.0)
        a2.scatter(f[ok], yy[ok] + off, s=12, facecolors="white", edgecolors=S_COL[s], lw=0.8)
        a2.scatter(ad[ok], yy[ok] + off, s=16, color=S_COL[s],
                   label=f"adapted to {SITES.get(s, s)}" + (" (control)" if s == dev else ""))
    lab = [f"medication flags (largest: {v})" if v.startswith("med_") else v + (" (unobserved)" if lb == "unobserved"
                                                                                  else "") for v, lb in keys]
    a1.set_yticks(yy, lab, fontsize=7)
    a1.invert_xaxis()
    a1.invert_yaxis()
    PT._style(a1, "stays with the value missing", "% of the site's stays", "")
    PT._style(a2, "distance from the observed values (metric units): ○ frozen → ● adapted", "distance", "")
    for ax in (a1, a2):
        ax.grid(axis="x", color=GRID, lw=0.8)
    a2.legend(fontsize=7, frameon=False, ncol=3, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.4, -0.08))
    return fig


# --------------------------------------------------------------------------- report
def _ll(x: dict | None) -> str:
    return "—" if x is None else f"{x['logloss']:.4f} [{x['ci_low']:.4f}, {x['ci_high']:.4f}]"


def _d(x: dict | None) -> str:
    return "—" if x is None else D._ci(x["mean"], x["ci_low"], x["ci_high"])


def _pct(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.0%}"


def render(res: dict, sites: list[str], dev: str) -> str:
    sec = [box("<b>Exploratory, post hoc.</b> Added on 2026-10-28 at the lead's request, after the drift analysis; not "
               "pre-registered, no decision rule. The frozen INSPIRE metric with each site's own case base (as in the "
               "drift analysis, same test set, labels and k); <b>only the points of the missing-data values move</b>, "
               "all at once: the 'missing' value of every continuous variable and missingness indicator (it starts at "
               "the median's point, as INSPIRE imputes the median) and the 'unobserved' value of every binary. Every "
               "other point is frozen. The groups: <b>measurement</b> (a value not recorded), <b>structural</b> "
               "(hours since a support that was never used) and <b>source</b> (a binary whose source was not recorded, "
               "e.g. the medication tables); each group's row takes only its own adapted points (no refit). INSPIRE is "
               "the control (in-sample).")]
    rows = []
    for s in sites:
        S = res[s]["summary"]
        rows.append([SITES.get(s, s) + (" (control)" if s == dev else ""), _ll(S.get("frozen")), _ll(S.get("missing")),
                     _d(S.get("missing_minus_frozen")), _ll(S.get("joint")), _pct(S.get("share_of_joint_gain")),
                     _ll(S.get("local")), _pct(S.get("share_of_local_gain")), _d(S.get("missing_minus_local")),
                     f"{S['split_half_r']:.2f}" if np.isfinite(S["split_half_r"]) else "—"])
    sec.append("<h2>Missing points re-deformed at once</h2>" + table(
        ["Site", "frozen metric", "missing points adapted", "missing − frozen", "all points adapted (drift)",
         "share of the drift gain", "PERISIM refitted locally", "share of the gap to the refit", "missing − refit",
         "split-half r of the moves"], rows))
    rows = []
    for s in sites:
        S = res[s]["summary"]
        for g in GROUPS:
            if f"g:{g}_minus_frozen" in S:
                rows.append([SITES.get(s, s), g, _ll(S[f"g:{g}"]), _d(S[f"g:{g}_minus_frozen"])])
    sec.append("<h2>By kind of missing value</h2>" + table(["Site", "group", "log-loss", "− frozen"], rows)
               + fig_img(gain_figure(res, sites, dev)))
    sec.append("<h2>Which missing values move</h2>" + box(
        "Left: share of the site's stays with the value missing. Right: distance of the missing value's point from the "
        "mean of the observed values (in the metric, so it includes the variable's weight); ○ frozen (where INSPIRE "
        "puts it), ● adapted to the site. Moving away = the site wants 'missing' to mean something of its own. Values "
        "with < 10 stays at a site are not shown. The medication flags share one 'unobserved' pattern (the "
        "medication tables of the stay) and are summarised by their largest move.") + fig_img(points_figure(res, sites, dev)))
    hdr = ["Site", "variable", "value", "group", "% stays", "distance frozen → adapted", "move", "half cosine",
           "weight frozen → adapted"]
    body = []
    for s in sites:
        for r in sorted(res[s]["points"], key=lambda r: -np.nan_to_num(r["move"])):
            if np.isfinite(r["move"]):
                body.append([SITES.get(s, s), r["variable"], r["label"], r["group"], _pct(r["share"]),
                             f"{r['dist_observed_frozen']:.3f} → {r['dist_observed_adapted']:.3f}", r["move"],
                             r["half_cosine"], f"{r['w_site_frozen']:.1%} → {r['w_site_adapted']:.1%}"])
    sec.append(box("<b>Move</b>: distance between the frozen and the adapted point. <b>Half cosine</b>: agreement in "
                   "direction of the moves fitted on two patient halves (1 = the same direction). <b>Weight</b>: "
                   "effective weight of the whole variable at the site (% of Σ w).") + table(hdr, body, nd=3))
    return page("PERISIM post-hoc drift: missing values", sec, "Exploratory · not pre-registered · aggregates only")


# --------------------------------------------------------------------------- driver
def run(pipe, force: bool = False) -> int:
    cfg, ws = pipe.cfg, pipe.ws
    dev = cfg.experiment["cohorts"]["development"]
    externals = [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()]
    targets = list(cfg.targets_to_model)
    for t in targets:
        if not (ws.predictions("external", t) / "final_model.pkl").exists():
            raise SystemExit("drift-missing: run the external stage first (no final model)")
    before = frozen_hashes(ws, targets)
    args = {"names": pipe.feature_names(), "dev": dev, "externals": externals}
    out: dict = {"note": "Exploratory, post hoc (2026-10-28): not pre-registered.", "targets": {}}
    rd = ws.root / "results" / "posthoc_drift_missing"
    rd.mkdir(parents=True, exist_ok=True)
    for t in targets:
        fm = PT._load(ws.predictions("external", t) / "final_model.pkl")
        enc, w = fm["enc"], np.asarray(fm["w"], float)
        k = int(cfg.experiment["knn"]["k_primary"])
        alpha = float(dict(fm["alpha"]).get(k, 1.0))
        look = D.frozen_points(enc, w)
        free = missing_rows(look)
        sites = [dev] + [c for c in externals if _eval_target_for(t, c, cfg) == t]
        key, dkey = _cache_key(ws, t), D._cache_key(ws, t)
        res = {}
        for s in sites:
            LOG.info("drift-missing: %s/%s", t, s)
            d = D.site_data(cfg, ws, t, s, args, enc, w, look)
            C = int(1 + max(d["y_pool"].max(), d["y_test"].max()))
            fits = site_fits(ws.predictions("posthoc", t) / f"drift_missing_{s}.pkl", key, look, free, d, C, k,
                             alpha, force)
            joint = None
            dp = ws.predictions("posthoc", t) / f"drift_{s}.pkl"
            if dp.exists():
                df = D._load(dp)
                if df.get("key") == dkey and df.get("joint") is not None:
                    joint = df["joint"]["points"]
            if joint is None:
                LOG.warning("drift-missing: no joint drift fit for %s (run posthoc_drift.py first)", s)
            local = None
            npz = ws.predictions("posthoc", t) / f"{s}_preds.npz"
            if s != dev and npz.exists():
                z = np.load(npz)
                if np.array_equal(z["y_test"], d["y_test"]) and np.array_equal(z["s_test"], d["s_test"]):
                    local = M.row_logloss(z["perisim__local__test"], d["y_test"])
            LOG.info("drift-missing: evaluating %s/%s (test, bootstrap)", t, s)
            res[s] = evaluate_site(cfg, s, d, fits, look, free, w, C, k, alpha, joint, local)
        out["targets"][t] = {s: {kk: v for kk, v in r.items() if kk != "points"} for s, r in res.items()}
        _csv([x for s in sites for x in res[s]["points"]], rd / f"{t}_drift_missing_points.csv")
        _csv([{"site": s, **{f"{a}_{b}": v for a, x in res[s]["summary"].items() if isinstance(x, dict)
                             for b, v in x.items()},
               **{a: x for a, x in res[s]["summary"].items() if not isinstance(x, dict)}} for s in sites],
             rd / f"{t}_drift_missing_summary.csv")
        PT._save(gain_figure(res, sites, dev), rd / f"{t}_fig_drift_missing_gain.png")
        PT._save(points_figure(res, sites, dev), rd / f"{t}_fig_drift_missing_points.png")
        (ws.reports / "posthoc_drift_missing.html").write_text(render(res, sites, dev), encoding="utf-8")
    after = frozen_hashes(ws, targets)
    check_hashes(before, after)
    out["frozen_hashes"] = {"start": before, "end": after, "unchanged": True}
    write_json(ws.artifacts() / "posthoc_drift_missing.json", out)
    LOG.info("drift-missing: results in %s; report %s", rd, ws.reports / "posthoc_drift_missing.html")
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
    ap.add_argument("--threads", type=int, default=None, help="torch / FAISS threads (default: all CPUs)")
    ap.add_argument("--force", action="store_true", help="refit the points instead of reusing the cached fits")
    a = ap.parse_args(argv)
    over = {"experiment": {"paths": {"workdir": str(Path(a.workdir).resolve())}}} if a.workdir else {}
    pipe = Pipeline(load_config(None, a.profile, over))
    setup_logging(pipe.ws.logs / f"drift_missing_{now_iso().replace(':', '')}.log")
    set_threads(a.threads or os.cpu_count() or 4)
    return run(pipe, force=a.force)


if __name__ == "__main__":
    raise SystemExit(main())
