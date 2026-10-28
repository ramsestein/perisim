"""Post-hoc, EXPLORATORY drift analysis (not pre-registered: added on 2026-10-28 at the lead's request, after the
transport analyses). The frozen INSPIRE metric is kept as it is and only its case base is swapped for each site's own
patients (analysis A(i) of the transport stage). The question: which variables would have to change their shape
function g_j, or their importance, for the metric to fit the site, and how.

Every variable of the metric is a lookup. Each stay has one VALUE per variable (a bin of INSPIRE's quantile grid, a
category, 0 / 1 for binaries and missingness indicators) and the variable's block of the weighted embedding is the
POINT of that value, √w_j · g_j(value) / s_j ("metric coordinates": squared distances between points add up to d²).
Continuous variables get one extra value, "missing", that starts at the point of the median bin (INSPIRE imputes the
median), so that informative missingness can show up; binaries keep "unobserved" at the training prevalence.

1. ICE through the frozen geometry. N_ICE test stays of the site are moved along every value of x_j, the rest of each
   patient unchanged, and predicted by the hard kNN of the transport stage (k_primary, frozen α) with the site's case
   base: one curve per stay in centred log-odds per class (the units of g_j), centred over the values with INSPIRE's
   value counts; reported as the mean (PDP) and the 10–90 % band. Blind where INSPIRE's g_j is flat: there the stays
   do not move in the metric.
2. Re-deformation, one variable at a time. The points of x_j are freed, for the case base and the queries alike,
   and fitted to the site with everything else frozen: gradient (L-BFGS) on a smoothed version of the same kNN (each
   candidate votes σ((r_a − d²)/τ_a) around the anchor's k-th distance r_a; frozen α prior), patient-level
   leave-one-out over the site's case base, ridge RIDGE towards the frozen points. Reported: the adapted points, the
   change in TEST log-loss of the real kNN (patient bootstrap, paired with the frozen metric), the variable's effective
   weight (mean squared distance between two random stays of the site, in the units of w_j) with INSPIRE's points and
   with the adapted ones, the shape change (share of the adapted points' spread that a scaled, rotated copy of
   INSPIRE's points cannot reproduce) and the split-half reliability of the deformation (the case base cut in two
   patient halves, each refitted alone).
3. All variables at once: how much of the distance to the locally refitted PERISIM (A(iii)) deforming the INSPIRE
   metric can recover.

INSPIRE itself is run as the CONTROL, with the same split rule (test 20 % of its patients, the pool as case base) and
the same fits. The final metric was read from all of INSPIRE, so the control is in-sample; and g_j was read from
surrogate models, not fitted to the kNN loss, so the optimiser moves points even at home. The drift of a site is what
moves there beyond what moves in INSPIRE.

Usage:  python scripts/posthoc_drift.py [--profile local] [--workdir <work>] [--force]      (run_local.ps1 drift)

Outside src/perisim on purpose (the frozen code hash covers all of it). Only reads the frozen artefacts (hashes checked
at the start and the end); uses the transport stage's split, labels and k. The fitted points are cached per site in
predictions/<variant>/posthoc/<t>/drift_<site>.pkl (resumable, never exported); --force refits. Outputs (aggregates
only; adapted points of values with < 10 stays at a site are not exported): results/posthoc_drift/,
artifacts/<variant>/posthoc_drift.json, reports/posthoc_drift.html.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import posthoc_transport as PT  # noqa: E402
from perisim.evaluation import metrics as M  # noqa: E402
from perisim.evaluation.html import box, fig_img, page, table  # noqa: E402
from perisim.experiment import load_cohort_data  # noqa: E402
from perisim.external import _eval_target_for  # noqa: E402
from perisim.importance.shapes import centred_logodds  # noqa: E402
from perisim.knn.index import search, search_excluding  # noqa: E402
from perisim.metric.encoding import OTHER, _norm_cat  # noqa: E402
from perisim.transport.analyses import context  # noqa: E402
from perisim.transport.core import (  # noqa: E402
    SEED_OFFSET, check_hashes, frozen_hashes, patient_split, source_labels, variable_class,
)
from perisim.transport.run import _csv  # noqa: E402
from perisim.utils import write_json  # noqa: E402

LOG = logging.getLogger("perisim.drift")
PHASE = "drift-1"             # change when the cached fits change meaning
K_CAND = 150                  # candidates per anchor in the smoothed kNN (k = 50 plus a margin)
SMOOTH = 10                   # τ_a = (d²_(k+SMOOTH) − d²_(k−SMOOTH)) / 4: the width of the smoothed vote
RIDGE = 20.0                  # a move of 1 metric unit at one value costs RIDGE / n_anchors nats
ROUNDS, ITERS = 3, 30         # candidate refreshes × L-BFGS iterations, one variable
J_ROUNDS, J_ITERS, J_CHUNK = 2, 25, 2048   # the same, all variables at once (anchors in chunks)
N_ICE = 1000                  # test stays moved along every variable
MIN_N = PT.MIN_N              # §9 small-cell threshold
SITES, S_COL, INK, MUTED, GRID, V_COL = PT.SITES, PT.S_COL, PT.INK, PT.MUTED, PT.GRID, PT.V_COL


# --------------------------------------------------------------------------- the metric as a lookup (pure helpers)
def frozen_points(enc, w) -> list[dict]:
    """Per metric variable: its embedding columns, value labels and frozen points (one row per value, metric
    coordinates). `exact` is False only for a neutral rank axis, quantised here to each bin's mid-rank."""
    sw = np.sqrt(np.asarray(w, float))
    out = []
    for j, b in enumerate(enc.blocks):
        f = sw[j] / b.scale
        exact = True
        if b.kind in ("continuous", "ordinal"):
            e = b.params["edges"]
            if b.shape is not None:
                pts = b.shape * f
            else:
                srt = b.params["sorted"]
                mid = (np.searchsorted(srt, e[:-1], "left") + np.searchsorted(srt, e[1:], "right")) / (2.0 * srt.size)
                pts, exact = mid[:, None] * f, False
            med = int(np.clip(np.searchsorted(e, b.params["median"], side="right") - 1, 0, e.size - 2))
            pts, labels = np.vstack([pts, pts[med]]), PT._bin_labels(b) + ["missing"]
        elif b.kind == "nominal" and b.shape is not None:
            pts = np.vstack([b.shape, np.zeros((1, b.shape.shape[1]))]) * f
            labels = [str(v) for v in b.params["values"]] + ["unseen"]
        elif b.kind == "nominal":
            pts, labels = np.eye(len(b.params["onehot"])) / np.sqrt(2.0) * f, [str(v) for v in b.params["onehot"]]
        elif b.kind == "hier":
            K = len(b.params["codes"])
            pts = (b.shape if b.shape is not None else np.eye(K + 2) / np.sqrt(2.0)) * f
            labels = [str(c) for c in b.params["codes"]] + ["other", "missing"]
        elif b.kind == "binary":
            pts, labels = np.array([[0.0], [1.0], [b.params["prevalence"]]]) * f, ["0", "1", "unobserved"]
        else:                                                   # missingness indicator
            pts, labels = np.array([[0.0], [1.0]]) * f, ["observed", "missing"]
        out.append({"name": b.name, "kind": b.kind, "cols": np.where(enc.dim_feature == j)[0],
                    "points": np.asarray(pts, float), "labels": labels, "shaped": b.shape is not None,
                    "exact": exact})
    return out


def value_codes(enc, ft, look: list[dict]) -> np.ndarray:
    """Each stay's value in every metric variable (n × F): the row of its point in `frozen_points`."""
    codes = np.zeros((len(ft), len(enc.blocks)), dtype=np.int64)
    for j, b in enumerate(enc.blocks):
        V = look[j]["points"].shape[0]
        if b.kind in ("continuous", "ordinal"):
            x = ft.num[b.source]
            codes[:, j] = np.where(np.isfinite(x), enc._bin_index(b, ft), V - 1)
        elif b.kind == "nominal" and b.shape is not None:
            idx = enc.value_index(b, ft)
            codes[:, j] = np.where(idx >= 0, idx, V - 1)        # never seen in INSPIRE → zero contribution
        elif b.kind == "nominal":
            v = _norm_cat(ft.cat[b.source])
            pos = {c: i for i, c in enumerate(b.params["onehot"])}
            codes[:, j] = [pos[c if c in b.params["keep"] else OTHER] for c in v]
        elif b.kind == "hier":
            codes[:, j] = enc._hier_index(b.params, np.asarray(ft.cat[b.source], dtype=object))
        elif b.kind == "binary":
            x = ft.num[b.source]
            codes[:, j] = np.where(np.isfinite(x), (x > 0.5).astype(np.int64), 2)
        else:
            codes[:, j] = (~np.isfinite(ft.num[b.source])).astype(np.int64)
    return codes


def embed(points: list[np.ndarray], codes: np.ndarray) -> np.ndarray:
    """Weighted embedding of the stays from the points of their values (the metric's own column order)."""
    return np.hstack([points[j][codes[:, j]] for j in range(len(points))]).astype(np.float32)


def knn_probs(E_ref: np.ndarray, y_ref: np.ndarray, E_q: np.ndarray, k: int, alpha: float, C: int) -> np.ndarray:
    """The hard kNN of the transport stage: p_c = (n_c + α·P(c)·C) / (k + α·C), P(c) the case base's class share."""
    _, I = search(E_ref, E_q, k)
    L = y_ref[I]
    n = np.stack([(L == c).sum(1) for c in range(C)], 1).astype(float)
    prior = np.bincount(y_ref, minlength=C) / len(y_ref)
    return (n + alpha * prior * C) / (n.sum(1, keepdims=True) + alpha * C)


def smoothed_probs(d2: torch.Tensor, valid: torch.Tensor, yc: torch.Tensor, C: int, k: int, alpha: float,
                   prior: torch.Tensor) -> torch.Tensor:
    """Differentiable version of `knn_probs` over each anchor's candidates: a candidate's vote is σ((r − d²)/τ), r the
    anchor's k-th distance and τ a quarter of the spread between its (k − SMOOTH)-th and (k + SMOOTH)-th distances (both
    held fixed within a step), so the votes add up to ≈ k as in the real kNN."""
    big = torch.where(valid, d2.detach(), torch.full_like(d2, torch.finfo(d2.dtype).max))
    srt = torch.sort(big, dim=1).values
    K = d2.shape[1]
    r = srt[:, k - 1]
    hi = srt[:, min(k - 1 + SMOOTH, K - 1)]
    hi = torch.where(hi < torch.finfo(d2.dtype).max / 2, hi, r)
    tau = ((hi - srt[:, max(k - 1 - SMOOTH, 0)]) / 4.0).clamp_min(1e-6)
    s = torch.sigmoid((r[:, None] - d2) / tau[:, None]) * valid
    n = torch.stack([(s * (yc == c)).sum(1) for c in range(C)], 1)
    return (n + alpha * prior * C) / (n.sum(1, keepdim=True) + alpha * C)


def _candidates(E: np.ndarray, subj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return search_excluding(E, E, K_CAND, subj, subj, self_query=True)


def fit_variable(points: list[np.ndarray], j: int, codes: np.ndarray, y: np.ndarray, subj: np.ndarray, C: int,
                 k: int, alpha: float, rounds: int = ROUNDS, iters: int = ITERS, ridge: float = RIDGE) -> dict:
    """Points of variable j fitted to the case base (patient-level LOO), every other variable frozen. The other
    variables' part of each anchor–candidate distance is fixed within a round, so only V_j × d_j numbers move."""
    P0 = np.asarray(points[j], float)
    P = torch.tensor(P0, dtype=torch.float64, requires_grad=True)
    P0t = torch.tensor(P0, dtype=torch.float64)
    n = len(y)
    prior = torch.as_tensor(np.bincount(y, minlength=C) / n, dtype=torch.float64)
    y_t, va = torch.as_tensor(y), torch.as_tensor(codes[:, j])
    trace = []
    for _ in range(rounds):
        cur = list(points)
        cur[j] = P.detach().numpy()
        D, I, valid = _candidates(embed(cur, codes), subj)
        own = ((cur[j][codes[:, j]][:, None, :] - cur[j][codes[I, j]]) ** 2).sum(-1)
        rest = torch.as_tensor(np.maximum(D.astype(np.float64) - own, 0.0))
        vc, yc, vt = torch.as_tensor(codes[I, j]), torch.as_tensor(y[I]), torch.as_tensor(valid)
        opt = torch.optim.LBFGS([P], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def objective():
            diff = P[va][:, None, :] - P[vc]
            p = smoothed_probs(rest + (diff * diff).sum(-1), vt, yc, C, k, alpha, prior)
            return -torch.log(p[torch.arange(n), y_t]).mean() + ridge / n * ((P - P0t) ** 2).sum()

        def closure():
            opt.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        with torch.no_grad():
            start = float(objective())
        opt.step(closure)
        with torch.no_grad():
            trace.append((start, float(objective())))
    return {"points": P.detach().numpy().copy(), "trace": trace}


def fit_joint(points: list[np.ndarray], codes: np.ndarray, y: np.ndarray, subj: np.ndarray, C: int, k: int,
              alpha: float, rounds: int = J_ROUNDS, iters: int = J_ITERS, ridge: float = RIDGE) -> dict:
    """All the variables' points at once (float32, anchors in chunks with the gradient accumulated)."""
    F, n = len(points), len(y)
    Ps = [torch.tensor(p, dtype=torch.float32, requires_grad=True) for p in points]
    P0 = [torch.tensor(p, dtype=torch.float32) for p in points]
    prior = torch.as_tensor(np.bincount(y, minlength=C) / n, dtype=torch.float32)
    ct, y_t = torch.as_tensor(codes), torch.as_tensor(y)
    trace = []
    for _ in range(rounds):
        _, I, valid = _candidates(embed([p.detach().numpy() for p in Ps], codes), subj)
        It, vt, yc = torch.as_tensor(I), torch.as_tensor(valid), torch.as_tensor(y[I])
        opt = torch.optim.LBFGS(Ps, lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def closure(grad: bool = True):
            if grad:
                opt.zero_grad()
            tot = 0.0
            for s in range(0, n, J_CHUNK):
                a = slice(s, min(s + J_CHUNK, n))
                Ea = torch.cat([Ps[j][ct[a, j]] for j in range(F)], 1)
                Ec = torch.cat([Ps[j][ct[It[a], j]] for j in range(F)], 2)
                p = smoothed_probs(((Ea[:, None, :] - Ec) ** 2).sum(-1), vt[a], yc[a], C, k, alpha, prior)
                part = -torch.log(p[torch.arange(p.shape[0]), y_t[a]]).sum() / n
                if grad:
                    part.backward()
                tot += float(part.detach())
            reg = ridge / n * sum(((Ps[j] - P0[j]) ** 2).sum() for j in range(F))
            if grad:
                reg.backward()
            return torch.tensor(tot + float(reg))

        with torch.no_grad():
            start = float(closure(False))
        opt.step(closure)
        with torch.no_grad():
            trace.append((start, float(closure(False))))
    return {"points": [p.detach().numpy().astype(float) for p in Ps], "trace": trace}


def effective_weight(P: np.ndarray, pi: np.ndarray) -> float:
    """Mean squared distance between the points of two random stays drawn from the value distribution π, in the
    units of w_j (with INSPIRE's points and INSPIRE's training distribution it is w_j itself)."""
    pi = pi / max(pi.sum(), 1e-12)
    mu = (pi[:, None] * P).sum(0)
    return float(2.0 * (pi * ((P - mu) ** 2).sum(1)).sum())


def shape_change(P0: np.ndarray, P1: np.ndarray, pi: np.ndarray) -> float:
    """Share of the adapted points' spread (weighted by π) that a scaled, rotated and shifted copy of the frozen
    points cannot reproduce (weighted Procrustes): 0 = the same shape, only stretched; 1 = unrelated."""
    ok = pi > 0
    if ok.sum() < 3:
        return float("nan")
    p = pi[ok] / pi[ok].sum()
    X = P0[ok] - (p[:, None] * P0[ok]).sum(0)
    Y = P1[ok] - (p[:, None] * P1[ok]).sum(0)
    sx, sy = float((p * (X ** 2).sum(1)).sum()), float((p * (Y ** 2).sum(1)).sum())
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    sv = np.linalg.svd((X * p[:, None]).T @ Y, compute_uv=False)
    return float(np.clip(1.0 - sv.sum() ** 2 / (sx * sy), 0.0, 1.0))


def split_half_r(P0: np.ndarray, PA: np.ndarray, PB: np.ndarray, nA: np.ndarray, nB: np.ndarray) -> float:
    """Correlation between the deformations (adapted − frozen) fitted on two patient halves, over the values with
    ≥ MIN_N stays in both."""
    ok = (nA >= MIN_N) & (nB >= MIN_N)
    a, b = (PA - P0)[ok].ravel(), (PB - P0)[ok].ravel()
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ice_curves(points: list[np.ndarray], j: int, look_j: dict, codes_ref: np.ndarray, y_ref: np.ndarray,
               codes_q: np.ndarray, k: int, alpha: float, C: int, ref_counts: np.ndarray) -> dict:
    """ICE of the hard kNN along variable j: every query set to each value in turn (the case base as it is), centred
    log-odds per class, each stay's curve centred over the values with INSPIRE's value counts."""
    E_r, E_q = embed(points, codes_ref), embed(points, codes_q)
    V = points[j].shape[0]
    cur = np.zeros((len(codes_q), V, C))
    for v in range(V):
        Eq = E_q.copy()
        Eq[:, look_j["cols"]] = points[j][v]
        cur[:, v] = centred_logodds(knn_probs(E_r, y_ref, Eq, k, alpha, C))
    wv = ref_counts / max(ref_counts.sum(), 1e-12)
    cur -= (wv[None, :, None] * cur).sum(1, keepdims=True)
    return {"pdp": cur.mean(0), "p10": np.percentile(cur, 10, axis=0), "p90": np.percentile(cur, 90, axis=0)}


# --------------------------------------------------------------------------- data and fits per site (cached)
def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _dump(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    tmp.replace(path)


def site_data(cfg, ws, t: str, s: str, args: dict, enc, w, look: list[dict]) -> dict:
    """Codes, labels and subjects of the site's case base (pool) and test set; the lookup is checked to reproduce the
    frozen embedding exactly."""
    tc = cfg.experiment["transport"]
    if s == args["dev"]:
        X = load_cohort_data(ws.common(s), ws.derived(s), args["names"])
        y = source_labels(t, t, X, cfg)["own"]
        seed = int(cfg.seed) + 101 * SEED_OFFSET.get(s, 9)
        split = patient_split(X.ft.subjects, float(tc["test_fraction"]), int(tc["tuning_max_patients"]), seed)
    else:
        ctx = context(cfg, ws, t, s, args)
        X, y, split, seed = ctx["X"], ctx["lab"]["own"], ctx["split"], ctx["seed"]
    codes = value_codes(enc, X.ft, look)
    exact = [j for j, lj in enumerate(look) if lj["exact"]]
    E = enc.transform(X.ft) * enc.weights_dims(w)
    cols = np.concatenate([look[j]["cols"] for j in exact])
    err = float(np.abs(embed([lj["points"] for lj in look], codes)[:, cols] - E[:, cols]).max())
    if err > 1e-4:
        raise RuntimeError(f"drift: the lookup does not reproduce the frozen embedding of {s} (max error {err:.2g})")
    pool, test = (split[r][y[split[r]] >= 0] for r in ("pool", "test"))
    subj = X.ft.subjects.astype(str)
    return {"codes_pool": codes[pool], "y_pool": y[pool], "s_pool": subj[pool], "codes_test": codes[test],
            "y_test": y[test], "s_test": subj[test], "seed": seed, "codes_all": codes if s == args["dev"] else None}


def _cache_key(ws, t: str) -> str:
    h = hashlib.sha256(f"{PHASE}|{K_CAND}|{SMOOTH}|{RIDGE}|{ROUNDS}|{ITERS}|{J_ROUNDS}|{J_ITERS}".encode())
    h.update((ws.predictions("external", t) / "final_model.pkl").read_bytes())
    return h.hexdigest()[:16]


def site_fits(path: Path, key: str, look: list[dict], d: dict, C: int, k: int, alpha: float, force: bool) -> dict:
    """Per-variable fits (full case base and two patient halves for the shaped variables) and the joint fit; saved
    after every variable, so an interrupted run resumes."""
    fits = _load(path) if path.exists() and not force else {}
    if fits.get("key") != key:
        fits = {"key": key, "single": {}, "halves": {}, "joint": None}
    P0 = [lj["points"] for lj in look]
    cp, yp, sp = d["codes_pool"], d["y_pool"], d["s_pool"]
    pats = np.random.default_rng(d["seed"] + 23).permutation(np.unique(sp))
    half = np.isin(sp, pats[: pats.size // 2])
    for j, lj in enumerate(look):
        name = lj["name"]
        if name not in fits["single"]:
            t0 = time.time()
            fits["single"][name] = fit_variable(P0, j, cp, yp, sp, C, k, alpha)
            LOG.info("drift: %-30s %.1fs  smoothed LOO %.4f → %.4f", name, time.time() - t0,
                     fits["single"][name]["trace"][0][0], fits["single"][name]["trace"][-1][1])
        if lj["kind"] not in ("binary", "missing") and name not in fits["halves"]:
            fits["halves"][name] = [fit_variable(P0, j, cp[m], yp[m], sp[m], C, k, alpha)["points"]
                                    for m in (half, ~half)]
        _dump(fits, path)
    if fits["joint"] is None:
        t0 = time.time()
        fits["joint"] = fit_joint(P0, cp, yp, sp, C, k, alpha)
        LOG.info("drift: all variables at once %.0fs  smoothed LOO %.4f → %.4f", time.time() - t0,
                 fits["joint"]["trace"][0][0], fits["joint"]["trace"][-1][1])
        _dump(fits, path)
    fits["half_counts"] = [[np.bincount(cp[m][:, j], minlength=P0[j].shape[0]) for j in range(len(P0))]
                           for m in (half, ~half)]
    return fits


# --------------------------------------------------------------------------- evaluation per site
def evaluate_site(cfg, s: str, d: dict, fits: dict, look: list[dict], ins_counts: list[np.ndarray], w_raw: np.ndarray,
                  C: int, k: int, alpha: float, local_rows: np.ndarray | None) -> dict:
    P0 = [lj["points"] for lj in look]
    cp, yp, cq, yq, sq = d["codes_pool"], d["y_pool"], d["codes_test"], d["y_test"], d["s_test"]
    nb, seed = int(cfg.experiment["evaluation"]["n_bootstrap"]), int(cfg.seed)

    def rows(pts):
        return M.row_logloss(knn_probs(embed(pts, cp), yp, embed(pts, cq), k, alpha, C), yq)

    vals = {"frozen": rows(P0), "joint": rows(fits["joint"]["points"])}
    if local_rows is not None:
        vals["local"] = local_rows
    for j, lj in enumerate(look):
        pts = list(P0)
        pts[j] = fits["single"][lj["name"]]["points"]
        vals[f"v:{lj['name']}"] = rows(pts)
    B = M.bootstrap_rowwise(vals, sq, nb, seed)

    def diff(a, b):
        return PT._pv(B[a]["_boot"] - B[b]["_boot"], B[a]["mean"] - B[b]["mean"])

    q = np.random.default_rng(seed + 31).choice(len(cq), size=min(N_ICE, len(cq)), replace=False)
    W_tot = float(np.sum(w_raw))
    variables, points, ice = [], [], []
    for j, lj in enumerate(look):
        name, P1 = lj["name"], fits["single"][lj["name"]]["points"]
        pi = np.bincount(cp[:, j], minlength=P0[j].shape[0]).astype(float)
        dd = diff(f"v:{name}", "frozen")
        hv = fits["halves"].get(name)
        r = {"site": s, "variable": name, "kind": lj["kind"], "variable_class": variable_class(name),
             "w_inspire": float(w_raw[j] / W_tot), "w_site_frozen": effective_weight(P0[j], pi) / W_tot,
             "w_site_adapted": effective_weight(P1, pi) / W_tot,
             "shape_change": shape_change(P0[j], P1, pi) if lj["kind"] not in ("binary", "missing") else np.nan,
             "split_half_r": split_half_r(P0[j], hv[0], hv[1], fits["half_counts"][0][j], fits["half_counts"][1][j])
             if hv is not None else np.nan,
             "d_logloss": dd["mean"], "d_ci_low": dd["ci_low"], "d_ci_high": dd["ci_high"], "d_p": dd["p_two_sided"],
             "n_values": int((pi >= MIN_N).sum())}
        r["weight_ratio"] = r["w_site_adapted"] / r["w_site_frozen"] if r["w_site_frozen"] > 1e-12 else np.nan
        variables.append(r)
        share = PT.suppressed_shares(pi)
        for v, lab in enumerate(lj["labels"]):
            for c in range(P0[j].shape[1]):
                keep = pi[v] >= MIN_N
                points.append({"site": s, "variable": name, "value": v, "label": lab, "component": c,
                               "share": float(share[v]), "frozen": float(P0[j][v, c]),
                               "adapted": float(P1[v, c]) if keep else np.nan,
                               "half_a": float(hv[0][v, c]) if hv is not None and keep else np.nan,
                               "half_b": float(hv[1][v, c]) if hv is not None and keep else np.nan})
        if lj["kind"] in ("binary", "missing"):
            continue
        pts1 = list(P0)
        pts1[j] = P1
        fr = ice_curves(P0, j, lj, cp, yp, cq[q], k, alpha, C, ins_counts[j])
        ad = ice_curves(pts1, j, lj, cp, yp, cq[q], k, alpha, C, ins_counts[j])
        for v, lab in enumerate(lj["labels"]):
            for c in range(C):
                ice.append({"site": s, "variable": name, "value": v, "label": lab, "class": c,
                            "pdp_frozen": float(fr["pdp"][v, c]), "p10_frozen": float(fr["p10"][v, c]),
                            "p90_frozen": float(fr["p90"][v, c]), "pdp_adapted": float(ad["pdp"][v, c])})
    joint = {k_: {"logloss": B[k_]["mean"], "ci_low": B[k_]["ci_low"], "ci_high": B[k_]["ci_high"]}
             for k_ in ("frozen", "joint", "local") if k_ in B}
    joint["joint_minus_frozen"] = diff("joint", "frozen")
    if "local" in B:
        joint["joint_minus_local"] = diff("joint", "local")
        joint["frozen_minus_local"] = diff("frozen", "local")
    joint["sum_single_gains"] = float(sum(min(r["d_logloss"], 0.0) for r in variables))
    jw = fits["joint"]["points"]
    for r, (j, lj) in zip(variables, enumerate(look)):
        pi = np.bincount(cp[:, j], minlength=P0[j].shape[0]).astype(float)
        r["w_site_joint"] = effective_weight(jw[j], pi) / W_tot
    return {"site": s, "n_test": int(len(yq)), "test_patients": int(np.unique(sq).size),
            "pool_patients": int(np.unique(d["s_pool"]).size), "variables": variables, "points": points, "ice": ice,
            "joint": joint, "trace_joint": fits["joint"]["trace"]}


# --------------------------------------------------------------------------- figures
def gain_figure(res: dict, sites: list[str], dev: str, top: int = 22):
    import matplotlib.pyplot as plt

    V = {(r["site"], r["variable"]): r for s in sites for r in res[s]["variables"]}
    ext = [s for s in sites if s != dev]
    names = sorted({r["variable"] for s in sites for r in res[s]["variables"]},
                   key=lambda n: min(V[(s, n)]["d_logloss"] for s in ext or sites))[:top]
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(names) + 1.6))
    h = 0.8 / len(sites)
    for i, s in enumerate(sites):
        ys = np.arange(len(names)) + (i - (len(sites) - 1) / 2) * h
        g = np.array([-V[(s, n)]["d_logloss"] for n in names])
        lo = np.array([-V[(s, n)]["d_ci_high"] for n in names])
        hi = np.array([-V[(s, n)]["d_ci_low"] for n in names])
        ax.barh(ys, g, h, color=S_COL[s], label=SITES.get(s, s) + (" (control)" if s == dev else ""),
                alpha=0.55 if s == dev else 0.9)
        ax.errorbar(g, ys, xerr=[g - lo, hi - g], fmt="none", ecolor=INK, elinewidth=0.6, capsize=0)
    ax.set_yticks(np.arange(len(names)), names, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.6)
    PT._style(ax, "Log-loss recovered by re-deforming ONE variable (test set, the rest of the metric frozen)",
              "log-loss gain (frozen − adapted), 95 % CI", "")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(fontsize=7, frameon=False, ncol=3, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.45, -0.07))
    return fig


def drift_map_figure(res: dict, sites: list[str], dev: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sites), figsize=(4.6 * len(sites), 4.3), squeeze=False, sharey=True)
    for ax, s in zip(axes[0], sites):
        R = [r for r in res[s]["variables"] if r["kind"] not in ("binary", "missing")]
        x = np.array([np.log2(r["weight_ratio"]) if np.isfinite(r["weight_ratio"]) and r["weight_ratio"] > 0
                      else np.nan for r in R])
        y = np.array([r["shape_change"] for r in R])
        g = np.array([max(-r["d_logloss"], 0.0) for r in R])
        for cls in V_COL:
            sel = np.isfinite(x) & np.isfinite(y) & np.array([r["variable_class"] == cls for r in R])
            if sel.any():
                ax.scatter(np.clip(x[sel], -3, 3), y[sel], s=18 + 6000 * g[sel], color=V_COL[cls], alpha=0.75, lw=0,
                           label=cls)
        for i in np.argsort(-g)[:7]:
            if np.isfinite(x[i]) and np.isfinite(y[i]):
                ax.annotate(R[i]["variable"], (np.clip(x[i], -3, 3), y[i]), fontsize=6.5, color=INK, xytext=(3, 3),
                            textcoords="offset points")
        ax.axvline(0, color=MUTED, lw=0.6)
        PT._style(ax, SITES.get(s, s) + (" — control" if s == dev else ""),
                  "importance change: log₂(adapted / frozen effective weight)", "shape change" if s == sites[0] else "")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-0.03, 1.03)
    leg = axes[0][0].legend(fontsize=7, frameon=False, ncol=5, labelcolor=INK, loc="upper center",
                            bbox_to_anchor=(len(sites) / 2, -0.16))
    for hd in leg.legend_handles:
        hd.set_sizes([30])
    fig.text(0.01, -0.1, "Shaped variables. Point size: log-loss gain of re-deforming the variable alone. Shape change: "
             "share of the adapted points' spread a scaled, rotated copy of INSPIRE's cannot reproduce.", color=MUTED,
             fontsize=7)
    return fig


def variable_figure(name: str, res: dict, sites: list[str], dev: str, look_j: dict, class_names: list[str]):
    import matplotlib.pyplot as plt

    pts = {s: [r for r in res[s]["points"] if r["variable"] == name] for s in sites}
    ice = {s: [r for r in res[s]["ice"] if r["variable"] == name] for s in sites}
    var = {s: next(r for r in res[s]["variables"] if r["variable"] == name) for s in sites}
    labels, Cd = look_j["labels"], look_j["points"].shape[1]
    nv = len(labels)
    C = 1 + max(r["class"] for s in sites for r in ice[s])
    cont = look_j["kind"] in ("continuous", "ordinal")
    nb = nv - 1 if cont else nv                 # continuous: the last value is "missing", drawn apart
    fig = plt.figure(figsize=(12.5, 8.2))
    gsp = fig.add_gridspec(3, max(C, Cd), height_ratios=[1.6, 1.6, 1.0], hspace=0.5 if cont else 0.85, wspace=0.12)
    top = [fig.add_subplot(gsp[0, c]) for c in range(Cd)]
    mid = [fig.add_subplot(gsp[1, c]) for c in range(C)]
    bot = fig.add_subplot(gsp[2, :])
    for row in (top, mid):
        for ax in row[1:]:
            ax.sharey(row[0])
    xs = np.arange(nb) + 0.5

    def series(rows, key, c, ckey):
        rr = sorted((r for r in rows if r[ckey] == c), key=lambda r: r["value"])
        return np.array([r[key] for r in rr])

    for c, ax in enumerate(top):
        f0 = series(pts[dev], "frozen", c, "component")
        ax.plot(xs, f0[:nb], color=INK, lw=1.2, ls=(0, (3, 2)), marker="o", ms=2.5, label="INSPIRE (frozen)")
        if cont:
            ax.plot([nb + 0.7], [f0[nb]], color=INK, marker="o", ms=3, ls="none")
        for s in sites:
            a = series(pts[s], "adapted", c, "component")
            for hk in ("half_a", "half_b"):
                hh = series(pts[s], hk, c, "component")
                ax.plot(xs, hh[:nb], color=S_COL[s], lw=0.6, alpha=0.35)
            ax.plot(xs, a[:nb], color=S_COL[s], lw=1.6, marker="o", ms=2.5,
                    label=f"adapted to {SITES.get(s, s)}" + (" (control)" if s == dev else ""))
            if cont:
                ax.plot([nb + 0.7], [a[nb]], color=S_COL[s], marker="D", ms=4, ls="none")
        ax.axhline(0, color=MUTED, lw=0.6)
        PT._style(ax, f"component {class_names[c] if c < len(class_names) else c}", "",
                  "point (metric coordinates)" if c == 0 else "")
        if c:
            ax.tick_params(labelleft=False)
    for c, ax in enumerate(mid):
        for s in sites:
            fr, ad = series(ice[s], "pdp_frozen", c, "class"), series(ice[s], "pdp_adapted", c, "class")
            lo, hi = series(ice[s], "p10_frozen", c, "class"), series(ice[s], "p90_frozen", c, "class")
            ax.fill_between(xs, lo[:nb], hi[:nb], color=S_COL[s], alpha=0.08, lw=0)
            ax.plot(xs, fr[:nb], color=S_COL[s], lw=1.1, ls=(0, (3, 2)))
            ax.plot(xs, ad[:nb], color=S_COL[s], lw=1.6)
            if cont:
                ax.plot([nb + 0.7], [fr[nb]], color=S_COL[s], marker="o", mfc="white", ms=4, ls="none")
                ax.plot([nb + 0.7], [ad[nb]], color=S_COL[s], marker="D", ms=4, ls="none")
        ax.axhline(0, color=MUTED, lw=0.6)
        PT._style(ax, class_names[c] if c < len(class_names) else f"class {c}", "",
                  "ICE mean (centred log-odds)" if c == 0 else "")
        if c:
            ax.tick_params(labelleft=False)
    width = 0.8 / len(sites)
    for i, s in enumerate(sites):
        sh = series(pts[s], "share", 0, "component") * 100
        xb = np.r_[xs, nb + 0.7] if cont else xs
        bot.bar(xb + (i - (len(sites) - 1) / 2) * width, sh, width, color=S_COL[s], label=SITES.get(s, s))
    if cont:                    # ticks at INSPIRE's inner bin edges (the outer ones are single patients' values)
        step = max(1, int(np.ceil((nb - 1) / 9)))
        pos = list(range(1, nb, step))
        lab = [labels[p].split(",")[0].lstrip("[").replace("≥ ", "") for p in pos]
        for ax in top + mid + [bot]:
            ax.set_xlim(0, nb + 1.2)
            ax.set_xticks(pos + [nb + 0.7], lab + ["miss."])
    else:
        for ax in top + mid + [bot]:
            ax.set_xticks(xs, labels, rotation=40, ha="right", fontsize=6)
    PT._style(bot, "Where each site's case base falls (values with < 10 stays not shown)", "", "% of the site's stays")
    top[0].legend(fontsize=7, frameon=False, ncol=4, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.15))
    bot.legend(fontsize=7, frameon=False, ncol=3, labelcolor=INK, loc="upper right")
    fig.text(0.01, -0.04 if cont else -0.09, "Top: points of every value (metric coordinates); thin lines: the two patient halves. Middle: "
             "ICE mean, dashed = frozen INSPIRE metric (band: 10–90 % of the stays), solid = this variable re-deformed. "
             "Values with < 10 stays at a site: adapted point not shown.", fontsize=7, color=MUTED)
    stats = "  ·  ".join(
        f"{SITES.get(s, s)}: Δ log-loss {var[s]['d_logloss']:+.4f}, weight {var[s]['w_site_frozen']:.1%} → "
        f"{var[s]['w_site_adapted']:.1%}, shape change {var[s]['shape_change']:.2f}, split-half r "
        f"{var[s]['split_half_r']:.2f}" for s in sites)
    fig.suptitle(f"{name}  ·  {variable_class(name)}  ·  INSPIRE weight {var[dev]['w_inspire']:.1%}\n{stats}",
                 color=INK, fontsize=8.5, x=0.01, ha="left", y=1.0)
    return fig


def binary_figure(res: dict, sites: list[str], dev: str):
    import matplotlib.pyplot as plt

    R = {(r["site"], r["variable"]): r for s in sites for r in res[s]["variables"] if r["kind"] in ("binary", "missing")}
    names = sorted({n for _, n in R}, key=lambda n: -R[(dev, n)]["w_inspire"])
    fig, ax = plt.subplots(figsize=(8.5, 0.24 * len(names) + 1.5))
    yy = np.arange(len(names))
    ax.scatter([R[(dev, n)]["w_inspire"] * 100 for n in names], yy, marker="|", s=60, color=INK, label="INSPIRE w_j")
    for i, s in enumerate(sites):
        off = (i - (len(sites) - 1) / 2) * 0.22
        f = np.array([R[(s, n)]["w_site_frozen"] * 100 for n in names])
        a = np.array([R[(s, n)]["w_site_adapted"] * 100 for n in names])
        ax.hlines(yy + off, f, a, color=S_COL[s], lw=1.0)
        ax.scatter(a, yy + off, s=14, color=S_COL[s], label=f"adapted to {SITES.get(s, s)}" +
                   (" (control)" if s == dev else ""))
    ax.set_yticks(yy, names, fontsize=6.5)
    ax.invert_yaxis()
    PT._style(ax, "Binary variables and missingness indicators: effective weight before → after re-deformation",
              "effective weight (% of Σ w)", "")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(fontsize=7, frameon=False, ncol=4, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.45, -0.06))
    return fig


# --------------------------------------------------------------------------- report
def _ci(v, lo, hi, nd: int = 4) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def render(out: dict, res: dict, sites: list[str], dev: str, t: str) -> str:
    ext = [s for s in sites if s != dev]
    sec = [box("<b>Exploratory, post hoc.</b> Added on 2026-10-28 at the lead's request, after the transport analyses; "
               "not pre-registered, no decision rule. The frozen INSPIRE metric (shapes g<sub>j</sub>, weights "
               "w<sub>j</sub>, α, k) is kept and only its case base is swapped for each site's own patients (the pool "
               "of the transport stage; same test set, labels and k). Each variable's values (INSPIRE's bins, "
               "categories, 0 / 1) are <b>points</b> in the metric. <b>ICE</b>: test stays moved along every value "
               "through the frozen geometry (blind where g<sub>j</sub> is flat). <b>Re-deformation</b>: the points of "
               "one variable freed and fitted to the site (case base and queries alike, everything else frozen), by "
               f"gradient on a smoothed version of the same kNN, patient-level leave-one-out, ridge {RIDGE:g} towards "
               "INSPIRE's points; evaluated with the real kNN on the test set. Continuous variables get a separate "
               "'missing' value (INSPIRE imputes the median). <b>INSPIRE is the control</b>: the same fits at home "
               "(in-sample: the metric was read from all of INSPIRE). The class components of a point are anchored to "
               "INSPIRE's by the ridge (distances alone do not fix them), so read them as directions, not as "
               "calibrated log-odds.")]
    J = {s: res[s]["joint"] for s in sites}
    rows = []
    for s in sites:
        j = J[s]
        rows.append([SITES.get(s, s) + (" (control)" if s == dev else ""),
                     f"{j['frozen']['logloss']:.4f} [{j['frozen']['ci_low']:.4f}, {j['frozen']['ci_high']:.4f}]",
                     f"{j['joint']['logloss']:.4f} [{j['joint']['ci_low']:.4f}, {j['joint']['ci_high']:.4f}]",
                     _ci(j["joint_minus_frozen"]["mean"], j["joint_minus_frozen"]["ci_low"],
                         j["joint_minus_frozen"]["ci_high"]),
                     f"{j['local']['logloss']:.4f}" if "local" in j else "—",
                     _ci(j["frozen_minus_local"]["mean"], j["frozen_minus_local"]["ci_low"],
                         j["frozen_minus_local"]["ci_high"]) if "local" in j else "—",
                     _ci(j["joint_minus_local"]["mean"], j["joint_minus_local"]["ci_low"],
                         j["joint_minus_local"]["ci_high"]) if "local" in j else "—",
                     f"{j['sum_single_gains']:+.4f}"])
    sec.append("<h2>All variables re-deformed at once</h2>" + table(
        ["Site", "frozen metric, local case base", "all variables adapted", "adapted − frozen",
         "PERISIM refitted locally (A(iii))", "frozen − refit", "adapted − refit", "Σ single-variable gains"], rows))
    sec.append("<h2>Which variables drift</h2>" + fig_img(gain_figure(res, sites, dev)))
    sec.append(fig_img(drift_map_figure(res, sites, dev)))
    V = {(r["site"], r["variable"]): r for s in sites for r in res[s]["variables"]}
    names = sorted({r["variable"] for r in res[dev]["variables"]},
                   key=lambda n: min(V[(s, n)]["d_logloss"] for s in ext) if ext else 0)
    hdr = ["Variable", "class", "INSPIRE w"] + [f"{SITES.get(s, s)}: {x}" for s in sites
                                                for x in ("Δ log-loss", "weight frozen → adapted", "shape change",
                                                          "split-half r")]
    body = []
    for n in names:
        r0 = V[(dev, n)]
        row = [n, r0["variable_class"], f"{r0['w_inspire']:.1%}"]
        for s in sites:
            r = V[(s, n)]
            row += [_ci(r["d_logloss"], r["d_ci_low"], r["d_ci_high"]),
                    f"{r['w_site_frozen']:.1%} → {r['w_site_adapted']:.1%}", r["shape_change"], r["split_half_r"]]
        body.append(row)
    sec.append(box("<b>Δ log-loss</b>: test log-loss with this variable re-deformed minus the frozen metric (negative "
                   "= the site wants this variable different). <b>Weight</b>: effective weight of the variable at the "
                   "site (mean squared distance between two random stays of the site's case base, % of Σ w), with "
                   "INSPIRE's points (a different population already changes it) and with the adapted points. "
                   "<b>Shape change</b>: 0 = the same shape, only stretched; 1 = unrelated. <b>Split-half r</b>: "
                   "agreement of the deformation fitted on two patient halves (low = not reproducible)."))
    sec.append(table(hdr, body, nd=2))
    sec.append("<h2>Binary variables and missingness indicators</h2>" + fig_img(binary_figure(res, sites, dev)))
    sec.append("<h2>Shaped variables, one figure each</h2>" + box(
        "Top: the points of every value per component (dashed black: INSPIRE's frozen points; solid: adapted to each "
        "site, thin: the two patient halves; diamonds at the right: the 'missing' value). Middle: ICE mean per class "
        "through the frozen metric (dashed, band 10–90 % of the stays) and with this variable re-deformed (solid). "
        "Bottom: where each site's case base falls on INSPIRE's grid.") + "".join(
        f"<p><img alt='{n}' src='../results/posthoc_drift/shapes/{t}_drift_{n}.png'/></p>"
        for n in names if V[(dev, n)]["kind"] not in ("binary", "missing")))
    return page("PERISIM post-hoc drift analysis", sec, "Exploratory · not pre-registered · aggregates only")


# --------------------------------------------------------------------------- driver
def run(pipe, force: bool = False) -> int:
    cfg, ws = pipe.cfg, pipe.ws
    dev = cfg.experiment["cohorts"]["development"]
    externals = [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()]
    targets = list(cfg.targets_to_model)
    for t in targets:
        if not (ws.predictions("external", t) / "final_model.pkl").exists():
            raise SystemExit("drift: run the external stage first (no final model)")
    before = frozen_hashes(ws, targets)
    args = {"names": pipe.feature_names(), "dev": dev, "externals": externals}
    out: dict = {"note": "Exploratory, post hoc (2026-10-28): not pre-registered.", "targets": {}}
    rd = ws.root / "results" / "posthoc_drift"
    (rd / "shapes").mkdir(parents=True, exist_ok=True)
    for t in targets:
        fm = PT._load(ws.predictions("external", t) / "final_model.pkl")
        enc, w = fm["enc"], np.asarray(fm["w"], float)
        kcfg = cfg.experiment["knn"]
        k = int(kcfg["k_primary"])
        alpha = float(dict(fm["alpha"]).get(k, 1.0))
        look = frozen_points(enc, w)
        if not all(lj["exact"] for lj in look):
            LOG.warning("drift: neutral rank axes quantised to their bins: %s",
                        [lj["name"] for lj in look if not lj["exact"]])
        sites = [dev] + [c for c in externals if _eval_target_for(t, c, cfg) == t]
        key = _cache_key(ws, t)
        res, ins_counts = {}, None
        for s in sites:
            LOG.info("drift: %s/%s", t, s)
            d = site_data(cfg, ws, t, s, args, enc, w, look)
            C = int(1 + max(d["y_pool"].max(), d["y_test"].max()))
            if s == dev:
                ins_counts = [np.bincount(d["codes_all"][:, j], minlength=lj["points"].shape[0]).astype(float)
                              for j, lj in enumerate(look)]
            fits = site_fits(ws.predictions("posthoc", t) / f"drift_{s}.pkl", key, look, d, C, k, alpha, force)
            local = None
            npz = ws.predictions("posthoc", t) / f"{s}_preds.npz"
            if s != dev and npz.exists():
                z = np.load(npz)
                if np.array_equal(z["y_test"], d["y_test"]) and np.array_equal(z["s_test"], d["s_test"]):
                    local = M.row_logloss(z["perisim__local__test"], d["y_test"])
            LOG.info("drift: evaluating %s/%s (test, ICE, bootstrap)", t, s)
            res[s] = evaluate_site(cfg, s, d, fits, look, ins_counts, w, C, k, alpha, local)
        out["targets"][t] = {s: {kk: v for kk, v in r.items() if kk not in ("points", "ice")} for s, r in res.items()}
        for name in ("variables", "points", "ice"):
            _csv([x for s in sites for x in res[s][name]], rd / f"{t}_drift_{name}.csv")
        _csv([{"site": s, **{f"{a}_{b}": v for a, x in res[s]["joint"].items() if isinstance(x, dict)
                             for b, v in x.items()}, "sum_single_gains": res[s]["joint"]["sum_single_gains"]}
              for s in sites], rd / f"{t}_drift_joint.csv")
        PT._save(gain_figure(res, sites, dev), rd / f"{t}_fig_drift_gain.png")
        PT._save(drift_map_figure(res, sites, dev), rd / f"{t}_fig_drift_map.png")
        PT._save(binary_figure(res, sites, dev), rd / f"{t}_fig_drift_binary.png")
        names = (cfg.targets.get(t) or {}).get("labels") or []
        for lj in look:
            if lj["kind"] not in ("binary", "missing"):
                PT._save(variable_figure(lj["name"], res, sites, dev, lj, names),
                         rd / "shapes" / f"{t}_drift_{lj['name']}.png")
        (ws.reports / "posthoc_drift.html").write_text(render(out, res, sites, dev, t), encoding="utf-8")
    after = frozen_hashes(ws, targets)
    check_hashes(before, after)
    out["frozen_hashes"] = {"start": before, "end": after, "unchanged": True}
    write_json(ws.artifacts() / "posthoc_drift.json", out)
    LOG.info("drift: results in %s; report %s", rd, ws.reports / "posthoc_drift.html")
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
    setup_logging(pipe.ws.logs / f"drift_{now_iso().replace(':', '')}.log")
    set_threads(a.threads or os.cpu_count() or 4)
    return run(pipe, force=a.force)


if __name__ == "__main__":
    raise SystemExit(main())
