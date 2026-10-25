"""Transport analyses per (target, external cohort): phase-1 fits (local PERISIM, multi-source PERISIM, per-site
shapes for E) and the analyses A (loss decomposition), B (case-base composition and site weight γ), C (accumulation
curves), D (minimal recalibration), F (eICU by hospital) and the per-cohort hypotheses H1–H5."""

from __future__ import annotations

import logging
import pickle
import shutil

import numpy as np

from perisim.data.io import find_table, load_table, norm_id
from perisim.evaluation import metrics as M
from perisim.experiment import TARGET_CLASSES, _nca, fit_perisim, inner_splits, load_cohort_data
from perisim.external import _eval_target_for
from perisim.transport.core import (
    SEED_OFFSET, assert_disjoint, boot_auroc_difference, boot_cumulative, case_base, concat_ft, gamma_schedule,
    patient_orders, patient_split, point_metrics, predict_gammas, source_labels, target_labels, time_blocks,
)

LOG = logging.getLogger("perisim.transport")


# --------------------------------------------------------------------------- shared context
def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _dump(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def context(cfg, ws, target: str, c: str, args: dict) -> dict:
    """Data, frozen model, labels and the patient-level split of one external cohort (the same in every job)."""
    tc = cfg.experiment["transport"]
    names, dev = args["names"], args["dev"]
    et = _eval_target_for(target, c, cfg)
    C = 2 if et == "t1_bin" else TARGET_CLASSES[target]
    X = load_cohort_data(ws.common(c), ws.derived(c), names)
    seed = int(cfg.seed) + 101 * SEED_OFFSET.get(c, 9)
    split = patient_split(X.ft.subjects, float(tc["test_fraction"]), int(tc["tuning_max_patients"]), seed)
    assert_disjoint(split, X.ft.subjects)
    return {"tc": tc, "et": et, "C": C, "X": X, "seed": seed, "split": split, "names": names, "dev": dev,
            "lab": target_labels(target, et, X, cfg)}


def _fit(cfg, ft, y, C: int, seed: int, args: dict, ck):
    """PERISIM with the same nested procedure and stopping rule as the final development model."""
    d7 = args["d7"]
    st = {k: d7[k] for k in ("W", "P", "n_min", "n_max", "min_delta", "tol_delta") if k in d7}
    ok = np.where(y >= 0)[0]
    fp = fit_perisim(cfg, ft.take(ok), y[ok], C, seed=seed, threads=int(args["threads"]), stopping_override=st,
                     exclude_families=set(d7.get("exclude_families", [])), early_stop=True,
                     need_comparator_weights=False, checkpoint_dir=ck)
    shutil.rmtree(ck, ignore_errors=True)
    return fp


# --------------------------------------------------------------------------- phase 1
def fit_local(cfg, ws, target: str, c: str, args: dict) -> dict:
    """A(iii): full PERISIM fitted on the local pool of `c` (its own shapes and weights)."""
    ctx = context(cfg, ws, target, c, args)
    pool = ctx["split"]["pool"]
    out = ws.predictions("transport", target) / f"{c}_local_model.pkl"
    LOG.info("transport: local PERISIM on the pool of %s (%s, %d stays)", c, ctx["et"], pool.size)
    fp = _fit(cfg, ctx["X"].ft.take(pool), ctx["lab"]["own"][pool], ctx["C"], ctx["seed"], args,
              out.with_name(f"{c}_local_ckpt"))
    _dump({"enc": fp.enc, "w": fp.w, "knn": fp.knn, "lambda": fp.info["lambda"], "n_accepted": fp.loop["n_accepted"],
           "stop_reason": fp.loop["stop_reason"], "weights": dict(zip(fp.enc.feature_names, fp.w.tolist()))}, out)
    return {"file": str(out)}


def fit_multisource(cfg, ws, target: str, c: str, args: dict) -> dict:
    """PERISIM refitted on INSPIRE ∪ the other development cohorts (each with its own cut-points)."""
    tc = cfg.experiment["transport"]
    dev, names = args["dev"], args["names"]
    et = _eval_target_for(target, c, cfg)
    C = 2 if et == "t1_bin" else TARGET_CLASSES[target]
    D = load_cohort_data(ws.common(dev), ws.derived(dev), names)
    ft, y = D.ft, source_labels(target, et, D, cfg)["own"]
    for o in [x for x in (tc.get("multisource") or {}).get("development", []) if x != dev]:
        O = load_cohort_data(ws.common(o), ws.derived(o), names)
        ft, y = concat_ft(ft, O.ft), np.concatenate([y, source_labels(target, et, O, cfg)["own"]])
    out = ws.predictions("transport", target) / f"{c}_multisource_model.pkl"
    LOG.info("transport: PERISIM refitted on %s (n=%d) for %s", (tc.get("multisource") or {}).get("development"),
             int((y >= 0).sum()), c)
    fp = _fit(cfg, ft, y, C, int(cfg.seed), args, out.with_name(f"{c}_multisource_ckpt"))
    _dump({"enc": fp.enc, "w": fp.w, "lambda": fp.info["lambda"], "n_train": int((y >= 0).sum()),
           "n_accepted": fp.loop["n_accepted"], "stop_reason": fp.loop["stop_reason"],
           "weights": dict(zip(fp.enc.feature_names, fp.w.tolist()))}, out)
    return {"file": str(out)}


def fit_site_shapes(cfg, ws, target: str, site: str, args: dict) -> dict:
    """E: reduced consensus (same model space and procedure, `shapes.n_models` models) in one site, on INSPIRE's
    FROZEN value grid, so the curves of the sites are comparable value by value. Returns per variable the plain mean
    curve over the competitive models and its standard error (values × classes)."""
    from perisim.importance.loop import run_importance_loop

    tc = cfg.experiment["transport"]
    fm = _load(ws.predictions("external", target) / "final_model.pkl")
    enc, gof, gnames = fm["enc"], np.asarray(fm["group_of_feature"]), list(fm["group_names"])
    X = load_cohort_data(ws.common(site), ws.derived(site), args["names"])
    et = _eval_target_for(target, site, cfg) if site != args["dev"] else target
    C = 2 if et == "t1_bin" else TARGET_CLASSES[target]
    y = (source_labels(target, et, X, cfg) if site == args["dev"] else target_labels(target, et, X, cfg))["own"]
    ok = np.where(y >= 0)[0]
    ft, yy = X.ft.take(ok), y[ok]
    Xm, owner = enc.transform_model(ft)
    seed = int(cfg.seed) + 101 * SEED_OFFSET.get(site, 9)
    a, b = inner_splits(yy, ft.subjects, int(cfg.experiment["cv"]["inner_folds"]), seed)[0]
    nm = int(tc["shapes"]["n_models"])
    out = ws.predictions("transport", target) / f"shapes_{site}.pkl"
    LOG.info("transport: shapes of %s on INSPIRE's grid (%d models, n=%d)", site, nm, ok.size)
    res = run_importance_loop(X_model=Xm, owner=owner, col_group=gof[owner], n_groups=len(gnames), group_names=gnames,
                              y=yy, n_classes=C, inner_train=a, inner_val=b, enc=enc, ft=ft, group_of_feature=gof,
                              subjects=ft.subjects, cfg=cfg, seed=seed, threads=int(args["threads"]),
                              stopping_override={"n_max": nm, "n_min": nm, "min_delta": None}, early_stop=False,
                              checkpoint=out.with_name(f"shapes_{site}_ckpt.pkl"))
    shapes = {}
    for j, cl in (res.get("curves_kept") or {}).items():
        cl = [np.asarray(x, dtype=float) for x in cl if x is not None]
        if len(cl) >= 3:
            S = np.stack(cl)
            shapes[enc.blocks[j].name] = {"mean": S.mean(0), "se2": S.var(0, ddof=1) / len(cl), "n_models": len(cl)}
    _dump({"site": site, "eval_target": et, "C": C, "n": int(ok.size), "shapes": shapes,
           "n_competitive": res["n_competitive_final"]}, out)
    out.with_name(f"shapes_{site}_ckpt.pkl").unlink(missing_ok=True)
    return {"file": str(out)}


# --------------------------------------------------------------------------- phase 2
def cohort_analyses(cfg, ws, target: str, c: str, args: dict) -> dict:
    """A, B, C, D (and F for eICU) for one external cohort, plus the per-cohort hypotheses H1–H5."""
    ctx = context(cfg, ws, target, c, args)
    tc, et, C, X, split, lab = ctx["tc"], ctx["et"], ctx["C"], ctx["X"], ctx["split"], ctx["lab"]
    kcfg = cfg.experiment["knn"]
    kp, kg = int(kcfg["k_primary"]), [int(k) for k in kcfg["k_grid"]]
    nb = int(cfg.experiment["evaluation"]["n_bootstrap"])
    fm = _load(ws.predictions("external", target) / "final_model.pkl")
    enc, alpha = fm["enc"], dict(fm["alpha"])
    wd = enc.weights_dims(fm["w"])
    dev = args["dev"]
    D = load_cohort_data(ws.common(dev), ws.derived(dev), args["names"])
    others = [o for o in args["externals"] if o != c]
    O = load_cohort_data(ws.common(others[0]), ws.derived(others[0]), args["names"]) if others else None
    labD = source_labels(target, et, D, cfg)
    labO = source_labels(target, et, O, cfg) if O is not None else None
    E_X, E_D = enc.transform(X.ft), enc.transform(D.ft)
    E_O = enc.transform(O.ft) if O is not None else None
    N_X = enc.transform_neutral(X.ft)
    nd = enc.neutral_weights_dims(np.ones(enc.n_features))
    te, tu, pool = split["test"], split["tuning"], split["pool"]
    sX, sD = X.ft.subjects.astype(str), D.ft.subjects.astype(str)
    sO = O.ft.subjects.astype(str) if O is not None else None
    gam = {f"{g:g}": float(g) for g in tc["gamma"]}
    n0 = float(tc["gamma_schedule_n0"])
    res: dict = {"cohort": c, "eval_target": et, "other": others[0] if others else None,
                 "sizes": {k: {"stays": int(v.size), "patients": int(np.unique(sX[v]).size)} for k, v in split.items()},
                 "A": [], "B": [], "C": [], "D": [], "F": [], "hypotheses": {}, "calibration_curves": {}}

    local_fit = ws.predictions("transport", target) / f"{c}_local_model.pkl"
    fpL = _load(local_fit) if local_fit.exists() else None
    ms_fit = ws.predictions("transport", target) / f"{c}_multisource_model.pkl"
    fpS = _load(ms_fit) if ms_fit.exists() else None

    def part(E, y, s, idx, local):
        return (E[idx], y[idx], s[idx], local) if idx is not None else (E, y, s, local)

    for name, y in lab.items():
        q = te[y[te] >= 0]
        if q.size < 20:
            LOG.warning("transport: only %d evaluable test stays in %s/%s/%s → skipped", q.size, c, target, name)
            continue
        Eq, sq, yq = E_X[q], sX[q], y[q]
        primary = name == "own"
        loc = part(E_X, y, sX, pool, True)
        inspire = (E_D, labD[name], sD, False)
        other = (E_O, labO[name], sO, False) if O is not None and (labO[name] >= 0).any() else None
        rows_b, preds = [], {}

        def add_b(comp, kn, gammas, n_cases, E_query=None):
            if kn is None:
                LOG.warning("transport: %s not usable in %s/%s/%s", comp, c, target, name)
                return {}
            P, ood = predict_gammas(kn, Eq if E_query is None else E_query, sq, kp, gammas)
            for g, Pg in P.items():
                rows_b.append({"labels": name, "composition": comp, "gamma": g, "n_case_base": n_cases,
                               **point_metrics(Pg, yq, et, ood)})
            return {"P": P, "ood": ood}

        # ---- B: case-base composition (frozen metric) and site weight γ
        kn_loc = case_base([loc], wd, C, kcfg, kg, alpha)
        preds["local"] = add_b("local", kn_loc, {"1": 1.0}, int(pool.size))
        preds["inspire_only"] = add_b("inspire_only", case_base([inspire], wd, C, kcfg, kg, alpha), {"1": 1.0},
                                      int(len(D)))
        g_full = {**gam, "schedule": gamma_schedule(np.unique(sX[pool]).size, n0)}
        add_b("inspire+local", case_base([inspire, loc], wd, C, kcfg, kg, alpha), g_full, int(len(D) + pool.size))
        if other is not None:
            add_b("inspire+other+local", case_base([inspire, other, loc], wd, C, kcfg, kg, alpha), g_full,
                  int(len(D) + len(O) + pool.size))
        add_b("local_euclidean", case_base([part(N_X, y, sX, pool, True)], nd, C, kcfg, [kp]), {"1": 1.0},
              int(pool.size), E_query=N_X[q])
        yl = y[pool][y[pool] >= 0]
        prior = np.bincount(yl, minlength=C) / max(yl.size, 1)
        rows_b.append({"labels": name, "composition": "local_prevalence", "gamma": "1", "n_case_base": int(pool.size),
                       **point_metrics(np.tile(prior, (q.size, 1)), yq, et)})
        if fpS is not None:      # metric refitted on INSPIRE ∪ eICU (MIMIC-IV only)
            ft_S = concat_ft(D.ft, O.ft) if O is not None else D.ft
            y_S = np.concatenate([labD[name], labO[name]]) if O is not None else labD[name]
            wS = fpS["enc"].weights_dims(fpS["w"])
            ES, ESx = fpS["enc"].transform(ft_S), fpS["enc"].transform(X.ft)
            sS = ft_S.subjects.astype(str)
            EqS = ESx[q]
            for comp, parts in (("S_transport", [(ES, y_S, sS, False)]), ("S_local", [part(ESx, y, sX, pool, True)]),
                                ("S_pooled", [(ES, y_S, sS, False), part(ESx, y, sX, pool, True)])):
                kn = case_base(parts, wS, C, kcfg, kg, alpha)
                if kn is not None:
                    P, ood = predict_gammas(kn, EqS, sq, kp, {"1": 1.0})
                    rows_b.append({"labels": name, "composition": comp, "gamma": "1",
                                   "n_case_base": int(sum(len(p[1]) for p in parts)), **point_metrics(P["1"], yq, et, ood)})
        res["B"] += rows_b
        if not primary or not preds.get("local") or not preds.get("inspire_only"):
            continue
        P_L, ood_L = preds["local"]["P"]["1"], preds["local"]["ood"]
        P_A, ood_A = preds["inspire_only"]["P"]["1"], preds["inspire_only"]["ood"]
        res["calibration_curves"] = {
            mode: [{"threshold": t, "points": M.calibration_curve(Pm[:, t:].sum(1), (yq >= t).astype(float))}
                   for t in range(1, C)] for mode, Pm in (("local", P_L), ("mode_A", P_A))}

        # ---- A: decomposition of the transport loss (same local case base, same test)
        A = {"A(i) frozen INSPIRE metric": P_L}
        ok = pool[y[pool] >= 0]
        nca = _nca(cfg, enc, C, ctx["seed"])
        prior_w = fm["w"] / max(float(np.sum(fm["w"])), 1e-12)
        nca.fit(E_X[ok], y[ok], prior=prior_w, groups=np.asarray(fm["group_of_feature"]), lam=float(fm["lambda"]),
                epochs=int(cfg.experiment["metric"]["nca"]["epochs"]), subjects=sX[ok])
        kn2 = case_base([loc], enc.weights_dims(nca.weights), C, kcfg, kg)          # α fitted on the case base
        if kn2 is not None:
            A["A(ii) local NCA, INSPIRE prior"] = predict_gammas(kn2, Eq, sq, kp, {"1": 1.0})[0]["1"]
        if fpL is not None:
            A["A(iii) local PERISIM"] = predict_gammas(_with_local(fpL["knn"]), fpL["enc"].transform(X.ft.take(q)), sq,
                                                       kp, {"1": 1.0})[0]["1"]
        A["mode A (INSPIRE case base)"] = P_A
        for meth, P in A.items():
            ll = M.row_logloss(P, yq)
            ci = M.bootstrap_rowwise({"ll": ll}, sq, nb, cfg.seed)["ll"]
            res["A"].append({"variant": meth, **point_metrics(P, yq, et), "logloss_ci_low": ci["ci_low"],
                             "logloss_ci_high": ci["ci_high"]})
        keys = list(A)
        for a_, b_ in ((keys[0], "A(iii) local PERISIM"), (keys[0], "A(ii) local NCA, INSPIRE prior"),
                       ("A(ii) local NCA, INSPIRE prior", "A(iii) local PERISIM")):
            if a_ in A and b_ in A:
                d = M.paired_difference(M.row_logloss(A[a_], yq), M.row_logloss(A[b_], yq), sq, nb, cfg.seed)
                d.pop("_boot", None)
                res["A"].append({"variant": f"Δ log-loss: {a_} − {b_}", "logloss": d["mean"],
                                 "logloss_ci_low": d["ci_low"], "logloss_ci_high": d["ci_high"]})

        # ---- hypotheses H1–H4 (this cohort, primary labelling)
        H = res["hypotheses"]
        d1, d2, d3 = (float(tc["margins"][k]) for k in ("citl", "slope", "auroc"))
        cal = {mode: [{"threshold": t, "citl": boot_cumulative(Pm, yq, sq, t, "citl", nb, cfg.seed),
                       "slope": boot_cumulative(Pm, yq, sq, t, "slope", nb, cfg.seed)} for t in range(1, C)]
               for mode, Pm in (("local", P_L), ("mode_A", P_A))}
        inside = all(abs(x["citl"]["ci_low"]) <= d1 and abs(x["citl"]["ci_high"]) <= d1 for x in cal["local"])
        beyond = any(x["citl"]["ci_low"] > d1 or x["citl"]["ci_high"] < -d1 for x in cal["mode_A"])
        H["H1"] = {"holds": bool(inside and beyond), "local_inside_margin": bool(inside), "modeA_beyond_margin": bool(beyond),
                   "margin": d1, "detail": cal}
        H["H2"] = {"holds": bool(all(x["slope"]["ci_low"] >= 1 - d2 for x in cal["local"])), "margin": d2,
                   "slopes_local": [x["slope"] for x in cal["local"]]}
        if "A(iii) local PERISIM" in A:
            dA = boot_auroc_difference(P_L, A["A(iii) local PERISIM"], yq, sq, nb, cfg.seed)
            H["H3"] = {"holds": bool(dA["ci_low"] > -d3), "margin": d3, "auroc_difference_frozen_minus_local": dA}
        dO = M.paired_difference(ood_A, ood_L, sq, nb, cfg.seed)
        dO.pop("_boot", None)
        H["H4"] = {"holds": bool(dO["ci_low"] > 0), "ood_modeA_minus_local": dO}

        # ---- C: accumulation curves
        res["C"] = _accumulation(cfg, ws, c, tc, et, C, kcfg, kg, kp, alpha, wd, gam, n0, E_X, y, sX, pool, Eq, sq, yq,
                                 inspire, other, X.ft.ids)
        res["hypotheses"]["H5"] = _h5(res["C"])

        # ---- D: minimal recalibration without touching the geometry
        res["D"] = _recalibration(cfg, tc, et, C, kcfg, kg, kp, alpha, kn_loc, E_X, y, sX, tu, Eq, sq, yq, ctx["seed"])

        # ---- F: eICU by hospital
        if c == "eicu" and args.get("eicu_src"):
            res["F"] = _hospitals(cfg, ws, args, tc, et, C, kcfg, kg, kp, alpha, wd, gam, n0, X, E_X, y, sX, inspire,
                                  other, ctx["seed"])
    return res


def _with_local(kn):
    kn.is_local = np.ones(kn.Ew.shape[0], dtype=bool)
    return kn


def _accumulation(cfg, ws, c, tc, et, C, kcfg, kg, kp, alpha, wd, gam, n0, E_X, y, sX, pool, Eq, sq, yq, inspire,
                  other, ids, reps=None, orders=None) -> list[dict]:
    """C: local cases added in batches (patients), for every start and γ; α frozen; k_eff = min(k, case base)."""
    acc = tc["accumulation"]
    reps = int(acc["repetitions"]) if reps is None else reps
    ps = sX[pool]
    n_pat = np.unique(ps).size
    grid = sorted({n_pat if v == "all" else min(int(v), n_pat) for v in acc["n_local"]})
    blocks = time_blocks(ws.common(c), ids[pool]) if ws is not None else None
    allowed = orders or acc.get("orders", ["blocks", "random"])
    ords = {k: v for k, v in patient_orders(ps, blocks, reps, int(cfg.seed) + 7).items() if k in allowed}
    starts = {"empty": [], "inspire": [inspire]}
    if other is not None:
        starts["inspire+other"] = [inspire, other]
    n_ood = int(acc.get("ood_repetitions", reps))
    rows = []
    for order, perms in ords.items():
        for r, perm in enumerate(perms):
            for n in grid:
                idx = pool[np.isin(ps, perm[:n])]
                local = (E_X[idx], y[idx], sX[idx], True)
                for start, parts in starts.items():
                    base = {"order": order, "rep": r, "start": start, "n_local": int(n)}
                    if start == "empty" and n == 0:
                        rows.append({**base, "gamma": "1", **point_metrics(np.full((len(yq), C), 1.0 / C), yq, et)})
                        continue
                    kn = case_base(parts + ([local] if n else []), wd, C, kcfg, kg, alpha, with_ood=r < n_ood)
                    if kn is None:
                        continue
                    gs = {"1": 1.0} if start == "empty" else {**gam, "schedule": gamma_schedule(n, n0)}
                    P, ood = predict_gammas(kn, Eq, sq, kp, gs)
                    for g, Pg in P.items():
                        rows.append({**base, "gamma": g, **point_metrics(Pg, yq, et, ood)})
    return rows


def _h5(rows: list[dict]) -> dict:
    """H5 on the random order: (a) a crossing — the empty start reaches the INSPIRE start (γ = 1); (b) at n_local ∈
    {50, 100} the INSPIRE start with the declared schedule beats the empty start over the repetitions."""
    rr = [r for r in rows if r["order"] == "random"]
    if not rr:
        return {"holds": None, "note": "no random-order accumulation"}

    def ll(start, gamma, n):
        return {r["rep"]: r["logloss"] for r in rr if r["start"] == start and r["gamma"] == gamma and r["n_local"] == n}

    ns = sorted({r["n_local"] for r in rr if r["n_local"] > 0})
    crossing = None
    for n in ns:
        e, i1 = ll("empty", "1", n), ll("inspire", "1", n)
        if e and i1 and np.mean(list(e.values())) <= np.mean(list(i1.values())):
            crossing = n
            break
    early = {}
    for n in (50, 100):
        e, s = ll("empty", "1", n), ll("inspire", "schedule", n)
        common = sorted(set(e) & set(s))
        if common:
            d = np.array([s[k] - e[k] for k in common])
            early[str(n)] = {"mean": float(d.mean()), "low": float(np.percentile(d, 2.5)),
                             "high": float(np.percentile(d, 97.5)), "reps": len(common)}
    b = bool(early) and all(v["high"] < 0 for v in early.values())
    return {"holds": bool(crossing is not None and b), "crossing_n_local": crossing,
            "schedule_minus_empty": early}


def _recalibration(cfg, tc, et, C, kcfg, kg, kp, alpha, kn_loc, E_X, y, sX, tu, Eq, sq, yq, seed) -> list[dict]:
    """D: α (and k) chosen on local tuning sets of growing size, the frozen metric and the local case base; logistic
    recalibration (intercept + slope on logit p per class) as a sensitivity only."""
    if kn_loc is None:
        return []
    rows = []
    Dq, Iq, vq = kn_loc.neighbors(Eq, sq)
    a0 = kn_loc.alpha.get(kp, 1.0)
    P0 = kn_loc.probs(Iq, vq, min(kp, kn_loc.k_max), alpha=a0)
    rows.append({"method": "none (frozen α, k)", "tuning_patients": 0, "k": kp, "alpha": a0, **point_metrics(P0, yq, et)})
    tu = tu[y[tu] >= 0]
    pats = np.random.default_rng(seed + 11).permutation(np.unique(sX[tu]))
    for size in [int(s) for s in tc["tuning_sizes"] if int(s) <= pats.size]:
        sel = tu[np.isin(sX[tu], pats[:size])]
        Dt, It, vt = kn_loc.neighbors(E_X[sel], sX[sel])
        yt = y[sel]

        def ll_t(k, a):
            return float(M.row_logloss(kn_loc.probs(It, vt, min(k, kn_loc.k_max), alpha=a), yt).mean())

        a1 = min(kcfg["alpha_grid"], key=lambda a: ll_t(kp, a))
        rows.append({"method": "α tuned", "tuning_patients": size, "k": kp, "alpha": a1,
                     **point_metrics(kn_loc.probs(Iq, vq, min(kp, kn_loc.k_max), alpha=a1), yq, et)})
        k2, a2 = min(((k, a) for k in kg for a in kcfg["alpha_grid"]), key=lambda ka: ll_t(*ka))
        rows.append({"method": "α and k tuned", "tuning_patients": size, "k": k2, "alpha": a2,
                     **point_metrics(kn_loc.probs(Iq, vq, min(k2, kn_loc.k_max), alpha=a2), yq, et)})
        Pt = kn_loc.probs(It, vt, min(kp, kn_loc.k_max), alpha=a0)
        P = P0.copy()
        for cc in range(C):
            yb = (yt == cc).astype(float)
            if 0 < yb.sum() < len(yb):
                a, b = M.calibration_line(Pt[:, cc], yb)
                lg = np.log(np.clip(P0[:, cc], 1e-6, 1 - 1e-6) / (1 - np.clip(P0[:, cc], 1e-6, 1 - 1e-6)))
                P[:, cc] = 1 / (1 + np.exp(-np.clip(a + b * lg, -500, 500)))
        P = P / P.sum(1, keepdims=True)
        rows.append({"method": "logistic recalibration (sensitivity)", "tuning_patients": size, "k": kp, "alpha": a0,
                     **point_metrics(P, yq, et)})
    return rows


def hospital_map(src_dir: str, ids: np.ndarray, cache_dir) -> np.ndarray | None:
    """eICU hospital of each stay, read from the raw `patient` table (not part of the extraction: adding it there
    would re-key the extraction and the development stages)."""
    p = find_table(src_dir, "patient") if src_dir else None
    if p is None:
        return None
    df = load_table(p, {"unit": "patientunitstayid", "hospital": "hospitalid"}, None, cache_dir)
    lut = dict(zip(df["unit"].cast(str).to_list(), df["hospital"].cast(str).to_list()))
    raw = [norm_id(str(s).split("-", 1)[-1]) for s in np.asarray(ids).astype(str)]
    return np.array([lut.get(r) for r in raw], dtype=object)


def _hospitals(cfg, ws, args, tc, et, C, kcfg, kg, kp, alpha, wd, gam, n0, X, E_X, y, sX, inspire, other,
               seed) -> list[dict]:
    """F: B and C with the case base and test of each large eICU hospital (≥ `hospitals.min_index_stays`)."""
    hosp = hospital_map(args["eicu_src"], X.ft.ids, ws.cache("eicu"))
    if hosp is None:
        LOG.warning("transport: eICU hospital not available → F skipped")
        return []
    lab_ok = y >= 0
    vals, counts = np.unique(hosp[lab_ok & (hosp != None)].astype(str), return_counts=True)  # noqa: E711
    big = [v for v, n in sorted(zip(vals, counts), key=lambda t: -t[1]) if n >= int(tc["hospitals"]["min_index_stays"])]
    rows = []
    for rank, h in enumerate(big, start=1):
        idx = np.where((hosp.astype(str) == h) & lab_ok)[0]
        sp = patient_split(sX[idx], float(tc["test_fraction"]), 0, seed + rank)
        te, pool = idx[sp["test"]], idx[sp["pool"]]
        if te.size < 20 or pool.size < 20:
            continue
        Eq, sq, yq = E_X[te], sX[te], y[te]
        loc = (E_X[pool], y[pool], sX[pool], True)
        label = f"H{rank}"
        g_full = {**gam, "schedule": gamma_schedule(np.unique(sX[pool]).size, n0)}
        for comp, parts, gs in (("local", [loc], {"1": 1.0}), ("inspire_only", [inspire], {"1": 1.0}),
                                ("inspire+local", [inspire, loc], g_full),
                                ("inspire+other+local", [inspire, other, loc] if other is not None else None, g_full)):
            if parts is None:
                continue
            kn = case_base(parts, wd, C, kcfg, kg, alpha)
            if kn is None:
                continue
            P, ood = predict_gammas(kn, Eq, sq, kp, gs)
            for g, Pg in P.items():
                rows.append({"hospital": label, "n_index_stays": int(idx.size), "analysis": "B", "composition": comp,
                             "gamma": g, **point_metrics(Pg, yq, et, ood)})
        acc = _accumulation(cfg, None, "eicu", tc, et, C, kcfg, kg, kp, alpha, wd, gam, n0, E_X, y, sX, pool, Eq, sq,
                            yq, inspire, other, X.ft.ids, orders=["random"])
        rows += [{"hospital": label, "n_index_stays": int(idx.size), "analysis": "C", **r} for r in acc]
    return rows
