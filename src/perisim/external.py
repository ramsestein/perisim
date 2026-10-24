"""External validation (§6.2) — only after the freeze (§6.3).

Final model: PERISIM fitted on ALL of INSPIRE with the same nested procedure. The final models of the
targets (T1, T2, T3) are independent, so they run in parallel processes (at most `jobs` × `threads`).
- Mode A (full transport): weights, VDM tables and INSPIRE case base; queries from the target cohort.
- Mode B (metric transport): INSPIRE weights and scales, case base of the target cohort, patient-level LOO
  within the target cohort (the smoothing prior also excludes the query's patient). B-VDM variant: VDM tables
  recomputed in the target cohort with patient-level CV.
- eICU: T1 reduced to severe / not severe (declared as a reduced target).
- T2/T3 (AUDIT C4, decided 2026-10-22): the target cohort's own tertiles are CROSS-FITTED — the cohort is split
  into two patient-level halves and the stays of each half are labelled with the cut-points of the other half
  (mode B case bases use the same labels, so a query never influences its own categories). Absolute categories
  (≤ 3 / 4–7 / > 7 d) are a co-primary analysis: the INSPIRE case base is relabelled with the same absolute
  categories (same learned metric) and the logistic regression is refitted on them.
- Sensitivity: INSPIRE percentile discretization applied to MIMIC-IV / eICU (§4.1).
- §5.6 per-patient output for every external query (mode A, primary analysis), inside the pod only (E14).
- Geometry shape (v0.5): modes A and B transport the weights AND the consensus shapes of INSPIRE. The B-VDM variant
  exists only for the HVDM geometry (re-estimating the shapes in the target cohort would require training the
  consensus models there; declared). Comparators: unsupervised Euclidean kNN (neutral axes), trivial prevalence
  (INSPIRE's, mode A; the target cohort's patient-level LOO, mode B), logistic regression and SWIFT.
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import logging
import multiprocessing as mp
import traceback

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from perisim.baselines import parametric
from perisim.evaluation.html import box, page, table
from perisim.evaluation.report import eval_sections, evaluate_predictions, p4_metrics, paired_between
from perisim.experiment import SEVERE, TARGET_CLASSES, fit_perisim, knn_predict, labels_for, load_cohort_data, swift_inputs
from perisim.knn.predict import KNNModel, merge_patient_outputs, patient_pieces
from perisim.targets.build import t2_categories, t3_labels
from perisim.utils import read_jsonl, set_threads, write_json, write_jsonl, write_parquet

LOG = logging.getLogger("perisim.external")


def _pred_frame(ids, subj, y, method, k, P) -> pl.DataFrame:
    d = {"stay_id": ids.astype(str), "subject_id": subj.astype(str), "y": y.astype(np.int64),
         "method": [method] * len(y), "k": np.full(len(y), int(k), dtype=np.int64)}
    for c in range(P.shape[1]):
        d[f"p{c}"] = P[:, c].astype(float)
    return pl.DataFrame(d)


def discretize_like(ft_ext, ft_dev):
    """Apply INSPIRE's percentile discretization (steps of 5, clipped at p2.5–p97.5) to the target cohort."""
    grid_q = np.array([0.025] + [i / 20 for i in range(1, 20)] + [0.975])
    out = copy.copy(ft_ext)
    out.num = dict(ft_ext.num)
    for s in ft_ext.specs:
        if s.type == "continuous" and (s.name.startswith("last_") or s.name.startswith("preop_")):
            ref = ft_dev.num[s.name]
            ref = ref[np.isfinite(ref)]
            if ref.size < 20:
                continue
            g = np.unique(np.quantile(ref, grid_q))
            x = ft_ext.num[s.name]
            ok = np.isfinite(x)
            idx = np.abs(x[ok, None] - g[None, :]).argmin(1)
            v = x.copy()
            v[ok] = g[idx]
            out.num[s.name] = v
    return out


def _eval_target_for(target: str, cohort: str, cfg) -> str:
    reduced = cohort in set(cfg.targets["t1"].get("reduced_cohorts") or [])
    if target == "t1" and reduced:
        return "t1_bin"
    return target


def absolute_labels(target: str, data, tcfg: dict) -> np.ndarray:
    """T2 / T3 with the absolute cut-points (≤ a / (a, b] / > b days): no threshold is estimated from data."""
    a, b = (float(x) for x in tcfg["t2"]["absolute_cut_days"])
    if target == "t2":
        return t2_categories(data.los, (a, b), data.death)
    y = t3_labels(data.t1, data.los, (a, b))
    y[data.t1 < 0] = -1
    return y


def patient_halves(subjects: np.ndarray, seed: int) -> np.ndarray:
    """Patient-level random split of a cohort into two halves (0 / 1) for cross-fitted cut-points (C4)."""
    u = np.unique(subjects.astype(str))
    perm = np.random.default_rng(seed).permutation(u.size)
    half = dict(zip(u[perm], (np.arange(u.size) % 2)))
    return np.array([half[s] for s in subjects.astype(str)], dtype=np.int64)


def run_external(pipe) -> None:
    from perisim.pipeline import _child_threads

    cfg, ws = pipe.cfg, pipe.ws
    dev = cfg.experiment["cohorts"]["development"]
    externals = [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()]
    d7 = pipe.d7()
    targets = list(cfg.targets_to_model)
    common = {"cfg": cfg.to_dict(), "profile": cfg.profile, "base": str(ws.base), "experiment": ws.experiment,
              "names": pipe.feature_names(),
              "d7": d7, "threads": pipe.threads(), "dev": dev, "externals": externals}
    args = [{**common, "target": t} for t in targets]
    par = max(1, min(len(targets), pipe.jobs()))
    LOG.info("external: final models for %s in %d process(es) × %d threads", targets, par, pipe.threads())
    results = []
    if par == 1:
        results = [_external_target(a) for a in args]
    else:
        with _child_threads(pipe.threads()), \
                cf.ProcessPoolExecutor(max_workers=par, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_external_target, a): a["target"] for a in args}
            for fu in cf.as_completed(futs):
                try:
                    results.append(fu.result())
                except Exception as exc:  # worker died (e.g. OOM)
                    results.append({"target": futs[fu], "error": f"process terminated ({exc!r})"})
    errors = [f"{r['target']}: {r['error']}" for r in results if r.get("error")]
    if errors:
        raise RuntimeError("External validation failed:\n" + "\n".join(errors))
    out = {"targets": {r["target"]: r["result"] for r in sorted(results, key=lambda r: targets.index(r["target"]))},
           "note": "External validation with a frozen pipeline (§6.3)."}
    kp = int(cfg.experiment["knn"]["k_primary"])
    for c in externals:   # §5.6 records per external cohort (inside the pod only, E14)
        parts = [read_jsonl(ws.predictions("external", t) / f"{c}_patient.jsonl") for t in targets]
        if any(parts):
            write_jsonl(ws.predictions("external", "patient_output") / f"{c}.jsonl", merge_patient_outputs(parts))
    write_json(ws.artifacts() / "external.json", out)
    (ws.reports / "external.html").write_text(render_external(out, kp, int(cfg.experiment["evaluation"]["n_bootstrap"])),
                                           encoding="utf-8")


def _external_target(args: dict) -> dict:
    """Final model of one target on the development cohort + evaluation in every external cohort."""
    target = args["target"]
    try:
        from perisim.config import Config
        from perisim.utils import setup_logging
        from perisim.workspace import Workspace

        setup_logging()
        set_threads(int(args["threads"]))
        cfg = Config(**args["cfg"], profile=args["profile"], variant="main")
        ws = Workspace(args["base"], experiment=args["experiment"])
        res = _evaluate_target(cfg, ws, target, args)
        import shutil

        shutil.rmtree(ws.predictions("external", target) / "ckpt", ignore_errors=True)
        return {"target": target, "result": res}
    except Exception:
        return {"target": target, "error": traceback.format_exc()}


def _case_base(E_dev, w_dims, y, C, ft, k_grid, kcfg, carried: dict) -> dict | None:
    """INSPIRE case base for mode A with a given labelling (same learned metric); None if unusable."""
    ok = np.where(y >= 0)[0]
    if ok.size < 20 or np.unique(y[ok]).size < 2:
        return None
    sub = ft.take(ok)
    kn = KNNModel(E_dev[ok], w_dims, y[ok], C, sub.subjects, k_grid, ids_ref=sub.ids)
    kn.fit_alpha_and_ood(kcfg["alpha_grid"], int(kcfg["k_primary"]), kcfg["ood_percentile"])
    return {"knn": kn, "ok": ok, "carried": {k: v[ok] for k, v in carried.items()}}


def _evaluate_target(cfg, ws, target: str, args: dict) -> dict:
    names, d7, dev, threads = args["names"], args["d7"], args["dev"], int(args["threads"])
    st = {k: d7[k] for k in ("W", "P", "n_min", "n_max", "min_delta", "tol_delta") if k in d7}
    kcfg = cfg.experiment["knn"]
    kp = int(kcfg["k_primary"])
    k_grid = [int(k) for k in kcfg["k_grid"]]
    nb = int(cfg.experiment["evaluation"]["n_bootstrap"])
    C_grid = cfg.model_space["baselines"]["logreg_C_grid"]
    D = load_cohort_data(ws.common(dev), ws.derived(dev), names)
    idx = np.arange(len(D))
    y, thr = labels_for(target, D, idx, cfg.targets)
    tr = idx[y >= 0]
    C = TARGET_CLASSES[target]
    LOG.info("external: final %s model on %s (n=%d)", target, dev, len(tr))
    ft_dev = D.ft.take(tr)
    ck = ws.predictions("external", target) / "ckpt"
    fp = fit_perisim(cfg, ft_dev, y[tr], C, seed=cfg.seed, threads=threads, stopping_override=st,
                     exclude_families=set(d7.get("exclude_families", [])), early_stop=True, checkpoint_dir=ck)
    E_dev = fp.enc.transform(ft_dev)
    N_dev = fp.enc.transform_neutral(ft_dev)          # unsupervised axes for the Euclidean comparator
    ones = np.ones(fp.enc.n_features)
    w_dims, e_dims = fp.enc.weights_dims(fp.w), fp.enc.neutral_weights_dims(ones)
    Xm_dev, _ = fp.enc.transform_model(ft_dev)
    car_dev = {k: v[tr] for k, v in D.carried.items() if np.isfinite(v[tr]).any()}

    def _lr(yy: np.ndarray, CC: int):
        ok = np.where(yy >= 0)[0]
        sp = [(a, b) for a, b in GroupKFold(3).split(Xm_dev[ok], yy[ok], ft_dev.subjects[ok])]
        f, _ = parametric.fit_logreg(Xm_dev[ok], yy[ok], C_grid, sp, CC)
        return f

    # Mode A case bases, one per labelling of INSPIRE: the target's own, the eICU-reduced T1 (AUDIT E1) and the
    # absolute T2/T3 categories (C4). The learned metric is the same; only the labels the kNN counts change.
    euc = KNNModel(N_dev, e_dims, y[tr], C, ft_dev.subjects, k_grid)
    euc.fit_alpha_and_ood(kcfg["alpha_grid"], kp, kcfg["ood_percentile"])
    cases = {"own": {"knn": fp.knn, "ok": np.arange(len(tr)), "carried": car_dev, "euc": euc, "lr": _lr(y[tr], C),
                     "swift": None, "C": C}}
    if target == "t1":
        cases["own"]["swift"] = parametric.fit_swift(swift_inputs(ft_dev), (y[tr] == SEVERE[target]).astype(int))[0]
    if target == "t1" and any(_eval_target_for(target, c, cfg) == "t1_bin" for c in args["externals"]):
        yr = D.extra["t1_reduced"][tr]
        cb = _case_base(E_dev, w_dims, yr, 2, ft_dev, k_grid, kcfg, car_dev)
        if cb is not None and 0 < yr[cb["ok"]].sum() < cb["ok"].size:
            ok = cb["ok"]
            cb.update(euc=_case_base(N_dev, e_dims, yr, 2, ft_dev, [kp], kcfg, {})["knn"], lr=_lr(yr, 2), C=2,
                      swift=parametric.fit_swift(swift_inputs(ft_dev.take(ok)), yr[ok])[0],
                      prevalence=float(yr[ok].mean()))
            cases["reduced"] = cb
    if target in ("t2", "t3"):
        ya = absolute_labels(target, D, cfg.targets)[tr]
        cb = _case_base(E_dev, w_dims, ya, C, ft_dev, k_grid, kcfg, car_dev)
        if cb is not None:
            cb.update(euc=_case_base(N_dev, e_dims, ya, C, ft_dev, [kp], kcfg, {})["knn"], lr=_lr(ya, C), swift=None, C=C)
            cases["absolute"] = cb

    tres: dict = {"final_model": {"lambda": fp.info["lambda"], "m": fp.info["m"], "alpha": fp.info["alpha"],
                                  "weights": dict(zip(fp.enc.feature_names, np.round(fp.w, 8).tolist())),
                                  "vdm_tables": fp.enc.vdm_tables(int(cfg.experiment["evaluation"]["small_cell_threshold"])),
                                  "shape_tables": fp.enc.shape_tables(int(cfg.experiment["evaluation"]["small_cell_threshold"])),
                                  "geometry": fp.enc.geometry,
                                  "n_accepted": fp.loop["n_accepted"], "stop_reason": fp.loop["stop_reason"],
                                  "thresholds": thr}, "cohorts": {}}
    for c in args["externals"]:
        E = load_cohort_data(ws.common(c), ws.derived(c), names)
        et = _eval_target_for(target, c, cfg)
        if target == "t3" and (E.t1 < 0).all():
            LOG.info("external: %s has no full T1 → T3 not evaluable (skipped, declared)", c)
            continue
        if et == "t1_bin" and "reduced" not in cases:
            LOG.warning("external: no usable reduced INSPIRE label for %s → T1 not evaluated", c)
            continue
        eidx = np.arange(len(E))
        # labellings of the external cohort: (name, query labels, [(query mask, case-base labels)], INSPIRE case base)
        analyses = []
        if et == "t1_bin":
            analyses.append(("main", E.t1_severe.copy(), None, "reduced", {}))
        elif target == "t1":
            analyses.append(("main", labels_for(target, E, eidx, cfg.targets)[0], None, "own", {}))
        else:
            half = patient_halves(E.ft.subjects, cfg.seed)
            V = {}
            thr_h = {}
            for h in (0, 1):            # V[h] = labels with the cut-points estimated on half h
                V[h], thr_h[h] = labels_for(target, E, eidx[half == h], cfg.targets)
            yq = np.where(half == 0, V[1], V[0])
            analyses.append(("crossfit", yq, (half, V), "own",
                             {"thresholds_half0": thr_h[0], "thresholds_half1": thr_h[1]}))
            if "absolute" in cases:
                analyses.append(("absolute", absolute_labels(target, E, cfg.targets), None, "absolute",
                                 {"cut_days": list(cfg.targets["t2"]["absolute_cut_days"])}))
        res_c: dict = {"eval_target": et, "analyses": {},
                       "note": ("Mode A uses the INSPIRE case base labelled with the reduced definition "
                                "(readmission or death), as in eICU." if et == "t1_bin" else "")}
        for i, (name, ye, crossfit, case_key, extra) in enumerate(analyses):
            te = eidx[ye >= 0]
            if te.size < 20:
                LOG.warning("external: only %d evaluable stays in %s for %s/%s → skipped", te.size, c, target, name)
                continue
            primary = i == 0
            r = _analysis(cfg, ws, fp, cases[case_key], E, te, ye, crossfit, et, c, target, k_grid, kp, nb, ft_dev,
                          name=name, primary=primary)
            r.update(extra)
            res_c["analyses"][name] = r
        if not res_c["analyses"]:
            continue
        if c == "eicu" and target == "t2":
            st_e = pl.read_parquet(ws.common(c) / "stays_index.parquet").select("stay_id", "apache_pred_hosp_los")
            j = pl.DataFrame({"stay_id": E.ft.ids.astype(str), "los": E.los}).join(st_e, on="stay_id", how="left") \
                .drop_nulls().filter(pl.col("los").is_not_nan())
            if j.height >= 20:
                rho = spearmanr(j["apache_pred_hosp_los"].to_numpy(), j["los"].to_numpy()).correlation
                res_c["apache_iv_los_descriptive"] = {
                    "spearman_rho_vs_post_icu_los": float(rho), "n": int(j.height),
                    "note": "Different estimand (total hospital stay predicted at ICU admission vs observed post-ICU "
                            "stay): descriptive reference only (§6.4.10)."}
        tres["cohorts"][c] = res_c
    return tres


def _mode_b(Ee: np.ndarray, w_dims: np.ndarray, subjects: np.ndarray, groups: list, Cb: int, k_grid: list[int],
            alpha: dict) -> tuple[dict, KNNModel, np.ndarray, np.ndarray]:
    """Mode B: patient-level LOO over the target cohort. `groups` = [(query mask, case-base labels)]: the
    neighbours are the same, the labels they carry (and the LOO prior) follow the query's cut-points."""
    kb = KNNModel(Ee, w_dims, groups[0][1], Cb, subjects, k_grid)
    kb.alpha = dict(alpha)
    _, Ib, vb = kb.neighbors(None, loo=True)
    P = {k: np.zeros((len(Ee), Cb)) for k in k_grid}
    for mask, lab in groups:
        pri = kb.loo_prior(lab)
        for k in k_grid:
            P[k][mask] = kb.probs(Ib[mask], vb[mask], min(k, kb.k_max), labels=lab, prior=pri[mask])
    return P, kb, Ib, vb


def _analysis(cfg, ws, fp, case: dict, E, te: np.ndarray, ye: np.ndarray, crossfit, et: str, c: str, target: str,
              k_grid: list[int], kp: int, nb: int, ft_dev, name: str, primary: bool) -> dict:
    ft_e = E.ft.take(te)
    yt = ye[te]
    Cb = case["C"]
    knnA = case["knn"]
    if crossfit is None:
        groups = [(np.ones(te.size, dtype=bool), yt)]
    else:
        half, V = crossfit
        h = half[te]
        groups = [(h == 0, V[1][te]), (h == 1, V[0][te])]   # each half labelled with the other half's cut-points
    res: dict = {"n": int(te.size)}
    frames = []
    variants = {"": ft_e, "_discretized": discretize_like(ft_e, ft_dev)} if primary else {"": ft_e}
    for suf, fte in variants.items():
        Ee = fp.enc.transform(fte)
        # ---- mode A: INSPIRE case base
        probs, (Dd, I, valid) = knn_predict(knnA, Ee, fte.subjects, k_grid)
        _, pct, ood = knnA.kth_info(Dd, valid, kp)
        res[f"ood_rate_modeA{suf}"] = float(ood.mean())
        for k in k_grid:
            frames.append(_pred_frame(fte.ids, fte.subjects, yt, f"A_perisim{suf}", k, probs[k]))
        if not suf and primary:   # §5.6 records of the external queries (mode A), inside the pod only
            nbrs = target == cfg.patient_output_target
            car = knnA.carried_summary(I, valid, kp, case["carried"]) if nbrs else None
            write_jsonl(ws.predictions("external", target) / f"{c}_patient.jsonl",
                        patient_pieces(fte.ids, E.t0[te], et, probs[kp], kp,
                                       **({"kth_pct": pct, "ood": ood, "I": I, "D": Dd, "valid": valid,
                                           "ref_ids": knnA.ids, "carried": car} if nbrs else {})))
        # ---- mode B: target-cohort case base, patient-level LOO, INSPIRE weights (α from the same task)
        Pb, kb, Ib, vb = _mode_b(Ee, fp.enc.weights_dims(fp.w), fte.subjects, groups, Cb, k_grid, knnA.alpha)
        for k in k_grid:
            frames.append(_pred_frame(fte.ids, fte.subjects, yt, f"B_perisim{suf}", k, Pb[k]))
        if suf:
            continue
        # ---- B-VDM: VDM tables recomputed in the target cohort (patient-level CV); HVDM geometry only
        if fp.enc.geometry == "hvdm":
            Pv = {k: np.zeros((te.size, Cb)) for k in k_grid}
            for mask, lab in groups:
                P_ = _mode_b_vdm(fp, fte, lab, Cb, k_grid, alpha=dict(knnA.alpha))
                for k in k_grid:
                    Pv[k][mask] = P_[k][mask]
            for k in k_grid:
                frames.append(_pred_frame(fte.ids, fte.subjects, yt, "B_vdm_target", k, Pv[k]))
        # ---- comparators: unsupervised kNN (neutral axes) and trivial prevalences
        eA = case["euc"]
        Ne = fp.enc.transform_neutral(fte)
        Pe, _ = knn_predict(eA, Ne, fte.subjects, [kp])
        frames.append(_pred_frame(fte.ids, fte.subjects, yt, "A_euclidean", kp, Pe[kp]))
        Pbe, _, Ibe, vbe = _mode_b(Ne, fp.enc.neutral_weights_dims(np.ones(fp.enc.n_features)), fte.subjects, groups,
                                   Cb, [kp], eA.alpha)
        frames.append(_pred_frame(fte.ids, fte.subjects, yt, "B_euclidean", kp, Pbe[kp]))
        frames.append(_pred_frame(fte.ids, fte.subjects, yt, "A_prevalence", 0,
                                  np.tile(knnA.prior, (te.size, 1))))
        Pp = np.zeros((te.size, Cb))
        for mask, lab in groups:           # the target cohort's own prevalence, without the query's patient
            Pp[mask] = kb.loo_prior(lab)[mask]
        frames.append(_pred_frame(fte.ids, fte.subjects, yt, "B_prevalence", 0, Pp))
        Xe, _ = fp.enc.transform_model(fte)
        frames.append(_pred_frame(fte.ids, fte.subjects, yt, "A_logreg", 0, case["lr"](Xe)))
        if case.get("swift") is not None:
            pg = case["swift"](swift_inputs(fte))
            frames.append(_pred_frame(fte.ids, fte.subjects, yt, "swift", 0, np.column_stack([1 - pg, pg])))
        # ---- Q4 in the target cohort (mode B neighbours from the target cohort itself)
        if primary:
            car = {k_: v[te] for k_, v in E.carried.items() if np.isfinite(v[te]).any()}
            rows = []
            for meth, (II, vv) in {"B_perisim": (Ib, vb), "B_euclidean": (Ibe, vbe)}.items():
                for var, pr in kb.carried_summary(II, vv, kp, car).items():
                    rows.append(pl.DataFrame({"stay_id": fte.ids.astype(str), "method": meth, "variable": var,
                                              "pred": pr.astype(float), "obs": car[var].astype(float)}))
            res["q4"] = p4_metrics(pl.concat(rows)) if rows else {}
    pred = pl.concat(frames, how="diagonal_relaxed")
    # Reference = PERISIM mode A at the primary k (explicit; no renaming, so the k-curve stays complete)
    res["evaluation"] = evaluate_predictions(pred, et, kp, nb, cfg.seed, ref=("A_perisim", kp))
    # Q3 and metric-transport contrasts (a − b on common stays; negative = a better)
    res["q3_contrasts"] = {k: v for k, v in {
        "B_perisim − A_perisim": paired_between(pred, ("B_perisim", kp), ("A_perisim", kp), nb, cfg.seed),
        "B_perisim − B_euclidean": paired_between(pred, ("B_perisim", kp), ("B_euclidean", kp), nb, cfg.seed),
        "B_vdm_target − B_perisim": paired_between(pred, ("B_vdm_target", kp), ("B_perisim", kp), nb, cfg.seed),
        "A_perisim − A_euclidean": paired_between(pred, ("A_perisim", kp), ("A_euclidean", kp), nb, cfg.seed),
        "B_perisim − B_prevalence": paired_between(pred, ("B_perisim", kp), ("B_prevalence", 0), nb, cfg.seed),
    }.items() if v is not None}
    write_parquet(pred, ws.predictions("external", target) / f"{c}{'' if primary else '_' + name}_pred.parquet")
    return res


def _mode_b_vdm(fp, ft, y, C, k_grid, n_splits: int = 5, alpha: dict | None = None) -> dict[int, np.ndarray]:
    """Mode B with VDM tables recomputed in the target cohort: for each fold f the tables are fitted on the
    other folds (patient-level) and the queries of f are resolved against the target cohort excluding their
    own patient (whose stays are also left out of the smoothing prior)."""
    from perisim.metric.encoding import OTHER, _norm_cat

    P = {k: np.zeros((len(y), C)) for k in k_grid}
    ok = y >= 0
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(ft.subjects))))
    for tr, te in gkf.split(np.zeros(len(y)), y, ft.subjects):
        tr = tr[ok[tr]]
        prior = np.bincount(y[tr], minlength=C) / max(tr.size, 1)
        enc = copy.deepcopy(fp.enc)
        for b in enc.blocks:
            if b.kind != "nominal":
                continue
            v = _norm_cat(ft.cat[b.source])[tr]
            vv = np.where(np.isin(v, b.params["keep"]), v, OTHER)
            m = enc.m
            table_ = {}
            for cat in np.unique(vv):
                sel = vv == cat
                nc = sel.sum()
                ph = np.bincount(y[tr][sel], minlength=C) / max(nc, 1)
                table_[cat] = ((nc * ph + m * prior) / (nc + m), int(nc))
            b.params = {**b.params, "table": table_, "unseen": prior, "seen": sorted(set(v.tolist()))}
            b.dims = C
            raw = enc._raw_block(b, ft.take(tr))          # δ_j renormalized on the fitting part of the target cohort
            b.scale = float(np.sqrt(2.0 * raw.var(axis=0).sum())) or 1.0
        enc._index()
        Ee = enc.transform(ft)
        kn = KNNModel(Ee, enc.weights_dims(fp.w), y, C, ft.subjects, k_grid)
        kn.alpha = {k: (alpha or fp.knn.alpha).get(k, 1.0) for k in k_grid}
        _, Iq, vq = kn.neighbors(Ee[te], ft.subjects[te])
        pri = kn.loo_prior(y)[te]
        for k in k_grid:
            P[k][te] = kn.probs(Iq, vq, min(k, kn.k_max), prior=pri)
    return P


def _fmt_cuts(v) -> str:
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_fmt_cuts(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return " / ".join(_fmt_cuts(x) for x in v)
    return f"{float(v):.2f}"


def render_external(out: dict, kp: int, n_boot: int = 2000) -> str:
    sec = [box("Mode A: INSPIRE case base. Mode B: INSPIRE weights with the target cohort's case base (patient-level "
               "LOO, the query's patient also left out of the smoothing prior). The B − A contrast answers Q3. eICU: "
               "reduced target (severe = readmission or death), with the INSPIRE case base labelled the same way. "
               "T2/T3: target-cohort tertiles cross-fitted between two patient-level halves, and absolute categories "
               "(≤ 3 / 4–7 / > 7 d) as a co-primary analysis with the INSPIRE case base relabelled (AUDIT C4).")]
    for t, r in out["targets"].items():
        sec.append(f"<h2>{t.upper()}</h2>")
        fm = r["final_model"]
        sec.append(table(["Final model (INSPIRE)", "value"], [["λ", fm["lambda"]], ["m (VDM)", fm["m"]],
                                                              ["accepted models", fm["n_accepted"]],
                                                              ["stopping", fm["stop_reason"]]]))
        for c, rc in r["cohorts"].items():
            for name, ra in rc.get("analyses", {}).items():
                sec.append(f"<h3>{c} — {name} (n={ra['n']}, target {rc['eval_target']})</h3>")
                if rc.get("note"):
                    sec.append(box(rc["note"]))
                cuts = [(k, v) for k, v in ra.items() if k.startswith("thresholds_") or k == "cut_days"]
                if cuts:
                    sec.append(table(["Cut-points (days)", "value"], [[k, _fmt_cuts(v)] for k, v in cuts]))
                sec.extend(eval_sections(ra["evaluation"], n_boot))
                if ra.get("q3_contrasts"):
                    sec.append("<h4>Q3 / metric-transport contrasts at the primary k (Δ log-loss; negative = first better)</h4>")
                    sec.append(table(["Contrast", "Δ log-loss", "CI low", "CI high", "p", "common stays"],
                                     [[k, v["mean"], v["ci_low"], v["ci_high"], v.get("p_two_sided"), v["n_common"]]
                                      for k, v in ra["q3_contrasts"].items()], nd=4))
                sec.append(table(["OOD mode A", "OOD mode A (discretized)"],
                                 [[ra.get("ood_rate_modeA"), ra.get("ood_rate_modeA_discretized")]]))
                if ra.get("q4"):
                    rows = [[var, meth, v.get("logloss"), v.get("brier"), v.get("mae_days"), v["n"]]
                            for var, d in ra["q4"].items() for meth, v in d.items()]
                    sec.append("<h4>Q4 in the target cohort (mode B neighbours)</h4>")
                    sec.append(table(["Variable", "Method", "Log-loss", "Brier", "MAE days", "n"], rows))
            if rc.get("apache_iv_los_descriptive"):
                a = rc["apache_iv_los_descriptive"]
                sec.append(box(f"APACHE IV (eICU): Spearman ρ with the post-ICU stay = {a['spearman_rho_vs_post_icu_los']:.3f} "
                               f"(n={a['n']}). {a['note']}", "warn"))
    return page("PERISIM external validation (MIMIC-IV, eICU)", sec, "PERISIM · §6.2 · aggregates only")
