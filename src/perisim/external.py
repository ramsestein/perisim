"""External validation (§6.2) — only after the freeze (§6.3).

Final model: PERISIM fitted on ALL of INSPIRE with the same nested procedure. The final models of the
targets (T1, T2, T3) are independent, so they run in parallel processes (at most `jobs` × `threads`).
- Mode A (full transport): weights, VDM tables and INSPIRE case base; queries from the target cohort.
- Mode B (metric transport): INSPIRE weights and scales, case base of the target cohort, patient-level LOO
  within the target cohort. B-VDM variant: VDM tables recomputed in the target cohort with patient-level CV.
- eICU: T1 reduced to severe / not severe (declared as a reduced target). T2 uses each cohort's own tertiles.
- Sensitivity: INSPIRE percentile discretization applied to MIMIC-IV / eICU (§4.1).
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import logging
import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from perisim.baselines import parametric
from perisim.evaluation.html import box, page, table
from perisim.evaluation.report import eval_sections, evaluate_predictions, p4_metrics, paired_between
from perisim.experiment import SEVERE, TARGET_CLASSES, fit_perisim, knn_predict, labels_for, load_cohort_data, swift_inputs
from perisim.knn.predict import KNNModel
from perisim.utils import set_threads, write_json, write_parquet

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


def run_external(pipe) -> None:
    from perisim.pipeline import _child_threads

    cfg, ws = pipe.cfg, pipe.ws
    dev = cfg.experiment["cohorts"]["development"]
    externals = [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()]
    d7 = pipe.d7()
    targets = list(cfg.experiment["targets_to_model"])
    common = {"cfg": cfg.to_dict(), "profile": cfg.profile, "workdir": str(ws.root), "names": pipe.feature_names(),
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
        ws = Workspace(args["workdir"])
        return {"target": target, "result": _evaluate_target(cfg, ws, target, args)}
    except Exception:
        return {"target": target, "error": traceback.format_exc()}


def _evaluate_target(cfg, ws, target: str, args: dict) -> dict:
    names, d7, dev, threads = args["names"], args["d7"], args["dev"], int(args["threads"])
    st = {k: d7[k] for k in ("W", "P", "n_min", "n_max", "min_delta", "tol_delta") if k in d7}
    kcfg = cfg.experiment["knn"]
    kp = int(kcfg["k_primary"])
    k_grid = [int(k) for k in kcfg["k_grid"]]
    nb = int(cfg.experiment["evaluation"]["n_bootstrap"])
    D = load_cohort_data(ws.common(dev), ws.derived(dev), names)
    idx = np.arange(len(D))
    y, thr = labels_for(target, D, idx, cfg.targets)
    tr = idx[y >= 0]
    C = TARGET_CLASSES[target]
    LOG.info("external: final %s model on %s (n=%d)", target, dev, len(tr))
    ft_dev = D.ft.take(tr)
    fp = fit_perisim(cfg, ft_dev, y[tr], C, seed=cfg.seed, threads=threads, stopping_override=st,
                     exclude_families=set(d7.get("exclude_families", [])), early_stop=True)
    E_dev = fp.enc.transform(ft_dev)
    ones = np.ones(fp.enc.n_features)
    euc = KNNModel(E_dev, fp.enc.weights_dims(ones), y[tr], C, ft_dev.subjects, k_grid)
    euc.fit_alpha_and_ood(kcfg["alpha_grid"], kp, kcfg["ood_percentile"])
    Xm_dev, _ = fp.enc.transform_model(ft_dev)
    splits = [(a, b) for a, b in GroupKFold(3).split(Xm_dev, y[tr], ft_dev.subjects)]
    lr_f, _ = parametric.fit_logreg(Xm_dev, y[tr], cfg.model_space["baselines"]["logreg_C_grid"], splits, C)
    sev = SEVERE[target]
    sw = None
    if target == "t1":
        sw, _ = parametric.fit_swift(swift_inputs(ft_dev), (y[tr] == sev).astype(int))

    # AUDIT E1: for a cohort whose T1 is reduced (eICU: readmission or death only) the INSPIRE case base must
    # carry the SAME reduced label; the learned metric (weights) is unchanged — the kNN only counts labels.
    red = None
    if target == "t1" and any(_eval_target_for(target, c, cfg) == "t1_bin" for c in args["externals"]):
        yr = D.extra["t1_reduced"][tr]
        ok = np.where(yr >= 0)[0]
        if ok.size >= 20 and 0 < yr[ok].sum() < ok.size:
            ftr = ft_dev.take(ok)
            kr = KNNModel(E_dev[ok], fp.enc.weights_dims(fp.w), yr[ok], 2, ftr.subjects, k_grid)
            kr.fit_alpha_and_ood(kcfg["alpha_grid"], kp, kcfg["ood_percentile"])
            er = KNNModel(E_dev[ok], fp.enc.weights_dims(ones), yr[ok], 2, ftr.subjects, [kp])
            er.fit_alpha_and_ood(kcfg["alpha_grid"], kp, kcfg["ood_percentile"])
            sp = [(a, b) for a, b in GroupKFold(3).split(Xm_dev[ok], yr[ok], ftr.subjects)]
            lr_r, _ = parametric.fit_logreg(Xm_dev[ok], yr[ok], cfg.model_space["baselines"]["logreg_C_grid"], sp, 2)
            sw_r, _ = parametric.fit_swift(swift_inputs(ftr), yr[ok])
            red = {"knn": kr, "euc": er, "lr": lr_r, "swift": sw_r, "prevalence": float(yr[ok].mean())}

    tres: dict = {"final_model": {"lambda": fp.info["lambda"], "m": fp.info["m"], "alpha": fp.info["alpha"],
                                  "weights": dict(zip(fp.enc.feature_names, np.round(fp.w, 8).tolist())),
                                  "vdm_tables": fp.enc.vdm_tables(int(cfg.experiment["evaluation"]["small_cell_threshold"])),
                                  "n_accepted": fp.loop["n_accepted"], "stop_reason": fp.loop["stop_reason"],
                                  "thresholds": thr}, "cohorts": {}}
    for c in args["externals"]:
        E = load_cohort_data(ws.common(c), ws.derived(c), names)
        et = _eval_target_for(target, c, cfg)
        reduced = et == "t1_bin"
        if target == "t3" and (E.t1 < 0).all():
            LOG.info("external: %s has no full T1 → T3 not evaluable (skipped, declared)", c)
            continue
        if reduced and red is None:
            LOG.warning("external: no usable reduced INSPIRE label for %s → T1 not evaluated", c)
            continue
        eidx = np.arange(len(E))
        if reduced:
            ye = E.t1_severe.copy()
        else:
            ye, _ = labels_for(target, E, eidx, cfg.targets)   # target cohort's own tertiles (see AUDIT C4)
        te = eidx[ye >= 0]
        if te.size < 20:
            LOG.warning("external: only %d evaluable stays in %s for %s → skipped", te.size, c, target)
            continue
        ft_e = E.ft.take(te)
        yt = ye[te]
        Cb = 2 if reduced else C
        knnA = red["knn"] if reduced else fp.knn
        frames = []
        res_c: dict = {"n": int(te.size), "eval_target": et,
                       "note": ("Mode A uses the INSPIRE case base labelled with the reduced definition "
                                "(readmission or death), as in eICU." if reduced else "")}
        for suf, fte in {"": ft_e, "_discretized": discretize_like(ft_e, ft_dev)}.items():
            Ee = fp.enc.transform(fte)
            # ---- mode A: INSPIRE case base
            probs, (Dd, I, valid) = knn_predict(knnA, Ee, fte.subjects, k_grid)
            _, _, ood = knnA.kth_info(Dd, valid, kp)
            res_c[f"ood_rate_modeA{suf}"] = float(ood.mean())
            for k in k_grid:
                frames.append(_pred_frame(fte.ids, fte.subjects, yt, f"A_perisim{suf}", k, probs[k]))
            # ---- mode B: target-cohort case base, patient-level LOO, INSPIRE weights (α from the same task)
            kb = KNNModel(Ee, fp.enc.weights_dims(fp.w), yt, Cb, fte.subjects, k_grid)
            kb.alpha = dict(knnA.alpha)
            _, Ib, vb = kb.neighbors(None, loo=True)
            for k in k_grid:
                frames.append(_pred_frame(fte.ids, fte.subjects, yt, f"B_perisim{suf}", k,
                                          kb.probs(Ib, vb, min(k, kb.k_max))))
            if suf:
                continue
            # ---- B-VDM: VDM tables recomputed in the target cohort (patient-level CV)
            Pv = _mode_b_vdm(fp, fte, yt, Cb, k_grid, alpha=dict(knnA.alpha))
            for k in k_grid:
                frames.append(_pred_frame(fte.ids, fte.subjects, yt, "B_vdm_target", k, Pv[k]))
            # ---- comparators
            eA = red["euc"] if reduced else euc
            Pe, _ = knn_predict(eA, Ee, fte.subjects, [kp])
            frames.append(_pred_frame(fte.ids, fte.subjects, yt, "A_euclidean", kp, Pe[kp]))
            kbe = KNNModel(Ee, fp.enc.weights_dims(ones), yt, Cb, fte.subjects, [kp])
            kbe.alpha = dict(eA.alpha)
            _, Ibe, vbe = kbe.neighbors(None, loo=True)
            frames.append(_pred_frame(fte.ids, fte.subjects, yt, "B_euclidean", kp, kbe.probs(Ibe, vbe, min(kp, kbe.k_max))))
            Xe, _ = fp.enc.transform_model(fte)
            frames.append(_pred_frame(fte.ids, fte.subjects, yt, "A_logreg", 0, (red["lr"] if reduced else lr_f)(Xe)))
            swf = red["swift"] if reduced else sw
            if swf is not None:
                pg = swf(swift_inputs(fte))
                frames.append(_pred_frame(fte.ids, fte.subjects, yt, "swift", 0, np.column_stack([1 - pg, pg])))
            # ---- Q4 in the target cohort (mode B: neighbours from the target cohort itself)
            car = {k_: v[te] for k_, v in E.carried.items() if np.isfinite(v[te]).any()}
            rows = []
            for meth, (II, vv) in {"B_perisim": (Ib, vb), "B_euclidean": (Ibe, vbe)}.items():
                for var, pr in kb.carried_summary(II, vv, kp, car).items():
                    rows.append(pl.DataFrame({"stay_id": fte.ids.astype(str), "method": meth, "variable": var,
                                              "pred": pr.astype(float), "obs": car[var].astype(float)}))
            res_c["q4"] = p4_metrics(pl.concat(rows)) if rows else {}
        pred = pl.concat(frames, how="diagonal_relaxed")
        # Reference = PERISIM mode A at the primary k (explicit; no renaming, so the k-curve stays complete)
        res_c["evaluation"] = evaluate_predictions(pred, et, kp, nb, cfg.seed, ref=("A_perisim", kp))
        # Q3 and metric-transport contrasts (a − b on common stays; negative = a better)
        res_c["q3_contrasts"] = {k: v for k, v in {
            "B_perisim − A_perisim": paired_between(pred, ("B_perisim", kp), ("A_perisim", kp), nb, cfg.seed),
            "B_perisim − B_euclidean": paired_between(pred, ("B_perisim", kp), ("B_euclidean", kp), nb, cfg.seed),
            "B_vdm_target − B_perisim": paired_between(pred, ("B_vdm_target", kp), ("B_perisim", kp), nb, cfg.seed),
            "A_perisim − A_euclidean": paired_between(pred, ("A_perisim", kp), ("A_euclidean", kp), nb, cfg.seed),
        }.items() if v is not None}
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
        write_parquet(pred, ws.predictions("external", target) / f"{c}_pred.parquet")
        tres["cohorts"][c] = res_c
    return tres


def _mode_b_vdm(fp, ft, y, C, k_grid, n_splits: int = 5, alpha: dict | None = None) -> dict[int, np.ndarray]:
    """Mode B with VDM tables recomputed in the target cohort: for each fold f the tables are fitted on the
    other folds (patient-level) and the queries of f are resolved against the target cohort excluding their
    own patient."""
    from perisim.metric.encoding import OTHER, _norm_cat

    P = {k: np.zeros((len(y), C)) for k in k_grid}
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(ft.subjects))))
    prior = np.bincount(y, minlength=C) / len(y)
    for tr, te in gkf.split(np.zeros(len(y)), y, ft.subjects):
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
            b.params = {**b.params, "table": table_, "unseen": prior}
            b.dims = C
            raw = enc._raw_block(b, ft.take(tr))          # δ_j renormalized on the fitting part of the target cohort
            b.scale = float(np.sqrt(2.0 * raw.var(axis=0).sum())) or 1.0
        enc._index()
        Ee = enc.transform(ft)
        kn = KNNModel(Ee, enc.weights_dims(fp.w), y, C, ft.subjects, k_grid)
        kn.alpha = {k: (alpha or fp.knn.alpha).get(k, 1.0) for k in k_grid}
        _, Iq, vq = kn.neighbors(Ee[te], ft.subjects[te])
        for k in k_grid:
            P[k][te] = kn.probs(Iq, vq, min(k, kn.k_max))
    return P


def render_external(out: dict, kp: int, n_boot: int = 2000) -> str:
    sec = [box("Mode A: INSPIRE case base. Mode B: INSPIRE weights with the target cohort's case base (patient-level "
               "LOO). The B − A contrast answers Q3. eICU: reduced target (severe = readmission or death), with the "
               "INSPIRE case base labelled the same way. Target-cohort tertiles for T2/T3 (see AUDIT C4).")]
    for t, r in out["targets"].items():
        sec.append(f"<h2>{t.upper()}</h2>")
        fm = r["final_model"]
        sec.append(table(["Final model (INSPIRE)", "value"], [["λ", fm["lambda"]], ["m (VDM)", fm["m"]],
                                                              ["accepted models", fm["n_accepted"]],
                                                              ["stopping", fm["stop_reason"]]]))
        for c, rc in r["cohorts"].items():
            sec.append(f"<h3>{c} (n={rc['n']}, target {rc['eval_target']})</h3>")
            if rc.get("note"):
                sec.append(box(rc["note"]))
            sec.extend(eval_sections(rc["evaluation"], n_boot))
            if rc.get("q3_contrasts"):
                sec.append("<h4>Q3 / metric-transport contrasts at the primary k (Δ log-loss; negative = first better)</h4>")
                sec.append(table(["Contrast", "Δ log-loss", "CI low", "CI high", "p", "common stays"],
                                 [[k, v["mean"], v["ci_low"], v["ci_high"], v.get("p_two_sided"), v["n_common"]]
                                  for k, v in rc["q3_contrasts"].items()], nd=4))
            sec.append(table(["OOD mode A", "OOD mode A (discretized)"],
                             [[rc.get("ood_rate_modeA"), rc.get("ood_rate_modeA_discretized")]]))
            if rc.get("q4"):
                rows = [[var, meth, v.get("logloss"), v.get("brier"), v.get("mae_days"), v["n"]]
                        for var, d in rc["q4"].items() for meth, v in d.items()]
                sec.append("<h4>Q4 in the target cohort (mode B neighbours)</h4>")
                sec.append(table(["Variable", "Method", "Log-loss", "Brier", "MAE days", "n"], rows))
            if rc.get("apache_iv_los_descriptive"):
                a = rc["apache_iv_los_descriptive"]
                sec.append(box(f"APACHE IV (eICU): Spearman ρ with the post-ICU stay = {a['spearman_rho_vs_post_icu_los']:.3f} "
                               f"(n={a['n']}). {a['note']}", "warn"))
    return page("PERISIM external validation (MIMIC-IV, eICU)", sec, "PERISIM · §6.2 · aggregates only")
