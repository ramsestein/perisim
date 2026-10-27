"""k sweep of the internal cross-validation (secondary, descriptive; decided 2026-10-25).

k is a run-time parameter (§5.5): the metric of an outer fold does not depend on it. The fold models the CV saved are
queried again over a wider grid than ``experiment.knn.k_grid`` (from k = 1 to a fifth of the case base), with α
refitted per k by the same patient-level LOO and nothing else refitted. It answers how much the choice of k matters:

  - the curve of every kNN method over k (log-loss, Brier, RPS, AUROC, calibration slope);
  - the paired log-loss difference of each k against the primary k (D11, which stays the primary analysis);
  - every comparator against PERISIM at the same k (does the ranking depend on k?);
  - k chosen inside the training fold, by the LOO log-loss that already fits α, against the fixed primary k.

Reproduction check: on the k of the CV grid the sweep must give the CV's own predictions (exactly for PERISIM, the
Euclidean kNN and RF-GAP; the weighted comparators use the weights the CV stored, w/Σw rounded to 6 decimals, so
approximately; and a neighbour tied with the k-th may be taken in one search and not in the other, which mostly
affects the categorical HVDM geometry). Not a pipeline stage, which would re-key the pilot and the CV of a finished run: ``python -m perisim
ksweep`` after the CV, keyed by the CV it was computed from. Patient-level predictions stay under predictions/ (never
exported); results/, reports/ and artifacts/ get aggregates only.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing as mp
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl

from perisim.config import Config, code_hash, stable_hash
from perisim.evaluation import metrics as M
from perisim.evaluation.html import box, esc, fig_img, page, table
from perisim.evaluation.report import T_LABELS, _probs
from perisim.experiment import ORDINAL, SEVERE, TARGET_CLASSES, fold_seed, labels_for, load_cohort_data
from perisim.knn.predict import KNNModel
from perisim.sensitivity import _frame
from perisim.utils import read_json, write_json, write_parquet

LOG = logging.getLogger("perisim.ksweep")

#: geometry of each re-queried kNN method's weights, as in experiment.run_fold
GEOMETRY = {"perisim": "shape", "euclidean": "neutral", "hvdm": "hvdm", "nca_diag": "shape", "power_prior": "shape",
            "shap_lgbm": "neutral", "rf_importance": "neutral", "relieff": "neutral", "rf_gap": "rf_gap"}
REFERENCES = ("lgbm", "logreg", "group_prevalence", "prevalence")      # no k: horizontal references
LABELS = {"perisim": "PERISIM", "perisim_kernel": "PERISIM, exp(−d²) kernel", "euclidean": "Euclidean",
          "hvdm": "HVDM (v0.4 geometry)", "nca_diag": "NCA, diagonal", "power_prior": "NCA, power prior",
          "shap_lgbm": "SHAP (LightGBM) weights", "rf_importance": "RF importance weights", "relieff": "ReliefF weights",
          "rf_gap": "RF-GAP proximities", "lgbm": "LightGBM", "logreg": "logistic regression",
          "group_prevalence": "prevalence by surgical group", "prevalence": "prevalence"}


# --------------------------------------------------------------------------- small pure pieces (tested)
def k_grid_for(cfg, n_train: int) -> list[int]:
    """The sweep's k: its own grid, the CV grid (for the reproduction check) and the primary k, up to a fraction of the
    smallest training case base."""
    kc, sw = cfg.experiment["knn"], cfg.experiment.get("k_sweep") or {}
    kp = int(kc["k_primary"])
    ks = {int(k) for k in sw.get("k_grid", [])} | {int(k) for k in kc["k_grid"]} | {kp}
    cap = max(kp, int(float(sw.get("max_fraction", 0.2)) * n_train))
    return sorted(k for k in ks if 1 <= k <= min(cap, n_train - 1))


def k_star(loo_logloss: dict) -> int:
    """k chosen inside the training fold: the smallest k with the lowest patient-level LOO log-loss."""
    return min((float(v), int(k)) for k, v in loo_logloss.items())[1]


def stored_weights(stored: dict, names: list[str]) -> np.ndarray:
    """A comparator's weights as the CV stored them (w/Σw, 6 decimals), in the encoder's order. The 1e-8 floor of the
    importance-based weights was rounded to 0: put back a floor of the same order, so that those variables still
    only break ties."""
    return np.maximum(np.array([float(stored.get(n, 0.0)) for n in names]), 1e-9)


def reproduction(sweep: pl.DataFrame, cv: pl.DataFrame) -> list[dict]:
    """Sweep vs CV predictions on every (method, k) both have: largest |Δp| and log-loss difference (sweep − CV)."""
    rows = []
    for (meth, k), g in cv.filter(pl.col("k") > 0).group_by(["method", "k"]):
        s = sweep.filter((pl.col("method") == meth) & (pl.col("k") == k))
        if not s.height:
            continue
        j = g.join(s, on="stay_id", how="inner", suffix="_sw").sort("stay_id")
        pc = [c for c in s.columns if c.startswith("p") and c[1:].isdigit()]
        A, B = j.select(pc).to_numpy(), j.select([c + "_sw" for c in pc]).to_numpy()
        y = j["y"].to_numpy()
        rows.append({"method": meth, "k": int(k), "n": int(j.height), "n_cv": int(g.height),
                     "max_abs_diff": float(np.max(np.abs(A - B))) if j.height else float("nan"),
                     "d_logloss": float(M.row_logloss(B, y).mean() - M.row_logloss(A, y).mean())
                     if j.height else float("nan")})
    return sorted(rows, key=lambda r: (r["method"], r["k"]))


def sweep_key(pipe, t: str) -> str:
    cfg = pipe.cfg
    return stable_hash({"cv": pipe.key("cv"), "target": t, "k_sweep": cfg.experiment.get("k_sweep"),
                        "knn": cfg.experiment["knn"], "rf_gap": cfg.model_space["baselines"].get("rf_gap"),
                        "seed": cfg.seed, "n_bootstrap": cfg.experiment["evaluation"]["n_bootstrap"],
                        "code": code_hash(["ksweep.py", "knn", "metric", "baselines/rfgap.py", "evaluation/metrics.py"])},
                       16)


# --------------------------------------------------------------------------- one outer fold (own process)
def _fold_job(args: dict) -> dict | str:
    """Re-queries one outer fold's metric (and the comparators) over the sweep grid. Returns the fold's info or an
    error message."""
    try:
        from perisim.baselines.rfgap import RFGAP
        from perisim.metric.encoding import Encoder
        from perisim.resources import apply_library_limits
        from perisim.utils import set_threads, setup_logging

        setup_logging()
        set_threads(int(args["threads"]))
        apply_library_limits(int(args["threads"]))
        t0 = time.perf_counter()
        cfg = Config(**args["cfg"], profile=args["profile"], variant=args["variant"])
        kc = cfg.experiment["knn"]
        t, f, ks = args["target"], int(args["fold"]), [int(k) for k in args["k_grid"]]
        C, kp = TARGET_CLASSES[t], int(kc["k_primary"])
        data = load_cohort_data(Path(args["common_dir"]), Path(args["derived_dir"]), args["features"])
        fp = pl.read_parquet(args["folds"])
        of = dict(zip(fp["stay_id"].to_list(), fp["outer_fold"].to_list()))
        folds = np.array([of.get(s, -1) for s in data.ft.ids.astype(str)])
        if (folds < 0).any():
            raise RuntimeError("folds.parquet does not match the development data (re-run the CV)")
        train_idx, test_idx = np.where(folds != f)[0], np.where(folds == f)[0]   # exactly the CV's partition
        y_all, _ = labels_for(t, data, train_idx, cfg.targets)       # T2/T3 cut-points from the training fold only
        tr, te = train_idx[y_all[train_idx] >= 0], test_idx[y_all[test_idx] >= 0]
        with open(args["model"], "rb") as fh:
            model = pickle.load(fh)
        ft_tr, ft_te = data.ft.take(tr), data.ft.take(te)
        if not (np.array_equal(ft_tr.ids.astype(str), np.asarray(model["train_ids"]).astype(str)) and
                np.array_equal(ft_te.ids.astype(str), np.asarray(model["test_ids"]).astype(str))):
            raise RuntimeError("the fold's stays differ from the CV's (data or folds changed since the CV)")
        y_tr, y_te, s_tr, s_te = y_all[tr], y_all[te], ft_tr.subjects, ft_te.subjects
        stored = (read_json(Path(args["fold_json"])) or {}).get("baselines") or {}
        enc, w = model["enc"], model["w"]
        frames: list[pl.DataFrame] = []
        info: dict = {"fold": f, "n_train": int(tr.size), "n_test": int(te.size), "k_grid": ks, "methods": {},
                      "skipped": {}, "seconds": {}}

        def knn(name: str, Er: np.ndarray, Eq: np.ndarray, sw: np.ndarray) -> None:
            kn = KNNModel(Er, sw, y_tr, C, s_tr, ks)
            loo = kn.fit_alpha_and_ood(kc["alpha_grid"], kp, kc["ood_percentile"])
            D, I, valid = kn.neighbors(Eq, s_te)
            for k in ks:
                frames.append(_frame(ft_te.ids, s_te, y_te, name, k, kn.probs(I, valid, min(k, kn.k_max))))
                if name == "perisim":   # same neighbours and α, votes weighted by exp(−d²)
                    frames.append(_frame(ft_te.ids, s_te, y_te, "perisim_kernel", k,
                                         kn.probs(I, valid, min(k, kn.k_max), D=D, kernel=True)))
            ll = {str(k): float(loo[k]["loo_logloss"]) for k in ks}
            info["methods"][name] = {"alpha": {str(k): float(loo[k]["alpha"]) for k in ks}, "loo_logloss": ll,
                                     "k_star": k_star(ll)}

        E_tr, E_te = enc.transform(ft_tr), enc.transform(ft_te)
        neutral: tuple | None = None
        for name in args["methods"]:
            tb = time.perf_counter()
            geo = GEOMETRY[name]
            if name not in ("perisim", "euclidean") and \
                    ("alpha" if geo == "rf_gap" else "weights") not in (stored.get(name) or {}):
                info["skipped"][name] = "not fitted in the CV (absent or failed)"
                continue
            if name == "perisim":
                knn(name, E_tr, E_te, enc.weights_dims(w))
            elif geo == "shape":
                knn(name, E_tr, E_te, enc.weights_dims(stored_weights(stored[name]["weights"], enc.feature_names)))
            elif geo == "neutral":
                if neutral is None:
                    neutral = (enc.transform_neutral(ft_tr), enc.transform_neutral(ft_te))
                wv = (np.ones(enc.n_features) if name == "euclidean"
                      else stored_weights(stored[name]["weights"], enc.feature_names))
                knn(name, neutral[0], neutral[1], enc.neutral_weights_dims(wv))
            elif geo == "hvdm":         # the v0.4 encoder, refitted exactly as in fit_perisim (m = 2nd value of the grid)
                grid_m = cfg.experiment["metric"]["vdm_m_grid"]
                enc_h = Encoder.from_config(cfg, float(grid_m[min(1, len(grid_m) - 1)]), geometry="hvdm").fit(
                    ft_tr, y_tr, C)
                knn(name, enc_h.transform(ft_tr), enc_h.transform(ft_te),
                    enc_h.weights_dims(stored_weights(stored[name]["weights"], enc_h.feature_names)))
            else:                       # RF-GAP: same forest (fold seed) and OOB α per k on the same subsample
                g = cfg.model_space["baselines"]["rf_gap"]
                seed = fold_seed(cfg, f)
                Xm_tr, _ = enc.transform_model(ft_tr)
                Xm_te, _ = enc.transform_model(ft_te)
                rfg = RFGAP(g["n_estimators"], g["min_samples_leaf"], g["max_features"], seed).fit(Xm_tr, y_tr)
                sub = np.random.default_rng(seed).choice(len(y_tr), size=min(3000, len(y_tr)), replace=False)
                NBs = rfg.topk(Xm_tr[sub], max(ks), train_idx=sub, subjects_q=s_tr[sub], subjects_train=s_tr)
                alpha, ll = {}, {}
                for k in ks:
                    lls = {}
                    for a in kc["alpha_grid"]:
                        pr = RFGAP.probs_from_topk(NBs, y_tr, C, k, float(a))
                        lls[a] = float(-np.mean(np.log(np.clip(pr[np.arange(len(sub)), y_tr[sub]], 1e-15, 1))))
                    best = min(lls, key=lls.get)
                    alpha[str(k)], ll[str(k)] = float(best), lls[best]
                NBt = rfg.topk(Xm_te, max(ks))
                for k in ks:
                    frames.append(_frame(ft_te.ids, s_te, y_te, name, k,
                                         RFGAP.probs_from_topk(NBt, y_tr, C, k, alpha[str(k)])))
                info["methods"][name] = {"alpha": alpha, "loo_logloss": ll, "k_star": k_star(ll)}
            info["seconds"][name] = round(time.perf_counter() - tb, 1)
        sweep = pl.concat(frames, how="diagonal_relaxed")
        write_parquet(sweep, Path(args["out"]))
        info["reproduction"] = reproduction(sweep, pl.read_parquet(args["cv_pred"]))
        info["seconds"]["total"] = round(time.perf_counter() - t0, 1)
        info["_key"] = args["key"]
        write_json(Path(args["info"]), info)
        return info
    except Exception:
        return f"ksweep {args.get('target')} fold {args.get('fold')}:\n{traceback.format_exc()}"


def _run_jobs(jobs: list[dict], pipe) -> list[dict]:
    from perisim.pipeline import _child_threads

    par = max(1, min(len(jobs), pipe.jobs()))
    LOG.info("ksweep: %d fold jobs in %d processes × %d threads", len(jobs), par, pipe.threads())
    out, failures = [], []

    def collect(j: dict, r) -> None:
        if isinstance(r, dict):
            out.append(r)
            LOG.info("  ✓ ksweep %s fold %d (%.0f s)", j["target"], j["fold"], r["seconds"]["total"])
        else:
            failures.append(str(r))

    if par == 1:
        for j in jobs:
            collect(j, _fold_job(j))
    else:
        with _child_threads(pipe.threads()), \
                cf.ProcessPoolExecutor(max_workers=par, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_fold_job, j): j for j in jobs}
            for fu in cf.as_completed(futs):
                try:
                    collect(futs[fu], fu.result())
                except Exception as exc:   # worker died (e.g. OOM)
                    collect(futs[fu], f"ksweep fold {futs[fu]['fold']}: process terminated ({exc!r})")
    if failures:
        raise RuntimeError("ksweep jobs failed (relaunching resumes only these):\n" + "\n".join(failures))
    return out


# --------------------------------------------------------------------------- pooling over the outer folds
def _with_p(v: dict) -> dict:
    bs = v.pop("_boot")
    return {**v, "p_two_sided": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))}


def aggregate(cfg, ws, t: str, infos: list[dict]) -> dict:
    kp = int(cfg.experiment["knn"]["k_primary"])
    nb = int(cfg.experiment["evaluation"]["n_bootstrap"])
    sev, ordinal = SEVERE[t], ORDINAL[t]
    per_fold = {int(i["fold"]): pl.read_parquet(ws.predictions("ksweep", t) / f"fold{i['fold']}_pred.parquet")
                for i in infos}
    pred = pl.concat(list(per_fold.values()), how="diagonal_relaxed")
    refs = pl.concat([pl.read_parquet(ws.predictions("cv", t) / f"fold{f}_pred.parquet")
                      .filter(pl.col("method").is_in(list(REFERENCES)) & (pl.col("k") == 0)) for f in per_fold],
                     how="diagonal_relaxed")
    tuned = [per_fold[int(i["fold"])].filter((pl.col("method") == m) & (pl.col("k") == mi["k_star"]))
             .with_columns(pl.lit(f"{m}@tuned").alias("method"), pl.lit(-1, dtype=pl.Int64).alias("k"))
             for i in infos for m, mi in i["methods"].items()]
    order = pred.filter((pl.col("method") == "perisim") & (pl.col("k") == kp)).sort("stay_id")
    ids, subj, y = order["stay_id"], order["subject_id"].to_numpy(), order["y"].to_numpy()

    point, ll, series, incomplete = {}, {}, {}, []
    for (meth, k), g in pl.concat([pred, refs, *tuned], how="diagonal_relaxed").group_by(["method", "k"]):
        g = g.sort("stay_id")
        P, yy = _probs(g), g["y"].to_numpy()
        s = M.summary(P, yy, ordinal, sev)
        slopes = [c["slope"] for c in s["calibration"].values() if np.isfinite(c["slope"])]
        point[(meth, int(k))] = {"n": int(g.height), "logloss": s["logloss"], "brier": s["brier"], "rps": s.get("rps"),
                                 "auroc_macro": s["auroc_ovr_macro"], "auroc_severe": s.get("auroc_severe"),
                                 "auprc_severe": s.get("auprc_severe"),
                                 "cal_slope_severe": s["calibration"].get(str(sev), {}).get("slope"),
                                 "cal_slope_mean": float(np.mean(slopes)) if slopes else None}
        if g.height != order.height or not (g["stay_id"] == ids).all():
            incomplete.append(f"{meth}@{k}")     # e.g. a comparator that failed in one fold: point estimates only
            continue
        ll[(meth, int(k))] = M.row_logloss(P, yy)
        series[f"ll|{meth}|{k}"] = ll[(meth, int(k))]
        series[f"br|{meth}|{k}"] = M.row_brier(P, yy)
        if ordinal:
            series[f"rps|{meth}|{k}"] = M.row_rps(P, yy)
    for (meth, k), v in list(ll.items()):
        if k > 0 and (meth, kp) in ll:
            series[f"d50|{meth}|{k}"] = v - ll[(meth, kp)]
        if k > 0 and meth != "perisim" and ("perisim", k) in ll:
            series[f"dP|{meth}|{k}"] = v - ll[("perisim", k)]
        if meth == "perisim" and k > 0:
            for r in ("lgbm", "logreg"):
                if (r, 0) in ll:
                    series[f"d{r}|{meth}|{k}"] = v - ll[(r, 0)]
        if meth.endswith("@tuned") and (meth[:-6], kp) in ll:
            series[f"dtuned|{meth[:-6]}|0"] = v - ll[(meth[:-6], kp)]
    LOG.info("ksweep %s: patient bootstrap (%d resamples) of %d series", t, nb, len(series))
    ci = M.bootstrap_rowwise(series, subj, nb, cfg.seed)
    ci = {k: (_with_p(v) if k.split("|")[0][0] == "d" else {kk: vv for kk, vv in v.items() if kk != "_boot"})
          for k, v in ci.items()}

    def c(kind: str, m: str, k: int, prefix: str) -> dict:
        v = ci.get(f"{kind}|{m}|{k}")
        if not v:
            return {}
        return {prefix: v["mean"], f"{prefix}_ci_low": v["ci_low"], f"{prefix}_ci_high": v["ci_high"],
                **({f"{prefix}_p": v["p_two_sided"]} if "p_two_sided" in v else {})}

    curves = []
    for (m, k), pt in sorted(point.items(), key=lambda x: (x[0][0], x[0][1])):
        if k <= 0:
            continue
        curves.append({"method": m, "k": k, **pt, **c("ll", m, k, "logloss"), **c("br", m, k, "brier"),
                       **c("rps", m, k, "rps"), **c("d50", m, k, "d_vs_primary_k"), **c("dP", m, k, "d_vs_perisim"),
                       **c("dlgbm", m, k, "d_vs_lgbm"), **c("dlogreg", m, k, "d_vs_logreg"),
                       "complete": f"{m}@{k}" not in incomplete})
    references = [{"method": m, **pt, **c("ll", m, 0, "logloss")} for (m, k), pt in point.items() if k == 0]
    tuned_rows = []
    for m in sorted({m for i in infos for m in i["methods"]}):
        per = {int(i["fold"]): i["methods"][m]["k_star"] for i in infos if m in i["methods"]}
        mine = [r for r in curves if r["method"] == m]
        best = min(mine, key=lambda r: r["logloss"]) if mine else {}
        tuned_rows.append({"method": m, "k_star_by_fold": per, **point.get((f"{m}@tuned", -1), {}),
                           **c("ll", f"{m}@tuned", -1, "logloss"),
                           "logloss_primary_k": point.get((m, kp), {}).get("logloss"),
                           **c("dtuned", m, 0, "d_tuned_vs_primary_k"),
                           "best_k_pooled": best.get("k"), "logloss_best_k_pooled": best.get("logloss"),
                           **({f"d_best_vs_primary_k{s}": best.get(f"d_vs_primary_k{s}")
                               for s in ("", "_ci_low", "_ci_high", "_p")} if best else {})})
    repro = [{"fold": int(i["fold"]), **r} for i in infos for r in i["reproduction"]]
    return {"target": t, "k_primary": kp, "k_grid": sorted({k for i in infos for k in i["k_grid"]}),
            "n_bootstrap": nb, "curves": curves, "references": references, "tuned": tuned_rows,
            "reproduction": repro, "incomplete": incomplete,
            "folds": [{k: v for k, v in i.items() if k not in ("reproduction", "_key")} for i in infos]}


# --------------------------------------------------------------------------- entry point
def run_ksweep(pipe, force: bool = False) -> int:
    cfg, ws = pipe.cfg, pipe.ws
    if not pipe.done("cv"):
        LOG.error("ksweep: the cross-validation of this configuration is not complete; run the pipeline first")
        return 2
    dev = cfg.experiment["cohorts"]["development"]
    methods = [m for m in (cfg.experiment.get("k_sweep") or {}).get("methods", ["perisim", "euclidean"])
               if m == "perisim" or m in cfg.experiment["baselines"]]
    report = {}
    for t in cfg.targets_to_model:
        key = sweep_key(pipe, t)
        out = ws.artifacts() / f"ksweep_{t}.json"
        prev = read_json(out)
        if prev and prev.get("_key") == key and not force:
            LOG.info("ksweep %s: up to date (%s)", t, out)
            report[t] = prev
            continue
        fjs = [r for r in (read_json(p) for p in sorted(ws.artifacts("cv", t).glob("fold*.json")))
               if r and r.get("_key") == pipe.key("cv")]
        folds = sorted(int(r["fold"]) for r in fjs)
        ks = k_grid_for(cfg, min(int(r["n_train"]) for r in fjs))
        common = {"cfg": cfg.to_dict(), "profile": cfg.profile, "variant": cfg.variant, "threads": pipe.threads(),
                  "common_dir": str(ws.common(dev)), "derived_dir": str(ws.derived(dev)),
                  "features": pipe.feature_names(), "folds": str(ws.artifacts() / "folds.parquet"), "target": t,
                  "k_grid": ks, "methods": methods, "key": key}
        infos, jobs = [], []
        for f in folds:
            pd = ws.predictions("ksweep", t)
            j = {**common, "fold": f, "model": str(ws.predictions("cv", t) / f"fold{f}_model.pkl"),
                 "fold_json": str(ws.artifacts("cv", t) / f"fold{f}.json"),
                 "cv_pred": str(ws.predictions("cv", t) / f"fold{f}_pred.parquet"),
                 "out": str(pd / f"fold{f}_pred.parquet"), "info": str(pd / f"fold{f}_info.json")}
            done = read_json(Path(j["info"]))
            if done and done.get("_key") == key and Path(j["out"]).exists() and not force:
                infos.append(done)
            else:
                jobs.append(j)
        LOG.info("ksweep %s: k = %s; methods %s; %d folds to compute", t, ks, methods, len(jobs))
        infos += _run_jobs(jobs, pipe) if jobs else []
        res = aggregate(cfg, ws, t, sorted(infos, key=lambda i: i["fold"]))
        res.update({"_key": key, "cv_key": pipe.key("cv")})
        write_json(out, res)
        _tables(ws, res)
        report[t] = res
    if report:
        (ws.reports / "ksweep.html").write_text(render(report, cfg, ws), encoding="utf-8")
        LOG.info("ksweep: report %s · tables %s", ws.reports / "ksweep.html", ws.root / "results" / "ksweep")
    return 0


def _results_dir(ws) -> Path:
    d = ws.root / "results" / "ksweep"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tables(ws, res: dict) -> None:
    rd, t = _results_dir(ws), res["target"]
    for name, rows in (("k_curve", res["curves"]), ("tuned_k", res["tuned"]), ("references", res["references"]),
                       ("reproduction", res["reproduction"])):
        flat = [{k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows]
        if flat:
            pl.DataFrame(flat, infer_schema_length=None, strict=False).write_csv(rd / f"{t}_{name}.csv")


# --------------------------------------------------------------------------- report
INK, MUTED, GRID, OTHER = "#1d2433", "#6b7c93", "#e6e9ee", "#b9bfc8"
#: fixed colour per method (validated categorical order, light surface); the other weighted kNN are drawn in grey
SLOTS = {"perisim": "#2a78d6", "euclidean": "#eb6834", "hvdm": "#1baf7a", "nca_diag": "#eda100", "rf_gap": "#e87ba4"}
MARKERS = {"perisim": "o", "euclidean": "s", "hvdm": "^", "nca_diag": "D", "rf_gap": "v"}
REF_STYLE = {"lgbm": (0, (5, 3)), "logreg": (0, (1.5, 2))}


def _ax(ax, ks: list[int], kp: int, ylabel: str, title: str) -> None:
    from matplotlib.ticker import NullLocator, ScalarFormatter

    ax.set_xscale("log")
    ticks = [k for k in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if min(ks) <= k <= max(ks)]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(kp, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=0)
    ax.text(kp, 1.0, f" primary k = {kp}", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=7, va="top")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("k (neighbours)", color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


def _legend(ax) -> None:
    """Below the plot, so that it never covers a line or the primary-k label."""
    ax.legend(fontsize=7, frameon=False, ncol=3, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.2))


def _curve_fig(res: dict, field: str, ylabel: str, title: str, band: bool = False, refs: bool = True,
               hline: float | None = None):
    import matplotlib.pyplot as plt

    kp, cur = res["k_primary"], res["curves"]
    meths = sorted({r["method"] for r in cur})
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    others = [m for m in meths if m not in SLOTS and m != "perisim_kernel"]
    for m in others:
        rr = [r for r in cur if r["method"] == m and r.get(field) is not None]
        ax.plot([r["k"] for r in rr], [r[field] for r in rr], color=OTHER, lw=1.0, zorder=1,
                label="other weighted kNN" if m == others[0] else None)
    for m in [m for m in SLOTS if m in meths] + (["perisim_kernel"] if "perisim_kernel" in meths else []):
        rr = [r for r in cur if r["method"] == m and r.get(field) is not None]
        if not rr:
            continue
        xs, ys = [r["k"] for r in rr], [r[field] for r in rr]
        if m == "perisim_kernel":
            ax.plot(xs, ys, color=SLOTS["perisim"], lw=1.4, ls=(0, (4, 2)), zorder=2, label=LABELS[m])
            continue
        if band and m == "perisim" and all(f"{field}_ci_low" in r for r in rr):
            ax.fill_between(xs, [r[f"{field}_ci_low"] for r in rr], [r[f"{field}_ci_high"] for r in rr],
                            color=SLOTS[m], alpha=0.14, lw=0, zorder=1)
        ax.plot(xs, ys, color=SLOTS[m], lw=1.8, marker=MARKERS[m], ms=4.5, mec="white", mew=1.0, zorder=3,
                label=LABELS[m])
    if refs:
        for r in res["references"]:
            if r["method"] in REF_STYLE and r.get(field) is not None:
                ax.axhline(r[field], color=INK, lw=0.9, ls=REF_STYLE[r["method"]], zorder=1,
                           label=f"{LABELS[r['method']]} (no k)")
    if hline is not None:
        ax.axhline(hline, color=INK, lw=0.8, zorder=1)
    _ax(ax, res["k_grid"], kp, ylabel, title)
    _legend(ax)
    return fig


def _diff_fig(res: dict, field: str, ylabel: str, title: str, include: list[str]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for m in include:
        rr = [r for r in res["curves"] if r["method"] == m and r.get(field) is not None]
        if not rr:
            continue
        xs = [r["k"] for r in rr]
        ax.fill_between(xs, [r[f"{field}_ci_low"] for r in rr], [r[f"{field}_ci_high"] for r in rr],
                        color=SLOTS.get(m, OTHER), alpha=0.12, lw=0, zorder=1)
        ax.plot(xs, [r[field] for r in rr], color=SLOTS.get(m, OTHER), lw=1.8, marker=MARKERS.get(m, "o"), ms=4.5,
                mec="white", mew=1.0, zorder=3, label=LABELS[m])
    ax.axhline(0, color=INK, lw=0.8, zorder=2)
    _ax(ax, res["k_grid"], res["k_primary"], ylabel, title)
    _legend(ax)
    return fig


def _ci_txt(r: dict, f: str, nd: int = 3) -> str:
    if r.get(f) is None:
        return "—"
    s = f"{r[f]:.{nd}f}"
    if r.get(f"{f}_ci_low") is not None:
        s += f" [{r[f'{f}_ci_low']:.{nd}f}, {r[f'{f}_ci_high']:.{nd}f}]"
    return s


def _p_txt(p, n_boot: int) -> str:
    if p is None:
        return "—"
    return f"< {1 / n_boot:.4f}" if p == 0 else f"{p:.4f}"


def render(report: dict, cfg, ws) -> str:
    kp = int(cfg.experiment["knn"]["k_primary"])
    sec = [box("Secondary, descriptive analysis added on 2026-10-25, after the provisional results of the first CV "
               f"folds had been seen. The primary k (D11, k = {kp}) and the confirmatory comparisons are unchanged. "
               "Each outer fold's metric is queried again over a wider grid of k, with α refitted per k by the same "
               "patient-level LOO on the training fold and nothing else refitted. Pooled out-of-fold predictions; 95 % "
               "CIs and paired differences by patient bootstrap. Aggregates only (§9)."),
           box("<b>Best k pooled on the test folds</b> is optimistic (chosen on the data it is evaluated on): it shows "
               "how much k could matter at most. <b>k tuned in the training fold</b> (patient-level LOO log-loss, "
               "the same LOO that fits α) is the honest estimate of what tuning k would add.", "warn")]
    for t, res in report.items():
        nb, sev = res["n_bootstrap"], T_LABELS[t][SEVERE[t]]
        sec.append(f"<h2>{esc(t.upper())}</h2>")
        rep = res["reproduction"]
        bad = [r for r in rep if r["method"] in ("perisim", "euclidean", "rf_gap") and r["max_abs_diff"] > 1e-6]
        worst = max((abs(r["d_logloss"]) for r in rep), default=float("nan"))
        if rep:
            sec.append(box(f"Reproduction of the CV on its own grid of k ({len(rep)} fold × method × k checks): "
                           f"largest |Δ log-loss| = {worst:.2g}. " +
                           ("PERISIM, Euclidean and RF-GAP reproduce the CV exactly (|Δp| ≤ 1e-6); the weighted "
                            "comparators differ only through their stored, rounded weights and neighbours tied with "
                            "the k-th." if not bad else
                            f"<b>{len(bad)} checks of PERISIM / Euclidean / RF-GAP differ from the CV</b>: see the "
                            "table at the end."), "ok" if not bad and worst < 1e-3 else "warn"))
        if res["incomplete"]:
            sec.append(box("Point estimates only (not every fold): " + esc(", ".join(res["incomplete"])), "warn"))
        tr = res["tuned"]
        sec.append("<h3>How much does k contribute?</h3>")
        sec.append(table(["method", "k tuned per fold", "log-loss, k tuned", f"log-loss, k = {kp}",
                          f"Δ tuned − k = {kp}", "p", "best k pooled (optimistic)", "log-loss there",
                          f"Δ best − k = {kp}"],
                         [[LABELS.get(r["method"], r["method"]),
                           ", ".join(str(v) for _, v in sorted(r["k_star_by_fold"].items(), key=lambda x: int(x[0]))),
                           _ci_txt(r, "logloss"), r.get("logloss_primary_k"), _ci_txt(r, "d_tuned_vs_primary_k", 4),
                           _p_txt(r.get("d_tuned_vs_primary_k_p"), nb), r.get("best_k_pooled"),
                           r.get("logloss_best_k_pooled"), _ci_txt(r, "d_best_vs_primary_k", 4)]
                          for r in sorted(tr, key=lambda r: r.get("logloss") or 9)], nd=4))
        figs = [("logloss", _curve_fig(res, "logloss", "log-loss (lower is better)",
                                       f"{t.upper()}: log-loss over k (band: PERISIM 95 % CI)", band=True)),
                ("auroc_macro", _curve_fig(res, "auroc_macro", "AUROC, one-vs-rest macro",
                                           f"{t.upper()}: discrimination over k")),
                ("auroc_severe", _curve_fig(res, "auroc_severe", f"AUROC, {sev}", f"{t.upper()}: {sev} over k")),
                ("cal_slope", _curve_fig(res, "cal_slope_mean", "calibration slope (mean one-vs-rest)",
                                         f"{t.upper()}: calibration over k (1 = ideal; < 1 over-confident, "
                                         "> 1 over-smoothed)", refs=False, hline=1.0)),
                ("d_vs_primary_k", _diff_fig(res, "d_vs_primary_k", f"Δ log-loss vs k = {kp}",
                                             f"{t.upper()}: each k against the primary k (below 0: better)",
                                             [m for m in SLOTS])),
                ("d_vs_perisim", _diff_fig(res, "d_vs_perisim", "Δ log-loss, comparator − PERISIM",
                                           f"{t.upper()}: comparators against PERISIM at the same k "
                                           "(above 0: PERISIM better)", [m for m in SLOTS if m != "perisim"]))]
        rd = _results_dir(ws)
        for name, fig in figs:
            fig.savefig(rd / f"{t}_{name}.png", dpi=110, bbox_inches="tight")
            sec.append(fig_img(fig))
        ref_rows = sorted(res["references"], key=lambda r: r["logloss"])
        sec.append("<h3>References without k</h3>")
        sec.append(table(["method", "log-loss [95 % CI]", "Brier", "AUROC macro", f"AUROC {sev}"],
                         [[LABELS.get(r["method"], r["method"]), _ci_txt(r, "logloss"), r["brier"], r["auroc_macro"],
                           r.get("auroc_severe")] for r in ref_rows]))
        sec.append("<h3>Curves (every method and k)</h3>")
        for m in sorted({r["method"] for r in res["curves"]}, key=lambda m: (m not in SLOTS, m)):
            rr = [r for r in res["curves"] if r["method"] == m]
            sec.append(f"<details><summary>{esc(LABELS.get(m, m))}</summary>" + table(
                ["k", "log-loss [95 % CI]", f"Δ vs k = {kp} [95 % CI]", "Δ vs PERISIM, same k [95 % CI]", "Brier",
                 "RPS", "AUROC macro", f"AUROC {sev}", f"slope {sev}", "slope (mean)"],
                [[r["k"], _ci_txt(r, "logloss"), _ci_txt(r, "d_vs_primary_k", 4), _ci_txt(r, "d_vs_perisim", 4),
                  r["brier"], r.get("rps"), r["auroc_macro"], r.get("auroc_severe"), r.get("cal_slope_severe"),
                  r.get("cal_slope_mean")] for r in rr]) + "</details>")
        sec.append("<h3>Reproduction of the CV</h3>")
        agg: dict = {}
        for r in rep:
            a = agg.setdefault(r["method"], {"checks": 0, "max_abs_diff": 0.0, "max_d_logloss": 0.0})
            a["checks"] += 1
            a["max_abs_diff"] = max(a["max_abs_diff"], r["max_abs_diff"])
            a["max_d_logloss"] = max(a["max_d_logloss"], abs(r["d_logloss"]))
        sec.append(table(["method", "fold × k checked", "largest |Δp|", "largest |Δ log-loss|"],
                         [[LABELS.get(m, m), a["checks"], f"{a['max_abs_diff']:.2g}", f"{a['max_d_logloss']:.2g}"]
                          for m, a in sorted(agg.items())]))
        secs = {m: round(sum(i["seconds"].get(m, 0) for i in res["folds"])) for i in res["folds"]
                for m in i["seconds"]}
        sec.append(f"<p class='muted'>k grid: {esc(res['k_grid'])} · compute time summed over folds (s): "
                   f"{esc(secs)} · CV key {esc(res.get('cv_key'))}</p>")
    return page("PERISIM — k sweep (secondary)", sec, "PERISIM · internal validation · aggregates only")
