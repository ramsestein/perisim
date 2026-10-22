"""Aggregation of results (out-of-fold predictions) and TRIPOD+AI aggregate report (§6.5, §6.6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from perisim.evaluation import metrics as M
from perisim.evaluation.html import box, fig_img, page, table
from perisim.experiment import ORDINAL, SEVERE
from perisim.utils import read_json

T_LABELS = {"t1": ["none", "mild", "severe"], "t2": ["short", "medium", "long"],
            "t3": ["early", "expected", "prolonged/mild", "severe/death"], "t1_bin": ["not severe", "severe"]}


def _probs(df: pl.DataFrame) -> np.ndarray:
    cols = sorted([c for c in df.columns if c.startswith("p") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    P = df.select(cols).to_numpy()
    keep = ~np.isnan(P).all(0)
    return P[:, keep]


def _rows(df: pl.DataFrame) -> pl.DataFrame:
    return df.select("stay_id", "subject_id", "y").with_columns(
        pl.Series("ll", M.row_logloss(_probs(df), df["y"].to_numpy())))


def evaluate_predictions(pred: pl.DataFrame, target: str, k_primary: int, n_boot: int, seed: int,
                         ref: tuple[str, int | None] = ("perisim", None)) -> dict:
    """Metrics per (method, k) on the aggregated predictions, with patient-level bootstrap CIs and paired
    log-loss differences against the reference method (default PERISIM at the primary k).

    Paired differences are aligned by stay (inner join), so a comparator that failed in some folds is compared
    only on the stays it predicted, and it is listed under `incomplete` (AUDIT E2)."""
    ref_m, ref_k = ref[0], (k_primary if ref[1] is None else int(ref[1]))
    out: dict = {"reference": f"{ref_m}@{ref_k}", "methods": {}, "paired_vs_reference": {}, "binary_severe": {},
                 "paired_binary_vs_reference": {}, "incomplete": {}}
    sev = SEVERE[target]
    refdf = pred.filter((pl.col("method") == ref_m) & (pl.col("k") == ref_k)).sort("stay_id")
    ref_rows = _rows(refdf) if refdf.height else None
    for (meth, k), g in pred.group_by(["method", "k"], maintain_order=True):
        g = g.sort("stay_id")
        P, y, subj = _probs(g), g["y"].to_numpy(), g["subject_id"].to_numpy()
        key = f"{meth}@{k}" if k else meth
        if ref_rows is not None and g.height < refdf.height:
            out["incomplete"][key] = {"n": int(g.height), "n_reference": int(refdf.height)}
        if meth == "swift":
            yb = y if target == "t1_bin" else (y == sev).astype(int)
            out["binary_severe"][key] = _binary_block(P[:, 1], yb, subj, n_boot, seed)
            if refdf.height:
                rp, ry = _probs(refdf), refdf["y"].to_numpy()
                ryb = ry if target == "t1_bin" else (ry == sev).astype(int)
                out["binary_severe"][out["reference"]] = _binary_block(rp[:, sev], ryb, refdf["subject_id"].to_numpy(),
                                                                       n_boot, seed)
                a = pl.DataFrame({"stay_id": g["stay_id"], "subject_id": g["subject_id"],
                                  "ll": M.row_logloss(np.column_stack([1 - P[:, 1], P[:, 1]]), yb)})
                b = pl.DataFrame({"stay_id": refdf["stay_id"],
                                  "ll_ref": M.row_logloss(np.column_stack([1 - rp[:, sev], rp[:, sev]]), ryb)})
                j = a.join(b, on="stay_id", how="inner")
                if j.height:
                    d = M.paired_difference(j["ll"].to_numpy(), j["ll_ref"].to_numpy(), j["subject_id"].to_numpy(),
                                            n_boot, seed)
                    out["paired_binary_vs_reference"][key] = {**d, "n_common": int(j.height)}
            continue
        s = M.summary(P, y, ORDINAL[target], sev)
        rows = {"logloss": M.row_logloss(P, y), "brier": M.row_brier(P, y)}
        if ORDINAL[target]:
            rows["rps"] = M.row_rps(P, y)
        ci = M.bootstrap_rowwise(rows, subj, n_boot, seed)
        for v in ci.values():
            v.pop("_boot", None)
        s["ci"] = ci
        if (k == k_primary or k == 0) and sev is not None and 0 < (y == sev).sum() < len(y):
            from sklearn.metrics import average_precision_score, roc_auc_score

            s["ci"]["auroc_severe"] = M.bootstrap_metric(lambda p, yy: roc_auc_score(yy == sev, p[:, sev]), P, y, subj,
                                                         min(n_boot, 500), seed)
            s["ci"]["auprc_severe"] = M.bootstrap_metric(lambda p, yy: average_precision_score(yy == sev, p[:, sev]), P,
                                                         y, subj, min(n_boot, 500), seed)
        out["methods"][key] = s
        if ref_rows is not None and (meth, int(k)) != (ref_m, ref_k) and (k == k_primary or k == 0):
            j = _rows(g).join(ref_rows.select("stay_id", pl.col("ll").alias("ll_ref")), on="stay_id", how="inner")
            if j.height:
                d = M.paired_difference(j["ll"].to_numpy(), j["ll_ref"].to_numpy(), j["subject_id"].to_numpy(),
                                        n_boot, seed)
                out["paired_vs_reference"][key] = {**d, "n_common": int(j.height)}
    return out


def paired_between(pred: pl.DataFrame, a: tuple[str, int], b: tuple[str, int], n_boot: int, seed: int) -> dict | None:
    """Paired log-loss difference a − b on the common stays (e.g. mode B PERISIM vs mode B Euclidean)."""
    ga = pred.filter((pl.col("method") == a[0]) & (pl.col("k") == a[1]))
    gb = pred.filter((pl.col("method") == b[0]) & (pl.col("k") == b[1]))
    if not ga.height or not gb.height:
        return None
    j = _rows(ga).join(_rows(gb).select("stay_id", pl.col("ll").alias("ll_b")), on="stay_id", how="inner")
    d = M.paired_difference(j["ll"].to_numpy(), j["ll_b"].to_numpy(), j["subject_id"].to_numpy(), n_boot, seed)
    return {**d, "n_common": int(j.height), "a": f"{a[0]}@{a[1]}", "b": f"{b[0]}@{b[1]}"}


def _binary_block(p: np.ndarray, y: np.ndarray, subj: np.ndarray, n_boot: int, seed: int) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    P2 = np.column_stack([1 - p, p])
    ll = M.row_logloss(P2, y)
    br = M.row_brier_binary(p, y)
    ci = M.bootstrap_rowwise({"logloss": ll, "brier": br}, subj, n_boot, seed)
    for v in ci.values():
        v.pop("_boot", None)
    out = {"n": int(len(y)), "logloss": float(ll.mean()), "brier": float(br.mean()), "ci": ci}
    if 0 < y.sum() < len(y):
        out["auroc"] = M.bootstrap_metric(lambda pp, yy: roc_auc_score(yy, pp[:, 1]), P2, y, subj, min(n_boot, 500), seed)
        out["auprc"] = M.bootstrap_metric(lambda pp, yy: average_precision_score(yy, pp[:, 1]), P2, y, subj,
                                          min(n_boot, 500), seed)
        a, b = M.calibration_line(p, y.astype(float))
        out["calibration"] = {"intercept": a, "slope": b}
    return out


def p4_metrics(p4: pl.DataFrame) -> dict:
    out = {}
    if p4.height == 0:
        return out
    for (meth, var), g in p4.group_by(["method", "variable"], maintain_order=True):
        g = g.filter(pl.col("obs").is_not_nan() & pl.col("obs").is_not_null() & pl.col("pred").is_not_nan())
        if g.height == 0:
            continue
        p, o = g["pred"].to_numpy(), g["obs"].to_numpy()
        if var.startswith("los"):
            out.setdefault(var, {})[meth] = {"mae_days": float(np.mean(np.abs(p - o))), "n": int(g.height)}
        else:
            P2 = np.column_stack([1 - np.clip(p, 1e-6, 1 - 1e-6), np.clip(p, 1e-6, 1 - 1e-6)])
            yy = (o > 0.5).astype(int)
            out.setdefault(var, {})[meth] = {"logloss": float(M.row_logloss(P2, yy).mean()),
                                             "brier": float(M.row_brier_binary(P2[:, 1], yy).mean()), "n": int(g.height),
                                             "prevalence": (float(yy.mean()) if yy.sum() >= 10 and
                                                            len(yy) - yy.sum() >= 10 else None)}
    return out


def aggregate_cv(art_dir: Path, pred_dir: Path, targets: list[str], cfg, tag: str = "cv",
                 expected_key: str | None = None, expected_folds: set[int] | None = None) -> dict:
    """Pools the out-of-fold predictions of the CURRENT run only: fold files whose key or fold number does not
    match (left over from a previous configuration) are ignored (AUDIT E15)."""
    kp = int(cfg.experiment["knn"]["k_primary"])
    nb = int(cfg.experiment["evaluation"]["n_bootstrap"])
    res: dict = {"targets": {}}
    for t in targets:
        fj = []
        for f in sorted((art_dir / tag / t).glob("fold*.json")):
            r = read_json(f)
            if not r or (expected_key and r.get("_key") != expected_key) or \
                    (expected_folds is not None and int(r.get("fold", -1)) not in expected_folds):
                continue
            fj.append(r)
        if not fj:
            continue
        ids = [int(f["fold"]) for f in fj]
        rd = pred_dir / tag / t
        pred = pl.concat([pl.read_parquet(rd / f"fold{i}_pred.parquet") for i in ids], how="diagonal_relaxed")
        p4 = [pl.read_parquet(rd / f"fold{i}_p4.parquet") for i in ids if (rd / f"fold{i}_p4.parquet").exists()]
        p4 = pl.concat([x for x in p4 if x.height], how="diagonal_relaxed") if any(x.height for x in p4) else pl.DataFrame()
        ood = pl.concat([pl.read_parquet(rd / f"fold{i}_ood.parquet") for i in ids])
        ev = evaluate_predictions(pred, t, kp, nb, cfg.seed)
        errors: dict[str, list[int]] = {}
        for f in fj:
            for name, info in (f.get("baselines") or {}).items():
                if isinstance(info, dict) and info.get("error"):
                    errors.setdefault(name, []).append(int(f["fold"]))
        ev["comparator_errors_by_fold"] = errors
        imp_disp = _importance_dispersion(fj)
        res["targets"][t] = {
            "evaluation": ev, "p4": p4_metrics(p4), "ood_rate": float(ood["ood"].mean()),
            "folds": [{"fold": i, "n_train": f["n_train"], "n_test": f["n_test"], "m": f["perisim"]["m"],
                       "lambda": f["perisim"]["lambda"], "lambda_grid_loss": f["perisim"].get("lambda_grid_loss"),
                       "alpha": f["perisim"]["alpha"], "n_accepted": f["importance_loop"]["n_accepted"],
                       "n_attempts": f["importance_loop"]["n_attempts"], "stop_reason": f["importance_loop"]["stop_reason"],
                       "plateau": f["importance_loop"]["plateau"],
                       "plateau_above_tol": f["importance_loop"]["plateau_above_tol"],
                       "delta": f["importance_loop"]["delta"], "delta_smooth": f["importance_loop"]["delta_smooth"],
                       "spearman_W": f["importance_loop"]["spearman_W"], "seconds": f["seconds"],
                       "ood_rate_test": f["ood_rate_test"], "thresholds": f.get("thresholds")}
                      for f in fj for i in [int(f["fold"])]],
            "importance": imp_disp,
            "weights_mean": _mean_dict([f["perisim"]["weights"] for f in fj]),
        }
    return res


def _mean_dict(ds: list[dict]) -> dict:
    keys = sorted({k for d in ds for k in d})
    return {k: float(np.mean([d[k] for d in ds if k in d])) for k in keys}


def _importance_dispersion(fj: list[dict]) -> dict:
    """Group importance with its dispersion across accepted models (all folds)."""
    rows: dict[str, list[float]] = {}
    agg: dict[str, list[float]] = {}
    for f in fj:
        loop = f["importance_loop"]
        names = loop["group_names"]
        mm = np.array(loop["model_matrix"]) if loop["model_matrix"] else np.zeros((0, len(names)))
        for j, g in enumerate(names):
            rows.setdefault(g, []).extend(mm[:, j].tolist() if mm.size else [])
            agg.setdefault(g, []).append(loop["group_importance"][g])
    out = {}
    for g, vals in rows.items():
        v = np.array(vals) if vals else np.array([np.nan])
        out[g] = {"aggregated_mean": float(np.mean(agg[g])), "models_mean": float(np.nanmean(v)),
                  "models_sd": float(np.nanstd(v)), "models_p10": float(np.nanpercentile(v, 10)),
                  "models_p90": float(np.nanpercentile(v, 90)), "n_models": int(np.isfinite(v).sum())}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["aggregated_mean"]))


# --------------------------------------------------------------------------- HTML
def _ci(d: dict | None) -> str:
    return f"{d['ci_low']:.3f}–{d['ci_high']:.3f}" if d and d.get("ci_low") is not None else "—"


def _p(p: float | None, n_boot: int) -> str:
    if p is None:
        return "—"
    return f"< {1 / n_boot:.4f}" if p == 0 else f"{p:.4f}"


def eval_sections(ev: dict, n_boot: int) -> list[str]:
    """Tables shared by the internal and external reports: metrics with CIs, calibration, paired differences
    against the reference, incomplete comparators and the severe / not-severe block (SWIFT)."""
    sec = []
    rows = []
    for key, s in ev["methods"].items():
        ci = s["ci"]
        rows.append([key, s["n"], ci["logloss"]["mean"], _ci(ci["logloss"]), ci["brier"]["mean"], _ci(ci["brier"]),
                     ci.get("rps", {}).get("mean"), s.get("auroc_ovr_macro"), s.get("auroc_severe"),
                     _ci(ci.get("auroc_severe")), s.get("auprc_severe")])
    sec.append(table(["Method@k", "n", "Log-loss", "95 % CI", "Brier (Σ classes)", "95 % CI", "RPS", "AUROC OvR",
                      "AUROC severe", "95 % CI", "AUPRC severe"], rows))
    cal_rows = [[key, c, v["intercept"], v["slope"], v["citl"], v["ece"]]
                for key, s in ev["methods"].items() if key.endswith("@" + ev["reference"].split("@")[1]) or "@" not in key
                for c, v in s.get("calibration", {}).items()]
    if cal_rows:
        sec.append("<h3>One-vs-rest calibration (primary k and parametric models)</h3>")
        sec.append(table(["Method", "class", "intercept", "slope", "CITL", "ECE"], cal_rows))
    if ev.get("incomplete"):
        sec.append(box("Incomplete methods (fewer stays than the reference): " +
                       "; ".join(f"{k}: {v['n']} / {v['n_reference']}" for k, v in ev["incomplete"].items()), "warn"))
    if ev.get("paired_vs_reference"):
        sec.append(f"<h3>Paired log-loss difference vs {ev['reference']} (positive = worse than the reference)</h3>")
        sec.append(table(["Method", "Δ log-loss", "95 % CI", "p", "common stays"],
                         [[k, v["mean"], _ci(v), _p(v.get("p_two_sided"), n_boot), v.get("n_common")]
                          for k, v in ev["paired_vs_reference"].items()], nd=4))
    if ev.get("binary_severe"):
        sec.append("<h3>Severe / not severe (SWIFT vs the reference; binary Brier)</h3>")
        sec.append(box("SWIFT was developed for unplanned ICU readmission / death after ICU discharge; here it is "
                       "recalibrated on the training fold for the severe class of the target — a different estimand.",
                       "note"))
        sec.append(table(["Method", "n", "Log-loss", "95 % CI", "Brier", "AUROC", "AUPRC", "Cal. slope"],
                         [[k, v["n"], v["logloss"], _ci(v["ci"]["logloss"]), v["brier"], v.get("auroc", {}).get("mean"),
                           v.get("auprc", {}).get("mean"), v.get("calibration", {}).get("slope")]
                          for k, v in ev["binary_severe"].items()]))
        for k, v in ev.get("paired_binary_vs_reference", {}).items():
            sec.append(table([f"Paired binary log-loss {k} − reference", "Δ", "95 % CI", "p", "common stays"],
                             [["", v["mean"], _ci(v), _p(v.get("p_two_sided"), n_boot), v.get("n_common")]], nd=4))
    return sec


def render_results(res: dict, cfg, title: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kp = int(cfg.experiment["knn"]["k_primary"])
    sec = [box("Nested patient-level cross-validation (INSPIRE). Aggregated out-of-fold predictions; 95 % CIs by "
               "patient bootstrap. Aggregates only (§9).")]
    for t, r in res["targets"].items():
        ev = r["evaluation"]
        sec.append(f"<h2>{t.upper()}</h2>")
        sec.extend(eval_sections(ev, int(cfg.experiment["evaluation"]["n_bootstrap"])))
        if ev.get("comparator_errors_by_fold"):
            sec.append(box("Comparators that FAILED in some folds (evaluated only on the stays they predicted): " +
                           "; ".join(f"{k}: folds {v}" for k, v in ev["comparator_errors_by_fold"].items()), "warn"))
        # k curve
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        meths = sorted({k.split("@")[0] for k in ev["methods"] if "@" in k})
        for m_ in meths:
            ks = sorted(int(k.split("@")[1]) for k in ev["methods"] if k.startswith(m_ + "@"))
            ax.plot(ks, [ev["methods"][f"{m_}@{k}"]["logloss"] for k in ks], marker="o", label=m_)
        ax.set_xscale("log")
        ax.set_xlabel("k (run-time parameter)")
        ax.set_ylabel("log-loss")
        ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"{t.upper()}: performance across the k grid")
        sec.append(fig_img(fig))
        # PERISIM calibration
        cal = ev["methods"].get(f"perisim@{kp}", {}).get("calibration", {})
        if cal:
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            for c, cc in cal.items():
                xs = [b["p_mean"] for b in cc["curve"]]
                ys = [b["obs"] for b in cc["curve"]]
                lab = T_LABELS[t][int(c)] if int(c) < len(T_LABELS[t]) else c
                ax.plot(xs, ys, marker="o", label=f"{lab} (slope {cc['slope']:.2f})")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("predicted probability")
            ax.set_ylabel("observed frequency")
            ax.legend(fontsize=7)
            ax.set_title(f"PERISIM@{kp} calibration (one-vs-rest)")
            sec.append(fig_img(fig))
        # convergence
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for f in r["folds"]:
            ax.plot(range(1, len(f["delta"]) + 1), f["delta"], alpha=0.3, lw=0.8)
            ax.plot(range(1, len(f["delta_smooth"]) + 1), f["delta_smooth"], lw=1.6, label=f"fold {f['fold']} ({f['stop_reason']})")
        ax.set_yscale("log")
        ax.set_xlabel("accepted models")
        ax.set_ylabel("Δ (thin) and Δ̃ (thick)")
        ax.legend(fontsize=7)
        ax.set_title("Metric convergence (§5.3.2)")
        sec.append(fig_img(fig))
        sec.append(table(["Fold", "n train", "n test", "m", "λ", "accepted models", "attempts", "stop reason", "Δ̃ plateau",
                          "plateau > tol", "OOD test", "seconds"],
                         [[f["fold"], f["n_train"], f["n_test"], f["m"], f["lambda"], f["n_accepted"], f["n_attempts"],
                           f["stop_reason"], f["plateau"], f["plateau_above_tol"], f["ood_rate_test"], f["seconds"]]
                          for f in r["folds"]], nd=4))
        imp = list(r["importance"].items())[:30]
        sec.append("<h3>Consensus SAGE importance (groups) and dispersion across models</h3>")
        sec.append(table(["Group", "aggregated", "model mean", "SD", "p10", "p90", "n models"],
                         [[g, v["aggregated_mean"], v["models_mean"], v["models_sd"], v["models_p10"], v["models_p90"],
                           v["n_models"]] for g, v in imp], nd=4))
        if r["p4"]:
            rows = []
            for var, d in r["p4"].items():
                for meth, v in d.items():
                    rows.append([var, meth, v.get("logloss"), v.get("brier"), v.get("mae_days"), v.get("prevalence"), v["n"]])
            sec.append("<h3>Q4: future variables carried by the neighborhood (primary k)</h3>")
            sec.append(table(["Variable", "Method", "Log-loss", "Brier", "MAE days", "Prevalence", "n"], rows))
    return page(title, sec, "PERISIM · TRIPOD+AI · aggregates only")
