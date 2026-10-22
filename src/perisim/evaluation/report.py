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


def evaluate_predictions(pred: pl.DataFrame, target: str, k_primary: int, n_boot: int, seed: int) -> dict:
    """Metrics per (method, k) on the aggregated predictions, with patient-level bootstrap CIs and paired
    differences against PERISIM at the primary k."""
    out: dict = {"methods": {}, "paired_vs_perisim": {}, "binary_severe": {}}
    sev = SEVERE[target]
    ref = pred.filter((pl.col("method") == "perisim") & (pl.col("k") == k_primary)).sort("stay_id")
    ref_ll = M.row_logloss(_probs(ref), ref["y"].to_numpy()) if ref.height else None
    for (meth, k), g in pred.group_by(["method", "k"], maintain_order=True):
        g = g.sort("stay_id")
        P, y, subj = _probs(g), g["y"].to_numpy(), g["subject_id"].to_numpy()
        key = f"{meth}@{k}" if k else meth
        if meth == "swift":
            yb = (y == sev).astype(int)
            out["binary_severe"][key] = _binary_block(P[:, 1], yb, subj, n_boot, seed)
            if ref.height:
                out["binary_severe"]["perisim@%d" % k_primary] = _binary_block(_probs(ref)[:, sev], (ref["y"].to_numpy() == sev).astype(int),
                                                                             ref["subject_id"].to_numpy(), n_boot, seed)
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
        if ref_ll is not None and meth != "perisim" and (k == k_primary or k == 0) and g.height == ref.height:
            other = M.row_logloss(P, y)
            out["paired_vs_perisim"][key] = M.paired_difference(other, ref_ll, subj, n_boot, seed)
    return out


def _binary_block(p: np.ndarray, y: np.ndarray, subj: np.ndarray, n_boot: int, seed: int) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    P2 = np.column_stack([1 - p, p])
    ll = M.row_logloss(P2, y)
    br = M.row_brier(P2, y)
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
                                             "brier": float(M.row_brier(P2, yy).mean()), "n": int(g.height),
                                             "prevalence": float(yy.mean())}
    return out


def aggregate_cv(art_dir: Path, pred_dir: Path, targets: list[str], cfg, tag: str = "cv") -> dict:
    kp = int(cfg.experiment["knn"]["k_primary"])
    nb = int(cfg.experiment["evaluation"]["n_bootstrap"])
    res: dict = {"targets": {}}
    for t in targets:
        folds = sorted((art_dir / tag / t).glob("fold*.json"))
        if not folds:
            continue
        fj = [read_json(f) for f in folds]
        pred = pl.concat([pl.read_parquet(p) for p in sorted((pred_dir / tag / t).glob("fold*_pred.parquet"))],
                         how="diagonal_relaxed")
        p4 = [pl.read_parquet(p) for p in sorted((pred_dir / tag / t).glob("fold*_p4.parquet"))]
        p4 = pl.concat([x for x in p4 if x.height], how="diagonal_relaxed") if any(x.height for x in p4) else pl.DataFrame()
        ood = pl.concat([pl.read_parquet(p) for p in sorted((pred_dir / tag / t).glob("fold*_ood.parquet"))])
        ev = evaluate_predictions(pred, t, kp, nb, cfg.seed)
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
                       "ood_rate_test": f["ood_rate_test"], "thresholds": f.get("thresholds")} for i, f in enumerate(fj)],
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
        rows = []
        for key, s in ev["methods"].items():
            ci = s["ci"]
            rows.append([key, s["n"], ci["logloss"]["mean"], f"{ci['logloss']['ci_low']:.3f}–{ci['logloss']['ci_high']:.3f}",
                         ci["brier"]["mean"], ci.get("rps", {}).get("mean"), s.get("auroc_ovr_macro"),
                         s.get("auroc_severe"), s.get("auprc_severe")])
        sec.append(table(["Method@k", "n", "Log-loss", "95 % CI", "Brier", "RPS", "AUROC OvR", "AUROC severe",
                          "AUPRC severe"], rows))
        if ev["paired_vs_perisim"]:
            sec.append("<h3>Paired log-loss difference vs PERISIM@%d (positive = worse than PERISIM)</h3>" % kp)
            sec.append(table(["Method", "Δ log-loss", "CI low", "CI high", "p"],
                             [[k, v["mean"], v["ci_low"], v["ci_high"], v["p_two_sided"]] for k, v in ev["paired_vs_perisim"].items()]))
        if ev["binary_severe"]:
            sec.append("<h3>Severe / not severe (includes SWIFT)</h3>")
            sec.append(table(["Method", "n", "Log-loss", "Brier", "AUROC", "AUPRC", "Cal. slope"],
                             [[k, v["n"], v["logloss"], v["brier"], v.get("auroc", {}).get("mean"),
                               v.get("auprc", {}).get("mean"), v.get("calibration", {}).get("slope")]
                              for k, v in ev["binary_severe"].items()]))
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
            sec.append("<h3>P4: future variables carried by the neighborhood (primary k)</h3>")
            sec.append(table(["Variable", "Metric", "Log-loss", "Brier", "MAE days", "Prevalence", "n"], rows))
    return page(title, sec, "PERISIM · TRIPOD+AI · aggregates only")
