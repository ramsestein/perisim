"""Aggregate report of the internal validation (Phase 2): <workdir>/reports/cv[_quick].md / .json.

Per development site, on the landmarks every method predicted (out of fold): AUROC, log-loss, Brier, calibration in
the large and slope, for PERISIM (one line per partner site: the metric learned with it) and each local comparator;
then PERISIM − comparator, paired, in log-loss and AUROC. 95 % CIs by patient-level bootstrap (the same resamples
for every method of a site). PERISIM's weights and shapes are summarised per rotation over the folds. Aggregates
only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from perisim.evaluation.metrics import calibration_in_the_large, calibration_line

from .config import load, workdir
from .cv import Sites, cv_dir

LOG = logging.getLogger("hypo.report")
NAMES = {"last_map": "last MAP (logistic)", "logreg": "logistic, all inputs", "lightgbm": "LightGBM",
         "euclidean": "Euclidean kNN", "prevalence": "prevalence"}


# ---------------------------------------------------------------------------- weighted AUROC for the bootstrap
def auc_prep(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort order and tie groups of the scores, computed once per method."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    return order, np.r_[0, np.cumsum(s[1:] != s[:-1])]


def auc_weighted(y: np.ndarray, w: np.ndarray, prep: tuple[np.ndarray, np.ndarray]) -> float:
    """AUROC with row weights (ties count ½), O(n) given `auc_prep`."""
    order, grp = prep
    yo, wo = y[order].astype(float), w[order]
    pos = np.bincount(grp, weights=wo * yo)
    neg = np.bincount(grp, weights=wo * (1 - yo))
    below = np.cumsum(neg) - neg
    den = pos.sum() * neg.sum()
    return float((pos * (below + 0.5 * neg)).sum() / den) if den > 0 else float("nan")


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ci(b: np.ndarray) -> list[float]:
    b = b[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b.size else [float("nan")] * 2


def site_metrics(P: dict[str, np.ndarray], y: np.ndarray, subjects: np.ndarray, n_boot: int, seed: int,
                 reference: list[str]) -> dict:
    """Metrics of every method in P (same rows) and paired differences reference − others, patient bootstrap."""
    _, inv = np.unique(subjects.astype(str), return_inverse=True)
    S = int(inv.max()) + 1
    ll = {m: _ll(p, y) for m, p in P.items()}
    prep = {m: auc_prep(p) for m, p in P.items()}
    rng = np.random.default_rng(seed)
    B = {m: {"auroc": [], "logloss": []} for m in P}
    for _ in range(n_boot):
        w = rng.multinomial(S, np.full(S, 1.0 / S)).astype(float)[inv]
        for m in P:
            B[m]["auroc"].append(auc_weighted(y, w, prep[m]))
            B[m]["logloss"].append(float((w * ll[m]).sum() / w.sum()))
    B = {m: {k: np.asarray(v) for k, v in d.items()} for m, d in B.items()}
    one = np.ones(len(y))
    out = {"n_landmarks": int(len(y)), "n_patients": S, "prevalence": float(y.mean()), "methods": {}, "paired": {}}
    for m, p in P.items():
        a, b = calibration_line(p, y) if m != "prevalence" else (float("nan"), float("nan"))
        out["methods"][m] = {"auroc": auc_weighted(y, one, prep[m]), "auroc_ci": _ci(B[m]["auroc"]),
                             "logloss": float(ll[m].mean()), "logloss_ci": _ci(B[m]["logloss"]),
                             "brier": float(np.mean((p - y) ** 2)), "citl": calibration_in_the_large(p, y),
                             "slope": b, "mean_p": float(p.mean())}
    for r in reference:
        for m in P:
            if m == r or m in reference:
                continue
            d_ll = B[r]["logloss"] - B[m]["logloss"]
            d_auc = B[r]["auroc"] - B[m]["auroc"]
            out["paired"][f"{r} − {m}"] = {
                "logloss": out["methods"][r]["logloss"] - out["methods"][m]["logloss"], "logloss_ci": _ci(d_ll),
                "auroc": out["methods"][r]["auroc"] - out["methods"][m]["auroc"], "auroc_ci": _ci(d_auc)}
    return out


def rotation_summary(infos: list[dict]) -> dict:
    W = {}
    for i in infos:
        for n, w in i["weights"].items():
            W.setdefault(n, []).append(w)
    shaped = {n: sum(i["shaped"].get(n, False) for i in infos) for n in W}
    return {"folds": len(infos), "development": infos[0]["development"],
            "weights": {n: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                        for n, v in sorted(W.items(), key=lambda x: -np.mean(x[1]))},
            "shaped_folds": shaped, "dropped_site_constant": sorted({d for i in infos for d in i["dropped_site_constant"]}),
            "n_accepted": [i["n_accepted"] for i in infos], "minutes": float(sum(i["seconds"] for i in infos) / 60),
            "alpha": {c: [i["sites"][c]["alpha"] for i in infos] for c in infos[0]["sites"]},
            "code_hashes": sorted({i["code_hash"] for i in infos})}


def report(quick: bool = False) -> dict:
    cfg = load()
    d = cv_dir(quick)
    S = Sites(cfg, d)
    n_boot = int(cfg["cv"]["n_boot"])
    res = {"quick": quick, "sites": {}, "rotations": {}, "code_hashes": set()}
    for ext in cfg["cohorts"]:
        files = sorted(d.glob(f"perisim_ext-{ext}_f*.parquet"))
        if files:
            infos = [json.loads(f.with_suffix(".json").read_text(encoding="utf-8")) for f in files]
            res["rotations"][f"external-{ext}"] = rotation_summary(infos)
    for c in cfg["cohorts"]:
        if not (d / f"folds_{c}.parquet").exists():
            continue
        lm = S.lm(c)
        preds: dict[str, pl.DataFrame] = {}
        for ext in cfg["cohorts"]:
            fs = sorted(d.glob(f"perisim_ext-{ext}_f*.parquet"))
            parts = [pl.read_parquet(f).filter(pl.col("site") == c) for f in fs]
            parts = [p for p in parts if p.height]
            if parts:
                partner = [x for x in cfg["cohorts"] if x not in (ext, c)][0]
                preds[f"PERISIM (with {partner})"] = pl.concat(parts).select("row", "p")
        for name in cfg["cv"]["comparators"]:
            fs = sorted(d.glob(f"comp_{c}_f*_{name}.parquet"))
            if fs:
                preds[name] = pl.concat([pl.read_parquet(f) for f in fs]).select("row", "p")
        if not preds:
            continue
        rows = None
        for p in preds.values():
            r = set(p["row"].to_list())
            rows = r if rows is None else rows & r
        rows = np.array(sorted(rows))
        P = {m: p.sort("row").filter(pl.col("row").is_in(rows))["p"].to_numpy() for m, p in preds.items()}
        ref = [m for m in P if m.startswith("PERISIM")]
        res["sites"][c] = site_metrics(P, lm.y[rows], lm.ft.subjects[rows], n_boot, int(cfg["seed"]), ref)
        res["sites"][c]["folds"] = sorted({int(x) for x in S.fold(c)[rows]})
        LOG.info("%s: %d landmarks, %d methods", c, rows.size, len(P))
    for r in res["rotations"].values():
        res["code_hashes"] |= set(r["code_hashes"])
    res["code_hashes"] = sorted(res["code_hashes"])
    rep = workdir() / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    name = "cv_quick" if quick else "cv"
    (rep / f"{name}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (rep / f"{name}.md").write_text(render(res), encoding="utf-8")
    LOG.info("report: %s", rep / f"{name}.md")
    return res


def _f(x, nd=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def render(res: dict) -> str:
    L = ["# `hypo` Phase 2: internal validation by rotation" + (" (quick)" if res["quick"] else ""), "",
         "Out-of-fold predictions of each development site; PERISIM is shown once per partner site (the metric "
         "learned on both sites' training patients, each site queried with its own case base); comparators are "
         f"local. 95 % CIs by patient bootstrap. Code: {', '.join(res['code_hashes']) or '—'}.", ""]
    for c, s in res["sites"].items():
        L += [f"## {c}: {s['n_landmarks']:,} landmarks, {s['n_patients']:,} patients, prevalence "
              f"{s['prevalence']:.1%} (folds {s['folds']})", "",
              "| method | AUROC (95 % CI) | log-loss (95 % CI) | Brier | CITL | slope | mean p |",
              "|---|---|---|---|---|---|---|"]
        for m, r in s["methods"].items():
            L.append(f"| {NAMES.get(m, m)} | {_f(r['auroc'])} ({_f(r['auroc_ci'][0])}–{_f(r['auroc_ci'][1])}) | "
                     f"{_f(r['logloss'], 4)} ({_f(r['logloss_ci'][0], 4)}–{_f(r['logloss_ci'][1], 4)}) | "
                     f"{_f(r['brier'], 4)} | {_f(r['citl'])} | {_f(r['slope'])} | {_f(r['mean_p'])} |")
        L += ["", "| paired | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |", "|---|---|---|"]
        for k, r in s["paired"].items():
            a, b = k.split(" − ")
            L.append(f"| {a} − {NAMES.get(b, b)} | {_f(r['logloss'], 4)} ({_f(r['logloss_ci'][0], 4)} to "
                     f"{_f(r['logloss_ci'][1], 4)}) | {_f(r['auroc'])} ({_f(r['auroc_ci'][0])} to "
                     f"{_f(r['auroc_ci'][1])}) |")
        L.append("")
    for k, r in res["rotations"].items():
        L += [f"## Rotation {k}: metric learned on {' + '.join(r['development'])} ({r['folds']} folds, "
              f"{r['minutes']:.0f} min)", "",
              f"Accepted models per fold {r['n_accepted']}; α per site {r['alpha']}; left out of the metric "
              f"(constant within each site): {', '.join(r['dropped_site_constant']) or 'none'}.", "",
              "| variable | weight, mean (min–max over folds) | shaped in folds |", "|---|---|---|"]
        for n, w in r["weights"].items():
            L.append(f"| {n} | {w['mean']:.3f} ({w['min']:.3f}–{w['max']:.3f}) | {r['shaped_folds'][n]}/{r['folds']} |")
        L.append("")
    return "\n".join(L) + "\n"
