"""Post-hoc, EXPLORATORY transport analysis (not pre-registered: added on 2026-10-28, after the external validation and
the transport analyses had been seen, at the lead's request). Two questions:

1. How much does PERISIM lose when it is transported, compared with the parametric comparators of the CV (LightGBM
   with temperature scaling, multinomial logistic regression), and how: base rates, spread or ranking? Per external
   cohort, on the fixed test set of the transport stage, every method in four situations:
   - trained on INSPIRE and applied as it is;
   - recalibrated with n ∈ transport.tuning_sizes local patients (the nested tuning subsets of analysis D):
     (a) intercepts only (class offsets: the base rates), (b) intercepts + one common slope on log p (the spread);
     multinomial, penalised maximum likelihood (Gaussian prior, sd RECAL_SD, on the offsets and on log slope);
   - PERISIM only: INSPIRE's case base plus those n local patients, with the declared site weight γ(n) (its own way
     of adapting, without touching the model);
   - refitted on the local pool (the ceiling): PERISIM A(iii) of the transport stage, LightGBM and the logistic
     regression on an encoder fitted on the pool, the kNN references on the pool.
   Patient-bootstrap CIs (the same resamples for every method, so differences are paired): log-loss, loss against the
   method's own ceiling, PERISIM − comparator, macro and death AUROC.
2. Where do the sites differ? The per-site shape functions of analysis E (INSPIRE's value grid), one figure per
   variable: g_j of every class at INSPIRE, MIMIC-IV and eICU (each divided by its site scale, as in E) above the share
   of each site's stays in every INSPIRE bin (bins with < 10 stays suppressed). Per variable: distribution shift (PSI
   over INSPIRE's bins, % outside INSPIRE's range, missingness) next to the shape deviation and the weight w_j —
   a different population (covariate shift) vs a different relationship.

Usage:  python scripts/posthoc_transport.py [--profile local] [--workdir <work>] [--force]      (run_local.ps1 posthoc)

Outside src/perisim on purpose: the frozen code hash covers all of src/perisim. The analysis only reads the frozen
artefacts (final model, transport fits; hashes checked at the start and the end) and uses the transport stage's split,
labels and k. Per-patient predictions are cached in predictions/<variant>/posthoc/<t>/ (never exported); --force
refits. Outputs (aggregates only): results/posthoc/, artifacts/<variant>/posthoc_transport.json,
reports/posthoc_transport.html.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perisim.baselines import parametric  # noqa: E402
from perisim.evaluation import metrics as M  # noqa: E402
from perisim.evaluation.html import box, fig_img, page, table  # noqa: E402
from perisim.experiment import SEVERE, TARGET_CLASSES, inner_splits, load_cohort_data  # noqa: E402
from perisim.external import _eval_target_for  # noqa: E402
from perisim.metric.encoding import Encoder  # noqa: E402
from perisim.transport.analyses import context  # noqa: E402
from perisim.transport.core import (  # noqa: E402
    case_base, check_hashes, frozen_hashes, gamma_schedule, macro_auroc, point_metrics, predict_gammas,
    source_labels, variable_class,
)
from perisim.transport.run import _csv, shape_homogeneity  # noqa: E402
from perisim.utils import write_json  # noqa: E402

LOG = logging.getLogger("perisim.posthoc")
PHASE1 = "posthoc-1"          # change when the cached predictions change meaning
RECAL_SD = 2.0                # prior sd of the recalibration offsets and of log(slope)
MIN_N = 10                    # §9 small-cell threshold (bin shares)
METHODS = {"perisim": "PERISIM", "lgbm": "LightGBM", "logreg": "Logistic regression", "euclidean": "Euclidean kNN",
           "prevalence": "Prevalence"}
SITES = {"inspire": "INSPIRE", "mimic": "MIMIC-IV", "eicu": "eICU"}
INK, MUTED, GRID = "#1d2433", "#6b7c93", "#e6e9ee"
M_COL = {"perisim": "#2a78d6", "lgbm": "#eb6834", "logreg": "#1baf7a", "euclidean": "#eda100", "prevalence": "#b9bfc8"}
S_COL = {"inspire": "#2a78d6", "mimic": "#eb6834", "eicu": "#1baf7a"}
V_COL = {"physiological": "#2a78d6", "practice": "#eb6834", "coding": "#1baf7a", "patient": "#eda100",
         "other": "#b9bfc8"}


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- recalibration and shift (pure helpers)
def fit_recal(P: np.ndarray, y: np.ndarray, slope: bool, sd: float = RECAL_SD) -> tuple[np.ndarray, float]:
    """Multinomial recalibration p'_c ∝ exp(a_c + b·log p_c), a_0 = 0; b = 1 when `slope` is False (offsets only).
    Penalised maximum likelihood: N(0, sd²) on a and on log b, so a handful of patients cannot run away."""
    L = np.log(np.clip(P, 1e-12, 1.0))
    C, n_a = P.shape[1], P.shape[1] - 1
    rows = np.arange(len(y))

    def nll(th):
        a = np.concatenate([[0.0], th[:n_a]])
        z = (np.exp(th[n_a]) if slope else 1.0) * L + a
        z = z - z.max(1, keepdims=True)
        return float((np.log(np.exp(z).sum(1)) - z[rows, y]).sum() + 0.5 * float(th @ th) / sd ** 2)

    th = minimize(nll, np.zeros(n_a + int(slope)), method="L-BFGS-B").x
    return np.concatenate([[0.0], th[:n_a]]), float(np.exp(th[n_a])) if slope else 1.0


def apply_recal(P: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    z = b * np.log(np.clip(P, 1e-12, 1.0)) + a
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def tuning_subsets(subj_tune: np.ndarray, sizes: list[int], seed: int) -> dict[int, np.ndarray]:
    """The local patients of analysis D: the first n of one seeded permutation of the tuning patients (nested)."""
    pats = np.random.default_rng(seed + 11).permutation(np.unique(subj_tune))
    return {int(n): np.isin(subj_tune, pats[:n]) for n in sizes if int(n) <= pats.size}


def psi(ref_counts: np.ndarray, counts: np.ndarray, eps: float = 1e-4) -> float:
    """Population stability index over the reference bins. Inflated by a different recording resolution (values
    rounded to 0.1 leave some fine INSPIRE bins empty), so it is reported but not used as the shift measure."""
    p = ref_counts / max(ref_counts.sum(), 1)
    q = counts / max(counts.sum(), 1)
    return float(np.sum((q - p) * np.log((q + eps) / (p + eps))))


def percentile_shift(ref_sorted: np.ndarray, x: np.ndarray, grid: int = 1000) -> tuple[float, float]:
    """Distribution shift on the reference's percentile scale: every value → its mid-rank percentile u in the
    reference (as the neutral axis does), for the site and for the reference itself; returns the Wasserstein-1
    distance between the two (the mean displacement in reference percentiles, 0–1; robust to a coarser recording
    resolution and to ties such as GCS = 15) and the signed shift E[u_site] − E[u_ref] (+ = higher values)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0 or ref_sorted.size == 0:
        return float("nan"), float("nan")

    def u(v):
        return (np.searchsorted(ref_sorted, v, "left") + np.searchsorted(ref_sorted, v, "right")) / (2.0 * ref_sorted.size)

    us, ur = u(x), u(ref_sorted)
    t = (np.arange(grid) + 0.5) / grid
    return (float(np.mean(np.abs(np.quantile(us, t) - np.quantile(ur, t)))), float(us.mean() - ur.mean()))


def total_variation(ref_counts: np.ndarray, counts: np.ndarray) -> float:
    """Share of the stays that would have to change category to match the reference (nominal variables)."""
    return float(0.5 * np.abs(ref_counts / max(ref_counts.sum(), 1) - counts / max(counts.sum(), 1)).sum())


def suppressed_shares(counts: np.ndarray, min_n: int = MIN_N) -> np.ndarray:
    """Share of the stays in each bin; bins with fewer than `min_n` stays → NaN (never exported)."""
    s = counts / max(counts.sum(), 1)
    return np.where(counts >= min_n, s, np.nan)


# --------------------------------------------------------------------------- phase 1: predictions (cached)
def _fit_parametric(cfg, X, y, subjects, C, seed, threads) -> tuple[dict, dict]:
    """LightGBM (grid × early stopping + temperature) and L2 logistic regression, nested exactly as in the CV."""
    ms = cfg.model_space["baselines"]
    inner = inner_splits(y, subjects, int(cfg.experiment["cv"]["inner_folds"]), seed)
    lg, i_lg = parametric.fit_lgbm_calibrated(X, y, ms["lgbm"], inner, C, seed, threads, grid=ms.get("lgbm_grid"))
    lr, i_lr = parametric.fit_logreg(X, y, ms["logreg_C_grid"], inner, C)
    return {"lgbm": lg, "logreg": lr}, {"lgbm": i_lg, "logreg": i_lr}


def _cache_key(ws, t: str, c: str) -> str:
    h = hashlib.sha256(PHASE1.encode())
    for f in (ws.predictions("external", t) / "final_model.pkl", ws.predictions("transport", t) / f"{c}_local_model.pkl"):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def inspire_models(cfg, ws, t: str, args: dict) -> dict:
    """The INSPIRE side, shared by every external cohort: frozen final model, labels and the parametric comparators
    trained on the same stays as the final model."""
    fm = _load(ws.predictions("external", t) / "final_model.pkl")
    D = load_cohort_data(ws.common(args["dev"]), ws.derived(args["dev"]), args["names"])
    y = source_labels(t, t, D, cfg)["own"]
    ok = np.where(y >= 0)[0]
    Xm, _ = fm["enc"].transform_model(D.ft.take(ok))
    LOG.info("posthoc: LightGBM and logistic regression on %s (n=%d)", args["dev"], ok.size)
    models, info = _fit_parametric(cfg, Xm, y[ok], D.ft.subjects[ok], TARGET_CLASSES[t], int(cfg.seed),
                                   args["threads"])
    return {"fm": fm, "D": D, "y": y, "models": models, "info": info}


def cohort_predictions(cfg, ws, t: str, c: str, args: dict, ins: dict) -> dict:
    """Test-set probabilities of every (method, situation) and the tuning-set probabilities of the transported models."""
    ctx = context(cfg, ws, t, c, args)
    X, C, y, split = ctx["X"], ctx["C"], ctx["lab"]["own"], ctx["split"]
    tc, kcfg = cfg.experiment["transport"], cfg.experiment["knn"]
    kp, kg = int(kcfg["k_primary"]), [int(k) for k in kcfg["k_grid"]]
    fm, D, yD = ins["fm"], ins["D"], ins["y"]
    enc, alpha = fm["enc"], dict(fm["alpha"])
    wd, nd = enc.weights_dims(fm["w"]), enc.neutral_weights_dims(np.ones(enc.n_features))
    q, tu, pool = (v[y[v] >= 0] for v in (split["test"], split["tuning"], split["pool"]))
    sX, sD = X.ft.subjects.astype(str), D.ft.subjects.astype(str)
    sub = tuning_subsets(sX[tu], [int(s) for s in tc["tuning_sizes"]], ctx["seed"])
    parts = {"test": q, "tune": tu}
    E_X, E_D = enc.transform(X.ft), enc.transform(D.ft)
    N_X, N_D = enc.transform_neutral(X.ft), enc.transform_neutral(D.ft)
    Xm_X, _ = enc.transform_model(X.ft)
    inspire = (E_D, yD, sD, False)
    P: dict[str, np.ndarray] = {}

    def knn(kn, E, idx, g=1.0):
        return predict_gammas(kn, E[idx], sX[idx], kp, {"g": g})[0]["g"]

    # ---- trained on INSPIRE (test and tuning rows: the tuning rows feed the recalibration)
    kA = case_base([inspire], wd, C, kcfg, kg, alpha, with_ood=False)
    kE = case_base([(N_D, yD, sD, False)], nd, C, kcfg, [kp])
    prior = np.bincount(yD[yD >= 0], minlength=C) / max(int((yD >= 0).sum()), 1)
    for part, idx in parts.items():
        P[f"perisim__inspire__{part}"] = knn(kA, E_X, idx)
        P[f"euclidean__inspire__{part}"] = knn(kE, N_X, idx)
        P[f"prevalence__inspire__{part}"] = np.tile(prior, (idx.size, 1))
        for m, f in ins["models"].items():
            P[f"{m}__inspire__{part}"] = f(Xm_X[idx])
    # ---- PERISIM's own adaptation: INSPIRE ∪ the n local tuning patients, declared site weight γ(n)
    for n, mask in sub.items():
        loc = (E_X[tu[mask]], y[tu[mask]], sX[tu[mask]], True)
        kn = case_base([inspire, loc], wd, C, kcfg, kg, alpha, with_ood=False)
        P[f"perisim__casebase{n}__test"] = knn(kn, E_X, q, gamma_schedule(n, float(tc["gamma_schedule_n0"])))
    # ---- refitted on the local pool (ceilings)
    LOG.info("posthoc: local refits on the pool of %s (%d stays)", c, pool.size)
    P["perisim__frozenlocal__test"] = knn(case_base([(E_X[pool], y[pool], sX[pool], True)], wd, C, kcfg, kg, alpha,
                                                    with_ood=False), E_X, q)
    fpL = _load(ws.predictions("transport", t) / f"{c}_local_model.pkl")
    kL = fpL["knn"]
    kL.is_local = np.ones(kL.Ew.shape[0], dtype=bool)
    P["perisim__local__test"] = predict_gammas(kL, fpL["enc"].transform(X.ft.take(q)), sX[q], kp, {"g": 1.0})[0]["g"]
    P["euclidean__local__test"] = knn(case_base([(N_X[pool], y[pool], sX[pool], True)], nd, C, kcfg, [kp]), N_X, q)
    P["prevalence__local__test"] = np.tile(np.bincount(y[pool], minlength=C) / pool.size, (q.size, 1))
    encL = Encoder.from_config(cfg).fit(X.ft.take(pool), y[pool], C)
    models, info = _fit_parametric(cfg, encL.transform_model(X.ft.take(pool))[0], y[pool], sX[pool], C, ctx["seed"],
                                   args["threads"])
    XLq = encL.transform_model(X.ft.take(q))[0]
    for m, f in models.items():
        P[f"{m}__local__test"] = f(XLq)
    out = {**P, "y_test": y[q], "s_test": sX[q], "y_tune": y[tu], "s_tune": sX[tu],
           **{f"tune{n}": m.astype(np.int8) for n, m in sub.items()}}
    return {"arrays": out, "info": {"inspire": ins["info"], "local": info, "et": ctx["et"], "C": C,
                                    "pool_patients": int(np.unique(sX[pool]).size), "seed": ctx["seed"]}}


# --------------------------------------------------------------------------- phase 2: evaluation
def _pv(bs: np.ndarray, point: float) -> dict:
    return {"mean": float(point), "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))}


def evaluate_cohort(cfg, t: str, c: str, d: dict, info: dict) -> dict:
    et, C = info["et"], int(info["C"])
    nb, seed = int(cfg.experiment["evaluation"]["n_bootstrap"]), int(cfg.seed)
    yq, sq, yt = d["y_test"], d["s_test"], d["y_tune"]
    sizes = sorted(int(k[4:]) for k in d if k.startswith("tune") and k[4:].isdigit())
    preds, recal = {}, []
    for m in METHODS:
        P0, Pt = d[f"{m}__inspire__test"], d[f"{m}__inspire__tune"]
        preds[(m, "inspire")] = P0
        for n in sizes:
            mask = d[f"tune{n}"].astype(bool)
            for lvl, slope in (("int", False), ("full", True)):
                a, b = fit_recal(Pt[mask], yt[mask], slope)
                preds[(m, f"{lvl}{n}")] = apply_recal(P0, a, b)
                recal.append({"method": m, "level": lvl, "n_local": n, "slope": b,
                              **{f"offset_{k}": float(a[k]) for k in range(1, C)}})
        preds[(m, "local")] = d[f"{m}__local__test"]
    for n in sizes:
        preds[("perisim", f"casebase{n}")] = d[f"perisim__casebase{n}__test"]
    preds[("perisim", "frozenlocal")] = d["perisim__frozenlocal__test"]
    key = {k: f"{k[0]}|{k[1]}" for k in preds}
    B = M.bootstrap_rowwise({key[k]: M.row_logloss(P, yq) for k, P in preds.items()}, sq, nb, seed)

    def diff(a, b):
        return _pv(B[key[a]]["_boot"] - B[key[b]]["_boot"], B[key[a]]["mean"] - B[key[b]]["mean"])

    rows = []
    for (m, s), P in preds.items():
        n_loc = 0 if s == "inspire" else info["pool_patients"] if s in ("local", "frozenlocal") else int(
            "".join(ch for ch in s if ch.isdigit()))
        dl = diff((m, s), (m, "local")) if s != "local" else None
        rows.append({"method": m, "situation": _situation(s), "n_local": n_loc, **point_metrics(P, yq, et),
                     "logloss_ci_low": B[key[(m, s)]]["ci_low"], "logloss_ci_high": B[key[(m, s)]]["ci_high"],
                     "d_vs_local": dl["mean"] if dl else 0.0, "d_vs_local_low": dl["ci_low"] if dl else np.nan,
                     "d_vs_local_high": dl["ci_high"] if dl else np.nan})
    contrasts = []
    for m in ("lgbm", "logreg", "euclidean"):
        pairs = [("inspire", "inspire"), ("local", "local")] + [(f"{lv}{n}", f"{lv}{n}") for n in sizes
                                                                for lv in ("int", "full")]
        pairs += [(f"casebase{n}", f"full{n}") for n in sizes]
        for sp, sm in pairs:
            r = diff(("perisim", sp), (m, sm))
            contrasts.append({"comparator": m, "perisim": _situation(sp), "comparator_situation": _situation(sm),
                              "d_logloss": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                              "p": r["p_two_sided"]})
    return {"cohort": c, "n_test": int(len(yq)), "test_patients": int(np.unique(sq).size), "sizes": sizes,
            "pool_patients": info["pool_patients"], "rows": rows, "contrasts": contrasts, "recalibration": recal,
            "auroc": _auroc_boot({k: preds[k] for k in preds if k[0] != "prevalence" and k[1] in ("inspire", "local")},
                                 yq, sq, SEVERE[et], nb, seed),
            "hyperparameters": info["inspire"] | {f"local_{k}": v for k, v in info["local"].items()}}


def _situation(s: str) -> str:
    for p, lab in (("int", "recal. intercept"), ("full", "recal. intercept+slope"), ("casebase", "+ local cases")):
        if s.startswith(p) and s[len(p):].isdigit():
            return f"{lab} (n={s[len(p):]})"
    return {"inspire": "INSPIRE as is", "local": "local refit", "frozenlocal": "INSPIRE metric + local case base"}[s]


def _auroc_boot(preds: dict, y: np.ndarray, subj: np.ndarray, sev: int, n_boot: int, seed: int) -> list[dict]:
    """Macro and death AUROC with one set of patient resamples for every prediction (paired differences vs PERISIM)."""
    inv = np.unique(subj, return_inverse=True)[1]
    S = int(inv.max()) + 1
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(S))
    ends = np.append(starts[1:], len(inv))
    rng = np.random.default_rng(seed)

    def both(P, yy):
        return macro_auroc(P, yy), M.auroc_binary(yy == sev, P[:, sev])

    point = {k: both(P, y) for k, P in preds.items()}
    draws = {k: [] for k in preds}
    for _ in range(n_boot):
        rows = np.concatenate([order[starts[s]:ends[s]] for s in rng.integers(0, S, S)])
        for k, P in preds.items():
            draws[k].append(both(P[rows], y[rows]))
    out = []
    for (m, s), v in draws.items():
        A = np.array(v, dtype=float)
        r = {"method": m, "situation": _situation(s)}
        for j, name in enumerate(("auroc_macro", "auroc_death")):
            ok = np.isfinite(A[:, j])
            r.update({name: point[(m, s)][j], f"{name}_low": float(np.percentile(A[ok, j], 2.5)),
                      f"{name}_high": float(np.percentile(A[ok, j], 97.5))})
            if m != "perisim" and ("perisim", s) in draws:
                R = np.array(draws[("perisim", s)], dtype=float)
                dd = R[:, j] - A[:, j]
                dd = dd[np.isfinite(dd)]
                r.update({f"perisim_minus_{name}": point[("perisim", s)][j] - point[(m, s)][j],
                          f"perisim_minus_{name}_low": float(np.percentile(dd, 2.5)),
                          f"perisim_minus_{name}_high": float(np.percentile(dd, 97.5))})
        out.append(r)
    return out


# --------------------------------------------------------------------------- phase 3: shapes and shift by site
def shapes_by_site(cfg, ws, t: str, args: dict) -> dict:
    fm = _load(ws.predictions("external", t) / "final_model.pkl")
    enc, dev = fm["enc"], args["dev"]
    sites = {}
    for s in [dev] + args["externals"]:
        f = ws.predictions("transport", t) / f"shapes_{s}.pkl"
        if f.exists():
            sites[s] = _load(f)
    E = shape_homogeneity(sites, dev, float(cfg.experiment["transport"]["homogeneity_alpha"]))
    use, scale = E.get("sites") or [], E.get("scale") or {}
    if dev not in use:
        return {"variables": [], "curves": [], "shares": [], "scale": scale, "sites": use}
    data = {s: load_cohort_data(ws.common(s), ws.derived(s), args["names"]).ft for s in use}
    w = np.asarray(fm["w"], float) / max(float(np.sum(fm["w"])), 1e-12)
    wj = dict(zip(enc.feature_names, w))
    dev_rows = {}
    for r in E["rows"]:
        for s in use:
            if s != dev:
                dev_rows.setdefault((r["variable"], s), []).append(r.get(f"rms_dev_{s}", np.nan))
    block = {b.name: b for b in enc.blocks}
    variables, curves, shares = [], [], []
    for name in sorted(sites[dev]["shapes"]):
        b = block.get(name)
        if b is None or not all(name in sites[s]["shapes"] for s in use):
            continue
        counts, extra = {}, {}
        for s in use:
            counts[s], extra[s] = _bin_counts(enc, b, data[s])
        labels = _bin_labels(b)
        v = {"variable": name, "variable_class": variable_class(name), "kind": b.kind, "weight": float(wj.get(name, 0.0)),
             "n_values": len(labels)}
        for s in use:
            v[f"missing_pct_{s}"] = extra[s]["missing_pct"]
            if s != dev:
                if b.kind in ("continuous", "ordinal"):
                    v[f"shift_{s}"], v[f"shift_direction_{s}"] = percentile_shift(b.params["sorted"],
                                                                                  data[s].num[b.source])
                else:
                    v[f"shift_{s}"] = total_variation(np.r_[counts[dev], extra[dev]["n_unseen"]],
                                                      np.r_[counts[s], extra[s]["n_unseen"]])
                    v[f"shift_direction_{s}"] = np.nan
                v[f"psi_{s}"] = psi(counts[dev], counts[s])
                v[f"outside_range_pct_{s}"] = extra[s]["outside_pct"]
                v[f"shape_rms_dev_{s}"] = float(np.nanmean(dev_rows.get((name, s), [np.nan])))
        variables.append(v)
        for s in use:
            sh = sites[s]["shapes"][name]
            for i, lab in enumerate(labels):
                shares.append({"variable": name, "site": s, "bin": i, "value": lab,
                               "share": float(suppressed_shares(counts[s])[i])})
                for k in range(sh["mean"].shape[1]):
                    curves.append({"variable": name, "site": s, "bin": i, "value": lab, "class": k,
                                   "g_scaled": float(sh["mean"][i, k] / (scale.get(s) or 1.0)),
                                   "se_scaled": float(np.sqrt(sh["se2"][i, k]) / abs(scale.get(s) or 1.0))})
    return {"variables": variables, "curves": curves, "shares": shares, "scale": scale, "sites": use}


def _bin_counts(enc, b, ft) -> tuple[np.ndarray, dict]:
    """Stays of one site in every value of block b (INSPIRE's bins / categories); out-of-range values go to the end
    bins, as in the metric (g_j is constant beyond INSPIRE's range); missing values are left out and counted apart."""
    if b.kind in ("continuous", "ordinal"):
        x = ft.num[b.source]
        fin = np.isfinite(x)
        e = b.params["edges"]
        xv = x[fin]
        idx = np.clip(np.searchsorted(e, xv, side="right") - 1, 0, e.size - 2)
        out = float(((xv < e[0]) | (xv > e[-1])).mean() * 100) if xv.size else np.nan
        return np.bincount(idx, minlength=e.size - 1).astype(float), {"missing_pct": float((~fin).mean() * 100),
                                                                       "outside_pct": out}
    idx = enc.value_index(b, ft)
    n_val = enc._n_values(b)
    raw = np.asarray(ft.cat[b.source], dtype=object)
    miss = np.array([r is None or (isinstance(r, float) and r != r) or r == "" for r in raw])
    return np.bincount(idx[idx >= 0], minlength=n_val).astype(float), {
        "missing_pct": float(miss.mean() * 100), "outside_pct": float((idx < 0).mean() * 100),
        "n_unseen": float((idx < 0).sum())}


def _bin_labels(b) -> list[str]:
    """Bin labels from INSPIRE's INNER edges only (the outer edges are single patients' extreme values)."""
    if b.kind in ("continuous", "ordinal"):
        e = b.params["edges"]
        nb = e.size - 1
        if nb == 1:
            return ["all"]
        return [f"< {e[1]:.4g}"] + [f"[{e[i]:.4g}, {e[i + 1]:.4g})" for i in range(1, nb - 1)] + [f"≥ {e[nb - 1]:.4g}"]
    if b.kind == "nominal":
        return [str(v) for v in b.params["values"]]
    return [str(v) for v in b.params["codes"]] + ["other", "missing"]


# --------------------------------------------------------------------------- figures
def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.set_xlabel(xlabel, color=INK, fontsize=8)
    ax.set_ylabel(ylabel, color=INK, fontsize=8)


def comparators_figure(res: dict):
    import matplotlib.pyplot as plt

    rows = {(r["method"], r["situation"]): r["logloss"] for r in res["rows"]}
    sizes = res["sizes"]
    xs = list(range(len(sizes) + 2))
    xt = ["INSPIRE\nas is"] + [str(n) for n in sizes] + [f"local refit\n({res['pool_patients']:,} pat.)"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, lvl, title in ((axes[0], "recal. intercept", "Recalibrated: intercepts only (base rates)"),
                           (axes[1], "recal. intercept+slope", "Recalibrated: intercepts + slope (spread)")):
        for m, lab in METHODS.items():
            ys = [rows[(m, "INSPIRE as is")]] + [rows[(m, f"{lvl} (n={n})")] for n in sizes] + [rows[(m, "local refit")]]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=M_COL[m], label=lab,
                    ls=(0, (3, 2)) if m == "prevalence" else "-")
        ys = [rows[("perisim", "INSPIRE as is")]] + [rows[("perisim", f"+ local cases (n={n})")] for n in sizes] + \
             [rows[("perisim", "INSPIRE metric + local case base")]]
        ax.plot(xs, ys, marker="s", ms=4, lw=1.4, color=M_COL["perisim"], ls=(0, (1.5, 1.5)),
                label="PERISIM, INSPIRE metric + local cases (γ(n))")
        ax.set_xticks(xs, xt)
        _style(ax, title, "local patients used", "log-loss (test)")
    axes[0].legend(fontsize=7, frameon=False, ncol=3, labelcolor=INK, loc="upper center", bbox_to_anchor=(1.05, -0.22))
    fig.suptitle(f"{SITES.get(res['cohort'], res['cohort'])}: transported vs locally refitted "
                 f"(test n={res['n_test']:,})", color=INK, fontsize=10, x=0.01, ha="left")
    return fig


def shift_map_figure(sh: dict, dev: str):
    import matplotlib.pyplot as plt

    ext = [s for s in sh["sites"] if s != dev]
    fig, axes = plt.subplots(1, len(ext), figsize=(5.6 * len(ext), 4.6), squeeze=False)
    V = sh["variables"]
    for ax, s in zip(axes[0], ext):
        x = np.array([v.get(f"shift_{s}", np.nan) for v in V])
        y = np.array([v.get(f"shape_rms_dev_{s}", np.nan) for v in V])
        w = np.array([v["weight"] for v in V])
        ok = np.isfinite(x) & np.isfinite(y)
        for cls in V_COL:
            sel = ok & np.array([v["variable_class"] == cls for v in V])
            if sel.any():
                ax.scatter(x[sel], y[sel], s=25 + 900 * w[sel], color=V_COL[cls], alpha=0.75, label=cls, lw=0)
        rank = np.argsort(-(np.argsort(np.argsort(np.nan_to_num(x))) + np.argsort(np.argsort(np.nan_to_num(y)))))
        for i in rank[:10]:
            if ok[i]:
                ax.annotate(V[i]["variable"], (x[i], y[i]), fontsize=6.5, color=INK, xytext=(3, 3),
                            textcoords="offset points")
        rho = spearmanr(x[ok], y[ok]).correlation if ok.sum() > 2 else np.nan
        _style(ax, f"{SITES.get(s, s)} vs INSPIRE (Spearman ρ = {rho:.2f})",
               "different population: distribution shift", "different relationship: shape deviation (RMS, scaled)")
    leg = axes[0][0].legend(fontsize=7, frameon=False, ncol=5, labelcolor=INK, loc="upper center",
                            bbox_to_anchor=(0.5, -0.14))
    for h in leg.legend_handles:
        h.set_sizes([30])
    fig.text(0.01, -0.08, "Distribution shift: continuous = mean displacement in INSPIRE percentiles (Wasserstein-1); "
             "nominal = share of stays in other categories. Point size: weight w_j.", color=MUTED, fontsize=7)
    return fig


def shape_figure(name: str, sh: dict, dev: str, enc_block, class_names: list[str]):
    import matplotlib.pyplot as plt

    V = next(v for v in sh["variables"] if v["variable"] == name)
    cur = [r for r in sh["curves"] if r["variable"] == name]
    shr = [r for r in sh["shares"] if r["variable"] == name]
    nv, C = V["n_values"], 1 + max(r["class"] for r in cur)
    cont = V["kind"] in ("continuous", "ordinal")
    sites = sh["sites"]
    fig = plt.figure(figsize=(12, 5.8))
    gs = fig.add_gridspec(2, C, height_ratios=[2.0, 1.1], hspace=0.55, wspace=0.1)
    axes = [fig.add_subplot(gs[0, k]) for k in range(C)]
    for ax in axes[1:]:
        ax.sharey(axes[0])
    axd = fig.add_subplot(gs[1, :])
    edges = np.arange(nv + 1)
    for i, s in enumerate(sites):
        sc = sh["scale"].get(s, 1.0)
        lab = SITES.get(s, s) + ("" if s == dev else f" (÷ {sc:.2f})")
        for k, ax in enumerate(axes):
            rr = sorted((r for r in cur if r["site"] == s and r["class"] == k), key=lambda r: r["bin"])
            g = np.array([r["g_scaled"] for r in rr])
            se = np.array([r["se_scaled"] for r in rr])
            if cont:
                ax.stairs(g, edges, color=S_COL[s], lw=1.6, label=lab)
                ax.fill_between(edges, np.r_[g - 1.96 * se, g[-1] - 1.96 * se[-1]],
                                np.r_[g + 1.96 * se, g[-1] + 1.96 * se[-1]], step="post", color=S_COL[s], alpha=0.13,
                                lw=0)
            else:
                ax.errorbar(np.arange(nv) + (i - 1) * 0.2, g, yerr=1.96 * se, fmt="o", ms=3, color=S_COL[s],
                            elinewidth=1, label=lab)
        share = np.array([r["share"] for r in sorted((r for r in shr if r["site"] == s), key=lambda r: r["bin"])])
        extra = f" · missing {V.get(f'missing_pct_{s}', np.nan):.1f} %"
        if s != dev:
            extra = f" · outside INSPIRE's range {V.get(f'outside_range_pct_{s}', np.nan):.1f} %" + extra
        width = 0.8 / len(sites)
        axd.bar(np.arange(nv) + (0.5 if cont else 0.0) + (i - (len(sites) - 1) / 2) * width, share * 100, width,
                color=S_COL[s], label=SITES.get(s, s) + extra)
    labels = _bin_labels(enc_block)
    if cont:
        step = max(1, int(np.ceil((nv - 1) / 9)))
        pos = list(range(1, nv, step))
        inner = enc_block.params["edges"]
        tick = ([pos, [f"{inner[p]:.3g}" for p in pos]] if nv > 1 else [[], []])
        for ax in axes + [axd]:
            ax.set_xticks(*tick)
            ax.set_xlim(0, nv)
    else:
        for ax in axes + [axd]:
            ax.set_xticks(range(nv), labels, rotation=40, ha="right", fontsize=6)
    g_all = np.array([r["g_scaled"] for r in cur])
    g_all = g_all[np.isfinite(g_all)]
    if g_all.size:          # y range from the curves: the bands of a site with a small scale must not flatten them
        lo, hi = float(g_all.min()), float(g_all.max())
        pad = 0.25 * max(hi - lo, 0.1)
        axes[0].set_ylim(lo - pad, hi + pad)
    for k, ax in enumerate(axes):
        ax.axhline(0, color=MUTED, lw=0.6)
        _style(ax, class_names[k] if k < len(class_names) else f"class {k}", "",
               "g (centred log-odds, ÷ site scale)" if k == 0 else "")
        if k:
            ax.tick_params(labelleft=False)
    _style(axd, "Where each site's stays fall on INSPIRE's grid (bins with < 10 stays not shown)",
           "value (INSPIRE quantile bins)" if cont else "", "% of the site's stays")
    axes[0].legend(fontsize=7, frameon=False, ncol=3, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.12))
    axd.legend(fontsize=7, frameon=False, ncol=1, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ext = [s for s in sites if s != dev]
    stats = " · ".join(f"{SITES.get(s, s)}: distribution shift {V.get(f'shift_{s}', np.nan):.2f}, shape deviation "
                       f"{V.get(f'shape_rms_dev_{s}', np.nan):.2f}" for s in ext)
    fig.suptitle(f"{name}  ·  {V['variable_class']}  ·  weight {V['weight']:.1%}\n{stats}", color=INK, fontsize=10,
                 x=0.01, ha="left", y=1.02)
    return fig


def _save(fig, path: Path) -> str:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


# --------------------------------------------------------------------------- report
def _ci(v, lo, hi, nd: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]" if np.isfinite(lo) else f"{v:.{nd}f}"


def _delta(v, lo, hi) -> str:
    return "—" if v is None or not np.isfinite(lo) else f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _cal(citl, slope) -> str:
    """CITL / slope; 'n.e.' when the calibration line does not converge (near-constant predictions)."""
    def one(v, f):
        return f.format(v) if v is not None and np.isfinite(v) and abs(v) < 20 else "n.e."
    return f"{one(citl, '{:+.2f}')} / {one(slope, '{:.2f}')}"


def render(out: dict, full: dict, dev: str) -> str:
    sec = [box("<b>Exploratory, post hoc.</b> Added on 2026-10-28 after the external validation and the transport "
               "analyses had been seen; not part of the pre-registration (SPECS §6.2b, amendment of 2026-10-24) and "
               "no decision rule. The frozen artefacts were only read (SHA-256 identical at the start and the end); the "
               "split, labels (cross-fitted tertiles) and k are those of the transport stage. LightGBM and the logistic "
               "regression are nested as in the CV. Recalibration: multinomial, p'<sub>c</sub> ∝ exp(a<sub>c</sub> + "
               "b·log p<sub>c</sub>), on the nested local tuning subsets of analysis D, Gaussian prior sd "
               f"{RECAL_SD:g} on a and log b. 95 % CIs: patient bootstrap, the same resamples for every method "
               "(paired). Shapes: the per-site consensus of analysis E on INSPIRE's grid, each site divided by its "
               "scale.")]
    for t, r in out["targets"].items():
        sec.append(f"<h2>{t.upper()}</h2>")
        for c, rc in r["cohorts"].items():
            n_max = max(rc["sizes"]) if rc["sizes"] else None
            R = {(x["method"], x["situation"]): x for x in rc["rows"]}
            sec.append(f"<h3>{SITES.get(c, c)} — test set {rc['n_test']:,} stays ({rc['test_patients']:,} patients); "
                       f"local pool {rc['pool_patients']:,} patients</h3>")
            hdr = ["Method", "local refit (ceiling)", "INSPIRE as is", "loss vs ceiling"]
            sits = ["INSPIRE as is"]
            if n_max:
                hdr += [f"+ recal. intercept (n={n_max})", "loss", f"+ recal. intercept+slope (n={n_max})", "loss"]
                sits += [f"recal. intercept (n={n_max})", f"recal. intercept+slope (n={n_max})"]
            rows = []
            for m, lab in METHODS.items():
                row = [lab, _ci(R[(m, "local refit")]["logloss"], R[(m, "local refit")]["logloss_ci_low"],
                                R[(m, "local refit")]["logloss_ci_high"])]
                for s in sits:
                    x = R[(m, s)]
                    row += [_ci(x["logloss"], x["logloss_ci_low"], x["logloss_ci_high"]),
                            _delta(x["d_vs_local"], x["d_vs_local_low"], x["d_vs_local_high"])]
                rows.append(row)
            if n_max:
                for s in (f"+ local cases (n={n_max})", "INSPIRE metric + local case base"):
                    x = R[("perisim", s)]
                    rows.append([f"PERISIM, {s}", "", _ci(x["logloss"], x["logloss_ci_low"], x["logloss_ci_high"]),
                                 _delta(x["d_vs_local"], x["d_vs_local_low"], x["d_vs_local_high"])] +
                                [""] * (len(hdr) - 4))
            sec.append("<h4>Log-loss: how much each method loses (loss = situation − its own local refit)</h4>" +
                       table(hdr, rows))
            sec.append("<h4>PERISIM − comparator (Δ log-loss on the same stays; negative = PERISIM better)</h4>" + table(
                ["PERISIM", "Comparator", "LightGBM", "Logistic regression", "Euclidean kNN"],
                [[a, b] + [next((_delta(x["d_logloss"], x["ci_low"], x["ci_high"]) for x in rc["contrasts"]
                                 if x["comparator"] == m and x["perisim"] == a and x["comparator_situation"] == b), "—")
                           for m in ("lgbm", "logreg", "euclidean")]
                 for a, b in dict.fromkeys((x["perisim"], x["comparator_situation"]) for x in rc["contrasts"])]))
            A = {(x["method"], x["situation"]): x for x in rc["auroc"]}
            sec.append("<h4>Discrimination (recalibration barely moves it)</h4>" + table(
                ["Method", "AUROC macro, INSPIRE as is", "local refit", "AUROC death, INSPIRE as is", "local refit",
                 "PERISIM − method, macro (INSPIRE / local)", "PERISIM − method, death (INSPIRE / local)"],
                [[lab] + [_ci(A[(m, s)][f"{k}"], A[(m, s)][f"{k}_low"], A[(m, s)][f"{k}_high"])
                          for k in ("auroc_macro", "auroc_death") for s in ("INSPIRE as is", "local refit")] +
                 ["" if m == "perisim" else " / ".join(_delta(A[(m, s)][f"perisim_minus_{k}"],
                                                              A[(m, s)][f"perisim_minus_{k}_low"],
                                                              A[(m, s)][f"perisim_minus_{k}_high"])
                                                       for s in ("INSPIRE as is", "local refit"))
                  for k in ("auroc_macro", "auroc_death")]
                 for m, lab in METHODS.items() if (m, "INSPIRE as is") in A]))
            cal_s = ["INSPIRE as is"] + ([f"recal. intercept (n={n_max})", f"recal. intercept+slope (n={n_max})"]
                                         if n_max else []) + ["local refit"]
            thr = sorted({int(k[7:]) for k in rc["rows"][0] if k.startswith("citl_ge")})
            sec.append("<h4>Calibration of the cumulative thresholds P(Y ≥ c): CITL / slope</h4>" + table(
                ["Method", "Situation"] + [f"Y ≥ {c}: CITL / slope" for c in thr],
                [[METHODS[m], s] + [_cal(R[(m, s)][f"citl_ge{c}"], R[(m, s)][f"slope_ge{c}"]) for c in thr]
                 for m in METHODS if m != "prevalence" for s in cal_s]))
            sec.append(fig_img(comparators_figure(rc)))
        sh = full[t]
        if sh["variables"]:
            ext = [s for s in sh["sites"] if s != dev]
            sec.append("<h3>Where the sites differ: distribution vs relationship</h3>")
            sec.append(box("Two different things make a variable behave differently at another site. A <b>different "
                           "population</b> (covariate shift): the site's stays fall in other parts of INSPIRE's grid — "
                           "distribution shift (continuous: mean displacement in INSPIRE percentiles, Wasserstein-1 "
                           "between the site's and INSPIRE's own percentiles, with its sign: + = higher values; "
                           "nominal: share of stays in other categories; the PSI over "
                           "INSPIRE's bins is in the CSV, but a coarser recording resolution inflates it), stays beyond "
                           "INSPIRE's range (where every g<sub>j</sub> is flat) and missingness. A <b>different "
                           "relationship</b>: the shape g<sub>j</sub> re-estimated at the site (analysis E, divided by "
                           "the site scale) departs from INSPIRE's. The weight w<sub>j</sub> says how much the variable "
                           "counts in the metric."))
            sec.append(fig_img(shift_map_figure(sh, dev)))
            V = sorted(sh["variables"], key=lambda v: -max(v.get(f"shape_rms_dev_{s}", 0) for s in ext))
            sec.append(table(["Variable", "class", "weight"] +
                             [f"{SITES.get(s, s)}: {k}" for s in ext for k in ("shift", "direction", "% outside range",
                                                                                 "% missing", "shape dev.")],
                             [[v["variable"], v["variable_class"], f"{v['weight']:.1%}"] +
                              [v.get(f"{k}_{s}") for s in ext for k in ("shift", "shift_direction", "outside_range_pct",
                                                                        "missing_pct", "shape_rms_dev")] for v in V],
                             nd=2))
            sec.append("<h4>Shape functions by site (one figure per variable)</h4>" + "".join(
                f"<p><img alt='{v['variable']}' src='../results/posthoc/shapes/{t}_shape_{v['variable']}.png'/></p>"
                for v in V))
    return page("PERISIM post-hoc transport comparison", sec, "Exploratory · not pre-registered · aggregates only")


# --------------------------------------------------------------------------- driver
def run(pipe, force: bool = False, threads: int | None = None) -> int:
    cfg, ws = pipe.cfg, pipe.ws
    targets = list(cfg.targets_to_model)
    dev = cfg.experiment["cohorts"]["development"]
    externals = [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()]
    for t in targets:
        need = [ws.predictions("external", t) / "final_model.pkl"] + \
               [ws.predictions("transport", t) / f"{c}_local_model.pkl" for c in externals]
        miss = [str(p) for p in need if not p.exists()]
        if miss:
            raise SystemExit(f"posthoc: run the external and transport stages first (missing: {miss})")
    before = frozen_hashes(ws, targets)
    args = {"names": pipe.feature_names(), "dev": dev, "externals": externals,
            "threads": int(threads or pipe.res.get("cpus") or pipe.threads())}
    out: dict = {"note": "Exploratory, post hoc (2026-10-28): not pre-registered.", "targets": {}}
    full = {}                                  # shape curves and bin shares: CSV and report only
    rd = ws.root / "results" / "posthoc"
    rd.mkdir(parents=True, exist_ok=True)
    for t in targets:
        ins = None
        res_t: dict = {"cohorts": {}}
        for c in externals:
            if _eval_target_for(t, c, cfg) != t:
                LOG.info("posthoc: %s evaluates %s, not %s → skipped", c, _eval_target_for(t, c, cfg), t)
                continue
            path = ws.predictions("posthoc", t) / f"{c}_preds.npz"
            key = _cache_key(ws, t, c)
            if force or not path.exists() or str(np.load(path)["key"]) != key:
                ins = ins or inspire_models(cfg, ws, t, args)
                pr = cohort_predictions(cfg, ws, t, c, args, ins)
                np.savez_compressed(path, key=np.array(key), **pr["arrays"])
                with open(path.with_suffix(".info.pkl"), "wb") as fh:
                    pickle.dump(pr["info"], fh)
            d = dict(np.load(path, allow_pickle=False))
            LOG.info("posthoc: evaluating %s/%s (bootstrap)", t, c)
            rc = evaluate_cohort(cfg, t, c, d, _load(path.with_suffix(".info.pkl")))
            res_t["cohorts"][c] = rc
            _save(comparators_figure(rc), rd / f"{t}_fig_comparators_{c}.png")
        sh = full[t] = shapes_by_site(cfg, ws, t, args)
        res_t["shapes"] = {k: v for k, v in sh.items() if k not in ("curves", "shares")}
        out["targets"][t] = res_t
        _write_tables(t, res_t, sh, rd)
        if sh["variables"]:
            _save(shift_map_figure(sh, dev), rd / f"{t}_fig_shift_map.png")
            blocks = {b.name: b for b in _load(ws.predictions("external", t) / "final_model.pkl")["enc"].blocks}
            names = (cfg.targets.get(t) or {}).get("labels") or []
            for v in sh["variables"]:
                _save(shape_figure(v["variable"], sh, dev, blocks[v["variable"]], names),
                      rd / "shapes" / f"{t}_shape_{v['variable']}.png")
    after = frozen_hashes(ws, targets)
    check_hashes(before, after)
    out["frozen_hashes"] = {"start": before, "end": after, "unchanged": True}
    (ws.reports / "posthoc_transport.html").write_text(render(out, full, dev), encoding="utf-8")
    write_json(ws.artifacts() / "posthoc_transport.json", out)
    LOG.info("posthoc: results in %s; report %s", rd, ws.reports / "posthoc_transport.html")
    return 0


def _write_tables(t: str, res_t: dict, sh: dict, rd: Path) -> None:
    for name, key in (("comparators", "rows"), ("contrasts", "contrasts"), ("auroc", "auroc"),
                      ("recalibration", "recalibration")):
        rows = [{"cohort": c, **x} for c, rc in res_t["cohorts"].items() for x in rc[key]]
        if rows:
            _csv(rows, rd / f"{t}_{name}.csv")
    for name in ("variables", "curves", "shares"):
        if sh.get(name):
            _csv(sh[name], rd / f"{t}_shift_{name}.csv")


def main(argv=None) -> int:
    import matplotlib

    from perisim.config import load_config
    from perisim.pipeline import Pipeline
    from perisim.utils import now_iso, set_threads, setup_logging

    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--threads", type=int, default=None, help="threads for LightGBM (default: all usable CPUs)")
    ap.add_argument("--force", action="store_true", help="refit the models instead of reusing the cached predictions")
    a = ap.parse_args(argv)
    over = {"experiment": {"paths": {"workdir": str(Path(a.workdir).resolve())}}} if a.workdir else {}
    pipe = Pipeline(load_config(None, a.profile, over))
    setup_logging(pipe.ws.logs / f"posthoc_{now_iso().replace(':', '')}.log")
    if a.threads:
        set_threads(a.threads)
    return run(pipe, force=a.force, threads=a.threads)


if __name__ == "__main__":
    raise SystemExit(main())
