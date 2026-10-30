"""Post-hoc, EXPLORATORY follow-ups of `hypo`, second round: the TRIPOD+AI metrics the external report lacks and an
ablation of the metric (not pre-registered; plan fixed before running, docs/DECISIONS.md 2026-10-30). The frozen
artefacts are only read; the freeze record is checked at the start and the end. Three parts:

M. Metrics of the external stage's own predictions (no refit), per external site, on S2's rows: AUPRC, Brier,
   log-loss and AUROC with 500 patient-level resamples and paired differences against PERISIM transported;
   calibration in the large and slope (point estimates; their CIs are in the external report); calibration curves
   (10 equal-count bins of each method's prediction, observed rate with a patient-level CI); decision curves (net
   benefit, thresholds 0.01–0.60, treat-all and treat-none; CIs of the difference against PERISIM transported at
   0.05, 0.10, …, 0.50). Methods: S2's, plus the transported comparators recalibrated on 500 local patients
   (draw 0). Everything per landmark.
R. The weights of the ablation, per rotation, on the frozen metric's learning sample (same rows, same seed):
   - SAGE aggregated over the competitive models of the frozen Rashomon set, refitted with its seed (each model's
     family and validation log-loss checked against the frozen record), the site's nuisance group always present;
   - mean |SHAP| of a LightGBM (T2's comparator 2: model_space.yaml baselines.lgbm, the learning sample);
   - permutation importance of the same LightGBM (trained on two thirds, log-loss on the held-out third,
     `N_PERM` repeats, negatives → 0);
   - diagonal NCA (hypo's SiteNCA and settings) on the neutral, HVDM and z-score geometries.
X. The ablation, in the transported role: 14 variants (geometry × weights, `VARIANTS`) evaluated as the external
   stage evaluates PERISIM (each external site's five folds, case base = the training folds, α by leave-one-out with
   the stage's seed). Shapes × NCA is the frozen metric and must reproduce the stage's predictions exactly.

Usage:  python scripts/posthoc_hypo_ablation.py [--parts M R X] [--sites inspire mimic eicu] [--threads 16]
        (run_local.ps1 hypo-ablation)

Outside src/hypo on purpose (the frozen code hash covers it). Caches and predictions (patient level, never exported)
in work/experiments/hypo/posthoc_ablation/; outputs (aggregates only): posthoc_ablation/*.json,
reports/posthoc_ablation.md and the figures in reports/posthoc_ablation/.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from hypo.casebase import CaseBase  # noqa: E402
from hypo.data import concat, learning_sample  # noqa: E402
from hypo.external import Stage  # noqa: E402
from hypo.external_report import _on, _phase2, _units  # noqa: E402
from hypo.frozen import check, dump, external_dir, final_dir, load_pickle  # noqa: E402
from hypo.metric import C, Metric, SiteNCA, fit_encoder  # noqa: E402
from hypo.report import auc_prep, auc_weighted  # noqa: E402
from hypo.stats import PatientBootstrap, calibration_line_w, ci, citl_w, logit, logloss_rows  # noqa: E402
from perisim.baselines.weights import shap_lgbm  # noqa: E402
from perisim.importance.sage import aggregate, normalize, sage_permutation  # noqa: E402
from perisim.importance.surrogate import Surrogate, log_loss, sample_params  # noqa: E402
from perisim.metric.encoding import Encoder, _scale, build_groups  # noqa: E402

LOG = logging.getLogger("hypo.ablation")
VDM_M = 5.0
N_PERM = 3
SAGE = {"batch_size": 512, "max_passes": 3, "threshold": 0.025}          # T2's experiment.yaml importance.sage
N_BINS = 10
DCA_GRID = np.round(np.arange(0.01, 0.6001, 0.01), 2)
DCA_CI_AT = np.round(np.arange(0.05, 0.5001, 0.05), 2)
RECAL_N = 500
VARIANTS = [("zscore", "uniform"), ("zscore", "nca"),
            ("neutral", "uniform"), ("neutral", "sage"), ("neutral", "shap"), ("neutral", "perm"), ("neutral", "nca"),
            ("hvdm", "uniform"), ("hvdm", "nca"),
            ("shape", "uniform"), ("shape", "sage"), ("shape", "shap"), ("shape", "perm"), ("shape", "nca")]
FROZEN = ("shape", "nca")
CONTRASTS = [  # (label, a, b): a − b
    ("shapes, uniform weights", ("shape", "uniform"), ("neutral", "uniform")),
    ("shapes, SAGE weights", ("shape", "sage"), ("neutral", "sage")),
    ("shapes, NCA weights", ("shape", "nca"), ("neutral", "nca")),
    ("SAGE weights, neutral axes", ("neutral", "sage"), ("neutral", "uniform")),
    ("SAGE weights, shapes", ("shape", "sage"), ("shape", "uniform")),
    ("NCA weights, neutral axes", ("neutral", "nca"), ("neutral", "uniform")),
    ("NCA weights, shapes", ("shape", "nca"), ("shape", "uniform")),
    ("NCA − SAGE, shapes", ("shape", "nca"), ("shape", "sage")),
    ("SHAP − SAGE, shapes", ("shape", "shap"), ("shape", "sage")),
    ("permutation − SAGE, shapes", ("shape", "perm"), ("shape", "sage")),
    ("HVDM − neutral, uniform", ("hvdm", "uniform"), ("neutral", "uniform")),
    ("HVDM − neutral, NCA", ("hvdm", "nca"), ("neutral", "nca")),
    ("shapes − HVDM, NCA", ("shape", "nca"), ("hvdm", "nca")),
]
GEO_LABEL = {"zscore": "z-score", "neutral": "neutral", "hvdm": "HVDM", "shape": "shapes"}
W_LABEL = {"uniform": "uniform", "sage": "SAGE", "shap": r"mean \|SHAP\|", "perm": "permutation", "nca": "NCA"}  # table cells


def out_dir() -> Path:
    return external_dir().with_name("posthoc_ablation")


def report_dir() -> Path:
    return external_dir().parent / "reports"


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def vname(v: tuple[str, str]) -> str:
    return f"{v[0]}+{v[1]}"


# --------------------------------------------------------------------------- statistics
def auprc_weighted(y: np.ndarray, w: np.ndarray, prep: tuple[np.ndarray, np.ndarray]) -> float:
    """Average precision with row weights (sklearn's definition: Σ over distinct scores, highest first, of the
    recall gained × the precision there; tied scores form one threshold)."""
    order, grp = prep
    yo, wo = y[order].astype(float), w[order]
    pos = np.bincount(grp, weights=wo * yo)[::-1]
    neg = np.bincount(grp, weights=wo * (1 - yo))[::-1]
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    P = tp[-1] if tp.size else 0.0
    if P <= 0:
        return float("nan")
    return float((pos * tp / np.maximum(tp + fp, 1e-300)).sum() / P)


def net_benefit(y: np.ndarray, w: np.ndarray, order_desc: np.ndarray, p_desc: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Net benefit at thresholds t: TP/N − FP/N · t/(1 − t), treating every row with p ≥ t (row weights)."""
    yo, wo = y[order_desc].astype(float), w[order_desc]
    ctp, cfp = np.cumsum(wo * yo), np.cumsum(wo * (1 - yo))
    k = np.searchsorted(-p_desc, -t, side="right")
    tp = np.where(k > 0, ctp[np.maximum(k - 1, 0)], 0.0)
    fp = np.where(k > 0, cfp[np.maximum(k - 1, 0)], 0.0)
    W = wo.sum()
    return tp / W - fp / W * t / (1 - t)


def treat_all(y: np.ndarray, w: np.ndarray, t: np.ndarray) -> np.ndarray:
    pi = float((w * y).sum() / w.sum())
    return pi - (1 - pi) * t / (1 - t)


def calibration_bins(p: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Bin of each row: equal-count bins of p (ties kept together)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(edges.size - 2, 0))


class Scores:
    """Statistics of several methods on the same rows and the same patient resamples."""

    def __init__(self, y: np.ndarray, subjects: np.ndarray, methods: dict[str, np.ndarray], ref: str, n_boot: int,
                 seed: int):
        self.y, self.methods, self.ref = y.astype(np.int64), methods, ref
        self.boot = PatientBootstrap(subjects, n_boot, seed)
        self.prep = {m: auc_prep(p) for m, p in methods.items()}
        self.ll = {m: logloss_rows(p, y) for m, p in methods.items()}
        self.br = {m: (p - y) ** 2 for m, p in methods.items()}
        self.desc = {m: np.argsort(-p, kind="mergesort") for m, p in methods.items()}

    def _stats(self, w: np.ndarray, m: str, constant: bool) -> dict:
        s = {"logloss": float((w * self.ll[m]).sum() / w.sum()), "brier": float((w * self.br[m]).sum() / w.sum())}
        if not constant:
            s["auroc"] = auc_weighted(self.y, w, self.prep[m])
            s["auprc"] = auprc_weighted(self.y, w, self.prep[m])
        return s

    def table(self, dca_at: np.ndarray = DCA_CI_AT, bins: dict | None = None) -> dict:
        y = self.y
        one = np.ones(len(y))
        # prevalence varies a little between folds but ranks nothing: no AUROC / AUPRC / calibration line (as S2)
        const = {m: bool(m.startswith("prevalence") or np.ptp(p) < 1e-12) for m, p in self.methods.items()}
        point = {m: self._stats(one, m, const[m]) for m in self.methods}
        B = {m: {k: [] for k in point[m]} for m in self.methods}
        nb_p = {m: net_benefit(y, one, self.desc[m], self.methods[m][self.desc[m]], dca_at) for m in self.methods}
        NB = {m: [] for m in self.methods}
        bins = bins or {}
        obs_B = {m: [] for m in bins}
        for wp in self.boot:
            w = self.boot.rows(wp)
            for m in self.methods:
                for k, v in self._stats(w, m, const[m]).items():
                    B[m][k].append(v)
                NB[m].append(net_benefit(y, w, self.desc[m], self.methods[m][self.desc[m]], dca_at))
            for m, b in bins.items():
                obs_B[m].append(np.bincount(b, weights=w * y) / np.maximum(np.bincount(b, weights=w), 1e-12))
        out = {"n": int(len(y)), "n_patients": int(self.boot.uniq.size), "prevalence": float(y.mean()),
               "methods": {}, "paired": {}, "net_benefit_ci": {"thresholds": dca_at.tolist(), "methods": {}}}
        for m in self.methods:
            r = {k: {"est": v, "ci": ci(np.asarray(B[m][k]))} for k, v in point[m].items()}
            if not const[m]:
                x = logit(self.methods[m])
                r["citl"] = citl_w(x, y.astype(float))
                r["slope"] = calibration_line_w(x, y.astype(float))[1]
            r["mean_p"] = float(self.methods[m].mean())
            out["methods"][m] = r
            if m != self.ref:
                out["paired"][m] = {k: {"est": point[self.ref][k] - v,
                                        "ci": ci(np.asarray(B[self.ref][k]) - np.asarray(B[m][k]))}
                                    for k, v in point[m].items() if k in point[self.ref]}
            nbB = np.asarray(NB[m])
            out["net_benefit_ci"]["methods"][m] = {
                "est": nb_p[m].tolist(), "ci": [ci(nbB[:, i]) for i in range(len(dca_at))],
                "vs_ref": None if m == self.ref else {
                    "est": (nb_p[self.ref] - nb_p[m]).tolist(),
                    "ci": [ci(np.asarray(NB[self.ref])[:, i] - nbB[:, i]) for i in range(len(dca_at))]}}
        out["bins_ci"] = {m: np.asarray(v).T for m, v in obs_B.items()}
        return out


# --------------------------------------------------------------------------- M: metrics of the stage's predictions
def methods_of(X: str, st: Stage) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """S2's methods on S2's rows (the external report's own assembly), plus the transported comparators
    recalibrated on RECAL_N local patients (draw 0)."""
    cfg, x = st.cfg, st.x
    T = _units(f"transported_{X}_f*.parquet")
    rows = T["row"].to_numpy()
    methods = {"PERISIM transported": T["p"].to_numpy().astype(np.float64)}
    L = _on(_units(f"local_{X}_f*.parquet"), rows, "p")
    if L is not None:
        methods["PERISIM local"] = L
    Cu = _units(f"comps_{X}_f*.parquet")
    for name in x["comparators"]:
        v = _on(Cu, rows, f"p_{name}")
        if v is not None:
            methods[f"{name} transported"] = v
        v = _on(Cu, rows, f"p_{name}_n{RECAL_N}_r0")
        if v is not None:
            methods[f"{name} transported, recalibrated ({RECAL_N})"] = v
    for name in cfg["cv"]["comparators"]:
        v = _on(_phase2(X, name), rows, "p")
        if v is not None:
            methods[f"{name} local"] = v
    from hypo.cv import cv_dir
    for e in cfg["cohorts"]:
        if e == X:
            continue
        fs = sorted(cv_dir(False).glob(f"perisim_ext-{e}_f*.parquet"))
        parts = [pl.read_parquet(f).filter(pl.col("site") == X) for f in fs]
        parts = [p for p in parts if p.height]
        if parts:
            partner = [c for c in cfg["cohorts"] if c not in (e, X)][0]
            methods[f"PERISIM with {partner} (Phase 2)"] = _on(pl.concat(parts).sort("row"), rows, "p")
    common = np.all([np.isfinite(v) for v in methods.values()], axis=0)
    return rows[common], {m: v[common] for m, v in methods.items()}


def metrics_part(st: Stage, X: str) -> dict:
    out = out_dir() / f"metrics_{X}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    t = time.perf_counter()
    rows, methods = methods_of(X, st)
    lm = st.S.lm(X)
    y, subj = lm.y[rows], lm.ft.subjects[rows]
    curve_methods = [m for m in methods if not m.startswith("prevalence")]
    bins = {m: calibration_bins(methods[m]) for m in curve_methods}
    sc = Scores(y, subj, methods, "PERISIM transported", int(st.x["n_boot"]), st.seed + 77 * st.ci(X) + 9)
    res = sc.table(bins=bins)
    calib = {}
    for m in curve_methods:
        b, p = bins[m], methods[m]
        n = np.bincount(b)
        obs_ci = res["bins_ci"][m]
        calib[m] = [{"bin": int(i), "n": int(n[i]), "mean_p": float(p[b == i].mean()), "observed": float(y[b == i].mean()),
                     "observed_ci": ci(obs_ci[i])} for i in range(n.size) if n[i] >= int(st.x["min_cell"])]
    res.pop("bins_ci")
    one = np.ones(len(y))
    res["decision_curve"] = {"thresholds": DCA_GRID.tolist(), "treat_all": treat_all(y, one, DCA_GRID).tolist(),
                             "methods": {m: net_benefit(y, one, sc.desc[m], methods[m][sc.desc[m]], DCA_GRID).tolist()
                                         for m in methods}}
    res.update(site=X, calibration_curves=calib, seconds=time.perf_counter() - t)
    _write(out, res)
    LOG.info("M (metrics) %s: %.0f s", X, res["seconds"])
    return res


# --------------------------------------------------------------------------- R: the weights
class ZScore:
    """The standardised Euclidean geometry: the models' matrix (standardised values, ordinal scaled to [0, 1], 0 / 1,
    missing indicators, one-hot nominal), each variable's block scaled to unit mean square between random pairs."""

    def __init__(self, base: Encoder, ft):
        self.base = base
        self.blocks = base.blocks
        self.feature_names = list(base.feature_names)
        self.n_features = len(self.blocks)
        widths, self.scales = [], []
        for b in self.blocks:
            e = base._model_block(b, ft)
            widths.append(e.shape[1])
            s = _scale(e)
            self.scales.append(s if s > 1e-12 else 1.0)
        self.dim_feature = np.concatenate([np.full(w, j) for j, w in enumerate(widths)]).astype(np.int64)
        self.n_dims = int(self.dim_feature.size)

    def transform(self, ft) -> np.ndarray:
        return np.hstack([self.base._model_block(b, ft) / s for b, s in zip(self.blocks, self.scales)]).astype(np.float32)

    def weights_dims(self, w: np.ndarray) -> np.ndarray:
        return np.sqrt(np.asarray(w, dtype=np.float64))[self.dim_feature].astype(np.float32)


def _drop(enc: Encoder, names: list[str]) -> Encoder:
    if names:
        enc.blocks = [b for b in enc.blocks if b.name not in set(names)]
        enc._index()
    return enc


def learning(st: Stage, X: str):
    """The frozen metric's learning sample of rotation X (final(): development pool, learn settings, final seed)."""
    dev = [c for c in st.cfg["cohorts"] if c != X]
    pool = concat([st.S.lm(c) for c in dev])
    seed = int(st.cfg["seed"]) + int(st.x["final_seed_offset"])
    rows = learning_sample(pool, int(st.mcfg["learn"]["per_patient"]), int(st.mcfg["learn"]["per_site"]), seed)
    return pool.take(rows), seed


def rashomon_models(enc: Encoder, lm, scfg: dict, space: dict, *, seed: int, threads: int) -> dict:
    """hypo.metric.shape_loop's draws, step by step with the same generator, keeping the fitted models (the curves
    and the stability trace are not recomputed; the curves' subsample is still drawn so the sequence is the same)."""
    rng = np.random.default_rng(seed)
    ft, y = lm.ft, lm.y
    Xm, owner = enc.transform_model(ft)
    gof, gnames = build_groups(enc, enc.transform(ft), float(scfg["corr_cluster_threshold"]), 10 ** 9, seed=seed)
    col_group = gof[owner]
    G = G_in = len(gnames)
    sites = np.unique(lm.site)
    if scfg.get("site_adjusted", True) and sites.size > 1:
        S = (lm.site[:, None] == sites[None, 1:]).astype(np.float32)
        Xm = np.hstack([Xm, S])
        owner = np.concatenate([owner, np.full(S.shape[1], -1)])
        col_group = np.concatenate([col_group, np.full(S.shape[1], G)])
        G += 1
    spline_cols = np.array([o >= 0 and enc.blocks[o].kind in ("continuous", "ordinal") for o in owner])
    a, b = next(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(Xm, y, groups=ft.subjects))
    a = rng.choice(a, size=min(int(scfg["max_train"]), a.size), replace=False)
    b = rng.choice(b, size=min(int(scfg["max_val"]), b.size), replace=False)
    exp_idx = rng.choice(b, size=min(int(scfg["n_explain"]), b.size), replace=False)
    n_sub = min(int(scfg["subsample"]), exp_idx.size)
    eps, n_models = float(scfg["epsilon"]), int(scfg["n_models"])
    nR = int(scfg["n_reference_pairs"])
    rng.integers(0, len(y), size=(nR * 2, 2))                       # the stability pairs, as the frozen loop draws
    vls, rows, models = [], [], []
    best, attempts = np.inf, 0
    while len(vls) < n_models and attempts < n_models * int(scfg["max_attempts_factor"]):
        attempts += 1
        fam, params = sample_params(space, rng)
        mseed = int(rng.integers(0, 2 ** 31 - 1))
        sur = Surrogate(fam, params, col_group, G, C, seed=mseed, threads=threads, spline_cols=spline_cols)
        sur.fit(Xm[a], y[a], augment_copies=int(scfg["augment_copies"]))
        vloss = log_loss(sur.predict_proba(Xm[b]), y[b])
        accepted = vloss <= (1 + eps) * best if np.isfinite(best) else True
        best = min(best, vloss)
        rows.append({"family": fam, "val_logloss": float(vloss), "accepted": bool(accepted)})
        if accepted:
            rng.choice(exp_idx, size=n_sub, replace=False)
            vls.append(vloss)
            models.append(sur)
    keep = np.asarray(vls) <= (1 + eps) * best
    return {"models": models, "keep": keep, "rows": rows, "Xm": Xm, "exp_idx": exp_idx, "gof": gof, "G_in": G_in,
            "G": G, "owner": owner}


class _InputsOnly:
    """A surrogate seen by SAGE over the input groups only (the site's nuisance group always present)."""

    def __init__(self, model, G_in: int, G: int):
        self.model, self.G_in, self.G = model, G_in, G

    def predict_proba(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.G > self.G_in:
            mask = np.hstack([mask, np.ones((mask.shape[0], self.G - self.G_in), dtype=bool)])
        return self.model.predict_proba(X, mask)


def per_variable(imp_group: np.ndarray, gof: np.ndarray) -> np.ndarray:
    imp = np.zeros(gof.size)
    for g in range(imp_group.size):
        m = np.where(gof == g)[0]
        if m.size:
            imp[m] = imp_group[g] / m.size
    return imp / imp.sum() if imp.sum() > 0 else np.full(gof.size, 1.0 / gof.size)


def permutation_importance(X: np.ndarray, y: np.ndarray, owner: np.ndarray, n_features: int, groups: np.ndarray,
                           params: dict, seed: int, threads: int, repeats: int = N_PERM) -> np.ndarray:
    """Increase in log-loss on a held-out third when the columns of a metric variable are permuted together."""
    import warnings

    import lightgbm as lgb

    tr, va = next(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, groups=groups))
    m = lgb.LGBMClassifier(**params, n_jobs=threads, random_state=seed, verbose=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X[tr], y[tr])
    Xv, yv = X[va], y[va]
    base = float(logloss_rows(m.predict_proba(Xv)[:, 1], yv).mean())
    rng = np.random.default_rng(seed)
    imp = np.zeros(n_features)
    for j in range(n_features):
        cols = np.where(owner == j)[0]
        d = []
        for _ in range(repeats):
            Xp = Xv.copy()
            Xp[:, cols] = Xv[rng.permutation(len(Xv))][:, cols]
            d.append(float(logloss_rows(m.predict_proba(Xp)[:, 1], yv).mean()) - base)
        imp[j] = max(float(np.mean(d)), 0.0)
    return imp / imp.sum() if imp.sum() > 0 else np.full(n_features, 1.0 / n_features)


def geometry(kind: str, metric, learn, dropped: list[str]):
    if kind == "shape":
        return metric.enc
    if kind == "neutral":
        return metric.enc.with_shapes({}, learn.ft)
    base = _drop(Encoder(geometry="hvdm", m=VDM_M).fit(learn.ft, learn.y, C), dropped)
    return base if kind == "hvdm" else ZScore(base, learn.ft)


def nca_weights(enc, learn, st: Stage, seed: int) -> tuple[np.ndarray, list[float]]:
    n = st.mcfg["nca"]
    nca = SiteNCA(enc.dim_feature, enc.n_features, C, n_candidates=int(n["n_candidates"]),
                  batch_size=int(n["batch_size"]), lr=float(n["lr"]), refresh_every=int(n["refresh_every"]),
                  eps0=float(n["eps0"]), label_smoothing=float(n["label_smoothing"]), seed=seed, sites=learn.site)
    nca.fit(enc.transform(learn.ft), learn.y, epochs=int(n["epochs"]), subjects=learn.ft.subjects)
    return nca.weights, list(nca.history)


def weights_part(st: Stage, X: str) -> dict:
    out = out_dir() / f"weights_ext-{X}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    t = time.perf_counter()
    cache = out.with_suffix(".pkl")
    done = load_pickle(cache) if cache.exists() else {}
    metric = st.metric(X)
    names = list(metric.enc.feature_names)
    learn, seed = learning(st, X)
    dropped = list(metric.info["dropped_site_constant"])
    res = {"rotation": f"external-{X}", "n_learn": len(learn), "seed": seed,
           "learn_reproduced": bool(len(learn) == metric.info["n_learn"])}
    # SAGE over the refitted Rashomon set
    if "sage" not in done:
        enc0, dropped0 = fit_encoder(learn, st.mcfg["shapes"])
        rs = rashomon_models(enc0, learn, st.mcfg["shapes"], st.space, seed=seed, threads=st.threads)
        frozen_rows = metric.info["models"]
        same_fam = [r["family"] for r in rs["rows"]] == [r["family"] for r in frozen_rows]
        dv = float(np.max(np.abs(np.array([r["val_logloss"] for r in rs["rows"]])
                                 - np.array([r["val_logloss"] for r in frozen_rows])))) if same_fam else None
        rng = np.random.default_rng([seed, 31, st.ci(X)])
        imps, ses = [], []
        Xe, ye = rs["Xm"][rs["exp_idx"]], learn.y[rs["exp_idx"]]
        for mdl, k in zip(rs["models"], rs["keep"]):
            if not k:
                continue
            r = sage_permutation(_InputsOnly(mdl, rs["G_in"], rs["G"]), Xe, ye, rs["G_in"], rng,
                                 batch_size=SAGE["batch_size"], max_passes=SAGE["max_passes"],
                                 threshold=SAGE["threshold"])
            v, s = normalize(r["values"], r["se"])
            imps.append(v)
            ses.append(s)
        w_sage = per_variable(aggregate(np.array(imps), np.array(ses)), rs["gof"])
        done["sage"] = {"weights": dict(zip(enc0.feature_names, w_sage.tolist())),
                        "reproduction": {"same_families": same_fam, "max_abs_val_logloss_diff": dv,
                                         "n_attempts": len(rs["rows"]), "n_competitive": int(rs["keep"].sum()),
                                         "same_variables": enc0.feature_names == names,
                                         "same_dropped": sorted(dropped0) == sorted(dropped)}}
        dump(done, cache)
        LOG.info("R %s: SAGE over %d competitive models", X, len(imps))
    # SHAP and permutation importance of one LightGBM
    if "shap" not in done:
        Xl, owner = metric.enc.transform_model(learn.ft)
        lgbp = st.space["baselines"]["lgbm"]
        done["shap"] = dict(zip(names, shap_lgbm(Xl, learn.y, owner, len(names), lgbp, seed, st.threads).tolist()))
        done["perm"] = dict(zip(names, permutation_importance(Xl, learn.y, owner, len(names), learn.ft.subjects,
                                                              lgbp, seed, st.threads).tolist()))
        dump(done, cache)
    # NCA on the other geometries
    for kind in ("neutral", "hvdm", "zscore"):
        if f"nca_{kind}" in done:
            continue
        enc = geometry(kind, metric, learn, dropped)
        w, hist = nca_weights(enc, learn, st, seed)
        done[f"nca_{kind}"] = {"weights": dict(zip(enc.feature_names, w.tolist())), "history": hist}
        dump(done, cache)
        LOG.info("R %s: NCA on the %s geometry", X, kind)
    sh = lambda d: {k: v / sum(d.values()) for k, v in d.items()}  # noqa: E731
    res.update(sage=sh(done["sage"]["weights"]), sage_reproduction=done["sage"]["reproduction"],
               shap=sh(done["shap"]), perm=sh(done["perm"]), frozen_nca=metric.weights_table(),
               **{f"nca_{k}": sh(done[f"nca_{k}"]["weights"]) for k in ("neutral", "hvdm", "zscore")},
               nca_history={k: done[f"nca_{k}"]["history"] for k in ("neutral", "hvdm", "zscore")},
               seconds=time.perf_counter() - t)
    _write(out, res)
    return res


# --------------------------------------------------------------------------- X: the ablation
def variant_metric(v: tuple[str, str], st: Stage, X: str, W: dict, cache: dict):
    """Metric(geometry, weights) of a variant for rotation X (geometries cached per rotation)."""
    g, wk = v
    metric = st.metric(X)
    if (X, g) not in cache:
        if g == "shape":
            cache[(X, g)] = metric.enc
        else:
            learn, _ = learning(st, X)
            cache[(X, g)] = geometry(g, metric, learn, list(metric.info["dropped_site_constant"]))
    enc = cache[(X, g)]
    names = list(enc.feature_names)
    if wk == "uniform":
        w = np.ones(len(names))
    elif wk == "nca":
        w = np.asarray(metric.w, float) if g == "shape" else np.array([W[f"nca_{g}"][n] for n in names])
    else:
        w = np.array([W[wk].get(n, 0.0) for n in names]) + 1e-4
    return Metric(enc, np.asarray(w, float))


NO_WEIGHTS_NEEDED = [v for v in VARIANTS if v[1] == "uniform"] + [FROZEN]


def run_units(st: Stage, X: str, W: dict | None, folds: list[int], variants: list[tuple[str, str]]) -> None:
    """The predictions of each variant on each fold of site X (patient level, cached in posthoc_ablation/units)."""
    d = out_dir() / "units"
    d.mkdir(parents=True, exist_ok=True)
    lm = st.S.lm(X)
    geo_cache: dict = {}
    for v in variants:
        if all((d / f"{vname(v)}_{X}_f{f}.parquet").exists() for f in folds):
            continue
        m = variant_metric(v, st, X, W, geo_cache)
        E = m.enc.transform(lm.ft)
        for f in folds:
            path = d / f"{vname(v)}_{X}_f{f}.parquet"
            if path.exists():
                continue
            tr, te = st.S.split(X, f)
            cb, a = st.case_base(m, lm, tr, E, st.seed + 100 * f)
            pr = cb.predict(lm.take(te), E=E[te])
            pl.DataFrame({"row": te, "p": pr["p"]}).write_parquet(path)
            path.with_suffix(".json").write_text(json.dumps({"variant": vname(v), "fold": f, **a}, default=float),
                                                 encoding="utf-8")
        LOG.info("X %s: %s done", X, vname(v))


def ablation_part(st: Stage, X: str, W: dict, folds: list[int]) -> dict:
    out = out_dir() / f"ablation_{X}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    t = time.perf_counter()
    d = out_dir() / "units"
    run_units(st, X, W, folds, VARIANTS)
    lm = st.S.lm(X)
    # scores on the stage's rows
    T = _units(f"transported_{X}_f*.parquet")
    rows = T["row"].to_numpy()
    methods = {}
    for v in VARIANTS:
        parts = [pl.read_parquet(d / f"{vname(v)}_{X}_f{f}.parquet") for f in folds]
        methods[vname(v)] = _on(pl.concat(parts).sort("row"), rows, "p")
    repro = float(np.nanmax(np.abs(methods[vname(FROZEN)] - T["p"].to_numpy())))
    ok = np.all([np.isfinite(p) for p in methods.values()], axis=0)
    y, subj = lm.y[rows[ok]], lm.ft.subjects[rows[ok]]
    meth = {k: p[ok] for k, p in methods.items()}
    sc = Scores(y, subj, meth, vname(FROZEN), int(st.x["n_boot"]), st.seed + 77 * st.ci(X) + 11)
    res = sc.table(dca_at=np.array([0.2]))
    res.pop("bins_ci")
    res.pop("net_benefit_ci")
    # every contrast on the same resamples
    con = {}
    B = {m: {"logloss": [], "auroc": []} for m in meth}
    for wp in sc.boot:
        w = sc.boot.rows(wp)
        for m in meth:
            B[m]["logloss"].append(float((w * sc.ll[m]).sum() / w.sum()))
            B[m]["auroc"].append(auc_weighted(sc.y, w, sc.prep[m]))
    for label, a, b in CONTRASTS + [(f"{vname(v)} − z-score uniform", v, ("zscore", "uniform")) for v in VARIANTS
                                    if v != ("zscore", "uniform")]:
        A, Bn = vname(a), vname(b)
        con[label] = {"a": A, "b": Bn, **{k: {"est": res["methods"][A][k]["est"] - res["methods"][Bn][k]["est"],
                                               "ci": ci(np.asarray(B[A][k]) - np.asarray(B[Bn][k]))}
                                           for k in ("logloss", "auroc")}}
    alphas = {vname(v): [json.loads((d / f"{vname(v)}_{X}_f{f}.json").read_text(encoding="utf-8"))["alpha"]
                         for f in folds] for v in VARIANTS}
    res.update(site=X, folds=folds, reproduces_stage=repro, contrasts=con, alpha=alphas, seconds=time.perf_counter() - t)
    _write(out, res)
    return res


# --------------------------------------------------------------------------- report
def _e(r: dict | None, d: int = 3, sign: bool = False) -> str:
    if not r or r.get("est") is None or not np.isfinite(r["est"]):
        return "—"
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    lo, hi = r["ci"]
    return f"{f.format(r['est'])} ({f.format(lo)} to {f.format(hi)})"


def figures(M: dict, folder: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    show = ["PERISIM transported", "PERISIM local", "last_map transported", "logreg transported",
            "lightgbm transported", f"lightgbm transported, recalibrated ({RECAL_N})", "lightgbm local"]
    made = []
    for X, r in M.items():
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for m in show:
            if m not in r["calibration_curves"]:
                continue
            c = r["calibration_curves"][m]
            ax[0].plot([b["mean_p"] for b in c], [b["observed"] for b in c], "o-", ms=3, label=m)
        lim = max(max(b["observed"] for b in r["calibration_curves"]["PERISIM transported"]), 0.6)
        ax[0].plot([0, lim], [0, lim], color="grey", lw=0.8, ls="--")
        ax[0].set(xlabel="mean predicted risk (10 bins)", ylabel="observed rate", title=f"{X}: calibration")
        ax[0].legend(fontsize=6)
        dc = r["decision_curve"]
        tt = np.array(dc["thresholds"])
        for m in show:
            if m in dc["methods"]:
                ax[1].plot(tt, dc["methods"][m], label=m, lw=1)
        ax[1].plot(tt, dc["treat_all"], color="grey", ls=":", label="treat all")
        ax[1].axhline(0, color="black", lw=0.6)
        ax[1].set(xlabel="threshold probability", ylabel="net benefit", title=f"{X}: decision curve",
                  ylim=(-0.05, max(r["prevalence"], 0.05) * 1.05))
        ax[1].legend(fontsize=6)
        fig.tight_layout()
        p = folder / f"calibration_dca_{X}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p.name)
    return made


def render(M: dict, R: dict, Xa: dict, figs: list[str]) -> str:
    L = ["# `hypo` post hoc, second round: missing metrics and ablation (exploratory, not pre-registered)", "",
         "Plan fixed before running (docs/DECISIONS.md, 2026-10-30). Frozen artefacts only read; the freeze record "
         "checked at the start and the end. Differences: PERISIM transported (M) or the frozen metric (ablation) "
         "minus the other method; 95 % CIs from 500 patient-level resamples. Per landmark.", ""]
    if M:
        L += ["## M. Metrics of the external stage's predictions", ""]
        for X, r in M.items():
            L += [f"### {X} ({r['n']:,} landmarks, {r['n_patients']:,} patients, prevalence {r['prevalence']:.1%})", "",
                  "| method | AUROC | AUPRC | Brier | log-loss | CITL | slope | AUPRC: PERISIM − method | Brier: PERISIM − method |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for m, s in r["methods"].items():
                pr = r["paired"].get(m, {})
                L.append(f"| {m} | {_e(s.get('auroc'))} | {_e(s.get('auprc'))} | {_e(s['brier'], 4)} | "
                         f"{_e(s['logloss'], 4)} | {s.get('citl', float('nan')):+.3f} | {s.get('slope', float('nan')):.3f} | "
                         f"{_e(pr.get('auprc'), 3, True)} | {_e(pr.get('brier'), 4, True)} |"
                         if "citl" in s else
                         f"| {m} | — | — | {_e(s['brier'], 4)} | {_e(s['logloss'], 4)} | — | — | — | "
                         f"{_e(pr.get('brier'), 4, True)} |")
            nb = r["net_benefit_ci"]
            th = nb["thresholds"]
            L += ["", "Net benefit of PERISIM transported minus each method (thresholds " +
                  ", ".join(f"{x:.2f}" for x in th[1::2]) + "):", "",
                  "| method | " + " | ".join(f"{x:.2f}" for x in th[1::2]) + " |", "|---|" + "---|" * len(th[1::2])]
            for m, v in nb["methods"].items():
                if v["vs_ref"] is None:
                    continue
                L.append(f"| {m} | " + " | ".join(
                    _e({"est": v["vs_ref"]["est"][i], "ci": v["vs_ref"]["ci"][i]}, 4, True) for i in range(1, len(th), 2)) + " |")
            L.append("")
        L += [f"Figures (calibration curves and decision curves per site): {', '.join(figs)}.", ""]
    if R:
        L += ["## R. The weights (share of the distance, per rotation)", ""]
        for X, r in R.items():
            rp = r["sage_reproduction"]
            L.append(f"- external-{X}: learning sample reproduced {r['learn_reproduced']}; Rashomon refit: same "
                     f"families {rp['same_families']}, max |Δ validation log-loss| {rp['max_abs_val_logloss_diff']}, "
                     f"{rp['n_competitive']} competitive models.")
        L.append("")
        names = list(next(iter(R.values()))["frozen_nca"])
        L += ["| variable | " + " | ".join(f"{X}: NCA / SAGE / SHAP / perm." for X in R) + " |",
              "|---|" + "---|" * len(R)]
        for n in names:
            L.append(f"| {n} | " + " | ".join(
                f"{r['frozen_nca'].get(n, 0):.3f} / {r['sage'].get(n, 0):.3f} / {r['shap'].get(n, 0):.3f} / "
                f"{r['perm'].get(n, 0):.3f}" for r in R.values()) + " |")
        L.append("")
    if Xa:
        L += ["## X. Ablation in the transported role (each external site's five folds)", ""]
        for X, r in Xa.items():
            L += [f"### {X} (reproduction of the stage by shapes + NCA: max |Δp| = {r['reproduces_stage']:.2e})", "",
                  "| geometry | weights | log-loss | AUROC | AUPRC | Brier | CITL | Δ log-loss vs frozen | Δ AUROC vs frozen |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for v in VARIANTS:
                m = vname(v)
                s, pr = r["methods"][m], r["paired"].get(m, {})
                neg = lambda d: None if not d else {"est": -d["est"], "ci": [-d["ci"][1], -d["ci"][0]]}  # noqa: E731
                L.append(f"| {GEO_LABEL[v[0]]} | {W_LABEL[v[1]]} | {_e(s['logloss'], 4)} | {_e(s.get('auroc'))} | "
                         f"{_e(s.get('auprc'))} | {_e(s['brier'], 4)} | {s.get('citl', float('nan')):+.3f} | "
                         f"{_e(neg(pr.get('logloss')), 4, True)} | {_e(neg(pr.get('auroc')), 3, True)} |")
            L += ["", "| contrast (a − b) | Δ log-loss | Δ AUROC |", "|---|---|---|"]
            for label, c in r["contrasts"].items():
                L.append(f"| {label} | {_e(c['logloss'], 4, True)} | {_e(c['auroc'], 3, True)} |")
            L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="*", default=["M", "R", "X"], choices=["M", "R", "X"])
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    rec = check()
    st = Stage(threads=a.threads)
    sites = a.sites or list(st.cfg["cohorts"])
    out_dir().mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    M = {X: metrics_part(st, X) for X in sites} if "M" in a.parts else {}
    if "X" in a.parts:                       # the variants that need no refitted weight first: a fault shows early
        for X in sites:
            run_units(st, X, None, a.folds, NO_WEIGHTS_NEEDED)
    R = {X: weights_part(st, X) for X in sites} if ("R" in a.parts or "X" in a.parts) else {}
    Xa = {X: ablation_part(st, X, R[X], a.folds) for X in sites} if "X" in a.parts else {}
    check()
    figs = figures(M, report_dir() / "posthoc_ablation") if M else []
    summary = {"freeze": rec["tag"], "parts": a.parts, "sites": sites, "M": M, "R": R, "X": Xa,
               "figures": figs, "seconds": time.perf_counter() - t}
    _write(out_dir() / "posthoc_ablation.json", summary)
    rep = report_dir() / "posthoc_ablation.md"
    rep.write_text(render(M, R, Xa, figs), encoding="utf-8")
    LOG.info("post hoc report: %s", rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
