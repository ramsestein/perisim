"""Fitting and evaluation of PERISIM and comparators within a fold (§5, §6.1, §6.4).

Everything learned from the data (standardization, VDM and m, groups, SAGE, weights, λ, α, T2/T3 thresholds)
is fitted on the training indices the function receives; k is not tuned (it is evaluated over the grid).
λ is selected with a SAGE prior recomputed inside each inner training fold, so the prior that scores an inner
validation fold never saw its labels (AUDIT C3, decided 2026-10-22). Geometry "shape" (v0.5, 2026-10-24): the
consensus loop also produces the per-variable shape functions, installed in the encoder before the NCA.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from perisim.baselines import parametric, weights as bw
from perisim.baselines.rfgap import RFGAP
from perisim.features.table import FeatureTable
from perisim.importance.loop import run_importance_loop
from perisim.knn.predict import KNNModel, patient_pieces
from perisim.metric.encoding import Encoder, build_groups
from perisim.metric.nca import DiagonalNCA
from perisim.targets.build import t2_categories, t2_thresholds, t3_labels, t3_thresholds
from perisim.utils import rss_gb

LOG = logging.getLogger("perisim.experiment")
TARGET_CLASSES = {"t1": 3, "t2": 4, "t3": 4, "t1_bin": 2}
ORDINAL = {"t1": True, "t2": True, "t3": True, "t1_bin": False}
SEVERE = {"t1": 2, "t2": 3, "t3": 3, "t1_bin": 1}     # worst class (T2: in-hospital death)
KNN_METHODS = {"perisim", "perisim_kernel", "euclidean", "shap_lgbm", "rf_importance", "relieff", "nca_diag",
               "power_prior", "rf_gap"}


@dataclass
class CohortData:
    cohort: str
    ft: FeatureTable
    t1: np.ndarray
    t1_severe: np.ndarray
    los: np.ndarray
    carried: dict[str, np.ndarray]
    t0: np.ndarray
    year: np.ndarray
    extra: dict = field(default_factory=dict)
    death: np.ndarray | None = None      # 1 = in-hospital death within the follow-up, 0 = survivor, −1 = unknown

    def __len__(self) -> int:
        return len(self.ft)


def load_cohort_data(common_dir: Path, derived_dir: Path, feature_names: list[str] | None,
                     targets_dir: Path | None = None) -> CohortData:
    """`targets_dir` (relabelled sensitivity variants): outcomes from another derived folder, inputs from this one."""
    stays = pl.read_parquet(common_dir / "stays_index.parquet").sort("stay_id")
    ft = FeatureTable.load(derived_dir, stays, feature_names)
    tg = pl.read_parquet((targets_dir or derived_dir) / "targets.parquet")
    car = pl.read_parquet((targets_dir or derived_dir) / "carried.parquet")
    order = pl.DataFrame({"stay_id": ft.ids.astype(str)})
    tg = order.join(tg, on="stay_id", how="left")
    car = order.join(car, on="stay_id", how="left")
    st = order.join(stays.select("stay_id", "icu_out_min", "year"), on="stay_id", how="left")
    carried = {}
    for c in car.columns:
        if c == "stay_id":
            continue
        if car.schema[c] == pl.Utf8:
            continue
        carried[c] = car[c].cast(pl.Float64).to_numpy()
    return CohortData(
        cohort=str(stays["cohort"][0]) if stays.height else "",
        ft=ft,
        t1=tg["t1"].fill_null(-1).cast(pl.Int64).to_numpy(),
        t1_severe=tg["t1_severe"].fill_null(-1).cast(pl.Int64).to_numpy(),
        los=tg["los_post_icu_days"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        carried=carried,
        t0=st["icu_out_min"].to_numpy(),
        year=st["year"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        extra={"t1_reduced": _reduced_severe(tg)},
        death=(tg["death"].cast(pl.Int64).fill_null(-1).to_numpy() if "death" in tg.columns
               else np.full(tg.height, -1, dtype=np.int64)),
    )


def _reduced_severe(tg: pl.DataFrame) -> np.ndarray:
    """Severe restricted to the components eICU can observe (ICU readmission or in-hospital death); used as the
    INSPIRE case-base label when INSPIRE predicts the reduced eICU target (AUDIT E1). −1 when unobservable."""
    r = tg["icu_readmission"].cast(pl.Float64) if "icu_readmission" in tg.columns else pl.Series([None] * tg.height)
    d = tg["death"].cast(pl.Float64) if "death" in tg.columns else pl.Series([None] * tg.height)
    out = pl.DataFrame({"r": r, "d": d}).select(
        pl.when(pl.col("r").is_null() & pl.col("d").is_null()).then(-1)
        .when((pl.col("r").fill_null(0) == 1) | (pl.col("d").fill_null(0) == 1)).then(1).otherwise(0)
        .cast(pl.Int64).alias("x"))
    return out["x"].to_numpy()


def fold_seed(cfg, fold: int) -> int:
    """Seed of an outer-fold job (part of the hashed scientific code, unlike the orchestration in pipeline.py)."""
    return int(cfg.seed) + 1000 * int(fold)


# --------------------------------------------------------------------------- partitions
def outer_folds(data: CohortData, n_splits: int, seed: int) -> np.ndarray:
    """Outer fold by patient (subject_id), stratified by T1 (§6.1)."""
    y = np.where(data.t1 >= 0, data.t1, 0)
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(data), -1)
    for f, (_, te) in enumerate(sgk.split(np.zeros(len(y)), y, groups=data.ft.subjects)):
        fold[te] = f
    return fold


def inner_splits(y: np.ndarray, subjects: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(a, b) for a, b in sgk.split(np.zeros(len(y)), y, groups=subjects)]


def labels_for(target: str, data: CohortData, train_idx: np.ndarray, tcfg: dict) -> tuple[np.ndarray, dict]:
    """Target labels with thresholds computed ONLY on the training set (−1 = excluded from the analysis)."""
    if target == "t1":
        return data.t1.copy(), {}
    if target == "t1_bin":
        return data.t1_severe.copy(), {}
    if target == "t2":
        thr = t2_thresholds(data.los[train_idx], tcfg)      # survivors' stay only (los is NaN for deaths)
        return t2_categories(data.los, thr, data.death), {"t2_thresholds": thr}
    if target == "t3":
        tr = train_idx[data.t1[train_idx] >= 0]
        thr = t3_thresholds(data.los[tr], data.t1[tr], tcfg)
        y = t3_labels(data.t1, data.los, thr)
        y[data.t1 < 0] = -1
        return y, {"t3_thresholds": thr}
    raise ValueError(target)


# --------------------------------------------------------------------------- building blocks
def _nca(cfg, enc: Encoder, C: int, seed: int, **kw) -> DiagonalNCA:
    m = cfg.experiment["metric"]
    n = m["nca"]
    return DiagonalNCA(enc.dim_feature, enc.n_features, C, n_candidates=kw.get("n_candidates", m["n_candidates"]),
                       batch_size=n["batch_size"], lr=n["lr"], refresh_every=n["refresh_every"], eps0=n["eps0"],
                       label_smoothing=n["label_smoothing"], seed=seed)


def _map_prior(imp: np.ndarray, names_src: list[str], names_dst: list[str]) -> np.ndarray:
    d = dict(zip(names_src, imp))
    floor = float(np.min(imp)) if len(imp) else 1e-6
    out = np.array([d.get(n, floor) for n in names_dst])
    return out / out.sum() if out.sum() > 0 else np.full(len(names_dst), 1.0 / max(len(names_dst), 1))


def select_m(cfg, ft: FeatureTable, y: np.ndarray, C: int, inner, seed: int) -> tuple[float, dict]:
    """VDM m by inner CV with the soft-kernel log-loss (uniform weights + fitted scale)."""
    res = {}
    for m in cfg.experiment["metric"]["vdm_m_grid"]:
        ll = []
        for a, b in inner:
            enc = Encoder.from_config(cfg, m).fit(ft.take(a), y[a], C)
            Ea, Eb = enc.transform(ft.take(a)), enc.transform(ft.take(b))
            nca = _nca(cfg, enc, C, seed)
            base = np.ones(enc.n_features)
            nca.fit(Ea, y[a], epochs=3, mode="scale", base_w=base, subjects=ft.subjects[a])
            ll.append(nca.soft_loss(Ea, y[a], Eb, y[b], subj_ref=ft.subjects[a], subj_q=ft.subjects[b]))
        res[float(m)] = float(np.mean(ll))
    best = min(res, key=res.get)
    return best, res


def importance_prior(cfg, ft: FeatureTable, y: np.ndarray, C: int, m: float, *, seed: int, threads: int,
                     stopping_override: dict | None = None, exclude_families: set[str] | None = None,
                     early_stop: bool = True, checkpoint: Path | None = None) -> tuple:
    """Encoder, SAGE groups, consensus loop (importances and shapes, §5.2–§5.3) and the final embedding, all
    fitted on `ft` only."""
    ecfg = cfg.experiment
    enc = Encoder.from_config(cfg, m).fit(ft, y, C)
    E0 = enc.transform(ft)                  # neutral axes (geometry shape) or the metric itself (hvdm)
    Xm, owner = enc.transform_model(ft)
    gof, gnames = build_groups(enc, E0, ecfg["features"]["corr_cluster_threshold"],
                               int(ecfg["features"]["block_group_threshold"]), seed=seed)
    a0, b0 = inner_splits(y, ft.subjects, int(ecfg["cv"]["inner_folds"]), seed)[0]
    loop = run_importance_loop(X_model=Xm, owner=owner, col_group=gof[owner], n_groups=len(gnames), group_names=gnames,
                               y=y, n_classes=C, inner_train=a0, inner_val=b0, enc=enc, ft=ft, group_of_feature=gof,
                               subjects=ft.subjects, cfg=cfg, seed=seed, threads=threads,
                               stopping_override=stopping_override, exclude_families=exclude_families,
                               early_stop=early_stop, checkpoint=checkpoint)
    if enc.geometry == "shape":
        enc = enc.with_shapes(loop["shapes"], ft)
    return enc, enc.transform(ft), gof, gnames, loop


def select_lambda(cfg, ft: FeatureTable, y: np.ndarray, C: int, inner, m: float, *, seed: int, epochs: int,
                  threads: int, outer_prior: tuple | None = None, checkpoint_dir: Path | None = None,
                  **loop_kw) -> tuple[float, dict, dict]:
    """λ by inner CV with the soft-kNN log-loss. With `metric.lambda_prior: inner_fold_sage` (C3) the SAGE prior —
    and, in geometry shape, the shapes — of each inner fold are recomputed on its training part; `outer` reuses the
    fold's encoder and prior (v0, declared bias)."""
    res: dict[float, list[float]] = {}
    loops: list[dict] = []
    per_inner = str(cfg.experiment["metric"].get("lambda_prior", "inner_fold_sage")) == "inner_fold_sage"
    for i, (a, b) in enumerate(inner):
        if per_inner:
            ck = checkpoint_dir / f"loop_inner{i}.pkl" if checkpoint_dir else None
            enc, Ea, gof, _, lp = importance_prior(cfg, ft.take(a), y[a], C, m, seed=seed + 7919 * (i + 1),
                                                   threads=threads, checkpoint=ck, **loop_kw)
            prior = lp["feature_importance"]
            loops.append({"n_accepted": lp["n_accepted"], "stop_reason": lp["stop_reason"], "seconds": lp["seconds"]})
        else:
            enc, prior, gof = outer_prior
            Ea = enc.transform(ft.take(a))
        Eb = enc.transform(ft.take(b))
        for lam in cfg.experiment["metric"]["lambda_grid"]:
            nca = _nca(cfg, enc, C, seed)
            nca.fit(Ea, y[a], prior=prior, groups=gof, lam=float(lam), epochs=epochs, subjects=ft.subjects[a])
            res.setdefault(float(lam), []).append(
                nca.soft_loss(Ea, y[a], Eb, y[b], subj_ref=ft.subjects[a], subj_q=ft.subjects[b]))
    mean = {lam: float(np.mean(v)) for lam, v in res.items()}
    for lam, v in mean.items():
        LOG.info("    λ=%g → inner soft-kNN log-loss %.4f", lam, v)
    best = min(mean, key=mean.get)
    return best, mean, {"prior": "inner_fold_sage" if per_inner else "outer", "inner_loops": loops}


def swift_inputs(ft: FeatureTable) -> np.ndarray:
    g = lambda n: ft.num.get(n, np.full(len(ft), np.nan))  # noqa: E731
    fio2 = g("last_fio2")
    pf = g("last_po2") / np.where(np.isfinite(fio2) & (fio2 > 0), fio2 / 100.0, np.nan)
    src = ft.cat.get("admit_source", np.full(len(ft), None, dtype=object))
    return parametric.swift_score(g("icu_los_h"), pf, g("last_gcs"), g("last_pco2"), src)


# --------------------------------------------------------------------------- full PERISIM fit
@dataclass
class FittedPerisim:
    enc: Encoder
    w: np.ndarray
    knn: KNNModel
    info: dict
    imp_feature: np.ndarray
    loop: dict
    extra_weights: dict
    group_of_feature: np.ndarray | None = None
    group_names: list[str] | None = None


def fit_perisim(cfg, ft: FeatureTable, y: np.ndarray, C: int, *, seed: int, threads: int,
                stopping_override: dict | None = None, exclude_families: set[str] | None = None,
                early_stop: bool = True, select_hparams: bool = True, timings: dict | None = None,
                need_comparator_weights: bool = True, need_hvdm: bool = False,
                checkpoint_dir: Path | None = None) -> FittedPerisim:
    ecfg = cfg.experiment
    timings = {} if timings is None else timings
    inner = inner_splits(y, ft.subjects, int(ecfg["cv"]["inner_folds"]), seed)
    from perisim.evaluation.privacy import counts as _counts

    info: dict = {"n_train": int(len(y)),
                  "class_counts": _counts({str(c): int(v) for c, v in enumerate(np.bincount(y, minlength=C))})}

    t = time.perf_counter()
    geometry = ecfg["metric"].get("geometry", "shape")
    m_default = float(ecfg["metric"]["vdm_m_grid"][min(1, len(ecfg["metric"]["vdm_m_grid"]) - 1)])
    if select_hparams and geometry == "hvdm" and len(ecfg["metric"]["vdm_m_grid"]) > 1:
        m, info["m_grid_loss"] = select_m(cfg, ft, y, C, inner, seed)
    else:
        m = m_default                       # geometry shape: no VDM (m only for the HVDM comparator)
    info["m"] = m if geometry == "hvdm" else None
    info["geometry"] = geometry
    timings["select_m"] = time.perf_counter() - t

    t = time.perf_counter()
    loop_kw = {"stopping_override": stopping_override, "exclude_families": exclude_families, "early_stop": early_stop}
    ck = checkpoint_dir / "loop_main.pkl" if checkpoint_dir else None
    enc, E, gof, gnames, loop = importance_prior(cfg, ft, y, C, m, seed=seed, threads=threads, checkpoint=ck, **loop_kw)
    info.update(n_features=enc.n_features, n_dims=enc.n_dims, n_groups=len(gnames),
                n_shaped=int(sum(b.shape is not None for b in enc.blocks)))
    timings["importance_loop"] = time.perf_counter() - t
    imp = loop["feature_importance"]

    epochs = int(ecfg["metric"]["nca"]["epochs"])
    t = time.perf_counter()
    if select_hparams and len(ecfg["metric"]["lambda_grid"]) > 1:
        lam, info["lambda_grid_loss"], info["lambda_selection"] = select_lambda(
            cfg, ft, y, C, inner, m, seed=seed, epochs=epochs, threads=threads, outer_prior=(enc, imp, gof),
            checkpoint_dir=checkpoint_dir, **loop_kw)
    else:
        lam = float(ecfg["metric"].get("lambda_default", ecfg["metric"]["lambda_grid"][0]))
    info["lambda"] = lam
    timings["select_lambda"] = time.perf_counter() - t

    t = time.perf_counter()
    nca = _nca(cfg, enc, C, seed)
    nca.fit(E, y, prior=imp, groups=gof, lam=lam, epochs=epochs, subjects=ft.subjects)
    w = nca.weights
    info["nca_history"] = nca.history
    info["nca_params"] = getattr(nca, "params_", {})
    timings["nca_final"] = time.perf_counter() - t

    extra = {}
    if need_comparator_weights:
        t = time.perf_counter()
        if lam == 0.0:
            extra["nca_diag"] = w
        else:
            n0 = _nca(cfg, enc, C, seed)
            n0.fit(E, y, prior=None, lam=0.0, epochs=epochs, subjects=ft.subjects)
            extra["nca_diag"] = n0.weights
        npw = _nca(cfg, enc, C, seed)
        npw.fit(E, y, prior=imp, epochs=max(epochs // 2, 1), mode="power", subjects=ft.subjects)
        extra["power_prior"] = npw.weights
        info["power_params"] = getattr(npw, "params_", {})
        timings["comparator_nca"] = time.perf_counter() - t
    if need_hvdm and geometry == "shape":
        # v0.4 geometry with the SAME importance prior and λ: isolates the effect of the shapes
        t = time.perf_counter()
        enc_h = Encoder.from_config(cfg, m_default, geometry="hvdm").fit(ft, y, C)
        E_h = enc_h.transform(ft)
        by_name = {n: g for n, g in zip(enc.feature_names, gof)}
        gof_h = np.unique([str(by_name.get(n, n)) for n in enc_h.feature_names], return_inverse=True)[1]
        nh = _nca(cfg, enc_h, C, seed)
        nh.fit(E_h, y, prior=_map_prior(imp, enc.feature_names, enc_h.feature_names), groups=gof_h, lam=lam,
               epochs=epochs, subjects=ft.subjects)
        extra["hvdm"] = (enc_h, nh.weights)
        timings["comparator_hvdm"] = time.perf_counter() - t

    knn = KNNModel(E, enc.weights_dims(w), y, C, ft.subjects, ecfg["knn"]["k_grid"], ids_ref=ft.ids)
    info["alpha"] = knn.fit_alpha_and_ood(ecfg["knn"]["alpha_grid"], ecfg["knn"]["k_primary"],
                                          ecfg["knn"]["ood_percentile"])
    # scale for the exp(−d²) kernel variant: that of the learned metric itself (NCA)
    info["rss_gb"] = rss_gb()
    return FittedPerisim(enc, w, knn, info, imp, loop, extra, gof, gnames)


def knn_predict(model: KNNModel, E_q: np.ndarray, subj_q: np.ndarray, k_grid: list[int], kernel_k: int | None = None
                ) -> tuple[dict, tuple]:
    D, I, valid = model.neighbors(E_q, subj_q)
    probs = {k: model.probs(I, valid, min(k, model.k_max)) for k in k_grid}
    if kernel_k is not None:
        probs[("kernel", kernel_k)] = model.probs(I, valid, min(kernel_k, model.k_max), D=D, kernel=True)
    return probs, (D, I, valid)


# --------------------------------------------------------------------------- one outer fold
def _density(E_ref: np.ndarray, sw: np.ndarray, kth: np.ndarray, w: np.ndarray, seed: int) -> dict:
    """Gap 9 diagnostics: effective dimension of the weights (participation ratio) and contrast = median distance
    to the k-th neighbour / median distance between random training pairs (→ 1: neighbours are no closer than any
    patient, i.e. no real profiles)."""
    rng = np.random.default_rng(seed)
    n = E_ref.shape[0]
    a, b = rng.integers(0, n, 4000), rng.integers(0, n, 4000)
    ok = a != b
    d = np.sqrt((((E_ref[a[ok]] - E_ref[b[ok]]) * sw) ** 2).sum(1))
    wn = np.asarray(w, dtype=float) / max(float(np.sum(w)), 1e-12)
    med = float(np.median(d)) if d.size else float("nan")
    return {"effective_dimension": float(1.0 / max(float((wn * wn).sum()), 1e-12)),
            "contrast_kth_vs_random": float(np.median(kth)) / med if med and np.isfinite(med) else None}


def run_fold(cfg, data: CohortData, target: str, train_idx: np.ndarray, test_idx: np.ndarray, *, seed: int,
             threads: int, baselines: list[str], stopping_override: dict | None = None,
             exclude_families: set[str] | None = None, early_stop: bool = True, select_hparams: bool = True,
             checkpoint_dir: Path | None = None) -> tuple[dict, pl.DataFrame]:
    ecfg = cfg.experiment
    t_all = time.perf_counter()
    y_all, thr = labels_for(target, data, train_idx, cfg.targets)
    C = TARGET_CLASSES[target]
    tr = train_idx[y_all[train_idx] >= 0]
    te = test_idx[y_all[test_idx] >= 0]
    ft_tr, ft_te = data.ft.take(tr), data.ft.take(te)
    y_tr, y_te = y_all[tr], y_all[te]
    timings: dict = {}
    LOG.info("[%s] train=%d test=%d classes=%s", target, len(tr), len(te), np.bincount(y_tr, minlength=C).tolist())

    fp = fit_perisim(cfg, ft_tr, y_tr, C, seed=seed, threads=threads, stopping_override=stopping_override,
                     exclude_families=exclude_families, early_stop=early_stop, select_hparams=select_hparams,
                     timings=timings, need_comparator_weights=bool(set(baselines) & {"nca_diag", "power_prior"}),
                     need_hvdm="hvdm" in baselines, checkpoint_dir=checkpoint_dir)
    k_grid = [int(k) for k in ecfg["knn"]["k_grid"]]
    kp = int(ecfg["knn"]["k_primary"])
    E_te = fp.enc.transform(ft_te)
    preds: dict[tuple[str, int], np.ndarray] = {}
    probs, (D, I, valid) = knn_predict(fp.knn, E_te, ft_te.subjects, k_grid, kernel_k=kp)
    for k in k_grid:
        preds[("perisim", k)] = probs[k]
    preds[("perisim_kernel", kp)] = probs[("kernel", kp)]
    kth, pct, ood = fp.knn.kth_info(D, valid, kp)

    # Q4: carried variables summarized over the k neighbours (learned metric vs Euclidean)
    car_tr = {k: v[tr] for k, v in data.carried.items() if np.isfinite(v[tr]).any()}
    car_te = {k: data.carried[k][te] for k in car_tr}
    p4 = {"perisim": fp.knn.carried_summary(I, valid, kp, car_tr)}

    t = time.perf_counter()
    E_tr = fp.enc.transform(ft_tr)
    N_tr, N_te = fp.enc.transform_neutral(ft_tr), fp.enc.transform_neutral(ft_te)   # no outcome information
    Xm_tr, owner = fp.enc.transform_model(ft_tr)
    Xm_te, _ = fp.enc.transform_model(ft_te)
    ms = cfg.model_space["baselines"]
    inner = inner_splits(y_tr, ft_tr.subjects, int(ecfg["cv"]["inner_folds"]), seed)
    bl_info: dict = {}
    density = {"perisim": {**_density(E_tr, fp.enc.weights_dims(fp.w), kth, fp.w, seed),
                           "ood_rate_test": float(np.mean(ood)) if len(ood) else None}}

    def _knn_weights(name: str, w: np.ndarray, geometry: str = "neutral", enc_h=None) -> None:
        """Weighted kNN comparators. neutral: literature-style weights on unsupervised axes; shape: PERISIM's own
        axes (metric-learning comparators); hvdm: the v0.4 geometry."""
        if geometry == "hvdm":
            Er, Eq, sw, names = enc_h.transform(ft_tr), enc_h.transform(ft_te), enc_h.weights_dims(w), enc_h.feature_names
        elif geometry == "shape":
            Er, Eq, sw, names = E_tr, E_te, fp.enc.weights_dims(w), fp.enc.feature_names
        else:
            Er, Eq, sw, names = N_tr, N_te, fp.enc.neutral_weights_dims(w), fp.enc.feature_names
        mdl = KNNModel(Er, sw, y_tr, C, ft_tr.subjects, k_grid)
        bl_info[name] = {"alpha": mdl.fit_alpha_and_ood(ecfg["knn"]["alpha_grid"], kp, ecfg["knn"]["ood_percentile"]),
                         "weights": dict(zip(names, np.round(w / w.sum(), 6).tolist())), "geometry": geometry}
        pr, (Db, Ib, vb) = knn_predict(mdl, Eq, ft_te.subjects, k_grid)
        for k in k_grid:
            preds[(name, k)] = pr[k]
        if name in ("euclidean", "hvdm"):
            p4[name] = mdl.carried_summary(Ib, vb, kp, car_tr)
            kth_b, _, ood_b = mdl.kth_info(Db, vb, kp)
            density[name] = {**_density(Er, sw, kth_b, w, seed), "ood_rate_test": float(np.mean(ood_b))}

    F = fp.enc.n_features
    for name in baselines:
        try:
            tb = time.perf_counter()
            if name == "prevalence":            # trivial: training class frequencies for everyone
                pri = np.bincount(y_tr, minlength=C) / len(y_tr)
                preds[(name, 0)] = np.tile(pri, (len(te), 1))
            elif name == "group_prevalence":    # trivial: prevalence by surgical group (m-estimate)
                if "surgical_group" in ft_tr.cat:
                    preds[(name, 0)] = _group_prevalence(ft_tr.cat["surgical_group"], y_tr,
                                                         ft_te.cat["surgical_group"], C,
                                                         float(ms.get("group_prevalence_m", 20)))
            elif name == "euclidean":
                _knn_weights(name, np.ones(F))
            elif name == "hvdm" and "hvdm" in fp.extra_weights:
                enc_h, w_h = fp.extra_weights["hvdm"]
                _knn_weights(name, w_h, "hvdm", enc_h)
            elif name == "shap_lgbm":
                _knn_weights(name, np.maximum(bw.shap_lgbm(Xm_tr, y_tr, owner, F, ms["lgbm"], seed, threads), 1e-8))
            elif name == "rf_importance":
                _knn_weights(name, np.maximum(bw.rf_importance(Xm_tr, y_tr, owner, F, ms["rf"], seed, threads), 1e-8))
            elif name == "relieff":
                rf = ms["relieff"]
                _knn_weights(name, np.maximum(bw.relieff(N_tr, fp.enc.neutral_dim_feature(), F, y_tr, rf["n_samples"],
                                                         rf["n_neighbors"], seed), 1e-8))
            elif name in ("nca_diag", "power_prior"):
                _knn_weights(name, fp.extra_weights[name], "shape")
            elif name == "rf_gap":
                g = ms["rf_gap"]
                rfg = RFGAP(g["n_estimators"], g["min_samples_leaf"], g["max_features"], seed).fit(Xm_tr, y_tr)
                sub = np.random.default_rng(seed).choice(len(y_tr), size=min(3000, len(y_tr)), replace=False)
                kmax = max(k_grid)
                NBs = rfg.topk(Xm_tr[sub], kmax, train_idx=sub, subjects_q=ft_tr.subjects[sub],
                               subjects_train=ft_tr.subjects)
                best = {}
                for k in k_grid:   # α per k with LOO (OOB) on a training subsample
                    lls = {}
                    for a in ecfg["knn"]["alpha_grid"]:
                        pr = RFGAP.probs_from_topk(NBs, y_tr, C, k, float(a))
                        lls[a] = float(-np.mean(np.log(np.clip(pr[np.arange(len(sub)), y_tr[sub]], 1e-15, 1))))
                    best[k] = float(min(lls, key=lls.get))
                NBt = rfg.topk(Xm_te, kmax)
                for k in k_grid:
                    preds[(name, k)] = RFGAP.probs_from_topk(NBt, y_tr, C, k, best[k])
                bl_info[name] = {"alpha": best}
            elif name == "logreg":
                f, inf = parametric.fit_logreg(Xm_tr, y_tr, ms["logreg_C_grid"], inner, C)
                preds[(name, 0)] = f(Xm_te)
                bl_info[name] = inf
            elif name == "lgbm":
                f, inf = parametric.fit_lgbm_calibrated(Xm_tr, y_tr, ms["lgbm"], inner, C, seed, threads,
                                                        grid=ms.get("lgbm_grid"))
                preds[(name, 0)] = f(Xm_te)
                bl_info[name] = inf
            elif name == "swift" and target in ("t1", "t1_bin"):
                sev = SEVERE[target]
                yb = (y_tr == sev).astype(int)
                if 0 < yb.sum() < len(yb):
                    f, inf = parametric.fit_swift(swift_inputs(ft_tr), yb)
                    pg = f(swift_inputs(ft_te))
                    preds[("swift", 0)] = np.column_stack([1 - pg, pg])  # severe / not severe
                    bl_info[name] = inf
            timings[f"baseline_{name}"] = time.perf_counter() - tb
        except Exception as exc:  # a failed comparator does not bring down the fold; it is logged
            LOG.exception("Comparator %s failed: %s", name, exc)
            bl_info[name] = {"error": repr(exc)}
    timings["baselines"] = time.perf_counter() - t

    # per-patient predictions (NEVER exported)
    rows = []
    for (meth, k), p in preds.items():
        df = {"stay_id": ft_te.ids.astype(str), "subject_id": ft_te.subjects.astype(str), "y": y_te.astype(np.int64),
              "method": [meth] * len(te), "k": np.full(len(te), int(k), dtype=np.int64)}
        for c in range(p.shape[1]):
            df[f"p{c}"] = p[:, c].astype(np.float64)
        rows.append(pl.DataFrame(df))
    pred_df = pl.concat(rows, how="diagonal_relaxed")
    ood_df = pl.DataFrame({"stay_id": ft_te.ids.astype(str), "kth_dist": kth, "kth_pct": pct, "ood": ood})
    p4_rows = []
    for meth, summ in p4.items():
        for var, pred in summ.items():
            p4_rows.append(pl.DataFrame({"stay_id": ft_te.ids.astype(str), "method": meth, "variable": var,
                                         "pred": pred.astype(float), "obs": car_te[var].astype(float)}))
    p4_df = pl.concat(p4_rows) if p4_rows else pl.DataFrame()
    # §5.6 per-patient output (E14): this target's probabilities; neighbours for the clinical-output target only
    nbrs = target == cfg.patient_output_target
    patient = patient_pieces(ft_te.ids, data.t0[te], target, probs[kp], kp,
                             **({"kth_pct": pct, "ood": ood, "I": I, "D": D, "valid": valid, "ref_ids": ft_tr.ids,
                                 "carried": p4["perisim"]} if nbrs else {}))

    loop = fp.loop
    result = {
        "target": target, "n_classes": C, "n_train": int(len(tr)), "n_test": int(len(te)), "thresholds": thr,
        "perisim": {**fp.info, "weights": dict(zip(fp.enc.feature_names, np.round(fp.w, 8).tolist())),
                    "feature_importance": dict(zip(fp.enc.feature_names, np.round(fp.imp_feature, 8).tolist())),
                    "feature_groups": dict(zip(fp.enc.feature_names, fp.enc.feature_groups)),
                    "vdm_tables": fp.enc.vdm_tables(int(ecfg["evaluation"]["small_cell_threshold"])),
                    "shape_tables": fp.enc.shape_tables(int(ecfg["evaluation"]["small_cell_threshold"]))},
        "importance_loop": {k: v for k, v in loop.items() if k not in ("weights_loop", "feature_importance", "shapes")},
        "density": density,
        "ood_rate_test": float(np.mean(ood)) if len(ood) else float("nan"),
        "baselines": bl_info, "timings": timings, "seconds": time.perf_counter() - t_all, "rss_gb": rss_gb(),
    }
    result["importance_loop"]["group_importance"] = dict(zip(loop["group_names"], np.round(loop["group_importance"], 8)
                                                            .tolist()))
    # SAGE group of each metric variable, so that dispersion can be reported per variable across folds (E11)
    result["importance_loop"]["group_of_feature"] = {n: fp.group_names[g] for n, g in
                                                     zip(fp.enc.feature_names, fp.group_of_feature)}
    # fitted metric of the fold, for the outcome-definition sensitivity variants (relabelled case base; pod only)
    model = {"enc": fp.enc, "w": fp.w, "alpha": dict(fp.knn.alpha), "train_ids": ft_tr.ids.astype(str),
             "test_ids": ft_te.ids.astype(str)}
    return result, {"pred": pred_df, "ood": ood_df, "p4": p4_df, "patient": patient, "model": model}


def _group_prevalence(g_tr: np.ndarray, y_tr: np.ndarray, g_te: np.ndarray, C: int, m: float) -> np.ndarray:
    """Class frequencies per surgical group, shrunk towards the overall frequencies (m-estimate)."""
    pri = np.bincount(y_tr, minlength=C) / len(y_tr)
    gtr = np.asarray(g_tr, dtype=object).astype(str)
    tab = {}
    for g in np.unique(gtr):
        sel = gtr == g
        tab[g] = (np.bincount(y_tr[sel], minlength=C) + m * pri) / (sel.sum() + m)
    return np.array([tab.get(str(g), pri) for g in np.asarray(g_te, dtype=object)])
