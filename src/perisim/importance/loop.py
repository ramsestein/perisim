"""Consensus importance loop with stopping based on the stability of the learned distance (§5.3.2).

In each iteration: a family + hyperparameters are sampled, the surrogate is trained with feature dropout on
the inner train set, it is accepted if log-loss_val ≤ (1+ε)·best, SAGE is computed on the explanation
subsample of the inner val set, results are aggregated (inverse variance), the weights w^(n) are recomputed
starting from w^(n−1), and Δ_n = median |d^(n) − d^(n−1)| / d^(n−1) is measured over a fixed set of pairs.
Early-stopping-style stopping on the moving median of Δ (window W, patience P, min_delta), with n_min / n_max.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.stats import spearmanr

from perisim.importance.sage import aggregate, normalize, sage_permutation
from perisim.importance.surrogate import Surrogate, log_loss, sample_params
from perisim.metric.encoding import distribute_group_importance
from perisim.metric.nca import DiagonalNCA, pair_sq_diffs

LOG = logging.getLogger("perisim.importance")


def run_importance_loop(*, X_model: np.ndarray, col_group: np.ndarray, n_groups: int, group_names: list[str],
                        y: np.ndarray, n_classes: int, inner_train: np.ndarray, inner_val: np.ndarray,
                        E: np.ndarray, enc, ft, group_of_feature: np.ndarray, subjects: np.ndarray,
                        cfg, seed: int, threads: int, stopping_override: dict | None = None,
                        exclude_families: set[str] | None = None, early_stop: bool = True) -> dict:
    icfg = cfg.experiment["importance"]
    st = {**icfg["stopping"], **(stopping_override or {})}
    W, P, n_min, n_max = int(st["W"]), int(st["P"]), int(st["n_min"]), int(st["n_max"])
    min_delta = st.get("min_delta")
    min_delta = None if min_delta in (None, "auto") else float(min_delta)
    eps = float(icfg["epsilon"])
    rng = np.random.default_rng(seed)
    mcfg = cfg.experiment["metric"]["nca"]

    # fixed (seeded) reference pairs from the training fold
    nR = int(min(icfg["n_reference_pairs"], E.shape[0] * (E.shape[0] - 1) // 2))
    pairs = rng.integers(0, E.shape[0], size=(nR * 2, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]][:nR]
    Dsq = pair_sq_diffs(E, enc.dim_feature, enc.n_features, pairs)

    # explanation subsample (inner val)
    n_exp = int(min(icfg["n_explain"], inner_val.size))
    exp_idx = rng.choice(inner_val, size=n_exp, replace=False)
    sage_cfg = icfg["sage"]

    nca = DiagonalNCA(enc.dim_feature, enc.n_features, n_classes, n_candidates=cfg.experiment["metric"]["n_candidates"],
                      batch_size=mcfg["batch_size"], lr=mcfg["lr"], refresh_every=10 ** 9, eps0=mcfg["eps0"],
                      label_smoothing=mcfg["label_smoothing"], seed=seed)
    w_prev = nca.weights.copy()
    d_prev = np.sqrt(np.maximum(Dsq @ w_prev, 1e-12))

    rows: list[dict] = []
    imps, ses = [], []
    deltas, smooth, spear = [], [], []
    best_loss = np.inf
    best_smooth = np.inf
    since_improve = 0
    attempts = 0
    stop_reason = "n_max"
    t_start = time.perf_counter()
    while len(imps) < n_max and attempts < n_max * int(st["max_attempts_factor"]):
        attempts += 1
        fam, params = sample_params(cfg.model_space, rng, exclude_families)
        mseed = int(rng.integers(0, 2 ** 31 - 1))
        sur = Surrogate(fam, params, col_group, n_groups, n_classes, seed=mseed, threads=threads)
        sur.fit(X_model[inner_train], y[inner_train], augment_copies=int(icfg["augment_copies"]))
        vloss = log_loss(sur.predict_proba(X_model[inner_val]), y[inner_val])
        accepted = vloss <= (1 + eps) * best_loss if np.isfinite(best_loss) else True
        best_loss = min(best_loss, vloss)
        row = {"attempt": attempts, "family": fam, "params": params, "seed": mseed, "val_logloss": vloss,
               "accepted": bool(accepted), "fit_seconds": sur.fit_seconds}
        if not accepted:
            rows.append(row)
            continue
        t0 = time.perf_counter()
        res = sage_permutation(sur, X_model[exp_idx], y[exp_idx], n_groups, rng, batch_size=sage_cfg["batch_size"],
                               max_passes=sage_cfg["max_passes"], threshold=sage_cfg["convergence_threshold"])
        imp_m, se_m = normalize(res["values"], res["se"])
        row.update(sage_seconds=time.perf_counter() - t0, sage_converged=res["converged"], sage_n_eval=res["n_eval"],
                   importance=imp_m.tolist(), se=se_m.tolist())
        imps.append(imp_m)
        ses.append(se_m)
        agg = aggregate(np.array(imps), np.array(ses), float(icfg.get("variance_floor", 1e-8)))
        imp_feat = distribute_group_importance(agg, group_of_feature, group_names, enc, ft.take(inner_train),
                                               y[inner_train], n_classes)
        # w^(n) from w^(n−1) (§5.4 conversion with the loop λ)
        t1 = time.perf_counter()
        if len(imps) == 1 or (len(imps) - 1) % int(icfg["candidate_refresh_every"]) == 0:
            nca.refresh_candidates(E, subjects, w_prev)
        nca.fit(E, y, prior=imp_feat, lam=float(icfg["lambda_loop"]), epochs=int(icfg["nca_epochs_per_iter"]),
                init_logw=np.log(w_prev) if len(imps) > 1 else None, subjects=subjects)
        w_cur = nca.weights.copy()
        d_cur = np.sqrt(np.maximum(Dsq @ w_cur, 1e-12))
        delta = float(np.median(np.abs(d_cur - d_prev) / d_prev))
        deltas.append(delta)
        sm = float(np.median(deltas[-W:]))
        smooth.append(sm)
        if len(imps) > W:
            prev_agg = aggregate(np.array(imps[:-W]), np.array(ses[:-W]))
            const = np.ptp(prev_agg) < 1e-12 or np.ptp(agg) < 1e-12
            spear.append(float("nan") if const else float(spearmanr(prev_agg, agg).correlation))
        else:
            spear.append(float("nan"))
        row.update(nca_seconds=time.perf_counter() - t1, delta=delta, delta_smooth=sm)
        rows.append(row)
        w_prev, d_prev = w_cur, d_cur
        n_acc = len(imps)
        LOG.info("  model %d (%s) val=%.4f Δ=%.4f Δ̃=%.4f", n_acc, fam, vloss, delta, sm)
        # stopping criterion (only with min_delta defined and n ≥ n_min)
        md = 0.0 if min_delta is None else min_delta
        if sm < best_smooth - md:
            best_smooth = sm
            since_improve = 0
        else:
            since_improve += 1
        if early_stop and min_delta is not None and n_acc >= n_min and since_improve >= P:
            stop_reason = "patience"
            break
    if attempts >= n_max * int(st["max_attempts_factor"]) and len(imps) < n_max and stop_reason != "patience":
        stop_reason = "max_attempts"
    agg = aggregate(np.array(imps), np.array(ses), float(icfg.get("variance_floor", 1e-8))) if imps else \
        np.full(n_groups, 1.0 / n_groups)
    imp_feat = distribute_group_importance(agg, group_of_feature, group_names, enc, ft.take(inner_train),
                                           y[inner_train], n_classes)
    plateau = smooth[-1] if smooth else float("nan")
    return {
        "group_importance": agg, "feature_importance": imp_feat, "weights_loop": w_prev,
        "model_matrix": np.array(imps) if imps else np.zeros((0, n_groups)),
        "model_se": np.array(ses) if ses else np.zeros((0, n_groups)),
        "models": rows, "delta": deltas, "delta_smooth": smooth, "spearman_W": spear,
        "n_accepted": len(imps), "n_attempts": attempts, "stop_reason": stop_reason,
        "converged": stop_reason == "patience", "plateau": plateau,
        "plateau_above_tol": bool(np.isfinite(plateau) and plateau > float(st["tol_delta"])),
        "seconds": time.perf_counter() - t_start, "group_names": group_names,
    }


def min_delta_from_pilot(delta_smooth: list[float], n_min: int, fraction: float) -> float:
    """Declared rule (experiment.yaml: pilot.min_delta_rule): fraction of Δ̃ at iteration n_min."""
    if not delta_smooth:
        return 0.0
    i = min(max(n_min, 1), len(delta_smooth)) - 1
    return float(fraction * delta_smooth[i])
