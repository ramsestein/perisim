"""Consensus importance loop with stopping based on the stability of the distance (§5.3.2, amended 2026-10-22/24).

In each iteration: a family + hyperparameters are sampled, the surrogate is trained with feature dropout on
the inner train set, it is accepted if log-loss_val ≤ (1+ε)·best, SAGE is computed on the explanation
subsample of the inner val set and the model's per-variable curves are read on a fresh subsample of it
(importance/shapes.py). Importances are aggregated (random effects) and the shapes by their consensus fit.
Δ_n = median |d^(n) − d^(n−1)| / d^(n−1) is measured over a fixed set of pairs on the metric IMPLIED BY THE PRIOR,
w^(n) = c · (p^(n) + ε₀), over the CURRENT consensus shapes: no NCA runs inside the loop (AUDIT B3), so Δ falls only
when both the importances and the shapes of the variables that matter have stabilised. The global scale c
cancels in Δ. The NCA conversion of §5.4 is applied once, after stopping.
Early-stopping-style stopping on the moving median of Δ (window W, patience P, min_delta), with n_min / n_max.

Models accepted early (when the best log-loss was still poor) leave the consensus as soon as they fall outside
(1+ε)·best (AUDIT B8): importances and shapes always aggregate the currently competitive models only.
The state is checkpointed after every accepted model, so a stopped pod resumes the loop where it was.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from perisim.config import stable_hash
from perisim.importance.sage import aggregate, normalize, sage_permutation
from perisim.importance.shapes import consensus_shapes, model_curves
from perisim.importance.surrogate import Surrogate, log_loss, sample_params
from perisim.metric.encoding import distribute_group_importance
from perisim.metric.nca import pair_sq_diffs

LOG = logging.getLogger("perisim.importance")


def _save(path: Path | None, state: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load(path: Path | None, signature: str) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    try:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
    except Exception as exc:                 # truncated by a kill: start again
        LOG.warning("Unreadable loop checkpoint %s (%s): starting the loop again", path, exc)
        return None
    return state if state.get("signature") == signature else None


def run_importance_loop(*, X_model: np.ndarray, owner: np.ndarray, col_group: np.ndarray, n_groups: int,
                        group_names: list[str], y: np.ndarray, n_classes: int, inner_train: np.ndarray,
                        inner_val: np.ndarray, enc, ft, group_of_feature: np.ndarray, subjects: np.ndarray,
                        cfg, seed: int, threads: int, stopping_override: dict | None = None,
                        exclude_families: set[str] | None = None, early_stop: bool = True,
                        checkpoint: Path | None = None) -> dict:
    icfg = cfg.experiment["importance"]
    st = {**icfg["stopping"], **(stopping_override or {})}
    W, P, n_min, n_max = int(st["W"]), int(st["P"]), int(st["n_min"]), int(st["n_max"])
    min_delta = st.get("min_delta")
    min_delta = None if min_delta in (None, "auto") else float(min_delta)
    eps = float(icfg["epsilon"])
    ccfg = icfg.get("curves") or {}
    rng = np.random.default_rng(seed)
    eps0 = float(cfg.experiment["metric"]["nca"]["eps0"])
    shapeable = enc.shapeable_blocks()
    spline_cols = np.array([enc.blocks[o].kind in ("continuous", "ordinal") for o in owner])
    F = enc.n_features

    # fixed (seeded) reference pairs from the training fold
    n = len(ft)
    nR = int(min(icfg["n_reference_pairs"], n * (n - 1) // 2))
    pairs = rng.integers(0, n, size=(nR * 2, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]][:nR]

    def _dsq(e) -> np.ndarray:
        return pair_sq_diffs(e.transform(ft), e.dim_feature, F, pairs)

    # explanation subsample (inner val)
    n_exp = int(min(icfg["n_explain"], inner_val.size))
    exp_idx = rng.choice(inner_val, size=n_exp, replace=False)
    sage_cfg = icfg["sage"]
    n_sub = int(min(int(ccfg.get("subsample", 1000)), n_exp))
    top_share = float(ccfg.get("top_importance_share", 1.0))
    flat_alpha = float(ccfg.get("flat_alpha", 0.05))

    signature = stable_hash({"seed": seed, "n": n, "F": F, "imp": icfg, "ms": cfg.model_space, "st": st,
                             "excl": sorted(exclude_families or []), "early": early_stop,
                             "groups": group_names, "y": stable_hash(np.asarray(y).tolist())})
    state = _load(checkpoint, signature)
    if state and state.get("done"):
        LOG.info("  importance loop already finished (checkpoint %s)", checkpoint)
        return state["result"]
    if state:
        rng.bit_generator.state = state["rng"]
        LOG.info("  resuming the importance loop from its checkpoint (%d accepted models)", len(state["imps"]))
    else:
        w0 = np.full(F, 1.0 / max(F, 1))                 # uniform metric on the neutral axes before the first model
        state = {"signature": signature, "rows": [], "imps": [], "ses": [], "vloss": [], "deltas": [], "smooth": [],
                 "spear": [],
                 "curves": {j: [] for j in shapeable}, "best_loss": np.inf, "best_smooth": np.inf,
                 "since_improve": 0, "attempts": 0, "seconds": 0.0, "w_prev": w0,
                 "d_prev": np.sqrt(np.maximum(_dsq(enc) @ w0, 1e-12)), "shape_diag": {}}
    rows, imps, ses, vls = state["rows"], state["imps"], state["ses"], state["vloss"]
    deltas, smooth, spear, curves = state["deltas"], state["smooth"], state["spear"], state["curves"]
    stop_reason = "n_max"
    t_start = time.perf_counter() - float(state["seconds"])
    ft_tr = ft.take(inner_train)
    shapes: dict = {}
    while len(imps) < n_max and state["attempts"] < n_max * int(st["max_attempts_factor"]):
        state["attempts"] += 1
        fam, params = sample_params(cfg.model_space, rng, exclude_families)
        mseed = int(rng.integers(0, 2 ** 31 - 1))
        sur = Surrogate(fam, params, col_group, n_groups, n_classes, seed=mseed, threads=threads,
                        spline_cols=spline_cols)
        sur.fit(X_model[inner_train], y[inner_train], augment_copies=int(icfg["augment_copies"]))
        vloss = log_loss(sur.predict_proba(X_model[inner_val]), y[inner_val])
        best_loss = state["best_loss"]
        accepted = vloss <= (1 + eps) * best_loss if np.isfinite(best_loss) else True
        state["best_loss"] = min(best_loss, vloss)
        row = {"attempt": state["attempts"], "family": fam, "params": params, "seed": mseed, "val_logloss": vloss,
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
        vls.append(vloss)
        # the model's curves, on a fresh subsample of the explanation set (gap 7: bootstrap-like)
        t1 = time.perf_counter()
        if shapeable:
            sub = rng.choice(exp_idx, size=n_sub, replace=False)
            got = model_curves(sur, X_model[sub], owner, enc, ft.take(sub), shapeable)
            for j in shapeable:
                curves[j].append(got[j][0] if j in got else None)
        row["curve_seconds"] = time.perf_counter() - t1
        keep = _competitive(vls, state["best_loss"], eps)
        agg = aggregate(np.array(imps)[keep], np.array(ses)[keep], float(icfg.get("variance_floor", 1e-8)))
        imp_feat = distribute_group_importance(agg, group_of_feature, group_names, enc, ft_tr, y[inner_train],
                                               n_classes)
        # w^(n) = c · (p^(n) + ε₀) over the current consensus shapes (B3); c cancels in the relative change Δ
        t2 = time.perf_counter()
        shapes, state["shape_diag"] = consensus_shapes(_kept(curves, keep), enc, imp_feat, top_share, flat_alpha) \
            if shapeable else ({}, {})
        enc_cur = enc.with_shapes(shapes, ft) if shapeable else enc
        w_cur = (imp_feat + eps0) / float(np.sum(imp_feat + eps0))
        d_cur = np.sqrt(np.maximum(_dsq(enc_cur) @ w_cur, 1e-12))
        d_prev = state["d_prev"]
        delta = float(np.median(np.abs(d_cur - d_prev) / d_prev))
        deltas.append(delta)
        sm = float(np.median(deltas[-W:]))
        smooth.append(sm)
        if len(imps) > W:
            kw = keep[:-W]
            prev_agg = aggregate(np.array(imps[:-W])[kw], np.array(ses[:-W])[kw]) if kw.any() else agg
            const = np.ptp(prev_agg) < 1e-12 or np.ptp(agg) < 1e-12
            spear.append(float("nan") if const else float(spearmanr(prev_agg, agg).correlation))
        else:
            spear.append(float("nan"))
        row.update(delta_seconds=time.perf_counter() - t2, delta=delta, delta_smooth=sm)
        rows.append(row)
        state["w_prev"], state["d_prev"] = w_cur, d_cur
        n_acc = len(imps)
        LOG.info("  model %d (%s) val=%.4f Δ=%.4f Δ̃=%.4f", n_acc, fam, vloss, delta, sm)
        # stopping criterion (only with min_delta defined and n ≥ n_min)
        md = 0.0 if min_delta is None else min_delta
        if sm < state["best_smooth"] - md:
            state["best_smooth"] = sm
            state["since_improve"] = 0
        else:
            state["since_improve"] += 1
        state["seconds"] = time.perf_counter() - t_start
        state["rng"] = rng.bit_generator.state
        if early_stop and min_delta is not None and n_acc >= n_min and state["since_improve"] >= P:
            stop_reason = "patience"
            break
        _save(checkpoint, state)
    if state["attempts"] >= n_max * int(st["max_attempts_factor"]) and len(imps) < n_max and stop_reason != "patience":
        stop_reason = "max_attempts"
    keep = _competitive(vls, state["best_loss"], eps) if imps else np.zeros(0, dtype=bool)
    agg = aggregate(np.array(imps)[keep], np.array(ses)[keep], float(icfg.get("variance_floor", 1e-8))) \
        if keep.any() else np.full(n_groups, 1.0 / n_groups)
    imp_feat = distribute_group_importance(agg, group_of_feature, group_names, enc, ft_tr, y[inner_train], n_classes)
    if shapeable:
        shapes, state["shape_diag"] = consensus_shapes(_kept(curves, keep), enc, imp_feat, top_share, flat_alpha)
    plateau = smooth[-1] if smooth else float("nan")
    result = {
        "group_importance": agg, "feature_importance": imp_feat, "weights_loop": state["w_prev"],
        "model_matrix": np.array(imps)[keep] if keep.any() else np.zeros((0, n_groups)),
        "model_se": np.array(ses)[keep] if keep.any() else np.zeros((0, n_groups)),
        "n_competitive_final": int(keep.sum()),
        "models": rows, "delta": deltas, "delta_smooth": smooth, "spearman_W": spear,
        "n_accepted": len(imps), "n_attempts": state["attempts"], "stop_reason": stop_reason,
        "converged": stop_reason == "patience", "plateau": plateau,
        "plateau_above_tol": bool(np.isfinite(plateau) and plateau > float(st["tol_delta"])),
        "seconds": time.perf_counter() - t_start, "group_names": group_names,
        "shapes": shapes, "shape_diag": state["shape_diag"],
    }
    _save(checkpoint, {"signature": signature, "done": True, "result": result})
    return result


def _competitive(vloss: list[float], best: float, eps: float) -> np.ndarray:
    """Accepted models still within (1+ε)·best (AUDIT B8)."""
    v = np.asarray(vloss, dtype=float)
    return v <= (1 + eps) * best if v.size else np.zeros(0, dtype=bool)


def _kept(curves: dict, keep: np.ndarray) -> dict:
    return {j: [c for c, k in zip(cl, keep) if k and c is not None] for j, cl in curves.items()}


def min_delta_from_pilot(delta_smooth: list[float], n_min: int, fraction: float) -> float:
    """Declared rule (experiment.yaml: pilot.min_delta_rule): fraction of Δ̃ at iteration n_min."""
    if not delta_smooth:
        return 0.0
    i = min(max(n_min, 1), len(delta_smooth)) - 1
    return float(fraction * delta_smooth[i])
