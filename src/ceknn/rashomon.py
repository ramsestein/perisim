"""The Rashomon set: competitive models drawn from the model space, and the consensus shapes read from them.

Models are drawn at random from the model space (family and hyperparameters), each trained with feature dropout on
a training split and scored by log-loss on a validation split (both group-level: a patient's rows never straddle
them). A model is accepted while its validation log-loss is ≤ (1 + ε)·best so far; at the end only the accepted
models still within (1 + ε) of the final best (the *competitive* set) enter the consensus.

After every accepted model the loop records how stable the metric is:
- Δ: median relative change of the distances between fixed random pairs, on the current consensus shapes, weighted
  by the current SAGE importance (`sage_in_loop`) or uniformly; Δ̃ is its moving median over `window` models;
- the range of each input's consensus curve (log-odds) and, with `sage_in_loop`, each input's aggregated SAGE
  importance, with the Spearman correlation of both against `window` models before (stability of the ranking).

Two stopping rules (`stopping`):
- ``"fixed"``: exactly `n_models` accepted models (`hypo`);
- ``"stability"``: stop once Δ̃ has not improved by more than `min_delta` for `patience` accepted models, after at
  least `n_min` and at most `n_max` (T2, SPECS §5.3.2). `min_delta=None` takes `min_delta_fraction` of Δ̃ at
  `n_min`. The final Δ̃ is reported against `tol_delta`.
In both, at most `max_attempts_factor` × the target number of models are tried.

With a site column and `site_adjusted`, the site enters the models as nuisance columns (not the metric), so each
g_j is a within-site effect, not a difference of base rates between sites.

Port of `hypo`'s shape loop (stopping "fixed", no SAGE) and of PERISIM's importance loop (stopping "stability",
SAGE in the loop), without the SAGE prior on the weights.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .encoding import Encoder, build_groups
from .models import ModelSpace, Surrogate, log_loss
from .sage import InputsOnly, aggregate, normalize, per_variable, sage_permutation
from .schema import Table
from .shapes import consensus_shapes, model_curves

LOG = logging.getLogger("ceknn.rashomon")
STOPPING = ("fixed", "stability")


@dataclass
class ShapeSettings:
    stopping: str = "fixed"            # "fixed" (n_models accepted) or "stability" (T2's rule on Δ̃)
    n_models: int = 40                 # accepted models with stopping "fixed"
    n_min: int = 20                    # stopping "stability": at least …
    n_max: int = 150                   # … and at most this many accepted models
    window: int = 5                    # W: moving median of Δ, and the lag of the ranking stability
    patience: int = 10                 # P: accepted models without Δ̃ improving by > min_delta
    min_delta: float | None = None     # None: min_delta_fraction × Δ̃ at n_min
    min_delta_fraction: float = 0.05
    tol_delta: float = 0.01            # a final Δ̃ above it is flagged
    max_attempts_factor: int = 3       # attempts ≤ factor × target
    epsilon: float = 0.05              # competitive: val log-loss ≤ (1 + ε)·best
    augment_copies: int = 2            # masked copies of the training rows (feature dropout)
    max_train: int = 20000             # rows to train each model
    max_val: int = 10000               # rows to score it
    n_explain: int = 2000              # rows the curves (and SAGE) are read on
    subsample: int = 1000              # fresh subsample of them per model, for the curves
    flat_alpha: float = 0.05           # level of the flatness test
    corr_cluster_threshold: float = 0.8
    site_adjusted: bool = True
    n_reference_pairs: int = 20000     # pairs for Δ
    sage_in_loop: bool = False         # SAGE of every accepted model (slower): importance and its stability
    sage: dict = field(default_factory=lambda: {"batch_size": 512, "max_passes": 3, "threshold": 0.025})

    def __post_init__(self):
        if self.stopping not in STOPPING:
            raise ValueError(f"stopping must be one of {STOPPING}, not {self.stopping!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def pair_sq_diffs(E: np.ndarray, dim_feature: np.ndarray, n_features: int, pairs: np.ndarray) -> np.ndarray:
    """δ_j(a, b)² per pair and variable (R × F)."""
    diff2 = (E[pairs[:, 0]].astype(np.float64) - E[pairs[:, 1]].astype(np.float64)) ** 2
    out = np.zeros((pairs.shape[0], n_features))
    np.add.at(out.T, dim_feature, diff2.T)
    return out


@dataclass
class RashomonSet:
    shapes: dict              # block → consensus g (None = neutral)
    diagnostics: dict         # input name → consensus diagnostics
    attempts: list            # one row per model tried
    models: list              # accepted Surrogates (in memory only)
    competitive: np.ndarray   # which accepted models are within (1 + ε) of the final best
    curves: dict              # input name → competitive models' curves
    groups: np.ndarray        # group of each metric variable
    group_names: list
    col_group: np.ndarray     # group of each model column (site columns: the last group)
    n_groups: int
    site_levels: list | None  # sites with a nuisance column (the first site has none)
    seconds: float
    delta: list
    stability: list = field(default_factory=list)   # one row per accepted model
    stop: dict = field(default_factory=dict)        # stopping rule, reason, Δ̃ at the end, min_delta used
    sage: dict | None = None                        # variable → SAGE aggregated over the competitive models


def _split(y: np.ndarray, groups: np.ndarray, seed: int):
    try:
        return next(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups=groups))
    except ValueError:
        return next(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def site_columns(sites: np.ndarray | None, levels: list | None) -> np.ndarray | None:
    if sites is None or not levels:
        return None
    return (np.asarray(sites)[:, None] == np.array(levels, dtype=object)[None, :]).astype(np.float32)


def curve_range(g: np.ndarray | None, n_classes: int) -> float:
    """Range of a consensus curve: in log-odds for two classes, the largest per-class range otherwise."""
    if g is None:
        return 0.0
    return float(np.ptp(g[:, 1] - g[:, 0])) if n_classes == 2 else float(np.ptp(g, 0).max())


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a is None or b is None or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return None
    return float(spearmanr(a, b).correlation)


def fit_rashomon(enc: Encoder, t: Table, y: np.ndarray, space: ModelSpace, s: ShapeSettings, *, seed: int = 0,
                 n_jobs: int = 4, verbose: bool = False) -> RashomonSet:
    rng = np.random.default_rng(seed)
    C = enc.n_classes
    Xm, owner = enc.transform_model(t)
    gof, gnames = build_groups(enc, enc.transform(t), float(s.corr_cluster_threshold), seed=seed)
    col_group = gof[owner]
    G = G_in = len(gnames)
    levels = None
    if s.site_adjusted and t.sites is not None and np.unique(t.sites).size > 1:
        levels = [str(x) for x in np.unique(t.sites)[1:]]
        S = site_columns(t.sites, levels)
        Xm = np.hstack([Xm, S])
        owner = np.concatenate([owner, np.full(S.shape[1], -1)])
        col_group = np.concatenate([col_group, np.full(S.shape[1], G)])
        G += 1
    spline_cols = np.array([o >= 0 and enc.blocks[o].kind in ("continuous", "ordinal") for o in owner])
    a, b = _split(y, t.groups, seed)
    a = rng.choice(a, size=min(int(s.max_train), a.size), replace=False)
    b = rng.choice(b, size=min(int(s.max_val), b.size), replace=False)
    exp_idx = rng.choice(b, size=min(int(s.n_explain), b.size), replace=False)
    n_sub = min(int(s.subsample), exp_idx.size)
    shapeable = enc.shapeable_blocks()
    eps = float(s.epsilon)
    stability_rule = s.stopping == "stability"
    target = int(s.n_max) if stability_rule else int(s.n_models)
    W, P, n_min = int(s.window), int(s.patience), int(s.n_min)
    min_delta = None if s.min_delta is None else float(s.min_delta)
    nR = int(s.n_reference_pairs)
    pairs = rng.integers(0, len(y), size=(nR * 2, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]][:nR]
    F = enc.n_features
    w_uni = np.full(F, 1.0 / max(F, 1))
    names = enc.feature_names

    def dist(e: Encoder, w: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(pair_sq_diffs(e.transform(t), e.dim_feature, e.n_features, pairs) @ w, 1e-12))

    curves = {j: [] for j in shapeable}
    models, vls, rows, deltas, smooth, stab = [], [], [], [], [], []
    imps, ses = [], []
    best, attempts = np.inf, 0
    best_smooth, since_improve = np.inf, 0
    d_prev = dist(enc, w_uni) if len(pairs) else None
    stop_reason = "n_models" if not stability_rule else "n_max"
    hist_range, hist_sage = [], []
    t0 = time.perf_counter()
    while len(vls) < target and attempts < target * int(s.max_attempts_factor):
        attempts += 1
        fam, params = space.sample(rng)
        mseed = int(rng.integers(0, 2 ** 31 - 1))
        sur = Surrogate(fam, params, col_group, G, C, seed=mseed, n_jobs=n_jobs, spline_cols=spline_cols, space=space)
        sur.fit(Xm[a], y[a], augment_copies=int(s.augment_copies))
        vloss = log_loss(sur.predict_proba(Xm[b]), y[b])
        accepted = vloss <= (1 + eps) * best if np.isfinite(best) else True
        best = min(best, vloss)
        row = {"attempt": attempts, "family": fam, "params": _jsonable(params), "seed": mseed,
               "val_logloss": float(vloss), "accepted": bool(accepted), "fit_seconds": float(sur.fit_seconds)}
        if not accepted:
            rows.append(row)
            continue
        if s.sage_in_loop:
            r = sage_permutation(InputsOnly(sur, G_in, G), Xm[exp_idx], y[exp_idx], G_in, rng,
                                 batch_size=int(s.sage["batch_size"]), max_passes=int(s.sage["max_passes"]),
                                 threshold=float(s.sage["threshold"]))
            im, se = normalize(r["values"], r["se"])
            imps.append(im)
            ses.append(se)
        sub = rng.choice(exp_idx, size=n_sub, replace=False)
        got = model_curves(sur, Xm[sub], owner, enc, t.take(sub), shapeable)
        for j in shapeable:
            curves[j].append(got[j][0] if j in got else None)
        vls.append(vloss)
        models.append(sur)
        keep = np.asarray(vls) <= (1 + eps) * best
        shapes, _ = consensus_shapes(_kept(curves, keep), enc, float(s.flat_alpha))
        rng_now = np.array([curve_range(shapes.get(j), C) for j in range(F)])
        imp_feat = per_variable(aggregate(np.array(imps)[keep], np.array(ses)[keep]), gof) if imps else None
        srow = {"n_accepted": len(vls), "attempt": attempts, "family": fam, "val_logloss": float(vloss),
                "n_competitive": int(keep.sum())}
        if d_prev is not None:
            w_cur = w_uni if imp_feat is None else (imp_feat + 1e-4) / float(np.sum(imp_feat + 1e-4))
            d_cur = dist(enc.with_shapes(shapes, t), w_cur)
            row["delta"] = srow["delta"] = float(np.median(np.abs(d_cur - d_prev) / d_prev))
            deltas.append(row["delta"])
            smooth.append(float(np.median(deltas[-W:])))
            srow["delta_smooth"] = smooth[-1]
            d_prev = d_cur
        hist_range.append(rng_now)
        srow["spearman_range"] = _spearman(hist_range[-1 - W], rng_now) if len(hist_range) > W else None
        if imp_feat is not None:
            hist_sage.append(imp_feat)
            srow["spearman_sage"] = _spearman(hist_sage[-1 - W], imp_feat) if len(hist_sage) > W else None
        srow.update({f"range[{n}]": float(v) for n, v in zip(names, rng_now)})
        if imp_feat is not None:
            srow.update({f"sage[{n}]": float(v) for n, v in zip(names, imp_feat)})
        stab.append(srow)
        rows.append(row)
        if verbose:
            LOG.info("model %d (%s) val=%.4f Δ=%s", len(vls), fam, vloss, srow.get("delta"))
        if stability_rule and smooth:
            if min_delta is None and len(vls) == n_min:
                min_delta = float(s.min_delta_fraction) * smooth[-1]
            md = 0.0 if min_delta is None else min_delta
            if smooth[-1] < best_smooth - md:
                best_smooth, since_improve = smooth[-1], 0
            else:
                since_improve += 1
            if len(vls) >= n_min and since_improve >= P:
                stop_reason = "patience"
                break
    if len(vls) < target and stop_reason != "patience":
        stop_reason = "max_attempts"
    keep = np.asarray(vls) <= (1 + eps) * best
    for r in rows:
        r["competitive"] = bool(r["accepted"] and r["val_logloss"] <= (1 + eps) * best)
    kept = _kept(curves, keep)
    shapes, diag = consensus_shapes(kept, enc, float(s.flat_alpha))
    sage = None
    if imps and keep.any():
        imp_feat = per_variable(aggregate(np.array(imps)[keep], np.array(ses)[keep]), gof)
        sage = {n: float(v) for n, v in zip(names, imp_feat)}
    plateau = smooth[-1] if smooth else None
    stop = {"rule": s.stopping, "reason": stop_reason, "n_accepted": len(vls), "n_attempts": attempts,
            "n_competitive": int(keep.sum()), "delta_smooth_final": plateau, "min_delta_used": min_delta,
            "plateau_above_tol": bool(plateau is not None and plateau > float(s.tol_delta))}
    return RashomonSet(shapes, diag, rows, models, keep, {enc.blocks[j].name: c for j, c in kept.items()},
                       gof, gnames, col_group, G, levels, time.perf_counter() - t0, deltas, stab, stop, sage)


def _kept(curves: dict, keep: np.ndarray) -> dict:
    return {j: [c for c, k in zip(cl, keep) if k and c is not None] for j, cl in curves.items()}


def _jsonable(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        out[k] = v.item() if isinstance(v, np.generic) else v
    return out
