"""The Rashomon set: competitive models drawn from the model space, and the consensus shapes read from them.

Models are drawn at random from the model space (family and hyperparameters), each trained with feature dropout on
a training split and scored by log-loss on a validation split (both group-level: a patient's rows never straddle
them). A model is accepted while its validation log-loss is ≤ (1 + ε)·best so far; at the end only the accepted
models still within (1 + ε) of the final best (the *competitive* set) enter the consensus. The loop stops after
`n_models` accepted models or `max_attempts_factor`·`n_models` attempts. After every accepted model the
stability Δ of the distances (median relative change between random pairs, uniform weights) is recorded.

With a site column and `site_adjusted`, the site enters the models as nuisance columns (not the metric), so each
g_j is a within-site effect, not a difference of base rates between sites.

Port of `hypo`'s shape loop (a fixed number of accepted models, no SAGE prior).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .encoding import Encoder, build_groups
from .models import ModelSpace, Surrogate, log_loss
from .schema import Table
from .shapes import consensus_shapes, model_curves

LOG = logging.getLogger("ceknn.rashomon")


@dataclass
class ShapeSettings:
    n_models: int = 40                 # accepted models
    max_attempts_factor: int = 3       # attempts ≤ factor × n_models
    epsilon: float = 0.05              # competitive: val log-loss ≤ (1 + ε)·best
    augment_copies: int = 2            # masked copies of the training rows (feature dropout)
    max_train: int = 20000             # rows to train each model
    max_val: int = 10000               # rows to score it
    n_explain: int = 2000              # rows the curves are read on
    subsample: int = 1000              # fresh subsample of them per model
    flat_alpha: float = 0.05           # level of the flatness test
    corr_cluster_threshold: float = 0.8
    site_adjusted: bool = True
    n_reference_pairs: int = 20000     # pairs for the stability trace

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


def _split(y: np.ndarray, groups: np.ndarray, seed: int):
    try:
        return next(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups=groups))
    except ValueError:
        return next(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def site_columns(sites: np.ndarray | None, levels: list | None) -> np.ndarray | None:
    if sites is None or not levels:
        return None
    return (np.asarray(sites)[:, None] == np.array(levels, dtype=object)[None, :]).astype(np.float32)


def fit_rashomon(enc: Encoder, t: Table, y: np.ndarray, space: ModelSpace, s: ShapeSettings, *, seed: int = 0,
                 n_jobs: int = 4, verbose: bool = False) -> RashomonSet:
    rng = np.random.default_rng(seed)
    C = enc.n_classes
    Xm, owner = enc.transform_model(t)
    gof, gnames = build_groups(enc, enc.transform(t), float(s.corr_cluster_threshold), seed=seed)
    col_group = gof[owner]
    G = len(gnames)
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
    eps, n_models = float(s.epsilon), int(s.n_models)
    nR = int(s.n_reference_pairs)
    pairs = rng.integers(0, len(y), size=(nR * 2, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]][:nR]
    w_uni = np.full(enc.n_features, 1.0 / max(enc.n_features, 1))

    def dist(e: Encoder) -> np.ndarray:
        return np.sqrt(np.maximum(pair_sq_diffs(e.transform(t), e.dim_feature, e.n_features, pairs) @ w_uni, 1e-12))

    curves = {j: [] for j in shapeable}
    models, vls, rows, deltas = [], [], [], []
    best, attempts = np.inf, 0
    d_prev = dist(enc) if len(pairs) else None
    t0 = time.perf_counter()
    while len(vls) < n_models and attempts < n_models * int(s.max_attempts_factor):
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
        if accepted:
            sub = rng.choice(exp_idx, size=n_sub, replace=False)
            got = model_curves(sur, Xm[sub], owner, enc, t.take(sub), shapeable)
            for j in shapeable:
                curves[j].append(got[j][0] if j in got else None)
            vls.append(vloss)
            models.append(sur)
            if d_prev is not None:
                keep = np.asarray(vls) <= (1 + eps) * best
                shapes, _ = consensus_shapes(_kept(curves, keep), enc, float(s.flat_alpha))
                d_cur = dist(enc.with_shapes(shapes, t))
                row["delta"] = float(np.median(np.abs(d_cur - d_prev) / d_prev))
                deltas.append(row["delta"])
                d_prev = d_cur
            if verbose:
                LOG.info("model %d/%d (%s) val=%.4f", len(vls), n_models, fam, vloss)
        rows.append(row)
    keep = np.asarray(vls) <= (1 + eps) * best
    for r in rows:
        r["competitive"] = bool(r["accepted"] and r["val_logloss"] <= (1 + eps) * best)
    kept = _kept(curves, keep)
    shapes, diag = consensus_shapes(kept, enc, float(s.flat_alpha))
    return RashomonSet(shapes, diag, rows, models, keep, {enc.blocks[j].name: c for j, c in kept.items()},
                       gof, gnames, col_group, G, levels, time.perf_counter() - t0, deltas)


def _kept(curves: dict, keep: np.ndarray) -> dict:
    return {j: [c for c, k in zip(cl, keep) if k and c is not None] for j, cl in curves.items()}


def _jsonable(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        out[k] = v.item() if isinstance(v, np.generic) else v
    return out
