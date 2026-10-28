"""The metric of `hypo`: consensus shape functions g_j and diagonal-NCA weights w_j (T2's procedure, no SAGE stage).

What T2 taught (DECISIONS 2026-10-28) sets the changes, all in this package; perisim's pieces are used unchanged:
- no SAGE: the importance prior was not used (λ ≈ 0), so the weights are a free diagonal NCA and the shape loop
  stops at a fixed number of accepted models instead of the stability of the prior-implied distance (the stability
  trace Δ is still recorded, with uniform weights);
- several development sites: the site enters the ensemble's models as a nuisance column (not the metric), so each
  g_j is a within-site effect, not a difference of base rates between sites; the NCA searches each anchor's
  candidates within its own site, as the metric is used (each site queries its own case base);
- an input that is constant within every development site (the missing indicator of a source gap is the site
  itself) carries no within-site information and leaves the metric.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from perisim.importance.shapes import consensus_shapes, model_curves
from perisim.importance.surrogate import Surrogate, log_loss, sample_params
from perisim.knn.index import search_excluding
from perisim.metric.encoding import Encoder, build_groups
from perisim.metric.nca import DiagonalNCA, pair_sq_diffs

from .data import Landmarks

LOG = logging.getLogger("hypo.metric")
C = 2


@dataclass
class Metric:
    enc: Encoder              # with the consensus shapes installed
    w: np.ndarray             # NCA weight of each metric variable
    info: dict = field(default_factory=dict)

    def embed(self, lm: Landmarks) -> np.ndarray:
        return self.enc.transform(lm.ft)

    @property
    def sqrt_w_dims(self) -> np.ndarray:
        return self.enc.weights_dims(self.w)

    def weights_table(self) -> dict[str, float]:
        """Weights rescaled to sum 1 (only the profile matters to the neighbours: k fixed, scale free)."""
        return {n: float(x) for n, x in zip(self.enc.feature_names, self.w / self.w.sum())}


class SiteNCA(DiagonalNCA):
    """Diagonal NCA whose candidates come from the anchor's own site (patient's own landmarks excluded)."""

    def __init__(self, *args, sites: np.ndarray | None = None, **kw):
        super().__init__(*args, **kw)
        self.sites = None if sites is None else np.asarray(sites)

    def refresh_candidates(self, E, subjects, w=None) -> None:
        if self.sites is None:
            return super().refresh_candidates(E, subjects, w)
        w = self.weights if w is None else w
        Ew = E * np.sqrt(w)[self.dim_feature.numpy()].astype(np.float32)
        cand = np.zeros((E.shape[0], self.K), dtype=np.int64)
        valid = np.zeros((E.shape[0], self.K), dtype=bool)
        for s in np.unique(self.sites):
            idx = np.where(self.sites == s)[0]
            sj = None if subjects is None else subjects[idx]
            _, I, v = search_excluding(Ew[idx], Ew[idx], self.K, sj, sj, self_query=True)
            cand[idx, :I.shape[1]] = idx[I]
            valid[idx, :I.shape[1]] = v
        self.cand, self.valid = cand, valid


def _site_constant(enc: Encoder, E: np.ndarray, site: np.ndarray) -> list[int]:
    """Metric variables constant within every site."""
    out = []
    for j in range(enc.n_features):
        cols = E[:, enc.dim_feature == j]
        if all(cols[site == s].var(axis=0).sum() <= 1e-12 for s in np.unique(site)):
            out.append(j)
    return out


def fit_encoder(lm: Landmarks, scfg: dict) -> tuple[Encoder, list[str]]:
    enc = Encoder(geometry="shape", max_bins=int(scfg["max_bins"]), min_bin=int(scfg["min_bin"])).fit(lm.ft, lm.y, C)
    drop = _site_constant(enc, enc.transform(lm.ft), lm.site)
    dropped = [enc.blocks[j].name for j in drop]
    if drop:
        enc.blocks = [b for j, b in enumerate(enc.blocks) if j not in set(drop)]
        enc._index()
    return enc, dropped


def shape_loop(enc: Encoder, lm: Landmarks, scfg: dict, model_space: dict, *, seed: int, threads: int) -> dict:
    """Consensus shapes from `n_models` accepted models of the model space (fixed number, no SAGE)."""
    rng = np.random.default_rng(seed)
    ft, y = lm.ft, lm.y
    Xm, owner = enc.transform_model(ft)
    gof, gnames = build_groups(enc, enc.transform(ft), float(scfg["corr_cluster_threshold"]), 10 ** 9, seed=seed)
    col_group = gof[owner]
    G = len(gnames)
    sites = np.unique(lm.site)
    if scfg.get("site_adjusted", True) and sites.size > 1:      # nuisance columns: one per site but the first
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
    shapeable = enc.shapeable_blocks()
    eps = float(scfg["epsilon"])
    n_models = int(scfg["n_models"])
    nR = int(scfg["n_reference_pairs"])
    pairs = rng.integers(0, len(y), size=(nR * 2, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]][:nR]
    w_uni = np.full(enc.n_features, 1.0 / max(enc.n_features, 1))

    def dist(e: Encoder) -> np.ndarray:
        return np.sqrt(np.maximum(pair_sq_diffs(e.transform(ft), e.dim_feature, e.n_features, pairs) @ w_uni, 1e-12))

    curves = {j: [] for j in shapeable}
    vls, rows, deltas = [], [], []
    best, attempts = np.inf, 0
    d_prev = dist(enc)
    t0 = time.perf_counter()
    while len(vls) < n_models and attempts < n_models * int(scfg["max_attempts_factor"]):
        attempts += 1
        fam, params = sample_params(model_space, rng)
        mseed = int(rng.integers(0, 2 ** 31 - 1))
        sur = Surrogate(fam, params, col_group, G, C, seed=mseed, threads=threads, spline_cols=spline_cols)
        sur.fit(Xm[a], y[a], augment_copies=int(scfg["augment_copies"]))
        vloss = log_loss(sur.predict_proba(Xm[b]), y[b])
        accepted = vloss <= (1 + eps) * best if np.isfinite(best) else True
        best = min(best, vloss)
        row = {"attempt": attempts, "family": fam, "val_logloss": vloss, "accepted": bool(accepted),
               "fit_seconds": sur.fit_seconds}
        if accepted:
            sub = rng.choice(exp_idx, size=n_sub, replace=False)
            got = model_curves(sur, Xm[sub], owner, enc, ft.take(sub), shapeable)
            for j in shapeable:
                curves[j].append(got[j][0] if j in got else None)
            vls.append(vloss)
            keep = np.asarray(vls) <= (1 + eps) * best
            shapes, _ = consensus_shapes(_kept(curves, keep), enc, np.ones(enc.n_features), 1.0,
                                         float(scfg["flat_alpha"]))
            d_cur = dist(enc.with_shapes(shapes, ft))
            row["delta"] = float(np.median(np.abs(d_cur - d_prev) / d_prev))
            deltas.append(row["delta"])
            d_prev = d_cur
            LOG.info("  model %d/%d (%s) val=%.4f Δ=%.4f", len(vls), n_models, fam, vloss, row["delta"])
        rows.append(row)
    keep = np.asarray(vls) <= (1 + eps) * best
    shapes, diag = consensus_shapes(_kept(curves, keep), enc, np.ones(enc.n_features), 1.0, float(scfg["flat_alpha"]))
    return {"shapes": shapes, "shape_diag": diag, "models": rows, "delta": deltas, "n_accepted": len(vls),
            "n_competitive": int(keep.sum()), "n_attempts": attempts, "seconds": time.perf_counter() - t0,
            "groups": {enc.feature_names[j]: gnames[g] for j, g in enumerate(gof)}}


def _kept(curves: dict, keep: np.ndarray) -> dict:
    return {j: [c for c, k in zip(cl, keep) if k and c is not None] for j, cl in curves.items()}


def fit_metric(lm: Landmarks, mcfg: dict, model_space: dict, *, seed: int, threads: int) -> Metric:
    """The metric learned on the learning sample `lm` (rows of one or several development sites)."""
    scfg, ncfg = mcfg["shapes"], mcfg["nca"]
    t = time.perf_counter()
    enc, dropped = fit_encoder(lm, scfg)
    loop = shape_loop(enc, lm, scfg, model_space, seed=seed, threads=threads)
    enc = enc.with_shapes(loop["shapes"], lm.ft)
    E = enc.transform(lm.ft)
    t_nca = time.perf_counter()
    nca = SiteNCA(enc.dim_feature, enc.n_features, C, n_candidates=int(ncfg["n_candidates"]),
                  batch_size=int(ncfg["batch_size"]), lr=float(ncfg["lr"]), refresh_every=int(ncfg["refresh_every"]),
                  eps0=float(ncfg["eps0"]), label_smoothing=float(ncfg["label_smoothing"]), seed=seed, sites=lm.site)
    nca.fit(E, lm.y, epochs=int(ncfg["epochs"]), subjects=lm.ft.subjects)
    info = {"n_learn": len(lm), "learn_per_site": {str(s): int((lm.site == s).sum()) for s in np.unique(lm.site)},
            "dropped_site_constant": dropped, "n_features": enc.n_features, "n_dims": enc.n_dims,
            "n_shaped": int(sum(b.shape is not None for b in enc.blocks)), "nca_history": nca.history,
            "seconds": {"shapes": loop["seconds"], "nca": time.perf_counter() - t_nca,
                        "total": time.perf_counter() - t},
            **{k: loop[k] for k in ("shape_diag", "models", "delta", "n_accepted", "n_competitive", "n_attempts",
                                    "groups")}}
    return Metric(enc, nca.weights, info)
