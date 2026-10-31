"""T2 in `roll`: secondary and descriptive only (design §12, approved by the lead on 2026-10-30).

The MIMIC-IV temporal stream (blocks in calendar order, random inside each, R orders) with T2's frozen method:
- the metric of T2's external stage, learned on INSPIRE (`final_model.pkl`): MIMIC-IV is external to it;
- MIMIC-IV's T2 labels as T2's external stage made them: cut-points cross-fitted between two patient halves, death
  the fourth class (`perisim.transport.core.target_labels`);
- T2's kNN (`perisim.knn.predict.KNNModel`): k_primary, the patient's own stays excluded, the smoothed vote; α chosen
  on the starting window by T2's LOO (its grid) and then fixed, as in the `hypo` arms.
Arms: PERISIM-FIFO, cumulative, static, and the window's class shares. The unit is the patient (all their stays).
Per window: the 4-class log-loss and the ranked probability score. No ML arms, no hypotheses: paired differences
with the same moving-block bootstrap as the primary streams, reported as estimates.
"""

from __future__ import annotations

import pickle
import time
from collections import deque

import numpy as np
import polars as pl

from . import endpoints as EP
from .config import base_workdir
from .streams import _perm, _rng, segments_of, steps_for, windows_for

C = 4
ARMS = ("perisim_fifo", "perisim_cumulative", "perisim_static", "class_shares")


def load() -> dict:
    """T2's frozen metric, MIMIC-IV's stays with their T2 labels, embeddings and blocks (patient level, local)."""
    from perisim.config import load_config
    from perisim.experiment import load_cohort_data
    from perisim.pipeline import Pipeline
    from perisim.transport.core import target_labels

    pipe = Pipeline(load_config(None, "local", {"experiment": {"paths": {"workdir": str(base_workdir())}}}))
    cfg, ws = pipe.cfg, pipe.ws
    with open(ws.predictions("external", "t2") / "final_model.pkl", "rb") as fh:
        fm = pickle.load(fh)
    X = load_cohort_data(ws.common("mimic"), ws.derived("mimic"), pipe.feature_names())
    y = target_labels("t2", "t2", X, cfg)["own"]
    ok = np.where(y >= 0)[0]
    enc, w = fm["enc"], np.asarray(fm["w"], float)
    Ew = np.ascontiguousarray(enc.transform(X.ft.take(ok)) * enc.weights_dims(w), dtype=np.float32)
    st = pl.read_parquet(ws.common("mimic") / "stays_index.parquet", columns=["stay_id", "year_group"])
    ids = pl.DataFrame({"stay_id": X.ft.ids[ok].astype(str)}).join(st, on="stay_id", how="left")
    kc = cfg.experiment["knn"]
    return {"Ew": Ew, "y": y[ok].astype(np.int64), "subject": X.ft.subjects[ok].astype(str),
            "block": ids["year_group"].to_numpy().astype(str), "k": int(kc["k_primary"]),
            "alpha_grid": [float(a) for a in kc["alpha_grid"]], "ood_percentile": float(kc["ood_percentile"]),
            "n_stays": int(ok.size), "n_unlabelled": int(len(y) - ok.size)}


def stream(d: dict, seed: int, order: int, N: int) -> dict | None:
    subj = d["subject"]
    uniq, inv = np.unique(subj, return_inverse=True)
    ublock = pl.DataFrame({"u": inv, "b": d["block"]}).group_by("u").agg(pl.col("b").min()).sort("u")["b"].to_numpy()
    rng = _rng(seed, "t2", "", order, N)
    labs = sorted(set(ublock))
    parts = [_perm(rng, np.where(ublock == b)[0]) for b in labs]
    if N >= parts[0].size:
        return None
    start, parts[0] = parts[0][:N], parts[0][N:]
    order_ = np.argsort(inv, kind="stable")
    cuts = np.searchsorted(inv[order_], np.arange(uniq.size + 1))
    rows = [order_[cuts[i]:cuts[i + 1]] for i in range(uniq.size)]
    seg = np.concatenate([np.full(p.size, lab, dtype=object) for lab, p in zip(labs, parts)])
    return {"start": start, "seq": np.concatenate(parts), "seg": seg, "rows": rows}


def run(d: dict, cfg: dict, order: int) -> tuple[pl.DataFrame, dict]:
    from perisim.knn.predict import KNNModel

    t0 = time.perf_counter()
    tc = cfg["t2"]
    N, W, B = int(tc["N"]), int(tc["W"]), int(cfg["loop"]["step"])
    s = stream(d, int(cfg["seed"]), order, N)
    segs = segments_of(s["seg"])
    windows = windows_for(segs, {lab: W for lab, _, _ in segs})
    Ew, y, subj, k = d["Ew"], d["y"], d["subject"], d["k"]
    ones = np.ones(Ew.shape[1])
    R = s["rows"]

    def cat(units):
        return np.concatenate([R[u] for u in units]) if len(units) else np.zeros(0, dtype=np.int64)

    def model(rows):
        return KNNModel(Ew[rows], ones, y[rows], C, subj[rows], [k])

    start = cat(s["start"])
    m0 = model(start)
    fit = m0.fit_alpha_and_ood(d["alpha_grid"], k, d["ood_percentile"])[k]
    alpha = fit["alpha"]
    units = {"perisim_fifo": deque(s["start"]), "perisim_cumulative": list(s["start"])}
    P = {a: np.full((len(y), C), np.nan) for a in ARMS}
    pos_of = np.full(len(y), -1)
    win_of = np.full(len(y), -1)
    for a0, a1, wi in steps_for(windows, B):
        q = cat(s["seq"][a0:a1])
        for a in ("perisim_fifo", "perisim_cumulative", "perisim_static"):
            m = m0 if a == "perisim_static" else model(cat(units[a]))
            m.alpha[k] = alpha
            D, I, valid = m.neighbors(Ew[q], subj[q])
            P[a][q] = m.probs(I, valid, k)
        fr = cat(units["perisim_fifo"])
        P["class_shares"][q] = np.bincount(y[fr], minlength=C) / max(fr.size, 1)
        pos_of[q] = np.repeat(np.arange(a0, a1), [len(R[u]) for u in s["seq"][a0:a1]])
        win_of[q] = wi
        for u in s["seq"][a0:a1]:
            units["perisim_fifo"].append(u)
            units["perisim_cumulative"].append(u)
            if len(units["perisim_fifo"]) > N:
                units["perisim_fifo"].popleft()
    ev = np.where(pos_of >= 0)[0]
    cols = {"row": ev, "pos": pos_of[ev], "window": win_of[ev], "y": y[ev]}
    for a in ARMS:
        for c in range(C):
            cols[f"p_{a}_{c}"] = P[a][ev, c]
    log = {"stream": {"stream": "t2_mimic_temporal", "order": order, "N": N, "evaluated": int(len(s["seq"])),
                      "windows": len(windows), "segments": [{"segment": g, "patients": b - a} for g, a, b in segs]},
           "windows": windows, "alpha": fit, "seconds": time.perf_counter() - t0}
    return pl.DataFrame(cols), log


def rps(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score of an ordinal outcome, per row: Σ_c (F_c − 1[y ≤ c])² / (C − 1)."""
    F = np.cumsum(P, 1)[:, :-1]
    Y = (y[:, None] <= np.arange(C - 1)[None, :]).astype(float)
    return ((F - Y) ** 2).sum(1) / (C - 1)


def endpoints(runs: list[tuple[pl.DataFrame, dict]], cfg: dict, seed: int) -> dict:
    windows = runs[0][1]["windows"]
    K = len(windows)
    seg = np.array([w["seg"] for w in windows], dtype=object)
    vals = {m: np.full((len(runs), K, len(ARMS)), np.nan) for m in ("logloss", "rps")}
    for r, (fr, _) in enumerate(runs):
        y, wid = fr["y"].to_numpy(), fr["window"].to_numpy()
        for j, a in enumerate(ARMS):
            P = np.column_stack([fr[f"p_{a}_{c}"].to_numpy() for c in range(C)])
            ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))
            rp = rps(P, y)
            for k in range(K):
                m = wid == k
                vals["logloss"][r, k, j], vals["rps"][r, k, j] = ll[m].mean(), rp[m].mean()
    ell = max(1, int(np.ceil(int(cfg["t2"]["N"]) / int(cfg["t2"]["W"]))))
    boot = EP.mbb(seg, ell, int(cfg["endpoints"]["boot"]), seed)
    out = {"note": "T2: secondary, descriptive only", "orders": len(runs), "windows": K, "arms": {}, "contrasts": {}}
    for m, V in vals.items():
        mean = np.nanmean(V, axis=(0, 1))
        sd = np.nanmean(np.nanstd(V, axis=1, ddof=1), axis=0)
        out["arms"][m] = {a: {"mean": float(mean[j]), "sd": float(sd[j])} for j, a in enumerate(ARMS)}
        bs = np.array([np.nanmean(V[:, b, :], axis=(0, 1)) for b in boot])
        out["contrasts"][m] = {a: {"estimate": float(mean[0] - mean[j]), "ci": EP.ci95(bs[:, 0] - bs[:, j])}
                               for j, a in enumerate(ARMS) if j > 0}
    return out
