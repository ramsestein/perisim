"""Synthetic cohorts for the tests and the Phase 2 dry run (design §16; never an analysis).

Three cohorts with `hypo`'s input schema and the same roles: "inspire" (no calendar, no infusion inputs),
"mimic" (five calendar blocks), "eicu" (hospitals, 2014 and 2015). The outcome depends on the MAP inputs and a
running vasopressor, with known changes: the base rate differs by cohort, drifts across the MIMIC-IV blocks (and
the MAP effect steepens in the fourth), and shifts in 2015 in half of the eICU hospitals. A quick `hypo` metric and
quick comparators are fitted on them the way the frozen ones were (each rotation on the two other cohorts).
"""

from __future__ import annotations

import copy

import numpy as np
import polars as pl

from hypo.compare import fit as fit_comp
from hypo.data import concat, learning_sample
from hypo.data import load as load_lm
from hypo.metric import fit_metric

from .data import Cohort, Frozen, build

BLOCKS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]
BASE = {"inspire": -1.6, "mimic": -0.6, "eicu": -1.2}
SPACE = {"families": {"logreg": {"weight": 1, "params": {"C": 1.0, "max_iter": 500}},
                      "lightgbm": {"weight": 1, "params": {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 8,
                                                           "min_child_samples": 20, "subsample": 1.0,
                                                           "colsample_bytree": 1.0, "reg_lambda": 1.0}}},
         "baselines": {"lgbm": {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 8, "min_child_samples": 20,
                                "subsample": 1.0, "colsample_bytree": 1.0},
                       "lgbm_grid": {"num_leaves": [8], "learning_rate": [0.1], "max_estimators": 200,
                                     "early_stopping_rounds": 20},
                       "logreg_C_grid": [0.1, 1.0]}}


def quick_mcfg(hcfg: dict) -> dict:
    m = copy.deepcopy(hcfg["metric"])
    m["shapes"].update(n_models=4, max_train=3000, max_val=1500, n_explain=600, subsample=300, n_reference_pairs=2000)
    m["nca"].update(epochs=4, n_candidates=100)
    m.update(alpha_queries=2000)
    return m


def quick_hcfg(hcfg: dict) -> dict:
    h = copy.deepcopy(hcfg)
    h["external"]["drift"].update(anchors=600, joint_rounds=1, joint_iters=5, k_candidates=80)
    return h


def frame(cohort: str, n_pat: int, per: int, hcfg: dict, seed: int, n_hosp: int = 6) -> tuple[pl.DataFrame, dict]:
    """(landmark frame, subject → (block, year, hospital))."""
    rng = np.random.default_rng(seed)
    inp = hcfg["model"]["inputs"]
    n = n_pat * per
    pid = np.repeat(np.arange(n_pat), per)
    block = rng.integers(0, len(BLOCKS), n_pat) if cohort == "mimic" else np.zeros(n_pat, int)
    year = rng.integers(2014, 2016, n_pat) if cohort == "eicu" else np.full(n_pat, -1)
    hosp = rng.integers(0, n_hosp, n_pat) if cohort == "eicu" else np.zeros(n_pat, int)
    cols = {v: rng.normal(0, 1, n) for v in inp["continuous"]}
    level = rng.normal(80, 8, n_pat)[pid]
    cols["map_last"] = level + rng.normal(0, 2, n)
    cols["map_mean_6h"] = level + rng.normal(0, 1.5, n)
    cols["map_min_6h"] = np.minimum(cols["map_last"], cols["map_mean_6h"]) - np.abs(rng.normal(3, 2, n))
    cols["age"] = rng.uniform(20, 90, n_pat)[pid]
    cols.update(hr_last=rng.normal(85, 12, n), rr_last=rng.normal(16, 4, n).clip(4, 60),
                spo2_last=rng.normal(97, 2, n).clip(80, 100), temp_last=rng.normal(37, 0.5, n),
                hb_last=rng.normal(11, 1.5, n).clip(3, 20), creatinine_last=np.exp(rng.normal(0, 0.4, n)),
                lactate_last=np.exp(rng.normal(0.3, 0.4, n)))
    cols["spo2_min_6h"] = cols["spo2_last"] - np.abs(rng.normal(1, 1, n))
    cols["L"] = np.tile(np.arange(2, 2 + 2 * per, 2), n_pat).astype(float)
    cols.update({v: (rng.random(n) < 0.3).astype(float) for v in inp["binary"]})
    b0 = BASE[cohort] + np.array([0.0, 0.0, 0.1, -0.2, -0.6])[block][pid]
    slope = np.where(block[pid] == 3, 1.3, 1.0)
    h_eff = rng.normal(0, 0.3, n_hosp)[hosp][pid] + np.where((hosp[pid] % 2 == 0) & (year[pid] == 2015), 0.4, 0.0)
    z = b0 + h_eff + slope * (-(cols["map_last"] - 75) / 4 - (cols["map_min_6h"] - 70) / 6) + 0.8 * cols["vaso_on"]
    y = rng.random(n) < 1 / (1 + np.exp(-z))
    df = pl.DataFrame({**cols, "surgical_group": rng.choice(["a", "b", "c"], n),
                       "stay_id": [f"{cohort.upper()}-{100000 + i}" for i in pid],
                       "subject_id": [f"{cohort.upper()}-{500000 + i}" for i in pid],
                       "y": y, "band": np.where(y, 0, 2).astype(np.int8), "normotensive": True, "known": True})
    if cohort == "eicu":
        df = df.with_columns(pl.Series("hospital", [f"{700 + h}" for h in hosp[pid]]))
    if cohort == "inspire":
        df = df.with_columns([pl.lit(None, pl.Float64).alias(v) for v in hcfg["model"]["source_missing"]["inspire"]])
    meta = {f"{cohort}:{cohort.upper()}-{500000 + i}": (BLOCKS[block[i]] if cohort == "mimic" else None, int(year[i]),
                                                        f"{700 + hosp[i]}") for i in range(n_pat)}
    return df.with_columns(pl.col("L").cast(pl.Int32)), meta


def cohorts(cfg: dict, hcfg: dict, seed: int = 7) -> dict[str, Cohort]:
    d = cfg["dry"]
    out = {}
    for i, c in enumerate(("inspire", "mimic", "eicu")):
        df, meta = frame(c, int(d["patients"][c]), int(d["landmarks_per_patient"]), hcfg, seed + i,
                         int(d["eicu_hospitals"]))
        lm = load_lm(c, hcfg, df)
        s = lm.ft.subjects.astype(str)
        rb = np.array([meta[x][0] for x in s], dtype=object) if c == "mimic" else None
        ry = np.array([meta[x][1] for x in s], dtype=np.int64) if c == "eicu" else None
        out[c] = build(c, lm, rb, ry, hcfg["bounds"])
    return out


def frozen(cs: dict[str, Cohort], hcfg: dict, families: list[str], seed: int = 11, threads: int = 4) -> Frozen:
    mcfg = quick_mcfg(hcfg)
    metrics, comps, infos = {}, {}, {}
    for x in ("inspire", "mimic", "eicu"):
        pool = concat([cs[c].lm for c in ("inspire", "mimic", "eicu") if c != x])
        rows = learning_sample(pool, int(mcfg["learn"]["per_patient"]), int(mcfg["learn"]["per_site"]), seed)
        metrics[x] = fit_metric(pool.take(rows), mcfg, SPACE, seed=seed, threads=threads)
        for f in families:
            comps[(x, f)], infos[(x, f)] = fit_comp(f, pool, mcfg, SPACE["baselines"], n_inner=2, seed=seed,
                                                    threads=threads)
    return Frozen(metrics, comps, infos)
