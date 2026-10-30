"""Synthetic data with known shapes (no real data ever enters the tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_data(n: int = 3000, seed: int = 0, n_classes: int = 2, rows_per_patient: int = 1,
              sites: int = 1) -> pd.DataFrame:
    """`u_shape` has a U-shaped effect, `linear` a straight one, `noise` none; `kind` (nominal) and `flag` (binary)
    move the risk; `lab` is missing for 20 % of the rows, and its missingness itself carries risk."""
    rng = np.random.default_rng(seed)
    n_pat = n // rows_per_patient
    pid = np.repeat(np.arange(n_pat), rows_per_patient)[:n]
    u = rng.normal(0, 1, n)
    lin = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    kind = rng.choice(["a", "b", "c", "d"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    flag = (rng.uniform(size=n) < 0.3).astype(float)
    lab = rng.normal(0, 1, n)
    miss = rng.uniform(size=n) < 0.2
    site = rng.integers(0, sites, n)
    eta = (-1.0 + 1.2 * u ** 2 - 0.6 + 1.0 * lin + 0.8 * (kind == "c") + 1.2 * (kind == "d") + 0.7 * flag
           + 0.6 * np.where(miss, 1.0, 0.0) + 0.3 * site)
    lab = np.where(miss, np.nan, lab)
    if n_classes == 2:
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    else:
        cuts = np.quantile(eta + rng.logistic(size=n), np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.searchsorted(cuts, eta + rng.logistic(size=n))
    df = pd.DataFrame({"pid": pid.astype(str), "u_shape": u, "linear": lin, "noise": noise, "kind": kind,
                       "flag": flag, "lab": lab, "y": y, "site": np.array(["s0", "s1", "s2"])[site]})
    return df


SMALL = dict(n_models=6, shapes={"max_train": 1500, "max_val": 800, "n_explain": 600, "subsample": 400,
                                 "n_reference_pairs": 2000},
             nca={"epochs": 4, "n_candidates": 100}, knn={"k": 30, "alpha_queries": 1000}, n_jobs=1)


def small_space():
    from ceknn import ModelSpace

    return ModelSpace.default(include=["logreg", "spline_logreg", "random_forest"])
