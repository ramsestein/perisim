"""§6.1 — Patient-level split: no patient in two folds (outer or inner)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from perisim.experiment import inner_splits


def _outer(subjects: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    fold = np.full(len(y), -1)
    for f, (_, te) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(y, y, subjects)):
        fold[te] = f
    return fold


def test_outer_and_inner_disjoint_by_subject():
    rng = np.random.default_rng(0)
    subjects = rng.integers(0, 300, 1200).astype(str)          # several hospitalizations per patient
    y = rng.integers(0, 3, 1200)
    fold = _outer(subjects, y, 5, 1)
    per_subject = pl.DataFrame({"s": subjects, "f": fold}).group_by("s").agg(pl.col("f").n_unique())
    assert per_subject["f"].max() == 1
    tr = np.where(fold != 0)[0]
    for a, b in inner_splits(y[tr], subjects[tr], 3, 2):
        assert not set(subjects[tr][a]) & set(subjects[tr][b])


def test_real_folds(real_workdir):
    if real_workdir is None:
        pytest.skip("no real workdir")
    wd, variant = real_workdir
    p = wd / "artifacts" / variant / "folds.parquet"
    if not p.exists():
        pytest.skip("no folds yet")
    f = pl.read_parquet(p)
    assert f.group_by("subject_id").agg(pl.col("outer_fold").n_unique())["outer_fold"].max() == 1
