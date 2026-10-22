"""§5.2 / §4.4.4 — The VDM (and everything that uses the target) is computed ONLY on the training fold."""

from __future__ import annotations

import numpy as np
import pytest

from perisim.features.build import FeatureSpec
from perisim.features.table import FeatureTable
from perisim.metric.encoding import Encoder
from perisim.utils import read_json


def _table(n: int, rng) -> FeatureTable:
    specs = [FeatureSpec("grp", "nominal", "grp", "surgery"), FeatureSpec("x", "continuous", "x", "icu")]
    cat = {"grp": rng.choice(["a", "b", "c"], n).astype(object)}
    num = {"x": rng.normal(size=n)}
    return FeatureTable(np.arange(n).astype(str).astype(object), np.arange(n).astype(str).astype(object), specs, num, cat)


def test_vdm_uses_only_training_labels():
    rng = np.random.default_rng(0)
    ft = _table(400, rng)
    y = rng.integers(0, 3, 400)
    tr, te = np.arange(300), np.arange(300, 400)
    e1 = Encoder(m=5).fit(ft.take(tr), y[tr], 3)
    y2 = y.copy()
    y2[te] = rng.integers(0, 3, te.size)                    # change the test labels
    e2 = Encoder(m=5).fit(ft.take(tr), y2[tr], 3)
    np.testing.assert_allclose(e1.transform(ft.take(te)), e2.transform(ft.take(te)))
    tab = next(b for b in e1.blocks if b.kind == "nominal").params["table"]
    assert sum(n for _, n in tab.values()) == tr.size


def test_vdm_smoothing_toward_prior():
    rng = np.random.default_rng(1)
    ft = _table(200, rng)
    y = rng.integers(0, 3, 200)
    big_m = Encoder(m=1e6, nominal_min_count=1).fit(ft, y, 3)
    tab = next(b for b in big_m.blocks if b.kind == "nominal").params["table"]
    prior = np.bincount(y, minlength=3) / len(y)
    for p, _ in tab.values():
        np.testing.assert_allclose(p, prior, atol=1e-3)


def test_real_vdm_counts_within_training(real_workdir):
    if real_workdir is None:
        pytest.skip("no real workdir")
    wd, variant = real_workdir
    folds = list((wd / "artifacts" / variant / "cv").glob("*/fold*.json"))
    if not folds:
        pytest.skip("no folds yet")
    for f in folds:
        r = read_json(f)
        for name, tab in r["perisim"]["vdm_tables"].items():
            assert sum(v["n"] for v in tab.values()) <= r["n_train"], f"{f}: {name}"
