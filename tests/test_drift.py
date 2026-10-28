"""Pure helpers of the exploratory post-hoc drift analysis (scripts/posthoc_drift.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from perisim.features.build import FeatureSpec
from perisim.features.table import FeatureTable
from perisim.metric.encoding import Encoder

_spec = importlib.util.spec_from_file_location(
    "posthoc_drift", Path(__file__).resolve().parents[1] / "scripts" / "posthoc_drift.py")
dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dr)


def _ft(n, rng):
    x = rng.uniform(0, 10, n)
    x[rng.random(n) < 0.1] = np.nan
    b = (rng.random(n) < 0.4).astype(float)
    b[rng.random(n) < 0.1] = np.nan
    specs = [FeatureSpec("x", "continuous", "x", "icu"), FeatureSpec("b", "binary", "b", "icu"),
             FeatureSpec("grp", "nominal", "grp", "surgery")]
    ids = np.arange(n).astype(str).astype(object)
    return FeatureTable(ids, ids, specs, {"x": x, "b": b}, {"grp": rng.choice(["a", "b", "c"], n).astype(object)})


def _shaped(rng):
    ft = _ft(3000, rng)
    y = rng.integers(0, 3, 3000)
    enc = Encoder(geometry="shape", max_bins=10, min_bin=50).fit(ft, y, 3)
    shapes = {j: rng.normal(size=(enc._n_values(enc.blocks[j]), 3)) for j in enc.shapeable_blocks()}
    return ft, enc.with_shapes(shapes, ft)


def test_lookup_reproduces_the_frozen_embedding():
    rng = np.random.default_rng(0)
    ft, enc = _shaped(rng)
    w = rng.uniform(0.2, 2.0, enc.n_features)
    look = dr.frozen_points(enc, w)
    assert {lj["kind"] for lj in look} == {"continuous", "binary", "nominal", "missing"}
    q = FeatureTable(np.array(["q1", "q2", "q3"], dtype=object), np.array(["q1", "q2", "q3"], dtype=object), ft.specs,
                     {"x": np.array([-5.0, 1e6, np.nan]), "b": np.array([1.0, np.nan, 0.0])},
                     {"grp": np.array(["never_seen", "a", "b"], dtype=object)})
    for t in (ft, q):
        codes = dr.value_codes(enc, t, look)
        np.testing.assert_allclose(dr.embed([lj["points"] for lj in look], codes),
                                   enc.transform(t) * enc.weights_dims(w), atol=1e-5)
    codes = dr.value_codes(enc, q, look)
    jx, jb, jg = (enc.feature_index()[n] for n in ("x", "b", "grp"))
    assert look[jx]["labels"][codes[2, jx]] == "missing"          # its own value, at the median bin's point
    assert look[jb]["labels"][codes[1, jb]] == "unobserved"
    assert look[jg]["labels"][codes[0, jg]] == "unseen"


def test_effective_weight_is_w_on_the_training_distribution():
    rng = np.random.default_rng(1)
    ft, enc = _shaped(rng)
    w = rng.uniform(0.2, 2.0, enc.n_features)
    look = dr.frozen_points(enc, w)
    codes = dr.value_codes(enc, ft, look)
    for j, lj in enumerate(look):
        pi = np.bincount(codes[:, j], minlength=lj["points"].shape[0]).astype(float)
        assert dr.effective_weight(lj["points"], pi) == pytest.approx(w[j], rel=1e-5)


def test_smoothed_knn_is_close_to_the_hard_knn():
    rng = np.random.default_rng(2)
    E = rng.normal(size=(800, 3)).astype(np.float32)
    y = (E[:, 0] > 0).astype(int) + (E[:, 1] > 1)
    D, I, valid = dr._candidates(E, np.arange(800).astype(str))
    prior = np.bincount(y, minlength=3) / 800
    P = dr.smoothed_probs(torch.as_tensor(D.astype(np.float64)), torch.as_tensor(valid), torch.as_tensor(y[I]), 3, 50,
                          1.0, torch.as_tensor(prior)).numpy()
    n = np.stack([(y[I[:, :50]] == c).sum(1) for c in range(3)], 1)
    H = (n + prior * 3) / (50 + 3)                                 # the hard kNN on the same (sorted) candidates
    assert np.allclose(P.sum(1), 1.0)
    assert np.abs(P - H).mean() < 0.02


def test_shape_change_ignores_stretch_and_rotation_and_split_half_agreement():
    rng = np.random.default_rng(3)
    P0, pi = rng.normal(size=(12, 4)), rng.uniform(1, 5, 12)
    R = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    assert dr.shape_change(P0, 2.5 * P0 @ R + 1.0, pi) < 1e-9
    assert dr.shape_change(P0, rng.normal(size=(12, 4)), pi) > 0.3
    d = rng.normal(size=(12, 4))
    n = np.full(12, 50)
    n[0] = 3                                                       # a value below the small-cell threshold: left out
    assert dr.split_half_r(P0, P0 + d, P0 + d, n, n) == pytest.approx(1.0)
    assert np.isnan(dr.split_half_r(P0, P0, P0, n, n))              # no deformation: undefined, not 'reliable'


def test_refit_of_one_variable_finds_a_relationship_the_frozen_metric_misses():
    rng = np.random.default_rng(4)
    n = 1500
    codes = np.column_stack([rng.integers(0, 6, n), rng.integers(0, 300, n)])
    y = (codes[:, 0] >= 3).astype(int)
    y = np.where(rng.random(n) < 0.1, 1 - y, y)                   # values 0–2 → class 0, 3–5 → class 1 (10 % noise)
    # variable 0 nearly ignored by the frozen metric; variable 1 is noise with many values, so it picks the neighbours
    points = [rng.normal(0, 0.05, (6, 2)), rng.normal(0, 1.0, (300, 2))]
    out = dr.fit_variable(points, 0, codes, y, np.arange(n).astype(str), 2, 50, 1.0, rounds=2, iters=20)
    P = out["points"]
    pi = np.bincount(codes[:, 0], minlength=6).astype(float)
    assert dr.effective_weight(P, pi) > 5 * dr.effective_weight(points[0], pi)
    assert out["trace"][-1][1] < out["trace"][0][0] - 0.05
    d2 = ((P[:, None] - P[None]) ** 2).sum(-1)
    same = np.add.outer(np.arange(6) >= 3, np.arange(6) >= 3) != 1
    off = ~np.eye(6, dtype=bool)
    assert d2[same & off].mean() < 0.5 * d2[~same].mean()          # the two groups of values pulled apart
    E0 = dr.embed(points, codes)
    E1 = dr.embed([P, points[1]], codes)
    l0 = dr.M.row_logloss(dr.knn_probs(E0, y, E0, 51, 1.0, 2), y).mean()
    l1 = dr.M.row_logloss(dr.knn_probs(E1, y, E1, 51, 1.0, 2), y).mean()
    assert l1 < l0 - 0.1                                           # the real (hard) kNN improves too


def test_ice_is_blind_where_the_frozen_points_coincide():
    rng = np.random.default_rng(5)
    n = 600
    codes = np.column_stack([rng.integers(0, 4, n), rng.integers(0, 5, n)])
    y = (codes[:, 0] >= 2).astype(int)
    flat = [np.zeros((4, 2)), rng.normal(0, 1.0, (5, 2))]
    look0 = {"cols": np.array([0, 1])}
    r = dr.ice_curves(flat, 0, look0, codes, y, codes[:100], 50, 1.0, 2, np.full(4, 1.0))
    assert np.abs(r["pdp"]).max() < 1e-9                          # the relationship is there, the ICE cannot see it
    shaped = [np.array([[0, 0], [0, 0], [3, 0], [3, 0]], float), flat[1]]
    r = dr.ice_curves(shaped, 0, look0, codes, y, codes[:100], 50, 1.0, 2, np.full(4, 1.0))
    assert r["pdp"][3, 1] - r["pdp"][0, 1] > 0.5
