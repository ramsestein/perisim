"""Pure helpers of the exploratory post-hoc missing-value drift analysis (scripts/posthoc_drift_missing.py)."""

import importlib.util
from pathlib import Path

import numpy as np

from perisim.features.build import FeatureSpec
from perisim.features.table import FeatureTable
from perisim.metric.encoding import Encoder

_spec = importlib.util.spec_from_file_location(
    "posthoc_drift_missing", Path(__file__).resolve().parents[1] / "scripts" / "posthoc_drift_missing.py")
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)


def test_only_the_missing_values_are_freed_and_grouped():
    rng = np.random.default_rng(0)
    n = 2000
    x, h, b = rng.uniform(0, 10, n), rng.uniform(0, 48, n), (rng.random(n) < 0.4).astype(float)
    x[rng.random(n) < 0.2] = np.nan
    h[rng.random(n) < 0.5] = np.nan
    b[rng.random(n) < 0.1] = np.nan
    specs = [FeatureSpec("x", "continuous", "x", "icu"), FeatureSpec("hours_since_vent_off", "continuous", "h", "icu"),
             FeatureSpec("b", "binary", "b", "icu"), FeatureSpec("grp", "nominal", "grp", "surgery")]
    ids = np.arange(n).astype(str).astype(object)
    ft = FeatureTable(ids, ids, specs, {"x": x, "hours_since_vent_off": h, "b": b},
                      {"grp": rng.choice(["a", "b"], n).astype(object)})
    enc = Encoder(geometry="shape", max_bins=8, min_bin=50).fit(ft, rng.integers(0, 3, n), 3)
    look = dm.D.frozen_points(enc, np.ones(enc.n_features))
    free = dm.missing_rows(look)
    got = {(lj["name"], lj["labels"][v], dm.missing_group(lj["name"], lj["labels"][v]))
           for lj, f in zip(look, free) for v in np.where(f)[0]}
    names = {lj["name"] for lj in look}
    assert ("x", "missing", "measurement") in got
    assert ("hours_since_vent_off", "missing", "structural") in got
    assert ("b", "unobserved", "source") in got
    assert all(lab in dm.MISSING_LABELS for _, lab, _ in got)            # bins, categories, 0 / 1 never freed
    assert not any(f.any() for lj, f in zip(look, free) if lj["name"] == "grp")
    only = dm.missing_rows(look, ("structural",))
    assert {lj["name"] for lj, f in zip(look, only) if f.any()} <= {n_ for n_ in names if "hours_since" in n_}


def test_fit_rows_moves_only_the_missing_point_and_finds_informative_missingness():
    rng = np.random.default_rng(1)
    n = 1500
    codes = np.column_stack([rng.integers(0, 5, n), rng.integers(0, 300, n)])
    y = np.where(codes[:, 0] == 4, 2, (codes[:, 0] >= 2).astype(int))   # value 4 ("missing") is a class of its own
    y = np.where(rng.random(n) < 0.1, rng.integers(0, 3, n), y)
    # the missing value starts at the point of value 2 (median imputation): its stays mix with value 2's
    v0 = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0], [6.0, 0.0]])
    points = [v0, rng.normal(0, 0.5, (300, 2))]
    free = [np.array([False, False, False, False, True]), np.zeros(300, bool)]
    out = dm.fit_rows(points, free, codes, y, np.arange(n).astype(str), 3, 50, 1.0, rounds=2, iters=20)
    P = out["points"]
    np.testing.assert_array_equal(P[0][:4], v0[:4])                     # observed values untouched
    np.testing.assert_array_equal(P[1], points[1])
    assert np.linalg.norm(P[0][4] - v0[4]) > 0.3                         # the missing point leaves value 2
    assert out["trace"][-1][1] < out["trace"][0][0] - 0.05
    E0, E1 = dm.D.embed(points, codes), dm.D.embed(P, codes)
    l0 = dm.M.row_logloss(dm.D.knn_probs(E0, y, E0, 51, 1.0, 3), y).mean()
    l1 = dm.M.row_logloss(dm.D.knn_probs(E1, y, E1, 51, 1.0, 3), y).mean()
    assert l1 < l0 - 0.05                                                # the real (hard) kNN improves too
    both = dm.with_rows(points, P, free)
    np.testing.assert_array_equal(both[0], P[0])
    assert dm.cosine(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == 1.0
