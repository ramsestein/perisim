"""Secondary k sweep (decided 2026-10-25): a wider grid reproduces the CV's own grid, the k chosen inside the
training fold, the stored comparator weights, the reproduction check and that the sweep re-keys no stage."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from perisim.knn.predict import KNNModel
from perisim.ksweep import k_grid_for, k_star, reproduction, stored_weights
from perisim.sensitivity import _frame


def _toy(n=600, d=5, C=3, seed=1):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, d))                       # continuous: no tied distances
    y = np.minimum((E[:, 0] > 0).astype(int) + (E[:, 1] > 0.8).astype(int), C - 1)
    return E, y, np.array([f"P{i // 2}" for i in range(n)])


def test_wider_grid_reproduces_the_cv_grid():
    E, y, s = _toy()
    sw = np.ones(E.shape[1])
    alpha_grid = [0, 0.5, 1, 2, 5]
    cv = KNNModel(E[:450], sw, y[:450], 3, s[:450], [10, 20])
    wide = KNNModel(E[:450], sw, y[:450], 3, s[:450], [1, 3, 10, 20, 50, 90])
    loo_cv = cv.fit_alpha_and_ood(alpha_grid, 10, 99)
    loo_w = wide.fit_alpha_and_ood(alpha_grid, 10, 99)
    Dc, Ic, vc = cv.neighbors(E[450:], s[450:])
    Dw, Iw, vw = wide.neighbors(E[450:], s[450:])
    for k in (10, 20):        # same neighbours, same α, same probabilities on the shared k
        assert loo_cv[k]["alpha"] == loo_w[k]["alpha"]
        assert loo_cv[k]["loo_logloss"] == loo_w[k]["loo_logloss"]
        np.testing.assert_array_equal(cv.probs(Ic, vc, k), wide.probs(Iw, vw, k))


def test_k_grid_includes_the_cv_grid_and_is_capped():
    cfg = SimpleNamespace(experiment={"knn": {"k_grid": [10, 20, 50], "k_primary": 50},
                                      "k_sweep": {"k_grid": [1, 5, 300, 2000], "max_fraction": 0.2}})
    assert k_grid_for(cfg, 10_000) == [1, 5, 10, 20, 50, 300, 2000]
    assert k_grid_for(cfg, 1_000) == [1, 5, 10, 20, 50]          # 20 % of 1000 = 200
    assert k_grid_for(cfg, 60) == [1, 5, 10, 20, 50]              # never below the primary k (if the base allows it)
    assert max(k_grid_for(cfg, 40)) == 20                         # k < size of the case base


def test_k_star_is_the_smallest_minimum():
    assert k_star({"10": 0.95, "20": 0.94, "50": 0.94, "100": 0.96}) == 20
    assert k_star({"1": 1.2}) == 1


def test_stored_weights_follow_the_encoder_and_keep_a_floor():
    w = stored_weights({"b": 0.75, "a": 0.25, "c": 0.0}, ["a", "b", "c", "d"])
    np.testing.assert_allclose(w, [0.25, 0.75, 1e-9, 1e-9])


def test_reproduction_reports_differences():
    ids, subj, y = np.array(["1", "2", "3"]), np.array(["a", "b", "c"]), np.array([0, 1, 1])
    P = np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
    cv = pl.concat([_frame(ids, subj, y, "perisim", 10, P), _frame(ids, subj, y, "lgbm", 0, P)])
    same = reproduction(_frame(ids, subj, y, "perisim", 10, P), cv)
    assert len(same) == 1 and same[0]["max_abs_diff"] == 0 and same[0]["d_logloss"] == 0
    Q = P.copy()
    Q[0] = [0.6, 0.4]
    diff = reproduction(_frame(ids, subj, y, "perisim", 10, Q), cv)[0]
    assert abs(diff["max_abs_diff"] - 0.1) < 1e-12 and diff["d_logloss"] > 0


def test_the_sweep_rekeys_no_stage():
    """It runs after a finished CV: neither its configuration nor its code may enter a stage's key."""
    from perisim.pipeline import STAGE_DEFS

    for name, sd in STAGE_DEFS.items():
        assert "experiment" not in sd.config, name
        assert not any(p.startswith("experiment.k_sweep") for p in sd.config), name
        assert "ksweep.py" not in sd.code, name
