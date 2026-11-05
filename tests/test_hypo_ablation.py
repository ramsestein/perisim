"""The helpers of scripts/posthoc_hypo_ablation.py (exploratory follow-ups of `hypo`, second round)."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthoc_hypo_ablation as A  # noqa: E402
from hypo.config import load  # noqa: E402
from hypo.data import concat  # noqa: E402
from hypo.data import load as load_lm  # noqa: E402
from hypo.metric import fit_encoder, fit_metric  # noqa: E402
from hypo.report import auc_prep  # noqa: E402
from perisim.metric.encoding import Encoder  # noqa: E402
from test_hypo_external import _xframe  # noqa: E402
from test_hypo_model import SPACE  # noqa: E402

CFG = load()


def test_weighted_auprc_is_sklearns_average_precision():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.uniform(size=500), 2)                       # ties
    assert A.auprc_weighted(y, np.ones(500), auc_prep(p)) == pytest.approx(average_precision_score(y, p))
    w = rng.integers(0, 3, 500).astype(float)                    # integer weights = repeated rows
    yr, pr = np.repeat(y, w.astype(int)), np.repeat(p, w.astype(int))
    assert A.auprc_weighted(y, w, auc_prep(p)) == pytest.approx(average_precision_score(yr, pr))


def test_net_benefit_by_its_definition():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = rng.uniform(size=400)
    t = np.array([0.1, 0.3, 0.5])
    o = np.argsort(-p, kind="mergesort")
    nb = A.net_benefit(y, np.ones(400), o, p[o], t)
    want = [((p >= x) & (y == 1)).mean() - ((p >= x) & (y == 0)).mean() * x / (1 - x) for x in t]
    assert np.allclose(nb, want)
    assert np.allclose(A.treat_all(y, np.ones(400), t), [y.mean() - (1 - y.mean()) * x / (1 - x) for x in t])


def test_calibration_bins_are_equal_count():
    p = np.random.default_rng(2).uniform(size=1000)
    b = A.calibration_bins(p, 10)
    assert b.min() == 0 and b.max() == 9 and np.bincount(b).min() >= 99


def test_per_variable_splits_a_group_equally():
    w = A.per_variable(np.array([0.6, 0.4]), np.array([0, 0, 1]))
    assert np.allclose(w, [0.3, 0.3, 0.4])


@pytest.fixture(scope="module")
def dev():
    return concat([load_lm("mimic", CFG, _xframe("mimic", 300, 5, 1)), load_lm("eicu", CFG, _xframe("eicu", 300, 5, 2))])


@pytest.fixture(scope="module")
def mcfg():
    m = copy.deepcopy(CFG["metric"])
    m["shapes"].update(n_models=4, max_train=2000, max_val=1000, n_explain=600, subsample=300, n_reference_pairs=2000)
    m["nca"].update(epochs=2, n_candidates=60)
    return m


def test_the_rashomon_refit_draws_the_frozen_models(dev, mcfg):
    metric = fit_metric(dev, mcfg, SPACE, seed=3, threads=1)
    enc0, _ = fit_encoder(dev, mcfg["shapes"])
    rs = A.rashomon_models(enc0, dev, mcfg["shapes"], SPACE, seed=3, threads=1)
    assert [r["family"] for r in rs["rows"]] == [r["family"] for r in metric.info["models"]]
    assert np.allclose([r["val_logloss"] for r in rs["rows"]], [r["val_logloss"] for r in metric.info["models"]])
    assert len(rs["models"]) == metric.info["n_accepted"] and rs["keep"].sum() == metric.info["n_competitive"]


def test_the_zscore_geometry_scales_every_block(dev):
    base = Encoder(geometry="hvdm", m=A.VDM_M).fit(dev.ft, dev.y, 2)
    z = A.ZScore(base, dev.ft)
    E = z.transform(dev.ft).astype(float)
    assert E.shape[1] == z.n_dims
    for j in range(z.n_features):
        cols = E[:, z.dim_feature == j]
        assert 2 * cols.var(0).sum() == pytest.approx(1.0, rel=1e-4)
    assert z.weights_dims(np.full(z.n_features, 4.0)).max() == pytest.approx(2.0)


def test_permutation_importance_finds_the_informative_column():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(1500, 3)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=1500) > 0).astype(int)
    imp = A.permutation_importance(X, y, np.array([0, 1, 2]), 3, np.arange(1500).astype(str),
                                   {"n_estimators": 50, "num_leaves": 8}, seed=0, threads=1, repeats=2)
    assert imp.argmax() == 0 and imp.sum() == pytest.approx(1.0)
