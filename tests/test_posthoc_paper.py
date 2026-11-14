"""Statistics of scripts/posthoc_paper.py on simulated data."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("posthoc_paper", Path(__file__).resolve().parents[1] / "scripts" / "posthoc_paper.py")
PP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PP)


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(0)
    n = 6000
    subj = np.repeat(np.arange(n // 3), 3)
    z = rng.normal(size=n)
    p_true = 1 / (1 + np.exp(-(-1.2 + 1.3 * z)))
    y = (rng.random(n) < p_true).astype(float)
    return rng, n, subj, z, p_true, y


def test_auprc_matches_sklearn(sim):
    from sklearn.metrics import average_precision_score
    _, n, _, _, p, y = sim
    assert PP.wauprc(p, y, np.ones(n)) == pytest.approx(average_precision_score(y, p), abs=1e-12)
    from sklearn.metrics import roc_auc_score
    assert PP.wauc(p, y, np.ones(n)) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_calibration_of_true_and_distorted_risk(sim):
    _, n, _, _, p, y = sim
    c = PP.calib(p, y)
    assert abs(c["citl"]) < 0.08 and abs(c["slope"] - 1) < 0.1 and c["ici"] < 0.02
    q = np.clip(p * 1.5, 0, 0.99)
    assert PP.calib(q, y)["ici"] > 3 * c["ici"]


def test_smoother_recovers_a_line():
    p = np.linspace(0.05, 0.95, 3000)
    assert np.abs(PP.smooth_observed(p, p.copy()) - p).max() < 1e-6


def test_net_benefit_and_crossfit(sim):
    rng, n, subj, z, p, y = sim
    assert PP.net_benefit(np.ones(n), y, 0.2) == pytest.approx(y.mean() - (1 - y.mean()) * 0.25)
    noise = np.clip(p + rng.normal(0, 0.15, n), 0.01, 0.99)
    fold = subj % 5
    mix, lam = PP.crossfit_lambda(p, noise, y, fold)
    assert all(v >= 0.8 for v in lam.values())


def test_analyse_runs(sim):
    rng, n, subj, z, p, y = sim
    P = {"transported": np.clip(p + rng.normal(0, 0.02, n), 0.01, 0.99), "local": p,
         "euclidean": np.clip(p + rng.normal(0, 0.08, n), 0.01, 0.99),
         "lgbm_transported": np.clip(p * 1.2, 0.01, 0.99), "lgbm_recal": p}
    out = PP.analyse(P, y, subj, subj % 5, "transported", "lgbm_transported", "lgbm_recal", 1, n_boot=60, n_boot_s=20)
    a = out["abs_equivalence"]
    assert a["G_local"]["est"] > 0 and out["calibration"]["local"]["ici"] < 0.03 and 0 <= a["share_G_local_below_margin"] <= 1
    assert set(out["dca"]) == {"transported", "lgbm_transported", "lgbm_recal", "treat_all"}
    assert "mixture" in out["logloss"] and PP.render({"tag": "t", "units": {"u": out}, "mi_audit": {}})
