"""Pure helpers of the exploratory post-hoc transport analysis (scripts/posthoc_transport.py)."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "posthoc_transport", Path(__file__).resolve().parents[1] / "scripts" / "posthoc_transport.py")
ph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ph)


def _draw(P, rng):
    return np.array([rng.choice(P.shape[1], p=p) for p in P])


def test_intercept_recalibration_recovers_a_base_rate_shift():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(3) * 3, size=4000)
    a_true = np.array([0.0, 0.8, -0.6])
    y = _draw(ph.apply_recal(P, a_true, 1.0), rng)
    a, b = ph.fit_recal(P, y, slope=False)
    assert b == 1.0
    assert np.allclose(a, a_true, atol=0.15)


def test_slope_recalibration_detects_overconfidence_and_identity_is_kept():
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(3), size=4000)
    y = _draw(ph.apply_recal(P, np.zeros(3), 0.5), rng)          # true spread is half the predicted one
    _, b = ph.fit_recal(P, y, slope=True)
    assert 0.4 < b < 0.6
    y2 = _draw(P, rng)                                             # already calibrated: (almost) no change
    a2, b2 = ph.fit_recal(P, y2, slope=True)
    assert abs(b2 - 1) < 0.1 and np.all(np.abs(a2) < 0.15)


def test_recalibration_cannot_run_away_with_few_patients():
    P = np.tile([0.5, 0.3, 0.2], (5, 1))
    a, b = ph.fit_recal(P, np.zeros(5, dtype=int), slope=True)    # one class only: the prior keeps it finite
    assert np.all(np.isfinite(a)) and np.isfinite(b) and np.all(np.abs(a) < 10)


def test_tuning_subsets_are_nested_patient_sets_as_in_analysis_D():
    subj = np.repeat(np.arange(30).astype(str), 2)
    sub = ph.tuning_subsets(subj, [5, 10, 100], seed=3)
    assert set(sub) == {5, 10}                                     # sizes above the available patients are dropped
    assert np.unique(subj[sub[5]]).size == 5 and np.unique(subj[sub[10]]).size == 10
    assert np.all(sub[10][sub[5]])                                 # nested
    pats = np.random.default_rng(3 + 11).permutation(np.unique(subj))
    assert set(subj[sub[5]]) == set(pats[:5])                      # the same draw as transport.analyses._recalibration


def test_percentile_shift_ignores_recording_resolution_but_sees_a_shift():
    rng = np.random.default_rng(4)
    ref = np.sort(rng.lognormal(0, 0.4, 20000))
    same = rng.lognormal(0, 0.4, 5000)
    w_same, d_same = ph.percentile_shift(ref, same)
    w_round, _ = ph.percentile_shift(ref, np.round(same, 1))      # the same population, coarser resolution
    w_up, d_up = ph.percentile_shift(ref, same * 1.5)
    assert w_same < 0.02 and abs(d_same) < 0.02
    assert w_round < 0.05
    assert w_up > 0.15 and d_up > 0.15
    assert ph.psi(np.histogram(ref, 20)[0] * 1.0, np.histogram(np.round(same, 1), np.histogram(ref, 20)[1])[0] * 1.0) \
        > ph.psi(np.histogram(ref, 20)[0] * 1.0, np.histogram(same, np.histogram(ref, 20)[1])[0] * 1.0)
    assert np.isclose(ph.total_variation(np.array([50.0, 50]), np.array([20.0, 80])), 0.3)
    gcs = np.sort(np.r_[np.full(900, 15.0), np.full(60, 14.0), np.full(40, 9.0)])   # heavy ties: same population → 0
    assert ph.percentile_shift(gcs, rng.permutation(gcs))[0] < 1e-9


def test_psi_and_small_cell_suppression():
    c = np.array([100.0, 200, 300, 400])
    assert ph.psi(c, c) == 0.0
    assert ph.psi(c, c[::-1]) > 0.25
    s = ph.suppressed_shares(np.array([5.0, 50, 45]))
    assert np.isnan(s[0]) and np.isclose(s[1], 0.5)
