"""Pure helpers of the exploratory post-hoc case-base analyses (scripts/posthoc_case_base.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "posthoc_case_base", Path(__file__).resolve().parents[1] / "scripts" / "posthoc_case_base.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def test_prior_correction_moves_base_rates_and_keeps_rows_normalised():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=500)
    frm, to = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    q = cb.prior_correct(p, to, frm)
    assert np.allclose(q.sum(1), 1.0)
    np.testing.assert_allclose(cb.prior_correct(p, frm, frm), p)          # same base rates: unchanged
    assert (q[:, 2] > p[:, 2]).all() and (q[:, 0] < p[:, 0]).all()         # mass moves towards the new frequent class
    np.testing.assert_allclose(cb.prior_correct(frm[None, :], to, frm), to[None, :])   # the prior itself maps to `to`


def test_n_to_reach_interpolates_in_log_n_and_per_doubling():
    ns = np.array([100, 200, 400, 800, 1600])
    ll = 1.2 - 0.01 * np.log2(ns / 100)                                    # −0.01 per doubling
    assert cb.per_doubling(ns, ll, n_min=100) == pytest.approx(-0.01)
    assert cb.n_to_reach(ns, ll, 1.185) == pytest.approx(100 * 2 ** 1.5, rel=1e-6)
    assert cb.n_to_reach(ns, ll, 1.3) == 100                                # already there at the first point
    assert np.isnan(cb.n_to_reach(ns, ll, 1.0))                             # never reached


def test_permuted_groups_keep_sizes_and_patients_together():
    rng = np.random.default_rng(1)
    subj = np.repeat(np.arange(300).astype(str), rng.integers(1, 4, 300))
    g_pat = rng.choice(np.array(["a", "b", "c", None], dtype=object), 300)
    groups = g_pat[subj.astype(int)]
    perm = cb.permuted_groups(subj, groups, seed=3)
    for s in np.unique(subj):                                                # one group per patient
        assert len(set(perm[subj == s])) == 1
    pats, first = np.unique(subj, return_index=True)
    assert sorted(map(str, groups[first])) == sorted(map(str, perm[first]))  # the same patient counts per group
    assert (perm[first] != groups[first]).any()


def test_knn_uses_at_most_the_case_base_and_returns_the_kth_distance():
    rng = np.random.default_rng(2)
    E = rng.normal(size=(30, 2)).astype(np.float32)
    y = rng.integers(0, 3, 30)
    p, I, dk = cb.knn(E, y, E[:5], 50, 1.0, 3)                               # k > case base: k_eff = 30
    assert I.shape == (5, 30)
    assert np.allclose(p.sum(1), 1.0)
    prior = np.bincount(y, minlength=3) / 30
    np.testing.assert_allclose(p, np.tile((np.bincount(y, minlength=3) + prior * 3) / (30 + 3), (5, 1)))
    assert (dk > 0).all()
