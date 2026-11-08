"""scripts/pram_factors_t1.py: PRAM's vote, the choice of a, the full NCA, and the chain on synthetic data."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pram_factors_t1 as T  # noqa: E402


def test_inv_vote_euclidean_and_cosine():
    D = np.array([[1.0, 4.0, 9.0]])                # squared distances 1, 4, 9 → d = 1, 2, 3
    I = np.array([[0, 1, 2]])
    valid = np.array([[True, True, False]])
    y = np.array([1.0, 0.0, 1.0])
    w = np.array([1 / (1 + 1e-6), 1 / (2 + 1e-6)])
    assert T.inv_vote(D, I, valid, y, False, 0.3)[0] == pytest.approx(w[0] / w.sum())
    wc = np.array([1 / (0.5 + 1e-6), 1 / (2 + 1e-6)])  # cosine: d = D / 2
    assert T.inv_vote(D, I, valid, y, True, 0.3)[0] == pytest.approx(wc[0] / wc.sum())
    assert T.inv_vote(D, I, np.zeros_like(valid), y, False, 0.3)[0] == pytest.approx(0.3)


def test_choose_a_prefers_the_better_mixture_and_the_smaller_a_on_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000).astype(float)
    base = rng.random(2000)                         # useless base model
    knn = np.clip(0.3 * y + 0.7 * rng.random(2000), 0, 1)
    assert T.choose_a(base, knn, y)["a"] == 0.5
    c = T.choose_a(knn, knn, y)                     # every a gives the same ranking: ties → 0
    assert c["a"] == 0.0 and len(c["auroc_curve"]) == T.A_GRID.size


def test_inner_groups_keep_patients_together():
    s = np.array(["a", "a", "b", "c", "c", "c", "d", "e"])
    g = T.inner_groups(s, 1)
    for p in np.unique(s):
        assert len(set(g[s == p])) == 1
    assert set(g) <= {0, 1, 2}


def test_linear_metric_and_full_nca():
    class Base:
        def embed(self, lm):
            return lm.E

    class L:
        def __init__(self, E, y, s):
            self.E, self.y, self.ft = E, y, type("F", (), {"subjects": s})()

        def take(self, r):
            return L(self.E[r], self.y[r], self.ft.subjects[r])

    rng = np.random.default_rng(2)
    n = 400
    y = rng.integers(0, 2, n)
    E = np.c_[y + 0.3 * rng.normal(size=n), rng.normal(size=(n, 3))]   # only axis 0 carries y
    lm = L(E, y, np.array([f"p{i // 2}" for i in range(n)]))
    Lm, info = T.fit_full_nca(Base(), [lm], 150)
    assert info["n"] == 150 and Lm.shape == (4, 4)                      # one landmark per patient: 200 available
    m = T.LinearMetric(Base(), Lm)
    assert m.embed(lm).shape == (n, 4) and np.allclose(m.sqrt_w_dims, 1)
    col = np.abs(Lm).sum(0)
    assert col[0] > col[1:].max()                                       # the learned map favours the signal axis


def test_t1_chain(tmp_path, monkeypatch):
    from test_hypo_external import _xframe
    from test_posthoc_hospital_leak import _chain

    import posthoc_fixed_metric as FM
    _chain(tmp_path, monkeypatch, ["transported", "local", "comps"])
    monkeypatch.setattr(FM, "N_BOOT", 100)
    assert FM.main(["--sites", "eicu", "--threads", "1"]) == 0
    nd = tmp_path / "experiments" / "nwicu" / "landmarks"
    nd.mkdir(parents=True)
    _xframe("mimic", 100, 4, 99).with_columns(pl.lit(None, pl.Boolean).alias("vent_on")).write_parquet(nd / "nwicu.parquet")
    import nwicu_transport as NT
    monkeypatch.setattr(NT, "N_BOOT", 100)
    assert NT.main(["--threads", "1", "--quick"]) == 0
    monkeypatch.setattr(T, "N_BOOT", 100)
    monkeypatch.setattr(T, "NCA_N", {"ncaf": 60, "ncaf2k": 30, "ncaf10k": 90})
    assert T.main(["--sites", "eicu", "nwicu", "--threads", "1", "--quick"]) == 0
    from hypo.frozen import external_dir
    out = json.loads((external_dir().with_name("pram_factors_t1") / "t1.json").read_text())
    r = out["sites"]["eicu"]["a"]
    assert set(r["rule"]["ncad"]["switch"]) == {"U", "V", "I", "UV", "UI", "VI", "UVI"}
    assert set(r["rule"]["ncad"]["effects"]) == {"U", "V", "I"}
    assert "ncaf10k|ours|alone" in r["cells"] and "mi|inv|mix_gbm" in r["cells"]
    assert set(out["sites"]["nwicu"]["rotations"]) == set(T.ROTATIONS)
    rep = (external_dir().parent / "reports" / "pram_factors_t1.md").read_text()
    assert "Rule (pre-specified)" in rep and "NWICU" in rep
