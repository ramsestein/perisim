"""scripts/posthoc_cosine.py: the cosine search, the weighted AUROC, and the chain on synthetic data."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthoc_cosine as P  # noqa: E402


class _Base:
    def __init__(self, E, w):
        self.E, self.w = E, w

    def embed(self, lm):
        return self.E[lm]

    @property
    def sqrt_w_dims(self):
        return np.sqrt(self.w)


def test_cosine_metric_ranks_as_cosine():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(200, 6)) + 3.0                 # off-centre on purpose: the centring matters
    w = rng.uniform(0.2, 3.0, 6)
    ref = np.arange(150)
    cm = P.CosineMetric(_Base(E, w), ref)
    U = cm.embed(np.arange(200))
    assert np.allclose(np.linalg.norm(U, axis=1), 1, atol=1e-5) and np.allclose(cm.sqrt_w_dims, 1)
    X = E * np.sqrt(w) - (E[ref] * np.sqrt(w)).mean(0)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    for q in (150, 170, 199):
        cos_rank = np.argsort(-(Xn[ref] @ Xn[q]), kind="stable")
        d_rank = np.argsort(((U[ref] - U[q]) ** 2).sum(1), kind="stable")
        assert np.array_equal(cos_rank[:20], d_rank[:20])


def test_wauc_matches_sklearn_and_weights():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + y, 1)           # ties
    w = rng.integers(0, 4, 500).astype(float)
    assert abs(P.wauc(s, y, np.ones(500)) - roc_auc_score(y, s)) < 1e-12
    assert abs(P.wauc(s, y, w) - roc_auc_score(y, s, sample_weight=w)) < 1e-12
    assert abs(P._AUC(s, y)(w) - roc_auc_score(y, s, sample_weight=w)) < 1e-12


def test_verdict():
    assert P.verdict([0.001, 0.01], "a", "b") == "a"
    assert P.verdict([-0.01, -0.001], "a", "b") == "b"
    assert P.verdict([-0.01, 0.01], "a", "b") == "no difference shown"


def test_cosine_chain(tmp_path, monkeypatch):
    from test_hypo_external import _xframe
    from test_posthoc_hospital_leak import _chain

    _chain(tmp_path, monkeypatch, ["transported", "local"])
    nd = tmp_path / "experiments" / "nwicu" / "landmarks"
    nd.mkdir(parents=True)
    _xframe("mimic", 100, 4, 99).with_columns(pl.lit(None, pl.Boolean).alias("vent_on")).write_parquet(nd / "nwicu.parquet")
    import nwicu_transport as NT
    monkeypatch.setattr(NT, "N_BOOT", 100)
    assert NT.main(["--threads", "1", "--quick"]) == 0
    monkeypatch.setattr(P, "N_BOOT", 100)
    assert P.main(["--sites", "eicu", "nwicu", "--threads", "1", "--quick"]) == 0
    from hypo.frozen import external_dir
    out = json.loads((external_dir().with_name("posthoc_cosine") / "cosine.json").read_text())
    assert set(out["sites"]) == {"eicu", "nwicu"}
    assert set(out["sites"]["nwicu"]["primary"]) == {f"nca_transported_{e}" for e in P.ROTATIONS}
    for r in out["sites"].values():
        for d in r["primary"].values():
            assert d["verdict"] in ("NCA beats cosine", "cosine beats NCA", "no difference shown")
    rep = (external_dir().parent / "reports" / "posthoc_cosine.md").read_text()
    assert "Rule (pre-specified)" in rep
