"""scripts/posthoc_fixed_metric.py: the mutual-information weights, the statistics and the hypo chain end to end."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthoc_fixed_metric as F  # noqa: E402


def test_plugin_mi_matches_closed_form():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 200_000)
    y = np.where(rng.random(x.size) < np.where(x == 1, 0.8, 0.2), 1, 0)
    # I = H(Y) − H(Y|X) with P(Y=1) = 0.5, H(Y|X) = H(0.2)
    h = lambda p: -(p * np.log(p) + (1 - p) * np.log(1 - p))  # noqa: E731
    assert F.plugin_mi(x, y) == pytest.approx(h(0.5) - h(0.2), abs=2e-3)
    assert F.plugin_mi(rng.integers(0, 5, 200_000), rng.integers(0, 2, 200_000)) < 1e-4


def test_rare_categories_are_pooled():
    v = np.array(["a"] * 100 + ["b"] * 60 + ["c"] * 3 + ["d"] * 2 + [None] * 5, dtype=object)
    code, obs = F._cat_code(v)
    assert obs.sum() == 165 and set(code.tolist()) == {0, 1, 2}
    assert (code[100:160] == code[100]).all() and code[160] == code[164]   # c and d share the pooled code


def test_weight_vector_order_scale_and_missing_names():
    class E:
        feature_names = ["a", "b", "c"]
    w = F.weight_vector(E, {"a": 0.2, "c": 0.6})
    assert w.tolist() == pytest.approx([0.75, 0.0, 2.25]) and w.sum() == pytest.approx(3)
    with pytest.raises(ValueError):
        F.weight_vector(E, {})


def test_share_gate_and_difference_ci():
    rng = np.random.default_rng(1)
    B = {a: rng.normal(m, 1e-4, 400) for a, m in (("euclidean", 0.50), ("x", 0.49), ("y", 0.48), ("z", 0.4999))}
    est = {a: float(v.mean()) for a, v in B.items()}
    s = F._share(est, B, ("euclidean", "x"), ("euclidean", "y"))
    assert s["est"] == pytest.approx(0.5, abs=0.02) and s["ci90"][0] <= s["est"] <= s["ci90"][1]
    assert F._share(est, B, ("euclidean", "x"), ("euclidean", "z"))["status"] == "not evaluable"
    d = F._diff(est, B, "x", "y")
    assert d["est"] == pytest.approx(0.01, abs=1e-3) and d["ci95"][0] > 0


def test_hypo_chain_on_a_synthetic_external_stage(tmp_path, monkeypatch):
    """Phase 2 with the Euclidean comparator, quick frozen artefacts, the external stage on eICU, then the script:
    the gate must reproduce the Euclidean comparator and the report must carry every arm."""
    from hypo.config import load
    from test_hypo_external import _xframe

    cfg = load()
    monkeypatch.setenv("PERISIM_WORKDIR", str(tmp_path))
    d = tmp_path / "experiments" / "hypo" / "landmarks"
    d.mkdir(parents=True)
    for i, c in enumerate(cfg["cohorts"]):
        _xframe(c, 120, 4, 10 + i).write_parquet(d / f"{c}.parquet")
    from hypo.cv import run
    from hypo.external import Stage
    from hypo.frozen import external_dir, final, freeze

    run(cfg["cohorts"], ["prevalence", "euclidean"], None, quick=False, threads=1)
    final(quick=True, threads=1)
    freeze("TEST", no_git=True, quick=True)
    Stage(quick=True, threads=1).run(["eicu"], folds=[0, 1], parts=["transported", "local"])
    assert F.main(["--sites", "eicu", "--threads", "1"]) == 0
    out = json.loads((external_dir().with_name("posthoc_fixed_metric") / "fixed_metric.json").read_text())
    u = out["sites"]["eicu"]["all"]
    assert set(u["arms"]) == set(F.ARMS)
    assert u["n"] > 0 and isinstance(u["delta1"]["learned_metric_adds"], bool)
    rep = (tmp_path / "experiments" / "hypo" / "reports" / "posthoc_fixed_metric.md").read_text()
    assert "Rule (pre-specified)" in rep and "eicu" in rep


def test_gate_fails_when_the_comparator_does_not_reproduce(monkeypatch):
    """A rebuilt comparator that differs from Phase 2 must stop the script."""
    import polars as pl

    class FakeSites:
        cfg = {"cohorts": ["a", "b"], "seed": 1}

        def lm(self, c):
            return None

    class St:
        S, mcfg = FakeSites(), {}

    monkeypatch.setattr(F, "knn_arm", lambda *a, **k: (np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]), {}))
    import hypo.external_report as R
    monkeypatch.setattr(R, "_phase2", lambda c, n: pl.DataFrame({"row": [0, 1, 2], "p": [0.1, 0.2, 0.31]}))
    with pytest.raises(RuntimeError, match="does not reproduce"):
        F.run_site(St(), "a", Path("/nonexistent"), gate_only=True)
