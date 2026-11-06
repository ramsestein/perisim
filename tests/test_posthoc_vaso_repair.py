"""scripts/posthoc_vaso_repair.py: the rule, the prior and the hypo chain end to end on synthetic landmarks."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthoc_vaso_repair as V  # noqa: E402


def _site(lower_vaso, upper_all):
    arms = V.REPAIRS + ["nca_transported"]
    vs = {a: {"est": 0.0, "ci95": [lower_vaso, lower_vaso + 0.01], "loss_upper95": upper_all} for a in arms}
    return {"all": {"vs_refit": vs}, "vaso_on = 1": {"vs_refit": vs}}


def test_rule_needs_repair_and_safety_everywhere():
    good = {"mimic": _site(0.002, 0.0002), "eicu": _site(0.0, 0.0005), "inspire": _site(0.0, 0.0001)}
    assert all(v["works"] for v in V.verdicts(good).values())
    unsafe = {**good, "eicu": _site(0.0, 0.002)}
    assert not any(v["works"] for v in V.verdicts(unsafe).values())
    no_repair = {**good, "mimic": _site(-0.001, 0.0002)}
    assert not any(v["repairs"] for v in V.verdicts(no_repair).values())


def test_hypo_chain_on_a_synthetic_external_stage(tmp_path, monkeypatch):
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
    Stage(quick=True, threads=1).run(["mimic"], folds=[0, 1], parts=["transported", "local"])
    monkeypatch.setattr(V, "LAMBDAS", [0.0, 1.0])                     # keep the test short
    assert V.main(["--sites", "mimic", "--threads", "1", "--quick"]) == 0
    out = json.loads((external_dir().with_name("posthoc_vaso_repair") / "vaso_repair.json").read_text())
    r = out["sites"]["mimic"]
    assert set(V.REPAIRS) <= set(r["all"]["arms"]) and "nca_refit" in r["all"]["arms"]
    assert r["weights"]["lambda"] in (0.0, 1.0)
    assert set(out["rule"]) == set(V.REPAIRS)
    rep = (tmp_path / "experiments" / "hypo" / "reports" / "posthoc_vaso_repair.md").read_text()
    assert "Rule (pre-specified)" in rep
