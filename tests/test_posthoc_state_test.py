"""scripts/posthoc_state_test.py: the paired stratum bootstrap, R, the rule and the hypo chain end to end."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthoc_state_test as T  # noqa: E402


def test_boot_strata_shares_weights_across_strata():
    rng = np.random.default_rng(0)
    subj = np.repeat(np.arange(300), 4)
    on = rng.random(subj.size) < 0.3
    a = rng.normal(0.5, 0.1, subj.size)
    B = T.boot_strata({"a": a, "b": a + 0.01}, subj, {"all": np.ones(subj.size, bool), "on": on, "off": ~on}, 400, 1)
    d = B["on"]["b"] - B["on"]["a"]
    assert np.allclose(d, 0.01)                                     # same weights for both arms
    assert B["all"]["a"].mean() == pytest.approx(a.mean(), abs=0.01)


def test_ratio_r_and_its_denominator_rule():
    rng = np.random.default_rng(1)
    B = {"euclidean": rng.normal(0.50, 1e-4, 500), "nca_local": rng.normal(0.48, 1e-4, 500),
         "nca_transported": rng.normal(0.482, 1e-4, 500)}
    e = {"est": {k: float(v.mean()) for k, v in B.items()}, "B": B}
    r = T.ratio_r(e, "nca_transported")
    assert r["est"] == pytest.approx(0.9, abs=0.02) and r["verdict"] == "equivalent"
    B2 = dict(B, nca_local=rng.normal(0.499, 1e-4, 500))
    e2 = {"est": {k: float(v.mean()) for k, v in B2.items()}, "B": B2}
    assert T.ratio_r(e2, "nca_transported")["verdict"] == "not evaluable"


def _v(failure, repairs, safe):
    return {"verdict": {"failure": failure, "repairs": repairs, "safe": safe}}


def test_rule_readings():
    w = {"sedation_on": {"inspire": 0.02, "mimic": 0.001, "eicu": 0.004}}
    sites = {"mimic": {"states": {"sedation_on": _v(True, True, True), "vent_on": _v(False, False, True)}},
             "eicu": {"states": {"sedation_on": _v(False, False, True), "vent_on": _v(False, False, True)}},
             "inspire": {"states": {"vent_on": _v(False, False, True)}}}
    r = T.rule(sites, w)
    assert r["P1_weight"] and r["P2_sedation"]["reading"] == "confirmed" and r["P3_vent_control"]["holds"]
    sites["mimic"]["states"]["sedation_on"] = _v(True, False, True)
    assert T.rule(sites, w)["P2_sedation"]["reading"] == "not confirmed"
    sites["mimic"]["states"]["sedation_on"] = _v(False, False, True)
    assert T.rule(sites, w)["P2_sedation"]["reading"].startswith("no failure")
    sites["inspire"]["states"]["vent_on"] = _v(True, True, True)
    assert not T.rule(sites, w)["P3_vent_control"]["holds"]


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

    run(cfg["cohorts"], ["prevalence", "euclidean", "lightgbm"], None, quick=False, threads=1)
    final(quick=True, threads=1)
    freeze("TEST", no_git=True, quick=True)
    Stage(quick=True, threads=1).run(["mimic"], folds=[0, 1], parts=["transported", "local", "comps"])
    monkeypatch.setattr(T, "N_BOOT", 200)
    assert T.main(["--sites", "mimic", "--threads", "1", "--quick"]) == 0
    out = json.loads((external_dir().with_name("posthoc_state_test") / "state_test.json").read_text())
    r = out["sites"]["mimic"]
    assert set(out["weights"]) == set(T.STATES)
    for s in r["states"].values():
        assert {"cost_on", "repair_on", "safety_all", "R", "verdict"} <= set(s)
        assert "b0" in s["all"]["arms"]
    rep = (tmp_path / "experiments" / "hypo" / "reports" / "posthoc_state_test.md").read_text()
    assert "Rule (pre-specified)" in rep
