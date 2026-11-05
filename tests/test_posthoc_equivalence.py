"""The statistic of scripts/posthoc_equivalence.py: R, its patient-level bootstrap and the verdict."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthoc_equivalence as E  # noqa: E402


def _rows(n_pat=400, per=3, gain=0.02, kept=0.9, seed=0):
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_pat), per)
    ll_l = rng.gamma(2.0, 0.25, subj.size)                      # local
    ll_e = ll_l + gain + rng.normal(0, 0.01, subj.size)         # Euclidean: worse by `gain`
    ll_t = ll_e - kept * (ll_e - ll_l)                          # transported keeps `kept` of it, row by row
    return ll_e, ll_t, ll_l, subj


def test_r_is_the_share_of_the_gain_kept():
    ll_e, ll_t, ll_l, subj = _rows(kept=0.9)
    r = E.ratio_bootstrap(ll_e, ll_t, ll_l, subj, n_boot=200)
    assert r["R"]["est"] == pytest.approx(0.9)
    assert r["cost"]["est"] == pytest.approx(0.1 * r["gain_local"]["est"])
    assert r["patients"] == 400 and r["n"] == 1200


def test_bounds_are_ordered_and_the_90_interval_is_inside_the_95():
    ll_e, ll_t, ll_l, subj = _rows(kept=0.85, seed=1)
    ll_t = ll_t + np.random.default_rng(2).normal(0, 0.02, ll_t.size)   # noise, so the CIs have width
    r = E.ratio_bootstrap(ll_e, ll_t, ll_l, subj, n_boot=400)
    lo90, hi90 = r["R"]["ci90"]
    lo95, hi95 = r["R"]["ci95"]
    assert lo95 <= lo90 <= r["R"]["est"] <= hi90 <= hi95


def test_resamples_are_by_patient_and_reproducible():
    ll_e, ll_t, ll_l, subj = _rows(seed=3)
    a = E.ratio_bootstrap(ll_e, ll_t, ll_l, subj, n_boot=300, seed=5)
    b = E.ratio_bootstrap(ll_e, ll_t, ll_l, subj, n_boot=300, seed=5)
    assert a["R"]["ci90"] == b["R"]["ci90"]
    # the order of the rows does not matter, only which patient they belong to
    p = np.random.default_rng(0).permutation(subj.size)
    c = E.ratio_bootstrap(ll_e[p], ll_t[p], ll_l[p], subj[p], n_boot=300, seed=5)
    assert c["R"]["est"] == pytest.approx(a["R"]["est"]) and c["R"]["ci90"] == pytest.approx(a["R"]["ci90"])


def test_blocks_do_not_change_the_replicates(monkeypatch):
    ll_e, ll_t, ll_l, subj = _rows(seed=4)
    ll_t = ll_t + np.random.default_rng(1).normal(0, 0.02, ll_t.size)
    a = E.ratio_bootstrap(ll_e, ll_t, ll_l, subj, n_boot=300, seed=9)
    monkeypatch.setattr(E, "BLOCK", 300)
    b = E.ratio_bootstrap(ll_e, ll_t, ll_l, subj, n_boot=300, seed=9)
    assert a["R"]["ci90"] == pytest.approx(b["R"]["ci90"])


def test_verdicts():
    def r(lo90, gain, gain_lo):
        return {"R": {"ci90": [lo90, 1.0]}, "gain_local": {"est": gain, "ci95": [gain_lo, gain + 0.01]}}

    assert E.verdict(r(0.82, 0.01, 0.005)) == "equivalent"
    assert E.verdict(r(0.80, 0.01, 0.005)) == "equivalent"                # the limit is inclusive
    assert E.verdict(r(0.79, 0.01, 0.005)) == "not equivalent"
    assert E.verdict(r(0.95, 0.004, 0.001)) == "not evaluable"            # gain below 0.005 nats
    assert E.verdict(r(0.95, 0.01, -0.001)) == "not evaluable"            # gain's CI includes 0


def test_no_gain_is_not_evaluable_end_to_end():
    ll_e, ll_t, ll_l, subj = _rows(gain=0.0, kept=0.5, seed=6)
    u = E.unit(ll_e, ll_t, ll_l, subj, seed=1)
    assert u["verdict"] == "not evaluable"


def test_hypo_part_on_a_synthetic_external_stage(tmp_path, monkeypatch):
    """The whole chain on synthetic landmarks (as tests/test_hypo_external.py): Phase 2 with the Euclidean kNN, quick
    frozen artefacts, the external stage on eICU, then the script's hypo part."""
    import json

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
    res = E.part_hypo(["eicu"])
    u = res["sites"]["eicu"]["all"]
    assert u["verdict"] in {"equivalent", "not equivalent", "not evaluable"}
    assert u["n"] > 0 and np.isfinite(u["R"]["est"]) and len(u["R"]["ci90"]) == 2
    saved = json.loads((external_dir().with_name("posthoc_equivalence") / "equivalence.json").read_text())
    assert saved["sites"]["eicu"]["all"]["R"]["est"] == pytest.approx(u["R"]["est"])
    assert (tmp_path / "experiments" / "hypo" / "reports" / "posthoc_equivalence.md").exists()
