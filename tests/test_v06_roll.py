"""v0.6 fixes of the `roll` experiment (audit 2026-11-05, items 11, 12 and the roll low-severity items): H4b's
monitoring window stops at the deployment, AUPRC is tie-neutral, H6 (c) leaves out windows without a test, H6 (a)'s
SE is a patient bootstrap, the W sensitivity reads the configured W, a cross-cohort stretch's delay uses its own rate.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from roll import endpoints as EP
from roll import metrics as Mx
from roll.arms import ML, Prevalence
from roll.config import load
from roll.loop import Run
from roll.streams import segment_pop, stretch_rate


# --------------------------------------------------------------------------- AUPRC (audit 12)
def _exact_tie_ap(p, y):
    """E[AP] over every order of the tied items (brute force)."""
    groups = [np.flatnonzero(p == v) for v in np.unique(p)[::-1]]
    vals = []
    for perm in itertools.product(*[itertools.permutations(g) for g in groups]):
        o = np.concatenate([np.array(x) for x in perm])
        ys = y[o]
        tp = np.cumsum(ys)
        vals.append(float((tp / np.arange(1, len(ys) + 1))[ys == 1].sum() / ys.sum()))
    return float(np.mean(vals))


def test_auprc_equals_scikit_learn_without_ties():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = rng.random(500)
        y = (rng.random(500) < p).astype(float)
        assert Mx.auprc(p, y, np.ones(500)) == pytest.approx(average_precision_score(y, p), abs=1e-12)


def test_auprc_is_the_expectation_over_random_tie_breaks():
    rng = np.random.default_rng(4)
    for _ in range(15):
        p = rng.integers(0, 3, 8).astype(float)
        y = (rng.random(8) < 0.5).astype(float)
        if y.sum() == 0:
            y[0] = 1.0
        assert Mx.auprc(p, y, np.ones(8)) == pytest.approx(_exact_tie_ap(p, y), abs=1e-12)
    # and against the Monte Carlo mean on a kNN-like vote (51 levels), where the old stepping was biased
    n = 1500
    r = 1 / (1 + np.exp(-rng.normal(-1.2, 1.3, n)))
    y = (rng.random(n) < r).astype(float)
    p = (rng.binomial(50, r) + 0.56) / 52
    mc = np.mean([average_precision_score(y, p + rng.uniform(0, 1e-9, n)) for _ in range(300)])
    assert Mx.auprc(p, y, np.ones(n)) == pytest.approx(mc, abs=1e-3)


def test_auprc_weights_are_copies_and_replicates_match_rows():
    rng = np.random.default_rng(5)
    p = rng.integers(0, 5, 60).astype(float)
    y = (rng.random(60) < 0.4).astype(float)
    w = rng.integers(0, 3, 60).astype(float)
    k = w.astype(int)
    assert Mx.auprc(p, y, w) == pytest.approx(Mx.auprc(np.repeat(p, k), np.repeat(y, k), np.ones(k.sum())), abs=1e-12)
    W = np.stack([w, np.ones(60), rng.integers(0, 4, 60).astype(float)])
    got = Mx.auprc(p, y, W)
    for i in range(3):
        assert got[i] == pytest.approx(Mx.auprc(p, y, W[i]), abs=1e-12)


# --------------------------------------------------------------------------- H4b window (audit 11)
class _Fixed:
    """An arm that only carries predictions (the run's `pred`)."""

    def __init__(self, name):
        self.name, self.log = name, []


def _stub(M, n=400, dep=200):
    y = (np.arange(n) % 10 < 3).astype(np.int64)                  # exactly 0.3 in every run of 10 patients
    p = np.where(np.arange(n) < dep, 0.6, 0.3)                    # miscalibrated before the deployment, fine after
    ml = ML("lightgbm_realistic", "lightgbm", "realistic")
    ml.log = [{"event": "deploy", "pos": 0}, {"event": "deploy", "pos": dep}]
    prev = Prevalence("prevalence")                               # never deploys: the last M
    cfg = {"monitor": {"M": {"mimic": M}, "E": 10}, "ml": {"primary_family": "none"},
           "triggers": {"primary": {"oe_low": 0.9, "oe_high": 1.1, "citl_band": 0.10, "z": 1.959964}}}
    s = SimpleNamespace(cfg=cfg, stream=SimpleNamespace(seq=np.arange(n), pop={"s": "mimic"}), y=y,
                        pcode=np.arange(n), arms=[ml, prev], checks=[],
                        pred={"lightgbm_realistic": p, "prevalence": p.copy()})
    s.M = lambda seg: Run.M(s, seg)
    s.rows_of = lambda u: np.array([int(u)])
    s._in_service_since = Run._in_service_since
    return s


def test_h4b_window_stops_at_the_deployment():
    s = _stub(M=80)
    Run._check_all(s, {"seg": "s", "index": 0}, 300)
    rec = s.checks[-1]
    assert rec["lightgbm_realistic"] is False                    # only the new model's 100 patients
    assert rec["prevalence"] is False                            # the last 80, all after 200
    s = _stub(M=150)
    Run._check_all(s, {"seg": "s", "index": 0}, 300)
    rec = s.checks[-1]
    assert "lightgbm_realistic" not in rec                       # 100 patients since the deployment < M: no check
    assert rec["prevalence"] is True                             # the same predictions, last 150 cross position 200
    # a deployment at the check's own position has predicted nobody: the window is that of the previous model
    s = _stub(M=80, dep=300)
    Run._check_all(s, {"seg": "s", "index": 0}, 300)
    assert s.checks[-1]["lightgbm_realistic"] is True


# --------------------------------------------------------------------------- H6 (audit, roll low)
def _h6_runs(cfg, seed=1, orders=1, ood_rate=0.05):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(4):
        for p in range(20):
            for _ in range(10):
                prev = 0.02 if w == 2 else 0.3
                rows.append((w, w * 100 + p, int(rng.random() < prev), rng.random() < ood_rate))
    w_, pat, y, ood = map(np.array, zip(*rows))
    fr = pl.DataFrame({"window": w_, "patient": pat, "y": y, "ood": ood, "p_ml": np.full(len(y), 0.6),
                       "p_perisim_fifo": np.full(len(y), 0.3), "pos": w_ * 20})
    wins = [{"start": i * 20, "end": (i + 1) * 20, "seg": "2014" if i < 2 else "2015", "index": i} for i in range(4)]
    return [(fr, {"windows": wins})] * orders


def test_h6c_leaves_out_windows_without_a_test():
    cfg = load()
    cfg["windows"]["boot"] = 20
    res = EP.h6_hospital(_h6_runs(cfg), cfg, "ml", 0)
    assert res["ml_false_alarms"][0] == 1.0                      # alarms in both control windows with a test
    assert np.isnan(res["ml_signal"][0])                         # the boundary window has no test


def test_h6a_se_is_a_sampling_se_that_does_not_shrink_with_the_orders():
    cfg = load()
    cfg["windows"]["boot"] = 200
    s2 = EP.h6_hospital(_h6_runs(cfg, orders=2), cfg, "ml", 0)["signal"]
    s8 = EP.h6_hospital(_h6_runs(cfg, orders=8), cfg, "ml", 0)["signal"]
    assert s2[1] > 0                                             # the orders agree exactly: their SD would be 0
    assert s8[1] == pytest.approx(s2[1], rel=1e-12)


# --------------------------------------------------------------------------- W sensitivity (audit, roll low)
def test_rewindow_uses_the_configured_w():
    cfg = load()                                                 # mimic W = 100
    n = 240                                                      # windows 100 + 140 (a merged remainder): median 120
    fr = pl.DataFrame({"pos": np.arange(n), "window": np.r_[np.zeros(100, int), np.ones(140, int)]})
    log = {"stream": {"stream": "mimic_temporal", "segments": [{"segment": "b1", "patients": n}]},
           "windows": [{"start": 0, "end": 100, "seg": "b1", "index": 0}, {"start": 100, "end": 240, "seg": "b1", "index": 1}],
           "checks": []}
    _, lg = EP.rewindow(fr, log, 0.5, cfg)
    assert [w["start"] for w in lg["windows"]] == [0, 50, 100, 150, 200]
    assert segment_pop(cfg["streams"]["eicu_within"], "2014") == "eicu_hospital"
    assert segment_pop(cfg["streams"]["eicu_mimic"], "eicu") == "eicu"


# --------------------------------------------------------------------------- cross-cohort delay (audit, roll low)
def test_a_stretch_rate_is_its_own():
    mo = load()["months"]
    blocks = np.array(["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"] * 400, dtype=object)
    mimic = SimpleNamespace(name="mimic", block=blocks, year=np.full(2000, -1))
    assert stretch_rate(mimic, np.arange(2000), mo) == pytest.approx(2000 / 180)
    assert stretch_rate(mimic, np.arange(0, 2000, 5), mo) == pytest.approx(400 / 36)   # one block only
    eicu = SimpleNamespace(name="eicu", block=np.full(1000, "", dtype=object), year=np.repeat([2014, 2015], 500))
    assert stretch_rate(eicu, np.arange(1000), mo) == pytest.approx(1000 / 24)
    insp = SimpleNamespace(name="inspire", block=np.full(500, "", dtype=object), year=np.full(500, -1))
    assert stretch_rate(insp, np.arange(500), mo) == pytest.approx(500 / 120)
