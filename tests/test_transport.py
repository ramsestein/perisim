"""§6.2b transport analyses (pre-registration amendment of 2026-10-24): site weights, disjoint splits, frozen-artefact
hashes, cumulative-threshold calibration, between-site homogeneity of the shapes and the job list."""

import json

import numpy as np
import pytest

from perisim.features.table import FeatureTable
from perisim.knn.predict import KNNModel
from perisim.transport.core import (
    assert_disjoint, case_base, check_hashes, concat_ft, cumulative_calibration, frozen_hashes, gamma_schedule,
    patient_orders, patient_split, predict_gammas,
)
from perisim.transport.run import shape_homogeneity, transport_jobs


def _toy(n=400, d=6, C=3, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, d))
    y = (E[:, 0] > 0).astype(int) + (E[:, 1] > 1).astype(int)
    return E, np.minimum(y, C - 1), np.array([f"P{i // 2}" for i in range(n)])


def test_gamma_one_reproduces_the_unweighted_knn():
    E, y, s = _toy()
    sw = np.ones(E.shape[1])
    kn = KNNModel(E[:300], sw, y[:300], 3, s[:300], [5, 10])
    D, I, v = kn.neighbors(E[300:], s[300:])
    P0 = kn.probs(I, v, 10, alpha=1.0)
    kn.set_ref_weight(np.ones(300))
    np.testing.assert_array_equal(kn.probs(I, v, 10, alpha=1.0), P0)
    # the same through the transport helpers (local and non-local parts, γ = 1)
    kcfg = {"alpha_grid": [1.0], "k_primary": 10, "ood_percentile": 99}
    cb = case_base([(E[:150], y[:150], s[:150], True), (E[150:300], y[150:300], s[150:300], False)],
                   sw, 3, kcfg, [5, 10], alpha={5: 1.0, 10: 1.0})
    P, _ = predict_gammas(cb, E[300:], s[300:], 10, {"1": 1.0, "0.5": 0.5, "0": 0.0})
    np.testing.assert_array_equal(P["1"], P0)
    assert not np.allclose(P["0.5"], P0)                      # γ < 1 changes the votes of the non-local cases
    assert np.allclose(P["0"].sum(1), 1.0)


def test_splits_are_disjoint_by_patient():
    subj = np.array([f"MIMIC-{i // 3}" for i in range(3000)])
    sp = patient_split(subj, 0.2, 500, seed=4)
    assert_disjoint(sp, subj)
    n_pat = {k: len(set(subj[v])) for k, v in sp.items()}
    assert n_pat["test"] == 200 and n_pat["tuning"] == 400      # 20 %; tuning ≤ half of the rest
    assert sum(len(v) for v in sp.values()) == len(subj)
    bad = {"test": np.array([0]), "pool": np.array([1])}        # two stays of patient 0 on both sides
    with pytest.raises(AssertionError):
        assert_disjoint(bad, subj)


def test_frozen_artefact_hashes(tmp_path):
    from perisim.workspace import Workspace

    ws = Workspace(tmp_path / "work", experiment="t2")
    (ws.state / "freeze.json").write_text(json.dumps({"tag": "x"}))
    (ws.state / "frozen_config").mkdir(parents=True, exist_ok=True)
    (ws.state / "frozen_config" / "experiment.yaml").write_text("seed: 1\n")
    (ws.predictions("external", "t2") / "final_model.pkl").write_bytes(b"\x80\x04frozen")
    h0 = frozen_hashes(ws, ["t2"])
    assert len(h0) == 3
    check_hashes(h0, frozen_hashes(ws, ["t2"]))                 # unchanged: no error
    (ws.predictions("external", "t2") / "final_model.pkl").write_bytes(b"\x80\x04other")
    with pytest.raises(RuntimeError):
        check_hashes(h0, frozen_hashes(ws, ["t2"]))


def test_cumulative_calibration_of_calibrated_predictions():
    rng = np.random.default_rng(1)
    P = rng.dirichlet([2, 2, 2, 0.3], size=20000)
    y = np.array([rng.choice(4, p=p) for p in P])
    for c in cumulative_calibration(P, y):
        assert abs(c["citl"]) < 0.1 and 0.85 < c["slope"] < 1.15


def test_shape_homogeneity_allows_a_uniform_scale():
    rng = np.random.default_rng(2)
    base = {f"v{j}": rng.normal(size=(8, 2)) for j in range(6)}
    se2 = np.full((8, 2), 0.002)

    def site(scale, distort=None):
        sh = {n: {"mean": g * scale, "se2": se2 * scale ** 2} for n, g in base.items()}
        if distort:
            sh[distort]["mean"] = sh[distort]["mean"][::-1].copy()
        return {"C": 2, "shapes": sh}

    E = shape_homogeneity({"inspire": site(1.0), "mimic": site(0.5)}, "inspire")
    assert not any(r["fails"] for r in E["rows"]) and abs(E["scale"]["mimic"] - 0.5) < 1e-6
    E2 = shape_homogeneity({"inspire": site(1.0), "mimic": site(0.5, distort="v3")}, "inspire")
    failing = {r["variable"] for r in E2["rows"] if r["fails"]}
    assert "v3" in failing and len(failing) <= 2


def test_block_order_follows_the_calendar():
    subj = np.array(["a", "a", "b", "c", "d"])
    blocks = np.array(["2017 - 2019", "2008 - 2010", "2011 - 2013", "2008 - 2010", "2017 - 2019"])
    o = patient_orders(subj, blocks, 3, seed=0)
    for perm in o["blocks"]:
        assert set(perm[:2]) == {"a", "c"} and perm[2] == "b"   # a's first stay is in 2008 - 2010
    assert len(o["random"]) == 3


def test_gamma_schedule_and_concat():
    assert gamma_schedule(0, 500) == 1.0 and gamma_schedule(500, 500) == 0.5
    a = FeatureTable(np.array(["INSPIRE-1"]), np.array(["INSPIRE-10"]), [], {"age": np.array([60.0])},
                     {"group": np.array(["cardiac"], dtype=object)})
    b = FeatureTable(np.array(["EICU-1"]), np.array(["EICU-9"]), [], {"age": np.array([50.0])},
                     {"group": np.array(["ortho"], dtype=object)})
    ab = concat_ft(a, b)
    assert ab.num["age"].tolist() == [60.0, 50.0] and ab.cat["group"].tolist() == ["cardiac", "ortho"]


def test_jobs_per_target(cfg):
    every = ["inspire", "mimic", "eicu"]
    j = transport_jobs(cfg, every, ["t2"])
    assert j["analyses"] == [("analysis", "t2", "mimic"), ("analysis", "t2", "eicu")]
    assert set(j["fits"]) == {("local", "t2", "mimic"), ("local", "t2", "eicu"), ("multisource", "t2", "mimic"),
                              ("shapes", "t2", "inspire"), ("shapes", "t2", "mimic"), ("shapes", "t2", "eicu")}
    j3 = transport_jobs(cfg, every, ["t3"])
    assert j3["analyses"] == [("analysis", "t3", "mimic")]                       # eICU has no full T1
    j1 = transport_jobs(cfg, every, ["t1"])
    assert ("shapes", "t1", "eicu") not in j1["fits"]                            # reduced T1: other classes
    assert transport_jobs(cfg, ["inspire", "mimic"], ["t2"])["fits"].count(("multisource", "t2", "mimic")) == 0
