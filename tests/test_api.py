"""The CEKNN estimator on synthetic data: fit, curves, weights, JSON, case bases, multiclass, SAGE."""

import json

import numpy as np
import pandas as pd
import pytest

from ceknn import CEKNN, ModelSpace
from synthetic import SMALL, make_data

INPUTS = ["u_shape", "linear", "noise", "kind", "flag", "lab"]


def _space():
    return ModelSpace.default(include=["logreg", "spline_logreg"])


@pytest.fixture(scope="module")
def data():
    return make_data(4000, seed=1, rows_per_patient=2)


@pytest.fixture(scope="module")
def fitted(data):
    return CEKNN(INPUTS, "y", group="pid", model_space=_space(), seed=0, **{**SMALL, "n_models": 10}).fit(data)


def test_the_curves_recover_the_shapes(fitted):
    u = fitted.shape("u_shape")
    lo = u["log_odds"].to_numpy()
    assert lo[0] > lo[len(lo) // 2] + 1 and lo[-1] > lo[len(lo) // 2] + 1          # U-shaped
    lin = fitted.shape("linear")["log_odds"].to_numpy()
    assert np.corrcoef(np.arange(len(lin)), lin)[0, 1] > 0.95                       # increasing
    k = fitted.shape("kind").set_index("value")["log_odds"]
    assert k["d"] > k["c"] > k["a"]
    # every model is trained on the same rows, so noise can get a small in-sample curve: it must stay small
    rng_noise = np.ptp(np.nan_to_num(fitted.shape("noise")["log_odds"].to_numpy()))
    assert rng_noise < 0.25 * np.ptp(lo)


def test_importance_ranks_the_informative_inputs(fitted):
    imp = fitted.importance().set_index("variable")
    assert imp["weight"].sum() == pytest.approx(1.0)
    assert imp.loc["noise", "weight"] < min(imp.loc["u_shape", "weight"], imp.loc["linear", "weight"])
    assert {"weight", "nca_weight", "shaped", "logodds_range", "r2"} <= set(imp.columns)


def test_rashomon_record_uses_only_the_chosen_families(fitted):
    r = fitted.rashomon()
    assert set(r["family"]) <= {"logreg", "spline_logreg"}
    assert r["accepted"].sum() == 10 and (r["competitive"] <= r["accepted"]).all()


def test_neighbours_never_come_from_the_query_patient(fitted, data):
    nb = fitted.neighbours(data.head(50), k=20)
    q_pid = data["pid"].to_numpy()[nb["query"]]
    assert (nb["case_group"].to_numpy() != q_pid).all()
    assert (nb.groupby("query")["case_group"].nunique() == nb.groupby("query").size()).all()   # distinct patients


def test_json_round_trip_gives_the_same_predictions(fitted, data, tmp_path):
    test = make_data(800, seed=5)
    path = tmp_path / "metric.json"
    fitted.save(path)
    m2 = CEKNN.load(path)
    with pytest.raises(RuntimeError):
        m2.predict_proba(test)
    m2.set_case_base(data)
    assert np.allclose(m2.predict_proba(test), fitted.predict_proba(test))
    assert m2.importance()["weight"].tolist() == pytest.approx(fitted.importance()["weight"].tolist())
    a, b = m2.shape("u_shape"), fitted.shape("u_shape")
    assert np.allclose(a["log_odds"], b["log_odds"], atol=1e-5)
    assert len(m2.rashomon()) == len(fitted.rashomon())


def test_the_json_holds_the_metric_and_no_rows(fitted, data):
    d = json.loads(fitted.to_json())
    assert d["format"] == "ceknn-metric" and d["metric"] is not None
    text = json.dumps(d)
    assert "pid" not in json.dumps(d["metric"]) and len(text) < 200_000
    for c in ("u_shape", "linear", "noise"):
        for x in (data[c].min(), data[c].max()):
            assert repr(float(x)) not in text


def test_an_unfitted_model_exports_its_settings(fitted):
    m = CEKNN({"u_shape": "continuous", "kind": "nominal"}, "y", k=25, n_models=12, model_space=_space())
    d = json.loads(m.to_json())
    assert d["metric"] is None and d["settings"]["knn"]["k"] == 25 and d["settings"]["shapes"]["n_models"] == 12
    m2 = CEKNN.from_dict(d)
    assert not m2.is_fitted and m2.knn_settings.k == 25 and set(m2.model_space.families) == {"logreg", "spline_logreg"}


def test_a_transported_metric_votes_with_a_local_case_base(fitted):
    local = make_data(1500, seed=9)
    local["y"] = np.where(np.random.default_rng(0).uniform(size=len(local)) < 0.3, 1, local["y"])   # higher rate
    m = CEKNN.from_json(fitted.to_json()).set_case_base(local)
    test = make_data(800, seed=10)
    p_local, p_dev = m.predict_proba(test)[:, 1].mean(), fitted.predict_proba(test)[:, 1].mean()
    assert p_local > p_dev                                       # the outcomes come from the local case base


def test_effects_transform_carry_and_evaluate(fitted, data):
    test = make_data(300, seed=3)
    eff = fitted.effects(test)
    shaped = set(fitted.importance().query("shaped")["variable"])
    assert set(eff.columns) == shaped and "flag" not in eff and len(eff) == 300
    assert fitted.transform(test).shape == (300, fitted.enc_.n_dims)
    car = fitted.carry(test, ["flag"], k=10)
    assert car["flag"].between(0, 1).all()
    ev = fitted.evaluate(test)
    assert ev["auroc"] > 0.7 and set(ev) == {"n", "log_loss", "brier", "auroc"}


def test_sage_is_normalised_and_saved(fitted):
    s = fitted.compute_sage(n_rows=300, max_passes=1)
    assert s["sage"].sum() == pytest.approx(1.0)
    assert s.set_index("variable").loc["noise", "sage"] < s.set_index("variable").loc["u_shape", "sage"]
    d = json.loads(fitted.to_json())
    assert d["metric"]["sage"]["u_shape"] == pytest.approx(s.set_index("variable").loc["u_shape", "sage"])
    assert "sage" in fitted.importance()


def test_multiclass_and_sites():
    df = make_data(3000, seed=4, n_classes=3, sites=2)
    m = CEKNN(INPUTS, "y", site="site", model_space=_space(), **SMALL).fit(df)
    assert m.schema.classes == [0, 1, 2]
    tab = m.shape("u_shape")
    assert {"g[0]", "g[1]", "g[2]"} <= set(tab.columns)
    p = m.predict_proba(make_data(200, seed=6, n_classes=3, sites=2))
    assert p.shape == (200, 3) and np.allclose(p.sum(1), 1)
    assert m.rashomon_.site_levels == ["s1"]
    assert set(m.info_["learn_per_site"]) == {"s0", "s1"}


def test_learning_sample_caps(data):
    m = CEKNN(INPUTS, "y", group="pid", model_space=_space(), learn={"max_rows": 1000, "rows_per_group": 1},
              **SMALL).fit(data)
    assert m.info_["n_learn"] == 1000 and m.info_["n_rows"] == len(data)
    assert len(m.case_base_) == len(data)


def test_errors_are_clear(data):
    m = CEKNN(INPUTS, "y", model_space=_space())
    with pytest.raises(RuntimeError):
        m.shape("u_shape")
    bad = data.copy()
    bad.loc[0, "y"] = np.nan
    with pytest.raises(ValueError):
        CEKNN(INPUTS, "y", model_space=_space(), **SMALL).fit(bad)
    with pytest.raises(KeyError):
        CEKNN(INPUTS + ["absent"], "y", model_space=_space()).fit(data)
