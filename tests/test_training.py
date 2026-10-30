"""The vote of the neighbours per class, the stopping rules, the k grid and the training report."""

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
    return make_data(3000, seed=21, rows_per_patient=2)


@pytest.fixture(scope="module")
def stab(data):
    shapes = {**SMALL["shapes"], "stopping": "stability", "n_min": 4, "n_max": 12, "window": 2, "patience": 2,
              "sage_in_loop": True, "sage": {"batch_size": 256, "max_passes": 1, "threshold": 0.05}}
    return CEKNN(INPUTS, "y", group="pid", model_space=_space(),
                 **{**SMALL, "shapes": shapes, "knn": {**SMALL["knn"], "k_grid": [5, 10, 30, 60], "select_k": True}}
                 ).fit(data)


def test_the_vote_gives_count_share_and_probability_of_each_class(stab):
    test = make_data(200, seed=22)
    v = stab.vote(test)
    for c in (0, 1):
        assert {f"n[{c}]", f"share[{c}]", f"p[{c}]"} <= set(v.columns)
    assert (v["n[0]"] + v["n[1]"] == v["n_neighbours"]).all()
    assert np.allclose(v["share[1]"], v["n[1]"] / v["n_neighbours"])
    assert np.allclose(v[["p[0]", "p[1]"]].to_numpy(), stab.predict_proba(test))
    assert (v["k"] == stab.case_base_info()["k"]).all() and v["ood"].dtype == bool
    p = stab.predict_proba(test, as_frame=True)
    assert list(p.columns) == [0, 1] and p.index.equals(test.index)


def test_the_vote_with_another_k(stab):
    test = make_data(50, seed=23)
    v = stab.vote(test, k=7)
    assert (v["n_neighbours"] <= 7).all() and (v["k"] == 7).all()


def test_stability_stopping_is_recorded(stab):
    st = stab.stopping()
    assert st["rule"] == "stability" and st["reason"] in ("patience", "n_max", "max_attempts")
    assert 4 <= st["n_accepted"] <= 12 and st["min_delta_used"] is not None
    tr = stab.stability()
    assert len(tr) == st["n_accepted"]
    assert {"delta", "delta_smooth", "spearman_range", "spearman_sage", "range[u_shape]", "sage[u_shape]"} <= set(tr)
    assert tr["delta_smooth"].notna().all()


def test_sage_in_the_loop_feeds_the_importance(stab):
    imp = stab.importance().set_index("variable")
    assert imp["sage"].sum() == pytest.approx(1.0)
    assert imp.loc["u_shape", "sage"] > imp.loc["noise", "sage"]


def test_k_is_chosen_by_leave_one_out(stab):
    cur = stab.knn_curve()
    assert set(cur["k"]) == {5, 10, 30, 60} and len(cur) == 4 * len(stab.knn_settings.alpha_grid)
    best = cur.loc[cur["loo_logloss"].idxmin()]
    info = stab.case_base_info()
    assert info["k_selected"] and info["k"] == best["k"] and cur["used"].sum() == 1


def test_settings_and_report_show_everything(stab):
    s = stab.settings()
    assert {"encoding", "shapes", "nca", "knn", "learn"} <= set(s["section"])
    assert (s["meaning"] != "").all()
    assert s.set_index(["section", "setting"]).loc[("shapes", "stopping"), "value"] == "stability"
    rep = stab.training_report()
    assert {"settings", "stopping", "stability", "rashomon", "nca_loss", "knn", "case_base"} <= set(rep)
    assert len(rep["nca_loss"]) == stab.nca_settings.epochs
    text = stab.summary()
    assert "stability" in text and "k = " in text and "SAGE" in text


def test_the_training_record_travels_in_the_json(stab):
    m = CEKNN.from_json(stab.to_json())
    a, b = m.stopping(), stab.stopping()
    assert (a["rule"], a["reason"], a["n_accepted"]) == (b["rule"], b["reason"], b["n_accepted"])
    assert len(m.stability()) == len(stab.stability())
    cur = m.knn_curve()                                  # no case base yet: the curve of the fit
    assert len(cur) == len(stab.knn_curve()) and cur["used"].sum() == 1
    assert m.importance()["sage"].notna().all()
    with pytest.raises(RuntimeError):
        m.case_base_info()


def test_fixed_stopping_is_the_default(data):
    m = CEKNN(INPUTS, "y", group="pid", model_space=_space(), **SMALL).fit(data)
    st = m.stopping()
    assert st["rule"] == "fixed" and st["n_accepted"] == SMALL["n_models"] and st["reason"] == "n_models"
    assert "spearman_sage" not in m.stability()
    with pytest.raises(ValueError):
        CEKNN(INPUTS, "y", shapes={"stopping": "never"})


def test_carry_gives_category_shares(stab, data):
    df = data.copy()
    df["unit"] = np.where(df["flag"] > 0, "icu", "ward")
    m = CEKNN.from_json(stab.to_json()).set_case_base(df)
    car = m.carry(make_data(100, seed=24), ["unit", "lab"], k=10)
    assert {"unit[icu]", "unit[ward]", "lab"} <= set(car.columns)
    assert np.allclose(car["unit[icu]"] + car["unit[ward]"], 1.0)


def test_the_plots_draw(stab):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    stab.plot_shape("u_shape")
    stab.plot_shape("kind")
    stab.plot_stability()
    stab.plot_knn()
    plt.close("all")
