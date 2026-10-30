"""The pieces: schema, encoding, consensus, NCA gradient, model space."""

import numpy as np
import pandas as pd
import pytest

from ceknn import ModelSpace
from ceknn.encoding import OTHER, Encoder
from ceknn.nca import DiagonalNCA
from ceknn.schema import Schema, Table, infer_type
from ceknn.shapes import consensus
from synthetic import make_data


def _table(df, inputs, output="y", **kw):
    sch = Schema.build(inputs, output, **kw)
    sch.resolve(df)
    return sch, Table.from_frame(df, sch), sch.labels(df)


def test_types_are_inferred_and_checked():
    df = make_data(200)
    assert infer_type(df["u_shape"]) == "continuous"
    assert infer_type(df["flag"]) == "binary"
    assert infer_type(df["kind"]) == "nominal"
    sch, _, _ = _table(df, ["u_shape", "flag", "kind"])
    assert [v.type for v in sch.inputs] == ["continuous", "binary", "nominal"] and sch.classes == [0, 1]
    with pytest.raises(ValueError):
        Schema.build(["y"], "y")
    with pytest.raises(KeyError):
        _table(df, ["nope"])


def test_ordinal_labels_follow_their_levels():
    df = pd.DataFrame({"sev": ["low", "high", "mid", None] * 50, "y": [0, 1] * 100})
    sch, t, _ = _table(df, {"sev": ("ordinal", ["low", "mid", "high"])})
    assert np.array_equal(t.num["sev"][:4], [0.0, 2.0, 1.0, np.nan], equal_nan=True)
    with pytest.raises(ValueError):
        _table(pd.DataFrame({"sev": ["odd"] * 4, "y": [0, 1, 0, 1]}), {"sev": ("ordinal", ["low", "high"])})


def test_encoder_bins_indicators_and_other():
    df = make_data(2000)
    df.loc[:30, "kind"] = "rare"
    _, t, y = _table(df, ["u_shape", "lab", "kind", "flag"])
    enc = Encoder(min_bin=50).fit(t, y, 2)
    names = enc.feature_names
    assert "lab__missing" in names                                  # 20 % missing → indicator
    b = enc.blocks[names.index("u_shape")]
    assert b.params["value_counts"].min() >= 50 and len(b.params["value_counts"]) == 20
    k = enc.blocks[names.index("kind")]
    assert OTHER in k.params["values"] and "rare" not in k.params["keep"]
    E = enc.transform(t)
    for j in range(enc.n_features):                                 # unit mean square between random pairs
        cols = E[:, enc.dim_feature == j].astype(float)
        assert 2 * cols.var(0).sum() == pytest.approx(1.0, rel=1e-3)


def test_encoder_export_holds_no_individual_value_and_round_trips():
    df = make_data(2000)
    _, t, y = _table(df, ["u_shape", "kind", "lab"])
    enc = Encoder().fit(t, y, 2)
    d = enc.to_dict()
    b = next(x for x in d["blocks"] if x["name"] == "u_shape")
    assert b["params"]["edges"][0] is None and b["params"]["edges"][-1] is None
    lo, hi = float(np.nanmin(df["u_shape"])), float(np.nanmax(df["u_shape"]))
    flat = str(d)
    assert repr(lo) not in flat and repr(hi) not in flat
    assert np.allclose(Encoder.from_dict(d).transform(t), enc.transform(t))


def test_consensus_keeps_a_real_shape_and_drops_noise():
    rng = np.random.default_rng(0)
    V, M = 10, 30
    real = np.linspace(-1, 1, V)[:, None] * np.array([[-0.5, 0.5]])
    shaped = [real * rng.uniform(0.7, 1.3) + rng.normal(0, 0.05, (V, 2)) for _ in range(M)]
    noise = [rng.normal(0, 0.05, (V, 1)) * np.array([[-1, 1]]) for _ in range(M)]
    counts = np.full(V, 100.0)
    r = consensus(shaped, counts)
    assert r["g"] is not None and np.corrcoef(r["g"][:, 1], real[:, 1])[0, 1] > 0.99
    assert consensus(noise, counts)["g"] is None


def test_nca_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    dim = np.array([0, 1, 1, 2])
    nca = DiagonalNCA(dim, 3, 2, label_smoothing=0.01)
    B, K = 6, 9
    E = rng.normal(size=(B + B * K, 4))
    anchors = np.arange(B)
    cand = (B + np.arange(B * K)).reshape(B, K)
    valid = rng.uniform(size=(B, K)) > 0.2
    y = rng.integers(0, 2, size=E.shape[0])
    same = (y[cand] == y[anchors][:, None]) & valid
    prior = np.full(B, 0.5)
    Delta = nca._delta(E, anchors, cand)
    w = np.array([0.7, 1.3, 0.4])
    _, g = nca._loss_grad(Delta, same, valid, prior, w)
    num = np.zeros(3)
    for j in range(3):
        h = np.zeros(3)
        h[j] = 1e-6
        num[j] = (nca._loss_grad(Delta, same, valid, prior, w + h)[0]
                  - nca._loss_grad(Delta, same, valid, prior, w - h)[0]) / 2e-6
    assert np.allclose(g, num, rtol=1e-4, atol=1e-7)


def test_nca_weights_the_informative_variable():
    rng = np.random.default_rng(2)
    n = 1500
    x = rng.normal(size=n)
    E = np.column_stack([x, rng.normal(size=n), rng.normal(size=n)]).astype(np.float32)
    y = (x + rng.normal(0, 0.5, n) > 0).astype(int)
    w = DiagonalNCA(np.arange(3), 3, 2, n_candidates=60, seed=0).fit(E, y, epochs=6).weights
    assert w[0] > 3 * w[1:].max()


def test_model_space_selection_and_custom_families():
    sp = ModelSpace.default(include=["logreg", "random_forest"])
    assert set(sp.families) == {"logreg", "random_forest"}
    with pytest.raises(ValueError):
        ModelSpace.default(include=["svm"])
    with pytest.raises(ValueError):
        ModelSpace({"svm": {"params": {}}})
    from sklearn.tree import DecisionTreeClassifier

    sp.add("tree", lambda p, seed, n_jobs: DecisionTreeClassifier(max_depth=p["depth"], random_state=seed),
           params={"depth": {"int": [2, 5]}}, handles_nan=True)
    fams = {sp.sample(np.random.default_rng(i))[0] for i in range(40)}
    assert fams == {"logreg", "random_forest", "tree"}
    d = sp.to_dict()
    assert d["tree"]["custom"] and set(ModelSpace.from_dict(d).families) == {"logreg", "random_forest"}
