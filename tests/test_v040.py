"""pmknn 0.4.0 (PERISIM v0.6): optional group / site columns outside fitting, no individual value in the exported
JSON, NCA with a single site, the discrete top value and the flatness test on local effects."""

import json

import numpy as np
import pandas as pd
import pytest

from pmknn import PMKNN
from pmknn.core import _public_count, _public_training
from pmknn.encoding import Encoder, _ecdf_knots
from pmknn.nca import DiagonalNCA
from pmknn.schema import Schema, Table
from pmknn.shapes import consensus
from synthetic import SMALL, make_data, small_space

INPUTS = {"u_shape": "continuous", "linear": "continuous", "noise": "continuous", "kind": "nominal",
          "flag": "binary", "lab": "continuous"}


@pytest.fixture(scope="module")
def fitted():
    df = make_data(3000, seed=31, rows_per_patient=3, sites=2)
    m = PMKNN(INPUTS, "y", group="pid", site="site", model_space=small_space(), seed=0, **SMALL).fit(df)
    return m, df


# ------------------------------------------------------------------ 1. group / site optional outside fitting
def test_readme_flow_without_group_and_site(fitted, tmp_path):
    m, _ = fitted
    path = tmp_path / "metric.json"
    m.save(path)
    local = make_data(1500, seed=32).drop(columns=["site"])
    m2 = PMKNN.load(path).set_case_base(local)
    q = make_data(200, seed=33).drop(columns=["pid", "site"])
    p = m2.predict_proba(q)
    assert p.shape == (200, 2) and np.allclose(p.sum(1), 1)
    assert len(m2.vote(q)) == 200 and len(m2.neighbours(q, k=5)) > 0
    m2.out_of_support(q)


def test_fit_still_requires_group_and_site():
    df = make_data(600, seed=34, sites=2)
    with pytest.raises(KeyError):
        PMKNN(INPUTS, "y", group="pid", site="site", model_space=small_space(), **SMALL).fit(df.drop(columns="site"))


def test_query_without_group_shares_no_patient(fitted):
    m, df = fitted
    q = df.head(50)
    with_g = m.neighbours(q, k=5)
    assert not (with_g["case_group"].to_numpy() == q["pid"].to_numpy()[with_g["query"]]).any()
    without = m.neighbours(q.drop(columns=["pid"]), k=5)
    # without the group column the query's own row (distance 0) may vote: it is not known to be the same patient
    first = without[without["rank"] == 1]
    assert (first["distance"] < 1e-6).mean() > 0.9


def test_case_base_without_group_column(fitted):
    m, df = fitted
    m2 = PMKNN.from_dict(m.to_dict()).set_case_base(df.drop(columns=["pid", "site"]))
    assert m2.case_base_.distinct is False
    q = df.head(20)                        # queries with a group column: the placeholder groups never match
    nb = m2.neighbours(q, k=3)
    assert (nb[nb["rank"] == 1]["distance"] < 1e-6).all()


# ------------------------------------------------------------------ 2. no individual value in the JSON
def _enc(df, inputs, min_bin=50):
    sch = Schema.build(inputs, "y")
    sch.resolve(df)
    return Encoder(min_bin=min_bin).fit(Table.from_frame(df, sch), sch.labels(df), 2)


def test_ordinal_without_levels_exports_no_min_max():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({"score": rng.integers(0, 30, n).astype(float), "y": rng.integers(0, 2, n)})
    df.loc[0, "score"] = 987.0
    b = _enc(df, {"score": "ordinal"}).to_dict()["blocks"][0]["params"]
    assert b["lo"] not in (0.0, 987.0) and b["hi"] not in (0.0, 987.0)
    assert 987.0 not in [x for x in b["edges"] if x is not None]
    assert 987.0 not in b["ecdf"][0]


def test_ecdf_knots_backed_by_min_bin_rows():
    rng = np.random.default_rng(1)
    for n in (40, 100, 101, 149, 150, 151, 299, 5000):
        x = rng.normal(size=n)
        r = _ecdf_knots(x, 50, 100)
        if n <= 100:
            assert r is None
            continue
        knots, levels = r
        assert len(knots) >= 2
        for kv in knots:
            assert (x < kv).sum() >= 49 and (x > kv).sum() >= 49
    # a small sample: the input is left out rather than exported with observed values
    x = np.array([3.1, 7.7, 12.9, 55.5, 91.2, 4.4, 8.8, 13.3, 60.1, 99.9, 1.0, 2.0, 30.0])
    df = pd.DataFrame({"v": x, "y": [0, 1] * 6 + [0]})
    assert _enc(df, {"v": "continuous"}).blocks == []


def test_public_counts():
    assert [_public_count(c) for c in (0, 1, 9, 10, 14, 15, 1234)] == [0, None, None, 10, 10, 20, 1230]
    info = {"n_rows": 1003, "n_learn": 1003, "class_shares": [0.997, 0.003], "class_counts": [1000, 3],
            "learn_per_site": {"A": 998, "B": 5},
            "case_base_at_fit": {"n_rows": 1003, "n_queries": 1003, "class_shares": [0.997, 0.003],
                                 "class_counts": [1000, 3]}}
    out = _public_training(info)
    assert out["n_rows"] == 1000 and out["learn_per_site"] == {"A": 1000, "B": None}
    assert out["class_shares"] == [1.0, None] and "class_counts" not in out
    cb = out["case_base_at_fit"]
    assert cb["class_shares"] == [1.0, None] and cb["n_rows"] == 1000 and "class_counts" not in cb
    assert _public_training(out) == out                       # idempotent on an exported record
    assert info["class_counts"] == [1000, 3]                  # the in-memory record keeps exact values


def test_exported_training_record_is_rounded(fitted):
    m, _ = fitted
    tr = json.loads(m.to_json())["metric"]["training"]
    counts = [tr["n_rows"], tr["n_learn"], *tr["learn_per_site"].values(), tr["case_base_at_fit"]["n_rows"]]
    assert all(c is None or c % 10 == 0 for c in counts)
    assert "class_counts" not in tr and "class_counts" not in tr["case_base_at_fit"]
    assert m.info_["n_rows"] == 3000                          # exact in memory


# ------------------------------------------------------------------ 3. NCA with a single site
@pytest.mark.parametrize("n", [50, 300])
def test_nca_single_site_small_n(n):
    rng = np.random.default_rng(0)
    E = rng.normal(size=(n, 3)).astype(np.float32)
    y = (E[:, 0] > 0).astype(int)
    nca = DiagonalNCA(np.arange(3), 3, 2, n_candidates=500).fit(E, y, epochs=1, groups=np.arange(n).astype(str),
                                                                 sites=np.array(["A"] * n, dtype=object))
    assert np.isfinite(nca.weights).all()


def test_fit_single_site_column():
    df = make_data(400, seed=35)            # one site level, n below n_candidates
    m = PMKNN(INPUTS, "y", group="pid", site="site", model_space=small_space(),
              **{**SMALL, "nca": {"epochs": 1, "n_candidates": 500}}).fit(df)
    assert m.is_fitted


# ------------------------------------------------------------------ 4a. discrete top value
def _asa(n_top, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.choice([1.0, 2.0, 3.0, 4.0], size=n, p=[0.1, 0.4, 0.35, 0.15])
    v[:n_top] = 5.0
    return pd.DataFrame({"asa": v, "y": rng.integers(0, 2, n)})


def test_top_value_gets_its_own_bin():
    df = _asa(60)
    enc = _enc(df, {"asa": "ordinal"})
    b = enc.blocks[0]
    t = Table.from_frame(df, Schema.build({"asa": "ordinal"}, "y"))
    idx = enc.value_index(b, t)
    x = df["asa"].to_numpy()
    assert np.unique(idx[x == 5.0]).size == 1 and np.unique(idx[x == 4.0]).size == 1
    assert idx[x == 5.0][0] != idx[x == 4.0][0]
    assert idx[x == 5.0][0] == enc.n_values(b) - 1
    # values above the training maximum stay in the top bin; the exported metric bins the same way
    enc2 = Encoder.from_dict(json.loads(json.dumps(enc.to_dict())))
    probe = pd.DataFrame({"asa": [1.0, 4.0, 4.5, 5.0, 6.0], "y": 0})
    tp = Table.from_frame(probe, Schema.build({"asa": "ordinal"}, "y"))
    assert (enc.value_index(b, tp) == enc2.value_index(enc2.blocks[0], tp)).all()
    assert enc.value_labels(b)[-1] == "≥ 5"


def test_lone_top_value_is_merged():
    df = _asa(3)
    enc = _enc(df, {"asa": "ordinal"})
    b = enc.blocks[0]
    idx = enc.value_index(b, Table.from_frame(df, Schema.build({"asa": "ordinal"}, "y")))
    x = df["asa"].to_numpy()
    assert idx[x == 5.0][0] == idx[x == 4.0][0]


def test_bin_labels_never_show_min_max(fitted):
    m, df = fitted
    tab = m.shape("u_shape")
    assert tab["value"].iloc[0].startswith("< ") and tab["value"].iloc[-1].startswith("≥ ")
    assert pd.isna(tab["lo"].iloc[0]) and pd.isna(tab["hi"].iloc[-1])
    lo, hi = df["u_shape"].min(), df["u_shape"].max()
    assert f"{lo:.4g}" not in " ".join(tab["value"]) and f"{hi:.4g}" not in " ".join(tab["value"])


# ------------------------------------------------------------------ 4b. flatness test on local effects
def _ale(le):
    G = np.vstack([np.zeros((1, le.shape[1])), np.cumsum(le, 0)])
    return 0.5 * (G[:-1] + G[1:])


def _rate(slope, local, reps=300, V=20, M=12, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        cl = []
        for _ in range(M):
            c = _ale(slope + rng.normal(0, 1, (V, 1)))
            cl.append(np.hstack([-c / 2, c / 2]))
        hits += consensus(cl, np.full(V, 100.0), local=local)["g"] is not None
    return hits / reps


def test_flatness_test_on_local_effects():
    assert _rate(0.0, local=False) > 0.12          # the old test: accumulated noise looks like a shape
    assert _rate(0.0, local=True) < 0.08           # the new one holds its level
    assert _rate(0.6, local=True, reps=100) > 0.95 # and still finds a straight line (not centred)
