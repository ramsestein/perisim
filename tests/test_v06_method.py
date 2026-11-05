"""v0.6 method-core fixes (AUDIT 7, 8, 1c shape_tables, low-severity "Método" items)."""

from __future__ import annotations

import numpy as np
import pytest

from perisim.features.build import FeatureSpec
from perisim.features.table import FeatureTable


def _ft(specs, num=None, cat=None, n=None):
    n = n or len(next(iter((num or cat).values())))
    ids = np.arange(n).astype(str).astype(object)
    return FeatureTable(ids, ids, specs, num or {}, cat or {})


# ---------------------------------------------------------------------------- 1. discrete top value
def test_frequent_maximum_gets_its_own_bin():
    from perisim.metric.encoding import Encoder

    rng = np.random.default_rng(0)
    n = 20000
    asa = rng.choice([1, 2, 3, 4, 5], n, p=[.1, .4, .4, .09, .01]).astype(float)          # 200 × ASA 5
    gcs = np.where(rng.uniform(size=n) < .75, 15, rng.integers(3, 15, n)).astype(float)
    specs = [FeatureSpec("asa", "ordinal", "asa", "baseline", levels=[1, 2, 3, 4, 5, 6]),
             FeatureSpec("last_gcs", "continuous", "g", "icu")]
    ft = _ft(specs, {"asa": asa, "last_gcs": gcs})
    enc = Encoder(geometry="shape").fit(ft, rng.integers(0, 2, n), 2)
    for b in enc.blocks:
        bi = enc._bin_index(b, ft)
        x = ft.num[b.source]
        top = x == x.max()
        assert (bi[top] == enc._n_values(b) - 1).all()
        assert not np.isin(bi[~top], bi[top]).any()            # nobody else shares the maximum's bin
        assert b.params["edges"][-1] == b.params["edges"][-2] == x.max()
        assert len(b.params["value_counts"]) == enc._n_values(b)
    a = enc.blocks[0]
    bi = enc._bin_index(a, ft)
    assert len({int(bi[asa == v][0]) for v in (1, 2, 3, 4, 5)}) == 5    # ASA 4 and 5 now differ


def test_lone_maximum_keeps_merged_bin():
    from perisim.metric.encoding import Encoder, top_value_edges

    rng = np.random.default_rng(1)
    n = 5000
    age = np.r_[rng.normal(60, 12, n - 1), 104.5]
    ft = _ft([FeatureSpec("age", "continuous", "age", "demographics")], {"age": age})
    enc = Encoder(geometry="shape").fit(ft, rng.integers(0, 2, n), 2)
    e = enc.blocks[0].params["edges"]
    assert e[-1] == age.max() and e[-2] < e[-1]                   # unchanged: max shares the last quantile bin
    # exact spec: own bin iff #(obs == max) ≥ min_bin
    obs = np.r_[np.arange(100.0), np.full(49, 100.0)]
    edges = np.array([0.0, 50.0, 100.0])
    assert top_value_edges(obs, edges, 50).tolist() == [0.0, 50.0, 100.0]
    obs = np.r_[obs, 100.0]
    assert top_value_edges(obs, edges, 50).tolist() == [0.0, 50.0, 100.0, 100.0]


# ---------------------------------------------------------------------------- 3. shape tables: no raw extremes
def test_shape_tables_do_not_export_min_or_max():
    from perisim.metric.encoding import Encoder, bin_labels

    rng = np.random.default_rng(0)
    n = 20000
    age = np.r_[rng.normal(60, 12, n - 1), 103.7]
    ft = _ft([FeatureSpec("age", "continuous", "age", "demographics")], {"age": age})
    e = Encoder(geometry="shape").fit(ft, rng.integers(0, 2, n), 2)
    V = e._n_values(e.blocks[0])
    g = np.c_[-np.arange(V), np.arange(V)].astype(float)
    st = e.with_shapes({0: g}, ft).shape_tables(10)["age"]
    txt = " ".join(r["value"] for r in st)
    assert f"{age.max():.4g}" not in txt and f"{age.min():.4g}" not in txt
    assert st[0]["value"].startswith("< ") and st[-1]["value"].startswith("≥ ")
    assert bin_labels(np.array([1.0, 2, 3, 4, 5, 5])) == ["< 2", "[2, 3)", "[3, 4)", "[4, 5)", "= 5"]


# ---------------------------------------------------------------------------- 2. flatness test on differences
def _ale(rng, V, C, sd, slope=0.0):
    le = rng.normal(0, sd, (V, C))
    le[:, 1] += slope
    le -= le.mean(1, keepdims=True)
    G = np.vstack([np.zeros((1, C)), np.cumsum(le, 0)])
    return 0.5 * (G[:-1] + G[1:])


def test_flatness_false_positive_rate_on_ale_noise():
    from perisim.importance.shapes import consensus

    rng = np.random.default_rng(0)
    V, C, M, R = 20, 3, 25, 200
    counts = np.full(V, 100.0)
    old = np.mean([consensus([_ale(rng, V, C, .05) for _ in range(M)], counts)["g"] is not None for _ in range(R)])
    new = np.mean([consensus([_ale(rng, V, C, .05) for _ in range(M)], counts, cumulative=True)["g"] is not None
                   for _ in range(R)])
    assert old > 0.25                    # the v0.5 test on the values (documents the defect)
    assert new < 0.15                    # near the nominal level (≈ 0.08–0.10 in 400-rep simulations)
    # a straight line (constant local effect) must still be detected: differences are tested against 0
    pw = np.mean([consensus([_ale(rng, V, C, .05, 0.02) for _ in range(M)], counts, cumulative=True)["g"] is not None
                  for _ in range(50)])
    assert pw > 0.9


def test_flatness_nominal_path_unchanged():
    from perisim.importance.shapes import _flat_pvalue

    rng = np.random.default_rng(2)
    Y = rng.normal(0, 1, (10, 6))
    y, M = Y.mean(0), Y.shape[0]
    s2p = np.mean(Y.var(0, ddof=1) / M)
    from scipy.stats import f as fdist
    ref = fdist.sf(((y - y.mean()) ** 2).sum() / s2p / 5, 5, 6 * (M - 1))
    assert _flat_pvalue(Y, False) == pytest.approx(ref)


def test_consensus_shapes_uses_cumulative_for_continuous(monkeypatch):
    import perisim.importance.shapes as sh

    seen = {}

    def fake(cl, counts, flat_alpha=0.05, cumulative=False):
        seen[len(seen)] = cumulative
        return {"g": None, "n_models": len(cl), "flat": True, "r2": None}

    monkeypatch.setattr(sh, "consensus", fake)
    blocks = [type("B", (), {"kind": k, "name": k, "params": {"value_counts": np.ones(3)}})()
              for k in ("continuous", "nominal", "ordinal", "hier")]
    enc = type("E", (), {"blocks": blocks})()
    sh.consensus_shapes({j: [np.zeros((3, 2))] for j in range(4)}, enc, np.ones(4))
    assert seen == {0: True, 1: False, 2: True, 3: False}


# ---------------------------------------------------------------------------- 4. α when k > case base
def test_alpha_used_when_k_exceeds_case_base():
    from perisim.experiment import knn_predict
    from perisim.knn.predict import KNNModel

    rng = np.random.default_rng(0)
    n = 120
    E = rng.normal(size=(n, 3)).astype(np.float32)
    y = (E[:, 0] + rng.normal(0, 1, n) > 0).astype(int)
    m = KNNModel(E, np.ones(3, np.float32), y, 2, np.arange(n), [10, 50, 200])
    m.fit_alpha_and_ood([0, 0.1, 0.5, 1, 2, 5, 10], 50, 99)
    assert m.k_max == n - 1 and m.alpha[200] != 1.0
    pr, (D, I, v) = knn_predict(m, E[:5] + 0.01, np.arange(1000, 1005), [10, 50, 200])
    np.testing.assert_allclose(pr[200], m.probs(I, v, m.k_max, alpha=m.alpha[200]))
    assert m.alpha_for(m.k_max) == m.alpha[200]
    with pytest.raises(KeyError):
        m.probs(I, v, 7)                 # neither a grid k nor an effective one: never a silent α = 1


# ---------------------------------------------------------------------------- 5. never-seen PCS codes
def test_unseen_pcs_code_parent_pool_or_neutral():
    from perisim.metric.encoding import Encoder

    rng = np.random.default_rng(0)
    n = 3000
    codes = rng.choice(["0DT", "0DB", "0FT"], n).astype(object)
    codes[:30] = "0FZ"                              # rare, pool 0F* too small → other
    codes[30:60], codes[60:90] = "0DX", "0DY"       # rare, pool 0D* = 60 ≥ 50 → pooled
    specs = [FeatureSpec("pcs_l3", "hier", "pcs", "surgery")]
    ft = _ft(specs, cat={"pcs_l3": codes})
    enc = Encoder(geometry="shape").fit(ft, rng.integers(0, 2, n), 2)
    b = enc.blocks[0]
    K = len(b.params["codes"])
    assert "0D*" in b.params["codes"]
    q = _ft(specs, cat={"pcs_l3": np.array(["ZZZ", "0DQ", "0FZ", "0FQ", None], dtype=object)})
    idx = enc.value_index(b, q)
    assert idx.tolist() == [-1, b.params["codes"].index("0D*"), K, -1, K + 1]
    g = np.arange(K + 2)[:, None] * np.array([[-1.0, 1.0]]) + 1.0
    e2 = enc.with_shapes({0: g}, ft)
    Z = e2.transform(q)
    assert np.all(Z[0] == 0) and np.all(Z[3] == 0)               # never seen, no parent pool → neutral 0
    np.testing.assert_allclose(Z[1], g[b.params["codes"].index("0D*")] / e2.blocks[0].scale, rtol=1e-6)
    # model / neutral one-hot matrices: unseen without pool → "other" column, unseen with pool → the pool
    X = enc._hier_onehot(b, q) * np.sqrt(2.0)
    assert X[0, K] == 1 and X[1, b.params["codes"].index("0D*")] == 1


# ---------------------------------------------------------------------------- 6. inner T2 cut-points
def test_inner_labels_refit_cut_points(cfg):
    from perisim.experiment import CohortData, _fold_labeler, _inner_labels, labels_for

    rng = np.random.default_rng(4)
    n = 300
    los = rng.exponential(5, n)
    D = CohortData("x", type("F", (), {"__len__": lambda self: n})(), np.zeros(n, int), np.zeros(n, int), los,
                   {}, np.zeros(n), np.zeros(n))
    tr = np.arange(0, n, 2)
    lab = _fold_labeler("t2", D, tr, cfg.targets)
    assert _fold_labeler("t1", D, tr, cfg.targets) is None
    a, b = np.arange(0, 100), np.arange(100, 150)
    y_outer = labels_for("t2", D, tr, cfg.targets)[0][tr]
    yi, a2, b2 = _inner_labels(y_outer, a, b, lab)
    thr = labels_for("t2", D, tr[a], cfg.targets)[1]["t2_thresholds"]
    assert thr != labels_for("t2", D, tr, cfg.targets)[1]["t2_thresholds"]
    # the inner validation part's own LOS no longer moves any label
    los2 = los.copy()
    los2[tr[b]] = 1000.0
    D2 = CohortData("x", D.ft, D.t1, D.t1_severe, los2, {}, D.t0, D.year)
    yi2, _, _ = _inner_labels(y_outer, a, b, _fold_labeler("t2", D2, tr, cfg.targets))
    assert (yi2[a] == yi[a]).all()
    assert np.array_equal(a2, a) and np.array_equal(b2, b)


def test_select_lambda_scores_with_inner_labels(monkeypatch):
    import perisim.experiment as ex

    calls = []

    class FakeNCA:
        def fit(self, Ea, y, **kw):
            calls.append(("fit", y.copy()))

        def soft_loss(self, Ea, ya, Eb, yb, **kw):
            calls.append(("score", yb.copy()))
            return 0.0

    monkeypatch.setattr(ex, "_nca", lambda *a, **k: FakeNCA())
    cfg = type("C", (), {"experiment": {"metric": {"lambda_prior": "outer", "lambda_grid": [0.0]}}})()
    n = 20
    ft = type("F", (), {"take": lambda self, idx: idx, "subjects": np.arange(n)})()
    enc = type("E", (), {"transform": lambda self, x: np.zeros((len(x), 1))})()
    y = np.zeros(n, dtype=np.int64)
    inner = [(np.arange(10), np.arange(10, 20))]
    ex.select_lambda(cfg, ft, y, 2, inner, 5.0, seed=0, epochs=1, threads=1, outer_prior=(enc, None, None),
                     labeler=lambda a: np.full(n, 1, dtype=np.int64))
    assert all((v == 1).all() for _, v in calls) and len(calls) == 2


# ---------------------------------------------------------------------------- 7. LightGBM determinism
def test_lgbm_deterministic_flags():
    from perisim.importance.surrogate import Surrogate

    p = {"n_estimators": 5, "learning_rate": 0.1, "num_leaves": 4, "min_child_samples": 5, "subsample": 0.8,
         "colsample_bytree": 0.8, "reg_lambda": 1.0}
    m = Surrogate("lightgbm", p, np.arange(3), 3, 2, seed=0, threads=2)._make(2)
    params = m.get_params()
    assert params.get("deterministic") is True and params.get("force_col_wise") is True
