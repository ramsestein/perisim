"""§5.1–§5.2 — El embedding reproduce exactamente d² = Σ_j w_j δ_j² y la normalización a δ² medio unitario."""

from __future__ import annotations

import numpy as np

from perisim.features.build import FeatureSpec
from perisim.features.table import FeatureTable
from perisim.metric.encoding import Encoder
from perisim.metric.nca import DiagonalNCA, pair_sq_diffs


def _ft(n, rng):
    specs = [FeatureSpec("age", "continuous", "age", "demografia"),
             FeatureSpec("asa", "ordinal", "asa", "basal", levels=[1, 2, 3, 4, 5, 6]),
             FeatureSpec("vent", "binary", "vent", "uci"),
             FeatureSpec("grp", "nominal", "grp", "cirugia"),
             FeatureSpec("pcs_l1", "hier", "pcs", "cirugia"), FeatureSpec("pcs_l2", "hier", "pcs", "cirugia")]
    age = rng.normal(60, 15, n)
    age[rng.random(n) < 0.2] = np.nan
    num = {"age": age, "asa": rng.integers(1, 5, n).astype(float), "vent": (rng.random(n) < 0.3).astype(float)}
    l2 = rng.choice(["0D", "0S", "B2"], n)
    cat = {"grp": rng.choice(["g1", "g2", "g3", None], n).astype(object),
           "pcs_l1": np.array([x[0] for x in l2], dtype=object), "pcs_l2": l2.astype(object)}
    ids = np.arange(n).astype(str).astype(object)
    return FeatureTable(ids, ids, specs, num, cat)


def test_embedding_matches_weighted_formula():
    rng = np.random.default_rng(0)
    ft = _ft(500, rng)
    y = rng.integers(0, 3, 500)
    enc = Encoder(m=5, nominal_min_count=1, hier_min_count=1).fit(ft, y, 3)
    E = enc.transform(ft).astype(np.float64)
    w = rng.uniform(0.1, 3, enc.n_features)
    pairs = rng.integers(0, 500, (300, 2))
    Dsq = pair_sq_diffs(E, enc.dim_feature, enc.n_features, pairs)
    d2_formula = Dsq @ w
    Ew = E * np.sqrt(w)[enc.dim_feature]
    d2_embed = ((Ew[pairs[:, 0]] - Ew[pairs[:, 1]]) ** 2).sum(1)
    np.testing.assert_allclose(d2_formula, d2_embed, rtol=1e-6, atol=1e-8)
    # δ_j² medio entre pares aleatorios ≈ 1 tras la normalización
    big = rng.integers(0, 500, (20000, 2))
    big = big[big[:, 0] != big[:, 1]]
    m = pair_sq_diffs(E, enc.dim_feature, enc.n_features, big).mean(0)
    np.testing.assert_allclose(m, 1.0, rtol=0.1)
    # indicador de ausencia de la edad (20 % > 5 %) y un nivel PCS por dimensión
    assert "age__missing" in enc.feature_names
    assert {"pcs_l1", "pcs_l2"} <= set(enc.feature_names)


def test_nca_learns_informative_weight():
    rng = np.random.default_rng(1)
    n = 600
    from perisim.features.build import FeatureSpec as FS

    x_inf = rng.normal(size=n)
    y = (x_inf > 0).astype(int)
    specs = [FS("inf", "continuous", "inf", "uci")] + [FS(f"noise{i}", "continuous", f"noise{i}", "uci") for i in range(5)]
    num = {"inf": x_inf, **{f"noise{i}": rng.normal(size=n) for i in range(5)}}
    ids = np.arange(n).astype(str).astype(object)
    ft = FeatureTable(ids, ids, specs, num, {})
    enc = Encoder().fit(ft, y, 2)
    E = enc.transform(ft)
    nca = DiagonalNCA(enc.dim_feature, enc.n_features, 2, n_candidates=60, batch_size=128, lr=0.1, seed=0)
    nca.fit(E, y, epochs=8, subjects=ids)
    w = nca.weights
    assert w[enc.feature_names.index("inf")] > 3 * np.median(w[1:])
