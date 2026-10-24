"""Methodology v0.5 (decided 2026-10-24): T2 with death, per-variable consensus shapes (ALE, consensus fit,
meta-analytic shrinkage, neutral axis), resumable consensus loop, experiments per target, trivial comparators."""

from __future__ import annotations

import numpy as np
import pytest

from perisim.features.build import FeatureSpec
from perisim.features.table import FeatureTable


# --------------------------------------------------------------------------- T2 with death
def test_t2_has_death_as_worst_class():
    from perisim.targets.build import t2_categories

    los = np.array([1.0, 5.0, 12.0, np.nan, np.nan])
    death = np.array([0, 0, 0, 1, -1])
    out = t2_categories(los, (3.0, 7.0), death)
    assert out.tolist() == [0, 1, 2, 3, -1]


# --------------------------------------------------------------------------- shapes
def _ft(n, rng):
    x = rng.uniform(0, 10, n)
    g = rng.choice(["a", "b", "c"], n).astype(object)
    specs = [FeatureSpec("x", "continuous", "x", "icu"), FeatureSpec("noise", "continuous", "noise", "icu"),
             FeatureSpec("grp", "nominal", "grp", "surgery")]
    ids = np.arange(n).astype(str).astype(object)
    return FeatureTable(ids, ids, specs, {"x": x, "noise": rng.normal(size=n)}, {"grp": g})


class _Known:
    """A 'model' whose class-1 log-odds is a known U-shaped function of x plus a category effect."""

    def __init__(self, enc, owner):
        self.enc, self.owner = enc, owner

    def predict_proba(self, Xm):
        bx = self.enc.blocks[self.enc.feature_index()["x"]]
        cx = np.where(self.owner == self.enc.feature_index()["x"])[0][0]
        x = Xm[:, cx] * bx.params["sd"] + bx.params["mean"]
        bg = self.enc.blocks[self.enc.feature_index()["grp"]]
        cols = np.where(self.owner == self.enc.feature_index()["grp"])[0]
        is_c = Xm[:, cols][:, bg.params["onehot"].index("c")] > 0
        z = 0.15 * (x - 5) ** 2 + 1.0 * is_c - 2
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


def test_ale_reads_the_shape_and_category_effects():
    from perisim.importance.shapes import model_curves
    from perisim.metric.encoding import Encoder

    rng = np.random.default_rng(0)
    ft = _ft(3000, rng)
    y = rng.integers(0, 2, 3000)
    enc = Encoder(geometry="shape", max_bins=20, min_bin=50).fit(ft, y, 2)
    Xm, owner = enc.transform_model(ft)
    cur = model_curves(_Known(enc, owner), Xm, owner, enc, ft, enc.shapeable_blocks())
    jx = enc.feature_index()["x"]
    vals = cur[jx][0][:, 1]
    e = enc.blocks[jx].params["edges"]
    mids = 0.5 * (e[:-1] + e[1:])
    true = 0.15 * (mids - 5) ** 2
    # centred log-odds of class 1 = half the logit change: a U shape, recovered up to that factor
    assert np.corrcoef(vals, true)[0, 1] > 0.97
    jg = enc.feature_index()["grp"]
    gv = cur[jg][0][:, 1]
    vals_g = enc.blocks[jg].params["values"]
    assert gv[vals_g.index("c")] - gv[vals_g.index("a")] == pytest.approx(0.5, abs=0.05)


def test_consensus_keeps_real_shapes_and_flattens_noise():
    from perisim.importance.shapes import consensus

    rng = np.random.default_rng(1)
    V = 15
    true = np.linspace(-1, 1, V) ** 2
    true = np.column_stack([-true / 2, true / 2])
    counts = np.full(V, 100.0)
    real = [rng.uniform(0.5, 1.5) * true + rng.normal(0, 0.05, true.shape) for _ in range(20)]
    r = consensus(real, counts)
    assert r["g"] is not None and r["r2"] > 0.8
    assert np.corrcoef(r["g"][:, 1], true[:, 1])[0, 1] > 0.99
    noise = [rng.normal(0, 0.05, true.shape) for _ in range(20)]
    rn = consensus(noise, counts)
    assert rn["g"] is None                              # τ² = 0 → not distinguishable from flat → neutral axis


def test_encoder_shapes_lookup_and_neutral_axes():
    from perisim.metric.encoding import Encoder

    rng = np.random.default_rng(2)
    ft = _ft(2000, rng)
    y = rng.integers(0, 2, 2000)
    enc = Encoder(geometry="shape", max_bins=10, min_bin=50).fit(ft, y, 2)
    jx, jg = enc.feature_index()["x"], enc.feature_index()["grp"]
    Vx = enc._n_values(enc.blocks[jx])
    shapes = {jx: np.column_stack([-np.arange(Vx), np.arange(Vx)]).astype(float), jg: np.array([[0, 1.0], [0, -1.0],
                                                                                                  [0, 0.5]])}
    e2 = enc.with_shapes(shapes, ft)
    assert e2.feature_names == enc.feature_names and e2.n_features == enc.n_features
    q = FeatureTable(np.array(["q1", "q2"], dtype=object), np.array(["q1", "q2"], dtype=object), ft.specs,
                     {"x": np.array([-50.0, 1e6]), "noise": np.array([0.0, 0.0])},
                     {"grp": np.array(["never_seen", "a"], dtype=object)})
    E = e2.transform(q)
    cols = np.where(e2.dim_feature == jx)[0]
    s = e2.blocks[jx].scale
    np.testing.assert_allclose(E[0, cols] * s, shapes[jx][0])        # below the range → first bin (constant)
    np.testing.assert_allclose(E[1, cols] * s, shapes[jx][-1])       # above the range → last bin
    gcols = np.where(e2.dim_feature == jg)[0]
    np.testing.assert_allclose(E[0, gcols], 0.0)                     # unseen category → neutral contribution
    # the neutral embedding has no outcome information: rank in [0, 1] for continuous variables
    N = enc.transform_neutral(ft)
    ncol = np.where(enc.neutral_dim_feature() == jx)[0][0]
    r = N[:, ncol] * enc.blocks[jx].neutral_scale
    assert r.min() >= 0 and r.max() <= 1 and np.corrcoef(r, ft.num["x"])[0, 1] > 0.99


# --------------------------------------------------------------------------- resumable loop
def _loop_inputs(cfg):
    from perisim.experiment import inner_splits
    from perisim.metric.encoding import Encoder, build_groups

    rng = np.random.default_rng(3)
    n = 400
    ft = _ft(n, rng)
    y = (ft.num["x"] + rng.normal(0, 2, n) > 5).astype(int)
    enc = Encoder(geometry="shape", max_bins=6, min_bin=30).fit(ft, y, 2)
    Xm, owner = enc.transform_model(ft)
    gof, gn = build_groups(enc, enc.transform(ft), 0.8, 50)
    a, b = inner_splits(y, ft.subjects, 2, 0)[0]
    return dict(X_model=Xm, owner=owner, col_group=gof[owner], n_groups=len(gn), group_names=gn, y=y, n_classes=2,
                inner_train=a, inner_val=b, enc=enc, ft=ft, group_of_feature=gof, subjects=ft.subjects, cfg=cfg,
                seed=5, threads=1, stopping_override={"n_max": 4, "n_min": 2, "min_delta": None}, early_stop=False)


def test_loop_checkpoint_resumes_identically(cfg, tmp_path, monkeypatch):
    import copy

    from perisim.importance import loop as L
    from perisim.importance.surrogate import Surrogate

    c = copy.deepcopy(cfg)
    c.model_space = {"families": {"logreg": {"weight": 1, "params": {"C": 1.0, "max_iter": 200}}}}
    c.experiment["importance"]["curves"] = {"subsample": 60, "max_bins": 6, "min_bin": 30}
    c.experiment["importance"]["n_explain"] = 80
    ref = L.run_importance_loop(**_loop_inputs(c))
    ck = tmp_path / "loop.pkl"
    calls = {"n": 0}
    real_fit = Surrogate.fit

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("pod stopped")
        return real_fit(self, *a, **k)

    monkeypatch.setattr(Surrogate, "fit", flaky)
    with pytest.raises(KeyboardInterrupt):
        L.run_importance_loop(**_loop_inputs(c), checkpoint=ck)
    assert ck.exists()
    monkeypatch.setattr(Surrogate, "fit", real_fit)
    res = L.run_importance_loop(**_loop_inputs(c), checkpoint=ck)
    np.testing.assert_allclose(res["model_matrix"], ref["model_matrix"])
    assert res["n_accepted"] == ref["n_accepted"] == 4
    again = L.run_importance_loop(**_loop_inputs(c), checkpoint=ck)      # finished: returned from the checkpoint
    np.testing.assert_allclose(again["feature_importance"], ref["feature_importance"])


# --------------------------------------------------------------------------- experiments and comparators
def test_workspace_shares_data_between_experiments(tmp_path):
    from perisim.workspace import Workspace

    a, b = Workspace(tmp_path, experiment="t2"), Workspace(tmp_path, experiment="t1")
    assert a.common("inspire") == b.common("inspire") == tmp_path / "common" / "inspire"
    assert a.cache("mimic") == b.cache("mimic")
    assert a.state != b.state and a.artifacts() != b.artifacts() and a.reports != b.reports
    assert a.for_variant("margin_4h").experiment == "t2"


def test_experiment_selects_the_target(cfg):
    import copy

    c = copy.deepcopy(cfg)
    c.experiment["experiment_name"] = "t1"
    assert c.targets_to_model == ["t1"] and c.pilot_target == "t1" and c.patient_output_target == "t1"
    c.experiment["experiment_name"] = "t9"
    with pytest.raises(ValueError):
        _ = c.targets_to_model


def test_group_prevalence_is_shrunk_towards_the_overall():
    from perisim.experiment import _group_prevalence

    g = np.array(["a"] * 80 + ["b"] * 20, dtype=object)
    y = np.array([1] * 40 + [0] * 40 + [1] * 20)
    p = _group_prevalence(g, y, np.array(["a", "b", "z"], dtype=object), 2, m=20)
    overall = 0.6
    assert p[0, 1] == pytest.approx((40 + 20 * overall) / 100)
    assert p[1, 1] == pytest.approx((20 + 20 * overall) / 40)
    assert p[2, 1] == pytest.approx(overall)                           # unseen group → overall prevalence
