"""Branch arc-metric (docs/prereg_arc.md): the arc-length placement of a shaped continuous / ordinal value, its
invariances against the published `value` placement, and the guard of the work folder."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from perisim.features.build import FeatureSpec
from perisim.features.table import FeatureTable
from perisim.metric.encoding import Encoder


def _ft(x: np.ndarray, grp: np.ndarray | None = None) -> FeatureTable:
    n = len(x)
    rng = np.random.default_rng(1)
    grp = rng.choice(["a", "b", "c"], n).astype(object) if grp is None else grp
    specs = [FeatureSpec("x", "continuous", "x", "icu"), FeatureSpec("grp", "nominal", "grp", "surgery")]
    ids = np.arange(n).astype(str).astype(object)
    return FeatureTable(ids, ids, specs, {"x": x}, {"grp": grp})


def _centred(z: np.ndarray) -> np.ndarray:
    """Binary centred log-odds of class-1 log-odds z (as importance/shapes.py: one component per class)."""
    return np.column_stack([-z / 2, z / 2])


def _fit(embedding: str, curve, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    ft = _ft(x)
    y = rng.integers(0, 2, n)
    enc = Encoder(geometry="shape", max_bins=20, min_bin=50, shape_embedding=embedding).fit(ft, y, 2)
    j = enc.feature_index()["x"]
    e = enc.blocks[j].params["edges"]
    mid = 0.5 * (e[:-1] + e[1:])
    g = _centred(curve(mid))
    g = g - g.mean(0)
    k = enc.feature_index()["grp"]
    gk = _centred(np.array([0.0, 0.5, -1.0])[: len(enc.blocks[k].params["values"])])
    return enc.with_shapes({j: g, k: gk}, ft), ft, j, k


def _u(x):
    return 0.15 * (x - 5) ** 2


def _col(enc, ft, name):
    E = enc.transform(ft)
    return E[:, enc.dim_feature == enc.feature_index()[name]]


# --------------------------------------------------------------------------- the placement
def test_arc_is_one_dimensional_monotone_and_continuous():
    enc, ft, j, _ = _fit("arc", _u)
    b = enc.blocks[j]
    assert b.dims == 1 and (enc.dim_feature == j).sum() == 1
    xs = np.linspace(-1, 11, 2001)
    phi = enc.transform(_ft(xs))[:, enc.dim_feature == j][:, 0]
    assert np.all(np.diff(phi) >= -1e-9)                              # monotone in x, although g is a U
    e = b.params["edges"]
    eps = 1e-6
    for edge in e[1:-1]:                                              # no jump at a bin edge
        lo, hi = enc.transform(_ft(np.array([edge - eps, edge + eps])))[:, enc.dim_feature == j][:, 0]
        assert abs(hi - lo) < 1e-3
    assert phi[0] == phi[1] and phi[-1] == phi[-2]                    # constant beyond the outer nodes


def test_value_placement_is_unchanged_and_steps():
    enc, ft, j, _ = _fit("value", _u)
    assert enc.blocks[j].dims == 2
    e = enc.blocks[j].params["edges"]
    a, b_ = enc.transform(_ft(np.array([e[3] + 1e-3, e[4] - 1e-3])))[:, enc.dim_feature == j]
    assert np.allclose(a, b_)                                         # constant within a bin


def test_a_u_shape_collapses_with_value_but_not_with_arc():
    """Two values on either side of the U's minimum, at the same height: net change ≈ 0, path walked > 0."""
    ev, _, jv, _ = _fit("value", _u)
    ea, _, ja, _ = _fit("arc", _u)
    xn, _ = Encoder.arc_nodes(ea.blocks[ja])                          # bin midpoints, where g is the U itself
    left, right = xn[xn < 3.5], xn[xn > 6.5]
    h = np.abs(_u(left)[:, None] - _u(right)[None, :])
    i, k = np.unravel_index(np.argmin(h), h.shape)                    # the pair at the most similar height
    q = np.array([left[i], right[k], 5.0])
    v = ev.transform(_ft(q))[:, ev.dim_feature == jv]
    a = ea.transform(_ft(q))[:, ea.dim_feature == ja]
    d_v = np.linalg.norm(v[0] - v[1])
    d_a = np.linalg.norm(a[0] - a[1])
    assert d_v < 0.15 * d_a
    # arc: 2 → 8 walks down to the minimum and up again = (2 → 5) + (5 → 8)
    assert np.isclose(d_a, np.linalg.norm(a[0] - a[2]) + np.linalg.norm(a[2] - a[1]), rtol=1e-6)


def test_on_a_monotone_curve_both_placements_agree_at_the_nodes():
    """Monotone g: the arc between two nodes equals the net change, so after s_j the distances are the same."""
    ev, _, jv, _ = _fit("value", lambda x: 0.4 * x + 0.05 * x ** 2)
    ea, _, ja, _ = _fit("arc", lambda x: 0.4 * x + 0.05 * x ** 2)
    e = ev.blocks[jv].params["edges"]
    nodes = 0.5 * (e[:-1] + e[1:])
    v = ev.transform(_ft(nodes))[:, ev.dim_feature == jv]
    a = ea.transform(_ft(nodes))[:, ea.dim_feature == ja]
    dv = np.linalg.norm(v[:, None] - v[None], axis=-1)
    da = np.abs(a[:, None, 0] - a[None, :, 0])
    # the normalisations differ (s_j is computed over the patients, placed differently within bins): compare shapes
    assert np.allclose(dv / dv.max(), da / da.max(), atol=1e-6)


def test_arc_leaves_the_nominal_variables_and_the_neutral_axes_alone():
    ev, ft, _, kv = _fit("value", _u)
    ea, _, _, ka = _fit("arc", _u)
    assert np.allclose(_col(ev, ft, "grp"), _col(ea, ft, "grp"))
    # a variable without a shape (flat) keeps its neutral rank axis in both
    rng = np.random.default_rng(0)
    ft2 = _ft(rng.uniform(0, 10, 2000))
    y = rng.integers(0, 2, 2000)
    for emb in ("value", "arc"):
        enc = Encoder(geometry="shape", shape_embedding=emb).fit(ft2, y, 2).with_shapes({}, ft2)
        assert enc.blocks[enc.feature_index()["x"]].shape is None
    n_v = Encoder(geometry="shape", shape_embedding="value").fit(ft2, y, 2).with_shapes({}, ft2).transform(ft2)
    n_a = Encoder(geometry="shape", shape_embedding="arc").fit(ft2, y, 2).with_shapes({}, ft2).transform(ft2)
    assert np.allclose(n_v, n_a)


def test_missing_values_go_to_the_median_and_scale_is_unit():
    enc, ft, j, _ = _fit("arc", _u)
    b = enc.blocks[j]
    a = enc.transform(_ft(np.array([np.nan, b.params["median"]])))[:, enc.dim_feature == j]
    assert np.allclose(a[0], a[1])
    col = _col(enc, ft, "x")[:, 0].astype(np.float64)
    assert np.isclose(2 * col.var(), 1.0, rtol=1e-4)                 # E_pairs[δ²] = 1 on the training data


def test_arc_nodes_and_shape_points():
    enc, _, j, _ = _fit("arc", _u)
    b = enc.blocks[j]
    xn, phi = Encoder.arc_nodes(b)
    assert phi[0] == 0 and np.all(np.diff(phi) >= 0) and len(xn) == len(phi) == b.shape.shape[0]
    pts, exact = enc.shape_points(b)
    assert not exact and pts.shape == (len(xn), 1)
    at_nodes = enc.transform(_ft(xn))[:, enc.dim_feature == j][:, 0] * b.scale
    assert np.allclose(at_nodes, pts[:, 0])
    ev, _, jv, _ = _fit("value", _u)
    pv, ex = ev.shape_points(ev.blocks[jv])
    assert ex and np.allclose(pv, ev.blocks[jv].shape)


def test_drift_lookup_reproduces_the_arc_embedding_at_the_nodes():
    from hypo.drift import lookup, value_codes

    enc, _, j, _ = _fit("arc", _u)
    w = np.ones(enc.n_features)
    look = lookup(enc, w)
    assert not look[j]["exact"]
    xn, _ = Encoder.arc_nodes(enc.blocks[j])
    ftn = _ft(xn, np.array(["a"] * len(xn), dtype=object))
    codes = value_codes(enc, ftn, look)
    E = enc.transform(ftn) * enc.weights_dims(w)
    got = look[j]["points"][codes[:, j]]
    assert np.allclose(got, E[:, look[j]["cols"]], atol=1e-5)


def test_old_pickles_keep_the_value_placement():
    enc, ft, j, _ = _fit("value", _u)
    old = pickle.loads(pickle.dumps(enc))
    del old.__dict__["shape_embedding"]                              # an encoder pickled before the option
    assert np.allclose(old.transform(ft), enc.transform(ft))


def test_unknown_embedding_is_refused():
    with pytest.raises(ValueError):
        Encoder(geometry="shape", shape_embedding="spline")


def test_the_arm_selects_the_placement_and_enters_the_hashes(monkeypatch):
    from hypo.config import load
    from hypo.frozen import config_hash
    from perisim.config import load_config

    monkeypatch.delenv("PERISIM_ARM", raising=False)                  # default arm: arc
    cfg = load_config(profile="smoke")
    assert cfg.experiment["metric"]["shape_embedding"] == "arc"
    assert Encoder.from_config(cfg).shape_embedding == "arc"
    assert load()["metric"]["shapes"]["embedding"] == "arc"
    h_arc, s_arc = config_hash(), cfg.scientific_hash()
    monkeypatch.setenv("PERISIM_ARM", "value")
    cfg_v = load_config(profile="smoke")
    assert cfg_v.experiment["metric"]["shape_embedding"] == "value"
    assert load()["metric"]["shapes"]["embedding"] == "value"
    assert config_hash() != h_arc and cfg_v.scientific_hash() != s_arc   # a freeze is bound to its arm


# --------------------------------------------------------------------------- the work folders
def test_guard_refuses_a_published_or_other_arm_folder(tmp_path, monkeypatch):
    from perisim.arc_guard import MARKER, NotArcWorkdir, ensure_arc_workdir, marker_arm

    monkeypatch.setenv("PERISIM_ARM", "arc")
    old = tmp_path / "work"
    (old / "experiments" / "hypo").mkdir(parents=True)
    with pytest.raises(NotArcWorkdir):
        ensure_arc_workdir(old)
    assert not (old / MARKER).exists()
    new = tmp_path / "work_arc"
    assert ensure_arc_workdir(new) == new.resolve() and marker_arm(new) == "arc"
    (new / "experiments").mkdir()
    ensure_arc_workdir(new)                                           # marked: accepted with data inside
    monkeypatch.setenv("PERISIM_ARM", "value")
    with pytest.raises(NotArcWorkdir):
        ensure_arc_workdir(new)                                       # the other arm's folder
    assert marker_arm(ensure_arc_workdir(tmp_path / "work_value")) == "value"
    monkeypatch.setenv("PERISIM_ARM", "spline")
    with pytest.raises(NotArcWorkdir):
        ensure_arc_workdir(tmp_path / "x")


def test_seed_copies_only_metric_independent_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("PERISIM_ARM", "value")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from arc_seed_inputs import seed

    from perisim.arc_guard import MARKER

    old = tmp_path / "work"
    for rel in ("common/inspire/stays.parquet", "cache/inspire/x.parquet", "state/_/extract__inspire.json",
                "artifacts/_data/discovery.json", "reports/schema_inspire.md",
                "experiments/hypo/extract/inspire/v.parquet", "experiments/hypo/landmarks/inspire.parquet",
                "experiments/hypo/final/metric_inspire.pkl", "experiments/t2/artifacts/main/weights.json",
                "experiments/nwicu/stays.parquet", "experiments/nwicu/landmarks/nwicu.parquet",
                "experiments/nwicu/results/transport.json"):
        p = old / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel)
    before = sorted(str(p) for p in old.rglob("*"))
    new = tmp_path / "work_value"
    man = seed(old, new)
    assert man["arm"] == "value"
    got = {f["path"] for f in man["files"]}
    assert "experiments/hypo/final/metric_inspire.pkl" not in got
    assert "experiments/t2/artifacts/main/weights.json" not in got
    assert "experiments/nwicu/results/transport.json" not in got
    assert "experiments/hypo/landmarks/inspire.parquet" not in got     # rebuilt unless --landmarks
    assert {"common/inspire/stays.parquet", "state/_/extract__inspire.json", "experiments/nwicu/stays.parquet",
            "experiments/hypo/extract/inspire/v.parquet"} <= got
    assert (new / MARKER).is_file() and (new / "SEED_MANIFEST.json").is_file()
    assert sorted(str(p) for p in old.rglob("*")) == before            # the published folder is untouched
    with pytest.raises(SystemExit):
        seed(old, new)                                                # never over a seeded folder


# --------------------------------------------------------------------------- the paired contrast
def _scripts():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _unit(n_pat: int, gain: float, delta: float, seed: int):
    """Per-row log-losses: Euclidean, value = Euclidean − gain, arc = value + delta (+ noise), locals alike."""
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_pat), 3)
    e = rng.gamma(2.0, 0.25, subj.size)
    tv = e - gain + rng.normal(0, 0.05, subj.size)
    ta = tv + delta + rng.normal(0, 0.005, subj.size)
    lv = e - gain * 1.05 + rng.normal(0, 0.05, subj.size)
    la = lv + delta + rng.normal(0, 0.005, subj.size)
    return e, tv, ta, lv, la, subj.astype(str)


def test_contrast_reads_equivalence_and_differences():
    _scripts()
    from arc_compare import contrast, readings

    r = contrast(*_unit(3000, 0.02, 0.0, 0), seed=1, n_boot=400)
    assert readings(r)["Q"] == "equivalent" and readings(r)["D"] == "no difference"
    assert abs(r["Q"]["est"] - 1) < 0.05
    r = contrast(*_unit(3000, 0.02, 0.01, 0), seed=1, n_boot=400)   # arc loses half the gain
    assert readings(r)["Q"] == "not equivalent (arc worse)" and readings(r)["D"] == "arc worse"
    r = contrast(*_unit(3000, 0.001, 0.0, 0), seed=1, n_boot=400)   # no gain to compare against
    assert readings(r)["Q"] == "not evaluable"


def test_compare_refuses_unpaired_units_and_reports(tmp_path, monkeypatch):
    _scripts()
    import arc_compare as AC

    from perisim.arc_guard import ensure_arc_workdir

    dirs = {}
    for arm in ("value", "arc"):
        d = ensure_arc_workdir(tmp_path / f"work_{arm}", arm=arm)
        (d / "arc_compare").mkdir()
        dirs[arm] = d
    e, tv, ta, lv, la, subj = _unit(500, 0.02, 0.0, 3)
    ids = np.arange(e.size).astype(str)
    y = (np.arange(e.size) % 2).astype(np.int64)
    np.savez(dirs["value"] / "arc_compare" / "hypo_inspire.npz", ids=ids, subjects=subj, y=y,
             ll_euclidean=e, ll_transported=tv, ll_local=lv)
    np.savez(dirs["arc"] / "arc_compare" / "hypo_inspire.npz", ids=ids, subjects=subj, y=y,
             ll_euclidean=e, ll_transported=ta, ll_local=la)
    monkeypatch.setattr(AC, "N_BOOT", 300)
    assert AC.main(["--value", str(dirs["value"]), "--arc", str(dirs["arc"])]) == 0
    assert "hypo_inspire" in (dirs["arc"] / "reports" / "arc_compare.md").read_text(encoding="utf-8")
    np.savez(dirs["arc"] / "arc_compare" / "hypo_inspire.npz", ids=ids[::-1], subjects=subj, y=y,
             ll_euclidean=e, ll_transported=ta, ll_local=la)
    with pytest.raises(SystemExit):
        AC.compare(dirs["value"], dirs["arc"])
    with pytest.raises(SystemExit):                                   # arms swapped
        AC.main(["--value", str(dirs["arc"]), "--arc", str(dirs["value"])])
