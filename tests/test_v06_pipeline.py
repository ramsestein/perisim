"""v0.6 stream D: §6.3 freeze gate of the transport stage, package versions in the freeze, frozen-artefact hashes,
small cells in the transport tables, stale outputs, --force of the phase-1 fits and the relabel cache key."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest


# --------------------------------------------------------------------------- helpers
def _pipe(cfg, tmp_path, overrides: dict | None = None, force: bool = False):
    from perisim.pipeline import Pipeline

    ov = {"experiment": {"paths": {"workdir": str(tmp_path / "work")}}}
    c = cfg.with_overrides(ov).with_overrides(overrides or {})
    return Pipeline(c, force=force)


def _freeze(pipe, packages: dict | None = None, **extra) -> dict:
    from perisim.config import code_hash
    from perisim.utils import package_versions

    fr = {"tag": "perisim-freeze-test", "code_hash": code_hash(), "config_hash": pipe.cfg.scientific_hash(),
          "packages": package_versions() if packages is None else packages, **extra}
    (pipe.ws.state / "freeze.json").write_text(json.dumps(fr))
    (pipe.ws.state / "frozen_config").mkdir(parents=True, exist_ok=True)
    (pipe.ws.state / "frozen_config" / "experiment.yaml").write_text("seed: 1\n")
    return fr


@pytest.fixture
def fake_run(monkeypatch):
    """Replaces run_transport / run_external and marks the external stage done; records the calls."""
    import perisim.external
    import perisim.transport

    calls = []
    monkeypatch.setattr(perisim.transport, "run_transport", lambda p: calls.append(("transport", p.governance)))
    monkeypatch.setattr(perisim.external, "run_external", lambda p: calls.append(("external", p.governance)))
    return calls


def _only_external_done(pipe):
    pipe.done = lambda st, c=None: st == "external"
    return pipe


# --------------------------------------------------------------------------- 1. freeze gate of the transport stage
def test_transport_stage_key_covers_its_config_and_code(cfg, tmp_path):
    from perisim.pipeline import STAGE_DEFS

    sd = STAGE_DEFS["transport"]
    assert "experiment.transport" in sd.config and "transport" in sd.code
    a = _pipe(cfg, tmp_path).key("transport")
    b = _pipe(cfg, tmp_path, {"experiment": {"transport": {"margins": {"citl": 0.2}}}}).key("transport")
    assert a != b


def test_transport_runs_on_the_frozen_pipeline(cfg, tmp_path, fake_run):
    pipe = _only_external_done(_pipe(cfg, tmp_path))
    _freeze(pipe)
    assert pipe.run(["transport"]) == 0
    assert fake_run and fake_run[0][0] == "transport"
    gov = fake_run[0][1]
    assert gov["freeze_gate"] is True and gov["confirmatory"] is True and gov["deviations"] == []


def test_transport_stops_after_a_post_freeze_config_change(cfg, tmp_path, fake_run):
    p0 = _pipe(cfg, tmp_path)
    _freeze(p0)
    pipe = _only_external_done(_pipe(cfg, tmp_path, {"experiment": {"transport": {"margins": {"citl": 0.2}}}}))
    assert pipe.run(["transport"]) == 3             # Stop
    assert not fake_run
    stop = json.loads((pipe.ws.state / "last_stop.json").read_text())
    assert stop["stage"] == "transport" and "freeze" in stop["message"]


def test_transport_stops_after_a_code_change(cfg, tmp_path, fake_run):
    pipe = _only_external_done(_pipe(cfg, tmp_path))
    _freeze(pipe, code_hash="000000000000")
    assert pipe.run(["transport"]) == 3 and not fake_run


def test_transport_waits_without_freeze(cfg, tmp_path, fake_run):
    pipe = _only_external_done(_pipe(cfg, tmp_path))
    assert pipe.run(["transport"]) == 0 and not fake_run


# --------------------------------------------------------------------------- 4. software environment
def test_env_drift_detection():
    from perisim.utils import env_drift, package_versions

    live = package_versions()
    assert env_drift(dict(live), live) == []
    assert env_drift(None, live)                           # no record: cannot verify
    assert any("numpy" in d for d in env_drift({**live, "numpy": "0.0.1"}, live))
    flipped = {**live, "faiss": None if live.get("faiss") else "1.0"}
    assert any("faiss" in d for d in env_drift(flipped, live))
    assert env_drift({**live, "pyarrow": "0.0.1"}, live) == []    # recorded, not compared
    other_faiss = {**live, "faiss": "9.9"} if live.get("faiss") else live
    assert env_drift(other_faiss, live) == []                     # faiss: presence only


@pytest.mark.parametrize("stage", ["external", "transport"])
def test_env_drift_stops_unless_explicitly_allowed(cfg, tmp_path, fake_run, monkeypatch, stage):
    from perisim.utils import ENV_DRIFT_VAR, package_versions

    pipe = _only_external_done(_pipe(cfg, tmp_path))
    if stage == "external":
        pipe.done = lambda st, c=None: False
    _freeze(pipe, packages={**package_versions(), "numpy": "0.0.1"})
    monkeypatch.delenv(ENV_DRIFT_VAR, raising=False)
    assert pipe.run([stage]) == 3 and not fake_run
    assert "numpy" in json.loads((pipe.ws.state / "last_stop.json").read_text())["message"]
    monkeypatch.setenv(ENV_DRIFT_VAR, "1")
    assert pipe.run([stage]) == 0
    gov = fake_run[0][1]
    assert any("numpy" in d and ENV_DRIFT_VAR in d for d in gov["deviations"])


def test_freeze_without_package_record_stops(cfg, tmp_path, fake_run, monkeypatch):
    from perisim.utils import ENV_DRIFT_VAR

    monkeypatch.delenv(ENV_DRIFT_VAR, raising=False)
    pipe = _only_external_done(_pipe(cfg, tmp_path))
    _freeze(pipe, packages={})
    assert pipe.run(["transport"]) == 3 and not fake_run


# --------------------------------------------------------------------------- 5. gate disabled
def test_disabled_freeze_gate_is_recorded_and_not_confirmatory(cfg, tmp_path, fake_run):
    from perisim.transport.run import _aggregate

    pipe = _only_external_done(_pipe(cfg, tmp_path, {"experiment": {"gates": {"external_requires_freeze": False}}}))
    assert pipe.run(["transport"]) == 0            # no freeze needed
    gov = fake_run[0][1]
    assert gov["freeze_gate"] is False and gov["confirmatory"] is False and gov["deviations"]
    assert _aggregate(pipe.cfg, pipe.ws, [], {}, [], confirmatory=False,
                      deviations=gov["deviations"])["status_of_hypotheses"] == {}


def test_status_of_hypotheses_follows_governance(cfg, tmp_path, monkeypatch):
    import perisim.transport.run as R

    pipe = _pipe(cfg, tmp_path)
    for name in ("shape_homogeneity", "coverage_missingness", "h6", "hospital_heterogeneity", "_figures"):
        monkeypatch.setattr(R, name, lambda *a, **k: {"rows": []} if a and isinstance(a[0], dict) else {})
    monkeypatch.setattr(R, "coverage_missingness", lambda *a, **k: [])
    monkeypatch.setattr(R, "_hypothesis_rows", lambda t, status, H, H6: [{"status": status}])
    ok = R._aggregate(pipe.cfg, pipe.ws, ["t2"], {}, [])["status_of_hypotheses"]["t2"]
    off = R._aggregate(pipe.cfg, pipe.ws, ["t2"], {}, [], confirmatory=False,
                       deviations=["gates.external_requires_freeze disabled"])["status_of_hypotheses"]["t2"]
    dev = R._aggregate(pipe.cfg, pipe.ws, ["t2"], {}, [], deviations=["environment drift"])["status_of_hypotheses"]["t2"]
    assert ok == "confirmatory" and off.startswith("descriptive") and "disabled" in off
    assert dev.startswith("confirmatory") and "deviation" in dev


# --------------------------------------------------------------------------- 2. frozen artefact hashes
def _frozen_ws(cfg, tmp_path):
    pipe = _pipe(cfg, tmp_path)
    _freeze(pipe)
    ws = pipe.ws
    (ws.predictions("external", "t2") / "final_model.pkl").write_bytes(b"\x80\x04frozen")
    return pipe, ws


def test_frozen_hashes_check_the_live_code_and_config(cfg, tmp_path):
    from perisim.transport.core import check_hashes, frozen_hashes

    pipe, ws = _frozen_ws(cfg, tmp_path)
    h = frozen_hashes(ws, ["t2"], cfg=pipe.cfg)
    check_hashes(h, frozen_hashes(ws, ["t2"], cfg=pipe.cfg))
    other = pipe.cfg.with_overrides({"experiment": {"transport": {"margins": {"citl": 0.3}}}})
    with pytest.raises(RuntimeError, match="config_hash"):
        frozen_hashes(ws, ["t2"], cfg=other)
    fr = json.loads((ws.state / "freeze.json").read_text())
    (ws.state / "freeze.json").write_text(json.dumps({**fr, "code_hash": "x"}))
    with pytest.raises(RuntimeError, match="code_hash"):
        frozen_hashes(ws, ["t2"])


def test_frozen_hashes_fail_on_missing_files_and_cover_the_fits(cfg, tmp_path):
    from perisim.transport.core import check_hashes, frozen_hashes

    pipe, ws = _frozen_ws(cfg, tmp_path)
    d = ws.predictions("transport", "t2")
    for name in ("mimic_local_model.pkl", "mimic_multisource_model.pkl", "shapes_inspire.pkl"):
        (d / name).write_bytes(name.encode())
    (ws.artifacts() / "transport.json").write_text("{}")
    h = frozen_hashes(ws, ["t2"], cfg=pipe.cfg)
    assert {k.split("/")[-1] for k in h} >= {"freeze.json", "experiment.yaml", "final_model.pkl",
                                             "mimic_local_model.pkl", "mimic_multisource_model.pkl",
                                             "shapes_inspire.pkl", "transport.json"}
    assert not any(k.endswith("transport.json") for k in frozen_hashes(ws, ["t2"], outputs=False))
    (d / "mimic_local_model.pkl").write_bytes(b"refit")
    with pytest.raises(RuntimeError):
        check_hashes(h, frozen_hashes(ws, ["t2"], cfg=pipe.cfg))
    with pytest.raises(RuntimeError, match="missing"):
        frozen_hashes(ws, ["t2"], expected=[d / "eicu_local_model.pkl"])
    with pytest.raises(RuntimeError, match="missing"):
        frozen_hashes(ws, ["t2", "t1"])                 # no final model for t1
    with pytest.raises(RuntimeError):
        check_hashes({}, {})
    (ws.state / "freeze.json").unlink()
    with pytest.raises(RuntimeError, match="freeze"):
        frozen_hashes(ws, ["t2"])


# --------------------------------------------------------------------------- 3. small cells
def test_ood_pct_never_reveals_small_counts():
    from perisim.transport.core import ood_share, point_metrics

    for k in (1, 5, 9, 191, 195, 199):
        assert np.isnan(ood_share(np.r_[np.ones(k), np.zeros(200 - k)]))
    assert ood_share(np.r_[np.ones(20), np.zeros(180)]) == 10.0
    assert ood_share(np.zeros(50)) == 0.0
    assert np.isnan(ood_share(np.full(10, np.nan))) and np.isnan(ood_share(None))
    rng = np.random.default_rng(0)
    P = rng.dirichlet([1, 1, 1, 1], size=200)
    y = rng.integers(0, 4, 200)
    assert np.isnan(point_metrics(P, y, "t2", np.r_[np.ones(3), np.zeros(197)])["ood_pct"])


def test_coverage_missingness_goes_through_share(cfg, tmp_path, monkeypatch):
    import perisim.experiment
    from perisim.transport.run import coverage_missingness

    def ft(vals):
        return types.SimpleNamespace(specs=[types.SimpleNamespace(name="last_hr", type="continuous")],
                                     num={"last_hr": np.asarray(vals, float)}, cat={})

    n = 1000
    dev = np.r_[np.full(3, np.nan), np.linspace(50, 150, n - 3)]          # 3 missing
    ext = np.r_[np.full(500, np.nan), np.full(4, 200.0), np.full(n - 504, 80.0)]   # 4 outside the range
    data = {"inspire": dev, "mimic": ext, "eicu": np.full(n, 80.0)}
    monkeypatch.setattr(perisim.experiment, "load_cohort_data",
                        lambda common, derived, names: types.SimpleNamespace(ft=ft(data[common.parent.name])))
    pipe = _pipe(cfg, tmp_path)
    pipe.ws.common = lambda c: tmp_path / c / "common"
    rows = coverage_missingness(pipe.ws, pipe.cfg, ["last_hr"])
    r = rows[0]
    assert r["missing_pct_inspire"] is None              # 3 missing: suppressed
    assert r["missing_pct_mimic"] == 50.0
    assert r["outside_range_pct_mimic"] is None          # 4 of 500 observed: suppressed
    assert r["outside_range_pct_eicu"] == 0.0 and r["missing_pct_eicu"] == 0.0


# --------------------------------------------------------------------------- 6. stale tables
def test_clear_outputs_removes_only_this_stage_files(cfg, tmp_path):
    from perisim.transport.run import _results_dir, clear_outputs

    pipe = _pipe(cfg, tmp_path)
    rd = _results_dir(pipe.ws)
    mine = ["t2_A_decomposition.csv", "t2_F_hospitals_part1.csv", "t2_E_coverage_missingness.csv",
            "t2_hypotheses.csv", "t2_fig_accumulation_mimic.png", "frozen_hashes.json"]
    keep = ["README.md", "t1_A_decomposition.csv", "notes.txt"]
    for f in mine + keep:
        (rd / f).write_text("x")
    (pipe.ws.artifacts() / "transport.json").write_text("{}")
    removed = clear_outputs(pipe.ws, ["t2"])
    assert set(removed) == set(mine) | {"transport.json"}
    assert sorted(p.name for p in rd.iterdir()) == sorted(keep)


# --------------------------------------------------------------------------- 7. --force refits phase 1
@pytest.mark.parametrize("force", [False, True])
def test_force_refits_phase1(cfg, tmp_path, monkeypatch, force):
    import perisim.transport.run as R

    pipe, ws = _frozen_ws(cfg, tmp_path)
    pipe.force = force
    pipe.governance = {"freeze_gate": True, "confirmatory": True, "deviations": []}
    key = pipe.key("transport")
    d = ws.predictions("transport", "t2")
    (d / "KEY").write_text(key)
    old = d / "mimic_local_model.pkl"
    old.write_bytes(b"old fit")
    pipe.available_cohorts = lambda: ["inspire", "mimic"]
    pipe.feature_names = lambda: ["age"]
    pipe.d7 = lambda: {}

    class Halt(Exception):
        pass

    seen = []

    def fake_jobs(p, common, jobs):
        seen.extend(jobs)
        raise Halt

    monkeypatch.setattr(R, "_run_jobs", fake_jobs)
    with pytest.raises(Halt):
        R.run_transport(pipe)
    assert old.exists() is (not force)
    assert (("local", "t2", "mimic") in seen) is force


# --------------------------------------------------------------------------- 8. relabel cache key
def test_relabel_cache_key_covers_bootstrap_and_code():
    import inspect

    from perisim import sensitivity

    src = inspect.getsource(sensitivity.relabel_variant)
    assert '"evaluation": cfg.experiment["evaluation"]' in src and "code_hash([\"sensitivity.py\", \"evaluation\"" in src


def test_transport_stops_on_a_changed_pipeline_even_before_external(cfg, tmp_path, fake_run):
    """The freeze check comes before the wait for the external stage: a post-freeze change stops, never waits."""
    pipe = _pipe(cfg, tmp_path)
    pipe.done = lambda st, c=None: False
    _freeze(pipe, code_hash="000000000000")
    assert pipe.run(["transport"]) == 3 and not fake_run
