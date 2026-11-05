"""v0.6 fixes of the `hypo` experiment (AUDIT 3, 15 and the low-severity hypo items): supports cut at the landmark,
eICU infusion stops, the LOO prior of α, MIMIC-IV lab result times, deaths in the ICU only, the freeze coverage."""

import json

import numpy as np
import polars as pl
import pytest

from hypo.config import code_hash, load
from hypo.extract import infusion_points, lab_hours
from hypo.landmarks import build, hourly, support_hours, support_hours_before, support_runs

CFG = load()
INPUTS = ["vaso_on", "sedation_on", "vent_on", "vaso_24h", "sedation_24h", "vaso_hours"]
SUP = {"stay_id": pl.Utf8, "kind": pl.Utf8, "h": pl.Float64, "h_end": pl.Float64}


def _stays(ids, los=60.0, death=None):
    n = len(ids)
    return pl.DataFrame({"stay_id": ids, "subject_id": [s.lower() for s in ids], "los_h": [los] * n,
                         "death_h": [death] * n, "limitation_h": [None] * n, "surgery_end_to_icu_h": [None] * n},
                        schema_overrides={"death_h": pl.Float64, "limitation_h": pl.Float64,
                                          "surgery_end_to_icu_h": pl.Float64})


def _flat_map(ids, value=80.0):
    return pl.DataFrame([(s, h + 0.5, "map_nibp", value) for s in ids for h in range(-2, 54)],
                        schema=["stay_id", "h", "var", "value"], orient="row")


def _nolabs():
    return pl.DataFrame(schema={"stay_id": pl.Utf8, "h": pl.Float64, "var": pl.Utf8, "value": pl.Float64})


def _sup(rows):
    return pl.DataFrame(rows, schema=SUP, orient="row")


# --------------------------------------------------------------------------- 1. supports cut at L (AUDIT 3)
def test_support_point_after_L_does_not_fill_hours_before_L():
    """The audit's reproduction: a vasopressor point at 11.5 h made vaso_on true and vaso_hours 3 at L = 10."""
    sup = _sup([("A", "vaso", 7.0, 7.0), ("B", "vaso", 7.0, 7.0), ("B", "vaso", 11.5, 11.5)])
    lm = build(_stays(["A", "B"]), hourly(_flat_map(["A", "B"])), _nolabs(), sup, CFG)
    r = {d["stay_id"]: d for d in lm.filter(pl.col("L") == 10).iter_rows(named=True)}
    assert not r["B"]["vaso_on"] and r["B"]["vaso_hours"] == 1
    assert all(r["A"][c] == r["B"][c] for c in INPUTS)
    assert r["B"]["vaso_start"] and not r["A"]["vaso_start"]       # the carried outcome still reads the window


def test_support_inputs_never_see_records_at_or_after_L():
    """Metamorphic: for every L, the support inputs of landmark L are the same whether or not the records from L on
    exist (points, intervals spanning L, stop markers)."""
    rng = np.random.default_rng(6)
    ids = [f"S{i}" for i in range(12)]
    rows = []
    for s in ids:
        for kind in ("vaso", "sedation", "vent"):
            for h in np.sort(rng.uniform(-6, 54, rng.integers(0, 12))):
                rows.append((s, kind, float(h), float(h)))
            if rng.random() < 0.5:
                a = float(rng.uniform(-4, 50))
                rows.append((s, kind, a, a + float(rng.uniform(0.5, 20))))
    sup = _sup(rows).with_columns(pl.Series("stop", rng.random(len(rows)) < 0.1)).with_columns(
        pl.col("stop") & (pl.col("h") == pl.col("h_end")))
    st, H = _stays(ids), hourly(_flat_map(ids))
    full = build(st, H, _nolabs(), sup, CFG)
    for L in range(int(CFG["landmarks"]["first_h"]), int(CFG["landmarks"]["last_h"]) + 1, int(CFG["landmarks"]["every_h"])):
        cut = build(st, H, _nolabs(), sup.filter(pl.col("h") < L).with_columns(
            pl.min_horizontal("h_end", pl.lit(float(L))).alias("h_end")), CFG)
        a = full.filter(pl.col("L") == L).sort("stay_id").select("stay_id", *INPUTS)
        b = cut.filter(pl.col("L") == L).sort("stay_id").select("stay_id", *INPUTS)
        assert a.equals(b), L


def test_interval_spanning_L_is_clipped_and_build_refuses_an_hourly_table():
    sup = _sup([("A", "vaso", 3.2, 30.0)])
    hrs = support_hours_before(sup, 10, CFG["landmarks"]["support_gap_h"])["hour"].sort().to_list()
    assert hrs == list(range(3, 10))
    lm = build(_stays(["A"]), hourly(_flat_map(["A"])), _nolabs(), sup, CFG)
    r = lm.filter(pl.col("L") == 10).row(0, named=True)
    assert r["vaso_on"] and r["vaso_hours"] == 7 and not r["vaso_start"]
    with pytest.raises(ValueError):
        build(_stays(["A"]), hourly(_flat_map(["A"])), _nolabs(), support_hours(sup), CFG)


# --------------------------------------------------------------------------- 2. eICU infusion stops
def test_infusion_rates_running_stop_and_unknown():
    inf = pl.DataFrame({"stay_id": ["A"] * 8, "infusionoffset": [str(60 * i) for i in range(8)],
                        "drugname": ["Norepinephrine"] * 8,
                        "drugrate": ["5", "OFF", "0", "", None, "Documentation undone", "-1", " 0.0 "]})
    p = infusion_points(inf)
    assert p["h"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0, 7.0]           # the free text and the negative rate: dropped
    assert p["stop"].to_list() == [False, True, True, True, True, True]
    assert (p["h"] == p["h_end"]).all()


def test_stop_marker_ends_a_run_and_is_not_running():
    sup = pl.DataFrame({"stay_id": ["A"] * 4, "kind": ["vaso"] * 4, "h": [1.0, 2.5, 4.0, 5.0],
                        "h_end": [1.0, 2.5, 4.0, 5.0], "stop": [False, True, False, False]})
    runs = support_runs(sup, 6).sort("h")
    assert runs.select("h", "h_end").rows() == [(1.0, 1.0), (4.0, 5.0)] and "stop" not in runs.columns
    assert support_hours(sup, 6)["hour"].sort().to_list() == [1, 4, 5]       # without the stop: 1–5 bridged
    assert support_hours(sup)["hour"].sort().to_list() == [1, 4, 5]           # no runs: the stop is not a support hour
    no_stop = sup.with_columns(pl.lit(False).alias("stop"))
    assert support_hours(no_stop, 6)["hour"].sort().to_list() == [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- 3. α LOO prior without the own patient
def _casebase(E, y, subj, k, alpha_grid=(0.1, 1.0, 10.0)):
    from hypo.casebase import CaseBase
    from perisim.knn.predict import KNNModel

    cb = CaseBase.__new__(CaseBase)
    KNNModel.__init__(cb, E, np.ones(E.shape[1]), y, 2, subj, [k])
    cb.E, cb.k, cb.distinct, cb.fetch = E, k, True, 6
    cb.mcfg = {"alpha_grid": list(alpha_grid), "alpha_queries": len(y), "ood_percentile": 99}
    cb.subj_s = np.asarray(subj, dtype=str)
    cb.uniq, cb.code = np.unique(cb.subj_s, return_inverse=True)
    return cb


def test_alpha_loo_prior_leaves_out_the_query_patient():
    rng = np.random.default_rng(3)
    subj = np.repeat([f"p{i}" for i in range(20)], 5)
    y = (rng.random(subj.size) < 0.2).astype(int)
    y[subj == "p0"] = 1                                       # one patient with events only
    E = rng.normal(size=(subj.size, 3))
    cb = _casebase(E, y, subj, k=5)
    info = cb.fit_alpha(1)
    rows, pri = cb.loo["rows"], cb.loo["prior"]
    for r, pr in zip(rows, pri):
        other = subj != subj[r]
        assert pr[1] == pytest.approx(y[other].mean())
    p = cb.probs(cb.loo["I"], cb.loo["valid"], 5, alpha=info["alpha"], prior=pri)
    ll = -np.mean(np.log(np.clip(p[np.arange(rows.size), y[rows]], 1e-15, 1)))
    assert info["loo_logloss"] == pytest.approx(ll)


# --------------------------------------------------------------------------- 4. MIMIC-IV lab result time
def test_mimic_lab_time_is_the_result_time_with_charttime_fallback():
    lab = pl.DataFrame({"charttime": ["2150-01-01 10:00:00", "2150-01-01 11:00:00"],
                        "storetime": ["2150-01-01 12:30:00", None], "icu_in_min": [0.0, 0.0]})

    def hours(col):
        return pl.col(col).str.to_datetime(strict=False).dt.epoch("s") / 3600

    st = lab.select(lab_hours(lab, "storetime", hours).alias("h"))["h"].to_list()
    ch = lab.select(lab_hours(lab, "charttime", hours).alias("h"))["h"].to_list()
    assert st[0] - ch[0] == pytest.approx(2.5) and st[1] == ch[1]
    assert CFG["sources"]["mimic"]["lab_time"] == "storetime"
    with pytest.raises(ValueError):
        lab_hours(lab, "resulttime", hours)


# --------------------------------------------------------------------------- 5. deaths in the ICU only
def test_death_after_icu_discharge_is_not_in_the_window():
    st = pl.concat([_stays(["A"], los=10.0, death=12.0), _stays(["B"], los=10.0, death=10.5)])
    lm = build(st, hourly(_flat_map(["A", "B"])), _nolabs(), _sup([]), CFG)
    r = {d["stay_id"]: d for d in lm.filter(pl.col("L") == 8).iter_rows(named=True)}
    assert not r["A"]["died_in_window"] and not r["A"]["y"]    # died on the ward, 2 h after the ICU discharge
    assert r["B"]["died_in_window"] and r["B"]["y"]            # within the common format's 60-min tolerance


# --------------------------------------------------------------------------- 6. freeze coverage
@pytest.fixture
def frozen_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PERISIM_WORKDIR", str(tmp_path))
    from hypo import frozen
    from hypo.cv import cv_dir

    w = tmp_path / "experiments" / "hypo"
    (w / "landmarks").mkdir(parents=True)
    cv = cv_dir(False)
    cv.mkdir(parents=True)
    frozen.final_dir().mkdir(parents=True)
    for c in CFG["cohorts"]:
        (w / "landmarks" / f"{c}.parquet").write_bytes(c.encode())
        (cv / f"folds_{c}.parquet").write_bytes(b"f" + c.encode())
    (cv / "perisim_ext-eicu_f0.parquet").write_bytes(b"p0")
    for a in frozen._expected_artefacts(CFG):
        (frozen.final_dir() / a).write_bytes(a.encode())
        (frozen.final_dir() / a).with_suffix(".json").write_text(json.dumps({"code_hash": code_hash(), "quick": True}))
    return frozen, w, cv


def test_freeze_records_and_checks_landmarks_phase2_perisim_and_prereg(frozen_setup):
    frozen, w, cv = frozen_setup
    rec = frozen.freeze("TEST", no_git=True, quick=True)
    assert set(rec["hashes"]["landmarks"]) == set(CFG["cohorts"]) and rec["hashes"]["phase2_perisim"]["files"] == 1
    assert frozen.check()["tag"] == rec["tag"]
    (cv / "perisim_ext-eicu_f0.parquet").write_bytes(b"p0 changed")
    with pytest.raises(frozen.Stop, match="phase2_perisim"):
        frozen.check()
    (cv / "perisim_ext-eicu_f0.parquet").write_bytes(b"p0")
    (w / "landmarks" / "mimic.parquet").write_bytes(b"other")
    with pytest.raises(frozen.Stop, match="landmarks"):
        frozen.check()
    (w / "landmarks" / "mimic.parquet").write_bytes(b"mimic")
    frozen.check()
    p = frozen.state_dir() / "freeze.json"
    r = json.loads(p.read_text())
    p.write_text(json.dumps({**r, "prereg_sha": "0" * 16}))
    with pytest.raises(frozen.Stop, match="prereg"):
        frozen.check()
    old = {**r, "hashes": {k: v for k, v in r["hashes"].items() if k != "landmarks"}}   # a pre-v0.6 record
    p.write_text(json.dumps(old))
    with pytest.raises(frozen.Stop, match="landmarks"):
        frozen.check()


def test_freeze_refused_with_any_external_output_or_missing_landmarks(frozen_setup):
    frozen, w, cv = frozen_setup
    ext = frozen.external_dir()
    ext.mkdir(parents=True)
    (ext / "drift_ext-eicu_mimic.pkl").write_bytes(b"cache")       # no parquet: refused all the same
    with pytest.raises(frozen.Stop, match="external results"):
        frozen.freeze("TEST", no_git=True, quick=True)
    (ext / "drift_ext-eicu_mimic.pkl").unlink()
    (w / "landmarks" / "inspire.parquet").unlink()
    with pytest.raises(frozen.Stop, match="landmarks missing"):
        frozen.freeze("TEST", no_git=True, quick=True)
