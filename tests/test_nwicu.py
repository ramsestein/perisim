"""scripts/nwicu_build.py on synthetic NWICU tables, and scripts/nwicu_transport.py on the frozen hypo chain."""

import gzip
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import nwicu_build as NB  # noqa: E402


def _gz(path: Path, df: pl.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(df.write_csv().encode())


def _raw(root: Path, n=60, seed=0):
    rng = np.random.default_rng(seed)
    d = root / "nwicu" / "northwestern-icu-nwicu-database-0.1.0" / "data"
    t0 = datetime(2021, 1, 1)
    fmt = lambda t: t.strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    stays = []
    for i in range(n):
        tin = t0 + timedelta(days=i)
        stays.append({"subject_id": 100 + i, "hadm_id": 200 + i, "stay_id": 300 + i,
                      "first_careunit": "SICU" if i % 3 else "MICU", "last_careunit": "SICU",
                      "intime": fmt(tin), "outtime": fmt(tin + timedelta(hours=40)), "los": 1.7})
    _gz(d / "nw_icu" / "icustays.csv.gz", pl.DataFrame(stays))
    _gz(d / "nw_hosp" / "patients.csv.gz", pl.DataFrame({"subject_id": [100 + i for i in range(n)],
        "gender": ["M" if i % 2 else "F" for i in range(n)], "anchor_age": [40 + i % 40 for i in range(n)],
        "anchor_year": [2021] * n, "anchor_year_group": ["2020 - 2022"] * n, "dod": [""] * n}))
    _gz(d / "nw_hosp" / "admissions.csv.gz", pl.DataFrame({"subject_id": [100 + i for i in range(n)],
        "hadm_id": [200 + i for i in range(n)], "admission_type": ["EMERGENCY" if i % 4 else "ELECTIVE" for i in range(n)],
        "deathtime": [""] * n}))
    ce, em, lab = [], [], []
    for s in stays:
        tin = datetime.strptime(s["intime"], "%Y-%m-%d %H:%M:%S")
        level = rng.normal(80, 10)
        for h in range(-1, 40):
            t = fmt(tin + timedelta(hours=h, minutes=5))
            sbp = level + 25 + rng.normal(0, 8)
            dbp = level - 10 + rng.normal(0, 5)
            for item, v in (("320179", sbp), ("320180", dbp), ("320045", rng.normal(85, 10)), ("320277", 97.0),
                            ("320210", 16.0), ("323761", 98.6)):
                ce.append({"subject_id": s["subject_id"], "hadm_id": s["hadm_id"], "stay_id": s["stay_id"],
                           "charttime": t, "itemid": item, "valuenum": round(float(v), 1)})
            if h % 2 == 0 and level < 78:
                em.append({"subject_id": s["subject_id"], "hadm_id": s["hadm_id"], "charttime": t,
                           "medication": "NOREPINEPHRINE BITARTRATE-D5W 4 MG/250 ML", "event_txt": "Confirmed"})
        em.append({"subject_id": s["subject_id"], "hadm_id": s["hadm_id"], "charttime": fmt(tin),
                   "medication": "LIDOCAINE-EPINEPHRINE 1 % INJ SOLN", "event_txt": "Confirmed"})
        lab.append({"subject_id": s["subject_id"], "hadm_id": s["hadm_id"], "itemid": "100007",
                    "charttime": fmt(tin - timedelta(hours=2)), "storetime": fmt(tin - timedelta(hours=1)), "valuenum": 11.0})
    _gz(d / "nw_icu" / "chartevents.csv.gz", pl.DataFrame(ce))
    _gz(d / "nw_hosp" / "emar.csv.gz", pl.DataFrame(em))
    _gz(d / "nw_hosp" / "labevents.csv.gz", pl.DataFrame(lab))


def test_build_on_synthetic_tables(tmp_path, monkeypatch):
    _raw(tmp_path / "data")
    monkeypatch.setattr(NB, "DATA", tmp_path / "data" / "nwicu" / "northwestern-icu-nwicu-database-0.1.0" / "data")
    monkeypatch.setattr(NB, "WORK", tmp_path / "work" / "experiments" / "nwicu")
    assert NB.main(["--threads", "1"]) == 0
    st = pl.read_parquet(NB.WORK / "stays.parquet")
    assert st.height == 40                                         # SICU only (2 of every 3)
    v = pl.read_parquet(NB.WORK / "vitals.parquet")
    assert {"map_nibp", "hr", "temp"} <= set(v["var"].unique().to_list())
    assert v.filter(pl.col("var") == "temp")["value"].mean() == pytest_approx(37.0, 0.1)
    sup = pl.read_parquet(NB.WORK / "supports.parquet")
    assert set(sup["kind"].unique().to_list()) <= {"vaso"}          # lidocaine-epinephrine is not a vasopressor
    lm = pl.read_parquet(NB.WORK / "landmarks" / "nwicu.parquet")
    assert lm.height > 0 and lm["vent_on"].null_count() == lm.height  # ventilation is source-missing
    assert "Outcomes are not summarised" in (NB.WORK / "build.md").read_text()


def pytest_approx(x, tol):
    import pytest
    return pytest.approx(x, abs=tol)


def test_transport_chain(tmp_path, monkeypatch):
    from hypo.config import load
    from test_hypo_external import _xframe

    cfg = load()
    monkeypatch.setenv("PERISIM_WORKDIR", str(tmp_path))
    d = tmp_path / "experiments" / "hypo" / "landmarks"
    d.mkdir(parents=True)
    for i, c in enumerate(cfg["cohorts"]):
        _xframe(c, 120, 4, 10 + i).write_parquet(d / f"{c}.parquet")
    nd = tmp_path / "experiments" / "nwicu" / "landmarks"
    nd.mkdir(parents=True)
    _xframe("mimic", 100, 4, 99).with_columns(pl.lit(None, pl.Boolean).alias("vent_on")).write_parquet(nd / "nwicu.parquet")
    from hypo.cv import run
    from hypo.frozen import final, freeze

    run(cfg["cohorts"], ["prevalence", "euclidean", "lightgbm"], None, quick=False, threads=1)
    final(quick=True, threads=1)
    freeze("TEST", no_git=True, quick=True)
    import nwicu_transport as NT
    monkeypatch.setattr(NT, "N_BOOT", 200)
    assert NT.main(["--threads", "1", "--quick"]) == 0
    out = json.loads((tmp_path / "experiments" / "nwicu" / "transport.json").read_text())
    assert set(out["R"]) == set(NT.ROTATIONS) and out["rule"]["primary"] in ("confirmed", "partial", "not confirmed")
    assert "Rule (pre-registered)" in (tmp_path / "experiments" / "nwicu" / "transport.md").read_text()
