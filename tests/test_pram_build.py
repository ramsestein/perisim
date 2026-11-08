"""scripts/pram_build.py on tiny hand-made MIMIC-IV and eICU tables: cohort rules, aggregates, outcomes."""

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest.importorskip("duckdb")
import pram_build as B  # noqa: E402


def _w(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(header + "\n" + "\n".join(rows) + "\n")


def _mimic(d: Path) -> None:
    m = d / "mimic4"
    _w(m / "icu" / "icustays.csv.gz", "subject_id,hadm_id,stay_id,first_careunit,last_careunit,intime,outtime,los", [
        "1,11,101,X,X,2150-01-01 00:00:00,2150-01-03 00:00:00,2",      # included
        "1,12,102,X,X,2151-01-01 00:00:00,2151-01-05 00:00:00,4",      # second stay of patient 1: excluded
        "2,21,201,X,X,2150-01-01 00:00:00,2150-01-01 20:00:00,0.8",    # < 24 h: excluded
        "3,31,301,X,X,2150-01-01 00:00:00,2150-01-04 00:00:00,3",      # anchor years 2020-2022: excluded
        "4,41,401,X,X,2150-01-01 00:00:00,2150-01-04 00:00:00,3",      # no creatinine in 24 h: excluded
        "5,51,501,X,X,2150-01-01 00:00:00,2150-01-04 00:00:00,3"])     # included, dies at 30 h
    _w(m / "hosp" / "patients.csv.gz", "subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod", [
        "1,M,60,2150,2014 - 2016,", "2,F,50,2150,2014 - 2016,", "3,F,50,2150,2020 - 2022,",
        "4,M,70,2150,2008 - 2010,", "5,F,17,2149,2011 - 2013,"])      # 17 + 1 = 18 at ICU admission
    _w(m / "hosp" / "admissions.csv.gz", "subject_id,hadm_id,admittime,dischtime,deathtime,hospital_expire_flag", [
        "1,11,x,2150-01-10 00:00:00,,0", "1,12,x,2151-01-10 00:00:00,,0", "2,21,x,2150-01-02 00:00:00,,0",
        "3,31,x,2150-01-09 00:00:00,,0", "4,41,x,2150-01-09 00:00:00,,0", "5,51,x,2150-01-02 06:00:00,2150-01-02 06:00:00,1"])
    ce = "subject_id,hadm_id,stay_id,caregiver_id,charttime,storetime,itemid,value,valuenum,valueuom,warning"
    _w(m / "icu" / "chartevents.csv.gz", ce, [
        "1,11,101,0,2150-01-01 01:00:00,,220045,80,80,,0", "1,11,101,0,2150-01-01 05:00:00,,220045,120,120,,0",
        "1,11,101,0,2150-01-01 09:00:00,,220045,100,100,,0", "1,11,101,0,2150-01-01 23:00:00,,220045,500,500,,0",  # out of range
        "1,11,101,0,2150-01-02 06:00:00,,220045,60,60,,0",                                                       # after 24 h
        "1,11,101,0,2150-01-01 02:00:00,,223761,98.6,98.6,,0",                                                   # 37 °C
        "1,11,101,0,2150-01-01 03:00:00,,220739,4,4,,0", "1,11,101,0,2150-01-01 03:00:00,,223900,5,5,,0",
        "1,11,101,0,2150-01-01 03:00:00,,223901,6,6,,0", "1,11,101,0,2150-01-01 04:00:00,,220739,1,1,,0",        # incomplete GCS
        "1,11,101,0,2150-01-01 00:30:00,,226531,220,220,,0",                                                     # 99.8 kg
        "5,51,501,0,2150-01-01 01:00:00,,220179,110,110,,0"])
    _w(m / "hosp" / "labevents.csv.gz", "labevent_id,subject_id,hadm_id,specimen_id,itemid,order_provider_id,charttime,"
       "storetime,value,valuenum,valueuom", [
        "1,1,11,900,50912,x,2150-01-01 02:00:00,,1.0,1.0,", "2,1,11,900,51006,x,2150-01-01 02:00:00,,20,20,",
        "3,1,11,901,50912,x,2150-01-01 10:00:00,,1.2,1.2,", "4,1,11,902,50912,x,2150-01-04 00:00:00,,1.3,1.3,",   # +0.3 at 72 h
        "5,4,41,903,50912,x,2150-01-02 06:00:00,,1.0,1.0,",                                                       # 30 h only
        "6,5,51,904,50912,x,2150-01-01 03:00:00,,0.9,0.9,", "7,5,51,905,50912,x,2150-01-09 01:00:00,,9.0,9.0,"])  # > 192 h


def _eicu(d: Path) -> None:
    e = d / "eicu"
    _w(e / "patient.csv.gz", "patientunitstayid,patienthealthsystemstayid,gender,age,hospitaldischargeyear,"
       "unitvisitnumber,unitdischargeoffset,hospitaldischargeoffset,hospitaldischargestatus,admissionweight,uniquepid", [
        "1001,1,Male,> 89,2014,1,3000,9000,Expired,70,u1",          # dies at 150 h: outcome
        "1002,1,Male,> 89,2014,2,3000,9000,Expired,70,u1",          # second unit visit: excluded
        "1003,2,Female,40,2015,1,1000,5000,Alive,60,u2",            # < 24 h
        "1004,3,Female,16,2015,1,3000,5000,Alive,60,u3",            # minor
        "1005,4,Female,50,2014,1,3000,20000,Expired,,u4"])          # dies after 192 h
    _w(e / "vitalPeriodic.csv.gz", "vitalperiodicid,patientunitstayid,observationoffset,temperature,sao2,heartrate,"
       "respiration,systemicsystolic,systemicdiastolic", [
        "1,1001,10,36.5,97,90,18,,", "2,1001,700,37.5,93,110,22,,", "3,1001,1500,39,80,150,30,,",
        "4,1005,20,,95,70,,,"])
    _w(e / "vitalAperiodic.csv.gz", "vitalaperiodicid,patientunitstayid,observationoffset,noninvasivesystolic,"
       "noninvasivediastolic", ["1,1001,100,120,60", "2,1001,1300,100,50"])
    _w(e / "nurseCharting.csv.gz", "nursingchartid,patientunitstayid,nursingchartoffset,nursingchartentryoffset,"
       "nursingchartcelltypecat,nursingchartcelltypevallabel,nursingchartcelltypevalname,nursingchartvalue", [
        "1,1001,60,60,x,Glasgow coma score,GCS Total,14", "2,1001,600,600,x,Glasgow coma score,GCS Total,9"])
    _w(e / "lab.csv.gz", "labid,patientunitstayid,labresultoffset,labtypeid,labname,labresult", [
        "1,1001,30,1,creatinine,2.0", "2,1001,30,1,BUN,40", "3,1001,3000,1,creatinine,2.2",          # +0.2 and 1.1×: no AKI
        "4,1005,40,1,creatinine,0.8", "5,1005,2000,1,creatinine,1.2"])                              # 1.5×: AKI


def test_build_on_synthetic_tables(tmp_path, monkeypatch):
    _mimic(tmp_path)
    _eicu(tmp_path)
    monkeypatch.setattr(B, "DATA", tmp_path)
    monkeypatch.setattr(B, "WORK", tmp_path / "work")
    assert B.main(["--threads", "1"]) == 0
    import polars as pl
    m = pl.read_parquet(tmp_path / "work" / "mimic.parquet").sort("id")
    assert m["id"].to_list() == ["101", "501"]
    r = m.row(0, named=True)
    assert (r["hr_min"], r["hr_max"], r["hr_last"]) == (80, 120, 100)
    assert r["temp_c_last"] == pytest.approx(37.0) and r["gcs_total_min"] == 15 and r["weight_kg_max"] == pytest.approx(99.79, abs=0.01)
    assert r["baseline_scr"] == 1.0 and r["scr_feat_last"] == 1.2 and r["bun_cr_ratio_last"] == pytest.approx(20)
    assert r["aki"] and not r["death"] and r["sex_male"] == 1
    r5 = m.row(1, named=True)
    assert r5["death"] and not r5["aki"] and r5["age"] == 18 and r5["sbp_last"] == 110
    e = pl.read_parquet(tmp_path / "work" / "eicu.parquet").sort("id")
    assert e["id"].to_list() == ["1001", "1005"]
    a = e.row(0, named=True)
    assert a["age"] == 90 and a["death"] and not a["aki"]
    assert (a["hr_min"], a["hr_max"], a["hr_last"]) == (90, 110, 110) and a["spo2_min"] == 93
    assert (a["sbp_min"], a["sbp_last"]) == (100, 100) and a["gcs_total_last"] == 9 and a["weight_kg_last"] == 70
    assert a["bun_cr_ratio_max"] == pytest.approx(20)
    b = e.row(1, named=True)
    assert b["aki"] and not b["death"] and b["weight_kg_last"] is None
    rep = (tmp_path / "work" / "build.md").read_text()
    assert "with_creatinine_24h" in rep and "aki" in rep
