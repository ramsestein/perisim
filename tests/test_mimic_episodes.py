"""AUDIT D3 / C2 — MIMIC-IV ICU episodes come from true ICU careunits in hosp.transfers, and the index
procedure code never comes from a same-day procedure that may follow t0."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl


def _w(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        df.write_csv(fh)


def _mimic(root: Path) -> Path:
    d = root / "mimic"
    _w(pl.DataFrame({"subject_id": [1, 2], "hadm_id": [11, 22], "admittime": ["2150-01-01 08:00:00", "2150-02-01 08:00:00"],
                     "dischtime": ["2150-01-10 12:00:00", "2150-02-08 12:00:00"], "deathtime": [None, None],
                     "admission_type": ["ELECTIVE", "ELECTIVE"], "admission_location": ["PHYSICIAN REFERRAL"] * 2,
                     "discharge_location": ["HOME", "HOME"], "hospital_expire_flag": [0, 0]}), d / "hosp/admissions.csv.gz")
    _w(pl.DataFrame({"subject_id": [1, 2], "gender": ["M", "F"], "anchor_age": [60, 70], "anchor_year": [2150, 2150]}),
       d / "hosp/patients.csv.gz")
    # stay 1: SICU 01-01 14:00 → 01-03 10:00, then Neuro Stepdown until 01-04 10:00 (icustays merges both)
    _w(pl.DataFrame({"subject_id": [1, 2], "hadm_id": [11, 22], "stay_id": [101, 202],
                     "intime": ["2150-01-01 14:00:00", "2150-02-01 15:00:00"],
                     "outtime": ["2150-01-04 10:00:00", "2150-02-01 22:00:00"], "first_careunit": ["SICU", "SICU"]}),
       d / "icu/icustays.csv.gz")
    _w(pl.DataFrame({"subject_id": [1, 1, 1, 1, 2, 2, 2], "hadm_id": [11, 11, 11, 11, 22, 22, 22],
                     "transfer_id": [1, 2, 3, 4, 5, 6, 7], "eventtype": ["admit"] + ["transfer"] * 3 + ["admit", "transfer", "transfer"],
                     "careunit": ["Med/Surg", "Surgical Intensive Care Unit (SICU)", "Neuro Stepdown", "Med/Surg",
                                  "Med/Surg", "Surgical Intensive Care Unit (SICU)", "Med/Surg"],
                     "intime": ["2150-01-01 08:00:00", "2150-01-01 14:00:00", "2150-01-03 10:00:00", "2150-01-04 10:00:00",
                                "2150-02-01 08:00:00", "2150-02-01 15:00:00", "2150-02-01 22:00:00"],
                     "outtime": ["2150-01-01 14:00:00", "2150-01-03 10:00:00", "2150-01-04 10:00:00", "2150-01-10 12:00:00",
                                 "2150-02-01 15:00:00", "2150-02-01 22:00:00", "2150-02-08 12:00:00"]}),
       d / "hosp/transfers.csv.gz")
    _w(pl.DataFrame({"subject_id": [1, 2], "hadm_id": [11, 22], "transfertime": ["2150-01-01 08:00:00"] * 2,
                     "curr_service": ["SURG", "SURG"]}), d / "hosp/services.csv.gz")
    # stay 2: ICU admission and t0 on the SAME day → a same-day code may be a reoperation after t0
    _w(pl.DataFrame({"subject_id": [1, 2], "hadm_id": [11, 22], "seq_num": [1, 1],
                     "chartdate": ["2150-01-01", "2150-02-01"], "icd_code": ["0DTJ0ZZ", "0DQ80ZZ"], "icd_version": [10, 10]}),
       d / "hosp/procedures_icd.csv.gz")
    return d


def test_episodes_and_pcs(tmp_path, cfg):
    from perisim.data.common import dt_minutes
    from perisim.data.mimic import MimicExtractor

    st = MimicExtractor(cfg, _mimic(tmp_path), tmp_path / "cache").extract_stays().sort("hadm_id")
    ref = pl.DataFrame({"t": ["2150-01-03 10:00:00"]})
    sicu_exit = ref.select(dt_minutes(ref, "t")).item()
    s1 = st.filter(pl.col("hadm_id") == "MIMIC-11").row(0, named=True)
    assert s1["icu_out_min"] == sicu_exit                    # not the step-down exit (icustays outtime)
    assert s1["pcs_code"] == "0DTJ0ZZ"                       # same day as ICU admission, t0 two days later: OK
    s2 = st.filter(pl.col("hadm_id") == "MIMIC-22").row(0, named=True)
    assert s2["pcs_code"] is None                            # same day as t0: could follow t0 → not used
    assert s2["postop"] is True
