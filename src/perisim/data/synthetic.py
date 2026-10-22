"""SYNTHETIC data with the INSPIRE, MIMIC-IV and eICU schemas (per config/harmonization.yaml).

Only for end-to-end pipeline testing (local smoke test and in the pod before the real data).
It neither contains nor derives from patient data: everything is generated from a simple latent model.
"""

from __future__ import annotations

import datetime as dt
import gzip
from pathlib import Path

import numpy as np
import polars as pl


def _w(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        with gzip.open(path, "wb") as fh:
            df.write_csv(fh)
    else:
        df.write_csv(path)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _latent(n: int, rng: np.random.Generator) -> dict:
    """Postoperative patients with an ICU stay and outcomes correlated with the state at discharge."""
    age = np.clip(rng.normal(62, 15, n), 19, 95)
    age[rng.random(n) < 0.02] = 16                           # minors: excluded
    sex = rng.random(n) < 0.55
    asa = rng.choice([1, 2, 3, 4], size=n, p=[0.1, 0.4, 0.4, 0.1])
    emerg = rng.random(n) < 0.2
    vent = rng.random(n) < 0.35
    vaso = rng.random(n) < 0.25
    z = -1.8 + 0.03 * (age - 60) + 0.45 * (asa - 2) + 0.6 * emerg + 0.5 * vent + 0.5 * vaso + rng.normal(0, 0.7, n)
    icu_h = np.exp(rng.normal(np.log(26), 0.6, n)) + 3 * np.maximum(z, 0)
    icu_h[rng.random(n) < 0.03] = rng.uniform(1, 3.5)          # < 4 h: excluded
    severe = rng.random(n) < _sig(z - 0.3)
    mild = (~severe) & (rng.random(n) < 0.35 + 0.1 * (z > 0))
    comp_g = rng.choice(["readmission", "death", "reop", "vent", "aki3", "vaso"], size=n)
    comp_l = rng.choice(["aki12", "antibiotic", "transfusion"], size=n)
    los = np.clip(np.exp(rng.normal(np.log(5) + 0.35 * z, 0.5, n)), 0.6, 45)
    status = np.where(rng.random(n) < 0.02, "icu_death", np.where(rng.random(n) < 0.02, "home_from_icu", "ward"))
    return dict(age=age, sex=sex, asa=asa, emerg=emerg, vent=vent, vaso=vaso, z=z, icu_h=icu_h, severe=severe, mild=mild,
                comp_g=comp_g, comp_l=comp_l, los=los, status=status,
                creat=np.exp(rng.normal(0, 0.35, n)) * (1 + 0.25 * (z > 0)),
                lact=np.clip(1.0 + 0.7 * np.maximum(z, 0) + rng.normal(0, 0.4, n), 0.3, 15),
                hb=np.clip(rng.normal(11 - 0.5 * np.maximum(z, 0), 1.4, n), 6, 17),
                group=rng.choice(["GS", "OS", "NS", "TS", "CS", "UR"], size=n),
                medical=rng.random(n) < 0.12)


LABS = {"creatinine": ("creat", 1.0), "hb": ("hb", 1.0), "wbc": (None, 9.0), "platelet": (None, 220.0),
        "lactate": ("lact", 1.0), "total_bilirubin": (None, 0.8), "albumin": (None, 3.4), "ph": (None, 7.38),
        "pco2": (None, 40.0), "po2": (None, 95.0)}
VITALS = {"hr": 85, "mbp": 80, "rr": 17, "spo2": 96, "bt": 36.8, "fio2": 35, "gcs": 15}


def _lab_value(name: str, L: dict, i: int, rng, drift: float = 0.0) -> float:
    key, base = LABS[name]
    v = L[key][i] if key else base * np.exp(rng.normal(0, 0.15))
    if name == "ph":
        v = base + rng.normal(0, 0.04) - 0.02 * max(L["z"][i], 0)
    return float(v * (1 + drift))


# ============================================================================ INSPIRE
def synth_inspire(root: Path, n: int, rng: np.random.Generator) -> None:
    L = _latent(n, rng)
    ops, labs, wv, meds, dx = [], [], [], [], []
    op_id = 1000
    for i in range(n):
        sid, hid = 1 + i, 50000 + i
        adm = 0.0
        op_s = adm + rng.uniform(600, 4000)
        op_e = op_s + rng.uniform(60, 360)
        icu_in = op_e + rng.uniform(5, 60)
        icu_out = icu_in + L["icu_h"][i] * 60
        disch = icu_out + L["los"][i] * 1440
        death = None
        st = L["status"][i]
        if st == "icu_death":
            death, disch = icu_out, icu_out
        elif st == "home_from_icu":
            disch = icu_out + 10
        g, c = L["severe"][i], L["comp_g"][i]
        if g and c == "death":
            death = icu_out + rng.uniform(0.3, 0.9) * (disch - icu_out)
            disch = death
        has_icu = not L["medical"][i]
        op_id += 1
        ops.append(dict(op_id=op_id, subject_id=sid, hadm_id=hid, age=round(L["age"][i] / 5) * 5,
                        sex="M" if L["sex"][i] else "F", weight=round(rng.normal(68, 12), 1),
                        height=round(rng.normal(165, 9), 1), asa=int(L["asa"][i]), emop=int(L["emerg"][i]),
                        department=L["group"][i], antype=rng.choice(["General", "Neuraxial", "MAC"], p=[.85, .1, .05]),
                        icd10_pcs=rng.choice(["0DTJ0", "0SRC0", "02100", "0FT44", "0TT10", "00B00"]),
                        opstart_time=op_s, opend_time=op_e, admission_time=adm, discharge_time=disch,
                        icuin_time=icu_in if has_icu else None, icuout_time=icu_out if has_icu else None,
                        inhosp_death_time=death))
        if g and c in ("reop", "readmission"):
            r_s = icu_out + rng.uniform(1, 0.6 * (disch - icu_out) / 1440 + 1) * 1440
            r_s = min(r_s, disch - 300)
            op_id += 1
            ops.append(dict(op_id=op_id, subject_id=sid, hadm_id=hid, age=round(L["age"][i] / 5) * 5,
                            sex="M" if L["sex"][i] else "F", weight=None, height=None, asa=int(L["asa"][i]), emop=1,
                            department=L["group"][i], antype="General", icd10_pcs="0DQ80", opstart_time=r_s,
                            opend_time=r_s + 90, admission_time=adm, discharge_time=disch,
                            icuin_time=(r_s + 120) if c == "readmission" else None,
                            icuout_time=(r_s + 120 + 1440) if c == "readmission" else None, inhosp_death_time=death))
        for name in LABS:
            labs.append((sid, adm + 30, name, _lab_value(name, L, i, rng)))
            for t in np.arange(icu_in + 120, icu_out - 60, 720):
                labs.append((sid, float(t), name, _lab_value(name, L, i, rng)))
        aki = (g and c == "aki3") or ((not g) and L["mild"][i] and L["comp_l"][i] == "aki12")
        ratio = 3.2 if (g and c == "aki3") else 1.7
        for d in range(1, int(min(L["los"][i], 12))):
            if rng.random() < 0.8:
                drift = (ratio - 1) * min(d / 2, 1) if aki else rng.normal(0, 0.05)
                labs.append((sid, icu_out + d * 1440, "creatinine", float(L["creat"][i] * (1 + drift))))
        for t in np.arange(icu_in, icu_out, 120.0):
            for k_, base in VITALS.items():
                v = base + rng.normal(0, 0.05 * base) + (8 * L["z"][i] if k_ == "hr" else 0)
                wv.append((sid, float(t), k_, float(min(v, 100) if k_ in ("spo2", "fio2") else v)))
        if L["vent"][i]:
            for t in np.arange(icu_in, icu_in + rng.uniform(2, 0.6 * L["icu_h"][i] * 60 + 3), 60.0):
                wv.append((sid, float(t), "vent", 1.0))
        for t in np.arange(icu_out + 120, disch, 480.0):
            for k_ in ("hr", "spo2", "bt"):
                v = VITALS[k_] + rng.normal(0, 0.03 * VITALS[k_])
                wv.append((sid, float(t), k_, float(min(v, 100))))
        if g and c == "vent":
            s0 = icu_out + rng.uniform(0.5, 3) * 1440
            for t in np.arange(s0, s0 + 600, 60.0):
                wv.append((sid, float(t), "vent", 1.0))
        meds.append((sid, op_s, "CEFAZOLIN", "J01DB04", None, None))
        meds.append((sid, op_s + 480, "CEFAZOLIN", "J01DB04", None, None))
        for t in np.arange(icu_in, icu_out, 720.0):
            meds.append((sid, float(t), "PANTOPRAZOLE", "A02BC02", None, None))
            meds.append((sid, float(t), "FENTANYL", "N02AB03", None, None))
        if L["vaso"][i]:
            for t in np.arange(icu_in, icu_in + 0.5 * L["icu_h"][i] * 60, 60.0):
                meds.append((sid, float(t), "NOREPINEPHRINE", "C01CA03", None, None))
        if g and c == "vaso":
            s0 = icu_out + rng.uniform(0.5, 3) * 1440
            meds.append((sid, s0, "NOREPINEPHRINE", "C01CA03", None, None))
        if (not g) and L["mild"][i] and L["comp_l"][i] == "antibiotic":
            s0 = icu_out + rng.uniform(1.2, 3) * 1440
            for t in np.arange(s0, s0 + 3 * 1440, 480.0):
                meds.append((sid, float(t), "PIPERACILLIN/TAZOBACTAM", "J01CR05", None, None))
        if (not g) and L["mild"][i] and L["comp_l"][i] == "transfusion":
            meds.append((sid, icu_out + 1440, "RBC (filtered)", None, None, None))
        for code, p in (("I10", .4), ("E11", .2), ("N18", .1), ("I25", .15), ("J44", .08), ("C18", .1), ("K80", .06)):
            if rng.random() < p * (1 + 0.3 * (L["z"][i] > 0)):
                dx.append((sid, adm + 15 if rng.random() < 0.8 else disch, code))
    d = root / "inspire"
    _w(pl.DataFrame(ops, infer_schema_length=None), d / "operations.csv")
    _w(pl.DataFrame(labs, schema=["subject_id", "chart_time", "item_name", "value"], orient="row"), d / "labs.csv")
    _w(pl.DataFrame(wv, schema=["subject_id", "chart_time", "item_name", "value"], orient="row"), d / "ward_vitals.csv")
    _w(pl.DataFrame(meds, schema={"subject_id": pl.Int64, "chart_time": pl.Float64, "drug_name": pl.Utf8,
                                  "atc_code": pl.Utf8, "atc_code2": pl.Utf8, "atc_code3": pl.Utf8}, orient="row"),
       d / "medications.csv")
    _w(pl.DataFrame(dx, schema=["subject_id", "chart_time", "icd10_cm"], orient="row"), d / "diagnosis.csv")


# ============================================================================ MIMIC-IV
LAB_ITEMS = {"creatinine": 50912, "hb": 51222, "wbc": 51301, "platelet": 51265, "lactate": 50813,
             "total_bilirubin": 50885, "albumin": 50862, "ph": 50820, "pco2": 50818, "po2": 50821}
CHART_ITEMS = {"hr": 220045, "mbp": 220052, "rr": 220210, "spo2": 220277, "bt": 223762, "fio2": 223835}


def _ts(base: dt.datetime, minutes: float) -> str:
    return (base + dt.timedelta(minutes=float(minutes))).strftime("%Y-%m-%d %H:%M:%S")


def synth_mimic(root: Path, n: int, rng: np.random.Generator) -> None:
    L = _latent(n, rng)
    adm_r, pat, icu, srv, prc, lab, ch, inp, pev, emar = [], [], [], [], [], [], [], [], [], []
    stay = 30000000
    trf, tid = [], 40000000
    ICU_UNIT, STEP_UNIT, WARD = "Surgical Intensive Care Unit (SICU)", "Neuro Stepdown", "Med/Surg"
    srv_map = {"GS": "SURG", "OS": "ORTHO", "NS": "NSURG", "TS": "TSURG", "CS": "CSURG", "UR": "GU"}
    for i in range(n):
        sid, hid = 10000000 + i, 20000000 + i
        base = dt.datetime(2150, 1, 1) + dt.timedelta(days=int(rng.integers(0, 15000)))
        anchor_year = base.year - int(rng.integers(0, 3))
        pat.append((sid, "M" if L["sex"][i] else "F", int(L["age"][i]), anchor_year))
        op_s = rng.uniform(300, 3000)
        icu_in = op_s + rng.uniform(120, 420)
        icu_out = icu_in + L["icu_h"][i] * 60
        disch = icu_out + L["los"][i] * 1440
        death = None
        st = L["status"][i]
        if st == "icu_death":
            death = disch = icu_out
        elif st == "home_from_icu":
            disch = icu_out + 10
        g, c = L["severe"][i], L["comp_g"][i]
        if g and c == "death":
            death = icu_out + 0.6 * (disch - icu_out)
            disch = death
        medical = L["medical"][i]
        adm_r.append((sid, hid, _ts(base, 0), _ts(base, disch), _ts(base, death) if death is not None else None,
                      "EW EMER." if L["emerg"][i] else "SURGICAL SAME DAY ADMISSION",
                      "EMERGENCY ROOM" if L["emerg"][i] else "PHYSICIAN REFERRAL",
                      "DIED" if death is not None else ("HOME" if rng.random() < .6 else "SKILLED NURSING FACILITY"),
                      1 if death is not None else 0))
        stay += 1
        idx_stay = stay
        # ~10 % of stays continue in a step-down unit: MIMIC's icustays merges it, transfers does not (AUDIT D3)
        step = rng.random() < 0.10
        icu.append((sid, hid, stay, _ts(base, icu_in), _ts(base, icu_out + (600 if step else 0)), "SICU"))
        tid += 1
        trf.append((sid, hid, tid, "admit", WARD, _ts(base, 0), _ts(base, icu_in)))
        tid += 1
        trf.append((sid, hid, tid, "transfer", ICU_UNIT, _ts(base, icu_in), _ts(base, icu_out)))
        if step:
            tid += 1
            trf.append((sid, hid, tid, "transfer", STEP_UNIT, _ts(base, icu_out), _ts(base, icu_out + 600)))
        tid += 1
        trf.append((sid, hid, tid, "transfer", WARD, _ts(base, icu_out + (600 if step else 0)), _ts(base, disch)))
        srv.append((sid, hid, _ts(base, 0), "MED" if medical else srv_map[L["group"][i]]))
        if not medical:
            prc.append((sid, hid, 1, _ts(base, op_s)[:10], "0DTJ0ZZ", 10))
        if g and c in ("reop", "readmission"):
            r = icu_out + rng.uniform(1.5, 4) * 1440
            r = min(r, disch - 600)
            prc.append((sid, hid, 2, _ts(base, r)[:10], "0DQ80ZZ", 10))
            if c == "readmission":
                stay += 1
                icu.append((sid, hid, stay, _ts(base, r + 60), _ts(base, r + 1500), "SICU"))
                tid += 1
                trf.append((sid, hid, tid, "transfer", ICU_UNIT, _ts(base, r + 60), _ts(base, r + 1500)))
        for name, it in LAB_ITEMS.items():
            lab.append((sid, hid, it, _ts(base, 30), _lab_value(name, L, i, rng)))
            for t in np.arange(icu_in + 120, icu_out - 60, 720):
                lab.append((sid, hid, it, _ts(base, t), _lab_value(name, L, i, rng)))
        aki = (g and c == "aki3") or ((not g) and L["mild"][i] and L["comp_l"][i] == "aki12")
        ratio = 3.2 if (g and c == "aki3") else 1.7
        for d in range(1, int(min(L["los"][i], 12))):
            if rng.random() < 0.8:
                drift = (ratio - 1) * min(d / 2, 1) if aki else rng.normal(0, 0.05)
                lab.append((sid, hid, 50912, _ts(base, icu_out + d * 1440), float(L["creat"][i] * (1 + drift))))
        for t in np.arange(icu_in, icu_out, 120.0):
            for k_, it in CHART_ITEMS.items():
                b = VITALS[k_]
                v = b + rng.normal(0, 0.05 * b) + (8 * L["z"][i] if k_ == "hr" else 0)
                ch.append((sid, hid, idx_stay, it, _ts(base, t), float(min(v, 100) if k_ in ("spo2", "fio2") else v)))
            for it, v in ((220739, 4), (223900, 5), (223901, 6)):
                ch.append((sid, hid, idx_stay, it, _ts(base, t), float(v)))
        ch.append((sid, hid, idx_stay, 226512, _ts(base, icu_in + 10), float(rng.normal(80, 15))))
        ch.append((sid, hid, idx_stay, 226730, _ts(base, icu_in + 10), float(rng.normal(170, 9))))
        if L["vent"][i]:
            pev.append((sid, hid, idx_stay, 225792, _ts(base, icu_in), _ts(base, icu_in + 0.5 * L["icu_h"][i] * 60)))
        if L["vaso"][i]:
            inp.append((sid, hid, idx_stay, 221906, _ts(base, icu_in + 30), _ts(base, icu_in + 0.4 * L["icu_h"][i] * 60)))
        if g and c == "vent":
            s0 = icu_out + 1.5 * 1440
            pev.append((sid, hid, idx_stay, 225792, _ts(base, s0), _ts(base, s0 + 720)))
        if g and c == "vaso":
            s0 = icu_out + 1.5 * 1440
            inp.append((sid, hid, idx_stay, 221906, _ts(base, s0), _ts(base, s0 + 300)))
        emar.append((sid, hid, _ts(base, op_s), "CeFAZolin", "Administered"))
        for t in np.arange(icu_in, icu_out, 720.0):
            emar.append((sid, hid, _ts(base, t), "Pantoprazole", "Administered"))
            emar.append((sid, hid, _ts(base, t), "Fentanyl Citrate", "Administered"))
        if (not g) and L["mild"][i] and L["comp_l"][i] == "antibiotic":
            s0 = icu_out + rng.uniform(1.2, 3) * 1440
            for t in np.arange(s0, s0 + 3 * 1440, 480.0):
                emar.append((sid, hid, _ts(base, t), "Piperacillin-Tazobactam", "Administered"))
        if (not g) and L["mild"][i] and L["comp_l"][i] == "transfusion":
            emar.append((sid, hid, _ts(base, icu_out + 1440), "Red Blood Cells", "Administered"))
    d = root / "mimic-iv" / "3.1"
    S = lambda rows, cols: pl.DataFrame(rows, schema=cols, orient="row")  # noqa: E731
    _w(S(adm_r, ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "admission_location",
                 "discharge_location", "hospital_expire_flag"]), d / "hosp" / "admissions.csv.gz")
    _w(S(pat, ["subject_id", "gender", "anchor_age", "anchor_year"]), d / "hosp" / "patients.csv.gz")
    _w(S(icu, ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit"]), d / "icu" / "icustays.csv.gz")
    _w(S(srv, ["subject_id", "hadm_id", "transfertime", "curr_service"]), d / "hosp" / "services.csv.gz")
    _w(S(trf, ["subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "intime", "outtime"]),
       d / "hosp" / "transfers.csv.gz")
    _w(S(prc, ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"]), d / "hosp" / "procedures_icd.csv.gz")
    _w(S(lab, ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]), d / "hosp" / "labevents.csv.gz")
    _w(S(ch, ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"]), d / "icu" / "chartevents.csv.gz")
    _w(S(inp, ["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime"]), d / "icu" / "inputevents.csv.gz")
    _w(S(pev, ["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime"]), d / "icu" / "procedureevents.csv.gz")
    _w(S(emar, ["subject_id", "hadm_id", "charttime", "medication", "event_txt"]), d / "hosp" / "emar.csv.gz")
    items = [(220045, "Heart Rate"), (220052, "Arterial Blood Pressure mean"), (220181, "Non Invasive Blood Pressure mean"),
             (220210, "Respiratory Rate"), (220277, "O2 saturation pulseoxymetry"), (223762, "Temperature Celsius"),
             (223761, "Temperature Fahrenheit"), (223835, "Inspired O2 Fraction"), (220739, "GCS - Eye Opening"),
             (223900, "GCS - Verbal Response"), (223901, "GCS - Motor Response"), (226512, "Admission Weight (Kg)"),
             (226730, "Height (cm)"), (225792, "Invasive Ventilation"), (221906, "Norepinephrine"),
             (221289, "Epinephrine"), (221749, "Phenylephrine"), (222315, "Vasopressin"), (221662, "Dopamine"),
             (225802, "Dialysis - CRRT"), (225803, "Dialysis - CVVHD"), (225805, "Peritoneal Dialysis"),
             (225809, "Dialysis - CVVHDF"), (225955, "Dialysis - SCUF"), (225441, "Hemodialysis"),
             (225168, "Packed Red Blood Cells")]
    _w(S(items, ["itemid", "label"]), d / "icu" / "d_items.csv.gz")
    _w(S([(v, k) for k, v in LAB_ITEMS.items()] + [(51300, "WBC Count")], ["itemid", "label"]), d / "hosp" / "d_labitems.csv.gz")


# ============================================================================ eICU
def synth_eicu(root: Path, n: int, rng: np.random.Generator) -> None:
    L = _latent(n, rng)
    pat, adx, apr, apv, lab, vp, va, rc, nc, inf, rcare, trt, med, dxr = ([] for _ in range(14))
    uid = 100000
    elabs = {"creatinine": "creatinine", "hb": "Hgb", "wbc": "WBC x 1000", "platelet": "platelets x 1000",
             "lactate": "lactate", "total_bilirubin": "total bilirubin", "albumin": "albumin", "ph": "pH",
             "pco2": "paCO2", "po2": "paO2"}
    grp = {"GS": "Gastrointestinal|bowel resection", "OS": "Musculoskeletal|hip fracture", "NS": "Neurologic|craniotomy",
           "TS": "Respiratory|lobectomy", "CS": "Cardiovascular|CABG alone", "UR": "Genitourinary|nephrectomy"}
    for i in range(n):
        hsid = 500000 + i
        upid = f"{i // 1000:03d}-{i:06d}"
        hoff = -float(rng.uniform(60, 2880))
        icu_los = L["icu_h"][i] * 60
        post = L["los"][i] * 1440
        st = L["status"][i]
        g, c = L["severe"][i], L["comp_g"][i]
        dead = st == "icu_death" or (g and c == "death")
        hdis = icu_los + (0 if st == "icu_death" else (10 if st == "home_from_icu" else (0.6 if dead else 1.0) * post))
        uid += 1
        src = "Emergency Department" if L["medical"][i] else "Operating Room"
        pat.append((uid, hsid, upid, "Male" if L["sex"][i] else "Female", "> 89" if L["age"][i] > 89 else str(int(L["age"][i])),
                    round(rng.normal(168, 9), 1), round(rng.normal(80, 15), 1), hoff,
                    "Emergency Department" if L["emerg"][i] else "Direct Admit", hdis, "Expired" if dead else "Alive",
                    "Death" if dead else "Home", 2015, src, 1, icu_los,
                    "Expired" if st == "icu_death" else "Alive",
                    "Home" if st == "home_from_icu" else ("Death" if st == "icu_death" else "Floor")))
        adx.append((uid, ("admission diagnosis|Operative|Diagnosis|" + grp[L["group"][i]]) if not L["medical"][i]
                    else "admission diagnosis|Non-operative|Diagnosis|Sepsis"))
        apr.append((uid, "IVa", round(float(L["los"][i] + L["icu_h"][i] / 24 + rng.normal(0, 2)), 2)))
        apv.append((uid, 0 if L["emerg"][i] else 1))
        if g and c == "readmission":
            uid += 1
            r = icu_los + rng.uniform(1, 3) * 1440
            pat.append((uid, hsid, upid, "Male" if L["sex"][i] else "Female", str(int(L["age"][i])), None, None,
                        hoff - r, "Emergency Department", hdis - r, "Alive", "Home", 2015, "Floor", 2, 1440.0, "Alive",
                        "Floor"))
        u0 = uid if not (g and c == "readmission") else uid - 1
        for name, ln in elabs.items():
            for t in np.arange(60, icu_los - 60, 720):
                lab.append((u0, float(t), ln, _lab_value(name, L, i, rng)))
        for t in np.arange(0, icu_los, 60.0):
            vp.append((u0, float(t), 85 + 8 * L["z"][i] + rng.normal(0, 5), 17 + rng.normal(0, 2), min(100, 96 + rng.normal(0, 1.5)),
                       36.8 + rng.normal(0, .3), 80 + rng.normal(0, 6)))
            if rng.random() < 0.3:
                va.append((u0, float(t), 78 + rng.normal(0, 7)))
        for t in np.arange(0, icu_los, 240.0):
            rc.append((u0, float(t), "FiO2", str(round(min(100, 35 + rng.normal(0, 5)), 0))))
            nc.append((u0, float(t), "GCS Total", "15"))
        if L["vaso"][i]:
            for t in np.arange(30, 0.4 * icu_los, 60.0):
                inf.append((u0, float(t), "Norepinephrine (mcg/min)"))
        if L["vent"][i]:
            rcare.append((u0, 1.0, float(0.5 * icu_los)))
        trt.append((u0, 30.0, "gastrointestinal|medications|stress ulcer prophylaxis|pantoprazole"))
        med.append((u0, 10.0, icu_los, "CEFAZOLIN 1 GM", "No"))
        med.append((u0, 10.0, icu_los, "PANTOPRAZOLE 40 MG", "No"))
        if rng.random() < 0.4:
            dxr.append((u0, 30.0, "401.9, I10"))
        if rng.random() < 0.2:
            dxr.append((u0, 40.0, "250.00, E11.9"))
    d = root / "eicu-crd" / "2.0"
    S = lambda rows, cols: pl.DataFrame(rows, schema=cols, orient="row", infer_schema_length=None)  # noqa: E731
    _w(S(pat, ["patientunitstayid", "patienthealthsystemstayid", "uniquepid", "gender", "age", "admissionheight",
               "admissionweight", "hospitaladmitoffset", "hospitaladmitsource", "hospitaldischargeoffset",
               "hospitaldischargestatus", "hospitaldischargelocation", "hospitaldischargeyear", "unitadmitsource",
               "unitvisitnumber", "unitdischargeoffset", "unitdischargestatus", "unitdischargelocation"]),
       d / "patient.csv.gz")
    _w(S(adx, ["patientunitstayid", "admitdxpath"]), d / "admissionDx.csv.gz")
    _w(S(apr, ["patientunitstayid", "apacheversion", "predictedhospitallos"]), d / "apachePatientResult.csv.gz")
    _w(S(apv, ["patientunitstayid", "electivesurgery"]), d / "apachePredVar.csv.gz")
    _w(S(lab, ["patientunitstayid", "labresultoffset", "labname", "labresult"]), d / "lab.csv.gz")
    _w(S(vp, ["patientunitstayid", "observationoffset", "heartrate", "respiration", "sao2", "temperature",
              "systemicmean"]), d / "vitalPeriodic.csv.gz")
    _w(S(va, ["patientunitstayid", "observationoffset", "noninvasivemean"]), d / "vitalAperiodic.csv.gz")
    _w(S(rc, ["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"]), d / "respiratoryCharting.csv.gz")
    _w(S(nc, ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevalname", "nursingchartvalue"]),
       d / "nurseCharting.csv.gz")
    _w(S(inf, ["patientunitstayid", "infusionoffset", "drugname"]), d / "infusionDrug.csv.gz")
    _w(S(rcare, ["patientunitstayid", "ventstartoffset", "ventendoffset"]), d / "respiratoryCare.csv.gz")
    _w(S(trt, ["patientunitstayid", "treatmentoffset", "treatmentstring"]), d / "treatment.csv.gz")
    _w(S(med, ["patientunitstayid", "drugstartoffset", "drugstopoffset", "drugname", "drugordercancelled"]),
       d / "medication.csv.gz")
    _w(S(dxr, ["patientunitstayid", "diagnosisoffset", "icd9code"]), d / "diagnosis.csv.gz")


def write_synthetic(root: Path, harmonization: dict | None = None, n_subjects: int = 700, seed: int = 7) -> Path:
    root = Path(root)
    marker = root / f".synthetic_{n_subjects}_{seed}"
    if marker.exists():
        return root
    rng = np.random.default_rng(seed)
    synth_inspire(root, n_subjects, rng)
    synth_mimic(root, n_subjects, rng)
    synth_eicu(root, n_subjects, rng)
    marker.write_text("PERISIM synthetic data")
    return root
