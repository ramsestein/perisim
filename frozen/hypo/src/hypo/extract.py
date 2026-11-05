"""Extraction from the ORIGINAL tables of each cohort to three long tables, all timed in hours after the ICU admission:

  vitals    stay_id, h, var (map_inv, map_nibp, hr, spo2, rr, temp), value
  labs      stay_id, h, var (hb, creatinine, lactate), value
  supports  stay_id, kind (vaso, sedation, vent), h, h_end   (charted points have h_end = h)

and the population (`stays`). Patient-level: cached in <workdir>/extract/ (git-ignored), never exported. The original
CSV files are filtered while streaming (DuckDB), so the large tables (MIMIC-IV chartevents / labevents, eICU
vitalPeriodic) are never loaded whole.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

from .config import common_dir

LOG = logging.getLogger("hypo.extract")
PREFIX = {"inspire": "INSPIRE", "mimic": "MIMIC", "eicu": "EICU"}


def _raw(col: str, cohort: str) -> pl.Expr:
    return pl.col(col).str.slice(len(PREFIX[cohort]) + 1)


def _csv(path: Path) -> str:
    return f"read_csv('{path.as_posix()}', header=true, all_varchar=true, ignore_errors=true)"


def _in(values) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


# --------------------------------------------------------------------------- population
def stays(cohort: str, cfg: dict, hospital_of=None) -> pl.DataFrame:
    """First postoperative adult ICU stay of each hospitalization (deaths kept), with what the landmarks need."""
    pop = cfg["population"]
    s = pl.read_parquet(common_dir(cohort) / "stays_candidates.parquet")
    s = s.filter(pl.col("postop").fill_null(False) & (pl.col("age") >= float(pop["min_age"])))
    if pop.get("exclude_organ_donors", True):
        s = s.filter(~pl.col("organ_donor").fill_null(False))
    if pop.get("first_postop_stay_of_hospitalization", True):
        s = s.sort("icu_in_min").group_by("hadm_id", maintain_order=True).first()
    def hours(col: str) -> pl.Expr:
        return ((pl.col(col) - pl.col("icu_in_min")) / 60).alias(col.replace("_min", "_h"))

    out = s.select(
        "stay_id", "subject_id", "hadm_id", "icu_in_min",
        ((pl.col("icu_out_min") - pl.col("icu_in_min")) / 60).alias("los_h"),
        hours("death_min"), hours("limitation_min"),
        ((pl.col("surgery_end_min") - pl.col("surgery_start_min")) / 60).alias("surgery_duration_h"),
        ((pl.col("icu_in_min") - pl.col("surgery_end_min")) / 60).alias("surgery_end_to_icu_h"),
        "age", "sex_male", "bmi", "emergency", "surgical_group", "anesthesia_type")
    if hospital_of is not None:
        out = out.with_columns(pl.Series("hospital", hospital_of(out["stay_id"].to_numpy()), dtype=pl.Utf8))
    return out


# --------------------------------------------------------------------------- per cohort
def _mimic(st: pl.DataFrame, src: Path, cfg: dict, con) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    S = cfg["sources"]["mimic"]
    con.register("k", st.select("stay_id", _raw("hadm_id", "mimic").alias("hadm"), "icu_in_min").to_arrow())
    item_var = {str(i): v for v, items in S["chartevents"].items() for i in items}
    vent_items = [str(i) for i in S.get("chartevents_vent", [])]

    def hours(col: str) -> pl.Expr:
        return (pl.col(col).str.to_datetime(strict=False).dt.epoch("s") / 60 - pl.col("icu_in_min")) / 60

    q = f"""SELECT k.stay_id, c.itemid, c.charttime, c.valuenum, k.icu_in_min FROM {_csv(src / 'icu' / 'chartevents.csv.gz')} c
            JOIN k ON c.hadm_id = k.hadm
            WHERE (c.itemid IN ({_in(item_var)}) AND c.valuenum IS NOT NULL) OR c.itemid IN ({_in(vent_items or ['-'])})"""
    ce = pl.from_arrow(con.execute(q).arrow())
    vent_pts = ce.filter(pl.col("itemid").is_in(vent_items)).select(
        "stay_id", pl.lit("vent").alias("kind"), hours("charttime").alias("h"), hours("charttime").alias("h_end"))
    v = ce.filter(~pl.col("itemid").is_in(vent_items)).select(
        "stay_id", hours("charttime").alias("h"), pl.col("itemid").replace_strict(item_var).alias("var"),
        pl.col("valuenum").cast(pl.Float64, strict=False).alias("value"))
    v = v.with_columns(pl.when(pl.col("var") == "temp_f").then((pl.col("value") - 32) * 5 / 9)
                       .otherwise(pl.col("value")).alias("value"),
                       pl.col("var").replace({"temp_f": "temp"}))
    item_lab = {str(i): v for v, items in S["labevents"].items() for i in items}
    q = f"""SELECT k.stay_id, l.itemid, l.charttime, l.valuenum, k.icu_in_min FROM {_csv(src / 'hosp' / 'labevents.csv.gz')} l
            JOIN k ON l.hadm_id = k.hadm WHERE l.itemid IN ({_in(item_lab)}) AND l.valuenum IS NOT NULL"""
    lab = pl.from_arrow(con.execute(q).arrow()).select(
        "stay_id", hours("charttime").alias("h"), pl.col("itemid").replace_strict(item_lab).alias("var"),
        pl.col("valuenum").cast(pl.Float64, strict=False).alias("value"))
    parts = []
    for kind, items in S["inputevents"].items():
        q = f"""SELECT k.stay_id, i.starttime, i.endtime, k.icu_in_min FROM {_csv(src / 'icu' / 'inputevents.csv.gz')} i
                JOIN k ON i.hadm_id = k.hadm WHERE i.itemid IN ({_in(items)})
                  AND coalesce(i.statusdescription, '') <> 'Rewritten'"""
        parts.append(pl.from_arrow(con.execute(q).arrow()).select(
            "stay_id", pl.lit(kind).alias("kind"), hours("starttime").alias("h"), hours("endtime").alias("h_end")))
    for kind, items in S["procedureevents"].items():
        q = f"""SELECT k.stay_id, p.starttime, p.endtime, k.icu_in_min FROM {_csv(src / 'icu' / 'procedureevents.csv.gz')} p
                JOIN k ON p.hadm_id = k.hadm WHERE p.itemid IN ({_in(items)})"""
        parts.append(pl.from_arrow(con.execute(q).arrow()).select(
            "stay_id", pl.lit(kind).alias("kind"), hours("starttime").alias("h"), hours("endtime").alias("h_end")))
    parts.append(vent_pts)
    con.unregister("k")
    return v, lab, pl.concat(parts)


def _eicu(st: pl.DataFrame, src: Path, cfg: dict, con) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    S, X = cfg["sources"]["eicu"], cfg["extraction"]
    con.register("k", st.select("stay_id", _raw("stay_id", "eicu").alias("unit")).to_arrow())
    lo, hi = (int(x * 60) for x in X["vitals_h"])
    per = S["vitalperiodic"]
    q = f"""SELECT k.stay_id, v.observationoffset, {", ".join(f"v.{c}" for c in per.values())}
            FROM {_csv(src / 'vitalPeriodic.csv.gz')} v JOIN k ON v.patientunitstayid = k.unit
            WHERE TRY_CAST(v.observationoffset AS DOUBLE) BETWEEN {lo} AND {hi}"""
    a = pl.from_arrow(con.execute(q).arrow())
    h = (pl.col("observationoffset").cast(pl.Float64, strict=False) / 60).alias("h")
    v = pl.concat([a.select("stay_id", h, pl.lit(var).alias("var"), pl.col(col).cast(pl.Float64, strict=False)
                            .alias("value")) for var, col in per.items()])
    ap = S["vitalaperiodic"]
    q = f"""SELECT k.stay_id, v.observationoffset, {", ".join(f"v.{c}" for c in ap.values())}
            FROM {_csv(src / 'vitalAperiodic.csv.gz')} v JOIN k ON v.patientunitstayid = k.unit
            WHERE TRY_CAST(v.observationoffset AS DOUBLE) BETWEEN {lo} AND {hi}"""
    b = pl.from_arrow(con.execute(q).arrow())
    v = pl.concat([v] + [b.select("stay_id", h, pl.lit(var).alias("var"), pl.col(col).cast(pl.Float64, strict=False)
                                  .alias("value")) for var, col in ap.items()])
    if S.get("nursecharting_temp"):
        nt = S["nursecharting_temp"]
        q = f"""SELECT k.stay_id, n.nursingchartoffset, n.nursingchartvalue FROM {_csv(src / 'nurseCharting.csv.gz')} n
                JOIN k ON n.patientunitstayid = k.unit
                WHERE n.nursingchartcelltypevallabel = '{nt["label"]}' AND n.nursingchartcelltypevalname = '{nt["name"]}'
                  AND TRY_CAST(n.nursingchartoffset AS DOUBLE) BETWEEN {lo} AND {hi}"""
        c = pl.from_arrow(con.execute(q).arrow())
        v = pl.concat([v, c.select("stay_id", (pl.col("nursingchartoffset").cast(pl.Float64, strict=False) / 60).alias("h"),
                                   pl.lit("temp").alias("var"),
                                   pl.col("nursingchartvalue").cast(pl.Float64, strict=False).alias("value"))])
    name_var = {n: var for var, names in S["lab"].items() for n in names}
    llo, lhi = (int(x * 60) for x in X["labs_h"])
    q = f"""SELECT k.stay_id, l.labresultoffset, l.labname, l.labresult FROM {_csv(src / 'lab.csv.gz')} l
            JOIN k ON l.patientunitstayid = k.unit WHERE l.labname IN ({_in(name_var)})
              AND TRY_CAST(l.labresultoffset AS DOUBLE) BETWEEN {llo} AND {lhi}"""
    lab = pl.from_arrow(con.execute(q).arrow()).select(
        "stay_id", (pl.col("labresultoffset").cast(pl.Float64, strict=False) / 60).alias("h"),
        pl.col("labname").replace_strict(name_var).alias("var"), pl.col("labresult").cast(pl.Float64, strict=False)
        .alias("value"))
    q = f"""SELECT k.stay_id, i.infusionoffset, i.drugname, i.drugrate FROM {_csv(src / 'infusionDrug.csv.gz')} i
            JOIN k ON i.patientunitstayid = k.unit"""
    inf = pl.from_arrow(con.execute(q).arrow()).filter(
        pl.col("drugrate").cast(pl.Float64, strict=False).fill_null(1.0) > 0)
    hinf = (pl.col("infusionoffset").cast(pl.Float64, strict=False) / 60)
    parts = [inf.filter(pl.col("drugname").str.contains(rx)).select(
        "stay_id", pl.lit(kind).alias("kind"), hinf.alias("h"), hinf.alias("h_end"))
        for kind, rx in S["infusiondrug_regex"].items()]
    if S.get("respiratorycharting_vent_labels"):
        q = f"""SELECT k.stay_id, r.respchartoffset FROM {_csv(src / 'respiratoryCharting.csv.gz')} r
                JOIN k ON r.patientunitstayid = k.unit
                WHERE r.respchartvaluelabel IN ({_in(S['respiratorycharting_vent_labels'])})"""
        hr_ = (pl.col("respchartoffset").cast(pl.Float64, strict=False) / 60)
        parts.append(pl.from_arrow(con.execute(q).arrow()).select(
            "stay_id", pl.lit("vent").alias("kind"), hr_.alias("h"), hr_.alias("h_end")))
    if S.get("treatment_vent_regex"):
        q = f"""SELECT k.stay_id, t.treatmentoffset, t.treatmentstring FROM {_csv(src / 'treatment.csv.gz')} t
                JOIN k ON t.patientunitstayid = k.unit"""
        tr = pl.from_arrow(con.execute(q).arrow()).filter(pl.col("treatmentstring").str.contains(S["treatment_vent_regex"]))
        ht = (pl.col("treatmentoffset").cast(pl.Float64, strict=False) / 60)
        parts.append(tr.select("stay_id", pl.lit("vent").alias("kind"), ht.alias("h"), ht.alias("h_end")))
    if S.get("respiratorycare_vent"):
        q = f"""SELECT k.stay_id, r.ventstartoffset, r.ventendoffset FROM {_csv(src / 'respiratoryCare.csv.gz')} r
                JOIN k ON r.patientunitstayid = k.unit"""
        r = pl.from_arrow(con.execute(q).arrow()).select(
            "stay_id", pl.lit("vent").alias("kind"),
            (pl.col("ventstartoffset").cast(pl.Float64, strict=False) / 60).alias("h"),
            (pl.col("ventendoffset").cast(pl.Float64, strict=False) / 60).alias("h_end"))
        parts.append(r.filter(pl.col("h_end") > pl.col("h")))
    con.unregister("k")
    return v, lab, pl.concat(parts)


def _inspire(st: pl.DataFrame, src: Path, cfg: dict, con) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    S, X = cfg["sources"]["inspire"], cfg["extraction"]
    con.register("k", st.select("stay_id", _raw("subject_id", "inspire").alias("subj"), "icu_in_min").to_arrow())
    item_var = {i: v for v, items in S["ward_vitals"].items() for i in items}
    vent = list(S["supports_ward_vitals"]["vent"])
    bp = ["nibp_sbp", "nibp_dbp"]
    lo, hi = (x * 60 for x in X["vitals_h"])
    q = f"""SELECT k.stay_id, w.chart_time, lower(trim(w.item_name)) AS item, w.value, k.icu_in_min
            FROM {_csv(src / 'ward_vitals.csv.gz')} w JOIN k ON w.subject_id = k.subj
            WHERE lower(trim(w.item_name)) IN ({_in(list(item_var) + vent + bp)})
              AND TRY_CAST(w.chart_time AS DOUBLE) BETWEEN k.icu_in_min + {lo} AND k.icu_in_min + {hi}"""
    w = pl.from_arrow(con.execute(q).arrow()).with_columns(
        ((pl.col("chart_time").cast(pl.Float64, strict=False) - pl.col("icu_in_min")) / 60).alias("h"),
        pl.col("value").cast(pl.Float64, strict=False))
    # MAP = (SBP + 2·DBP) / 3 at the time points with SBP and DBP but no charted MAP (T2's rule, lead 2026-10-24)
    pair = (w.filter(pl.col("item").is_in(bp)).drop_nulls("value").group_by("stay_id", "h")
            .agg(pl.col("value").filter(pl.col("item") == "nibp_sbp").first().alias("s"),
                 pl.col("value").filter(pl.col("item") == "nibp_dbp").first().alias("d"))
            .filter(pl.col("s") > pl.col("d")))
    charted = w.filter(pl.col("item").is_in([i for i, v in item_var.items() if v == "map_nibp"])).select(
        "stay_id", "h").unique()
    derived = pair.join(charted, on=["stay_id", "h"], how="anti").select(
        "stay_id", "h", pl.lit("map_nibp").alias("var"), ((pl.col("s") + 2 * pl.col("d")) / 3).alias("value"))
    v = pl.concat([w.filter(pl.col("item").is_in(list(item_var))).select(
        "stay_id", "h", pl.col("item").replace_strict(item_var).alias("var"), "value"), derived])
    on = pl.col("value").is_null() | (pl.col("value") != 0)          # 0 / 1 item: as T2's `active` rule
    sup = [w.filter(pl.col("item").is_in(vent) & on).select("stay_id", pl.lit("vent").alias("kind"), "h",
                                                             pl.col("h").alias("h_end"))]
    name_var = {n: var for var, names in S["labs"].items() for n in names}
    llo, lhi = (x * 60 for x in X["labs_h"])
    q = f"""SELECT k.stay_id, l.chart_time, lower(trim(l.item_name)) AS item, l.value, k.icu_in_min
            FROM {_csv(src / 'labs.csv.gz')} l JOIN k ON l.subject_id = k.subj
            WHERE lower(trim(l.item_name)) IN ({_in(name_var)})
              AND TRY_CAST(l.chart_time AS DOUBLE) BETWEEN k.icu_in_min + {llo} AND k.icu_in_min + {lhi}"""
    lab = pl.from_arrow(con.execute(q).arrow()).select(
        "stay_id", ((pl.col("chart_time").cast(pl.Float64, strict=False) - pl.col("icu_in_min")) / 60).alias("h"),
        pl.col("item").replace_strict(name_var).alias("var"), pl.col("value").cast(pl.Float64, strict=False))
    for kind, atc in S["medications_atc"].items():
        q = f"""SELECT k.stay_id, m.chart_time, k.icu_in_min FROM {_csv(src / 'medications.csv.gz')} m
                JOIN k ON m.subject_id = k.subj WHERE m.atc_code IN ({_in(atc)})
                  AND NOT regexp_matches(coalesce(m.route, ''), '(?i)^ex')"""
        hm = (pl.col("chart_time").cast(pl.Float64, strict=False) - pl.col("icu_in_min")) / 60
        sup.append(pl.from_arrow(con.execute(q).arrow()).select("stay_id", pl.lit(kind).alias("kind"), hm.alias("h"),
                                                                 hm.alias("h_end")))
    con.unregister("k")
    return v, lab, pl.concat(sup)


EXTRACTORS = {"mimic": _mimic, "eicu": _eicu, "inspire": _inspire}


def extract(cohort: str, st: pl.DataFrame, src: Path, cfg: dict, out: Path, threads: int = 4,
            force: bool = False) -> dict[str, pl.DataFrame]:
    """The three long tables of a cohort, cleaned (bounds, windows) and cached."""
    files = {n: out / f"{cohort}_{n}.parquet" for n in ("vitals", "labs", "supports")}
    if not force and all(f.exists() for f in files.values()):
        return {n: pl.read_parquet(f) for n, f in files.items()}
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    LOG.info("hypo: extracting %s from %s", cohort, src)
    v, lab, sup = EXTRACTORS[cohort](st, src, cfg, con)
    B, X = cfg["bounds"], cfg["extraction"]
    bounds = pl.DataFrame({"_b": list(B), "lo": [float(b[0]) for b in B.values()], "hi": [float(b[1]) for b in B.values()]})
    var_b = pl.col("var").str.replace("^map_.*", "map")
    lo_v, hi_v = X["vitals_h"]
    v = (v.drop_nulls().with_columns(var_b.alias("_b"))
         .join(bounds, on="_b", how="inner")
         .filter(pl.col("value").is_between(pl.col("lo"), pl.col("hi")) & pl.col("h").is_between(lo_v, hi_v))
         .select("stay_id", "h", "var", "value"))
    lo_l, hi_l = X["labs_h"]
    lab = (lab.drop_nulls().join(bounds.rename({"_b": "var"}), on="var", how="inner")
           .filter(pl.col("value").is_between(pl.col("lo"), pl.col("hi")) & pl.col("h").is_between(lo_l, hi_l))
           .select("stay_id", "h", "var", "value"))
    sup = (sup.drop_nulls("h").with_columns(pl.col("h_end").fill_null(pl.col("h")))
           .filter((pl.col("h_end") >= lo_v - 24) & (pl.col("h") <= hi_v)))
    tables = {"vitals": v, "labs": lab, "supports": sup}
    for n, df in tables.items():
        df.write_parquet(files[n])
    return tables
