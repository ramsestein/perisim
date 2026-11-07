"""NWICU → `hypo` landmarks (pre-registration: docs/prereg_nwicu.md).

Extracts the three long tables of `hypo` (vitals, labs, supports; hours after the ICU admission) and the population
from NWICU's original tables, then builds the landmarks with v0.6's landmark code (`hypo.landmarks`, records before L
only). Patient-level outputs stay in the work folder; the report holds aggregates only.

Usage:  python scripts/nwicu_build.py [--threads 8]
Inputs: <datasets>/nwicu/northwestern-icu-nwicu-database-0.1.0/data/{nw_icu,nw_hosp}
Outputs: <workdir>/experiments/nwicu/{stays,vitals,labs,supports}.parquet, landmarks/nwicu.parquet, build.md
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import polars as pl
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path(os.environ.get("PERISIM_DATASETS", "data")) / "nwicu" / "northwestern-icu-nwicu-database-0.1.0" / "data"
WORK = Path(os.environ.get("PERISIM_WORKDIR", "work")) / "experiments" / "nwicu"

ITEMS = {"art_sbp": "320050", "art_dbp": "320051", "nibp_sbp": "320179", "nibp_dbp": "320180", "hr": "320045",
         "rr": "320210", "spo2": "320277", "temp_f": "323761"}
LABS = {"100007": "hb", "100002": "creatinine", "100031": "lactate"}
VASO = r"(?i)(norepinephrine|epinephrine|phenylephrine|vasopressin|dopamine)"
VASO_NOT = r"(?i)(lidocaine|bupivacaine|ropivacaine|racepinephrine|articaine|prilocaine|nebu|inh)"
SEDATION = r"(?i)(propofol|dexmedetomidine|midazolam)"
SURGICAL_ICU = ["SICU", "CTICU"]
EMERGENCY = ["EMERGENCY", "URGENT", "TRAUMA"]


def landmarks_module():
    """v0.6's landmark code: hypo/landmarks.py beside this script (as nwicu_landmarks.py) or the repository's."""
    for p in (HERE / "nwicu_landmarks.py", ROOT / "src" / "hypo" / "landmarks.py"):
        if p.exists():
            spec = importlib.util.spec_from_file_location("nwicu_landmarks", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            if hasattr(m, "SupportsBefore"):
                return m
    raise RuntimeError("v0.6's hypo/landmarks.py (with SupportsBefore) not found")


def config() -> dict:
    for p in (HERE / "hypo_v06.yaml", ROOT / "config" / "hypo.yaml"):
        if p.exists():
            cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
            if "icu_death_tolerance_min" in cfg["landmarks"]:
                cfg.setdefault("model", {}).setdefault("source_missing", {})["nwicu"] = ["vent_on"]
                return cfg
    raise RuntimeError("v0.6's config/hypo.yaml not found")


def csv(p: Path) -> str:
    return f"read_csv('{p.as_posix()}', header=true, all_varchar=true, ignore_errors=true)"


def epoch_min(col: str) -> pl.Expr:
    return pl.col(col).str.to_datetime(strict=False).dt.epoch("s") / 60


def stays(con) -> pl.DataFrame:
    st = pl.from_arrow(con.execute(f"SELECT * FROM {csv(DATA / 'nw_icu' / 'icustays.csv.gz')}").arrow())
    pt = pl.from_arrow(con.execute(f"SELECT subject_id, gender, anchor_age FROM {csv(DATA / 'nw_hosp' / 'patients.csv.gz')}").arrow())
    ad = pl.from_arrow(con.execute(f"SELECT hadm_id, admission_type, deathtime FROM {csv(DATA / 'nw_hosp' / 'admissions.csv.gz')}").arrow())
    s = (st.join(pt, on="subject_id", how="left").join(ad, on="hadm_id", how="left")
         .with_columns(epoch_min("intime").alias("icu_in_min"), epoch_min("outtime").alias("icu_out_min"),
                       epoch_min("deathtime").alias("death_min"), pl.col("anchor_age").cast(pl.Float64).alias("age"))
         .filter(pl.col("first_careunit").is_in(SURGICAL_ICU) & (pl.col("age") >= 18) & pl.col("icu_in_min").is_not_null())
         .sort("icu_in_min").group_by("hadm_id", maintain_order=True).first())
    return s.select(
        pl.col("stay_id").cast(pl.Utf8), pl.col("subject_id").cast(pl.Utf8), pl.col("hadm_id").cast(pl.Utf8),
        "icu_in_min", ((pl.col("icu_out_min") - pl.col("icu_in_min")) / 60).alias("los_h"),
        ((pl.col("death_min") - pl.col("icu_in_min")) / 60).alias("death_h"),
        pl.lit(None, pl.Float64).alias("limitation_h"), pl.lit(None, pl.Float64).alias("surgery_duration_h"),
        pl.lit(None, pl.Float64).alias("surgery_end_to_icu_h"), "age",
        (pl.col("gender") == "M").alias("sex_male"), pl.lit(None, pl.Float64).alias("bmi"),
        pl.col("admission_type").is_in(EMERGENCY).alias("emergency"), pl.lit("unknown").alias("surgical_group"),
        pl.lit(None, pl.Utf8).alias("anesthesia_type"), "first_careunit")


def tables(con, st: pl.DataFrame, cfg: dict) -> dict[str, pl.DataFrame]:
    con.register("k", st.select("stay_id", "hadm_id", "icu_in_min").to_arrow())

    def hours(col: str) -> pl.Expr:
        return (epoch_min(col) - pl.col("icu_in_min")) / 60

    ce = pl.from_arrow(con.execute(
        f"""SELECT k.stay_id, c.itemid, c.charttime, c.valuenum, k.icu_in_min FROM {csv(DATA / 'nw_icu' / 'chartevents.csv.gz')} c
            JOIN k ON c.stay_id = k.stay_id WHERE c.itemid IN ({', '.join(repr(v) for v in ITEMS.values())})
            AND c.valuenum IS NOT NULL""").arrow())
    name = {v: k for k, v in ITEMS.items()}
    ce = ce.select("stay_id", hours("charttime").alias("h"), pl.col("itemid").replace_strict(name).alias("item"),
                   pl.col("valuenum").cast(pl.Float64, strict=False).alias("value")).drop_nulls()
    # MAP from SBP and DBP charted at the same time (prereg deviation 1)
    bp = []
    for kind, s, d in (("map_inv", "art_sbp", "art_dbp"), ("map_nibp", "nibp_sbp", "nibp_dbp")):
        a = ce.filter(pl.col("item") == s).group_by("stay_id", "h").agg(pl.col("value").median().alias("sbp"))
        b = ce.filter(pl.col("item") == d).group_by("stay_id", "h").agg(pl.col("value").median().alias("dbp"))
        bp.append(a.join(b, on=["stay_id", "h"]).filter(pl.col("sbp") > pl.col("dbp"))
                  .select("stay_id", "h", pl.lit(kind).alias("var"), ((pl.col("sbp") + 2 * pl.col("dbp")) / 3).alias("value")))
    other = ce.filter(pl.col("item").is_in(["hr", "rr", "spo2", "temp_f"])).with_columns(
        pl.when(pl.col("item") == "temp_f").then((pl.col("value") - 32) * 5 / 9).otherwise(pl.col("value")).alias("value"),
        pl.col("item").replace({"temp_f": "temp"}).alias("var")).select("stay_id", "h", "var", "value")
    v = pl.concat(bp + [other])
    lab = pl.from_arrow(con.execute(
        f"""SELECT k.stay_id, l.itemid, l.charttime, l.storetime, l.valuenum, k.icu_in_min
            FROM {csv(DATA / 'nw_hosp' / 'labevents.csv.gz')} l JOIN k ON l.hadm_id = k.hadm_id
            WHERE l.itemid IN ({', '.join(repr(i) for i in LABS)}) AND l.valuenum IS NOT NULL""").arrow())
    lab = lab.with_columns(pl.coalesce(pl.col("storetime"), pl.col("charttime")).alias("t")).select(
        "stay_id", hours("t").alias("h"), pl.col("itemid").replace_strict(LABS).alias("var"),
        pl.col("valuenum").cast(pl.Float64, strict=False).alias("value"))
    em = pl.from_arrow(con.execute(
        f"""SELECT k.stay_id, e.charttime, e.medication, k.icu_in_min FROM {csv(DATA / 'nw_hosp' / 'emar.csv.gz')} e
            JOIN k ON e.hadm_id = k.hadm_id WHERE e.event_txt IN ('Confirmed', 'Applied')
            AND (regexp_matches(e.medication, '{VASO[4:]}', 'i') OR regexp_matches(e.medication, '{SEDATION[4:]}', 'i'))""").arrow())
    med = pl.col("medication")
    kind = (pl.when(med.str.contains(VASO) & ~med.str.contains(VASO_NOT)).then(pl.lit("vaso"))
            .when(med.str.contains(SEDATION)).then(pl.lit("sedation")).otherwise(None))
    sup = em.with_columns(kind.alias("kind"), hours("charttime").alias("h")).drop_nulls(["kind", "h"]) \
        .select("stay_id", "kind", "h", pl.col("h").alias("h_end"))
    con.unregister("k")
    # cleaning as hypo.extract: bounds and windows
    B, X = cfg["bounds"], cfg["extraction"]
    bounds = pl.DataFrame({"_b": list(B), "lo": [float(b[0]) for b in B.values()], "hi": [float(b[1]) for b in B.values()]})
    lo_v, hi_v = X["vitals_h"]
    v = (v.drop_nulls().with_columns(pl.col("var").str.replace("^map_.*", "map").alias("_b")).join(bounds, on="_b")
         .filter(pl.col("value").is_between(pl.col("lo"), pl.col("hi")) & pl.col("h").is_between(lo_v, hi_v))
         .select("stay_id", "h", "var", "value"))
    lo_l, hi_l = X["labs_h"]
    lab = (lab.drop_nulls().join(bounds.rename({"_b": "var"}), on="var")
           .filter(pl.col("value").is_between(pl.col("lo"), pl.col("hi")) & pl.col("h").is_between(lo_l, hi_l))
           .select("stay_id", "h", "var", "value"))
    sup = sup.filter((pl.col("h_end") >= lo_v - 24) & (pl.col("h") <= hi_v))
    return {"vitals": v, "labs": lab, "supports": sup}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    import duckdb

    LM, cfg = landmarks_module(), config()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "landmarks").mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {int(a.threads)}")
    st = stays(con)
    T = tables(con, st, cfg)
    st.write_parquet(WORK / "stays.parquet")
    for n, df in T.items():
        df.write_parquet(WORK / f"{n}.parquet")
    lm = LM.build(st, LM.hourly(T["vitals"]), T["labs"], T["supports"], cfg, cohort="nwicu")
    lm.write_parquet(WORK / "landmarks" / "nwicu.parquet")
    an = lm.filter(pl.col("normotensive") & pl.col("known"))
    L = ["# NWICU build (aggregates)", "", f"- surgical-ICU stays (first per hospitalisation, adults): {st.height:,}",
         f"- of which SICU / CTICU: {st.filter(pl.col('first_careunit') == 'SICU').height:,} / "
         f"{st.filter(pl.col('first_careunit') == 'CTICU').height:,}",
         f"- landmark rows: {lm.height:,}; analysis landmarks (normotensive, window known): {an.height:,} "
         f"from {an['stay_id'].n_unique():,} stays",
         f"- supports records: vaso {T['supports'].filter(pl.col('kind') == 'vaso').height:,}, sedation "
         f"{T['supports'].filter(pl.col('kind') == 'sedation').height:,}",
         f"- vitals records: " + ", ".join(f"{k} {n:,}" for k, n in T["vitals"].group_by("var").len().sort("var").iter_rows()),
         "", "Outcomes are not summarised here; they are read only by scripts/nwicu_transport.py."]
    (WORK / "build.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"nwicu: {WORK / 'build.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
