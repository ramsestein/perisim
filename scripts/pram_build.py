"""Pre-registered (OSF; docs/prereg_pram_factors.md §4), Tier 2: the PRAM-like cohort from MIMIC-IV and eICU-CRD.

One row per first ICU stay of an adult with an ICU stay ≥ 24 h and a creatinine in the first 24 h. The anchor is
24 h after ICU admission. PRAM's 42 features come from [admission, admission + 24 h]; acute kidney injury and death
come from (24 h, 192 h].

Usage:  python scripts/pram_build.py [--sites mimic eicu] [--threads 8]
Inputs: <datasets>/mimic4/{hosp,icu}, <datasets>/eicu
Outputs: <workdir>/experiments/pram/{mimic,eicu}.parquet (row level, never published) and build.md (aggregates: counts,
missingness, medians; outcome prevalence only).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl

DATA = Path(os.environ.get("PERISIM_DATASETS", "data"))
WORK = Path(os.environ.get("PERISIM_WORKDIR", "work")) / "experiments" / "pram"

VARS = ["hr", "rr", "spo2", "temp_c", "sbp", "dbp", "gcs_total", "weight_kg", "bun", "calcium", "inr", "scr_feat",
        "bun_cr_ratio"]
FEATURES = ["age", "sex_male", "baseline_scr"] + [f"{v}_{s}" for v in VARS for s in ("min", "max", "last")]
OUTCOMES = ["aki", "death"]
SENSITIVITY = ["aki_kdigo48"]          # deviation 2026-11-08: KDIGO's 48 h window for the 0.3 mg/dL rise
WEIGHT_ITEMS = ["226512", "224639", "226531"]
RANGES = {"hr": (20, 300), "rr": (2, 80), "spo2": (50, 100), "temp_c": (25, 45), "sbp": (30, 300), "dbp": (10, 200),
          "gcs_total": (3, 15), "weight_kg": (20, 300), "bun": (1, 300), "calcium": (2, 20), "inr": (0.5, 20),
          "scr": (0.1, 30)}
MIMIC_YEARS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"]
MIMIC_CHART = {"220045": "hr", "220210": "rr", "224690": "rr", "220277": "spo2", "223762": "temp_c",
               "223761": "temp_f", "220050": "sbp", "220179": "sbp", "220051": "dbp", "220180": "dbp",
               "220739": "gcs_e", "223900": "gcs_v", "223901": "gcs_m", "226512": "weight_kg", "224639": "weight_kg",
               "226531": "weight_lb"}
MIMIC_LAB = {"51006": "bun", "50893": "calcium", "51237": "inr", "50912": "scr"}
EICU_LAB = {"BUN": "bun", "calcium": "calcium", "PT - INR": "inr", "creatinine": "scr"}
H24, H192 = 24.0, 192.0


def csv(p: Path) -> str:
    return f"read_csv('{p.as_posix()}', header=true, all_varchar=true, ignore_errors=true)"


def q(con, sql: str) -> pl.DataFrame:
    return pl.from_arrow(con.execute(sql).fetch_arrow_table())


def in_range(df: pl.DataFrame) -> pl.DataFrame:
    """Values outside the plausibility ranges become missing (dropped from the long table)."""
    rng = pl.DataFrame({"var": list(RANGES), "lo": [float(v[0]) for v in RANGES.values()],
                        "hi": [float(v[1]) for v in RANGES.values()]})
    return df.join(rng, on="var").filter(pl.col("value").is_between(pl.col("lo"), pl.col("hi"))).drop("lo", "hi")


# --------------------------------------------------------------------------- aggregation (shared)
def ratio(labs: pl.DataFrame) -> pl.DataFrame:
    """BUN / creatinine of the same specimen (`spec`), as a long table."""
    b = labs.filter(pl.col("var") == "bun").group_by("id", "spec").agg(pl.col("h").min(), pl.col("value").mean().alias("b"))
    c = labs.filter(pl.col("var") == "scr").group_by("id", "spec").agg(pl.col("value").mean().alias("c"))
    return b.join(c, on=["id", "spec"]).select("id", "h", pl.lit("bun_cr_ratio").alias("var"),
                                               (pl.col("b") / pl.col("c")).alias("value"))


def aggregate(long: pl.DataFrame) -> pl.DataFrame:
    """min, max and last value per stay and variable, wide."""
    a = (long.sort("id", "var", "h", "value")
         .group_by("id", "var").agg(pl.col("value").min().alias("min"), pl.col("value").max().alias("max"),
                                    pl.col("value").last().alias("last")))
    out = None
    for s in ("min", "max", "last"):
        w = a.select("id", "var", s).pivot(on="var", index="id", values=s)
        w = w.rename({c: f"{c}_{s}" for c in w.columns if c != "id"})
        out = w if out is None else out.join(w, on="id", how="full", coalesce=True)
    return out


def assemble(st: pl.DataFrame, feat_long: pl.DataFrame, scr_after: pl.DataFrame) -> pl.DataFrame:
    """Stays + aggregated features + outcomes; drops stays with no creatinine in the first 24 h."""
    W = aggregate(feat_long.with_columns(pl.col("var").replace({"scr": "scr_feat"})))
    df = st.join(W, on="id", how="left")
    for c in FEATURES:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, pl.Float64).alias(c))
    df = df.with_columns(pl.col("scr_feat_min").alias("baseline_scr")).filter(pl.col("baseline_scr").is_not_null())
    peak = scr_after.group_by("id").agg(pl.col("value").max().alias("_peak"),
                                        pl.col("value").filter(pl.col("h") <= H24 + 48).max().alias("_peak48"))
    df = df.join(peak, on="id", how="left").with_columns(
        ((pl.col("_peak") >= pl.col("baseline_scr") + 0.3) | (pl.col("_peak") >= 1.5 * pl.col("baseline_scr")))
        .fill_null(False).alias("aki"),
        ((pl.col("_peak48") >= pl.col("baseline_scr") + 0.3) | (pl.col("_peak") >= 1.5 * pl.col("baseline_scr")))
        .fill_null(False).alias("aki_kdigo48"),
        (pl.col("death_h").is_not_null() & (pl.col("death_h") > H24) & (pl.col("death_h") <= H192)).alias("death"))
    return df.select(["id", "site"] + FEATURES + OUTCOMES + SENSITIVITY + ["n_stays_before_filters"]).with_columns(
        [pl.col(c).cast(pl.Float64) for c in FEATURES])


# --------------------------------------------------------------------------- MIMIC-IV
def mimic(con) -> tuple[pl.DataFrame, dict]:
    d = DATA / "mimic4"
    ic = q(con, f"SELECT subject_id, hadm_id, stay_id, intime, outtime FROM {csv(d / 'icu' / 'icustays.csv.gz')}")
    pt = q(con, f"SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group FROM {csv(d / 'hosp' / 'patients.csv.gz')}")
    ad = q(con, f"SELECT hadm_id, dischtime, deathtime, hospital_expire_flag FROM {csv(d / 'hosp' / 'admissions.csv.gz')}")
    t = lambda c: pl.col(c).str.to_datetime(strict=False)  # noqa: E731
    st = (ic.with_columns(t("intime").alias("tin"), t("outtime").alias("tout"))
          .sort("tin").group_by("subject_id", maintain_order=True).first()          # the first ICU stay of each patient
          .join(pt, on="subject_id", how="left").join(ad, on="hadm_id", how="left"))
    flow = {"first_stays": st.height}
    st = st.filter(pl.col("anchor_year_group").is_in(MIMIC_YEARS))
    flow["in_anchor_years"] = st.height
    st = st.with_columns((pl.col("anchor_age").cast(pl.Float64) + pl.col("tin").dt.year()
                          - pl.col("anchor_year").cast(pl.Float64)).alias("age")).filter(pl.col("age") >= 18)
    flow["adults"] = st.height
    st = st.filter((pl.col("tout") - pl.col("tin")).dt.total_seconds() >= H24 * 3600)
    flow["icu_24h"] = st.height
    death = pl.when(pl.col("deathtime").is_not_null()).then(t("deathtime")) \
        .when(pl.col("hospital_expire_flag") == "1").then(t("dischtime")).otherwise(None)
    st = st.with_columns(((death - pl.col("tin")).dt.total_seconds() / 3600).alias("death_h"),
                         pl.col("stay_id").alias("id"), pl.lit("mimic").alias("site"),
                         (pl.col("gender") == "M").cast(pl.Float64).alias("sex_male"),
                         pl.lit(flow["icu_24h"]).alias("n_stays_before_filters"))
    con.register("k", st.select("id", "subject_id", pl.col("tin").cast(pl.Utf8).alias("tin")).to_arrow())
    ce = q(con, f"""SELECT k.id, c.itemid, c.charttime, c.valuenum, k.tin FROM {csv(d / 'icu' / 'chartevents.csv.gz')} c
                    JOIN k ON c.stay_id = k.id WHERE c.itemid IN ({', '.join(repr(i) for i in MIMIC_CHART)})
                    AND c.valuenum IS NOT NULL
                    AND TRY_CAST(c.charttime AS TIMESTAMP) <= CAST(k.tin AS TIMESTAMP) + INTERVAL 24 HOUR
                    AND (TRY_CAST(c.charttime AS TIMESTAMP) >= CAST(k.tin AS TIMESTAMP)
                         OR c.itemid IN ({', '.join(repr(i) for i in WEIGHT_ITEMS)}))""")
    lab = q(con, f"""SELECT k.id, l.itemid, l.specimen_id, l.charttime, l.valuenum, k.tin FROM {csv(d / 'hosp' / 'labevents.csv.gz')} l
                     JOIN k ON l.subject_id = k.subject_id WHERE l.itemid IN ({', '.join(repr(i) for i in MIMIC_LAB)})
                     AND l.valuenum IS NOT NULL
                     AND TRY_CAST(l.charttime AS TIMESTAMP) BETWEEN CAST(k.tin AS TIMESTAMP)
                         AND CAST(k.tin AS TIMESTAMP) + INTERVAL 192 HOUR""")
    con.unregister("k")
    hours = lambda c: (pl.col(c).str.to_datetime(strict=False) - pl.col("tin").str.to_datetime(strict=False)).dt.total_seconds() / 3600  # noqa: E731
    ce = ce.select("id", hours("charttime").alias("h"), pl.col("itemid").replace_strict(MIMIC_CHART).alias("var"),
                   pl.col("valuenum").cast(pl.Float64, strict=False).alias("value")).drop_nulls()
    ce = ce.with_columns(
        pl.when(pl.col("var") == "temp_f").then((pl.col("value") - 32) * 5 / 9)
        .when(pl.col("var") == "weight_lb").then(pl.col("value") * 0.453592).otherwise(pl.col("value")).alias("value"),
        pl.col("var").replace({"temp_f": "temp_c", "weight_lb": "weight_kg"}).alias("var"))
    gcs = (ce.filter(pl.col("var").is_in(["gcs_e", "gcs_v", "gcs_m"])).group_by("id", "h", "var")
           .agg(pl.col("value").mean()).pivot(on="var", index=["id", "h"], values="value"))
    for c in ("gcs_e", "gcs_v", "gcs_m"):
        if c not in gcs.columns:
            gcs = gcs.with_columns(pl.lit(None, pl.Float64).alias(c))
    gcs = gcs.drop_nulls(["gcs_e", "gcs_v", "gcs_m"]).select(
        "id", "h", pl.lit("gcs_total").alias("var"), (pl.col("gcs_e") + pl.col("gcs_v") + pl.col("gcs_m")).alias("value"))
    vit = pl.concat([ce.filter(~pl.col("var").str.starts_with("gcs_")), gcs])
    lab = lab.select("id", hours("charttime").alias("h"), pl.col("itemid").replace_strict(MIMIC_LAB).alias("var"),
                     pl.col("specimen_id").alias("spec"), pl.col("valuenum").cast(pl.Float64, strict=False).alias("value")
                     ).drop_nulls()
    return finish(st, vit, lab, flow)


# --------------------------------------------------------------------------- eICU-CRD
def eicu(con) -> tuple[pl.DataFrame, dict]:
    d = DATA / "eicu"
    pt = q(con, f"""SELECT patientunitstayid, patienthealthsystemstayid, uniquepid, gender, age, hospitaldischargeyear,
                    unitvisitnumber, unitdischargeoffset, hospitaldischargeoffset, hospitaldischargestatus,
                    admissionweight FROM {csv(d / 'patient.csv.gz')}""")
    num = lambda c: pl.col(c).cast(pl.Float64, strict=False)  # noqa: E731
    st = (pt.with_columns(num("hospitaldischargeyear").alias("_y"), num("patienthealthsystemstayid").alias("_h"),
                          num("unitvisitnumber").alias("_v"))
          .sort("_y", "_h", "_v", nulls_last=True).group_by("uniquepid", maintain_order=True).first())
    flow = {"first_stays": st.height}
    st = st.with_columns(pl.when(pl.col("age") == "> 89").then(90.0).otherwise(num("age")).alias("age"))
    st = st.filter(pl.col("age") >= 18)
    flow["adults"] = st.height
    st = st.filter(num("unitdischargeoffset") >= H24 * 60)
    flow["icu_24h"] = st.height
    st = st.with_columns(
        pl.col("patientunitstayid").alias("id"), pl.lit("eicu").alias("site"),
        (pl.col("gender") == "Male").cast(pl.Float64).alias("sex_male"),
        pl.when(pl.col("hospitaldischargestatus") == "Expired").then(num("hospitaldischargeoffset") / 60)
        .otherwise(None).alias("death_h"), pl.lit(flow["icu_24h"]).alias("n_stays_before_filters"))
    con.register("k", st.select("id").to_arrow())
    vp = q(con, f"""SELECT v.patientunitstayid AS id, v.observationoffset AS o, v.heartrate AS hr, v.respiration AS rr,
                    v.sao2 AS spo2, v.temperature AS temp_c, v.systemicsystolic AS sbp, v.systemicdiastolic AS dbp
                    FROM {csv(d / 'vitalPeriodic.csv.gz')} v JOIN k ON v.patientunitstayid = k.id
                    WHERE TRY_CAST(v.observationoffset AS DOUBLE) BETWEEN 0 AND 1440""")
    va = q(con, f"""SELECT v.patientunitstayid AS id, v.observationoffset AS o, v.noninvasivesystolic AS sbp,
                    v.noninvasivediastolic AS dbp FROM {csv(d / 'vitalAperiodic.csv.gz')} v JOIN k ON v.patientunitstayid = k.id
                    WHERE TRY_CAST(v.observationoffset AS DOUBLE) BETWEEN 0 AND 1440""")
    gc = q(con, f"""SELECT n.patientunitstayid AS id, n.nursingchartoffset AS o, n.nursingchartvalue AS gcs_total
                    FROM {csv(d / 'nurseCharting.csv.gz')} n JOIN k ON n.patientunitstayid = k.id
                    WHERE n.nursingchartcelltypevallabel = 'Glasgow coma score' AND n.nursingchartcelltypevalname = 'GCS Total'
                    AND TRY_CAST(n.nursingchartoffset AS DOUBLE) BETWEEN 0 AND 1440""")
    gcomp = q(con, f"""SELECT n.patientunitstayid AS id, n.nursingchartoffset AS o, n.nursingchartcelltypevalname AS part,
                       n.nursingchartvalue AS v FROM {csv(d / 'nurseCharting.csv.gz')} n JOIN k ON n.patientunitstayid = k.id
                       WHERE n.nursingchartcelltypevallabel = 'Glasgow coma score'
                       AND n.nursingchartcelltypevalname IN ('Eyes', 'Verbal', 'Motor')
                       AND TRY_CAST(n.nursingchartoffset AS DOUBLE) BETWEEN 0 AND 1440""")
    pe = q(con, f"""SELECT p.patientunitstayid AS id, p.physicalexamoffset AS o,
                    regexp_extract(p.physicalexampath, 'GCS/(Eyes|Verbal|Motor) Score', 1) AS part, p.physicalexamvalue AS v
                    FROM {csv(d / 'physicalExam.csv.gz')} p JOIN k ON p.patientunitstayid = k.id
                    WHERE regexp_matches(p.physicalexampath, 'Neurologic/GCS/(Eyes|Verbal|Motor) Score/')
                    AND TRY_CAST(p.physicalexamoffset AS DOUBLE) BETWEEN 0 AND 1440""")
    lb = q(con, f"""SELECT l.patientunitstayid AS id, l.labresultoffset AS o, l.labname, l.labresult
                    FROM {csv(d / 'lab.csv.gz')} l JOIN k ON l.patientunitstayid = k.id
                    WHERE l.labname IN ({', '.join(repr(n) for n in EICU_LAB)})
                    AND TRY_CAST(l.labresultoffset AS DOUBLE) BETWEEN 0 AND 11520""")
    con.unregister("k")

    def melt(df, cols):
        return (df.unpivot(index=["id", "o"], on=cols, variable_name="var", value_name="value")
                .select("id", (num("o") / 60).alias("h"), "var", num("value").alias("value")).drop_nulls())

    comp = (pl.concat([gcomp, pe]).select("id", (num("o") / 60).alias("h"), "part", num("v").alias("v")).drop_nulls()
            .group_by("id", "h", "part").agg(pl.col("v").mean()).pivot(on="part", index=["id", "h"], values="v"))
    for c in ("Eyes", "Verbal", "Motor"):
        if c not in comp.columns:
            comp = comp.with_columns(pl.lit(None, pl.Float64).alias(c))
    comp = comp.drop_nulls(["Eyes", "Verbal", "Motor"]).select(
        "id", "h", pl.lit("gcs_total").alias("var"), (pl.col("Eyes") + pl.col("Verbal") + pl.col("Motor")).alias("value"))
    vit = pl.concat([melt(vp, ["hr", "rr", "spo2", "temp_c", "sbp", "dbp"]), melt(va, ["sbp", "dbp"]),
                     melt(gc, ["gcs_total"]), comp,
                     st.select("id", pl.lit(0.0).alias("h"), pl.lit("weight_kg").alias("var"),
                               num("admissionweight").alias("value")).drop_nulls()])
    lab = lb.select("id", (num("o") / 60).alias("h"), pl.col("labname").replace_strict(EICU_LAB).alias("var"),
                    pl.col("o").alias("spec"), num("labresult").alias("value")).drop_nulls()
    return finish(st, vit, lab, flow)


def finish(st: pl.DataFrame, vit: pl.DataFrame, lab: pl.DataFrame, flow: dict) -> tuple[pl.DataFrame, dict]:
    st = st.with_columns(pl.col("id").cast(pl.Utf8), pl.col("age").cast(pl.Float64))
    vit = in_range(vit.with_columns(pl.col("id").cast(pl.Utf8)).filter(
        pl.col("h").is_between(0, H24) | ((pl.col("var") == "weight_kg") & (pl.col("h") <= H24))))
    lab = in_range(lab.with_columns(pl.col("id").cast(pl.Utf8), pl.col("spec").cast(pl.Utf8)))
    lab24 = lab.filter(pl.col("h").is_between(0, H24))
    feat = pl.concat([vit, lab24.drop("spec"), ratio(lab24)], how="diagonal_relaxed").select("id", "h", "var", "value")
    scr_after = lab.filter((pl.col("var") == "scr") & (pl.col("h") > H24) & (pl.col("h") <= H192))
    df = assemble(st, feat, scr_after)
    flow["with_creatinine_24h"] = df.height
    return df, flow


# --------------------------------------------------------------------------- report
def report(res: dict) -> str:
    L = ["# PRAM-like cohort (Tier 2): build summary (aggregates)", "",
         "Pre-registered: docs/prereg_pram_factors.md §4. Counts are stays.", "", "## Flow", "",
         "| step | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    steps = ["first_stays", "in_anchor_years", "adults", "icu_24h", "with_creatinine_24h"]
    for s in steps:
        L.append(f"| {s} | " + " | ".join(f"{r['flow'][s]:,}" if s in r["flow"] else "—" for r in res.values()) + " |")
    L += ["", "## Outcomes", "", "| outcome | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    for o in OUTCOMES + SENSITIVITY:
        L.append(f"| {o} | " + " | ".join(f"{r['df'][o].mean():.1%}" for r in res.values()) + " |")
    L += ["", "## Features: missing (%) and median [IQR]", "",
          "| feature | " + " | ".join(f"{s} missing | {s} median [IQR]" for s in res) + " |",
          "|---|" + "---|---|" * len(res)]
    for c in FEATURES:
        cells = []
        for r in res.values():
            v = r["df"][c].drop_nulls()
            cells.append(f"{100 * (1 - v.len() / r['df'].height):.1f}")
            cells.append(f"{v.median():.2f} [{v.quantile(0.25):.2f}, {v.quantile(0.75):.2f}]" if v.len() else "—")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="*", default=["mimic", "eicu"])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    import duckdb

    WORK.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {int(a.threads)}")
    res = {}
    for s in a.sites:
        df, flow = (mimic if s == "mimic" else eicu)(con)
        df.write_parquet(WORK / f"{s}.parquet")
        res[s] = {"df": df, "flow": flow}
        print(f"pram build {s}: {df.height:,} stays")
    (WORK / "build.md").write_text(report(res), encoding="utf-8")
    print(f"pram build: {WORK / 'build.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
