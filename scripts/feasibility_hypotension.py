"""Feasibility of the next use case: hypotension in the next 6 h in the ICU after surgery.

Reads the ORIGINAL tables (T2's common format holds only T2's index stays and, in MIMIC-IV / eICU, the vitals of a
window before the ICU discharge): the mean arterial pressure (invasive and non-invasive), heart rate, SpO₂ and the
vasopressor infusions of every postoperative adult ICU stay (the first of each hospitalization, ICU deaths included)
over its first 72 h. Per cohort:
  - measurement: MAP readings per hour, share of hours with a MAP, share of hours with an invasive (arterial) MAP;
  - landmarks every 2 h from 2 h to 48 h after the ICU admission (in the ICU and alive, a MAP in the last hour):
    share with hypotension in the next 6 h (an hour whose median MAP is < 65 mmHg), among all and among those
    normotensive in the last hour, by vasopressor status, and with alternative definitions (any reading < 65; one
    reading per hour, as hourly charting would see it) to size the effect of the measurement density;
  - the composite "hypotension or a vasopressor started" among those off vasopressors; tertile cuts of the lowest
    hourly MAP of the next 6 h (an ordinal target that is never rare);
  - inputs at the landmark (heart rate, SpO₂, a MAP trend); eICU: patients per hospital and the spread of the
    hospitals' rates.

Usage:  python scripts/feasibility_hypotension.py [--profile local] [--threads 4]
Patient-level extracts are cached in work/feasibility/hypotension/ (git-ignored, never exported); the report
(feasibility_hypotension.md / .json, shares and counts ≥ 10 only) is written next to them. Never touches src/ or
config/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

COHORTS = ("inspire", "mimic", "eicu")
HOURS = 72                                  # extracted window after the ICU admission
LANDMARKS = list(range(2, 49, 2))
HORIZON = 6
THRESH = 65.0
MAP_BOUNDS = (10.0, 250.0)
MIN_N = 10
MIMIC_ITEMS = {"220052": "map_inv", "225312": "map_inv", "220181": "map_nibp", "220045": "hr", "220277": "spo2"}
MIMIC_VASO = ["221906", "221289", "221749", "222315", "221662"]
VASO_REGEX = "(?i)(norepinephrine|levophed|epinephrine|adrenalin|phenylephrine|neo-?synephrine|vasopressin|dopamine)"
INSPIRE_VASO_ATC = ["C01CA03", "C01CA24", "C01CA04", "C01CA06", "H01BA01"]


def _raw(col: str, prefix: str) -> pl.Expr:
    return pl.col(col).str.slice(len(prefix) + 1)


# --------------------------------------------------------------------------- extraction (cached, patient level)
def first_postop(c: str) -> pl.DataFrame:
    """Postoperative adult ICU stays, the first of each hospitalization (T2's first cohort steps; ICU deaths kept)."""
    s = pl.read_parquet(ROOT / "work" / "common" / c / "stays_candidates.parquet")
    p = s.filter(pl.col("postop").fill_null(False)).sort("icu_in_min")
    return p.group_by("hadm_id", maintain_order=True).first()


def stays(c: str) -> pl.DataFrame:
    p = first_postop(c)
    return p.select("stay_id", "hadm_id", "subject_id", "icu_in_min", "icu_out_min", "death_min", "surgical_group",
                    ((pl.col("icu_out_min") - pl.col("icu_in_min")) / 60).alias("los_h"),
                    ((pl.col("death_min") - pl.col("icu_in_min")) / 60).alias("death_h"))


def extract(c: str, st: pl.DataFrame, src: Path, cache: Path, con) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Long vitals (stay_id, h, var, value) and vasopressor records (stay_id, h, h_end) over the first HOURS."""
    fv, fp = cache / f"{c}_vitals.parquet", cache / f"{c}_vaso.parquet"
    if fv.exists() and fp.exists():
        return pl.read_parquet(fv), pl.read_parquet(fp)
    if c == "mimic":
        keys = st.select("stay_id", _raw("hadm_id", "MIMIC").alias("hadm"), "icu_in_min")
        con.register("k", keys.to_arrow())
        q = f"""SELECT k.stay_id, c.itemid, c.charttime, c.valuenum, k.icu_in_min
                FROM read_csv('{src / 'icu' / 'chartevents.csv.gz'}', header=true, all_varchar=true) c
                JOIN k ON c.hadm_id = k.hadm
                WHERE c.itemid IN ({", ".join(f"'{i}'" for i in MIMIC_ITEMS)}) AND c.valuenum IS NOT NULL"""
        v = pl.from_arrow(con.execute(q).arrow())
        v = v.with_columns(((pl.col("charttime").str.to_datetime(strict=False).dt.epoch("s") / 60
                             - pl.col("icu_in_min")) / 60).alias("h"),
                           pl.col("itemid").replace_strict(MIMIC_ITEMS).alias("var"),
                           pl.col("valuenum").cast(pl.Float64, strict=False).alias("value"))
        q = f"""SELECT k.stay_id, i.starttime, i.endtime, k.icu_in_min
                FROM read_csv('{src / 'icu' / 'inputevents.csv.gz'}', header=true, all_varchar=true) i
                JOIN k ON i.hadm_id = k.hadm
                WHERE i.itemid IN ({", ".join(f"'{i}'" for i in MIMIC_VASO)})
                  AND coalesce(i.statusdescription, '') <> 'Rewritten'"""
        p = pl.from_arrow(con.execute(q).arrow())

        def hrs(col):
            return (pl.col(col).str.to_datetime(strict=False).dt.epoch("s") / 60 - pl.col("icu_in_min")) / 60

        p = p.select("stay_id", hrs("starttime").alias("h"), hrs("endtime").alias("h_end"))
    elif c == "eicu":
        keys = st.select("stay_id", _raw("stay_id", "EICU").alias("unit"))
        con.register("k", keys.to_arrow())
        q = f"""SELECT k.stay_id, v.observationoffset, v.systemicmean, v.heartrate, v.sao2
                FROM read_csv('{src / 'vitalPeriodic.csv.gz'}', header=true, all_varchar=true) v
                JOIN k ON v.patientunitstayid = k.unit
                WHERE TRY_CAST(v.observationoffset AS DOUBLE) BETWEEN -120 AND {HOURS * 60}"""
        a = pl.from_arrow(con.execute(q).arrow()).with_columns(
            (pl.col("observationoffset").cast(pl.Float64, strict=False) / 60).alias("h"))
        v = pl.concat([a.select("stay_id", "h", pl.lit(n).alias("var"), pl.col(col).cast(pl.Float64, strict=False)
                                .alias("value"))
                       for col, n in (("systemicmean", "map_inv"), ("heartrate", "hr"), ("sao2", "spo2"))])
        q = f"""SELECT k.stay_id, v.observationoffset, v.noninvasivemean
                FROM read_csv('{src / 'vitalAperiodic.csv.gz'}', header=true, all_varchar=true) v
                JOIN k ON v.patientunitstayid = k.unit
                WHERE TRY_CAST(v.observationoffset AS DOUBLE) BETWEEN -120 AND {HOURS * 60}"""
        b = pl.from_arrow(con.execute(q).arrow())
        v = pl.concat([v, b.select("stay_id", (pl.col("observationoffset").cast(pl.Float64, strict=False) / 60)
                                   .alias("h"), pl.lit("map_nibp").alias("var"),
                                   pl.col("noninvasivemean").cast(pl.Float64, strict=False).alias("value"))])
        q = f"""SELECT k.stay_id, i.infusionoffset, i.drugname, i.drugrate
                FROM read_csv('{src / 'infusionDrug.csv.gz'}', header=true, all_varchar=true) i
                JOIN k ON i.patientunitstayid = k.unit"""
        p = pl.from_arrow(con.execute(q).arrow())
        p = (p.filter(pl.col("drugname").str.contains(VASO_REGEX)
                      & (pl.col("drugrate").cast(pl.Float64, strict=False).fill_null(1.0) > 0))
             .select("stay_id", (pl.col("infusionoffset").cast(pl.Float64, strict=False) / 60).alias("h"))
             .with_columns(pl.col("h").alias("h_end")))
    else:
        keys = st.select("stay_id", _raw("subject_id", "INSPIRE").alias("subj"), "icu_in_min")
        con.register("k", keys.to_arrow())
        q = f"""SELECT k.stay_id, w.chart_time, lower(trim(w.item_name)) AS item, w.value, k.icu_in_min
                FROM read_csv('{src / 'ward_vitals.csv.gz'}', header=true, all_varchar=true, ignore_errors=true) w
                JOIN k ON w.subject_id = k.subj
                WHERE lower(trim(w.item_name)) IN ('nibp_mbp', 'nibp_sbp', 'nibp_dbp', 'hr', 'spo2')
                  AND TRY_CAST(w.chart_time AS DOUBLE) BETWEEN k.icu_in_min - 120 AND k.icu_in_min + {HOURS * 60}"""
        w = pl.from_arrow(con.execute(q).arrow()).with_columns(
            ((pl.col("chart_time").cast(pl.Float64, strict=False) - pl.col("icu_in_min")) / 60).alias("h"),
            pl.col("value").cast(pl.Float64, strict=False))
        # MAP = (SBP + 2·DBP) / 3 where only SBP and DBP were charted (the rule of T2's extraction)
        bp = (w.filter(pl.col("item").is_in(["nibp_sbp", "nibp_dbp"])).drop_nulls("value")
              .group_by("stay_id", "h").agg(pl.col("value").filter(pl.col("item") == "nibp_sbp").first().alias("s"),
                                            pl.col("value").filter(pl.col("item") == "nibp_dbp").first().alias("d"))
              .filter(pl.col("s") > pl.col("d")))
        charted = w.filter(pl.col("item") == "nibp_mbp").select("stay_id", "h").unique()
        derived = bp.join(charted, on=["stay_id", "h"], how="anti").select(
            "stay_id", "h", pl.lit("map_nibp").alias("var"), ((pl.col("s") + 2 * pl.col("d")) / 3).alias("value"))
        v = pl.concat([w.filter(pl.col("item").is_in(["nibp_mbp", "hr", "spo2"])).select(
            "stay_id", "h", pl.col("item").replace({"nibp_mbp": "map_nibp"}).alias("var"), "value"), derived])
        q = f"""SELECT k.stay_id, m.chart_time, k.icu_in_min
                FROM read_csv('{src / 'medications.csv.gz'}', header=true, all_varchar=true, ignore_errors=true) m
                JOIN k ON m.subject_id = k.subj
                WHERE m.atc_code IN ({", ".join(f"'{a}'" for a in INSPIRE_VASO_ATC)})
                  AND NOT regexp_matches(coalesce(m.route, ''), '(?i)^ex')"""
        p = pl.from_arrow(con.execute(q).arrow()).select(
            "stay_id", ((pl.col("chart_time").cast(pl.Float64, strict=False) - pl.col("icu_in_min")) / 60).alias("h")
        ).with_columns(pl.col("h").alias("h_end"))
    con.unregister("k")
    v = v.select("stay_id", "h", "var", "value").drop_nulls().filter(
        (pl.col("h") >= -2) & (pl.col("h") < HOURS)
        & (~pl.col("var").str.starts_with("map") | pl.col("value").is_between(*MAP_BOUNDS)))
    p = p.drop_nulls("h").filter((pl.col("h_end") >= -12) & (pl.col("h") < HOURS + 12))
    v.write_parquet(fv)
    p.write_parquet(fp)
    return v, p


# --------------------------------------------------------------------------- hourly grid and landmarks
def hourly(v: pl.DataFrame) -> pl.DataFrame:
    m = v.filter(pl.col("var").str.starts_with("map")).with_columns(pl.col("h").floor().cast(pl.Int32).alias("hr"))
    g = m.sort("h").group_by("stay_id", "hr").agg(
        pl.col("value").median().alias("map"), pl.len().alias("n_map"), (pl.col("value") < THRESH).any().alias("any_lt"),
        pl.col("value").first().alias("first"), (pl.col("var") == "map_inv").any().alias("inv"))
    o = (v.filter(pl.col("var").is_in(["hr", "spo2"])).with_columns(pl.col("h").floor().cast(pl.Int32).alias("hr"))
         .group_by("stay_id", "hr").agg((pl.col("var") == "hr").any().alias("has_hr"),
                                        (pl.col("var") == "spo2").any().alias("has_spo2")))
    return g.join(o, on=["stay_id", "hr"], how="full", coalesce=True)


def landmarks(st: pl.DataFrame, H: pl.DataFrame, P: pl.DataFrame) -> pl.DataFrame:
    """One row per (stay, landmark L): eligibility, state at L and the outcomes over [L, L + HORIZON)."""
    rows = []
    hmap = H.select("stay_id", "hr", "map", "any_lt", "first", "has_hr", "has_spo2")
    for L in LANDMARKS:
        el = st.filter((pl.col("los_h") > L) & (pl.col("death_h").is_null() | (pl.col("death_h") > L)))
        last = hmap.filter(pl.col("hr") == L - 1).select("stay_id", pl.col("map").alias("map_last"), "has_hr",
                                                         "has_spo2")
        prior = (hmap.filter((pl.col("hr") >= L - 6) & (pl.col("hr") < L)).group_by("stay_id")
                 .agg(pl.col("map").drop_nulls().len().alias("n_prior")))
        win = (hmap.join(el.select("stay_id", "los_h"), on="stay_id")
               .filter((pl.col("hr") >= L) & (pl.col("hr") < L + HORIZON) & (pl.col("hr") < pl.col("los_h")))
               .group_by("stay_id").agg((pl.col("map") < THRESH).any().alias("y_med"), pl.col("any_lt").any().alias("y_any"),
                                        (pl.col("first") < THRESH).any().alias("y_first"),
                                        pl.col("map").min().alias("map_min"), pl.col("map").drop_nulls().len().alias("n_win")))
        on = (P.filter((pl.col("h") <= L) & (pl.col("h_end") >= L - 2)).select("stay_id").unique()
              .with_columns(pl.lit(True).alias("vaso_on")))
        start = (P.filter((pl.col("h") > L) & (pl.col("h") <= L + HORIZON)).select("stay_id").unique()
                 .with_columns(pl.lit(True).alias("vaso_start")))
        d = (el.select("stay_id", "los_h", "death_h", "surgical_group").join(last, on="stay_id")
             .filter(pl.col("map_last").is_not_null())
             .join(prior, on="stay_id", how="left").join(win, on="stay_id", how="left")
             .join(on, on="stay_id", how="left").join(start, on="stay_id", how="left")
             .with_columns(pl.lit(L).alias("L"), pl.col("vaso_on").fill_null(False), pl.col("vaso_start").fill_null(False),
                           (pl.col("death_h") <= L + HORIZON).fill_null(False).alias("died_win")))
        rows.append(d)
    out = pl.concat(rows, how="diagonal_relaxed")
    return out.with_columns(*(pl.col(c).fill_null(False) for c in ("y_med", "y_any", "y_first")),
                            (pl.col("y_med") | pl.col("died_win")).alias("y"))


# --------------------------------------------------------------------------- aggregates
def summarise(c: str, st: pl.DataFrame, v: pl.DataFrame, H: pl.DataFrame, Lm: pl.DataFrame, hosp=None) -> dict:
    first48 = H.filter((pl.col("hr") >= 0) & (pl.col("hr") < 48))
    stay_hours = st.select(pl.col("los_h").clip(0, 48).sum()).item()
    with_map = first48.filter(pl.col("map").is_not_null())
    out = {"stays": st.height, "patients": int(st["subject_id"].n_unique()),
           "map_readings_per_hour_median": float(with_map["n_map"].median()),
           "hours_with_map_share": float(with_map.height / max(stay_hours, 1)),
           "map_hours_invasive_share": float(with_map["inv"].mean()),
           "stays_with_invasive_map": float(with_map.filter(pl.col("inv"))["stay_id"].n_unique() / st.height)}
    new = Lm.filter(pl.col("map_last") >= THRESH)
    out["landmarks"] = {"all": Lm.height, "normotensive": new.height,
                        "patients_all": int(Lm["stay_id"].n_unique()),
                        "patients_normotensive": int(new["stay_id"].n_unique())}
    out["outcome"] = {
        "hypotension next 6 h, all landmarks": float(Lm["y"].mean()),
        "… normotensive in the last hour": float(new["y"].mean()),
        "… normotensive, off vasopressors": float(new.filter(~pl.col("vaso_on"))["y"].mean()),
        "… normotensive, on vasopressors": float(new.filter(pl.col("vaso_on"))["y"].mean()),
        "landmarks on vasopressors": float(Lm["vaso_on"].mean()),
        "normotensive: any reading < 65 (not the hourly median)": float((new["y_any"] | new["died_win"]).mean()),
        "normotensive: one reading per hour (the first)": float((new["y_first"] | new["died_win"]).mean()),
        "normotensive, off vasopressors: hypotension or vasopressor started": float(
            new.filter(~pl.col("vaso_on")).select(pl.col("y") | pl.col("vaso_start")).to_series().mean()),
        "death within the 6 h (counted as positive)": float(Lm["died_win"].mean())}
    mm = new["map_min"].drop_nulls().to_numpy()
    out["lowest_map_next_6h_tertile_cuts"] = [round(float(x), 1) for x in np.quantile(mm, [1 / 3, 2 / 3])] if mm.size else []
    out["window_hours_with_map_median"] = float(new["n_win"].fill_null(0).median())
    out["inputs_at_landmark"] = {"heart rate in the last hour": float(new["has_hr"].fill_null(False).mean()),
                                 "SpO2 in the last hour": float(new["has_spo2"].fill_null(False).mean()),
                                 "≥ 3 hourly MAPs in the last 6 h": float((new["n_prior"].fill_null(0) >= 3).mean())}
    by_l = new.group_by("L").agg(pl.col("y").mean().alias("rate"), pl.len().alias("n")).sort("L")
    out["normotensive_rate_by_landmark"] = {int(r["L"]): round(float(r["rate"]), 3) for r in by_l.iter_rows(named=True)
                                            if r["n"] >= MIN_N}
    g = new.group_by("surgical_group").agg(pl.col("y").mean().alias("rate"), pl.len().alias("n")).sort("n", descending=True)
    out["normotensive_rate_by_surgical_group"] = {str(r["surgical_group"]): round(float(r["rate"]), 3)
                                                  for r in g.iter_rows(named=True) if r["n"] >= 100}
    if hosp is not None:
        h = hosp(new["stay_id"].to_numpy())
        hn = new.with_columns(pl.Series("hosp", h, dtype=pl.Utf8)).drop_nulls("hosp")
        per = hn.group_by("hosp").agg(pl.col("stay_id").n_unique().alias("patients"), pl.col("y").mean().alias("rate"))
        big = per.filter(pl.col("patients") >= 300)
        out["hospitals"] = {"with_landmarks": per.height, "≥ 300 patients": big.height,
                            "≥ 1000 patients": per.filter(pl.col("patients") >= 1000).height,
                            "rate_quantiles_hospitals_≥300": [round(float(x), 3) for x in
                                                               np.quantile(big["rate"].to_numpy(), [0.1, 0.5, 0.9])]
                            if big.height else []}
    return out


def render(res: dict) -> str:
    L = ["# Feasibility: hypotension in the next 6 h, ICU after surgery", "",
         "Original tables, every postoperative adult ICU stay (the first of each hospitalization, ICU deaths "
         f"included), first {HOURS} h. Landmarks every 2 h from 2 h to 48 h after the ICU admission, in the ICU and "
         f"alive, with a MAP in the last hour. Hypotension: an hour whose median MAP is < {THRESH:.0f} mmHg (death in "
         "the window counts as positive). Shares; counts ≥ 10.", ""]

    def row(lab, f):
        L.append(f"| {lab} | " + " | ".join(f(res[c]) for c in COHORTS) + " |")

    L += ["## Measurement", "", "| | " + " | ".join(COHORTS) + " |", "|---|" + "---|" * len(COHORTS)]
    row("stays (patients)", lambda r: f"{r['stays']:,} ({r['patients']:,})")
    row("MAP readings per hour (median, hours with a MAP)", lambda r: f"{r['map_readings_per_hour_median']:.0f}")
    row("ICU hours with a MAP, first 48 h", lambda r: f"{r['hours_with_map_share']:.0%}")
    row("MAP hours with an invasive (arterial) value", lambda r: f"{r['map_hours_invasive_share']:.0%}")
    row("stays with an arterial line", lambda r: f"{r['stays_with_invasive_map']:.0%}")
    L += ["", "## Landmarks and outcome", "", "| | " + " | ".join(COHORTS) + " |", "|---|" + "---|" * len(COHORTS)]
    row("landmarks (patients), all", lambda r: f"{r['landmarks']['all']:,} ({r['landmarks']['patients_all']:,})")
    row("landmarks (patients), normotensive in the last hour",
        lambda r: f"{r['landmarks']['normotensive']:,} ({r['landmarks']['patients_normotensive']:,})")
    for k_ in res["inspire"]["outcome"]:
        row(k_, lambda r, k_=k_: f"{r['outcome'][k_]:.1%}")
    row("hours with a MAP in the 6-h window (median)", lambda r: f"{r['window_hours_with_map_median']:.0f}")
    row("lowest hourly MAP of the next 6 h, tertile cuts (normotensive)",
        lambda r: ", ".join(str(x) for x in r["lowest_map_next_6h_tertile_cuts"]))
    for k_ in res["inspire"]["inputs_at_landmark"]:
        row(k_, lambda r, k_=k_: f"{r['inputs_at_landmark'][k_]:.0%}")
    L += ["", "## Rate by landmark (normotensive)", "", "| hours after ICU admission | " + " | ".join(COHORTS) + " |",
          "|---|" + "---|" * len(COHORTS)]
    for lm in LANDMARKS:
        L.append(f"| {lm} | " + " | ".join(
            f"{res[c]['normotensive_rate_by_landmark'].get(lm, float('nan')):.1%}" for c in COHORTS) + " |")
    L += ["", "## Rate by surgical group (normotensive, groups with ≥ 100 landmarks)", ""]
    groups = sorted({g for c in COHORTS for g in res[c]["normotensive_rate_by_surgical_group"]})
    L += ["| group | " + " | ".join(COHORTS) + " |", "|---|" + "---|" * len(COHORTS)]
    for g in groups:
        L.append(f"| {g} | " + " | ".join(
            (f"{res[c]['normotensive_rate_by_surgical_group'][g]:.1%}"
             if g in res[c]["normotensive_rate_by_surgical_group"] else "—") for c in COHORTS) + " |")
    if "hospitals" in res["eicu"]:
        H = res["eicu"]["hospitals"]
        L += ["", f"eICU: {H['with_landmarks']} hospitals with landmarks, {H['≥ 300 patients']} with ≥ 300 patients, "
                  f"{H['≥ 1000 patients']} with ≥ 1,000; hypotension rate (normotensive) across the hospitals with "
                  f"≥ 300 patients, p10 / p50 / p90: {', '.join(f'{x:.1%}' for x in H['rate_quantiles_hospitals_≥300'])}."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from perisim.config import load_config
    from perisim.pipeline import Pipeline
    from perisim.transport.analyses import hospital_map

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default="local")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    pipe = Pipeline(load_config(None, a.profile, {}))
    out = ROOT / "work" / "feasibility" / "hypotension"
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {a.threads}")
    srcs = {c: Path((pipe.disc.get(c) or {}).get("path")) for c in COHORTS}
    res = {}
    for c in COHORTS:
        st = stays(c)
        print(f"[hypotension] {c}: {st.height} stays, reading the original tables", flush=True)
        v, p = extract(c, st, srcs[c], out, con)
        H = (hourly(v).join(st.select("stay_id", "los_h"), on="stay_id")      # ICU hours only (INSPIRE: ward too)
             .filter(pl.col("hr") < pl.col("los_h")).drop("los_h"))
        Lm = landmarks(st, H, p)
        hof = (lambda ids: hospital_map(str(srcs["eicu"]), ids, pipe.ws.cache("eicu"))) if c == "eicu" else None
        res[c] = summarise(c, st, v, H, Lm, hof)
        print(f"[hypotension] {c}: done", flush=True)
    (out / "feasibility_hypotension.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (out / "feasibility_hypotension.md").write_text(render(res), encoding="utf-8")
    print(f"[hypotension] written {out / 'feasibility_hypotension.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
