"""Build the INSPIRE source cohort for the iPROVE transport experiment (plan: docs/DECISIONS.md 2026-11-07).

Target: any vasoactive drug during anaesthesia (iPROVE `usoFarmacosVasoactivos`). Predictors: the preoperative
variables of iPROVE that INSPIRE can derive, coded into iPROVE's own categories (iPROVE stores each variable as the
midpoint of its category; a raw INSPIRE value goes to the nearest iPROVE code, cut at the midpoints between codes).
Variables INSPIRE cannot derive (epidural, recruitment manoeuvres, Airtest, NMB reversal) are not used, nor are
ARISCAT (it contains the duration of surgery) and the intraoperative variables (concurrent with the target).

Cohort: adults, general anaesthesia, General Surgery, abdominal procedure (ICD-10-PCS 0D, 0F, or 0W on the abdominal
wall, peritoneal cavity, retroperitoneum or pelvic cavity), the patient's first such operation.

Usage:  python scripts/iprove_build_inspire.py   (PERISIM_DATASETS, PERISIM_WORKDIR as for the other experiments)
Outputs: <workdir>/experiments/iprove/inspire.parquet (patient level, stays local), iprove.parquet (iPROVE renamed,
same columns), cohort.md (aggregates only).
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

DATA = Path(os.environ.get("PERISIM_DATASETS", "data"))
WORK = Path(os.environ.get("PERISIM_WORKDIR", "work")) / "experiments" / "iprove"
TARGET = "usoFarmacosVasoactivos"
PREDICTORS = ["edad_d", "genero", "altura_d", "peso_d", "IMC_d", "cirugiaAbdominalProgramada", "esCirugiaOncologica",
              "ASA", "spO2pre_b", "hbPreoperatoria_d", "infeccionRespiratoriaUltimoMes", "hipertensionArterial", "DM",
              "cardiopatiaIsquemica", "fumador", "consumoAlcohol", "dislipemia", "ePOC", "insuficienciaRenal"]
BINNED = ["edad_d", "altura_d", "peso_d", "IMC_d", "hbPreoperatoria_d"]
VASO_ITEMS = ["phe", "eph", "epi", "epii", "nepi", "pepi", "vaso", "dopai"]
DX = {  # ICD-10-CM prefixes (dots removed), any record before the operation unless stated
    "hipertensionArterial": ["I10", "I11", "I12", "I13", "I15"],
    "DM": ["E10", "E11", "E12", "E13", "E14"],
    "cardiopatiaIsquemica": ["I20", "I21", "I22", "I23", "I24", "I25"],
    "fumador": ["F17", "Z720", "Z87891"],
    "consumoAlcohol": ["F10"],
    "dislipemia": ["E78"],
    "ePOC": ["J43", "J44"],
    "insuficienciaRenal": ["N17", "N18", "N19"],
}
RESP_INF = ["J0", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J20", "J21", "J22", "J40"]
DAY = 1440.0                                           # INSPIRE times are minutes


def sql(q: str, **frames) -> pl.DataFrame:
    con = duckdb.connect()
    for k, v in frames.items():
        con.register(k, v.to_arrow())
    return con.sql(q).pl()


def csv(path: Path) -> str:
    return f"read_csv('{path.as_posix()}', header=true)"


def cohort() -> pl.DataFrame:
    op = sql(f"SELECT * FROM {csv(DATA / 'inspire' / 'operations.csv.gz')}")
    pcs = pl.col("icd10_pcs").cast(pl.Utf8)
    ab = (pcs.str.slice(0, 2).is_in(["0D", "0F"])
          | ((pcs.str.slice(0, 2) == "0W") & pcs.str.slice(3, 1).is_in(["F", "G", "H", "J"])))
    return (op.filter((pl.col("department") == "GS") & (pl.col("antype") == "General") & (pl.col("age") >= 18) & ab)
            .sort("opstart_time").group_by("subject_id", maintain_order=True).first())


def codes_before(c: pl.DataFrame) -> pl.DataFrame:
    k = c.select("subject_id", "opstart_time", "admission_time", "discharge_time")
    dx = sql(f"SELECT d.subject_id, d.chart_time, d.icd10_cm, k.opstart_time, k.admission_time, k.discharge_time "
             f"FROM {csv(DATA / 'inspire' / 'diagnosis.csv.gz')} d JOIN k USING (subject_id)", k=k) \
        .with_columns(pl.col("icd10_cm").cast(pl.Utf8).str.replace_all(r"\.", "").alias("code"))
    before = dx.filter(pl.col("chart_time") < pl.col("opstart_time"))
    out = c.select("subject_id")
    for name, pre in DX.items():
        hit = before.filter(pl.any_horizontal([pl.col("code").str.starts_with(p) for p in pre])) \
            .select("subject_id").unique().with_columns(pl.lit(1).alias(name))
        out = out.join(hit, on="subject_id", how="left")
    month = before.filter(pl.col("chart_time") >= pl.col("opstart_time") - 30 * DAY)
    hit = month.filter(pl.any_horizontal([pl.col("code").str.starts_with(p) for p in RESP_INF])) \
        .select("subject_id").unique().with_columns(pl.lit(1).alias("infeccionRespiratoriaUltimoMes"))
    out = out.join(hit, on="subject_id", how="left")
    onco = dx.filter((pl.col("chart_time") >= pl.col("admission_time") - 365 * DAY)
                     & (pl.col("chart_time") <= pl.col("discharge_time"))
                     & pl.col("code").str.contains(r"^(C|D0)")).select("subject_id").unique() \
        .with_columns(pl.lit(1).alias("esCirugiaOncologica"))
    out = out.join(onco, on="subject_id", how="left")
    return out.fill_null(0)


def last_before(path: Path, item: str, c: pl.DataFrame, window_days: float, alias: str) -> pl.DataFrame:
    k = c.select("subject_id", "orin_time")
    return sql(f"SELECT w.subject_id, arg_max(TRY_CAST(w.value AS DOUBLE), w.chart_time) AS {alias} "
               f"FROM {csv(path)} w JOIN k USING (subject_id) WHERE w.item_name = '{item}' "
               f"AND w.chart_time < k.orin_time AND w.chart_time >= k.orin_time - {window_days * DAY} "
               f"GROUP BY w.subject_id", k=k)


def target(c: pl.DataFrame) -> pl.DataFrame:
    k = c.select("op_id", "anstart_time", "anend_time")
    items = ", ".join(f"'{i}'" for i in VASO_ITEMS)
    v = sql(f"SELECT DISTINCT v.op_id FROM {csv(DATA / 'inspire' / 'vitals.csv.gz')} v JOIN k USING (op_id) "
            f"WHERE v.item_name IN ({items}) AND TRY_CAST(v.value AS DOUBLE) > 0 "
            f"AND v.chart_time >= k.anstart_time AND v.chart_time <= k.anend_time", k=k) \
        .with_columns(pl.lit(1).alias(TARGET))
    return c.select("op_id").join(v, on="op_id", how="left").fill_null(0)


def edges(codes: np.ndarray) -> np.ndarray:
    c = np.sort(np.unique(codes))
    return (c[1:] + c[:-1]) / 2


def to_code(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    c = np.sort(np.unique(codes))
    return np.where(np.isfinite(x), c[np.searchsorted(edges(c), x, side="right")], np.nan)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    ip = pl.read_csv(WORK / "iprove_raw.csv")             # the Excel sheet, saved as CSV beside the outputs
    c = cohort()
    raw = c.select("subject_id", "op_id", pl.col("age").alias("edad"), (pl.col("sex") == "M").cast(pl.Int64)
                   .alias("genero"), pl.col("height").alias("altura"), pl.col("weight").alias("peso"),
                   (1 - pl.col("emop")).alias("cirugiaAbdominalProgramada"),
                   pl.col("asa").clip(1, 4).alias("ASA"))
    raw = raw.with_columns((pl.col("peso") / (pl.col("altura") / 100) ** 2).alias("IMC"))
    raw = raw.join(codes_before(c), on="subject_id", how="left")
    raw = raw.join(last_before(DATA / "inspire" / "ward_vitals.csv.gz", "spo2", c, 7, "spo2_pre"), on="subject_id",
                   how="left")
    raw = raw.join(last_before(DATA / "inspire" / "labs.csv.gz", "hb", c, 30, "hb_pre"), on="subject_id", how="left")
    raw = raw.join(c.select("subject_id", "op_id").join(target(c), on="op_id").drop("op_id"), on="subject_id")
    df = raw.with_columns(pl.when(pl.col("spo2_pre").is_null()).then(None)
                          .otherwise((pl.col("spo2_pre") >= 96).cast(pl.Int64)).alias("spO2pre_b"))
    miss = {}
    cols = {}
    for name, src in (("edad_d", "edad"), ("altura_d", "altura"), ("peso_d", "peso"), ("IMC_d", "IMC"),
                      ("hbPreoperatoria_d", "hb_pre")):
        x = df[src].cast(pl.Float64).to_numpy()
        cols[name] = to_code(x, ip[name].to_numpy().astype(float))
    for name in PREDICTORS:
        v = cols[name] if name in cols else df[name].cast(pl.Float64).to_numpy()
        miss[name] = float(np.isnan(v).mean())
        if np.isnan(v).any():                           # missing: the INSPIRE median (iPROVE has none missing)
            med = np.nanmedian(v)
            v = np.where(np.isnan(v), to_code(np.array([med]), ip[name].to_numpy().astype(float))[0]
                         if name in BINNED else np.round(med), v)
        cols[name] = v
    out = pl.DataFrame({**{k: cols[k] for k in PREDICTORS}, TARGET: df[TARGET].to_numpy()})
    out.write_parquet(WORK / "inspire.parquet")
    ip.select(PREDICTORS + [TARGET]).cast(pl.Float64).write_parquet(WORK / "iprove.parquet")
    L = ["# iPROVE transport: the INSPIRE source cohort", "", f"- INSPIRE patients: {out.height:,}; iPROVE patients: "
         f"{ip.height:,}", f"- target `{TARGET}`: INSPIRE {out[TARGET].mean():.1%}, iPROVE {ip[TARGET].mean():.1%}",
         "", "| predictor | INSPIRE missing (imputed) | INSPIRE mean | iPROVE mean |", "|---|---|---|---|"]
    for k in PREDICTORS:
        L.append(f"| {k} | {miss[k]:.1%} | {out[k].mean():.3f} | {ip[k].cast(pl.Float64).mean():.3f} |")
    (WORK / "cohort.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"iprove: {WORK / 'cohort.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
