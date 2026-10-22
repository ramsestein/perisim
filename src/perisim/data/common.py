"""Intermediate common format (§8.3) and conversion utilities shared by the extractors.

Time convention: all timestamps are minutes (float64) on a clock specific to each cohort
(INSPIRE: native relative minutes; MIMIC-IV: minutes since 1970 of the shifted dates;
eICU: minutes since hospital admission of the hospital stay). Nothing downstream
compares clocks across cohorts: only differences within the same stay.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

import polars as pl

STAYS_SCHEMA: dict[str, pl.DataType] = {
    "stay_id": pl.Utf8,
    "cohort": pl.Utf8,
    "subject_id": pl.Utf8,
    "hadm_id": pl.Utf8,
    "icu_in_min": pl.Float64,
    "icu_out_min": pl.Float64,
    "hosp_admit_min": pl.Float64,
    "hosp_disch_min": pl.Float64,
    "death_min": pl.Float64,
    "surgery_start_min": pl.Float64,
    "surgery_end_min": pl.Float64,
    "age": pl.Float64,
    "sex_male": pl.Float64,
    "bmi": pl.Float64,
    "asa": pl.Float64,
    "emergency": pl.Float64,
    "anesthesia_type": pl.Utf8,
    "surgical_group": pl.Utf8,
    "pcs_code": pl.Utf8,
    "admit_source": pl.Utf8,
    "year": pl.Float64,
    "postop": pl.Boolean,
    "organ_donor": pl.Boolean,
    "limitation_known": pl.Boolean,
    "died_in_icu": pl.Boolean,
    "icu_to_ward": pl.Boolean,
    "discharge_location": pl.Utf8,
    "unit_order": pl.Float64,
    "apache_pred_hosp_los": pl.Float64,
}

# inputs_long: stay, variable, value, timestamp (+ optional interval end: supports and orders)
INPUTS_SCHEMA: dict[str, pl.DataType] = {
    "stay_id": pl.Utf8,
    "variable": pl.Utf8,
    "value": pl.Float64,
    "time_min": pl.Float64,
    "end_min": pl.Float64,
}

# outcomes: discrete events after t0 (ICU readmission, surgery, transfusion). precision: minute | day
OUTCOMES_SCHEMA: dict[str, pl.DataType] = {
    "stay_id": pl.Utf8,
    "component": pl.Utf8,
    "time_min": pl.Float64,
    "precision": pl.Utf8,
    "value": pl.Float64,
}

MIN_PER_DAY = 1440.0


def conform(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Adds missing columns as nulls, casts and orders according to the common schema."""
    exprs = []
    for col, dtype in schema.items():
        if col in df.columns:
            exprs.append(pl.col(col).cast(dtype, strict=False).alias(col))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(col))
    return df.select(exprs)


def validate(df: pl.DataFrame, schema: dict[str, pl.DataType], name: str, unique: str | None = None) -> list[str]:
    errors = []
    for col, dtype in schema.items():
        if col not in df.columns:
            errors.append(f"{name}: missing column {col}")
        elif df.schema[col] != dtype:
            errors.append(f"{name}: {col} is {df.schema[col]}, expected {dtype}")
    if unique and unique in df.columns and df[unique].n_unique() != df.height:
        errors.append(f"{name}: {unique} is not unique")
    return errors


# --------------------------------------------------------------------------- type conversion
def as_float(col: str) -> pl.Expr:
    """Any column (text or numeric) -> Float64; non-convertible values -> null."""
    return pl.col(col).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def as_str(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars()


def as_id(col: str) -> pl.Expr:
    """Stable identifier as text ("123.0" -> "123")."""
    return pl.col(col).cast(pl.Utf8).str.strip_chars().str.replace(r"\.0+$", "")


_DT_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M")


def dt_minutes(df: pl.DataFrame, col: str) -> pl.Expr:
    """Date/time (text, Date or Datetime) -> minutes since 1970 (Float64)."""
    dtype = df.schema[col]
    if isinstance(dtype, pl.Datetime):
        return (pl.col(col).dt.epoch("s") / 60.0).cast(pl.Float64)
    if dtype == pl.Date:
        return (pl.col(col).cast(pl.Datetime("us")).dt.epoch("s") / 60.0).cast(pl.Float64)
    s = pl.col(col).cast(pl.Utf8).str.strip_chars()
    parsed = [s.str.to_datetime(f, strict=False, time_unit="us") for f in _DT_FORMATS]
    parsed.append(s.str.to_date("%Y-%m-%d", strict=False).cast(pl.Datetime("us")))
    return (pl.coalesce(parsed).dt.epoch("s") / 60.0).cast(pl.Float64)


def date_minutes(df: pl.DataFrame, col: str) -> pl.Expr:
    """Date (no time) -> minutes at midnight of that day."""
    m = dt_minutes(df, col)
    return (m / MIN_PER_DAY).floor() * MIN_PER_DAY


# --------------------------------------------------------------------------- transforms
TRANSFORMS: dict[str | None, Callable[[pl.Expr], pl.Expr]] = {
    None: lambda e: e,
    "f_to_c": lambda e: (e - 32.0) * 5.0 / 9.0,
    "fraction_to_percent": lambda e: pl.when(e <= 1.0).then(e * 100.0).otherwise(e),
}


def apply_bounds(expr: pl.Expr, bounds: Iterable[float] | None) -> pl.Expr:
    if not bounds:
        return expr
    lo, hi = bounds
    return pl.when((expr >= lo) & (expr <= hi)).then(expr).otherwise(None)


def variables_for(harmonization: dict, cohort: str, table: str) -> list[dict]:
    """Harmonized variables whose source in `cohort` is `table`."""
    out = []
    for name, spec in harmonization.get("variables", {}).items():
        src = (spec.get("sources") or {}).get(cohort)
        if src and src.get("table") == table:
            out.append({
                "variable": spec.get("rename_to", name),
                "items": [str(i) for i in src.get("items", [])],
                "transform": spec.get("transform"),
                "bounds": spec.get("bounds"),
                "aggregate": src.get("aggregate"),
            })
    return out


def map_measurements(df: pl.DataFrame, specs: list[dict], item_col: str = "item",
                     value_col: str = "value") -> pl.DataFrame:
    """Rows (item, value) -> (variable, value) with transform and physiological bounds."""
    frames = []
    item_lower = pl.col(item_col).cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    for spec in specs:
        if spec.get("aggregate"):
            continue  # handled separately (e.g. MIMIC GCS by components)
        items = [i.lower() for i in spec["items"]]
        sub = df.filter(item_lower.is_in(items))
        if sub.height == 0:
            continue
        val = TRANSFORMS[spec.get("transform")](as_float(value_col))
        val = apply_bounds(val, spec.get("bounds"))
        frames.append(sub.with_columns(val.alias("value"), pl.lit(spec["variable"]).alias("variable"))
                      .drop_nulls("value"))
    if not frames:
        return df.head(0).with_columns(pl.lit(None, pl.Utf8).alias("variable"), pl.lit(None, pl.Float64).alias("value"))
    return pl.concat(frames, how="diagonal_relaxed")


def regex_map_unique(values: pl.Series, rules: list[dict], key: str) -> dict[str, list[str]]:
    """Applies regex rules (with Python lookahead support) to unique values.

    Returns {value: [result, ...]} for the values with at least one match.
    """
    compiled = [(re.compile(r["pattern"]), r[key]) for r in rules]
    out: dict[str, list[str]] = {}
    for v in values.drop_nulls().unique().to_list():
        hits = [res for pat, res in compiled if pat.search(str(v))]
        if hits:
            out[str(v)] = hits
    return out


def regex_first_unique(values: pl.Series, rules: list[dict], key: str) -> dict[str, str]:
    m = regex_map_unique(values, rules, key)
    return {k: v[0] for k, v in m.items()}


def harmonized_category(col: pl.Expr, mapping: dict, default: str | None = "other") -> pl.Expr:
    mapping = {str(k).upper(): v for k, v in (mapping or {}).items()}
    return (col.cast(pl.Utf8).str.strip_chars().str.to_uppercase()
            .replace_strict(mapping, default=default, return_dtype=pl.Utf8))


def windowed_join(events: pl.DataFrame, stays: pl.DataFrame, key: str, lo: pl.Expr, hi: pl.Expr) -> pl.DataFrame:
    """Joins events by `key` to index stays and keeps those falling in [lo, hi] (expressions over stays)."""
    s = stays.with_columns(lo.alias("_lo"), hi.alias("_hi")).select([key, "stay_id", "_lo", "_hi"])
    j = events.join(s, on=key, how="inner")
    return j.filter((pl.col("time_min") >= pl.col("_lo")) & (pl.col("time_min") <= pl.col("_hi"))).drop("_lo", "_hi")


def empty(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)
