"""Index stay selection and flow diagram (§3.1–§3.3).

Order: adults → postoperative ICU → first postoperative stay of the hospitalization → discharged alive
from the ICU → stay ≥ 4 h → ICU discharge to the ward (not directly out of the hospital) → donors → limitation.
The unit of analysis is the FIRST postoperative stay: if it fails a later criterion,
the hospitalization is excluded (the next stay is not taken).
"""

from __future__ import annotations

import polars as pl


def _row(step: str, df: pl.DataFrame, prev: int | None) -> dict:
    n = df.height
    return {"step": step, "n_stays": n, "n_subjects": df["subject_id"].n_unique() if n else 0,
            "excluded": (prev - n) if prev is not None else None}


def select_index_stays(stays: pl.DataFrame, cohorts_cfg: dict, cohort: str) -> tuple[pl.DataFrame, list[dict]]:
    c = {**cohorts_cfg.get("common", {}), **cohorts_cfg.get(cohort, {})}
    flow: list[dict] = []
    df = stays.filter(pl.col("icu_in_min").is_not_null() & pl.col("icu_out_min").is_not_null()
                      & (pl.col("icu_out_min") > pl.col("icu_in_min")))
    flow.append(_row("ICU stays with valid admission and discharge times", df, None))

    steps: list[tuple[str, pl.Expr | str]] = [
        (f"Age ≥ {c['min_age']} years", pl.col("age") >= c["min_age"]),
        ("Postoperative ICU admission", pl.col("postop").fill_null(False)),
        ("First postoperative stay of the hospitalization", "first"),
        ("Discharged alive from the ICU (ICU deaths excluded)", ~pl.col("died_in_icu").fill_null(False)),
        (f"ICU stay ≥ {c['min_icu_hours']} h", (pl.col("icu_out_min") - pl.col("icu_in_min")) >= c["min_icu_hours"] * 60),
        ("ICU discharge within the hospital (ward follow-up)", pl.col("icu_to_ward").fill_null(False)),
    ]
    if c.get("exclude_organ_donors", True):
        steps.append(("Organ donors excluded", ~pl.col("organ_donor").fill_null(False)))
    if c.get("exclude_known_limitation", True):
        steps.append(("Known limitation of life-sustaining treatment before the cutoff excluded", ~pl.col("limitation_known").fill_null(False)))

    for label, cond in steps:
        prev = df.height
        if isinstance(cond, str) and cond == "first":
            df = (df.sort(["hadm_id", "icu_in_min"]).group_by("hadm_id", maintain_order=True).first())
        else:
            df = df.filter(cond)
        flow.append(_row(label, df, prev))
    return df.sort("stay_id"), flow


def add_cutoff(stays: pl.DataFrame, margin_hours: float) -> pl.DataFrame:
    """t0 = physical ICU exit; input cutoff = t0 − margin (§3.3, D2)."""
    return stays.with_columns(pl.col("icu_out_min").alias("t0_min"),
                              (pl.col("icu_out_min") - margin_hours * 60.0).alias("cutoff_min"))
