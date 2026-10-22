"""Selección de la estancia índice y diagrama de flujo (§3.1–§3.3).

Orden: adultos → UCI postoperatoria → primera estancia postoperatoria de la hospitalización → alta viva
de UCI → estancia ≥ 4 h → alta de UCI a planta (no directa fuera del hospital) → donantes → limitación.
La unidad de análisis es la PRIMERA estancia postoperatoria: si no cumple un criterio posterior,
la hospitalización queda fuera (no se toma la siguiente).
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
    flow.append(_row("Estancias en UCI con entrada y salida válidas", df, None))

    steps: list[tuple[str, pl.Expr | str]] = [
        (f"Edad ≥ {c['min_age']} años", pl.col("age") >= c["min_age"]),
        ("Ingreso en UCI postoperatorio", pl.col("postop").fill_null(False)),
        ("Primera estancia postoperatoria de la hospitalización", "first"),
        ("Alta viva de la UCI (muertes en UCI fuera)", ~pl.col("died_in_icu").fill_null(False)),
        (f"Estancia en UCI ≥ {c['min_icu_hours']} h", (pl.col("icu_out_min") - pl.col("icu_in_min")) >= c["min_icu_hours"] * 60),
        ("Alta de UCI dentro del hospital (con seguimiento en planta)", pl.col("icu_to_ward").fill_null(False)),
    ]
    if c.get("exclude_organ_donors", True):
        steps.append(("Excluidos donantes de órganos", ~pl.col("organ_donor").fill_null(False)))
    if c.get("exclude_known_limitation", True):
        steps.append(("Excluida limitación del esfuerzo conocida antes del corte", ~pl.col("limitation_known").fill_null(False)))

    for label, cond in steps:
        prev = df.height
        if isinstance(cond, str) and cond == "first":
            df = (df.sort(["hadm_id", "icu_in_min"]).group_by("hadm_id", maintain_order=True).first())
        else:
            df = df.filter(cond)
        flow.append(_row(label, df, prev))
    return df.sort("stay_id"), flow


def add_cutoff(stays: pl.DataFrame, margin_hours: float) -> pl.DataFrame:
    """t0 = salida física de UCI; corte de inputs = t0 − margen (§3.3, D2)."""
    return stays.with_columns(pl.col("icu_out_min").alias("t0_min"),
                              (pl.col("icu_out_min") - margin_hours * 60.0).alias("cutoff_min"))
