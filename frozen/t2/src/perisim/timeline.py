"""Shared temporal operations on inputs_long (support intervals)."""

from __future__ import annotations

import polars as pl


def merge_runs(rows: pl.DataFrame, gap_minutes: float, by: tuple[str, ...] = ("stay_id", "variable")) -> pl.DataFrame:
    """Point records or intervals → continuous intervals [start_min, end_min].

    A record without an end counts as a point. Two records belong to the same interval if the second starts
    before (accumulated end of the previous one + gap). This infers the end from the absence of records (§2.2).
    """
    if rows.height == 0:
        return pl.DataFrame(schema={**{b: pl.Utf8 for b in by}, "start_min": pl.Float64, "end_min": pl.Float64})
    b = list(by)
    df = (rows.with_columns(pl.coalesce(pl.col("end_min"), pl.col("time_min")).alias("_end"))
          .with_columns(pl.max_horizontal("_end", "time_min").alias("_end"))
          .sort(b + ["time_min"]))
    df = df.with_columns(pl.col("_end").cum_max().shift(1).over(b).alias("_prev"))
    df = df.with_columns((pl.col("_prev").is_null() | (pl.col("time_min") > pl.col("_prev") + gap_minutes))
                         .cast(pl.Int32).cum_sum().over(b).alias("_run"))
    return (df.group_by(b + ["_run"]).agg(pl.col("time_min").min().alias("start_min"), pl.col("_end").max().alias("end_min"))
            .drop("_run").sort(b + ["start_min"]))
