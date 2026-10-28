"""Landmarks: one row per (stay, L), L hours after the ICU admission, with the inputs known at L and the outcomes of
the window [L, L + horizon). Pure functions of the three long tables of `extract` (hours after the ICU admission).

Time rule (no leakage): an input uses only what happened before L (hours [L − history, L), labs in
[L − lab_lookback, L), supports up to hour L − 1); an outcome uses only the hours [L, L + horizon). Hour `hour` is
the interval [hour, hour + 1).
"""

from __future__ import annotations

import polars as pl

VITALS = ("hr", "spo2", "rr", "temp")
LABS = ("hb", "creatinine", "lactate")
SUPPORTS = ("vaso", "sedation", "vent")


def hourly(vitals: pl.DataFrame) -> pl.DataFrame:
    """Per stay and whole hour: the median MAP (invasive and non-invasive readings together), its number of readings,
    the first reading of the hour, whether any was invasive, and the medians of the other vitals."""
    v = vitals.with_columns(pl.col("h").floor().cast(pl.Int32).alias("hour"))
    m = (v.filter(pl.col("var").str.starts_with("map")).sort("h").group_by("stay_id", "hour")
         .agg(pl.col("value").median().alias("map"), pl.len().alias("map_n"), pl.col("value").first().alias("map_first"),
              (pl.col("var") == "map_inv").any().alias("map_inv")))
    o = (v.filter(pl.col("var").is_in(VITALS)).group_by("stay_id", "hour", "var").agg(pl.col("value").median())
         .pivot(on="var", index=["stay_id", "hour"], values="value"))
    for c in VITALS:
        if c not in o.columns:
            o = o.with_columns(pl.lit(None, pl.Float64).alias(c))
    return m.join(o, on=["stay_id", "hour"], how="full", coalesce=True)


def support_runs(supports: pl.DataFrame, gap_h: float) -> pl.DataFrame:
    """Charted points of a support (h_end == h) merged into runs while the gaps between them are < gap_h (T2's
    rule: the end is inferred from the absence of records); intervals are kept as they are."""
    pts = (supports.filter(pl.col("h_end") == pl.col("h")).sort("stay_id", "kind", "h")
           .with_columns(((pl.col("h") - pl.col("h").shift(1)).over("stay_id", "kind") >= gap_h).fill_null(True)
                         .cast(pl.Int32).cum_sum().over("stay_id", "kind").alias("run")))
    runs = pts.group_by("stay_id", "kind", "run").agg(pl.col("h").min(), pl.col("h").max().alias("h_end")).drop("run")
    return pl.concat([supports.filter(pl.col("h_end") != pl.col("h")), runs.select(supports.columns)])


def support_hours(supports: pl.DataFrame, gap_h: float | None = None) -> pl.DataFrame:
    """Whole hours with each support running: an interval (or a run of charted points, with `gap_h`) covers every
    hour it touches; a single charted point, its hour."""
    if gap_h:
        supports = support_runs(supports, gap_h)
    s = supports.with_columns(pl.col("h").floor().cast(pl.Int32).alias("a"),
                              pl.col("h_end").floor().cast(pl.Int32).alias("b"))
    s = s.filter(pl.col("b") >= pl.col("a")).with_columns(pl.int_ranges("a", pl.col("b") + 1).alias("hour"))
    return s.explode("hour").select("stay_id", "kind", pl.col("hour").cast(pl.Int32)).unique()


def _slope(x: str, y: str) -> pl.Expr:
    n = pl.col(y).is_not_null().sum()
    xs = pl.when(pl.col(y).is_not_null()).then(pl.col(x))
    sx, sy, sxy, sxx = xs.sum(), pl.col(y).sum(), (xs * pl.col(y)).sum(), (xs * xs).sum()
    den = n * sxx - sx * sx
    return pl.when((n >= 2) & (den > 0)).then((n * sxy - sx * sy) / den).otherwise(None)


def build(stays: pl.DataFrame, H: pl.DataFrame, labs: pl.DataFrame, S: pl.DataFrame, cfg: dict,
          cohort: str | None = None) -> pl.DataFrame:
    """Landmark rows for every L of the grid where the stay is in the ICU, alive, without a known limitation of
    treatment, and has a MAP in the last hour. `normotensive` marks the analysis population (last-hour MAP ≥ the
    threshold). Outcomes: `y` hypotension (an hour with median MAP < threshold, or death) in [L, L + horizon);
    `y_first` the same seen through one reading per hour; `band` of the lowest hourly MAP; `known` whether the
    window could be observed (a MAP hour or the death inside it). Features the cohort's sources cannot measure
    (cfg model.source_missing) are set missing."""
    lm = cfg["landmarks"]
    hist, hor, thr = int(lm["history_h"]), int(lm["horizon_h"]), float(lm["map_threshold"])
    b1, b2 = (float(x) for x in lm["map_bands"])
    look, lab_look = int(lm["support_lookback_h"]), float(lm["lab_lookback_h"])
    recent = int(lm.get("support_recent_h", 24))
    los = stays.select("stay_id", "los_h")
    rows = []
    for L in range(int(lm["first_h"]), int(lm["last_h"]) + 1, int(lm["every_h"])):
        el = stays.filter((pl.col("los_h") > L) & (pl.col("death_h").is_null() | (pl.col("death_h") > L))
                          & (pl.col("limitation_h").is_null() | (pl.col("limitation_h") > L)))
        past = H.filter((pl.col("hour") >= L - hist) & (pl.col("hour") < L)).with_columns((pl.col("hour") - L).alias("x"))
        last = H.filter(pl.col("hour") == L - 1).select("stay_id", pl.col("map").alias("map_last"))
        f_hist = past.sort("hour").group_by("stay_id").agg(
            pl.col("map").mean().alias("map_mean_6h"), pl.col("map").min().alias("map_min_6h"),
            pl.col("map").std().alias("map_sd_6h"), pl.col("map").is_not_null().sum().alias("map_hours_6h"),
            _slope("x", "map").alias("map_slope_6h"), _slope("x", "hr").alias("hr_slope_6h"),
            pl.col("spo2").min().alias("spo2_min_6h"), pl.col("map_inv").fill_null(False).any().alias("art_line"),
            *(pl.col(c).drop_nulls().last().alias(f"{c}_last") for c in VITALS))
        f_lab = (labs.filter((pl.col("h") < L) & (pl.col("h") >= L - lab_look)).sort("h").group_by("stay_id", "var")
                 .agg(pl.col("value").last(), (L - pl.col("h").last()).alias("age_h"))
                 .pivot(on="var", index="stay_id", values=["value", "age_h"]))
        f_lab = f_lab.rename({c: c.replace("value_", "").replace("age_h_", "") + ("_age_h" if c.startswith("age_h_")
                                                                                   else "_last")
                              for c in f_lab.columns if c != "stay_id"})
        sup_past = S.filter((pl.col("hour") < L) & (pl.col("hour") >= 0))      # ICU hours only
        f_sup = sup_past.group_by("stay_id").agg(
            *((pl.col("hour").filter(pl.col("kind") == k) >= L - look).any().alias(f"{k}_on") for k in SUPPORTS),
            *((pl.col("hour").filter(pl.col("kind") == k) >= L - recent).any().alias(f"{k}_24h")
              for k in ("vaso", "sedation")),
            (pl.col("kind") == "vaso").sum().alias("vaso_hours"))
        win = (H.join(los, on="stay_id").filter((pl.col("hour") >= L) & (pl.col("hour") < L + hor)
                                                & (pl.col("hour") < pl.col("los_h")))
               .group_by("stay_id").agg((pl.col("map") < thr).any().alias("y_med"),
                                        (pl.col("map_first") < thr).any().alias("y_first"),
                                        pl.col("map").min().alias("map_min_next"),
                                        pl.col("map").is_not_null().sum().alias("win_hours"),
                                        pl.col("map").filter(pl.col("hour") == L + hor - 1).first().alias("map_at_h")))
        start = (S.filter((pl.col("kind") == "vaso") & (pl.col("hour") >= L) & (pl.col("hour") < L + hor))
                 .select("stay_id").unique().with_columns(pl.lit(True).alias("vaso_in_window")))
        d = (el.join(last, on="stay_id").filter(pl.col("map_last").is_not_null())
             .join(f_hist, on="stay_id", how="left").join(f_lab, on="stay_id", how="left")
             .join(f_sup, on="stay_id", how="left").join(win, on="stay_id", how="left")
             .join(start, on="stay_id", how="left")
             .with_columns(pl.lit(L, pl.Int32).alias("L")))
        rows.append(d)
    out = pl.concat(rows, how="diagonal_relaxed")
    for k in [f"{s}_on" for s in SUPPORTS] + ["vaso_24h", "sedation_24h"]:
        out = out.with_columns(pl.col(k).fill_null(False))
    if "surgical_group" in out.columns:
        out = out.with_columns(pl.col("surgical_group").fill_null("unknown"))
    for c in [f"{x}_last" for x in LABS] + [f"{x}_age_h" for x in LABS]:
        if c not in out.columns:
            out = out.with_columns(pl.lit(None, pl.Float64).alias(c))
    died = (pl.col("death_h") > pl.col("L")) & (pl.col("death_h") <= pl.col("L") + hor)
    y_med, y_first = pl.col("y_med").fill_null(False), pl.col("y_first").fill_null(False)
    out = out.with_columns(
        (pl.col("map_last") >= thr).alias("normotensive"),
        died.fill_null(False).alias("died_in_window"),
        (pl.col("win_hours").fill_null(0) > 0).alias("_seen"),
        pl.col("vaso_hours").fill_null(0),
        (pl.col("L") + pl.col("surgery_end_to_icu_h")).alias("hours_since_surgery"),   # null without surgery times
    ).with_columns(
        (pl.col("_seen") | pl.col("died_in_window")).alias("known"),
        (y_med | pl.col("died_in_window")).alias("y"),
        (y_first | pl.col("died_in_window")).alias("y_first"),
        pl.when(pl.col("died_in_window") | (pl.col("map_min_next") < b1)).then(0)
        .when(pl.col("map_min_next") < b2).then(1).when(pl.col("map_min_next").is_not_null()).then(2)
        .otherwise(None).cast(pl.Int8).alias("band"),
        (~pl.col("vaso_on") & pl.col("vaso_in_window").fill_null(False)).alias("vaso_start"),
        (pl.col("los_h") <= pl.col("L") + 24).alias("icu_discharge_24h"),
    ).drop("_seen", "y_med", "vaso_in_window")
    unknown = (cfg.get("model", {}).get("source_missing") or {}).get(cohort, [])
    return out.with_columns(pl.lit(None, out.schema[c]).alias(c) for c in unknown if c in out.columns)
