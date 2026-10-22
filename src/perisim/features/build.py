"""Per-stay input matrix, with temporal provenance for the anti-leakage test (§4.4.1).

Only inputs_long rows with timestamp ≤ cutoff (t0 − margin) are used. Intervals (supports, medication
orders) are clipped at the cutoff. For each row of the matrix the maximum timestamp actually used is stored
(`feature_provenance`), which the test compares with the cutoff.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import polars as pl

from perisim.cohort.select import add_cutoff
from perisim.timeline import merge_runs

SLOPE_VITALS = ("hr", "map", "rr", "spo2", "temp")
SUPPORTS = ("vent", "vaso", "rrt")


@dataclass
class FeatureSpec:
    name: str
    type: str            # continuous | ordinal | binary | nominal | hier
    group: str           # group for feature dropout / SAGE (§5.3.1)
    block: str           # clinical block (§4.1)
    levels: list = field(default_factory=list)   # ordinal: order of the levels

    def to_dict(self) -> dict:
        return asdict(self)


def _last(df: pl.DataFrame, prefix_from: str, prefix_to: str) -> pl.DataFrame:
    return (df.sort("time_min").group_by(["stay_id", "variable"])
            .agg(pl.col("value").last(), pl.col("time_min").last().alias("t_used"))
            .with_columns((pl.lit(prefix_to) + pl.col("variable").str.replace(f"^{prefix_from}", "")).alias("name")))


def build_features(stays: pl.DataFrame, inputs: pl.DataFrame, cfg, cohort: str
                   ) -> tuple[pl.DataFrame, list[FeatureSpec], pl.DataFrame]:
    f = cfg.experiment["features"]
    margin = float(cfg.experiment["time"]["cutoff_margin_hours"])
    s = add_cutoff(stays, margin)
    base = s.select("stay_id", "cutoff_min", "icu_in_min", "surgery_start_min")
    L = inputs.join(base, on="stay_id", how="inner").filter(pl.col("time_min") <= pl.col("cutoff_min"))
    long_parts: list[pl.DataFrame] = []   # (stay_id, name, value, t_used)
    specs: list[FeatureSpec] = []

    # --- discharge status: last lab result within the stay (with a preceding margin)
    lab_lo = pl.col("icu_in_min") - float(f["lab_lookback_hours"]) * 60
    labs = L.filter(pl.col("variable").str.starts_with("lab_") & (pl.col("time_min") >= lab_lo))
    long_parts.append(_last(labs, "lab_", "last_"))

    # --- baseline: most recent preoperative lab result (before the start of the index surgery)
    preop_vars = cfg.harmonization.get("preop_labs", [])
    pre_lo = pl.col("surgery_start_min") - float(f["preop_lab_lookback_days"]) * 1440
    pre = L.filter(pl.col("variable").is_in(preop_vars) & pl.col("surgery_start_min").is_not_null()
                   & (pl.col("time_min") <= pl.col("surgery_start_min")) & (pl.col("time_min") >= pre_lo))
    long_parts.append(_last(pre, "lab_", "preop_"))

    # --- vital signs: last value and trend (slope per hour) in the 24 h before the cutoff
    vlo = pl.col("cutoff_min") - float(f["vital_window_hours"]) * 60
    vit = L.filter(pl.col("variable").str.starts_with("vital_") & (pl.col("time_min") >= vlo))
    long_parts.append(_last(vit, "vital_", "last_"))
    th = (pl.col("time_min") - pl.col("cutoff_min")) / 60.0
    slope = (vit.filter(pl.col("variable").is_in([f"vital_{v}" for v in SLOPE_VITALS]))
             .with_columns(th.alias("th"))
             .group_by(["stay_id", "variable"])
             .agg(pl.len().alias("n"), pl.col("th").mean().alias("mt"), pl.col("value").mean().alias("mv"),
                  (pl.col("th") * pl.col("value")).mean().alias("mtv"), (pl.col("th") ** 2).mean().alias("mtt"),
                  pl.col("time_min").max().alias("t_used"))
             .with_columns((pl.col("mtt") - pl.col("mt") ** 2).alias("vt"))
             .filter((pl.col("n") >= int(f["min_points_slope"])) & (pl.col("vt") > 1e-9))
             .with_columns(((pl.col("mtv") - pl.col("mt") * pl.col("mv")) / pl.col("vt")).alias("value"),
                           (pl.lit("slope_") + pl.col("variable").str.replace("^vital_", "")).alias("name")))
    long_parts.append(slope.select("stay_id", "variable", "value", "t_used", "name"))

    # --- supports during the ICU stay and hours since their withdrawal
    sup = L.filter(pl.col("variable").str.starts_with("support_"))
    runs = merge_runs(sup.select("stay_id", "variable", "time_min", "end_min"), float(f["support_gap_hours"]) * 60)
    runs = (runs.join(base, on="stay_id")
            .with_columns(pl.min_horizontal("end_min", "cutoff_min").alias("end_clip"))
            .filter((pl.col("end_clip") >= pl.col("icu_in_min")) & (pl.col("start_min") <= pl.col("cutoff_min"))))
    last_off = runs.group_by(["stay_id", "variable"]).agg(pl.col("end_clip").max().alias("t_used"),
                                                          pl.col("cutoff_min").first())
    long_parts.append(last_off.select("stay_id", "variable", pl.lit(1.0).alias("value"), "t_used",
                                      (pl.lit("used_") + pl.col("variable").str.replace("^support_", "")).alias("name")))
    long_parts.append(last_off.select("stay_id", "variable",
                                      ((pl.col("cutoff_min") - pl.col("t_used")) / 60.0).alias("value"), "t_used",
                                      (pl.lit("hours_since_") + pl.col("variable").str.replace("^support_", "")
                                       + pl.lit("_off")).alias("name")))

    # --- active medication: level-3 ATC groups in the 24 h before the cutoff
    lvl = int(cfg.harmonization["medications"]["atc_feature_level_chars"])
    mlo = pl.col("cutoff_min") - float(f["med_window_hours"]) * 60
    med = L.filter(pl.col("variable").str.starts_with("atc:")
                   & (pl.coalesce("end_min", "time_min") >= mlo))
    med = med.with_columns(pl.col("variable").str.slice(4, lvl).alias("atc3"),
                           pl.min_horizontal(pl.coalesce("end_min", "time_min"), "cutoff_min").alias("t_used"))
    med = med.filter(pl.col("atc3").str.len_chars() == lvl).group_by(["stay_id", "atc3"]).agg(pl.col("t_used").max())
    long_parts.append(med.select("stay_id", pl.lit("med").alias("variable"), pl.lit(1.0).alias("value"), "t_used",
                                 (pl.lit("med_") + pl.col("atc3")).alias("name")))

    # --- diagnoses: before the start of the index surgery (main) or before the cutoff (sensitivity)
    dx = L.filter(pl.col("variable").str.starts_with("dx:"))
    if f["dx_window"] == "before_surgery":
        dx = dx.filter(pl.col("surgery_start_min").is_null() | (pl.col("time_min") < pl.col("surgery_start_min")))
    dx = dx.group_by(["stay_id", "variable"]).agg(pl.col("time_min").max().alias("t_used"))
    long_parts.append(dx.select("stay_id", "variable", pl.lit(1.0).alias("value"), "t_used",
                                (pl.lit("dx_") + pl.col("variable").str.slice(3)).alias("name")))

    # --- recorded anthropometry (MIMIC-IV) for BMI if it is not provided in stays
    anth = _last(L.filter(pl.col("variable").str.starts_with("anthro_")), "anthro_", "anthro_")

    long = pl.concat([p.select("stay_id", "name", pl.col("value").cast(pl.Float64), pl.col("t_used").cast(pl.Float64))
                      for p in long_parts if p.height], how="vertical")
    wide = long.pivot(on="name", index="stay_id", values="value", aggregate_function="first") if long.height else \
        s.select("stay_id")

    # --- static variables and surgery
    st = s.select(
        "stay_id", "age", "sex_male", "asa", "emergency", "anesthesia_type", "surgical_group", "admit_source",
        ((pl.col("surgery_end_min") - pl.col("surgery_start_min"))).alias("surgery_duration_min"),
        ((pl.col("cutoff_min") - pl.col("icu_in_min")) / 60.0).alias("icu_los_h"),
        *[pl.col("pcs_code").str.slice(0, k).alias(f"pcs_l{k}") for k in range(1, 6)],
        "bmi",
    )
    st = st.with_columns([pl.when(pl.col(f"pcs_l{k}").str.len_chars() == k).then(pl.col(f"pcs_l{k}")).otherwise(None)
                          .alias(f"pcs_l{k}") for k in range(1, 6)])
    if anth.height:
        aw = anth.pivot(on="name", index="stay_id", values="value", aggregate_function="first")
        if {"anthro_weight", "anthro_height"} <= set(aw.columns):
            bmi = pl.col("anthro_weight") / (pl.col("anthro_height") / 100.0) ** 2
            st = st.join(aw.select("stay_id", pl.when(bmi.is_between(10, 80)).then(bmi).otherwise(None).alias("_bmi")),
                         on="stay_id", how="left").with_columns(pl.coalesce("bmi", "_bmi").alias("bmi")).drop("_bmi")
    X = st.join(wide, on="stay_id", how="left").sort("stay_id")

    # --- specification of metric types (§5.2) and groups (§5.3.1)
    for c in X.columns:
        if c == "stay_id":
            continue
        spec = _spec_for(c)
        if spec is not None:
            specs.append(spec)
    keep = ["stay_id"] + [sp.name for sp in specs]
    X = X.select(keep)
    # columns that are entirely empty in this cohort are kept (availability is measured later)

    # --- temporal provenance per stay
    static_t = s.select("stay_id", pl.max_horizontal(
        "icu_in_min", pl.col("surgery_start_min"), pl.col("surgery_end_min")).alias("t_used"))
    prov_parts = [long.select("stay_id", "t_used"), static_t]
    if anth.height:
        prov_parts.append(anth.select("stay_id", "t_used"))
    prov = (pl.concat(prov_parts, how="vertical").group_by("stay_id").agg(pl.col("t_used").max().alias("max_time_used_min"))
            .join(s.select("stay_id", "cutoff_min", "t0_min"), on="stay_id", how="right"))
    return X, specs, prov


def _spec_for(name: str) -> FeatureSpec | None:
    if name in ("age", "bmi"):
        return FeatureSpec(name, "continuous", name, "demographics")
    if name == "sex_male":
        return FeatureSpec(name, "binary", name, "demographics")
    if name == "asa":
        return FeatureSpec(name, "ordinal", name, "baseline", levels=[1, 2, 3, 4, 5, 6])
    if name == "emergency":
        return FeatureSpec(name, "binary", name, "baseline")
    if name.startswith("preop_"):
        return FeatureSpec(name, "continuous", name, "baseline")
    if name.startswith("pcs_l"):
        return FeatureSpec(name, "hier", "pcs", "surgery")
    if name in ("surgical_group", "anesthesia_type", "admit_source"):
        return FeatureSpec(name, "nominal", name, "surgery")
    if name == "surgery_duration_min":
        return FeatureSpec(name, "continuous", name, "surgery")
    if name == "icu_los_h":
        return FeatureSpec(name, "continuous", name, "icu")
    if name.startswith("used_"):
        k = name[5:]
        return FeatureSpec(name, "binary", f"support_{k}", "icu")
    if name.startswith("hours_since_"):
        k = name[len("hours_since_"):-len("_off")]
        return FeatureSpec(name, "continuous", f"support_{k}", "icu")
    if name.startswith("last_") or name.startswith("slope_"):
        return FeatureSpec(name, "continuous", name, "discharge_status")
    if name.startswith("dx_"):
        return FeatureSpec(name, "binary", name, "diagnoses")
    if name.startswith("med_"):
        return FeatureSpec(name, "binary", name, "medication")
    return None


def check_no_leakage(prov: pl.DataFrame) -> pl.DataFrame:
    """Rows that violate the anti-leakage rule (max_time_used > cutoff). Empty = OK."""
    return prov.filter(pl.col("max_time_used_min") > pl.col("cutoff_min") + 1e-6)
