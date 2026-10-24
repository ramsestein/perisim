"""INSPIRE (SNUH) → common format. Relative times in minutes since the subject's first admission (§2.2).

Critical points [VERIFY]:
- icuin_time / icuout_time only record the ICU within 24 h after surgery: ICU readmission is detected
  as a new ICU entry linked to another surgery in the same hospitalization OR, as the SPECS D1 proxy, a new
  invasive-ventilation / RRT / ECMO / IABP run after t0 in ward_vitals (targets.yaml: icu_readmission_proxy,
  AUDIT D10). The audit reports how many readmissions each source contributes.
- Devices (ventilation, RRT, ECMO, IABP) are read from ward_vitals as "active" records;
  the end is inferred from the absence of records (features.support_gap_hours).
- Chronic dialysis (D5): ESRD / dialysis codes and RRT device records before the index surgery are emitted as
  `marker:chronic_rrt` rows.
"""

from __future__ import annotations

import re

import polars as pl

from perisim.data.base import BaseExtractor
from perisim.data.common import (
    INPUTS_SCHEMA, OUTCOMES_SCHEMA, STAYS_SCHEMA, apply_bounds, as_float, as_id, as_str, conform,
    harmonized_category, map_measurements, variables_for, windowed_join,
)

CHRONIC_RRT_MARKER = "marker:chronic_rrt"

_TIME_COLS = ["op_start", "op_end", "admission", "discharge", "icu_in", "icu_out", "death_inhosp"]


class InspireExtractor(BaseExtractor):
    cohort = "inspire"
    prefix = "INSPIRE"
    _ops: pl.DataFrame | None = None

    def _operations(self) -> pl.DataFrame:
        if self._ops is None:
            df = self.table("operations", extra=self.optional_column_names("operations"))
            self._ops = df.with_columns([as_float(c).alias(c) for c in ["age", "weight", "height", "asa"] + _TIME_COLS])
        return self._ops

    # ------------------------------------------------------------------ candidate stays
    def extract_stays(self) -> pl.DataFrame:
        ops = self._operations()
        vals = self.src["values"]
        tol = float(self.ccfg.get("postop_negative_tolerance_min", 60))
        win = float(self.ccfg.get("postop_window_hours", 24)) * 60
        opt = self.optional_column_names("operations")

        icu = ops.filter(pl.col("icu_in").is_not_null() & pl.col("icu_out").is_not_null())
        gap = pl.col("icu_in") - pl.col("op_end")
        icu = icu.with_columns(((gap >= -tol) & (gap <= win)).fill_null(False).alias("postop"))
        # An ICU entry may appear in several surgery rows: take the most recent postoperative surgery
        icu = (icu.sort(["hadm_id", "icu_in", "postop", "op_end"], nulls_last=False)
               .unique(subset=["hadm_id", "icu_in"], keep="last", maintain_order=True))

        height_m = pl.col("height") / 100.0
        exprs = [
            self.sid("op_id").alias("stay_id"),
            pl.lit("inspire").alias("cohort"),
            self.sid("subject_id").alias("subject_id"),
            self.sid("hadm_id").alias("hadm_id"),
            pl.col("icu_in").alias("icu_in_min"),
            pl.col("icu_out").alias("icu_out_min"),
            pl.col("admission").alias("hosp_admit_min"),
            pl.col("discharge").alias("hosp_disch_min"),
            pl.col("death_inhosp").alias("death_min"),
            pl.col("op_start").alias("surgery_start_min"),
            pl.col("op_end").alias("surgery_end_min"),
            pl.col("age").alias("age"),
            as_str("sex").is_in(vals["sex_male"]).cast(pl.Float64).alias("sex_male"),
            apply_bounds(pl.col("weight") / (height_m * height_m), (10, 80)).alias("bmi"),
            apply_bounds(pl.col("asa"), (1, 6)).alias("asa"),
            as_str("emergency").is_in(vals["emergency_true"]).cast(pl.Float64).alias("emergency"),
            as_str("anesthesia_type").str.to_uppercase().alias("anesthesia_type"),
            harmonized_category(pl.col("department"), self.h["surgical_groups"]["inspire_department"]).alias("surgical_group"),
            as_str("pcs_code").str.to_uppercase().str.replace_all(r"[^0-9A-Z]", "").alias("pcs_code"),
            pl.col("postop"),
        ]
        if "year" in opt:
            exprs.append(as_float("year").alias("year"))
        if "donor_flag" in opt:
            exprs.append(as_str("donor_flag").is_in(vals["donor_true"]).alias("organ_donor"))
        if "limitation" in opt:
            exprs.append(as_str("limitation").is_in(vals["donor_true"]).alias("limitation_known"))
        if "discharge_location" in opt:
            exprs.append(harmonized_category(pl.col("discharge_location"),
                                             self.h["discharge_location"].get("inspire", {}), "other")
                         .alias("discharge_location"))
        # C5: minutes from the ICU exit to the start of the next operation of the hospitalization
        nxt = (icu.select("op_id", "hadm_id", "icu_out").join(ops.select("hadm_id", pl.col("op_start").alias("_s")),
                                                              on="hadm_id", how="inner")
               .filter(pl.col("_s") > pl.col("icu_out"))
               .group_by("op_id").agg((pl.col("_s") - pl.col("icu_out")).min().alias("return_within_min")))
        icu = icu.join(nxt, on="op_id", how="left")
        exprs.append(pl.col("return_within_min"))
        out = icu.with_columns(exprs).select([e.meta.output_name() for e in exprs])
        out = _common_flags(out, self.ccfg)
        if "discharge_location" not in opt:
            out = out.with_columns(pl.when(pl.col("death_min").is_not_null()).then(pl.lit("death"))
                                   .otherwise(pl.lit(None, pl.Utf8)).alias("discharge_location"))
        return conform(out, STAYS_SCHEMA)

    # ------------------------------------------------------------------ events
    def extract_events(self, index: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        subj = self.raw_ids(index, "subject_id")
        fcfg = self.cfg.experiment["features"]
        lookback = float(fcfg["preop_lab_lookback_days"]) * 1440
        w = index.select("stay_id", self.raw("subject_id").alias("subject_raw"), "hosp_admit_min", "hosp_disch_min",
                         "icu_in_min", "icu_out_min", "surgery_start_min")
        admit = pl.coalesce(pl.col("hosp_admit_min"), pl.col("icu_in_min") - 30 * 1440)
        hi = pl.min_horizontal(pl.col("hosp_disch_min").fill_null(float("inf")), pl.col("icu_out_min") + self.followup_min)
        inputs, outcomes = [], []

        def _prep(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(as_id("subject_id").alias("subject_raw"), as_float("time").alias("time_min"))

        # Labs (includes preoperative labs up to 6 months before and post-t0 labs for the targets)
        labs = _prep(self.table("labs", filters={"subject_id": subj}))
        m = map_measurements(labs, variables_for(self.h, "inspire", "labs"), "item", "value")
        m = windowed_join(m.select("subject_raw", "variable", "value", "time_min"), w, "subject_raw",
                          admit - lookback, hi)
        inputs.append(m)

        # Ward and ICU vital signs + devices
        wv = _prep(self.table("ward_vitals", filters={"subject_id": subj}))
        v = map_measurements(wv, variables_for(self.h, "inspire", "ward_vitals"), "item", "value")
        v = windowed_join(v.select("subject_raw", "variable", "value", "time_min"), w, "subject_raw",
                          pl.col("icu_in_min") - 1440, hi)
        inputs.append(v)
        item = pl.col("item").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
        active = as_float("value").is_null() | (as_float("value") != 0)
        surg = pl.coalesce(pl.col("surgery_start_min"), pl.col("icu_in_min"))
        for kind, items in self.src["devices"]["items"].items():
            d = wv.filter(item.is_in([i.lower() for i in items]) & active)
            d = d.select("subject_raw", pl.lit(f"support_{kind}").alias("variable"), pl.lit(1.0).alias("value"),
                         "time_min")
            inputs.append(windowed_join(d, w, "subject_raw", admit, hi))
            if kind == "rrt":   # D5: any RRT record before the index surgery (including earlier admissions)
                mk = d.with_columns(pl.lit(CHRONIC_RRT_MARKER).alias("variable"))
                inputs.append(windowed_join(mk, w, "subject_raw", admit - lookback, surg - 1e-6))

        # Medication (ATC) + vasopressors as support + transfusion as outcome
        atc_cols = self.src["tables"]["medications"].get("atc_columns") or []
        med = _prep(self.table("medications", filters={"subject_id": subj},
                               extra={f"atc_{i}": c for i, c in enumerate(atc_cols)}))
        codes = med.select("subject_raw", "time_min", "drug",
                           *[as_str(f"atc_{i}").str.to_uppercase().str.replace_all(r"\s", "").alias(f"atc_{i}")
                             for i in range(len(atc_cols))])
        long = (codes.unpivot(index=["subject_raw", "time_min", "drug"], on=[f"atc_{i}" for i in range(len(atc_cols))],
                              value_name="atc")
                .filter(pl.col("atc").str.len_chars() >= 3).unique(["subject_raw", "time_min", "atc"]))
        mrows = long.select("subject_raw", (pl.lit("atc:") + pl.col("atc")).alias("variable"),
                            pl.lit(1.0).alias("value"), "time_min")
        inputs.append(windowed_join(mrows, w, "subject_raw", pl.col("icu_in_min") - 1440, hi))
        vaso = long.filter(pl.col("atc").is_in(self.h["medications"]["vasopressor_atc"]))
        vaso = vaso.select("subject_raw", pl.lit("support_vaso").alias("variable"), pl.lit(1.0).alias("value"), "time_min")
        inputs.append(windowed_join(vaso, w, "subject_raw", admit, hi))
        rx = re.compile(self.src["transfusion_drug_regex"])
        tnames = [n for n in med["drug"].drop_nulls().unique().to_list() if rx.search(str(n))]
        tr = med.filter(pl.col("drug").is_in(tnames)).select("subject_raw", "time_min")
        tr = windowed_join(tr, w, "subject_raw", admit, hi)
        outcomes.append(tr.select("stay_id", pl.lit("transfusion_rbc").alias("component"), "time_min",
                                  pl.lit("minute").alias("precision")))

        # ICD-10 diagnoses at 3 characters, only up to t0 (the exact cutoff is applied in features)
        dx = _prep(self.table("diagnosis", filters={"subject_id": subj}))
        n = int(self.src.get("dx_code_length", 3))
        full = as_str("code").str.to_uppercase().str.replace_all(r"[^0-9A-Z]", "")
        if self.src.get("chronic_rrt_dx_regex"):   # D5: ESRD / dialysis-dependence codes before the index surgery
            mk = dx.filter(full.str.contains(self.src["chronic_rrt_dx_regex"])).select(
                "subject_raw", pl.lit(CHRONIC_RRT_MARKER).alias("variable"), pl.lit(1.0).alias("value"), "time_min")
            inputs.append(windowed_join(mk, w, "subject_raw", admit - lookback, surg - 1e-6))
        dx = dx.with_columns(full.str.slice(0, n).alias("c"))
        dx = dx.filter(pl.col("c").str.len_chars() == n).select(
            "subject_raw", (pl.lit("dx:") + pl.col("c")).alias("variable"), pl.lit(1.0).alias("value"), "time_min")
        inputs.append(windowed_join(dx, w, "subject_raw", admit - 1440, pl.col("icu_out_min")))

        # Structural outcomes: ICU entries and surgeries in the same hospitalization
        ops = self._operations().with_columns(self.sid("hadm_id").alias("hadm_id"))
        idx = index.select("stay_id", "hadm_id")
        icu_adm = (ops.filter(pl.col("icu_in").is_not_null()).select("hadm_id", pl.col("icu_in").alias("time_min"))
                   .unique().join(idx, on="hadm_id"))
        outcomes.append(icu_adm.select("stay_id", pl.lit("icu_admission").alias("component"), "time_min",
                                       pl.lit("minute").alias("precision")))
        surg = (ops.filter(pl.col("op_start").is_not_null()).select("hadm_id", pl.col("op_start").alias("time_min"))
                .unique().join(idx, on="hadm_id"))
        outcomes.append(surg.select("stay_id", pl.lit("surgery").alias("component"), "time_min",
                                    pl.lit("minute").alias("precision")))

        inp = pl.concat([conform(f, INPUTS_SCHEMA) for f in inputs if f.height], how="vertical") if inputs else None
        out = pl.concat([conform(f, OUTCOMES_SCHEMA) for f in outcomes if f.height], how="vertical") if outcomes else None
        return (inp if inp is not None else conform(pl.DataFrame(), INPUTS_SCHEMA),
                out if out is not None else conform(pl.DataFrame(), OUTCOMES_SCHEMA))


def _common_flags(df: pl.DataFrame, ccfg: dict) -> pl.DataFrame:
    """died_in_icu, icu_to_ward and order of the stay within the hospitalization (common to all cohorts)."""
    dtol = float(ccfg.get("icu_death_tolerance_min", 60))
    xtol = float(ccfg.get("discharge_tolerance_min", 60))
    died = (pl.col("death_min").is_not_null() & (pl.col("death_min") <= pl.col("icu_out_min") + dtol))
    ward = ((pl.col("hosp_disch_min") - pl.col("icu_out_min")) > xtol) & ~died
    return df.with_columns(
        died.alias("died_in_icu"),
        ward.alias("icu_to_ward"),
        pl.col("icu_in_min").rank("ordinal").over("hadm_id").cast(pl.Float64).alias("unit_order"),
    )
