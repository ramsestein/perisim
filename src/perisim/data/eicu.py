"""eICU-CRD → common format. Clock: minutes since hospital admission (offset − hospitaladmitoffset).

Data exist only inside the ICU (§2.2): after ICU discharge, only readmission (another stay in the
same patienthealthsystemstayid), death and the time/destination of hospital discharge are observed. T1 is
reduced to severe / non-severe (targets.yaml).
"""

from __future__ import annotations

import re

import polars as pl

from perisim.data.base import BaseExtractor
from perisim.data.common import (
    INPUTS_SCHEMA, OUTCOMES_SCHEMA, STAYS_SCHEMA, apply_bounds, as_float, as_id, as_str, conform,
    harmonized_category, map_measurements, variables_for, windowed_join,
)
from perisim.data.inspire import _common_flags
from perisim.data.mimic import _drug_rows


class EicuExtractor(BaseExtractor):
    cohort = "eicu"
    prefix = "EICU"
    _pat: pl.DataFrame | None = None

    def _patient(self) -> pl.DataFrame:
        if self._pat is None:
            p = self.table("patient")
            off = as_float("hospital_admit_offset")
            age_map = {str(k): float(v) for k, v in self.src["values"]["age_top_coded"].items()}
            self._pat = p.with_columns(
                as_id("unit_id").alias("unit_raw"), as_id("hosp_id").alias("hosp_raw"),
                as_str("subject_id").alias("subject_raw"),
                (-off).alias("icu_in_min"),
                (as_float("unit_discharge_offset") - off).alias("icu_out_min"),
                (as_float("hospital_discharge_offset") - off).alias("hosp_disch_min"),
                as_str("age").replace_strict(age_map, default=None, return_dtype=pl.Float64)
                .fill_null(as_float("age")).alias("age_num"),
            )
        return self._pat

    # ------------------------------------------------------------------ candidate stays
    def extract_stays(self) -> pl.DataFrame:
        p = self._patient()
        vals = self.src["values"]
        adx = self.table("admissiondx", optional=True)
        unit_dx = None
        if adx is not None:
            adx = adx.with_columns(as_id("unit_id").alias("unit_raw"), as_str("path").alias("path"))
            pats = self.ccfg["postop_admitdx_patterns"]
            rx = [re.compile(x) for x in pats]
            op_paths = [v for v in adx["path"].drop_nulls().unique().to_list() if any(r.search(v) for r in rx)]
            # AUDIT D11: the group comes only from the specific operative-diagnosis rows; per unit the
            # highest-priority (lowest pattern index) match wins, so it never depends on row order.
            sg = self.h["surgical_groups"]
            row_rx = re.compile(sg.get("eicu_admitdx_row_regex", r"(?i)\|Operative\|"))
            rules = [(re.compile(r["pattern"]), r["group"]) for r in sg["eicu_admitdx_regex"]]
            prio: dict[str, tuple[int, str]] = {}
            for v in adx["path"].drop_nulls().unique().to_list():
                if row_rx.search(v):
                    hit = next(((i, g) for i, (rx_, g) in enumerate(rules) if rx_.search(v)), None)
                    if hit is not None:
                        prio[v] = hit
            adx = adx.with_columns(
                pl.col("path").is_in(op_paths).alias("dx_postop"),
                pl.col("path").replace_strict({k: v[0] for k, v in prio.items()}, default=None,
                                              return_dtype=pl.Int64).alias("grp_rank"),
                pl.col("path").replace_strict({k: v[1] for k, v in prio.items()}, default=None,
                                              return_dtype=pl.Utf8).alias("grp"))
            unit_dx = adx.group_by("unit_raw").agg(
                pl.col("dx_postop").any(),
                pl.col("grp").sort_by(pl.col("grp_rank").fill_null(10 ** 6)).drop_nulls().first())
        df = p.join(unit_dx, on="unit_raw", how="left") if unit_dx is not None else \
            p.with_columns(pl.lit(False).alias("dx_postop"), pl.lit(None, pl.Utf8).alias("grp"))
        apr = self.table("apachepatientresult", optional=True)
        if apr is not None:
            apr = (apr.filter(as_str("apache_version").is_in(vals["apache_version"]))
                   .with_columns(as_id("unit_id").alias("unit_raw"), as_float("predicted_hospital_los").alias("pl"))
                   .filter(pl.col("pl") > 0).group_by("unit_raw").agg(pl.col("pl").first().alias("apache_pred_hosp_los")))
            df = df.join(apr, on="unit_raw", how="left")
        apv = self.table("apachepredvar", optional=True)
        if apv is not None:
            apv = (apv.with_columns(as_id("unit_id").alias("unit_raw"), as_float("elective_surgery").alias("elec"))
                   .group_by("unit_raw").agg(pl.col("elec").first()))
            df = df.join(apv, on="unit_raw", how="left")
        else:
            df = df.with_columns(pl.lit(None, pl.Float64).alias("elec"))

        expired = vals["expired"]
        postop = (as_str("unit_admit_source").is_in(self.ccfg["postop_admit_sources"]).fill_null(False)
                  | pl.col("dx_postop").fill_null(False))
        h_m = as_float("admission_height") / 100.0
        out = df.with_columns(
            self.sid("unit_raw").alias("stay_id"),
            pl.lit("eicu").alias("cohort"),
            self.sid("subject_raw").alias("subject_id"),
            self.sid("hosp_raw").alias("hadm_id"),
            pl.lit(0.0).alias("hosp_admit_min"),
            pl.when(as_str("hospital_discharge_status").is_in(expired)).then(pl.col("hosp_disch_min"))
            .when(as_str("unit_discharge_status").is_in(expired)).then(pl.col("icu_out_min"))
            .otherwise(None).alias("death_min"),
            pl.col("age_num").alias("age"),
            as_str("gender").is_in(vals["sex_male"]).cast(pl.Float64).alias("sex_male"),
            apply_bounds(as_float("admission_weight") / (h_m * h_m), (10, 80)).alias("bmi"),
            pl.when(pl.col("elec") == 1).then(0.0).when(pl.col("elec") == 0).then(1.0).otherwise(None).alias("emergency"),
            pl.col("grp").alias("surgical_group"),
            harmonized_category(pl.col("hospital_admit_source"), self.h["admit_source"]["eicu_hospital_admit_source"],
                                "other").alias("admit_source"),
            harmonized_category(pl.col("hospital_discharge_location"), self.h["discharge_location"]["eicu"], "other")
            .alias("discharge_location"),
            as_float("hospital_discharge_year").alias("year"),
            postop.alias("postop"),
        )
        out = _common_flags(out, self.ccfg)
        ward = as_str("unit_discharge_location").is_in(self.ccfg["ward_discharge_locations"]).fill_null(False)
        out = out.with_columns((pl.col("icu_to_ward") & ward).alias("icu_to_ward"),
                               as_float("unit_visit_number").fill_null(pl.col("unit_order")).alias("unit_order"))
        return conform(out, STAYS_SCHEMA)

    # ------------------------------------------------------------------ events
    def extract_events(self, index: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        p = self._patient()
        units = self.raw_ids(index, "stay_id")
        hosp = self.raw_ids(index, "hadm_id")
        w = index.select("stay_id", self.raw("stay_id").alias("unit_raw"), "icu_in_min", "icu_out_min",
                         "hosp_admit_min", "hosp_disch_min")
        offs = p.select("unit_raw", as_float("hospital_admit_offset").alias("hoff"))
        offs = offs.filter(pl.col("unit_raw").is_in(units))
        icu_win = (pl.col("icu_in_min") - 1440, pl.col("icu_out_min"))
        inputs, outcomes = [], []

        def _clock(df: pl.DataFrame, col: str = "offset", name: str = "time_min") -> pl.DataFrame:
            df = df.with_columns(as_id("unit_id").alias("unit_raw")).join(offs, on="unit_raw", how="inner")
            return df.with_columns((as_float(col) - pl.col("hoff")).alias(name))

        if self.has_table("lab"):
            lab = _clock(self.table("lab", filters={"unit_id": units}))
            m = map_measurements(lab, variables_for(self.h, "eicu", "lab"), "item", "value")
            inputs.append(windowed_join(m.select("unit_raw", "variable", "value", "time_min"), w, "unit_raw", *icu_win))

        vit_cols = self.src["vitals_columns"]
        bounds = {k: v.get("bounds") for k, v in self.h["variables"].items()}
        for tname in ("vitalperiodic", "vitalaperiodic"):
            if not self.has_table(tname):
                continue
            cols = [c for c in self.src["tables"][tname]["columns"] if c in vit_cols]
            vt = _clock(self.table(tname, filters={"unit_id": units}))
            # only the 48 h before ICU discharge: vital signs are used in the window before the cutoff
            vt = vt.join(w.select("unit_raw", "icu_out_min"), on="unit_raw").filter(
                pl.col("time_min") >= pl.col("icu_out_min") - 2880).with_columns([as_float(c).alias(c) for c in cols])
            long = vt.unpivot(index=["unit_raw", "time_min"], on=cols, variable_name="col", value_name="raw")
            long = long.with_columns(pl.col("col").replace_strict(vit_cols, return_dtype=pl.Utf8).alias("variable"),
                                     pl.col("raw").cast(pl.Float64).alias("value")).drop_nulls("value")
            long = pl.concat([long.filter(pl.col("variable") == v).with_columns(
                apply_bounds(pl.col("value"), bounds.get(v)).alias("value")) for v in set(vit_cols.values())]
            ).drop_nulls("value")
            inputs.append(windowed_join(long.select("unit_raw", "variable", "value", "time_min"), w, "unit_raw", *icu_win))

        if self.has_table("respiratorycharting"):
            rc = _clock(self.table("respiratorycharting", filters={"unit_id": units, "label": self.src["fio2_labels"]}))
            m = map_measurements(rc, variables_for(self.h, "eicu", "respiratorycharting"), "label", "value")
            inputs.append(windowed_join(m.select("unit_raw", "variable", "value", "time_min"), w, "unit_raw", *icu_win))
        if self.has_table("nursecharting"):
            nc = _clock(self.table("nursecharting", filters={"unit_id": units, "name": self.src["gcs_names"]}))
            m = map_measurements(nc, variables_for(self.h, "eicu", "nursecharting"), "name", "value")
            inputs.append(windowed_join(m.select("unit_raw", "variable", "value", "time_min"), w, "unit_raw", *icu_win))

        sp = self.src["support_patterns"]
        if self.has_table("infusiondrug"):
            inf = _clock(self.table("infusiondrug", filters={"unit_id": units}))
            rx = re.compile(sp["vaso_infusion_regex"])
            names = [n for n in inf["drug"].drop_nulls().unique().to_list() if rx.search(n)]
            v = inf.filter(pl.col("drug").is_in(names)).select(
                "unit_raw", pl.lit("support_vaso").alias("variable"), pl.lit(1.0).alias("value"), "time_min")
            inputs.append(windowed_join(v, w, "unit_raw", *icu_win))
        if self.has_table("respiratorycare"):
            rcare = self.table("respiratorycare", filters={"unit_id": units})
            rcare = _clock(rcare, "vent_start", "time_min").with_columns(
                (as_float("vent_end") - pl.col("hoff")).alias("end_min"))
            rcare = rcare.filter(as_float("vent_start") != 0).select(
                "unit_raw", pl.lit("support_vent").alias("variable"), pl.lit(1.0).alias("value"), "time_min", "end_min")
            inputs.append(windowed_join(rcare, w, "unit_raw", *icu_win))
        if self.has_table("treatment"):
            tr = _clock(self.table("treatment", filters={"unit_id": units}))
            for kind, pat in sp["treatment"].items():
                rx = re.compile(pat)
                names = [n for n in tr["text"].drop_nulls().unique().to_list() if rx.search(n)]
                s = tr.filter(pl.col("text").is_in(names)).select(
                    "unit_raw", pl.lit(f"support_{kind}").alias("variable"), pl.lit(1.0).alias("value"), "time_min")
                inputs.append(windowed_join(s, w, "unit_raw", *icu_win))
        if self.has_table("medication"):
            md = self.table("medication", filters={"unit_id": units}, extra=self.optional_column_names("medication"))
            md = md.filter(~as_str("cancelled").is_in(self.src["values"]["cancelled_true"]).fill_null(False))
            route_rx = self.h["medications"].get("route_exclude_regex")
            if route_rx and "route" in md.columns:   # systemic routes only (AUDIT D4)
                md = md.filter(~as_str("route").str.contains(route_rx).fill_null(False))
            md = _clock(md, "start", "time_min").with_columns((as_float("stop") - pl.col("hoff")).alias("end_min"),
                                                              as_str("drug").alias("drug"))
            inputs.append(_drug_rows(md, self.h, w, "unit_raw", *icu_win, with_end=True))
        if self.has_table("diagnosis"):
            dx = _clock(self.table("diagnosis", filters={"unit_id": units}))
            dx = dx.with_columns(as_str("icd").str.to_uppercase().str.extract_all(self.src["dx_code_regex"]).alias("codes"))
            dx = dx.explode("codes").drop_nulls("codes").select(
                "unit_raw", (pl.lit("dx:") + pl.col("codes")).alias("variable"), pl.lit(1.0).alias("value"), "time_min")
            inputs.append(windowed_join(dx.unique(), w, "unit_raw", *icu_win))

        # Outcomes: other ICU stays of the same hospitalization (readmission)
        allu = p.filter(pl.col("hosp_raw").is_in(hosp)).select(self.sid("hosp_raw").alias("hadm_id"),
                                                               pl.col("icu_in_min").alias("time_min"))
        outcomes.append(allu.join(index.select("stay_id", "hadm_id"), on="hadm_id")
                        .select("stay_id", pl.lit("icu_admission").alias("component"), "time_min",
                                pl.lit("minute").alias("precision")))
        inp = pl.concat([conform(f, INPUTS_SCHEMA) for f in inputs if f.height], how="vertical") \
            if any(f.height for f in inputs) else conform(pl.DataFrame(), INPUTS_SCHEMA)
        out = pl.concat([conform(f, OUTCOMES_SCHEMA) for f in outcomes if f.height], how="vertical") \
            if any(f.height for f in outcomes) else conform(pl.DataFrame(), OUTCOMES_SCHEMA)
        return inp, out
