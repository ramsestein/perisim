"""MIMIC-IV (BIDMC) → common format. Clock: minutes since 1970 of the shifted dates.

Particularities (§2.2) [VERIFY against the mounted version]:
- There are no ward vital signs (chartevents is ICU only): mild events on the ward are observed via labs and eMAR.
- procedures_icd only has a date: reoperation has day precision.
- diagnoses_icd has no timestamp (billing codes at discharge): NOT used as input (leakage, §4.4).
- icustays merges contiguous ICU and step-down careunits: ICU episodes are rebuilt from hosp.transfers with
  true ICU careunits only (AUDIT D3), so t0 is the exit from the last true ICU careunit and a step-down
  transfer is neither an index stay nor a readmission.
"""

from __future__ import annotations

import re

import polars as pl

from perisim.data.base import BaseExtractor
from perisim.data.common import (
    INPUTS_SCHEMA, OUTCOMES_SCHEMA, STAYS_SCHEMA, apply_bounds, as_float, as_id, as_str, conform, date_minutes,
    dt_minutes, harmonized_category, map_measurements, regex_map_unique, variables_for, windowed_join,
)
from perisim.data.inspire import _common_flags


class MimicExtractor(BaseExtractor):
    cohort = "mimic"
    prefix = "MIMIC"

    # ------------------------------------------------------------------ surgical rules
    def _or_procedures(self) -> pl.DataFrame:
        """Surgical procedures (ICD-10-PCS / ICD-9) with their day, per hospitalization."""
        prc = self.table("procedures_icd")
        prc = prc.with_columns(
            as_id("hadm_id").alias("hadm_raw"),
            date_minutes(prc, "chartdate").alias("day_min"),
            as_str("icd_code").str.to_uppercase().str.replace_all(r"[^0-9A-Z]", "").alias("code"),
            as_str("icd_version").alias("ver"),
            as_float("seq_num").alias("seq"),
        )
        c = self.ccfg
        code = pl.col("code")
        icd10 = ((pl.col("ver") == "10") & (code.str.slice(0, 1) == "0")
                 & code.str.slice(4, 1).is_in(c["or_icd10_approaches"])
                 & ~code.str.slice(2, 1).is_in(c["or_icd10_exclude_root_operations"]))
        lo, hi = c["or_icd9_range"]
        chap = code.str.slice(0, 2).cast(pl.Int32, strict=False)
        excl = pl.any_horizontal([code.str.starts_with(p) for p in c["or_icd9_exclude_prefixes"]])
        icd9 = (pl.col("ver") == "9") & chap.is_between(lo, hi) & ~excl
        return prc.filter(icd10 | icd9).select("hadm_raw", "day_min", "code", "ver", "seq")

    # ------------------------------------------------------------------ ICU episodes (AUDIT D3)
    def _icu_episodes(self, hadm: list[str] | None = None) -> pl.DataFrame:
        """ICU episodes = runs of consecutive transfers in true ICU careunits. Columns: unit_raw (first transfer_id
        of the episode), hadm_raw, subject_raw, icu_in_min, icu_out_min. Falls back to icustays without transfers."""
        filters = {"hadm_id": hadm} if hadm is not None else None
        if not self.has_table("transfers"):
            self.log.warning("MIMIC without hosp.transfers: using icustays (step-down units are not separated)")
            icu = self.table("icustays", filters=filters)
            return icu.select(as_id("stay_id").alias("unit_raw"), as_id("hadm_id").alias("hadm_raw"),
                              as_id("subject_id").alias("subject_raw"), dt_minutes(icu, "intime").alias("icu_in_min"),
                              dt_minutes(icu, "outtime").alias("icu_out_min"))
        tr = self.table("transfers", filters=filters)
        units = self.ccfg["icu_careunits"]
        gap = float(self.ccfg.get("icu_episode_merge_gap_min", 60))
        tr = (tr.with_columns(as_id("transfer_id").alias("tid"), as_id("hadm_id").alias("hadm_raw"),
                              as_id("subject_id").alias("subject_raw"), dt_minutes(tr, "intime").alias("t_in"),
                              dt_minutes(tr, "outtime").alias("t_out"))
              .filter(as_str("careunit").is_in(units) & pl.col("hadm_raw").is_not_null()
                      & pl.col("t_in").is_not_null() & pl.col("t_out").is_not_null())
              .sort(["hadm_raw", "t_in"]))
        tr = tr.with_columns(pl.col("t_out").cum_max().shift(1).over("hadm_raw").alias("_prev"))
        tr = tr.with_columns((pl.col("_prev").is_null() | (pl.col("t_in") > pl.col("_prev") + gap))
                             .cast(pl.Int32).cum_sum().over("hadm_raw").alias("_ep"))
        return (tr.group_by(["hadm_raw", "_ep"], maintain_order=True)
                .agg(pl.col("tid").first().alias("unit_raw"), pl.col("subject_raw").first(),
                     pl.col("t_in").min().alias("icu_in_min"), pl.col("t_out").max().alias("icu_out_min"))
                .drop("_ep"))

    # ------------------------------------------------------------------ candidate stays
    def extract_stays(self) -> pl.DataFrame:
        icu = self._icu_episodes()
        adm = self.table("admissions")
        adm = adm.with_columns(as_id("hadm_id").alias("hadm_raw"),
                               dt_minutes(adm, "admittime").alias("hosp_admit_min"),
                               dt_minutes(adm, "dischtime").alias("hosp_disch_min"),
                               dt_minutes(adm, "deathtime").alias("death_raw"),
                               as_str("hospital_expire_flag").alias("expire"),
                               as_str("admittime").str.slice(0, 4).cast(pl.Int32, strict=False).alias("admit_year"))
        pat = self.table("patients")
        pat = pat.with_columns(as_id("subject_id").alias("subject_raw"), as_float("anchor_age").alias("anchor_age"),
                               as_float("anchor_year").alias("anchor_year"))
        df = (icu.join(adm.drop("subject_id"), on="hadm_raw", how="inner")
              .join(pat.select("subject_raw", "gender", "anchor_age", "anchor_year"), on="subject_raw", how="left"))

        # Rule D10: surgical service active at ICU admission, or a surgical procedure on that day or the day before
        srv = self.table("services")
        srv = srv.with_columns(as_id("hadm_id").alias("hadm_raw"), dt_minutes(srv, "transfertime").alias("t_srv"),
                               as_str("curr_service").str.to_uppercase().alias("service"))
        lb = float(self.ccfg.get("service_lookback_min", 60))
        last_srv = (df.select("unit_raw", "hadm_raw", "icu_in_min")
                    .join(srv.select("hadm_raw", "t_srv", "service"), on="hadm_raw", how="inner")
                    .filter(pl.col("t_srv") <= pl.col("icu_in_min") + lb)
                    .sort("t_srv").group_by("unit_raw").agg(pl.col("service").last()))
        df = df.join(last_srv, on="unit_raw", how="left")
        surgical = self.ccfg["surgical_services"]
        days_before = int(self.ccfg.get("procedure_days_before", 1))
        orp = self._or_procedures()
        day = lambda c: (pl.col(c) / 1440).floor() * 1440  # noqa: E731
        # AUDIT C2: procedures are date-only, so a same-day procedure may follow t0. The index procedure code
        # (pcs) is taken preferably from the day BEFORE the ICU admission; a same-day code is only used when t0
        # falls on a later day. Earliest day first, then billing order (deterministic tie-break only).
        proc = (df.select("unit_raw", "hadm_raw", day("icu_in_min").alias("icu_day"), day("icu_out_min").alias("t0_day"))
                .join(orp, on="hadm_raw", how="inner")
                .filter((pl.col("day_min") <= pl.col("icu_day")) & (pl.col("day_min") >= pl.col("icu_day") - days_before * 1440))
                .with_columns(((pl.col("day_min") < pl.col("icu_day")) | (pl.col("t0_day") > pl.col("icu_day"))).alias("pcs_ok"))
                .sort(["day_min", "seq"])
                .group_by("unit_raw", maintain_order=True)
                .agg(pl.col("code").filter((pl.col("ver") == "10") & pl.col("pcs_ok")).first().alias("pcs_code"),
                     pl.len().alias("n_or")))
        df = df.join(proc, on="unit_raw", how="left")
        rule = self.ccfg.get("postop_rule", "service_or_procedure")
        by_srv = pl.col("service").is_in(surgical).fill_null(False)
        by_prc = pl.col("n_or").fill_null(0) > 0
        postop = {"service": by_srv, "procedure": by_prc, "service_and_procedure": by_srv & by_prc}.get(rule, by_srv | by_prc)

        death = pl.when(pl.col("death_raw").is_not_null()).then(pl.col("death_raw")) \
            .when(pl.col("expire") == "1").then(pl.col("hosp_disch_min")).otherwise(None)
        hmap = self.h
        out = df.with_columns(
            self.sid("unit_raw").alias("stay_id"),
            pl.lit("mimic").alias("cohort"),
            self.sid("subject_raw").alias("subject_id"),
            self.sid("hadm_raw").alias("hadm_id"),
            death.alias("death_min"),
            (pl.col("anchor_age") + (pl.col("admit_year") - pl.col("anchor_year"))).alias("age"),
            (as_str("gender").str.to_uppercase() == "M").cast(pl.Float64).alias("sex_male"),
            as_str("admission_type").str.to_uppercase().is_in([x.upper() for x in self.ccfg["emergency_admission_types"]])
            .cast(pl.Float64).alias("emergency"),
            harmonized_category(pl.col("service"), hmap["surgical_groups"]["mimic_service"], None).alias("surgical_group"),
            harmonized_category(pl.col("admission_location"), hmap["admit_source"]["mimic_admission_location"], "other")
            .alias("admit_source"),
            harmonized_category(pl.col("discharge_location"), hmap["discharge_location"]["mimic"], "other")
            .alias("discharge_location"),
            postop.alias("postop"),
        )
        out = _common_flags(out, self.ccfg)
        return conform(out, STAYS_SCHEMA)

    # ------------------------------------------------------------------ events
    def extract_events(self, index: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        subj = self.raw_ids(index, "subject_id")
        hadm = self.raw_ids(index, "hadm_id")
        w = index.select("stay_id", self.raw("subject_id").alias("subject_raw"), self.raw("hadm_id").alias("hadm_raw"),
                         "hosp_admit_min", "hosp_disch_min", "icu_in_min", "icu_out_min")
        lookback = float(self.cfg.experiment["features"]["preop_lab_lookback_days"]) * 1440
        admit = pl.coalesce(pl.col("hosp_admit_min"), pl.col("icu_in_min") - 30 * 1440)
        hi = pl.min_horizontal(pl.col("hosp_disch_min").fill_null(float("inf")), pl.col("icu_out_min") + self.followup_min)
        inputs, outcomes = [], []

        # Labs (per subject: includes prior outpatient labs, with their date)
        lab_specs = variables_for(self.h, "mimic", "labevents")
        items = sorted({i for s in lab_specs for i in s["items"]})
        lab = self.table("labevents", filters={"subject_id": subj, "itemid": items})
        lab = lab.with_columns(as_id("subject_id").alias("subject_raw"), dt_minutes(lab, "charttime").alias("time_min"))
        m = map_measurements(lab, lab_specs, "itemid", "valuenum")
        inputs.append(windowed_join(m.select("subject_raw", "variable", "value", "time_min"), w, "subject_raw",
                                    admit - lookback, hi))

        # ICU vital signs within the index ICU EPISODE (there are no ward vital signs)
        ch_specs = variables_for(self.h, "mimic", "chartevents")
        gcs_items = [str(i) for i in self.src.get("gcs_component_items", [])]
        citems = sorted({i for s in ch_specs for i in s["items"]} | set(gcs_items))
        ch = self.table("chartevents", filters={"hadm_id": hadm, "itemid": citems})
        ch = ch.with_columns(as_id("hadm_id").alias("hadm_raw"), dt_minutes(ch, "charttime").alias("time_min"))
        cm = map_measurements(ch, ch_specs, "itemid", "valuenum")
        inputs.append(windowed_join(cm.select("hadm_raw", "variable", "value", "time_min"), w, "hadm_raw",
                                    pl.col("icu_in_min"), pl.col("icu_out_min")))
        if gcs_items:
            g = (ch.filter(as_str("itemid").is_in(gcs_items))
                 .with_columns(as_float("valuenum").alias("v"))
                 .group_by(["hadm_raw", "time_min"]).agg(pl.col("v").sum().alias("value"), pl.len().alias("n"))
                 .filter(pl.col("n") == len(gcs_items))
                 .with_columns(apply_bounds(pl.col("value"), (3, 15)).alias("value"), pl.lit("vital_gcs").alias("variable"))
                 .drop_nulls("value"))
            inputs.append(windowed_join(g.select("hadm_raw", "variable", "value", "time_min"), w, "hadm_raw",
                                        pl.col("icu_in_min"), pl.col("icu_out_min")))

        # Supports (ventilation, vasopressors, RRT) from ICU charting of the hospitalization (IV infusions only)
        sup = self.src["support_items"]
        for tname in ("procedureevents", "inputevents"):
            kinds = {k: [str(i) for i in s["items"]] for k, s in sup.items() if s["table"] == tname}
            tr_items = [str(i) for i in self.src["transfusion_items"]["items"]] \
                if self.src["transfusion_items"]["table"] == tname else []
            all_items = sorted({i for v in kinds.values() for i in v} | set(tr_items))
            if not all_items:
                continue
            ev = self.table(tname, filters={"hadm_id": hadm, "itemid": all_items})
            ev = ev.with_columns(as_id("hadm_id").alias("hadm_raw"), as_str("itemid").alias("it"),
                                 dt_minutes(ev, "starttime").alias("time_min"), dt_minutes(ev, "endtime").alias("end_min"))
            for kind, its in kinds.items():
                s = ev.filter(pl.col("it").is_in(its)).select(
                    "hadm_raw", pl.lit(f"support_{kind}").alias("variable"), pl.lit(1.0).alias("value"), "time_min", "end_min")
                inputs.append(windowed_join(s, w, "hadm_raw", admit, hi))
            if tr_items:
                t = ev.filter(pl.col("it").is_in(tr_items)).select("hadm_raw", "time_min")
                t = windowed_join(t, w, "hadm_raw", admit, hi)
                outcomes.append(t.select("stay_id", pl.lit("transfusion_rbc").alias("component"), "time_min",
                                         pl.lit("minute").alias("precision")))

        # Administered medication (eMAR) → ATC by name (inputs and antibiotic course only; no support_vaso)
        if self.has_table("emar"):
            em = self.table("emar", filters={"hadm_id": hadm})
            em = em.filter(as_str("event_txt").is_in(self.src["emar_administered_events"]))
            em = em.with_columns(as_id("hadm_id").alias("hadm_raw"), dt_minutes(em, "charttime").alias("time_min"),
                                 as_str("medication").alias("drug"))
            inputs.append(_drug_rows(em, self.h, w, "hadm_raw", pl.col("icu_in_min") - 1440, hi))
            rx = re.compile(self.src["transfusion_drug_regex"])
            tn = [n for n in em["drug"].drop_nulls().unique().to_list() if rx.search(n)]
            t = windowed_join(em.filter(pl.col("drug").is_in(tn)).select("hadm_raw", "time_min"), w, "hadm_raw", admit, hi)
            outcomes.append(t.select("stay_id", pl.lit("transfusion_rbc").alias("component"), "time_min",
                                     pl.lit("minute").alias("precision")))
        else:
            self.log.warning("MIMIC without eMAR: antibiotics and active medication are not observed")

        # Outcomes: ICU episodes of the hospitalization (true ICU careunits) and surgeries (day precision)
        eps = self._icu_episodes(hadm)
        idx = w.select("stay_id", "hadm_raw")
        outcomes.append(eps.select("hadm_raw", pl.col("icu_in_min").alias("time_min")).join(idx, on="hadm_raw")
                        .select("stay_id", pl.lit("icu_admission").alias("component"), "time_min",
                                pl.lit("minute").alias("precision")))
        orp = self._or_procedures().filter(pl.col("hadm_raw").is_in(hadm))
        outcomes.append(orp.select("hadm_raw", pl.col("day_min").alias("time_min")).unique().join(idx, on="hadm_raw")
                        .select("stay_id", pl.lit("surgery").alias("component"), "time_min",
                                pl.lit("day").alias("precision")))

        inp = pl.concat([conform(f, INPUTS_SCHEMA) for f in inputs if f.height], how="vertical")
        out = pl.concat([conform(f, OUTCOMES_SCHEMA) for f in outcomes if f.height], how="vertical") \
            if any(f.height for f in outcomes) else conform(pl.DataFrame(), OUTCOMES_SCHEMA)
        return inp, out


def _drug_rows(df: pl.DataFrame, harmonization: dict, w: pl.DataFrame, key: str, lo: pl.Expr, hi: pl.Expr,
               with_end: bool = False) -> pl.DataFrame:
    """Drug names → atc:<code> rows. Shared with eICU.

    Local / topical / ophthalmic / otic / nasal / inhaled products and flushes are excluded before mapping, and no
    support_vaso is derived from names: vasopressor support comes only from IV infusion sources (AUDIT D4)."""
    med = harmonization["medications"]
    excl = re.compile(med["name_exclude_regex"]) if med.get("name_exclude_regex") else None
    names = df["drug"].drop_nulls().unique()
    if excl is not None:
        names = pl.Series([n for n in names.to_list() if not excl.search(str(n))], dtype=pl.Utf8)
    mapping = regex_map_unique(names, med["name_to_atc"], "atc")
    if not mapping:
        return conform(pl.DataFrame(), INPUTS_SCHEMA)
    lut = pl.DataFrame({"drug": list(mapping), "atc": list(mapping.values())}).explode("atc")
    cols = [key, "time_min", "drug"] + (["end_min"] if with_end else [])
    j = df.select(cols).join(lut, on="drug", how="inner")
    end = pl.col("end_min") if with_end else pl.lit(None, pl.Float64)
    rows = j.select(key, (pl.lit("atc:") + pl.col("atc")).alias("variable"), pl.lit(1.0).alias("value"), "time_min",
                    end.alias("end_min"))
    return windowed_join(rows, w, key, lo, hi)
