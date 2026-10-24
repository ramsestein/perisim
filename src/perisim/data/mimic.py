"""MIMIC-IV (BIDMC) → common format. Clock: minutes since 1970 of the shifted dates.

Particularities (§2.2) [VERIFY against the mounted version]:
- There are no ward vital signs (chartevents is ICU only): mild events on the ward are observed via labs and eMAR.
- Surgery is identified by timestamped operating-room evidence: an OR/PACU transfer in hosp.transfers (AUDIT D1/D8).
  procedures_icd (date only) is a fallback for hospitalizations without any OR/PACU transfer, so reoperation has
  minute precision when the hospitalization has OR/PACU transfers and day precision otherwise.
- diagnoses_icd has no timestamp (billing codes at discharge): NOT used as input (leakage, §4.4).
- icustays merges contiguous ICU and step-down careunits: ICU episodes are rebuilt from hosp.transfers with
  true ICU careunits only (AUDIT D3), so t0 is the exit from the last true ICU careunit and a step-down
  transfer is neither an index stay nor a readmission.
- eMAR does not cover every hospitalization: the first eMAR record of each hospitalization is emitted as a
  `src:medication` row, so that medication is unobservable (not "absent") where eMAR does not exist (D6).
"""

from __future__ import annotations

import re

import polars as pl

from perisim.config import REPO_ROOT
from perisim.data.base import BaseExtractor
from perisim.data.common import (
    INPUTS_SCHEMA, OUTCOMES_SCHEMA, STAYS_SCHEMA, apply_bounds, as_float, as_id, as_str, conform, date_minutes,
    dt_minutes, harmonized_category, map_measurements, regex_map_unique, variables_for, windowed_join,
)
from perisim.data.inspire import _common_flags

MEDICATION_MARKER = "src:medication"
CHRONIC_RRT_MARKER = "marker:chronic_rrt"


class MimicExtractor(BaseExtractor):
    cohort = "mimic"
    prefix = "MIMIC"
    _prc: pl.DataFrame | None = None

    # ------------------------------------------------------------------ surgical rules
    def _procedures(self) -> pl.DataFrame:
        if self._prc is None:
            prc = self.table("procedures_icd")
            self._prc = prc.with_columns(
                as_id("hadm_id").alias("hadm_raw"),
                as_id("subject_id").alias("subject_raw"),
                date_minutes(prc, "chartdate").alias("day_min"),
                as_str("icd_code").str.to_uppercase().str.replace_all(r"[^0-9A-Z]", "").alias("code"),
                as_str("icd_version").alias("ver"),
                as_float("seq_num").alias("seq"),
            )
        return self._prc

    def _hcup_major(self) -> dict[str, set[str]] | None:
        """HCUP Procedure Classes (class 4 = major therapeutic) if the files are configured; None otherwise."""
        files = self.ccfg.get("hcup_pclass_files") or {}
        out = {}
        for ver, key in (("9", "icd9"), ("10", "icd10")):
            f = files.get(key)
            if not f:
                return None
            df = pl.read_csv(REPO_ROOT / f, infer_schema_length=0)
            code, cls = df.columns[0], df.columns[1]
            out[ver] = set(df.filter(pl.col(cls).str.strip_chars() == "4")[code].str.replace_all(r"[^0-9A-Za-z]", "")
                           .str.to_uppercase().to_list())
        return out

    def _or_procedures(self) -> pl.DataFrame:
        """"Major therapeutic" procedures (ICD-10-PCS / ICD-9) with their day, per hospitalization: HCUP procedure
        classes when provided, otherwise a declared approximation (operative approaches / chapters minus bedside
        and diagnostic procedures, AUDIT D1)."""
        prc = self._procedures()
        code = pl.col("code")
        hcup = self._hcup_major()
        if hcup is not None:
            keep = ((pl.col("ver") == "9") & code.is_in(sorted(hcup["9"]))) | \
                   ((pl.col("ver") == "10") & code.is_in(sorted(hcup["10"])))
        else:
            c = self.ccfg
            icd10 = ((pl.col("ver") == "10") & (code.str.slice(0, 1) == "0")
                     & code.str.slice(4, 1).is_in(c["or_icd10_approaches"])
                     & ~code.str.slice(2, 1).is_in(c["or_icd10_exclude_root_operations"]))
            lo, hi = c["or_icd9_range"]
            chap = code.str.slice(0, 2).cast(pl.Int32, strict=False)
            excl = pl.any_horizontal([code.str.starts_with(p) for p in c["or_icd9_exclude_prefixes"]])
            keep = (icd10 | ((pl.col("ver") == "9") & chap.is_between(lo, hi) & ~excl))
        return prc.filter(keep).select("hadm_raw", "day_min", "code", "ver", "seq")

    def _transfers(self, hadm: list[str] | None = None) -> pl.DataFrame:
        filters = {"hadm_id": hadm} if hadm is not None else None
        tr = self.table("transfers", filters=filters)
        return (tr.with_columns(as_id("transfer_id").alias("tid"), as_id("hadm_id").alias("hadm_raw"),
                                as_id("subject_id").alias("subject_raw"), dt_minutes(tr, "intime").alias("t_in"),
                                dt_minutes(tr, "outtime").alias("t_out"), as_str("careunit").alias("unit"))
                .filter(pl.col("hadm_raw").is_not_null() & pl.col("t_in").is_not_null()))

    def _or_transfers(self, hadm: list[str] | None = None) -> pl.DataFrame:
        """Operating-room evidence: transfers into an OR/PACU careunit (hadm_raw, t_in, t_out)."""
        if not self.has_table("transfers"):
            return pl.DataFrame(schema={"hadm_raw": pl.Utf8, "t_in": pl.Float64, "t_out": pl.Float64})
        units = self.ccfg.get("or_careunits") or []
        return (self._transfers(hadm).filter(pl.col("unit").is_in(units))
                .select("hadm_raw", "t_in", pl.coalesce("t_out", "t_in").alias("t_out")))

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
        units = self.ccfg["icu_careunits"]
        gap = float(self.ccfg.get("icu_episode_merge_gap_min", 60))
        tr = (self._transfers(hadm).filter(pl.col("unit").is_in(units) & pl.col("t_out").is_not_null())
              .sort(["hadm_raw", "t_in"]))
        tr = tr.with_columns(pl.col("t_out").cum_max().shift(1).over("hadm_raw").alias("_prev"))
        tr = tr.with_columns((pl.col("_prev").is_null() | (pl.col("t_in") > pl.col("_prev") + gap))
                             .cast(pl.Int32).cum_sum().over("hadm_raw").alias("_ep"))
        return (tr.group_by(["hadm_raw", "_ep"], maintain_order=True)
                .agg(pl.col("tid").first().alias("unit_raw"), pl.col("subject_raw").first(),
                     pl.col("t_in").min().alias("icu_in_min"), pl.col("t_out").max().alias("icu_out_min"))
                .drop("_ep"))

    def _return_after_exit(self, eps: pl.DataFrame) -> pl.DataFrame:
        """C5: minutes from each ICU exit to the next OR/PACU (or other return careunit) or ICU entry."""
        if not self.has_table("transfers"):
            return eps.select("unit_raw", pl.lit(None, pl.Float64).alias("return_within_min"))
        units = set(self.ccfg.get("return_careunits") or []) | set(self.ccfg["icu_careunits"])
        tr = self._transfers().filter(pl.col("unit").is_in(sorted(units))).select("hadm_raw", "t_in")
        return (eps.select("unit_raw", "hadm_raw", "icu_out_min").join(tr, on="hadm_raw", how="inner")
                .filter(pl.col("t_in") >= pl.col("icu_out_min"))
                .group_by("unit_raw").agg((pl.col("t_in") - pl.col("icu_out_min")).min().alias("return_within_min")))

    def _limitation(self) -> pl.DataFrame:
        """D12: first timestamped limitation-of-treatment record (Code Status) per hospitalization."""
        lim = self.src.get("limitation") or {}
        if not lim or not self.has_table("code_status"):
            return pl.DataFrame(schema={"hadm_raw": pl.Utf8, "limitation_min": pl.Float64})
        cs = self.table("code_status", filters={"itemid": [str(i) for i in lim["items"]]})
        rx = re.compile(lim["value_regex"])
        vals = [v for v in cs["value"].drop_nulls().unique().to_list() if rx.search(str(v))]
        return (cs.filter(pl.col("value").is_in(vals))
                .with_columns(as_id("hadm_id").alias("hadm_raw"), dt_minutes(cs, "charttime").alias("t"))
                .group_by("hadm_raw").agg(pl.col("t").min().alias("limitation_min")))

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
        pat = self.table("patients", extra=self.optional_column_names("patients"))
        pat = pat.with_columns(as_id("subject_id").alias("subject_raw"), as_float("anchor_age").alias("anchor_age"),
                               as_float("anchor_year").alias("anchor_year"),
                               (as_str("anchor_year_group") if "anchor_year_group" in pat.columns
                                else pl.lit(None, pl.Utf8)).alias("year_group"))
        df = (icu.join(adm.drop("subject_id"), on="hadm_raw", how="inner")
              .join(pat.select("subject_raw", "gender", "anchor_age", "anchor_year", "year_group"), on="subject_raw",
                    how="left"))

        # Service in effect at the ICU admission (descriptive "service_only" count and the v0 rule)
        srv = self.table("services")
        srv = srv.with_columns(as_id("hadm_id").alias("hadm_raw"), dt_minutes(srv, "transfertime").alias("t_srv"),
                               as_str("curr_service").str.to_uppercase().alias("service"))
        lb = float(self.ccfg.get("service_lookback_min", 60))
        last_srv = (df.select("unit_raw", "hadm_raw", "icu_in_min")
                    .join(srv.select("hadm_raw", "t_srv", "service"), on="hadm_raw", how="inner")
                    .filter(pl.col("t_srv") <= pl.col("icu_in_min") + lb)
                    .sort("t_srv").group_by("unit_raw").agg(pl.col("service").last()))
        df = df.join(last_srv, on="unit_raw", how="left")

        # AUDIT D1/D8: OR/PACU transfer ending within or_window_hours before the ICU admission
        win = float(self.ccfg.get("or_window_hours", 24)) * 60
        tol = float(self.ccfg.get("or_tolerance_min", 60))
        ort = self._or_transfers()
        or_ev = (df.select("unit_raw", "hadm_raw", "icu_in_min").join(ort, on="hadm_raw", how="inner")
                 .filter((pl.col("t_out") >= pl.col("icu_in_min") - win) & (pl.col("t_out") <= pl.col("icu_in_min") + tol)
                         & (pl.col("t_in") <= pl.col("icu_in_min") + tol))
                 .group_by("unit_raw").agg(pl.len().alias("n_or_transfer")))
        df = df.join(or_ev, on="unit_raw", how="left")
        hadm_or = ort["hadm_raw"].unique().to_list()

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
        surgical = self.ccfg["surgical_services"]
        by_srv = pl.col("service").is_in(surgical).fill_null(False)
        by_or = pl.col("n_or_transfer").fill_null(0) > 0
        # the code fallback applies only to hospitalizations WITHOUT any OR/PACU transfer
        by_prc = (pl.col("n_or").fill_null(0) > 0) & ~pl.col("hadm_raw").is_in(hadm_or)
        evidence = (pl.when(by_or).then(pl.lit("or_transfer")).when(by_prc).then(pl.lit("procedure"))
                    .when(by_srv).then(pl.lit("service_only")).otherwise(pl.lit("none")))
        rule = self.ccfg.get("postop_rule", "or_evidence")
        if rule == "service_or_procedure":            # v0 rule, kept only for sensitivity analyses
            postop = by_srv | (pl.col("n_or").fill_null(0) > 0)
        else:
            postop = by_or | by_prc
        df = df.join(self._return_after_exit(df), on="unit_raw", how="left") \
            .join(self._limitation(), on="hadm_raw", how="left")
        lim_ok = pl.col("limitation_min") >= pl.col("hosp_admit_min").fill_null(-float("inf"))

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
            evidence.alias("postop_evidence"),
            pl.when(lim_ok).then(pl.col("limitation_min")).otherwise(None).alias("limitation_min"),
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

        # Labs (per subject: includes prior outpatient labs, with their date). Blood gases: arterial only (D13).
        lab_specs = variables_for(self.h, "mimic", "labevents")
        bgs = self.src.get("blood_gas_specimen") or {}
        items = sorted({i for s in lab_specs for i in s["items"]} | ({str(bgs["item"])} if bgs else set()))
        lab = self.table("labevents", filters={"subject_id": subj, "itemid": items},
                         extra=self.optional_column_names("labevents"))
        lab = lab.with_columns(as_id("subject_id").alias("subject_raw"), dt_minutes(lab, "charttime").alias("time_min"))
        art_only = [s for s in lab_specs if s.get("arterial_only")]
        if art_only and bgs and {"specimen_id", "value"} <= set(lab.columns):
            rx = re.compile(bgs["arterial_regex"])
            typ = lab.filter(as_str("itemid") == str(bgs["item"]))
            art_vals = [v for v in typ["value"].drop_nulls().unique().to_list() if rx.search(str(v))]
            art_ids = typ.filter(pl.col("value").is_in(art_vals))["specimen_id"].unique().to_list()
            art_items = sorted({i for s in art_only for i in s["items"]})
            lab = lab.filter(~as_str("itemid").is_in(art_items) | pl.col("specimen_id").is_in(art_ids))
        elif art_only:
            self.log.warning("MIMIC labevents without specimen_id/value: blood gases NOT restricted to arterial (D13)")
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
        with_emar: set[str] = set()
        if self.has_table("emar"):
            em_all = self.table("emar", filters={"hadm_id": hadm})
            em_all = em_all.with_columns(as_id("hadm_id").alias("hadm_raw"), dt_minutes(em_all, "charttime").alias("time_min"))
            with_emar = set(em_all["hadm_raw"].drop_nulls().unique().to_list())
            # D6: coverage marker = the first eMAR record of the hospitalization (any event)
            first = em_all.drop_nulls("time_min").group_by("hadm_raw").agg(pl.col("time_min").min())
            inputs.append(windowed_join(first.select("hadm_raw", pl.lit(MEDICATION_MARKER).alias("variable"),
                                                     pl.lit(1.0).alias("value"), "time_min"),
                                        w, "hadm_raw", admit - 30 * 1440, hi))
            em = em_all.filter(as_str("event_txt").is_in(self.src["emar_administered_events"]))
            em = em.with_columns(as_str("medication").alias("drug"))
            inputs.append(_drug_rows(em, self.h, w, "hadm_raw", pl.col("icu_in_min") - 1440, hi))
            rx = re.compile(self.src["transfusion_drug_regex"])
            tn = [n for n in em["drug"].drop_nulls().unique().to_list() if rx.search(n)]
            t = windowed_join(em.filter(pl.col("drug").is_in(tn)).select("hadm_raw", "time_min"), w, "hadm_raw", admit, hi)
            outcomes.append(t.select("stay_id", pl.lit("transfusion_rbc").alias("component"), "time_min",
                                     pl.lit("minute").alias("precision")))
        else:
            self.log.warning("MIMIC without eMAR: medication comes from prescriptions only (D6)")
        # Prescriptions (orders, start–stop) for the hospitalizations without any eMAR record (lead, 2026-10-24): the
        # same name → ATC mapping, the order interval kept as end_min (the antibiotic course counts the days it covers)
        no_emar = [h for h in hadm if h not in with_emar]
        if no_emar and self.has_table("prescriptions"):
            pr = self.table("prescriptions", filters={"hadm_id": no_emar}, extra=self.optional_column_names("prescriptions"))
            pr = pr.with_columns(as_id("hadm_id").alias("hadm_raw"), dt_minutes(pr, "starttime").alias("time_min"),
                                 dt_minutes(pr, "stoptime").alias("end_min"), as_str("drug").alias("drug"))
            pr = pr.drop_nulls("time_min")
            first = pr.group_by("hadm_raw").agg(pl.col("time_min").min())     # D6 coverage marker (any order)
            inputs.append(windowed_join(first.select("hadm_raw", pl.lit(MEDICATION_MARKER).alias("variable"),
                                                     pl.lit(1.0).alias("value"), "time_min"),
                                        w, "hadm_raw", admit - 30 * 1440, hi))
            route_rx = self.src.get("prescriptions_route_exclude_regex")
            if route_rx and "route" in pr.columns:   # systemic routes only (AUDIT D4)
                pr = pr.filter(~as_str("route").str.contains(route_rx).fill_null(False))
            # an order running into the window counts from the window start: ongoing therapy, never a new course
            lo = w.group_by("hadm_raw").agg((pl.col("icu_in_min") - 1440).min().alias("_lo"))
            pr = (pr.join(lo, on="hadm_raw").filter(pl.coalesce("end_min", "time_min") >= pl.col("_lo"))
                  .with_columns(pl.max_horizontal("time_min", "_lo").alias("time_min")).drop("_lo"))
            inputs.append(_drug_rows(pr, self.h, w, "hadm_raw", pl.col("icu_in_min") - 1440, hi, with_end=True))

        # D5: dialysis procedures of the subject dated before the ICU admission day (chronic / prior RRT)
        rrt_rx = self.src.get("chronic_rrt_procedure_regex")
        if rrt_rx:
            prc = self._procedures().filter(pl.col("subject_raw").is_in(subj) & pl.col("code").str.contains(rrt_rx))
            mk = prc.select("subject_raw", pl.lit(CHRONIC_RRT_MARKER).alias("variable"), pl.lit(1.0).alias("value"),
                            pl.col("day_min").alias("time_min"))
            icu_day = (pl.col("icu_in_min") / 1440).floor() * 1440
            inputs.append(windowed_join(mk, w, "subject_raw", admit - lookback, icu_day - 1440))

        # Outcomes: ICU episodes of the hospitalization (true ICU careunits) and surgeries
        eps = self._icu_episodes(hadm)
        idx = w.select("stay_id", "hadm_raw")
        outcomes.append(eps.select("hadm_raw", pl.col("icu_in_min").alias("time_min")).join(idx, on="hadm_raw")
                        .select("stay_id", pl.lit("icu_admission").alias("component"), "time_min",
                                pl.lit("minute").alias("precision")))
        # AUDIT D1: reoperation = a later OR/PACU transfer (minute precision); procedure codes (day precision) only
        # for hospitalizations without any OR/PACU transfer
        ort = self._or_transfers(hadm)
        outcomes.append(ort.select("hadm_raw", pl.col("t_in").alias("time_min")).unique().join(idx, on="hadm_raw")
                        .select("stay_id", pl.lit("surgery").alias("component"), "time_min",
                                pl.lit("minute").alias("precision")))
        no_or = [h for h in hadm if h not in set(ort["hadm_raw"].to_list())]
        orp = self._or_procedures().filter(pl.col("hadm_raw").is_in(no_or))
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
