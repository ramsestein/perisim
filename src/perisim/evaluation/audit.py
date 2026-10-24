"""Per-cohort completeness audit (§7): first technical deliverable and stop gate.

If any T1 component has prevalence < 0.5 % or missingness > 30 % in any cohort, the pipeline stops and
records it in OPEN_QUESTIONS (in the workdir) until `gates.audit_acknowledged` is confirmed.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from perisim.evaluation import privacy as PV
from perisim.evaluation.html import box, page, table

PREV_MIN = 0.005
MISS_MAX = 0.30


def completeness(cohort: str, flow: list[dict], stays: pl.DataFrame, inputs: pl.DataFrame, targets: pl.DataFrame,
                 X: pl.DataFrame, specs: list, cfg, postop_evidence: dict | None = None) -> tuple[dict, str]:
    min_n = int(cfg.experiment["evaluation"]["small_cell_threshold"])
    tcfg = cfg.targets
    observable = list(tcfg["t1"]["observable"].get(cohort, []))
    n = targets.height
    # Everything in `rep` is exportable: counts go through the small-cell rules of evaluation/privacy.py (A6)
    rep: dict = {"cohort": cohort, "n_index": PV.cell(n, min_n), "flow": PV.flow(flow, min_n), "issues": []}

    # --- inputs: missingness, distribution and units
    units = {}
    for vname, v in cfg.harmonization.get("variables", {}).items():
        units[vname.replace("lab_", "").replace("vital_", "")] = v.get("units")
    inp_rows = []
    for s in specs:
        col = X[s.name] if s.name in X.columns else pl.Series([None] * X.height)
        n_miss = int(col.is_null().sum()) if X.height else 0
        row = {"feature": s.name, "type": s.type, "block": s.block, "missing_pct": PV.share(n_miss, X.height, min_n)}
        base = s.name.split("_", 1)[1] if "_" in s.name else s.name
        row["units"] = units.get(base)
        if s.type in ("continuous", "ordinal"):
            v = col.drop_nulls().cast(pl.Float64)
            if v.len() >= min_n:
                q = np.quantile(v.to_numpy(), [0.05, 0.25, 0.5, 0.75, 0.95])
                row.update({f"p{int(p * 100)}": float(x) for p, x in zip([0.05, 0.25, 0.5, 0.75, 0.95], q)})
        elif s.type == "binary":
            k = int(col.fill_null(0).cast(pl.Float64).sum())
            row["n_pos"] = PV.cell(k, min_n, total=X.height)
            row["prev_pct"] = PV.share(k, X.height, min_n, digits=2)
        else:
            vc = col.drop_nulls().value_counts(sort=True).head(8)
            row["top"] = "; ".join(f"{r[s.name]} ({r['count']})" for r in vc.to_dicts()
                                   if r["count"] >= min_n and X.height - r["count"] >= min_n)
        inp_rows.append(row)
    rep["inputs"] = inp_rows

    # --- ward measurement frequency after t0 (feasibility of the mild category)
    fu = float(tcfg["window"]["followup_days"]) * 1440
    s = stays.select("stay_id", pl.col("icu_out_min").alias("t0"),
                     pl.min_horizontal(pl.col("icu_out_min") + fu, pl.col("hosp_disch_min").fill_null(float("inf"))).alias("we"))
    post = inputs.join(s, on="stay_id").filter((pl.col("time_min") > pl.col("t0")) & (pl.col("time_min") <= pl.col("we")))
    los_days = s.with_columns(((pl.col("we") - pl.col("t0")) / 1440).clip(1 / 24, None).alias("d"))
    freq = []
    for label, cond in [("creatinine", pl.col("variable") == "lab_creatinine"),
                        ("hemoglobin", pl.col("variable") == "lab_hb"),
                        ("J01 antibiotic", pl.col("variable").str.starts_with("atc:J01")),
                        ("ward SpO2", pl.col("variable") == "vital_spo2")]:
        c = post.filter(cond).group_by("stay_id").len()
        c = los_days.join(c, on="stay_id", how="left").with_columns(pl.col("len").fill_null(0))
        with_any = int((c["len"] > 0).sum())
        per_day = (c.filter(pl.col("len") > 0)["len"] / c.filter(pl.col("len") > 0)["d"]).to_numpy()
        freq.append({"measure": label, "stays_with_any": PV.cell(with_any, min_n, total=n),
                     "pct_stays": PV.share(with_any, n, min_n),
                     "median_per_day": float(np.median(per_day)) if per_day.size >= min_n else None})
    rep["ward_frequency"] = freq

    # --- component prevalences, classes and length of stay
    comp_rows = []
    # Missingness is only ASSESSABLE where an observability proxy exists (AKI: baseline + follow-up creatinine;
    # antibiotics: any medication record after t0). Structural events are reported as "not assessable", not 0 %.
    tracked = bool("med_coverage_tracked" in targets.columns and targets["med_coverage_tracked"].any())
    miss_map = {"aki_3": "aki_observable", "aki_12": "aki_observable",
                "antibiotic_course": "med_source_observed" if tracked else "med_observable",
                "hypoxemia_ward": None, "transfusion": "med_source_observed" if tracked else None}
    t1_def = set(tcfg["t1"]["severe_components"]) | set(tcfg["t1"]["mild_components"])
    for comp in ["icu_readmission", "death", "reoperation", "vent_start", "vaso_start", "rrt_start", "aki_3", "aki_12",
                 "antibiotic_course", "transfusion", "hypoxemia_ward"]:
        obs = comp in observable
        k = int(targets[comp].fill_null(0).sum()) if obs and comp in targets.columns else None
        prev = (k / n) if (k is not None and n) else None
        mcol = miss_map.get(comp)
        assessable = bool(obs and mcol and mcol in targets.columns and n)
        n_miss = int((~targets[mcol].fill_null(False)).sum()) if assessable else None
        miss = (n_miss / n) if assessable else None
        row = {"component": comp, "observable": obs, "n": PV.cell(k, min_n, total=n) if k is not None else None,
               "prev_pct": PV.share(k, n, min_n, digits=2) if k is not None else None,
               "missing_pct": PV.share(n_miss, n, min_n) if assessable else ("not assessable" if obs else None)}
        comp_rows.append(row)
        if obs and comp in t1_def:
            # issue texts are exported: they state the rule, never the exact (small) value
            if prev is not None and prev < PREV_MIN:
                rep["issues"].append(f"{cohort}: component {comp} with prevalence < 0.5 %")
            if miss is not None and miss > MISS_MAX:
                rep["issues"].append(f"{cohort}: component {comp} with missingness > 30 %")
    rep["components"] = comp_rows
    if "t1" in targets.columns and targets["t1"].drop_nulls().len():
        vc = targets["t1"].drop_nulls().value_counts().sort("t1")
        cc = PV.counts({["none", "mild", "severe"][int(r["t1"])]: int(r["count"]) for r in vc.to_dicts()}, min_n)
        rep["t1_classes"] = [{"class": k, "n": v} for k, v in cc.items()]
    kg = int(targets["t1_severe"].fill_null(0).sum())
    rep["t1_severe"] = {"n": PV.cell(kg, min_n, total=n), "pct": PV.share(kg, n, min_n, digits=2)}
    los = targets["los_post_icu_days"].drop_nulls().to_numpy()
    if los.size >= min_n:
        rep["t2"] = {"n_survivors": int(los.size), "quantiles": dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                                                         np.quantile(los, [.1, .25, .5, .75, .9]).tolist())),
                     "tertiles_descriptive": np.quantile(los, [1 / 3, 2 / 3]).tolist(),
                     "truncated_pct": PV.share(int(targets.filter(pl.col("los_post_icu_days").is_not_null())
                                                   ["los_truncated"].fill_null(False).sum()), int(los.size), min_n, 2)}
    if cohort == "inspire" and "icu_readmission_proxy" in targets.columns:
        e = targets["icu_readmission_icu_entry"].fill_null(0) == 1
        pr = targets["icu_readmission_proxy"].fill_null(0) == 1
        k = int(targets["icu_readmission"].fill_null(0).sum())
        rep["icu_readmission_inspire"] = {
            "n_detected": PV.cell(k, min_n, total=n), "pct": PV.share(k, n, min_n, digits=2),
            "by_source": PV.counts({"ICU entry after another surgery only": int((e & ~pr).sum()),
                                    "new organ-support run only (proxy)": int((~e & pr).sum()),
                                    "both": int((e & pr).sum())}, min_n),
            "note": "SPECS D1 proxy (AUDIT D10): a new ICU entry after another surgery OR a new invasive-ventilation / "
                    "RRT / ECMO / IABP run after t0 in ward_vitals. Validate this prevalence (vs MIMIC-IV / eICU "
                    "readmission rates) before acknowledging the audit."}
    elif cohort == "inspire":
        k = int(targets["icu_readmission"].fill_null(0).sum())
        rep["icu_readmission_inspire"] = {"n_detected": PV.cell(k, min_n, total=n), "note": "No proxy configured (D1)."}
    if "chronic_rrt" in targets.columns:
        k = int(targets["chronic_rrt"].fill_null(False).sum())
        rep["chronic_rrt"] = {"n": PV.cell(k, min_n, total=n), "pct": PV.share(k, n, min_n, digits=2),
                              "note": "RRT before the index surgery (D5): AKI and new RRT are not applicable."}
    if tracked:
        cov = targets.select("stay_id", "med_source_observed").join(
            stays.select("stay_id", *(["year_group"] if "year_group" in stays.columns else [])), on="stay_id", how="left")
        rows_cov = [{"period": "all", "n": PV.cell(n, min_n), "covered_pct":
                     PV.share(int(cov["med_source_observed"].sum()), n, min_n)}]
        if "year_group" in cov.columns and cov["year_group"].drop_nulls().len():
            for (g,), sub in cov.filter(pl.col("year_group").is_not_null()).group_by(["year_group"]):
                rows_cov.append({"period": str(g), "n": PV.cell(sub.height, min_n),
                                 "covered_pct": PV.share(int(sub["med_source_observed"].sum()), sub.height, min_n)})
        rep["medication_coverage"] = {"rows": sorted(rows_cov, key=lambda r: (r["period"] != "all", r["period"])),
                                      "note": "Stays whose hospitalization has no medication record are unobservable "
                                              "for medication inputs and the mild medication components (D6); their "
                                              "T1/T3 are undefined (coverage-based exclusion)."}
    if postop_evidence:
        tot = sum(int(v) for v in postop_evidence.values())
        rep["postop_evidence"] = {"counts": PV.counts({k: int(v) for k, v in sorted(postop_evidence.items())}, min_n),
                                  "total": PV.cell(tot, min_n),
                                  "note": "Candidate ICU episodes by surgical evidence (AUDIT D1/D8): or_transfer = "
                                          "OR/PACU transfer within 24 h before the ICU admission; procedure = major "
                                          "therapeutic code (hospitalizations without OR/PACU transfers); "
                                          "service_only = surgical service without either (NOT postoperative; "
                                          "sensitivity cohort)."}
    return rep, render(rep, min_n)


def render(rep: dict, min_n: int) -> str:
    c = rep["cohort"]
    sec = []
    if rep["issues"]:
        sec.append(box("<b>Gate §7: STOP.</b><br>" + "<br>".join(rep["issues"]) +
                       "<br>Recorded in docs/OPEN_QUESTIONS (workdir). To continue after reviewing it: "
                       "<code>PERISIM_ACK_AUDIT=true bash run_pod.sh</code>", "bad"))
    else:
        sec.append(box("No prevalence (< 0.5 %) or missingness (> 30 %) issues in the T1 components.", "ok"))
    sec.append("<h2>Flow diagram</h2>" + table(["Step", "Stays", "Patients", "Excluded"],
                                               [[f["step"], f["n_stays"], f["n_subjects"], f["excluded"]]
                                                for f in rep["flow"]])
               + ("<p class='muted'>Counts rounded to the nearest 10 from the first step that excludes &lt; 10 "
                  "stays (secondary suppression).</p>" if any(f.get("rounded") for f in rep["flow"]) else ""))
    sec.append("<h2>T1 components, classes and length of stay</h2>" + table(
        ["Component", "Observable", "n", "Prevalence %", "Missing %"],
        [[r["component"], "yes" if r["observable"] else "no (by design)", r["n"], r["prev_pct"], r["missing_pct"]]
         for r in rep["components"]], nd=2))
    if rep.get("t1_classes"):
        sec.append(table(["T1 class", "n"], [[r["class"], r["n"]] for r in rep["t1_classes"]]))
    sec.append(table(["T1 severe / not severe", "n", "%"], [["severe", rep["t1_severe"]["n"], rep["t1_severe"]["pct"]]], nd=2))
    if rep.get("t2"):
        t2 = rep["t2"]
        sec.append(table(["Post-ICU length of stay (survivors)", "value"],
                         [["n", t2["n_survivors"]]] + [[k, v] for k, v in t2["quantiles"].items()] +
                         [["tertiles (descriptive; for modeling they are computed within each fold)", " / ".join(f"{x:.2f}" for x in t2["tertiles_descriptive"])],
                          ["% truncated at 30 d", t2["truncated_pct"]]], nd=2))
    sec.append("<h2>Measurement frequency after t0 (feasibility of the mild category)</h2>" + table(
        ["Measure", "Stays with ≥1", "% of stays", "Median per day"],
        [[r["measure"], r["stays_with_any"], r["pct_stays"], r["median_per_day"]] for r in rep["ward_frequency"]], nd=2))
    if rep.get("icu_readmission_inspire"):
        r = rep["icu_readmission_inspire"]
        sec.append("<h2>Feasibility of ICU readmission (INSPIRE, D1 / D10)</h2>" +
                   box(f"Detected: {r['n_detected']} ({r.get('pct')} %). {r['note']}", "warn"))
        if r.get("by_source"):
            sec.append(table(["Source", "n"], [[k, v] for k, v in r["by_source"].items()]))
    if rep.get("chronic_rrt"):
        r = rep["chronic_rrt"]
        sec.append("<h2>Chronic / prior RRT (D5)</h2>" + box(f"Stays: {r['n']} ({r['pct']} %). {r['note']}"))
    if rep.get("medication_coverage"):
        r = rep["medication_coverage"]
        sec.append("<h2>Medication source coverage (D6)</h2>" + box(r["note"], "warn") +
                   table(["Period", "Stays", "% covered"], [[x["period"], x["n"], x["covered_pct"]] for x in r["rows"]],
                         nd=1))
    if rep.get("postop_evidence"):
        r = rep["postop_evidence"]
        sec.append("<h2>Surgical evidence of the candidate ICU episodes (D1 / D8)</h2>" + box(r["note"]) +
                   table(["Evidence", "Episodes"], [[k, v] for k, v in r["counts"].items()]))
    rows = []
    for r in rep["inputs"]:
        dist = ""
        if "p50" in r:
            dist = f"p5 {r['p5']:.2f} · p25 {r['p25']:.2f} · p50 {r['p50']:.2f} · p75 {r['p75']:.2f} · p95 {r['p95']:.2f}"
        elif "prev_pct" in r:
            dist = f"n={r['n_pos']} ({r['prev_pct']} %)" if r.get("prev_pct") is not None else f"n={r['n_pos']}"
        elif r.get("top"):
            dist = r["top"]
        rows.append([r["feature"], r["type"], r["block"], r["missing_pct"], r.get("units") or "", dist])
    sec.append("<h2>Inputs: missingness, distribution and units</h2>" + table(
        ["Variable", "Type", "Block", "Missing %", "Units", "Distribution"], rows, nd=1))
    return page(f"Completeness audit — {c}", sec,
                f"PERISIM · §7 · cells with n < {min_n} suppressed · aggregates only")
