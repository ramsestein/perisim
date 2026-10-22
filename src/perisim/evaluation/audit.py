"""Auditoría de completitud por cohorte (§7): primer entregable técnico y puerta de parada.

Si algún componente de T1 tiene prevalencia < 0,5 % o ausencia > 30 % en alguna cohorte, el pipeline se
detiene y lo anota en OPEN_QUESTIONS (en el workdir) hasta que se confirme `gates.audit_acknowledged`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from perisim.evaluation.html import box, page, suppress, table

PREV_MIN = 0.005
MISS_MAX = 0.30


def completeness(cohort: str, flow: list[dict], stays: pl.DataFrame, inputs: pl.DataFrame, targets: pl.DataFrame,
                 X: pl.DataFrame, specs: list, cfg) -> tuple[dict, str]:
    min_n = int(cfg.experiment["evaluation"]["small_cell_threshold"])
    tcfg = cfg.targets
    observable = list(tcfg["t1"]["observable"].get(cohort, []))
    n = targets.height
    rep: dict = {"cohort": cohort, "n_index": n, "flow": flow, "issues": []}

    # --- inputs: ausencia, distribución y unidades
    units = {}
    for vname, v in cfg.harmonization.get("variables", {}).items():
        units[vname.replace("lab_", "").replace("vital_", "")] = v.get("units")
    inp_rows = []
    for s in specs:
        col = X[s.name] if s.name in X.columns else pl.Series([None] * X.height)
        miss = float(col.is_null().mean()) if X.height else 1.0
        row = {"feature": s.name, "type": s.type, "block": s.block, "missing_pct": round(100 * miss, 1)}
        base = s.name.split("_", 1)[1] if "_" in s.name else s.name
        row["units"] = units.get(base)
        if s.type in ("continuous", "ordinal"):
            v = col.drop_nulls().cast(pl.Float64)
            if v.len() >= min_n:
                q = np.quantile(v.to_numpy(), [0.05, 0.25, 0.5, 0.75, 0.95])
                row.update({f"p{int(p * 100)}": float(x) for p, x in zip([0.05, 0.25, 0.5, 0.75, 0.95], q)})
        elif s.type == "binary":
            k = int(col.fill_null(0).cast(pl.Float64).sum())
            row["n_pos"] = suppress(k, min_n)
            row["prev_pct"] = round(100 * k / max(X.height, 1), 2) if k >= min_n else None
        else:
            vc = col.drop_nulls().value_counts(sort=True).head(8)
            row["top"] = "; ".join(f"{r[s.name]} ({r['count']})" for r in vc.to_dicts() if r["count"] >= min_n)
        inp_rows.append(row)
    rep["inputs"] = inp_rows

    # --- frecuencia de medición en planta tras t0 (viabilidad de la categoría leve)
    fu = float(tcfg["window"]["followup_days"]) * 1440
    s = stays.select("stay_id", pl.col("icu_out_min").alias("t0"),
                     pl.min_horizontal(pl.col("icu_out_min") + fu, pl.col("hosp_disch_min").fill_null(float("inf"))).alias("we"))
    post = inputs.join(s, on="stay_id").filter((pl.col("time_min") > pl.col("t0")) & (pl.col("time_min") <= pl.col("we")))
    los_days = s.with_columns(((pl.col("we") - pl.col("t0")) / 1440).clip(1 / 24, None).alias("d"))
    freq = []
    for label, cond in [("creatinina", pl.col("variable") == "lab_creatinine"),
                        ("hemoglobina", pl.col("variable") == "lab_hb"),
                        ("antibiótico J01", pl.col("variable").str.starts_with("atc:J01")),
                        ("SpO2 en planta", pl.col("variable") == "vital_spo2")]:
        c = post.filter(cond).group_by("stay_id").len()
        c = los_days.join(c, on="stay_id", how="left").with_columns(pl.col("len").fill_null(0))
        with_any = int((c["len"] > 0).sum())
        per_day = (c.filter(pl.col("len") > 0)["len"] / c.filter(pl.col("len") > 0)["d"]).to_numpy()
        freq.append({"measure": label, "stays_with_any": suppress(with_any, min_n),
                     "pct_stays": round(100 * with_any / max(n, 1), 1) if with_any >= min_n else None,
                     "median_per_day": float(np.median(per_day)) if per_day.size >= min_n else None})
    rep["ward_frequency"] = freq

    # --- prevalencias de componentes, clases y estancia
    comp_rows = []
    miss_map = {"aki_3": "aki_observable", "aki_12": "aki_observable", "antibiotic_course": "med_observable",
                "transfusion": None}
    t1_def = set(tcfg["t1"]["grave_components"]) | set(tcfg["t1"]["leve_components"])
    for comp in ["icu_readmission", "death", "reoperation", "vent_start", "vaso_start", "rrt_start", "aki_3", "aki_12",
                 "antibiotic_course", "transfusion", "hypoxemia_ward"]:
        obs = comp in observable
        k = int(targets[comp].fill_null(0).sum()) if obs and comp in targets.columns else None
        prev = (k / n) if (k is not None and n) else None
        mcol = miss_map.get(comp)
        miss = float(1 - targets[mcol].fill_null(False).mean()) if (obs and mcol and mcol in targets.columns and n) else 0.0
        row = {"component": comp, "observable": obs, "n": suppress(k, min_n) if k is not None else None,
               "prev_pct": round(100 * prev, 2) if (prev is not None and k >= min_n) else None,
               "missing_pct": round(100 * miss, 1) if obs else None}
        comp_rows.append(row)
        if obs and comp in t1_def:
            if prev is not None and prev < PREV_MIN:
                rep["issues"].append(f"{cohort}: componente {comp} con prevalencia {100 * prev:.2f} % (< 0,5 %)")
            if miss > MISS_MAX:
                rep["issues"].append(f"{cohort}: componente {comp} con ausencia {100 * miss:.1f} % (> 30 %)")
    rep["components"] = comp_rows
    if "t1" in targets.columns and targets["t1"].drop_nulls().len():
        vc = targets["t1"].drop_nulls().value_counts().sort("t1")
        rep["t1_classes"] = [{"class": ["ninguna", "leve", "grave"][int(r["t1"])], "n": suppress(int(r["count"]), min_n)}
                             for r in vc.to_dicts()]
    kg = int(targets["t1_grave"].fill_null(0).sum())
    rep["t1_grave"] = {"n": suppress(kg, min_n), "pct": round(100 * kg / max(n, 1), 2) if kg >= min_n else None}
    los = targets["los_post_icu_days"].drop_nulls().to_numpy()
    if los.size >= min_n:
        rep["t2"] = {"n_survivors": int(los.size), "quantiles": dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                                                         np.quantile(los, [.1, .25, .5, .75, .9]).tolist())),
                     "tertiles_descriptive": np.quantile(los, [1 / 3, 2 / 3]).tolist(),
                     "truncated_pct": round(100 * float(targets["los_truncated"].fill_null(False).mean()), 2)}
    if cohort == "inspire":
        k = int(targets["icu_readmission"].fill_null(0).sum())
        rep["icu_readmission_inspire"] = {
            "n_detected": suppress(k, min_n),
            "note": "Detectado como nueva entrada en UCI ligada a otra cirugía de la misma hospitalización. "
                    "Los reingresos no quirúrgicos no son observables por esta vía (D1)."}
    return rep, render(rep, min_n)


def render(rep: dict, min_n: int) -> str:
    c = rep["cohort"]
    sec = []
    if rep["issues"]:
        sec.append(box("<b>Puerta §7: PARAR.</b><br>" + "<br>".join(rep["issues"]) +
                       "<br>Anotado en docs/OPEN_QUESTIONS (workdir). Para continuar tras revisarlo: "
                       "<code>PERISIM_ACK_AUDIT=true bash run_pod.sh</code>", "bad"))
    else:
        sec.append(box("Sin incidencias de prevalencia (< 0,5 %) ni ausencia (> 30 %) en los componentes de T1.", "ok"))
    sec.append("<h2>Diagrama de flujo</h2>" + table(["Paso", "Estancias", "Pacientes", "Excluidas"],
                                                    [[f["step"], suppress(f["n_stays"], min_n), suppress(f["n_subjects"], min_n),
                                                      suppress(f["excluded"], min_n) if f["excluded"] is not None else None]
                                                     for f in rep["flow"]]))
    sec.append("<h2>Componentes de T1, clases y estancia</h2>" + table(
        ["Componente", "Observable", "n", "Prevalencia %", "Ausencia %"],
        [[r["component"], "sí" if r["observable"] else "no (por diseño)", r["n"], r["prev_pct"], r["missing_pct"]]
         for r in rep["components"]], nd=2))
    if rep.get("t1_classes"):
        sec.append(table(["Clase T1", "n"], [[r["class"], r["n"]] for r in rep["t1_classes"]]))
    sec.append(table(["T1 grave / no grave", "n", "%"], [["grave", rep["t1_grave"]["n"], rep["t1_grave"]["pct"]]], nd=2))
    if rep.get("t2"):
        t2 = rep["t2"]
        sec.append(table(["Estancia post-UCI (supervivientes)", "valor"],
                         [["n", t2["n_survivors"]]] + [[k, v] for k, v in t2["quantiles"].items()] +
                         [["terciles (descriptivos; en el modelado se calculan en cada fold)", " / ".join(f"{x:.2f}" for x in t2["tertiles_descriptive"])],
                          ["% truncada a 30 d", t2["truncated_pct"]]], nd=2))
    sec.append("<h2>Frecuencia de medición tras t0 (viabilidad de la categoría leve)</h2>" + table(
        ["Medida", "Estancias con ≥1", "% estancias", "Mediana por día"],
        [[r["measure"], r["stays_with_any"], r["pct_stays"], r["median_per_day"]] for r in rep["ward_frequency"]], nd=2))
    if rep.get("icu_readmission_inspire"):
        r = rep["icu_readmission_inspire"]
        sec.append("<h2>Viabilidad del reingreso en UCI (INSPIRE, D1)</h2>" + box(f"Detectados: {r['n_detected']}. {r['note']}", "warn"))
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
    sec.append("<h2>Inputs: ausencia, distribución y unidades</h2>" + table(
        ["Variable", "Tipo", "Bloque", "Ausencia %", "Unidades", "Distribución"], rows, nd=1))
    return page(f"Auditoría de completitud — {c}", sec,
                f"PERISIM · §7 · celdas con n < {min_n} suprimidas · solo agregados")
