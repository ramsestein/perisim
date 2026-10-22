"""Construcción de componentes de T1, T1, T2 (continua) y variables arrastradas desde el formato común.

T2 por categorías y T3 dependen de umbrales del fold de entrenamiento: se calculan en el modelado
(`t2_categories`, `t3_labels`), nunca aquí (§4.4.4).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from perisim.timeline import merge_runs

ALL_COMPONENTS = ["icu_readmission", "death", "reoperation", "vent_start", "vaso_start", "rrt_start", "aki_3",
                  "aki_12", "antibiotic_course", "transfusion", "hypoxemia_ward"]


def _flag(stays: pl.DataFrame, ids: pl.Series | list, name: str) -> pl.DataFrame:
    ids = ids.to_list() if isinstance(ids, pl.Series) else list(ids)
    return stays.select("stay_id").with_columns(pl.col("stay_id").is_in(ids).alias(name))


def _antibiotic_courses(times: pl.DataFrame, s: pl.DataFrame, cfg: dict) -> list[str]:
    """Curso nuevo de antibiótico sistémico: 1.ª administración tras t0 + 24 h, sin J01 en las 48 h previas,
    y administración en ≥ N días consecutivos (ventanas de 24 h desde esa administración)."""
    after = float(cfg["start_after_t0_hours"]) * 60
    wash = float(cfg["washout_hours"]) * 60
    ndays = int(cfg["min_consecutive_days"])
    j = times.join(s.select("stay_id", "t0", "win_end"), on="stay_id")
    out = []
    for (sid,), g in j.group_by(["stay_id"]):
        t = np.sort(g["time_min"].to_numpy())
        t0, wend = g["t0"][0], g["win_end"][0]
        for a in t[(t > t0 + after) & (t <= wend)]:
            prev = t[(t >= a - wash) & (t < a)]
            if prev.size:
                continue
            if all(((t >= a + d * 1440) & (t < a + (d + 1) * 1440)).any() for d in range(ndays)):
                out.append(sid)
                break
    return out


def _hypoxemia(spo2: pl.DataFrame, thr: float, k: int) -> list[str]:
    out = []
    for (sid,), g in spo2.sort("time_min").group_by(["stay_id"], maintain_order=True):
        low = (g["value"].to_numpy() < thr).astype(int)
        run = 0
        for v in low:
            run = run + 1 if v else 0
            if run >= k:
                out.append(sid)
                break
    return out


def build_targets(stays: pl.DataFrame, inputs: pl.DataFrame, outcomes: pl.DataFrame, tcfg: dict, cohort: str,
                  margin_hours: float, support_gap_hours: float) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    fu = float(tcfg["window"]["followup_days"]) * 1440
    s = stays.select(
        "stay_id", "subject_id", pl.col("icu_out_min").alias("t0"),
        (pl.col("icu_out_min") - margin_hours * 60).alias("cutoff"),
        pl.col("hosp_disch_min").alias("disch"), pl.col("death_min").alias("death"), "discharge_location",
    ).with_columns(pl.min_horizontal(pl.col("t0") + fu, pl.col("disch").fill_null(float("inf"))).alias("win_end"))
    observable = set(tcfg["t1"]["observable"].get(cohort, []))
    comp: dict[str, pl.DataFrame] = {}
    diag: dict = {}

    oc = outcomes.join(s, on="stay_id", how="inner")
    in_win = (pl.col("time_min") > pl.col("t0")) & (pl.col("time_min") <= pl.col("win_end"))
    comp["icu_readmission"] = _flag(s, oc.filter((pl.col("component") == "icu_admission") & in_win)["stay_id"],
                                    "icu_readmission")
    comp["death"] = s.select("stay_id", (pl.col("death").is_not_null() & (pl.col("death") > pl.col("t0"))
                                         & (pl.col("death") <= pl.col("t0") + fu)).alias("death"))
    day_t0 = (pl.col("t0") / 1440).floor() * 1440
    surg = oc.filter(pl.col("component") == "surgery").filter(
        pl.when(pl.col("precision") == "day")
        .then((pl.col("time_min") >= day_t0 + 1440) & (pl.col("time_min") <= pl.col("win_end")))
        .otherwise(in_win))
    comp["reoperation"] = _flag(s, surg["stay_id"], "reoperation")
    tr = oc.filter((pl.col("component") == "transfusion_rbc") & in_win)
    comp["transfusion"] = _flag(s, tr["stay_id"], "transfusion")

    li = inputs.join(s, on="stay_id", how="inner")
    sup = li.filter(pl.col("variable").str.starts_with("support_"))
    runs = merge_runs(sup.select("stay_id", "variable", "time_min", "end_min"), support_gap_hours * 60)
    runs = runs.join(s.select("stay_id", "t0", "win_end"), on="stay_id")
    for kind in ("vent", "vaso", "rrt"):
        r = runs.filter((pl.col("variable") == f"support_{kind}") & (pl.col("start_min") > pl.col("t0"))
                        & (pl.col("start_min") <= pl.col("win_end")))
        comp[f"{kind}_start"] = _flag(s, r["stay_id"], f"{kind}_start")

    # LRA por creatinina (KDIGO), basal = última creatinina antes del corte
    kd = tcfg["kdigo"]
    cr = li.filter(pl.col("variable") == "lab_creatinine")
    base = (cr.filter((pl.col("time_min") <= pl.col("cutoff"))
                      & (pl.col("time_min") >= pl.col("cutoff") - float(kd["baseline_lookback_days"]) * 1440))
            .sort("time_min").group_by("stay_id").agg(pl.col("value").last().alias("cr_base")))
    post = cr.filter(in_win)
    p_all = post.group_by("stay_id").agg(pl.col("value").max().alias("cr_max"), pl.len().alias("n_cr_post"))
    p48 = post.filter(pl.col("time_min") <= pl.col("t0") + 2880).group_by("stay_id").agg(pl.col("value").max().alias("cr_max48"))
    aki = s.select("stay_id").join(base, on="stay_id", how="left").join(p_all, on="stay_id", how="left") \
        .join(p48, on="stay_id", how="left")
    ratio = pl.col("cr_max") / pl.col("cr_base")
    st3 = (ratio >= kd["ratio_stage3"]) | ((pl.col("cr_max") >= kd["abs_stage3_mgdl"])
                                           & ((pl.col("cr_max") - pl.col("cr_base")) >= kd["abs_increase_48h"]))
    st2 = ratio >= kd["ratio_stage2"]
    st1 = (ratio >= kd["ratio_stage1"]) | ((pl.col("cr_max48") - pl.col("cr_base")) >= kd["abs_increase_48h"])
    aki = aki.with_columns(
        pl.when(st3.fill_null(False)).then(3).when(st2.fill_null(False)).then(2).when(st1.fill_null(False)).then(1)
        .otherwise(0).cast(pl.Int8).alias("aki_stage"),
        (pl.col("cr_base").is_not_null() & (pl.col("n_cr_post").fill_null(0) > 0)).alias("aki_observable"),
    )
    comp["aki_3"] = aki.select("stay_id", (pl.col("aki_stage") == 3).alias("aki_3"))
    comp["aki_12"] = aki.select("stay_id", pl.col("aki_stage").is_in([1, 2]).alias("aki_12"))

    ab = li.filter(pl.col("variable").str.starts_with("atc:" + "J01")).select("stay_id", "time_min")
    comp["antibiotic_course"] = _flag(s, _antibiotic_courses(ab, s, tcfg["antibiotic_course"]), "antibiotic_course")
    any_med_post = li.filter(pl.col("variable").str.starts_with("atc:") & in_win)["stay_id"].unique().to_list()

    hx = tcfg["hypoxemia_ward"]
    spo2 = li.filter((pl.col("variable") == "vital_spo2") & in_win).select("stay_id", "time_min", "value")
    comp["hypoxemia_ward"] = _flag(s, _hypoxemia(spo2, hx["spo2_threshold"], hx["min_consecutive"]), "hypoxemia_ward")

    # Ensamblado; lo no observable en la cohorte queda nulo
    t = s.select("stay_id", "subject_id")
    for name in ALL_COMPONENTS:
        col = comp[name].select("stay_id", pl.col(name).cast(pl.Int8))
        if name not in observable:
            col = col.with_columns(pl.lit(None, pl.Int8).alias(name))
        t = t.join(col, on="stay_id", how="left")
    t = t.join(aki.select("stay_id", "aki_stage", "aki_observable", "cr_base", "n_cr_post"), on="stay_id", how="left")
    t = t.with_columns(pl.col("stay_id").is_in(any_med_post).alias("med_observable"))

    t1c = tcfg["t1"]
    excl = set(t1c.get("exclude_components") or [])
    grave = [c for c in t1c["grave_components"] if c in observable and c not in excl]
    leve = [c for c in t1c["leve_components"] + list(t1c.get("include_sensitivity_components") or [])
            if c in observable and c not in excl]
    any_of = lambda cols: pl.any_horizontal([pl.col(c).fill_null(0) == 1 for c in cols]) if cols else pl.lit(False)  # noqa: E731
    reduced = cohort in set(t1c.get("reduced_cohorts") or [])
    t = t.with_columns(any_of(grave).cast(pl.Int8).alias("t1_grave"))
    t = t.with_columns(
        (pl.lit(None, pl.Int8) if reduced else
         pl.when(pl.col("t1_grave") == 1).then(2).when(any_of(leve)).then(1).otherwise(0).cast(pl.Int8)).alias("t1"))

    trunc = float(tcfg["t2"]["truncate_days"])
    los = (s["disch"] - s["t0"]) / 1440.0
    t = t.join(s.select("stay_id", ((pl.col("disch") - pl.col("t0")) / 1440.0).alias("los_raw"),
                        pl.col("death").is_null().alias("survivor"), "discharge_location"), on="stay_id", how="left")
    t = t.with_columns(
        (pl.col("los_raw") > trunc).alias("los_truncated"),
        pl.when(pl.col("survivor")).then(pl.col("los_raw").clip(0, trunc)).otherwise(None).alias("los_post_icu_days"),
    ).drop("los_raw")

    carried_cfg = tcfg["carried"]
    aki_any = (pl.col("aki_stage") >= 1).cast(pl.Int8) if "aki_12" in observable else pl.lit(None, pl.Int8)
    carried = t.with_columns(aki_any.alias("aki_any")).select(
        "stay_id",
        *[c for c in carried_cfg["binary"] if c in t.columns or c == "aki_any"],
        *carried_cfg["continuous"],
        *carried_cfg["nominal"],
    )
    diag["n_stays"] = s.height
    diag["los_days_finite"] = int(np.isfinite(los.to_numpy()).sum())
    return t, carried, diag


# --------------------------------------------------------------------------- dentro del fold
def t2_thresholds(los_train: np.ndarray, tcfg: dict) -> tuple[float, float]:
    if tcfg["t2"]["categories"] == "absolute":
        a, b = tcfg["t2"]["absolute_cut_days"]
        return float(a), float(b)
    v = los_train[np.isfinite(los_train)]
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    if q2 <= q1:  # estancias muy discretas: se garantiza un umbral estrictamente mayor
        q2 = np.nextafter(q1, np.inf)
    return float(q1), float(q2)


def t2_categories(los: np.ndarray, thr: tuple[float, float]) -> np.ndarray:
    out = np.full(los.shape, -1, dtype=np.int64)
    ok = np.isfinite(los)
    out[ok] = np.digitize(los[ok], [thr[0], thr[1]], right=True)
    return out


def t3_thresholds(los_train: np.ndarray, t1_train: np.ndarray, tcfg: dict) -> tuple[float, float]:
    v = los_train[np.isfinite(los_train) & (t1_train == 0)]
    if v.size < 3:
        v = los_train[np.isfinite(los_train)]
    q = np.quantile(v, [tcfg["t3"]["early_quantile"], tcfg["t3"]["prolonged_quantile"]])
    return float(q[0]), float(q[1])


def t3_labels(t1: np.ndarray, los: np.ndarray, thr: tuple[float, float]) -> np.ndarray:
    """0 alta precoz sin complicación < 1 esperada < 2 prolongada o leve < 3 grave o muerte."""
    out = np.full(t1.shape, -1, dtype=np.int64)
    grave = t1 == 2
    leve = t1 == 1
    none = t1 == 0
    out[grave] = 3
    out[leve] = 2
    fin = np.isfinite(los)
    out[none & fin & (los > thr[1])] = 2
    out[none & fin & (los <= thr[0])] = 0
    out[none & fin & (los > thr[0]) & (los <= thr[1])] = 1
    return out
