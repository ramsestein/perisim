"""Construction of T1 components, T1, T2 (continuous) and carried variables from the common format.

Categorical T2 and T3 depend on thresholds from the training fold: they are computed during modelling
(`t2_categories`, `t3_labels`), never here (§4.4.4).

Two kinds of per-stay null component (decided 2026-10-22):
- NOT APPLICABLE (D5): AKI and new RRT for patients with RRT before the index surgery. The component does not
  count for that patient, exactly like a component that a cohort cannot observe at all.
- NOT OBSERVED (D6): the mild medication components (antibiotic course, transfusion) where the medication source
  does not cover the hospitalization (MIMIC-IV without eMAR). T1 (and therefore T3) is then undefined for the stay,
  whatever its other outcomes, so the exclusion depends on data coverage only, never on the outcome.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from perisim.timeline import merge_runs

ALL_COMPONENTS = ["icu_readmission", "death", "reoperation", "vent_start", "vaso_start", "rrt_start", "aki_3",
                  "aki_12", "antibiotic_course", "transfusion", "hypoxemia_ward"]
MEDICATION_MARKER = "src:medication"
CHRONIC_RRT_MARKER = "marker:chronic_rrt"


def _flag(stays: pl.DataFrame, ids: pl.Series | list, name: str) -> pl.DataFrame:
    ids = ids.to_list() if isinstance(ids, pl.Series) else list(ids)
    return stays.select("stay_id").with_columns(pl.col("stay_id").is_in(ids).alias(name))


def _antibiotic_courses(times: pl.DataFrame, s: pl.DataFrame, cfg: dict) -> list[str]:
    """New systemic antibiotic course: first administration after t0 + 24 h, with no J01 in the previous 48 h,
    and administration on ≥ N consecutive days (24 h windows from that administration). A record covers
    [start, end]: an order (MIMIC-IV prescriptions, eICU) its whole interval, an administration (eMAR) one instant."""
    after = float(cfg["start_after_t0_hours"]) * 60
    wash = float(cfg["washout_hours"]) * 60
    ndays = int(cfg["min_consecutive_days"])
    j = times.join(s.select("stay_id", "t0", "win_end"), on="stay_id")
    out = []
    for (sid,), g in j.group_by(["stay_id"]):
        st = g["time_min"].to_numpy().astype(float)
        en = g["end_min"].to_numpy().astype(float) if "end_min" in g.columns else st.copy()
        en = np.where(np.isfinite(en), np.maximum(en, st), st)
        t0, wend = g["t0"][0], g["win_end"][0]
        for a in np.unique(st[(st > t0 + after) & (st <= wend)]):
            if ((st < a) & (en >= a - wash)).any():          # J01 given (or still running) in the previous 48 h
                continue
            if all(((st < a + (d + 1) * 1440) & (en >= a + d * 1440)).any() for d in range(ndays)):
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


def _kdigo(cr: pl.DataFrame, s: pl.DataFrame, rrt_runs: pl.DataFrame, kd: dict) -> pl.DataFrame:
    """KDIGO creatinine criteria (AUDIT D7, decided 2026-10-22).

    Baseline = last creatinine before the cutoff within `baseline_lookback_days` (SPECS), never one measured during
    RRT (D5). An AKI event is detected at a post-t0 measurement v(t) when v ≥ ratio_stage1 × the minimum of the
    previous `window_ratio_days` or v ≥ abs_increase_48h + the minimum of the previous `window_abs_hours`. The
    baseline always belongs to the 7-day reference set, and to the 48-h set during the first 48 h after t0, so the
    SPECS rule is a special case. Stage by the maximum ratio to baseline (≥ ratio_stage2 → 2, ≥ ratio_stage3 → 3);
    stage 3 also when v ≥ abs_stage3_mgdl at a detected measurement with a baseline below it (D5). Creatinine
    measured during an RRT run or within `rrt_exclusion_hours` after it is ignored."""
    w7 = float(kd.get("window_ratio_days", 7)) * 1440
    w48 = float(kd.get("window_abs_hours", 48)) * 60
    rx = float(kd.get("rrt_exclusion_hours", 24)) * 60
    r1, r2, r3 = float(kd["ratio_stage1"]), float(kd["ratio_stage2"]), float(kd["ratio_stage3"])
    a48, a3 = float(kd["abs_increase_48h"]), float(kd["abs_stage3_mgdl"])
    look = float(kd["baseline_lookback_days"]) * 1440
    if rrt_runs.height:
        r = rrt_runs.select("stay_id", "start_min", (pl.col("end_min") + rx).alias("_e"))
        during = cr.join(r, on="stay_id", how="inner").filter(
            (pl.col("time_min") >= pl.col("start_min")) & (pl.col("time_min") <= pl.col("_e")))
        cr = cr.join(during.select("stay_id", "time_min").unique(), on=["stay_id", "time_min"], how="anti")
    cr = cr.join(s.select("stay_id", "t0", "cutoff", "win_end"), on="stay_id", how="inner").sort(["stay_id", "time_min"])
    rows = []
    if cr.height:
        sid = cr["stay_id"].to_numpy()
        tt, vv = cr["time_min"].to_numpy(), cr["value"].to_numpy()
        t0s, cuts, wends = cr["t0"].to_numpy(), cr["cutoff"].to_numpy(), cr["win_end"].to_numpy()
        starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
        for a, b in zip(starts, np.r_[starts[1:], len(sid)]):
            t, v, t0, cut, we = tt[a:b], vv[a:b], t0s[a], cuts[a], wends[a]
            bsel = (t <= cut) & (t >= cut - look)
            base = float(v[bsel][-1]) if bsel.any() else np.nan
            post = np.flatnonzero((t > t0) & (t <= we))
            stage = 0
            if np.isfinite(base) and post.size:
                detected = abs3 = False
                for i in post:
                    ref7 = v[(t >= t[i] - w7) & (t < t[i])]
                    ref48 = v[(t >= t[i] - w48) & (t < t[i])]
                    m7 = min(base, ref7.min()) if ref7.size else base
                    m48 = ref48.min() if ref48.size else np.inf
                    if t[i] <= t0 + w48:
                        m48 = min(m48, base)
                    hit = (v[i] >= r1 * m7) or (v[i] - m48 >= a48)
                    detected |= hit
                    abs3 |= hit and v[i] >= a3 and base < a3
                ratio = float(v[post].max()) / base
                stage = 3 if (ratio >= r3 or abs3) else 2 if ratio >= r2 else 1 if detected else 0
            rows.append((sid[a], base, int(post.size), stage))
    out = pl.DataFrame(rows, schema={"stay_id": pl.Utf8, "cr_base": pl.Float64, "n_cr_post": pl.Int64,
                                     "aki_stage": pl.Int8}, orient="row")
    return s.select("stay_id").join(out, on="stay_id", how="left").with_columns(
        pl.when(pl.col("cr_base").is_not_null()).then(pl.col("aki_stage").fill_null(0)).otherwise(0)
        .cast(pl.Int8).alias("aki_stage"))


def build_targets(stays: pl.DataFrame, inputs: pl.DataFrame, outcomes: pl.DataFrame, tcfg: dict, cohort: str,
                  margin_hours: float, support_gap_hours: float, med_coverage_tracked: bool = False
                  ) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    fu = float(tcfg["window"]["followup_days"]) * 1440
    opt = lambda c: pl.col(c) if c in stays.columns else pl.lit(None, pl.Float64)  # noqa: E731
    s = stays.select(
        "stay_id", "subject_id", pl.col("icu_out_min").alias("t0"),
        (pl.col("icu_out_min") - margin_hours * 60).alias("cutoff"),
        pl.col("hosp_disch_min").alias("disch"), pl.col("death_min").alias("death"), "discharge_location",
        pl.coalesce(opt("surgery_start_min"), opt("icu_in_min")).alias("surg_ref"),
    ).with_columns(pl.min_horizontal(pl.col("t0") + fu, pl.col("disch").fill_null(float("inf"))).alias("win_end"))
    observable = set(tcfg["t1"]["observable"].get(cohort, []))
    comp: dict[str, pl.DataFrame] = {}
    diag: dict = {}

    oc = outcomes.join(s, on="stay_id", how="inner")
    in_win = (pl.col("time_min") > pl.col("t0")) & (pl.col("time_min") <= pl.col("win_end"))
    icu_entry = oc.filter((pl.col("component") == "icu_admission") & in_win)["stay_id"].unique().to_list()
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
    new_runs = runs.filter((pl.col("start_min") > pl.col("t0")) & (pl.col("start_min") <= pl.col("win_end")))
    for kind in ("vent", "vaso", "rrt"):
        r = new_runs.filter(pl.col("variable") == f"support_{kind}")
        comp[f"{kind}_start"] = _flag(s, r["stay_id"], f"{kind}_start")

    # D5: RRT before the index surgery (device / procedure record or ESRD code) → AKI and new RRT not applicable
    pre = li.filter(((pl.col("variable") == CHRONIC_RRT_MARKER) | (pl.col("variable") == "support_rrt"))
                    & (pl.col("time_min") < pl.col("surg_ref")))
    chronic = pre["stay_id"].unique().to_list()

    # ICU readmission: a new ICU entry, plus (D10, SPECS D1 proxy) a new organ-support run after t0 in cohorts whose
    # ICU stays are only recorded after a surgery (INSPIRE). Ward dialysis of a chronic-RRT patient is not a proxy.
    proxy_kinds = [f"support_{k}" for k in ((tcfg["t1"].get("icu_readmission_proxy") or {}).get(cohort) or [])]
    proxy = new_runs.filter(pl.col("variable").is_in(proxy_kinds)
                            & ~((pl.col("variable") == "support_rrt") & pl.col("stay_id").is_in(chronic)))
    proxy = proxy["stay_id"].unique().to_list()
    comp["icu_readmission"] = _flag(s, list(set(icu_entry) | set(proxy)), "icu_readmission")

    # AKI by creatinine (KDIGO with rolling windows, D7), baseline = last creatinine before the cutoff
    kd = tcfg["kdigo"]
    cr = li.filter(pl.col("variable") == "lab_creatinine").select("stay_id", "time_min", "value")
    rrt_runs = runs.filter(pl.col("variable") == "support_rrt") if kd.get("exclude_during_rrt", True) else runs.head(0)
    aki = _kdigo(cr, s, rrt_runs, kd).with_columns(
        (pl.col("cr_base").is_not_null() & (pl.col("n_cr_post").fill_null(0) > 0)).alias("aki_observable"))
    comp["aki_3"] = aki.select("stay_id", (pl.col("aki_stage") == 3).alias("aki_3"))
    comp["aki_12"] = aki.select("stay_id", pl.col("aki_stage").is_in([1, 2]).alias("aki_12"))

    ab = li.filter(pl.col("variable").str.starts_with("atc:" + "J01")).select("stay_id", "time_min", "end_min")
    comp["antibiotic_course"] = _flag(s, _antibiotic_courses(ab, s, tcfg["antibiotic_course"]), "antibiotic_course")
    any_med_post = li.filter(pl.col("variable").str.starts_with("atc:") & in_win)["stay_id"].unique().to_list()
    # D6: medication source coverage of the hospitalization (any record, any time)
    med_cov = li.filter(pl.col("variable") == MEDICATION_MARKER)["stay_id"].unique().to_list() \
        if med_coverage_tracked else s["stay_id"].to_list()

    hx = tcfg["hypoxemia_ward"]
    spo2 = li.filter((pl.col("variable") == "vital_spo2") & in_win).select("stay_id", "time_min", "value")
    comp["hypoxemia_ward"] = _flag(s, _hypoxemia(spo2, hx["spo2_threshold"], hx["min_consecutive"]), "hypoxemia_ward")

    # Assembly; components not observable in the cohort are left null
    t1c = tcfg["t1"]
    med_comps = set(t1c.get("medication_components") or [])
    t = s.select("stay_id", "subject_id").with_columns(
        pl.col("stay_id").is_in(chronic).alias("chronic_rrt"),
        pl.col("stay_id").is_in(med_cov).alias("med_source_observed"))
    for name in ALL_COMPONENTS:
        col = comp[name].select("stay_id", pl.col(name).cast(pl.Int8))
        if name not in observable:
            col = col.with_columns(pl.lit(None, pl.Int8).alias(name))
        t = t.join(col, on="stay_id", how="left")
        if name in ("aki_3", "aki_12", "rrt_start"):          # D5: not applicable
            t = t.with_columns(pl.when(pl.col("chronic_rrt")).then(None).otherwise(pl.col(name)).alias(name))
        if name in med_comps:                                 # D6: not observed
            t = t.with_columns(pl.when(pl.col("med_source_observed")).then(pl.col(name)).otherwise(None).alias(name))
    t = t.join(aki.select("stay_id", "aki_stage", "aki_observable", "cr_base", "n_cr_post"), on="stay_id", how="left")
    t = t.with_columns(
        pl.when(pl.col("chronic_rrt")).then(None).otherwise(pl.col("aki_stage")).alias("aki_stage"),
        (pl.col("aki_observable") & ~pl.col("chronic_rrt")).alias("aki_observable"),
        (pl.col("stay_id").is_in(any_med_post) & pl.col("med_source_observed")).alias("med_observable"),
        pl.lit(bool(med_coverage_tracked)).alias("med_coverage_tracked"),
        pl.col("stay_id").is_in(icu_entry).cast(pl.Int8).alias("icu_readmission_icu_entry"),
        pl.col("stay_id").is_in(proxy).cast(pl.Int8).alias("icu_readmission_proxy"))

    excl = set(t1c.get("exclude_components") or [])
    severe = [c for c in t1c["severe_components"] if c in observable and c not in excl]
    mild = [c for c in t1c["mild_components"] + list(t1c.get("include_sensitivity_components") or [])
            if c in observable and c not in excl]
    any_of = lambda cols: pl.any_horizontal([pl.col(c).fill_null(0) == 1 for c in cols]) if cols else pl.lit(False)  # noqa: E731
    reduced = cohort in set(t1c.get("reduced_cohorts") or [])
    unobserved = [c for c in mild if c in med_comps]
    undefined = (~pl.col("med_source_observed")) if (unobserved and t1c.get("unobserved_mild_policy", "exclude_stay")
                                                       == "exclude_stay") else pl.lit(False)
    t = t.with_columns(any_of(severe).cast(pl.Int8).alias("t1_severe"))
    t = t.with_columns(
        (pl.lit(None, pl.Int8) if reduced else
         pl.when(undefined).then(None).when(pl.col("t1_severe") == 1).then(2).when(any_of(mild)).then(1).otherwise(0)
         .cast(pl.Int8)).alias("t1"))

    # T2 = survivors' stay categories + death as the worst ordinal state (the thresholds use survivors only)
    trunc = float(tcfg["t2"]["truncate_days"])
    los = (s["disch"] - s["t0"]) / 1440.0
    t = t.join(s.select("stay_id", ((pl.col("disch") - pl.col("t0")) / 1440.0).alias("los_raw"),
                        (pl.col("death").is_null() | (pl.col("death") > pl.col("t0") + fu)).alias("survivor"), "discharge_location"), on="stay_id", how="left")
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
    diag["n_chronic_rrt"] = len(chronic)
    diag["n_medication_unobserved"] = int(s.height - len(set(med_cov)))
    return t, carried, diag


# --------------------------------------------------------------------------- within the fold
def t2_thresholds(los_train: np.ndarray, tcfg: dict) -> tuple[float, float]:
    if tcfg["t2"]["categories"] == "absolute":
        a, b = tcfg["t2"]["absolute_cut_days"]
        return float(a), float(b)
    v = los_train[np.isfinite(los_train)]
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    if q2 <= q1:  # highly discrete stays: guarantee a strictly greater threshold
        q2 = np.nextafter(q1, np.inf)
    return float(q1), float(q2)


def t2_categories(los: np.ndarray, thr: tuple[float, float], death: np.ndarray | None = None) -> np.ndarray:
    """T2 (decided 2026-10-24): 0 short < 1 medium < 2 long (tertiles of the SURVIVORS' post-ICU stay) < 3 in-hospital
    death within the follow-up. Defined for every patient at t0; −1 only when neither is known."""
    out = np.full(los.shape, -1, dtype=np.int64)
    ok = np.isfinite(los)
    out[ok] = np.digitize(los[ok], [thr[0], thr[1]], right=True)
    if death is not None:
        out[np.asarray(death) == 1] = 3
    return out


def t3_thresholds(los_train: np.ndarray, t1_train: np.ndarray, tcfg: dict) -> tuple[float, float]:
    v = los_train[np.isfinite(los_train) & (t1_train == 0)]
    if v.size < 3:
        v = los_train[np.isfinite(los_train)]
    q = np.quantile(v, [tcfg["t3"]["early_quantile"], tcfg["t3"]["prolonged_quantile"]])
    return float(q[0]), float(q[1])


def t3_labels(t1: np.ndarray, los: np.ndarray, thr: tuple[float, float]) -> np.ndarray:
    """0 early discharge without complication < 1 expected without complication < 2 prolonged or mild complication
    < 3 severe complication or death."""
    out = np.full(t1.shape, -1, dtype=np.int64)
    severe = t1 == 2
    mild = t1 == 1
    none = t1 == 0
    out[severe] = 3
    out[mild] = 2
    fin = np.isfinite(los)
    out[none & fin & (los > thr[1])] = 2
    out[none & fin & (los <= thr[0])] = 0
    out[none & fin & (los > thr[0]) & (los <= thr[1])] = 1
    return out
