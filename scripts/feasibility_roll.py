"""Phase 0 of experiment `roll`: can the prequential streams be built, and with how many patients and events?

Reads `hypo`'s analysis landmarks (work/experiments/hypo/landmarks, the same population, inputs and outcome) and the
calendar and hospital fields of the original tables. Aggregates only (counts ≥ 10; eICU hospitals labelled H1, H2, …
by size, as in `hypo`). Nothing is fitted. The freezes of T2 and `hypo` are checked at the start and at the end.

- MIMIC-IV: patients, landmarks and events per `anchor_year_group` block; how many years after its anchor year
  each stay falls (a stay can belong to a later block than its patient's group).
- eICU: per hospital and discharge year (2014, 2015), patients, landmarks and events; how many hospitals pass
  candidate thresholds; whether `patientunitstayid` grows with the year within a hospital.
- INSPIRE: whether any field orders the patients in calendar time.
- All three: totals for the cross-cohort streams; the precision of a window's AUROC against its size in patients,
  read from `hypo`'s own transported predictions (random windows of W patients); the admission gate (bounds,
  duplicates, impossible values) on the analysis landmarks.

Added in Phase 1 (`phase1` argument):
- T2 (secondary, descriptive): its analysis stays per block, year and hospital;
- the noise of the realistic ML arm's monitor: how often each proposed trigger fires in random windows of M
  patients, with no drift.

Added before the pre-registration, by the lead's decision (`s1cal` argument): the calibration of `hypo`'s S1 case
bases of n random patients (a new aggregate of examined predictions).

Usage:  python scripts/feasibility_roll.py [phase1 | s1cal]   (set PERISIM_WORKDIR / PERISIM_DATASETS from a worktree)
Output: <workdir>/experiments/roll/feasibility/feasibility_roll.json and .md (feasibility_roll_phase1.json,
        feasibility_roll_s1cal.json)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypo.config import load as load_hypo  # noqa: E402
from hypo.config import workdir as hypo_workdir  # noqa: E402
from hypo.external import hospital_labels  # noqa: E402
from hypo.data import load as load_lm  # noqa: E402
from hypo.frozen import check as hypo_check  # noqa: E402
from perisim.config import code_hash, load_config  # noqa: E402

MIN_CELL = 10
WINDOWS = [50, 100, 150, 200, 300, 500, 1000]
N_DRAWS = 300
EICU_PAT = [100, 150, 200, 300]
EICU_EV = [10, 20, 30]


def work() -> Path:
    return Path(os.environ.get("PERISIM_WORKDIR") or ROOT / "work")


def data_root() -> Path:
    return Path(os.environ.get("PERISIM_DATASETS") or ROOT / "data")


def out_dir() -> Path:
    return work() / "experiments" / "roll" / "feasibility"


# --------------------------------------------------------------------------- the frozen experiments
def frozen_guard() -> dict:
    """T2: code and configuration hashes against its freeze record; `hypo`: its own check (code, configuration,
    folds, Phase 2 comparators, final artefacts). Raises if anything changed."""
    fr = json.loads((work() / "experiments" / "t2" / "state" / "freeze.json").read_text(encoding="utf-8"))
    now = {"code_hash": code_hash(), "config_hash": load_config(None, "local", {}).scientific_hash()}
    bad = [k for k in now if now[k] != fr[k]]
    if bad:
        raise RuntimeError(f"T2's frozen {bad} changed: {now} vs the record {[fr[k] for k in bad]}")
    h = hypo_check()
    return {"t2": {"tag": fr["tag"], **now}, "hypo": {"tag": h["tag"], **{k: h["hashes"][k] for k in ("code", "config")}}}


# --------------------------------------------------------------------------- helpers
def cell(n: int | float) -> int | str:
    """A count, or "<10" (counts below MIN_CELL are not shown)."""
    return int(n) if n >= MIN_CELL or n == 0 else f"<{MIN_CELL}"


def landmarks(c: str) -> pl.DataFrame:
    """hypo's analysis landmarks of cohort c (normotensive now, outcome known), with the columns used here. The
    common format prefixes the identifiers with the cohort ("EICU-…"): `stay_n` / `subject_n` are the original
    numbers, for joining the original tables."""
    cols = ["stay_id", "subject_id", "L", "y"] + (["hospital"] if c == "eicu" else [])
    df = pl.read_parquet(hypo_workdir() / "landmarks" / f"{c}.parquet")
    df = df.filter(pl.col("normotensive") & pl.col("known")).select(cols)
    return df.with_columns(*(pl.col(k).cast(pl.Utf8).str.extract(r"(\d+)$").cast(pl.Int64).alias(k.replace("_id", "_n"))
                             for k in ("stay_id", "subject_id")))


def counts(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    pat = df.group_by(by + ["subject_id"]).agg(pl.col("y").max().alias("ev"))
    a = pat.group_by(by).agg(pl.len().alias("patients"), pl.col("ev").sum().alias("patients_with_event"))
    b = df.group_by(by).agg(pl.len().alias("landmarks"), pl.col("y").sum().alias("event_landmarks"))
    return a.join(b, on=by).with_columns((pl.col("event_landmarks") / pl.col("landmarks")).alias("prevalence")) \
        .sort(by)


def table(d: pl.DataFrame) -> list[dict]:
    out = []
    for r in d.to_dicts():
        out.append({k: (cell(v) if k in ("patients", "patients_with_event", "landmarks", "event_landmarks") else
                        (round(v, 4) if isinstance(v, float) else v)) for k, v in r.items()})
    return out


# --------------------------------------------------------------------------- MIMIC-IV
def mimic() -> dict:
    src = data_root() / "mimic4"
    lm = landmarks("mimic")
    con = duckdb.connect()
    pat = con.execute(f"SELECT subject_id, anchor_year, anchor_year_group FROM read_csv_auto('{(src / 'hosp' / 'patients.csv.gz').as_posix()}')").pl()
    icu = con.execute(f"SELECT stay_id, year(intime) AS stay_year FROM read_csv_auto('{(src / 'icu' / 'icustays.csv.gz').as_posix()}')").pl()
    df = lm.join(pat.rename({"subject_id": "subject_n"}), on="subject_n", how="left") \
        .join(icu.rename({"stay_id": "stay_n"}), on="stay_n", how="left")
    df = df.with_columns((pl.col("stay_year") - pl.col("anchor_year")).alias("years_after_anchor"))
    per_stay = df.unique("stay_id")
    yaa = per_stay["years_after_anchor"].drop_nulls().to_numpy()
    return {"blocks": table(counts(df, ["anchor_year_group"])),
            "missing_group": int(df["anchor_year_group"].null_count()),
            "years_after_anchor": {"stays": int(per_stay.height), "stays_without_icustays_match":
                                   int(per_stay["years_after_anchor"].null_count()), "share_0": float(np.mean(yaa == 0)),
                                   "share_ge3": float(np.mean(yaa >= 3)), "share_ge6": float(np.mean(yaa >= 6)),
                                   "q50": float(np.median(yaa)), "q90": float(np.quantile(yaa, 0.9))},
            "totals": table(counts(df.with_columns(pl.lit("all").alias("cohort")), ["cohort"]))[0]}


# --------------------------------------------------------------------------- eICU
def eicu_patient() -> pl.DataFrame:
    src = data_root() / "eicu"
    con = duckdb.connect()
    return con.execute(f"SELECT patientunitstayid AS stay_n, hospitalid, hospitaldischargeyear AS year FROM read_csv_auto('{(src / 'patient.csv.gz').as_posix()}')").pl()


def eicu_labels(cfg: dict, lm: pl.DataFrame) -> dict:
    """Hospital → label: `hypo`'s labels (H1, H2, … by size among hospitals with ≥ 300 patients), then the smaller
    hospitals by size."""
    hl = load_lm("eicu", cfg)
    labels = hospital_labels(hl, int(cfg["external"]["hospitals"]["min_patients"]))
    allh = lm.group_by("hospital").agg(pl.col("subject_id").n_unique().alias("p")).sort("p", descending=True)
    lab_all = {h: labels.get(str(h)) for h in allh["hospital"].to_list()}
    k = len(labels)
    for h in allh["hospital"].to_list():                      # hospitals below hypo's threshold: labelled after
        if lab_all[h] is None:
            k += 1
            lab_all[h] = f"H{k}"
    return lab_all


def eicu(cfg: dict) -> dict:
    lm = landmarks("eicu")
    df = lm.join(eicu_patient(), on="stay_n", how="left")
    lab_all = eicu_labels(cfg, lm)
    df = df.with_columns(pl.col("hospital").replace_strict(lab_all, return_dtype=pl.Utf8).alias("H"))
    hy = counts(df, ["H", "year"])
    wide = hy.pivot(on="year", index="H", values=["patients", "patients_with_event"])
    elig = {}
    for np_ in EICU_PAT:
        for ne in EICU_EV:
            ok = wide.filter((pl.col("patients_2014") >= np_) & (pl.col("patients_2015") >= np_)
                             & (pl.col("patients_with_event_2014") >= ne) & (pl.col("patients_with_event_2015") >= ne))
            elig[f"patients>={np_}, patients_with_event>={ne} each year"] = {
                "hospitals": int(ok.height), "patients": int(ok["patients_2014"].sum() + ok["patients_2015"].sum())}
    # does patientunitstayid grow with the year within a hospital? AUC of the id for 2015 vs 2014
    mono = []
    for (h,), g in df.unique("stay_id").group_by(["H"]):
        yr = g["year"].to_numpy()
        if (yr == 2014).sum() >= MIN_CELL and (yr == 2015).sum() >= MIN_CELL:
            ids = g["stay_n"].to_numpy().astype(float)
            r = ids.argsort().argsort() + 1.0
            n1, n0 = (yr == 2015).sum(), (yr == 2014).sum()
            mono.append((r[yr == 2015].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    mono = np.array(mono)
    rows = [r for r in table(hy)]
    return {"years": table(counts(df, ["year"])), "missing_year": int(df["year"].null_count()),
            "hospitals_total": int(allh.height), "hospital_year": rows, "eligibility": elig,
            "id_order": {"hospitals": int(mono.size), "auc_median": float(np.median(mono)) if mono.size else None,
                         "auc_q10": float(np.quantile(mono, 0.1)) if mono.size else None,
                         "share_auc_gt_0.9": float(np.mean(mono > 0.9)) if mono.size else None},
            "totals": table(counts(df.with_columns(pl.lit("all").alias("cohort")), ["cohort"]))[0]}


# --------------------------------------------------------------------------- T2 (secondary, descriptive)
def t2(cfg: dict) -> dict:
    """T2's analysis stays (its frozen targets: one row per ICU discharge) per MIMIC-IV block, eICU year and
    hospital, and cohort: stays, patients and deaths (T2's fourth class). For eICU, how many of `hypo`'s eligible
    hospitals (≥ 150 patients with landmarks in each year) also have enough T2 stays."""
    root = work() / "experiments" / "t2" / "derived" / "main"
    out = {}
    for c in ("inspire", "mimic", "eicu"):
        t = pl.read_parquet(root / c / "targets.parquet", columns=["stay_id", "subject_id", "survivor"])
        s = pl.read_parquet(work() / "common" / c / "stays_index.parquet", columns=["stay_id", "year", "year_group"])
        d = t.join(s, on="stay_id", how="left").with_columns((~pl.col("survivor")).cast(pl.Int64).alias("death"))
        d = d.with_columns(pl.col("stay_id").cast(pl.Utf8).str.extract(r"(\d+)$").cast(pl.Int64).alias("stay_n"))

        def cnt(df: pl.DataFrame, by: list[str]) -> list[dict]:
            a = df.group_by(by).agg(pl.len().alias("stays"), pl.col("subject_id").n_unique().alias("patients"),
                                    pl.col("death").sum().alias("deaths")).sort(by)
            return [{k: (cell(v) if k in ("stays", "patients", "deaths") else v) for k, v in r.items()}
                    for r in a.to_dicts()]

        r = {"totals": cnt(d.with_columns(pl.lit("all").alias("cohort")), ["cohort"])[0]}
        if c == "mimic":
            r["blocks"] = cnt(d, ["year_group"])
        if c == "eicu":
            lm = landmarks("eicu")
            lab = {str(h): v for h, v in eicu_labels(cfg, lm).items()}
            d = d.drop("year").join(eicu_patient(), on="stay_n", how="left") \
                .with_columns(pl.col("hospitalid").cast(pl.Utf8).replace_strict(lab, default=None).alias("H"))
            r["years"] = cnt(d, ["year"])
            r["stays_without_hypo_label"] = int(d["H"].null_count())
            hy = d.filter(pl.col("H").is_not_null()).group_by(["H", "year"]).agg(pl.len().alias("n"))
            wide = hy.pivot(on="year", index="H", values="n").fill_null(0)
            lmp = lm.join(eicu_patient(), on="stay_n", how="left").with_columns(
                pl.col("hospital").replace_strict(eicu_labels(cfg, lm), return_dtype=pl.Utf8).alias("H"))
            hp = lmp.group_by(["H", "year"]).agg(pl.col("subject_id").n_unique().alias("p")) \
                .pivot(on="year", index="H", values="p").fill_null(0)
            elig = hp.filter((pl.col("2014") >= 150) & (pl.col("2015") >= 150))["H"].to_list()
            w = wide.filter(pl.col("H").is_in(elig))
            r["hypo_eligible_hospitals"] = {
                "hospitals": len(elig),
                "t2_stays_per_year_median": float(np.median(np.concatenate([w["2014"].to_numpy(), w["2015"].to_numpy()]))),
                "t2_stays_per_year_min": int(min(w["2014"].min(), w["2015"].min())),
                **{f"with_t2_stays>={n}_each_year": int(w.filter((pl.col("2014") >= n) & (pl.col("2015") >= n)).height)
                   for n in (100, 150)}}
        out[c] = r
    return out


# --------------------------------------------------------------------------- INSPIRE
def inspire() -> dict:
    src = (data_root() / "inspire" / "operations.csv.gz").as_posix()
    con = duckdb.connect()
    r = con.execute(f"""SELECT count(*), count(DISTINCT opdate), min(admission_time), max(admission_time),
        corr(op_id, opdate), corr(subject_id, opdate), corr(hadm_id, opdate),
        approx_quantile(opdate, [0.5, 0.9]) FROM read_csv_auto('{src}')""").fetchone()
    lm = landmarks("inspire")
    return {"operations": int(r[0]), "opdate_distinct": int(r[1]), "admission_time_min": float(r[2]),
            "admission_time_max": float(r[3]), "corr_opdate_with": {"op_id": r[4], "subject_id": r[5], "hadm_id": r[6]},
            "opdate_q50_q90_min": [float(x) for x in r[7]],
            "reading": "opdate and every time are minutes from each patient's first admission (admission_time 0); "
                       "identifiers are random: no field orders the patients in calendar time",
            "totals": table(counts(lm.with_columns(pl.lit("all").alias("cohort")), ["cohort"]))[0]}


# --------------------------------------------------------------------------- windows and the admission gate
def window_precision(c: str) -> list[dict]:
    """AUROC of hypo's transported PERISIM in random windows of W patients of the external site's test rows."""
    from sklearn.metrics import roc_auc_score

    ext = hypo_workdir() / "external"
    fs = sorted(ext.glob(f"transported_{c}_f*.parquet"))
    T = pl.concat([pl.read_parquet(f, columns=["row", "p"]) for f in fs]).sort("row")
    cfg = load_hypo()
    lm = load_lm(c, cfg)
    rows = T["row"].to_numpy()
    y, subj, p = lm.y[rows], lm.ft.subjects[rows].astype(str), T["p"].to_numpy()
    uniq, inv = np.unique(subj, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1, len(inv)]
    rng = np.random.default_rng(20261030)
    out = []
    for W in WINDOWS:
        if W > uniq.size:
            continue
        aucs, ev, lms = [], [], []
        for _ in range(N_DRAWS):
            pick = rng.choice(uniq.size, size=W, replace=False)
            idx = np.concatenate([order[starts[i]:starts[i + 1]] for i in pick])
            if 0 < y[idx].sum() < idx.size:
                aucs.append(roc_auc_score(y[idx], p[idx]))
            ev.append(int(y[idx].sum()))
            lms.append(idx.size)
        a = np.array(aucs)
        out.append({"W_patients": W, "landmarks_mean": float(np.mean(lms)), "event_landmarks_mean": float(np.mean(ev)),
                    "auroc_sd": float(a.std(ddof=1)), "auroc_q025_q975": [float(np.quantile(a, 0.025)),
                                                                         float(np.quantile(a, 0.975))],
                    "half_width_95": float((np.quantile(a, 0.975) - np.quantile(a, 0.025)) / 2)})
    return out


def citl_cluster(p: np.ndarray, y: np.ndarray, g: np.ndarray, iters: int = 30) -> tuple[float, float]:
    """Calibration in the large (intercept with logit(p) as offset) and its patient-cluster robust SE."""
    off = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    a = 0.0
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(a + off)))
        step = float((y - mu).sum()) / (float((mu * (1 - mu)).sum()) + 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(a + off)))
    score = np.bincount(g, weights=y - mu)
    return a, float(np.sqrt((score ** 2).sum()) / (mu * (1 - mu)).sum())


def monitor_precision(c: str) -> list[dict]:
    """Added in Phase 1: how often each proposed trigger of the realistic ML arm fires on noise alone. Random
    monitoring windows of M patients (no drift by construction) of hypo's transported PERISIM predictions, which
    are calibrated in every site (hypo H2): the spread of O/E and CITL and the share of windows in which each rule
    would fire. Baselines (AUROC, log-loss) are the whole site's."""
    from sklearn.metrics import roc_auc_score

    ext = hypo_workdir() / "external"
    T = pl.concat([pl.read_parquet(f, columns=["row", "p"]) for f in sorted(ext.glob(f"transported_{c}_f*.parquet"))]).sort("row")
    lm = load_lm(c, load_hypo())
    rows = T["row"].to_numpy()
    y, subj, p = lm.y[rows].astype(float), lm.ft.subjects[rows].astype(str), T["p"].to_numpy()
    uniq, inv = np.unique(subj, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1, len(inv)]
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    ll_all = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    auc_all = float(roc_auc_score(y, p))
    rng = np.random.default_rng(20261031)
    out = []
    for M in (50, 100, 200, 300, 500, 1000):
        r = {k: [] for k in ("oe", "citl", "se", "auc", "ll", "ev")}
        for _ in range(N_DRAWS):
            pick = rng.choice(uniq.size, size=M, replace=False)
            idx = np.concatenate([order[starts[i]:starts[i + 1]] for i in pick])
            yy, pp = y[idx], p[idx]
            a, se = citl_cluster(pp, yy, np.unique(inv[idx], return_inverse=True)[1])
            r["oe"].append(yy.sum() / pp.sum())
            r["citl"].append(a)
            r["se"].append(se)
            r["auc"].append(roc_auc_score(yy, pp) if 0 < yy.sum() < yy.size else np.nan)
            q = np.clip(pp, 1e-12, 1 - 1e-12)
            r["ll"].append(float(-(yy * np.log(q) + (1 - yy) * np.log(1 - q)).mean()))
            r["ev"].append(yy.sum())
        oe, ci, se, auc, ll = (np.array(r[k]) for k in ("oe", "citl", "se", "auc", "ll"))
        lo, hi = ci - 1.96 * se, ci + 1.96 * se
        out.append({"M_patients": M, "event_landmarks_mean": float(np.mean(r["ev"])),
                    "oe_sd": float(oe.std(ddof=1)), "citl_sd": float(ci.std(ddof=1)), "citl_se_mean": float(se.mean()),
                    "fires": {"oe_outside_0.9_1.1": float(np.mean((oe < 0.9) | (oe > 1.1))),
                              "citl_ci_excludes_band_0.10": float(np.mean((lo > 0.10) | (hi < -0.10))),
                              "citl_ci_not_inside_band_0.10": float(np.mean((lo < -0.10) | (hi > 0.10))),
                              "auroc_drop_5pct": float(np.nanmean(auc < 0.95 * auc_all)),
                              "auroc_drop_10pct": float(np.nanmean(auc < 0.90 * auc_all)),
                              "logloss_rise_10pct": float(np.mean(ll > 1.10 * ll_all))}})
    return out


def s1_calibration(c: str) -> list[dict]:
    """Added before the pre-registration (lead, 2026-10-30): the calibration of `hypo`'s S1 case bases of n random
    patients (external stage, every fold, every draw), a new aggregate of examined predictions. Per n: O/E, CITL and
    calibration slope over the 5 folds × draws (mean and range), against the whole training pool."""
    ext = hypo_workdir() / "external"
    lm = load_lm(c, load_hypo())
    y_all = lm.y.astype(np.float64)
    out: dict = {}
    for f in range(5):
        cur = pl.read_parquet(ext / f"curve_{c}_f{f}.parquet")
        tr = pl.read_parquet(ext / f"transported_{c}_f{f}.parquet", columns=["row", "p"])
        if not np.array_equal(cur["row"].to_numpy(), tr["row"].to_numpy()):
            raise RuntimeError(f"curve and transported rows differ: {c} fold {f}")
        rows = cur["row"].to_numpy()
        y, code = y_all[rows], np.unique(lm.ft.subjects[rows].astype(str), return_inverse=True)[1]
        cols = {k: cur[k].to_numpy() for k in cur.columns if k.startswith("p_n")}
        cols["p_pool"] = tr["p"].to_numpy()
        for k, p in cols.items():
            n = "pool" if k == "p_pool" else k.split("_")[1][1:]
            a, se = citl_cluster(p, y, code)
            lo = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
            b = np.array([0.0, 1.0])
            X = np.column_stack([np.ones_like(lo), lo])
            for _ in range(50):
                mu = 1 / (1 + np.exp(-(X @ b)))
                step = np.linalg.solve(X.T @ (X * (mu * (1 - mu) + 1e-9)[:, None]) + 1e-8 * np.eye(2), X.T @ (y - mu))
                b += step
                if np.abs(step).max() < 1e-8:
                    break
            out.setdefault(n, []).append({"oe": float(y.sum() / p.sum()), "citl": a, "slope": float(b[1])})
    res = []
    for n in sorted(out, key=lambda x: float("inf") if x == "pool" else int(x)):
        v = out[n]
        res.append({"n_patients": n, "case_bases": len(v),
                    **{f"{m}_mean": float(np.mean([x[m] for x in v])) for m in ("oe", "citl", "slope")},
                    **{f"{m}_range": [float(np.min([x[m] for x in v])), float(np.max([x[m] for x in v]))]
                       for m in ("oe", "citl", "slope")}})
    return res


def admission_gate(c: str, cfg: dict) -> dict:
    """What the admission gate would refuse among the analysis landmarks: duplicates (stay, L), values outside
    `hypo`'s bounds, impossible values (negative L or ages, MAP summaries out of order)."""
    df = pl.read_parquet(hypo_workdir() / "landmarks" / f"{c}.parquet").filter(pl.col("normotensive") & pl.col("known"))
    b = cfg["bounds"]
    col = {"map": ["map_last", "map_mean_6h", "map_min_6h"], "hr": ["hr_last"], "spo2": ["spo2_last", "spo2_min_6h"],
           "rr": ["rr_last"], "temp": ["temp_last"], "hb": ["hb_last"], "creatinine": ["creatinine_last"],
           "lactate": ["lactate_last"]}
    out_of = {}
    for k, cols in col.items():
        lo, hi = b[k]
        for x in cols:
            if x in df.columns:
                out_of[x] = cell(int(df.filter((pl.col(x) < lo) | (pl.col(x) > hi)).height))
    dup = df.height - df.unique(["stay_id", "L"]).height
    imp = df.filter((pl.col("L") < 0) | (pl.col("age") < 18) | (pl.col("map_min_6h") > pl.col("map_mean_6h") + 1e-9)).height
    return {"landmarks": df.height, "duplicates": cell(dup), "outside_bounds": out_of, "impossible": cell(imp)}


def render(r: dict) -> str:
    L = ["# roll — Phase 0 feasibility (aggregates)", "", f"Freeze check: {r['freeze_start']}", ""]
    for c in ("mimic", "eicu", "inspire"):
        L += [f"## {c}", "", "```json", json.dumps(r[c], indent=1, default=str), "```", ""]
    L += ["## Window precision (AUROC of hypo's transported PERISIM in random windows of W patients)", ""]
    for c, rows in r["windows"].items():
        L += [f"### {c}", "", "| W patients | landmarks | event landmarks | AUROC SD | 2.5–97.5 % | half-width |",
              "|---|---|---|---|---|---|"]
        L += [f"| {w['W_patients']} | {w['landmarks_mean']:.0f} | {w['event_landmarks_mean']:.0f} | {w['auroc_sd']:.3f} | "
              f"{w['auroc_q025_q975'][0]:.3f}–{w['auroc_q025_q975'][1]:.3f} | {w['half_width_95']:.3f} |" for w in rows]
        L.append("")
    L += ["## Admission gate", "", "```json", json.dumps(r["gate"], indent=1, default=str), "```"]
    return "\n".join(L)


def main() -> int:
    t = time.perf_counter()
    start = frozen_guard()
    cfg = load_hypo()
    if sys.argv[1:] == ["s1cal"]:                              # added before the pre-registration (lead)
        r = {"freeze_start": start, "s1_calibration": {c: s1_calibration(c) for c in ("inspire", "mimic", "eicu")}}
        r["freeze_end"] = frozen_guard()
        out_dir().mkdir(parents=True, exist_ok=True)
        (out_dir() / "feasibility_roll_s1cal.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
        print(json.dumps(r, indent=1, default=str))
        return 0
    if sys.argv[1:] == ["phase1"]:                             # added in Phase 1: T2's counts, the monitor's noise
        r = {"freeze_start": start, "t2": t2(cfg),
             "monitor": {c: monitor_precision(c) for c in ("mimic", "eicu", "inspire")}}
        r["freeze_end"] = frozen_guard()
        out_dir().mkdir(parents=True, exist_ok=True)
        (out_dir() / "feasibility_roll_phase1.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
        print(json.dumps(r, indent=1, default=str))
        return 0
    r = {"freeze_start": start, "mimic": mimic(), "eicu": eicu(cfg), "inspire": inspire(),
         "windows": {c: window_precision(c) for c in ("mimic", "eicu", "inspire")},
         "gate": {c: admission_gate(c, cfg) for c in ("mimic", "eicu", "inspire")}}
    r["freeze_end"] = frozen_guard()
    r["seconds"] = time.perf_counter() - t
    out_dir().mkdir(parents=True, exist_ok=True)
    (out_dir() / "feasibility_roll.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    (out_dir() / "feasibility_roll.md").write_text(render(r), encoding="utf-8")
    print(out_dir() / "feasibility_roll.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
