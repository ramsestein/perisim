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

Usage:  python scripts/feasibility_roll.py        (set PERISIM_WORKDIR / PERISIM_DATASETS when run from a worktree)
Output: <workdir>/experiments/roll/feasibility/feasibility_roll.json and .md
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
def eicu(cfg: dict) -> dict:
    src = data_root() / "eicu"
    lm = landmarks("eicu")
    con = duckdb.connect()
    p = con.execute(f"SELECT patientunitstayid AS stay_id, hospitalid, hospitaldischargeyear AS year FROM read_csv_auto('{(src / 'patient.csv.gz').as_posix()}')").pl()
    df = lm.join(p.rename({"stay_id": "stay_n"}), on="stay_n", how="left")
    hl = load_lm("eicu", cfg)
    labels = hospital_labels(hl, int(cfg["external"]["hospitals"]["min_patients"]))
    allh = df.group_by("hospital").agg(pl.col("subject_id").n_unique().alias("p")).sort("p", descending=True)
    lab_all = {h: labels.get(str(h)) for h in allh["hospital"].to_list()}
    k = len(labels)
    for h in allh["hospital"].to_list():                      # hospitals below hypo's threshold: labelled after
        if lab_all[h] is None:
            k += 1
            lab_all[h] = f"H{k}"
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
