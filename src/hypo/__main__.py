"""Command line of the `hypo` experiment.

  python -m hypo landmarks [--cohorts inspire mimic eicu] [--threads 4] [--force-extract eicu ...]
  python -m hypo audit
  python -m hypo smoke --external eicu [--quick] [--threads 8]
  python -m hypo cv [--external inspire mimic eicu] [--methods perisim ...] [--folds 0 1] [--quick] [--threads 16]
  python -m hypo cv-report [--quick]

`landmarks` builds the landmark table of each cohort from its original tables: <workdir>/landmarks/<cohort>.parquet
(patient level, git-ignored) and an aggregate summary <workdir>/reports/landmarks_summary.md / .json; then runs
`audit`, the completeness audit of the metric's inputs (<workdir>/reports/audit.md / .json), the gate before the
model. `smoke` runs the whole model (metric, case bases, predictions) on the development sites of one rotation with
a plain patient split: mechanics and cost (Phase 1). `cv` is the internal validation by rotation (Phase 2,
resumable: finished units are kept) and ends with `cv-report`. Aggregates only: shares, quantiles, counts ≥ 10.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import polars as pl

from .config import code_hash, dataset_dirs, load, workdir
from .extract import extract, stays
from .landmarks import LABS, SUPPORTS, VITALS, build, hourly, support_hours

LOG = logging.getLogger("hypo")
MIN_N = 10
INPUTS = (["map_last", "map_mean_6h", "map_min_6h", "map_sd_6h", "map_slope_6h", "hr_slope_6h", "spo2_min_6h"]
          + [f"{v}_last" for v in VITALS] + [f"{v}_last" for v in LABS]
          + ["age", "sex_male", "bmi", "emergency", "surgical_group", "surgery_duration_h"])


def _f(x) -> float:
    return float("nan") if x is None else float(x)


def summary(c: str, st: pl.DataFrame, lm: pl.DataFrame) -> dict:
    a = lm.filter(pl.col("normotensive") & pl.col("known"))
    out = {"stays": st.height, "patients": int(st["subject_id"].n_unique()),
           "landmarks_all": lm.height, "landmarks": a.height, "patients_with_landmarks": int(a["subject_id"].n_unique()),
           "unknown_window": float(1 - lm.filter(pl.col("normotensive"))["known"].mean()),
           "y": float(a["y"].mean()), "y_first_reading": float(a["y_first"].mean()),
           "band": {str(b): float((a["band"] == b).mean()) for b in (0, 1, 2)},
           "vaso_start": _f(a.filter(~pl.col("vaso_on").fill_null(True))["vaso_start"].mean()),
           "art_line": _f(a["art_line"].fill_null(False).mean()),
           **{f"{k}_on": _f(a[f"{k}_on"].mean()) for k in SUPPORTS},
           **{f"{k}_24h": _f(a[f"{k}_24h"].mean()) for k in ("vaso", "sedation")},
           "y_by_art_line": {str(k): float(g["y"].mean()) for (k,), g in a.group_by("art_line") if g.height >= MIN_N},
           "inputs_available": {v: float(a[v].is_not_null().mean()) for v in INPUTS if v in a.columns}}
    if "hospital" in a.columns:
        per = a.drop_nulls("hospital").group_by("hospital").agg(pl.col("stay_id").n_unique().alias("p"))
        out["hospitals"] = {"with_landmarks": per.height, "≥ 300 patients": int((per["p"] >= 300).sum()),
                            "≥ 1000 patients": int((per["p"] >= 1000).sum())}
    return out


def render(res: dict, meta: dict) -> str:
    C = list(res)
    L = ["# Experiment `hypo`: landmark tables", "",
         f"Built {meta['time']} · code {meta['code_hash']} · config `config/hypo.yaml`. Analysis population: "
         "normotensive in the last hour, window observable. Shares; counts ≥ 10.", "",
         "| | " + " | ".join(C) + " |", "|---|" + "---|" * len(C)]

    def row(lab, f):
        L.append(f"| {lab} | " + " | ".join(f(res[c]) for c in C) + " |")

    row("stays (patients)", lambda r: f"{r['stays']:,} ({r['patients']:,})")
    row("landmarks (patients)", lambda r: f"{r['landmarks']:,} ({r['patients_with_landmarks']:,})")
    row("window not observable", lambda r: f"{r['unknown_window']:.1%}")
    row("**hypotension in the next 6 h**", lambda r: f"**{r['y']:.1%}**")
    row("… seen through one reading per hour", lambda r: f"{r['y_first_reading']:.1%}")
    row("lowest MAP < 65 / 65–75 / ≥ 75", lambda r: " / ".join(f"{r['band'][b]:.0%}" for b in ("0", "1", "2")))
    row("vasopressor started (off at L)", lambda r: f"{r['vaso_start']:.1%}")
    for k in SUPPORTS:
        row(f"{k} running at L", lambda r, k=k: f"{r[f'{k}_on']:.1%}")
    for k in ("vaso", "sedation"):
        row(f"{k} in the last 24 h", lambda r, k=k: f"{r[f'{k}_24h']:.1%}")
    row("arterial line in the last 6 h", lambda r: f"{r['art_line']:.0%}")
    L += ["", "Inputs available at the landmark:", "", "| input | " + " | ".join(C) + " |", "|---|" + "---|" * len(C)]
    for v in INPUTS:
        L.append(f"| {v} | " + " | ".join(f"{res[c]['inputs_available'].get(v, 0.0):.0%}" for c in C) + " |")
    for c in C:
        if "hospitals" in res[c]:
            H = res[c]["hospitals"]
            L += ["", f"{c}: {H['with_landmarks']} hospitals, {H['≥ 300 patients']} with ≥ 300 patients, "
                      f"{H['≥ 1000 patients']} with ≥ 1,000."]
    return "\n".join(L) + "\n"


MIN_PREVALENCE = 0.10                 # T2: a class this rare fails the kNN
MIN_COVERAGE = 0.50                   # below this an input must be a declared (source) missing value


def audit(cohorts: list[str] | None = None) -> dict:
    """Completeness audit of the metric's inputs on the analysis population of each cohort, and the gate."""
    cfg = load()
    wd = workdir()
    cohorts = cohorts or [c for c in cfg["cohorts"] if (wd / "landmarks" / f"{c}.parquet").exists()]
    spec = cfg["model"]["inputs"]
    src_missing = cfg["model"].get("source_missing") or {}
    meas_gaps = cfg["model"].get("measurement_gaps") or {}
    res, flags = {}, []
    for c in cohorts:
        a = pl.read_parquet(wd / "landmarks" / f"{c}.parquet").filter(pl.col("normotensive") & pl.col("known"))
        r = {"landmarks": a.height, "patients": int(a["subject_id"].n_unique()), "prevalence": float(a["y"].mean()),
             "inputs": {}}
        if r["prevalence"] < MIN_PREVALENCE:
            flags.append(f"{c}: outcome prevalence {r['prevalence']:.1%} < {MIN_PREVALENCE:.0%}")
        for kind in ("continuous", "binary", "nominal"):
            for v in spec[kind]:
                if v not in a.columns:
                    flags.append(f"{c}: input {v} not built")
                    continue
                col = a[v]
                cov = float(col.is_not_null().mean())
                d = {"kind": kind, "coverage": cov, "missing_type": "source" if v in src_missing.get(c, []) else
                     "measurement (declared)" if v in meas_gaps.get(c, []) else ("measurement" if cov < 1 else None)}
                if kind == "continuous" and cov > 0:
                    q = col.drop_nulls().cast(pl.Float64)
                    d["quantiles"] = [round(float(q.quantile(x)), 2) for x in (0.25, 0.5, 0.75)]
                elif kind == "binary" and cov > 0:
                    d["share_true"] = float(col.drop_nulls().cast(pl.Float64).mean())
                elif kind == "nominal":
                    vc = (col.fill_null("(missing)").value_counts().filter(pl.col("count") >= MIN_N)
                          .sort("count", descending=True))
                    d["shares"] = {str(x[v]): round(x["count"] / a.height, 3) for x in vc.iter_rows(named=True)}
                if cov < MIN_COVERAGE and d["missing_type"] not in ("source", "measurement (declared)"):
                    flags.append(f"{c}: {v} coverage {cov:.0%} (< {MIN_COVERAGE:.0%}, not a declared gap)")
                r["inputs"][v] = d
        if "hospital" in a.columns:
            per = a.drop_nulls("hospital").group_by("hospital").agg(pl.col("subject_id").n_unique().alias("p"))
            r["hospitals_300"] = int((per["p"] >= 300).sum())
        res[c] = r
    out = {"cohorts": res, "flags": flags, "gate": "pass" if not flags else "review"}
    (wd / "reports").mkdir(parents=True, exist_ok=True)
    (wd / "reports" / "audit.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (wd / "reports" / "audit.md").write_text(render_audit(out), encoding="utf-8")
    LOG.info("hypo: audit %s (%d flags) in %s", out["gate"], len(flags), wd / "reports" / "audit.md")
    return out


def render_audit(out: dict) -> str:
    R = out["cohorts"]
    C = list(R)
    L = ["# Experiment `hypo`: completeness audit of the metric's inputs", "",
         f"Gate: **{out['gate']}**. Population: normotensive in the last hour, window observable. Missing type: "
         "`source` = the cohort cannot measure it; `measurement (declared)` = measured only for some patients by "
         "practice (both declared in config/hypo.yaml); otherwise not measured for some landmarks. Quantiles "
         "p25 / p50 / p75.", "",
         "| | " + " | ".join(C) + " |", "|---|" + "---|" * len(C),
         "| landmarks (patients) | " + " | ".join(f"{R[c]['landmarks']:,} ({R[c]['patients']:,})" for c in C) + " |",
         "| hypotension in the next 6 h | " + " | ".join(f"{R[c]['prevalence']:.1%}" for c in C) + " |", "",
         "| input | " + " | ".join(C) + " |", "|---|" + "---|" * len(C)]
    names = list(R[C[0]]["inputs"]) if C else []
    for v in names:
        cells = []
        for c in C:
            d = R[c]["inputs"].get(v)
            if d is None:
                cells.append("—")
                continue
            s = f"{d['coverage']:.0%}"
            if d["missing_type"] in ("source", "measurement (declared)"):
                s += f" ({d['missing_type']})"
            if "quantiles" in d:
                s += " · " + " / ".join(f"{x:g}" for x in d["quantiles"])
            if "share_true" in d:
                s += f" · {d['share_true']:.1%} yes"
            if "shares" in d:
                s += " · " + ", ".join(f"{k} {x:.0%}" for k, x in list(d["shares"].items())[:4])
            cells.append(s)
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    L += ["", "Flags:" if out["flags"] else "No flags."] + [f"- {f}" for f in out["flags"]]
    return "\n".join(L) + "\n"


def landmarks(cohorts: list[str], threads: int, force: list[str]) -> int:
    cfg = load()
    wd = workdir()
    src = dataset_dirs()
    res = {}
    for c in cohorts:
        t0 = time.time()
        hof = None
        if c == "eicu":
            from perisim.transport.analyses import hospital_map

            def hof(ids, path=str(src["eicu"])):
                return hospital_map(path, ids, wd / "cache")
        st = stays(c, cfg, hof)
        tabs = extract(c, st, src[c], cfg, wd / "extract", threads=threads, force=c in force)
        lm = build(st, hourly(tabs["vitals"]), tabs["labs"],
                   support_hours(tabs["supports"], cfg["landmarks"].get("support_gap_h")), cfg, cohort=c)
        (wd / "landmarks").mkdir(parents=True, exist_ok=True)
        lm.write_parquet(wd / "landmarks" / f"{c}.parquet")
        res[c] = summary(c, st, lm)
        LOG.info("hypo: %s %d landmarks, hypotension %.1f %% (%.0f s)", c, res[c]["landmarks"], 100 * res[c]["y"],
                 time.time() - t0)
    meta = {"time": time.strftime("%Y-%m-%d %H:%M"), "code_hash": code_hash()}
    (wd / "reports").mkdir(parents=True, exist_ok=True)
    (wd / "reports" / "landmarks_summary.json").write_text(json.dumps({"meta": meta, **res}, indent=1, default=str),
                                                           encoding="utf-8")
    (wd / "reports" / "landmarks_summary.md").write_text(render(res, meta), encoding="utf-8")
    LOG.info("hypo: summary in %s", wd / "reports" / "landmarks_summary.md")
    audit()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypo", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("landmarks", help="build the landmark tables from the original tables")
    p.add_argument("--cohorts", nargs="*", default=load()["cohorts"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--force-extract", nargs="*", default=[], metavar="COHORT",
                   help="re-extract these cohorts from the original tables (default: reuse the cached extracts)")
    sub.add_parser("audit", help="completeness audit of the metric's inputs (the gate before the model)")
    p = sub.add_parser("smoke", help="Phase 1: the whole model on one rotation's development sites")
    p.add_argument("--external", required=True, choices=load()["cohorts"], help="the rotation's external site")
    p.add_argument("--quick", action="store_true", help="8 models and 10 NCA epochs (mechanics only)")
    p.add_argument("--threads", type=int, default=8)
    cfg = load()
    p = sub.add_parser("cv", help="Phase 2: internal validation by rotation (resumable)")
    p.add_argument("--external", nargs="*", default=cfg["cohorts"], choices=cfg["cohorts"],
                   help="rotations, named by their external site")
    p.add_argument("--methods", nargs="*", default=["perisim"] + cfg["cv"]["comparators"])
    p.add_argument("--folds", nargs="*", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="8 models and 10 NCA epochs, in a separate folder")
    p.add_argument("--threads", type=int, default=16)
    p = sub.add_parser("cv-report", help="aggregate report of the internal validation")
    p.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    if a.cmd == "audit":
        audit()
        return 0
    if a.cmd == "smoke":
        from .run import smoke

        smoke(a.external, quick=a.quick, threads=a.threads)
        return 0
    if a.cmd in ("cv", "cv-report"):
        from .report import report

        if a.cmd == "cv":
            from .cv import run

            run(a.external, a.methods, a.folds, quick=a.quick, threads=a.threads)
        report(a.quick)
        return 0
    return landmarks(a.cohorts, a.threads, a.force_extract)


if __name__ == "__main__":
    raise SystemExit(main())
