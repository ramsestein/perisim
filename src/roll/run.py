"""Running the streams and their endpoints, on the real data (Phase 4, frozen version only) or on synthetic data (the
Phase 2 dry run). Every stage checks T2's and `hypo`'s freezes at its start and its end."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from . import endpoints as EP
from .config import REPO, code_hash, config_hash, dry_settings, frozen_guard, load, workdir
from .loop import Run, _json, save
from .streams import build, eligible_hospitals, hospital_labels

LOG = logging.getLogger("roll")


def settings(dry: bool) -> tuple[dict, dict, dict, dict]:
    """(roll cfg, hypo cfg, metric cfg, model space)."""
    from hypo.config import load as load_hypo

    cfg, hcfg = load(), load_hypo()
    if dry:
        from .synthetic import SPACE, quick_hcfg, quick_mcfg

        return dry_settings(cfg), quick_hcfg(hcfg), quick_mcfg(hcfg), SPACE
    with open(REPO / "config" / "model_space.yaml", encoding="utf-8") as fh:
        space = yaml.safe_load(fh)
    return cfg, hcfg, hcfg["metric"], space


def jobs(cfg: dict, cohorts: dict, names: list[str] | None = None, primary_only: bool = False) -> list[dict]:
    """Every (stream, hospital, N) of the design, in a fixed order."""
    out = []
    eh = cfg["eicu_hospitals"]
    hosp = eligible_hospitals(cohorts["eicu"], list(eh["years"]), int(eh["min_patients_each_year"])) \
        if "eicu" in cohorts else []
    lab = hospital_labels(cohorts["eicu"]) if "eicu" in cohorts else {}
    for name, spec in cfg["streams"].items():
        if names and name not in names:
            continue
        Ns = [int(spec["N_primary"])] if primary_only else [int(n) for n in spec["N"]]
        for h in (hosp if spec["kind"] in ("within", "deploy") else [""]):
            for N in Ns:
                if spec["kind"] == "within" and N > int(spec["N_primary"]):
                    c = cohorts["eicu"]
                    first = int(((c.hospital == h) & (c.year == int(eh["years"][0]))).sum())
                    if first < int(eh["n250_min_first_year"]):
                        continue
                out.append({"stream": name, "spec": spec, "hospital": h, "label": lab.get(h, "all"), "N": N,
                            "orders": int(spec["orders"])})
    return out


def run_dir(root: Path, job: dict, order: int) -> Path:
    return root / job["stream"] / job["label"] / f"N{job['N']}" / f"o{order}"


def run_job(job: dict, order: int, cohorts: dict, frozen, cfg: dict, hcfg: dict, mcfg: dict, space: dict,
            root: Path, threads: int) -> dict | None:
    out = run_dir(root, job, order)
    if (out / "log.json").exists():
        return json.loads((out / "log.json").read_text(encoding="utf-8"))
    st = build(job["stream"], job["spec"], cohorts, cfg, order, job["N"], job["hospital"])
    if st is None:
        LOG.info("%s %s N=%d: too few patients, skipped", job["stream"], job["label"], job["N"])
        return None
    st.make_windows(cfg["windows"]["W"])
    rd = cfg["diagnostics"]["redeform"]
    red = job["stream"] in rd["streams"] and order in rd["orders"] and job["N"] == int(job["spec"]["N_primary"])
    run = Run(st, frozen, cfg, mcfg, space, threads=threads, redeform=red, hcfg=hcfg)
    res = run.execute()
    res["stream"]["label"] = job["label"]
    res["code_hash"] = code_hash()
    save(run, res, out)
    LOG.info("%s %s N=%d order %d: %.0f s", job["stream"], job["label"], job["N"], order, res["seconds"])
    return res


def endpoints(root: Path, cfg: dict, js: list[dict], seed: int) -> dict:
    """Endpoints of every (stream, hospital, N), then the eICU pooling and H6."""
    out: dict = {"streams": {}, "eicu": {}}
    ml5 = f"{cfg['ml']['primary_family']}_realistic"
    for job in js:
        runs = []
        for r in range(job["orders"]):
            d = run_dir(root, job, r)
            if (d / "log.json").exists():
                runs.append((pl.read_parquet(d / "predictions.parquet"),
                             json.loads((d / "log.json").read_text(encoding="utf-8"))))
        if not runs:
            continue
        key = f"{job['stream']}|{job['label']}|N{job['N']}"
        res = EP.stream_endpoints(runs, cfg, seed)
        if job["spec"]["kind"] == "within" and job["N"] == int(job["spec"]["N_primary"]):
            res["h6"] = EP.h6_hospital(runs, cfg, ml5, seed)
        out["streams"][key] = res
    for name, spec in cfg["streams"].items():
        if spec["kind"] not in ("within", "deploy"):
            continue
        for N in [int(n) for n in spec["N"]]:
            per = {k.split("|")[1]: v for k, v in out["streams"].items() if k.startswith(f"{name}|") and k.endswith(f"|N{N}")}
            if not per:
                continue
            comps = list(next(iter(per.values()))["contrasts"])
            out["eicu"][f"{name}|N{N}"] = {"hospitals": len(per), "pooled": {c: EP.pool_hospitals(per, c) for c in comps}}
            if spec["kind"] == "within" and N == int(spec["N_primary"]):
                h6 = {h: v["h6"] for h, v in per.items() if "h6" in v}
                out["eicu"][f"{name}|N{N}"]["h6"] = EP.pool_h6(h6, int(cfg["endpoints"]["boot"]), seed)
    return out


# --------------------------------------------------------------------------- the dry run
def dry(threads: int = 4, names: list[str] | None = None) -> Path:
    """Phase 2: every stream kind, every arm, trigger and endpoint on synthetic data (never an analysis)."""
    from . import synthetic

    t0 = time.perf_counter()
    start = frozen_guard()
    cfg, hcfg, mcfg, space = settings(True)
    root = workdir() / "dryrun"
    root.mkdir(parents=True, exist_ok=True)
    cs = synthetic.cohorts(cfg, hcfg)
    fz = synthetic.frozen(cs, hcfg, cfg["ml"]["families"], threads=threads)
    js = jobs(cfg, cs, names)
    hosp = sorted({j["label"] for j in js if j["label"] != "all"}, key=lambda x: int(x[1:]))[:2]
    js = [j for j in js if j["label"] == "all" or j["label"] in hosp]
    runs = {}
    for j in js:
        for r in range(j["orders"]):
            res = run_job(j, r, cs, fz, cfg, hcfg, mcfg, space, root / "runs", threads)
            if res is not None:
                runs[f"{j['stream']}|{j['label']}|N{j['N']}|o{r}"] = res
    # determinism: the first stream again, from scratch, with one thread
    j0 = js[0]
    from .streams import build as build_stream

    st = build_stream(j0["stream"], j0["spec"], cs, cfg, 0, j0["N"], j0["hospital"])
    st.make_windows(cfg["windows"]["W"])
    rerun = Run(st, fz, cfg, mcfg, space, threads=1).execute()
    first = runs[f"{j0['stream']}|{j0['label']}|N{j0['N']}|o0"]
    det = {"stream": j0["stream"], "digest_threads": first["predictions_digest"], "digest_one_thread":
           rerun["predictions_digest"], "identical": first["predictions_digest"] == rerun["predictions_digest"]}
    ep = endpoints(root / "runs", cfg, js, int(cfg["seed"]))
    end = frozen_guard()
    summary = {"note": "Phase 2 dry run on SYNTHETIC data: tests the code, never an analysis.",
               "freeze_start": start, "freeze_end": end, "code_hash": code_hash(), "config_hash": config_hash(cfg),
               "cohorts": {c: v.info for c, v in cs.items()}, "determinism": det,
               "runs": {k: {"seconds": v["seconds"], "evaluated": v["stream"]["evaluated"],
                            "windows": v["stream"]["windows"], "digest": v["predictions_digest"],
                            "triggers": {a: sum(1 for e in lg if e.get("event") == "trigger") for a, lg in v["arms"].items()},
                            "deployments": {a: sum(1 for e in lg if e.get("event") == "deploy") for a, lg in v["arms"].items()}}
                        for k, v in runs.items()},
               "seconds": time.perf_counter() - t0}
    (root / "endpoints.json").write_text(json.dumps(ep, indent=1, default=_json), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=_json), encoding="utf-8")
    (root / "summary.md").write_text(render_dry(summary, ep), encoding="utf-8")
    return root


def render_dry(s: dict, ep: dict) -> str:
    L = ["# roll — Phase 2 dry run (synthetic data; tests the code, never an analysis)", "",
         f"Code {s['code_hash']}, configuration {s['config_hash']}. Freezes of T2 and hypo unchanged at the start and "
         f"the end: {s['freeze_start'] == s['freeze_end']}. {s['seconds']:.0f} s.", "",
         f"Determinism (the first stream again with one thread): identical = {s['determinism']['identical']}.", "",
         "| run | evaluated | windows | seconds | triggers (LightGBM realistic) | deployments (LightGBM ideal) |",
         "|---|---|---|---|---|---|"]
    for k, v in s["runs"].items():
        L.append(f"| {k} | {v['evaluated']} | {v['windows']} | {v['seconds']:.0f} | "
                 f"{v['triggers'].get('lightgbm_realistic', 0)} | {v['deployments'].get('lightgbm_ideal', 0)} |")
    L += ["", "Every endpoint was computed (endpoints.json): "
          f"{len(ep['streams'])} stream units, {len(ep['eicu'])} eICU poolings."]
    return "\n".join(L) + "\n"
