"""Command line: python -m perisim <command>  (run_pod.sh wraps it in the pod)."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

if platform.system() == "Windows":  # FAISS and PyTorch each ship their own OpenMP on Windows (local development only)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _cfg(args):
    from perisim.config import load_config

    over = {}
    if getattr(args, "workdir", None):
        over = {"experiment": {"paths": {"workdir": str(Path(args.workdir).resolve())}}}
    return load_config(getattr(args, "config_dir", None), getattr(args, "profile", None), over)


def cmd_run(args) -> int:
    from perisim.pipeline import MAIN_STAGES, Pipeline
    from perisim.utils import LOG, now_iso, setup_logging

    cfg = _cfg(args)
    if args.ack_audit:
        cfg.experiment["gates"]["audit_acknowledged"] = True
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(cfg, force=args.force)
    setup_logging(pipe.ws.logs / f"run_{now_iso().replace(':', '')}.log")
    stages = MAIN_STAGES
    if args.stage:
        stages = [s for s in args.stage.split(",") if s]
    else:
        if args.from_stage:
            stages = stages[stages.index(args.from_stage):]
        if args.until:
            stages = stages[: stages.index(args.until) + 1]
    LOG.info("PERISIM · experiment=%s · workdir=%s · stages=%s · profile=%s", cfg.experiment_name, cfg.workdir,
             ",".join(stages), cfg.profile)
    rc = pipe.run(stages)
    if rc == 0:
        LOG.info("Reports: %s · Exportable: %s", pipe.ws.reports, pipe.ws.export)
    return rc


def cmd_status(args) -> int:
    from perisim.pipeline import COHORTS, MAIN_STAGES, STAGE_DEFS, Pipeline
    from perisim.utils import read_json

    cfg = _cfg(args)
    pipe = Pipeline(cfg)
    r = pipe.res
    print(f"experiment: {cfg.experiment_name} (target {', '.join(cfg.targets_to_model)}) · workdir: {cfg.workdir}")
    print(f"resources: {r['cpus']} CPUs, {r['memory_gb']} GB → {r['jobs']} jobs × {r['threads_per_job']} threads")
    for st in MAIN_STAGES:
        if STAGE_DEFS[st].per_cohort:
            s = ", ".join(f"{c}:{'✓' if pipe.done(st, c) else '·'}" for c in COHORTS if c in pipe.available_cohorts())
            print(f"  {st:<12} {s or '—'}")
        else:
            print(f"  {st:<12} {'✓' if pipe.done(st) else '·'}")
    stop = read_json(pipe.ws.state / "last_stop.json")
    if stop:
        print(f"\nLast stop at '{stop['stage']}' ({stop['at']}):\n{stop['message']}")
    err = read_json(pipe.ws.state / "last_error.json")
    if err:
        print(f"\nLast error at '{err['stage']}' ({err['at']}) — see the logs in {pipe.ws.logs}")
    fr = read_json(pipe.ws.state / "freeze.json")
    print(f"\nFreeze (§6.3): {'yes, ' + fr['tag'] if fr else 'no'}")
    return 0


def cmd_env_check(args) -> int:
    import importlib

    bad = []
    for m in ("numpy", "polars", "pyarrow", "scipy", "sklearn", "lightgbm", "xgboost", "torch", "yaml", "psutil",
              "matplotlib"):
        try:
            importlib.import_module(m)
        except Exception as exc:
            bad.append(f"{m}: {exc}")
    try:
        import faiss  # noqa: F401
    except Exception:
        print("[perisim] Warning: faiss not available; the exact numpy search will be used (slower)")
    if sys.version_info < (3, 11):
        bad.append(f"Python {sys.version.split()[0]} < 3.11")
    if bad:
        print("[perisim] Incomplete environment:\n  " + "\n  ".join(bad))
        return 1
    print(f"[perisim] Environment OK (Python {sys.version.split()[0]})")
    return 0


def cmd_code_hash(args) -> int:
    from perisim.config import code_hash

    print(code_hash())
    return 0


def cmd_resources(args) -> int:
    """Container limits and the jobs × threads plan; `--shell` prints exports for run_pod.sh."""
    import json

    from perisim.resources import plan

    cfg = _cfg(args)
    r = plan(cfg.experiment.get("resources"))
    if args.shell:
        print(f"export PERISIM_CPUS={r['cpus']}")
        print(f"export PERISIM_THREADS_PER_JOB={r['threads_per_job']}")
        print(f"export POLARS_MAX_THREADS={r['cpus']}")
    else:
        print(json.dumps(r, indent=2))
    return 0


def cmd_smoke(args) -> int:
    """Synthetic data with the schema of the three cohorts → full pipeline with the `smoke` profile."""
    from perisim.config import REPO_ROOT, load_config
    from perisim.data.synthetic import write_synthetic

    work = Path(args.workdir or Path.cwd() / "work_smoke").resolve()
    data_root = work / "datasets"
    cfg0 = load_config(profile="smoke")
    write_synthetic(data_root, cfg0.harmonization, n_subjects=args.n, seed=args.seed)
    base = [sys.executable, "-m", "perisim"]
    exps = [e for e in (args.experiments or "t2,t1,t3").split(",") if e]
    for exp in exps:   # the three experiments share the extraction; each one freezes its own version
        (work / "workdir" / "experiments" / exp / "state" / "freeze.json").unlink(missing_ok=True)
        env = {**os.environ, "PERISIM_DATASETS": str(data_root), "PERISIM_WORKDIR": str(work / "workdir"),
               "PERISIM_INSPIRE_DIR": "auto", "PERISIM_MIMIC_DIR": "auto", "PERISIM_EICU_DIR": "auto",
               "PERISIM_ACK_AUDIT": "true", "PERISIM_EXPERIMENT": exp,
               "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src")] + [p for p in [os.environ.get("PYTHONPATH")] if p])}
        steps = [base + ["run", "--profile", "smoke"] + (["--force"] if args.force else []),
                 base + ["freeze", "--profile", "smoke", "--no-git", "--registration", "SMOKE-TEST"],
                 base + ["run", "--profile", "smoke", "--stage", "external,export"]]
        for cmd in steps:  # internal pipeline → freeze → external validation (as in the pod, without the pauses)
            print(f"[perisim] smoke [{exp}]:", " ".join(cmd[2:]))
            rc = subprocess.call(cmd, env=env)
            if rc != 0:
                return rc
    print(f"[perisim] smoke OK ({', '.join(exps)}) · reports in {work / 'workdir' / 'experiments'}/<experiment>/reports")
    return 0


def cmd_freeze(args) -> int:
    """§6.3: local git tag, snapshot of the configuration and hashes; enables the external stage."""
    import shutil

    from perisim.config import REPO_ROOT, code_hash
    from perisim.utils import now_iso, write_json
    from perisim.workspace import Workspace

    cfg = _cfg(args)
    ws = Workspace(cfg.workdir, experiment=cfg.experiment_name)
    # one freeze (and pre-registration) per experiment: T2, T1 and T3 are separate experiments (2026-10-24)
    tag = f"perisim-freeze-{cfg.experiment_name}-{now_iso()[:19].replace(':', '').replace('T', '-')}"
    if (ws.artifacts() / "external.json").exists():
        print("[perisim] External results already exist: re-freezing would allow iterating after looking at them "
              "(§6.3). Refused.")
        return 2
    git = {}
    if not args.no_git and (REPO_ROOT / ".git").exists() and shutil.which("git"):
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
        if dirty.strip():
            print("[perisim] The repository has uncommitted changes: commit them (git commit) before freezing.")
            print(dirty)
            return 2
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "-c", "user.name=perisim", "-c", "user.email=perisim@localhost",
                            "tag", "-a", tag, "-m", "PERISIM freeze §6.3"], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[perisim] Could not create the git tag {tag}: {r.stderr.strip()}")
            return 2
        git = {"commit": subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(), "tag": tag}
    else:
        print("[perisim] Warning: no git; the freeze is recorded by hashes only.")
    if not args.registration:
        print("[perisim] Warning: no --registration (OSF); recorded as pending (D9).")
    snap = ws.state / "frozen_config"
    snap.mkdir(parents=True, exist_ok=True)
    for path in cfg.sources.values():
        shutil.copyfile(path, snap / Path(path).name)
    write_json(ws.state / "freeze.json", {"tag": tag, "at": now_iso(), "code_hash": code_hash(),
                                          "config_hash": cfg.scientific_hash(), "git": git,
                                          "registration": args.registration or "PENDING",
                                          "packages": _package_versions()})
    print(f"[perisim] Frozen ({tag}). The external stage can now run: bash run_pod.sh")
    return 0


def _package_versions() -> dict:
    import importlib

    out = {}
    for m in ("numpy", "polars", "pyarrow", "scipy", "sklearn", "lightgbm", "xgboost", "torch", "faiss"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:
            out[m] = None
    return out


def cmd_export(args) -> int:
    from perisim.evaluation.export import build_export
    from perisim.workspace import Workspace

    cfg = _cfg(args)
    print(build_export(Workspace(cfg.workdir, experiment=cfg.experiment_name), cfg))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="perisim", description="PERISIM — automatic pipeline (see SPECS.md)")
    p.add_argument("--config-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline (resumes whatever is pending)")
    r.add_argument("--profile", default=None)
    r.add_argument("--workdir", default=None)
    r.add_argument("--stage", default=None, help="comma-separated list of stages")
    r.add_argument("--from", dest="from_stage", default=None)
    r.add_argument("--until", default=None)
    r.add_argument("--force", action="store_true", help="redo stages even if they are up to date")
    r.add_argument("--ack-audit", action="store_true", help="continue after reviewing the §7 audit")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="stage status")
    s.add_argument("--profile", default=None)
    s.add_argument("--workdir", default=None)
    s.set_defaults(fn=cmd_status)

    sub.add_parser("env-check", help="check that the required packages import").set_defaults(fn=cmd_env_check)
    sub.add_parser("code-hash", help="hash of the source code").set_defaults(fn=cmd_code_hash)

    rs = sub.add_parser("resources", help="container CPU/memory limits and the jobs × threads plan")
    rs.add_argument("--shell", action="store_true", help="print shell exports (used by run_pod.sh)")
    rs.add_argument("--profile", default=None)
    rs.set_defaults(fn=cmd_resources)

    sm = sub.add_parser("smoke", help="end-to-end test on synthetic data")
    sm.add_argument("--workdir", default=None)
    sm.add_argument("--n", type=int, default=700, help="synthetic patients per cohort")
    sm.add_argument("--seed", type=int, default=7)
    sm.add_argument("--force", action="store_true")
    sm.add_argument("--experiments", default=None, help="comma-separated experiments (default t2,t1,t3)")
    sm.set_defaults(fn=cmd_smoke)

    f = sub.add_parser("freeze", help="§6.3 freeze (enables the external validation)")
    f.add_argument("--registration", default=None, help="registration ID (OSF or other)")
    f.add_argument("--no-git", action="store_true", help="do not tag in git (smoke test only)")
    f.add_argument("--profile", default=None)
    f.add_argument("--workdir", default=None)
    f.set_defaults(fn=cmd_freeze)

    e = sub.add_parser("export", help="rebuild the aggregate export package")
    e.add_argument("--profile", default=None)
    e.add_argument("--workdir", default=None)
    e.set_defaults(fn=cmd_export)

    args = p.parse_args(argv)
    return int(args.fn(args) or 0)
