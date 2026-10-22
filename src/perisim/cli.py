"""Línea de órdenes: python -m perisim <orden>  (run_pod.sh la envuelve en el pod)."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

if platform.system() == "Windows":  # FAISS y PyTorch traen cada uno su OpenMP en Windows (solo desarrollo local)
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
    LOG.info("PERISIM · workdir=%s · etapas=%s · perfil=%s", cfg.workdir, ",".join(stages), cfg.profile)
    rc = pipe.run(stages)
    if rc == 0:
        LOG.info("Informes: %s · Exportable: %s", pipe.ws.reports, pipe.ws.export)
    return rc


def cmd_status(args) -> int:
    from perisim.pipeline import COHORTS, MAIN_STAGES, STAGE_DEFS, Pipeline
    from perisim.utils import read_json

    cfg = _cfg(args)
    pipe = Pipeline(cfg)
    print(f"workdir: {cfg.workdir}")
    for st in MAIN_STAGES:
        if STAGE_DEFS[st].per_cohort:
            s = ", ".join(f"{c}:{'✓' if pipe.done(st, c) else '·'}" for c in COHORTS if c in pipe.available_cohorts())
            print(f"  {st:<12} {s or '—'}")
        else:
            print(f"  {st:<12} {'✓' if pipe.done(st) else '·'}")
    stop = read_json(pipe.ws.state / "last_stop.json")
    if stop:
        print(f"\nÚltima parada en '{stop['stage']}' ({stop['at']}):\n{stop['message']}")
    err = read_json(pipe.ws.state / "last_error.json")
    if err:
        print(f"\nÚltimo error en '{err['stage']}' ({err['at']}) — ver logs en {pipe.ws.logs}")
    fr = read_json(pipe.ws.state / "freeze.json")
    print(f"\nCongelación (§6.3): {'sí, ' + fr['tag'] if fr else 'no'}")
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
        print("[perisim] Aviso: faiss no disponible; se usará la búsqueda exacta en numpy (más lenta)")
    if sys.version_info < (3, 11):
        bad.append(f"Python {sys.version.split()[0]} < 3.11")
    if bad:
        print("[perisim] Entorno incompleto:\n  " + "\n  ".join(bad))
        return 1
    print(f"[perisim] Entorno OK (Python {sys.version.split()[0]})")
    return 0


def cmd_code_hash(args) -> int:
    from perisim.config import code_hash

    print(code_hash())
    return 0


def cmd_smoke(args) -> int:
    """Datos sintéticos con el esquema de las tres cohortes → pipeline completo con el perfil `smoke`."""
    from perisim.config import load_config
    from perisim.data.synthetic import write_synthetic

    work = Path(args.workdir or Path.cwd() / "work_smoke").resolve()
    data_root = work / "datasets"
    (work / "workdir" / "state" / "freeze.json").unlink(missing_ok=True)  # cada smoke congela su propia versión
    cfg0 = load_config(profile="smoke")
    write_synthetic(data_root, cfg0.harmonization, n_subjects=args.n, seed=args.seed)
    from perisim.config import REPO_ROOT

    env = {**os.environ, "PERISIM_DATASETS": str(data_root), "PERISIM_WORKDIR": str(work / "workdir"),
           "PERISIM_INSPIRE_DIR": "auto", "PERISIM_MIMIC_DIR": "auto", "PERISIM_EICU_DIR": "auto",
           "PERISIM_ACK_AUDIT": "true",
           "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src")] + [p for p in [os.environ.get("PYTHONPATH")] if p])}
    base = [sys.executable, "-m", "perisim"]
    steps = [base + ["run", "--profile", "smoke"] + (["--force"] if args.force else []),
             base + ["freeze", "--profile", "smoke", "--no-git", "--registration", "SMOKE-TEST"],
             base + ["run", "--profile", "smoke", "--stage", "external,export"]]
    for cmd in steps:  # pipeline interno → congelación → validación externa (como en el pod, pero sin pausas)
        print("[perisim] smoke:", " ".join(cmd[2:]))
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            return rc
    print(f"[perisim] smoke OK · informes en {work / 'workdir' / 'reports'}")
    return 0


def cmd_freeze(args) -> int:
    """§6.3: etiqueta git local, guarda configuración y hashes; habilita la etapa external."""
    import shutil

    from perisim.config import REPO_ROOT, code_hash
    from perisim.utils import now_iso, write_json

    cfg = _cfg(args)
    from perisim.workspace import Workspace

    ws = Workspace(cfg.workdir)
    tag = f"perisim-freeze-{now_iso()[:10]}"
    git = {}
    if not args.no_git and (REPO_ROOT / ".git").exists() and shutil.which("git"):
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
        if dirty.strip():
            print("[perisim] El repositorio tiene cambios sin confirmar: confírmalos (git commit) antes de congelar.")
            print(dirty)
            return 2
        subprocess.run(["git", "-C", str(REPO_ROOT), "tag", "-f", "-a", tag, "-m", "PERISIM freeze §6.3"], check=False)
        git = {"commit": subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(), "tag": tag}
    else:
        print("[perisim] Aviso: sin git; la congelación queda registrada solo por hashes.")
    if not args.registration:
        print("[perisim] Aviso: sin --registration (OSF); se registra como pendiente (D9).")
    snap = ws.state / "frozen_config"
    snap.mkdir(parents=True, exist_ok=True)
    for name, path in cfg.sources.items():
        shutil.copyfile(path, snap / Path(path).name)
    write_json(ws.state / "freeze.json", {"tag": tag, "at": now_iso(), "code_hash": code_hash(),
                                          "config_hash": cfg.hash(), "git": git,
                                          "registration": args.registration or "PENDIENTE"})
    print(f"[perisim] Congelado ({tag}). La etapa external ya puede ejecutarse: bash run_pod.sh")
    return 0


def cmd_export(args) -> int:
    from perisim.evaluation.export import build_export
    from perisim.workspace import Workspace

    cfg = _cfg(args)
    print(build_export(Workspace(cfg.workdir), cfg))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="perisim", description="PERISIM — pipeline automático (ver SPECS.md)")
    p.add_argument("--config-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="ejecutar el pipeline (reanuda lo pendiente)")
    r.add_argument("--profile", default=None)
    r.add_argument("--workdir", default=None)
    r.add_argument("--stage", default=None, help="lista de etapas separadas por comas")
    r.add_argument("--from", dest="from_stage", default=None)
    r.add_argument("--until", default=None)
    r.add_argument("--force", action="store_true", help="rehacer aunque la etapa esté al día")
    r.add_argument("--ack-audit", action="store_true", help="continuar tras revisar la auditoría §7")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="estado de las etapas")
    s.add_argument("--profile", default=None)
    s.add_argument("--workdir", default=None)
    s.set_defaults(fn=cmd_status)

    sub.add_parser("env-check").set_defaults(fn=cmd_env_check)
    sub.add_parser("code-hash").set_defaults(fn=cmd_code_hash)

    sm = sub.add_parser("smoke", help="prueba de punta a punta con datos sintéticos")
    sm.add_argument("--workdir", default=None)
    sm.add_argument("--n", type=int, default=700, help="pacientes sintéticos por cohorte")
    sm.add_argument("--seed", type=int, default=7)
    sm.add_argument("--force", action="store_true")
    sm.set_defaults(fn=cmd_smoke)

    f = sub.add_parser("freeze", help="congelación §6.3 (habilita la validación externa)")
    f.add_argument("--registration", default=None, help="ID de registro (OSF u otro)")
    f.add_argument("--no-git", action="store_true", help="no etiquetar en git (solo smoke test)")
    f.add_argument("--profile", default=None)
    f.add_argument("--workdir", default=None)
    f.set_defaults(fn=cmd_freeze)

    e = sub.add_parser("export", help="regenerar el paquete de exportación de agregados")
    e.add_argument("--profile", default=None)
    e.add_argument("--workdir", default=None)
    e.set_defaults(fn=cmd_export)

    args = p.parse_args(argv)
    return int(args.fn(args) or 0)
