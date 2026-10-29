"""Configuration, paths and code hash of the `hypo` experiment."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent


def load(path: str | Path | None = None) -> dict:
    with open(path or REPO / "config" / "hypo.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def workdir() -> Path:
    """<PERISIM_WORKDIR or repo/work>/experiments/hypo: patient-level data, git-ignored like the rest of work/."""
    return Path(os.environ.get("PERISIM_WORKDIR") or REPO / "work") / "experiments" / "hypo"


def common_dir(cohort: str) -> Path:
    """T2's common format of a cohort (its stays_candidates are this experiment's population source)."""
    return Path(os.environ.get("PERISIM_WORKDIR") or REPO / "work") / "common" / cohort


def dataset_dirs(profile: str | None = "local") -> dict[str, Path]:
    """Folders of the original tables, found as T2's pipeline finds them."""
    from perisim.config import load_config
    from perisim.data.discover import discover_cohorts

    disc = discover_cohorts(load_config(None, profile, {}))
    return {c: Path(disc[c]["path"]) for c in ("inspire", "mimic", "eicu") if (disc.get(c) or {}).get("path")}


def code_hash(n: int = 12) -> str:
    """Hash of this package plus the hash of the perisim code it uses."""
    from perisim.config import code_hash as perisim_hash

    h = hashlib.sha256(perisim_hash().encode())
    for f in sorted(PKG.rglob("*.py")):
        h.update(f.relative_to(PKG).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))           # same hash on any checkout (as perisim's)
    return h.hexdigest()[:n]
