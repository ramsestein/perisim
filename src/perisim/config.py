"""YAML configuration loading with environment-variable expansion, profiles and reproducible hashes."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "perisim"
CONFIG_SECTIONS = ("experiment", "cohorts", "harmonization", "targets", "model_space")
EXPERIMENTS = ("t1", "t2", "t3")

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SCALAR_RE = re.compile(r"(?i)true|false|yes|no|on|off|-?\d+(\.\d+)?([eE][-+]?\d+)?")


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str) and "${" in obj:
        full = _ENV_RE.fullmatch(obj)
        if full:  # whole value -> booleans and numbers are typed; anything else (paths…) stays a string
            raw = os.environ.get(full.group(1)) or (full.group(2) if full.group(2) is not None else "")
            if _SCALAR_RE.fullmatch(raw.strip()):
                return yaml.safe_load(raw)
            return raw
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1)) or m.group(2) or "", obj)
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def stable_hash(obj: Any, n: int = 12) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def code_hash(subpackages: list[str] | None = None, n: int = 12) -> str:
    """Hash of the source code (all of src/perisim, or only the given subpackages)."""
    h = hashlib.sha256()
    roots = [SRC_ROOT] if not subpackages else [SRC_ROOT / s for s in subpackages]
    files: list[Path] = []
    for r in roots:
        if r.is_file():
            files.append(r)
        elif r.with_suffix(".py").is_file():
            files.append(r.with_suffix(".py"))
        elif r.is_dir():
            files.extend(sorted(r.rglob("*.py")))
    for f in sorted(set(files)):
        h.update(f.relative_to(SRC_ROOT).as_posix().encode())   # same hash on Windows and Linux
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:n]


@dataclass
class Config:
    experiment: dict
    cohorts: dict
    harmonization: dict
    targets: dict
    model_space: dict
    profile: str | None = None
    variant: str = "main"
    sources: dict = field(default_factory=dict)

    def section(self, name: str) -> dict:
        return getattr(self, name)

    def hash(self, *sections: str) -> str:
        sections = sections or CONFIG_SECTIONS
        return stable_hash({s: self.section(s) for s in sections})

    def scientific_hash(self) -> str:
        """Hash of everything that can change a RESULT — i.e. without machine/run-specific settings (paths,
        compute resources, gate acknowledgements). Used by the §6.3 freeze (AUDIT A4)."""
        d = self.to_dict()
        exp = {k: v for k, v in d["experiment"].items() if k not in ("paths", "resources", "gates")}
        return stable_hash({**d, "experiment": exp})

    def get(self, dotted: str, default: Any = None) -> Any:
        """cfg.get('experiment.knn.k_primary')"""
        node: Any = self.section(dotted.split(".")[0])
        for part in dotted.split(".")[1:]:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def workdir(self) -> Path:
        return Path(self.experiment["paths"]["workdir"]).expanduser()

    @property
    def experiment_name(self) -> str:
        """One experiment per target (decided 2026-10-24): t2 (primary), t1, t3."""
        name = str(self.experiment.get("experiment_name", "t2"))
        if name not in EXPERIMENTS:
            raise ValueError(f"experiment_name must be one of {EXPERIMENTS} (PERISIM_EXPERIMENT), not {name!r}")
        return name

    @property
    def targets_to_model(self) -> list[str]:
        t = self.experiment.get("targets_to_model", "auto")
        return [self.experiment_name] if t in (None, "auto") else list(t)

    @property
    def pilot_target(self) -> str:
        t = self.experiment["pilot"].get("target", "auto")
        return self.targets_to_model[0] if t in (None, "auto") else str(t)

    @property
    def patient_output_target(self) -> str:
        t = self.experiment.get("patient_output_target", "auto")
        return self.targets_to_model[0] if t in (None, "auto") else str(t)

    @property
    def seed(self) -> int:
        return int(self.experiment["seed"])

    def with_overrides(self, overrides: dict, variant: str | None = None) -> "Config":
        kw = {s: deep_merge(self.section(s), overrides.get(s, {})) for s in CONFIG_SECTIONS}
        return Config(**kw, profile=self.profile, variant=variant or self.variant, sources=self.sources)

    def to_dict(self) -> dict:
        return {s: self.section(s) for s in CONFIG_SECTIONS}


def load_config(config_dir: str | Path | None = None, profile: str | None = None,
                overrides: dict | None = None) -> Config:
    cdir = Path(config_dir or os.environ.get("PERISIM_CONFIG_DIR") or REPO_ROOT / "config")
    data, sources = {}, {}
    for s in CONFIG_SECTIONS:
        p = cdir / f"{s}.yaml"
        with open(p, encoding="utf-8") as fh:
            data[s] = yaml.safe_load(fh) or {}
        sources[s] = str(p)
    if profile:
        pp = cdir / "profiles" / f"{profile}.yaml"
        with open(pp, encoding="utf-8") as fh:
            prof = yaml.safe_load(fh) or {}
        for s in CONFIG_SECTIONS:
            data[s] = deep_merge(data[s], prof.get(s, {}))
    if overrides:
        for s in CONFIG_SECTIONS:
            data[s] = deep_merge(data[s], overrides.get(s, {}))
    data = {s: _expand_env(v) for s, v in data.items()}
    return Config(**data, profile=profile, sources=sources)
