"""Configuration, paths, code hash and the freeze guard of experiment `roll`."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent


class Stop(RuntimeError):
    """A condition that must stop a stage (a changed freeze, missing artefacts)."""


def load(path: str | Path | None = None) -> dict:
    with open(path or REPO / "config" / "roll.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def base_workdir() -> Path:
    return Path(os.environ.get("PERISIM_WORKDIR") or REPO / "work")


def workdir() -> Path:
    """<PERISIM_WORKDIR or repo/work>/experiments/roll: patient-level data and predictions, git-ignored."""
    return base_workdir() / "experiments" / "roll"


def data_root() -> Path:
    """The original tables (PERISIM_DATASETS or repo/data), read only for calendar and hospital fields."""
    return Path(os.environ.get("PERISIM_DATASETS") or REPO / "data")


def code_hash(n: int = 12) -> str:
    """Hash of this package plus `hypo`'s code hash (which covers the perisim code it uses)."""
    from hypo.config import code_hash as hypo_hash

    h = hashlib.sha256(hypo_hash().encode())
    for f in sorted(PKG.rglob("*.py")):
        h.update(f.relative_to(PKG).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:n]


def config_hash(cfg: dict | None = None, n: int = 12) -> str:
    cfg = load() if cfg is None else cfg
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:n]


def frozen_guard() -> dict:
    """T2: code and configuration hashes against its freeze record; `hypo`: its own check (code, configuration,
    folds, Phase 2 comparators, final artefacts). Raises Stop if anything changed."""
    from hypo.frozen import Stop as HypoStop
    from hypo.frozen import check as hypo_check
    from perisim.config import code_hash as t2_code
    from perisim.config import load_config

    p = base_workdir() / "experiments" / "t2" / "state" / "freeze.json"
    if not p.exists():
        raise Stop(f"T2's freeze record is missing: {p}")
    fr = json.loads(p.read_text(encoding="utf-8"))
    now = {"code_hash": t2_code(), "config_hash": load_config(None, "local", {}).scientific_hash()}
    bad = [k for k in now if now[k] != fr[k]]
    if bad:
        raise Stop(f"T2's frozen {bad} changed: {now} vs the record {[fr[k] for k in bad]}")
    try:
        h = hypo_check()
    except HypoStop as e:
        raise Stop(f"hypo's freeze: {e}") from e
    return {"t2": {"tag": fr["tag"], **now}, "hypo": {"tag": h["tag"], **{k: h["hashes"][k] for k in ("code", "config")}}}


def dry_settings(cfg: dict) -> dict:
    """The configuration scaled down for the synthetic dry run (cfg['dry']['scale']); never used on real data."""
    c = copy.deepcopy(cfg)
    s = c["dry"]["scale"]
    for st in c["streams"].values():
        n = s["N_eicu"] if st["kind"] in ("within", "deploy") else s["N"]
        st.update(N=[n], N_primary=n, orders=int(s["orders"]))
        if st["kind"] == "cross":
            st.update(first_n=int(s["first_n"]), second_n=int(s["second_n"]))
    c["windows"]["W"] = {"mimic": s["W"], "inspire": s["W"], "eicu": s["W"], "eicu_hospital": s["W_eicu_hospital"]}
    c["windows"]["boot"] = int(s["window_boot"])
    c["monitor"]["M"] = {"mimic": s["M"], "inspire": s["M"], "eicu": s["M"], "eicu_hospital": s["M_eicu_hospital"]}
    c["monitor"]["E"] = int(s["E"])
    c["eicu_hospitals"]["min_patients_each_year"] = 2 * s["N_eicu"]
    c["eicu_hospitals"]["n250_min_first_year"] = 10 ** 9
    c["endpoints"]["boot"] = int(s["boot"])
    c["diagnostics"]["redeform"]["every_windows"] = 4
    c["t2"].update(N=s["N"], W=s["W"])
    return c
