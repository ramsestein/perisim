"""Running a post hoc script against a FROZEN experiment's code (v0.6; software audit 2026-11-05, item 10).

The published results of T2 and `hypo` are those of v0.5; v0.6 changed src/, so the live code no longer matches the
freeze records. A post hoc script that analyses a frozen experiment must run with that experiment's frozen code:

- `use_src()`: puts the source tree first on sys.path: $PERISIM_SRC when it is set (e.g. frozen/t2/src), otherwise
  <repo>/src. Every post hoc script calls it before importing perisim / hypo.
- `check_t2_freeze(cfg, ws)`: the T2 post hoc scripts call it at the start: the live code hash and the scientific
  configuration hash must equal those of T2's freeze record (experiments/t2/state/freeze.json), as the external
  stage requires; otherwise the script stops and says how to point PERISIM_SRC at T2's frozen code. (The start /
  end comparison of the frozen artefacts' hashes, transport.core.frozen_hashes, is kept: it checks that nothing
  changed DURING the analysis, not that the code is the frozen one.)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = "PERISIM_SRC"
T2_FROZEN_SRC = "frozen/t2/src"


def src_dir() -> Path:
    """$PERISIM_SRC (relative paths from the repository root) or <repo>/src."""
    env = os.environ.get(ENV, "").strip()
    if not env:
        return ROOT / "src"
    p = Path(env).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def use_src() -> Path:
    """Puts src_dir() first on sys.path (before <repo>/src or an installed copy) and returns it."""
    p = src_dir()
    if os.environ.get(ENV, "").strip() and not p.is_dir():
        raise SystemExit(f"{ENV}={os.environ[ENV]!r}: no such source directory ({p})")
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    return p


def check_t2_freeze(cfg, ws) -> dict:
    """Stops unless the live code and configuration hashes equal those of T2's freeze record. Returns
    {tag, code_hash, config_hash, src}."""
    from perisim.config import code_hash

    p = Path(ws.state) / "freeze.json"
    if not p.is_file():
        raise SystemExit(f"post hoc: T2 has no freeze record ({p}); these analyses only run on the frozen T2")
    fr = json.loads(p.read_text(encoding="utf-8"))
    now = {"code_hash": code_hash(), "config_hash": cfg.scientific_hash()}
    bad = [k for k in now if fr.get(k) != now[k]]
    if bad:
        raise SystemExit(
            f"post hoc: the live {' and '.join(bad)} differ from T2's freeze record {fr.get('tag', '')} "
            f"({', '.join(f'{k}: {now[k]} vs {fr.get(k)}' for k in bad)}). The T2 post hoc analyses must run on T2's "
            f"frozen code: set {ENV}={T2_FROZEN_SRC} (PowerShell: $env:{ENV} = '{T2_FROZEN_SRC}'; bash: "
            f"export {ENV}={T2_FROZEN_SRC}) and run again; if the configuration differs, point PERISIM_CONFIG_DIR at "
            f"T2's frozen configuration (experiments/t2/state/frozen_config) too.")
    return {"tag": fr.get("tag"), **now, "src": str(src_dir())}
