"""Branch arc-metric (docs/prereg_arc.md): the two arms of the re-run and the guard of their work folders.

Every experiment is re-run twice on this branch's code (v0.6), changing only how a shaped continuous / ordinal
value is placed on its axis (`Encoder.shape_embedding`):

- arm ``value``: the published placement (g_j of the value's bin), i.e. v0.6 as on main;
- arm ``arc``: the arc length of g_j up to the value.

The arm is chosen with the environment variable ``PERISIM_ARM`` (default ``arc``). It enters the configuration
(`experiment.metric.shape_embedding` and `hypo`'s `metric.shapes.embedding` read ``${PERISIM_ARM:-arc}``), so it is
part of T2's scientific hash and of `hypo`'s configuration hash, and therefore of every freeze record.

Each arm has its own work folder (default ``work_<arm>``), bound to the arm by the marker file ``ARC_METRIC_WORKDIR``.
A folder whose marker names the other arm is refused, and so is a folder without the marker that already holds
pipeline data (experiments/, common/, state/, cache/): a work folder of the published runs is never re-keyed,
overwritten or extended. An empty or new folder gets the marker. The metric-independent inputs of the published runs
(extraction, landmark tables) may be copied in with ``scripts/arc_seed_inputs.py``, which writes the marker first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MARKER = "ARC_METRIC_WORKDIR"
ARMS = ("arc", "value")
_DATA = ("experiments", "common", "state", "cache")


class NotArcWorkdir(SystemExit):
    pass


def current_arm() -> str:
    arm = (os.environ.get("PERISIM_ARM") or "arc").strip()
    if arm not in ARMS:
        raise NotArcWorkdir(f"[perisim] PERISIM_ARM must be one of {', '.join(ARMS)} (got {arm!r})")
    return arm


def default_workdir_name(arm: str | None = None) -> str:
    return f"work_{arm or current_arm()}"


def marker_arm(path: str | os.PathLike) -> str | None:
    """The arm a marked work folder belongs to (None: no marker)."""
    m = Path(path) / MARKER
    if not m.is_file():
        return None
    try:
        return json.loads(m.read_text(encoding="utf-8"))["arm"]
    except (ValueError, KeyError, TypeError):
        raise NotArcWorkdir(f"[perisim] unreadable marker {m}") from None


def ensure_arc_workdir(path: str | os.PathLike | None = None, arm: str | None = None) -> Path:
    """Checks (and marks) the work folder of the current arm. `path` defaults to $PERISIM_WORKDIR."""
    arm = arm or current_arm()
    p = Path(path if path is not None else os.environ.get("PERISIM_WORKDIR") or default_workdir_name(arm))
    p = p.expanduser().resolve()
    have = marker_arm(p)
    if have is not None:
        if have != arm:
            raise NotArcWorkdir(f"[perisim] {p} is the work folder of arm {have!r}, not {arm!r}: set PERISIM_ARM={have} "
                                f"or use the folder of arm {arm!r} (default {default_workdir_name(arm)})")
        return p
    held = [d for d in _DATA if (p / d).exists()]
    if held:
        raise NotArcWorkdir(
            f"[perisim] {p} holds pipeline data ({', '.join(held)}) but no {MARKER} marker: it looks like a work "
            "folder of the published runs. Branch arc-metric writes only into each arm's own folder (default "
            f"{default_workdir_name(arm)}). Set PERISIM_WORKDIR to a new folder, or copy the metric-independent "
            "inputs with scripts/arc_seed_inputs.py.")
    p.mkdir(parents=True, exist_ok=True)
    (p / MARKER).write_text(json.dumps({"arm": arm, "branch": "arc-metric", "prereg": "docs/prereg_arc.md"}) + "\n",
                            encoding="utf-8")
    return p
