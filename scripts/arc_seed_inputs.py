"""Branch arc-metric: copy the METRIC-INDEPENDENT inputs of a published run into this branch's work folder.

Usage:  PERISIM_ARM=<arc|value> python scripts/arc_seed_inputs.py --from <published work folder> [--to <folder>]
        [--landmarks]        (default --to: <repo>/work_<arm>; run it once per arm)

The re-run of docs/prereg_arc.md changes only how a shaped value is placed on its axis. Everything upstream of the
metric is the same code and configuration, so re-extracting the original tables would only cost time. This script
copies (never moves, never links: the published folder is left exactly as it was):

- T2's shared data layer: common/ (extraction), cache/ (filtered extracts), artifacts/_data/ (discovery),
  reports/schema_*.md and state/_/ (the markers of env, discover and extract, keyed by code that this branch does
  not change, so those stages are skipped and every later one — targets, features, … — is recomputed);
- `hypo`: experiments/hypo/extract/ and cache/ (the extracts of the original tables); with --landmarks also the
  landmark tables (otherwise `hypo landmarks` rebuilds them from the extracts: they should come out identical);
- NWICU: experiments/nwicu/ build outputs (stays, long tables, landmarks/, build.md), not the transport results.

Nothing that depends on the metric is copied (shapes, weights, folds, predictions, freezes, reports). The marker
ARC_METRIC_WORKDIR is written first; SEED_MANIFEST.json records what was copied, from where, and each file's size
and SHA-256. Refuses a destination that already holds pipeline data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perisim.arc_guard import MARKER, current_arm, default_workdir_name, ensure_arc_workdir  # noqa: E402

T2_SHARED = ["common", "cache", "artifacts/_data", "state/_"]
HYPO = ["experiments/hypo/extract", "experiments/hypo/cache"]
HYPO_LM = ["experiments/hypo/landmarks"]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(src: Path, landmarks: bool) -> list[str]:
    """Relative paths (folders or files) to copy that exist in `src`."""
    rel = T2_SHARED + HYPO + (HYPO_LM if landmarks else [])
    rel += [p.relative_to(src).as_posix() for p in sorted((src / "reports").glob("schema_*.md"))]
    nw = src / "experiments" / "nwicu"
    if nw.is_dir():
        rel += [p.relative_to(src).as_posix() for p in sorted(nw.glob("*.parquet"))]
        rel += [r for r in ("experiments/nwicu/landmarks", "experiments/nwicu/build.md") if (src / r).exists()]
    return [r for r in rel if (src / r).exists()]


def seed(src: Path, dst: Path, landmarks: bool = False) -> dict:
    src, dst = src.expanduser().resolve(), dst.expanduser().resolve()
    if src == dst:
        raise SystemExit("[seed] --from and --to are the same folder")
    if not (src / "common").is_dir() and not (src / "experiments").is_dir():
        raise SystemExit(f"[seed] {src} does not look like a PERISIM work folder (no common/ or experiments/)")
    if dst.exists() and any(dst.iterdir()) and not (dst / MARKER).is_file():
        raise SystemExit(f"[seed] {dst} is not empty and is not an arc-metric work folder: choose a new folder")
    held = [d for d in ("experiments", "common", "state", "cache") if (dst / d).exists()]
    if held:
        raise SystemExit(f"[seed] {dst} already holds pipeline data ({', '.join(held)}): seed only a new folder")
    ensure_arc_workdir(dst)                         # the marker first
    files = []
    for r in plan(src, landmarks):
        s, d = src / r, dst / r
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
        for f in ([d] if d.is_file() else sorted(p for p in d.rglob("*") if p.is_file())):
            files.append({"path": f.relative_to(dst).as_posix(), "bytes": f.stat().st_size, "sha256": _sha(f)})
        print(f"[seed] copied {r}")
    man = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": current_arm(), "from": str(src), "landmarks_copied": landmarks,
           "n_files": len(files), "bytes": sum(f["bytes"] for f in files), "files": files}
    (dst / "SEED_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"[seed] {len(files)} files ({man['bytes'] / 1e9:.2f} GB) in {dst}; manifest SEED_MANIFEST.json")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="src", required=True, help="work folder of a published run (left untouched)")
    ap.add_argument("--to", dest="dst", default=None, help="new work folder of the arm (default <repo>/work_<arm>)")
    ap.add_argument("--landmarks", action="store_true", help="also copy hypo's landmark tables (else rebuilt)")
    a = ap.parse_args(argv)
    seed(Path(a.src), Path(a.dst) if a.dst else ROOT / default_workdir_name(), a.landmarks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
