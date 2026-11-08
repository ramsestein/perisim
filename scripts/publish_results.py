"""Copies an experiment's AGGREGATE results into the repository (results/arc-metric/<arm>/<experiment>/ on branch arc-metric), for version control.

Usage:  python scripts/publish_results.py [t2 t1 t3] [--workdir <work folder>]   (default: <repo>/work_<PERISIM_ARM>, every
        experiment that has an export package)

Source: the experiment's export package (<work>/experiments/<t>/export/: aggregates only, already scanned for
identifiers by the pipeline), the experiment's docs (OSF note, generated DECISIONS / OPEN_QUESTIONS) and the
exploratory post-hoc analyses when they exist (results/posthoc*/, their reports and JSON). Copied:
HTML reports, tables (CSV) and figures, aggregate JSON and docs. Left out: the zip (a duplicate) and the per-fold CV
artifacts (bulky intermediates, summarised in results*.json). Never touched: data/ and the rest of work/ (common/,
derived/, predictions/, cache/: patient-level data, restricted by the PhysioNet data use agreements).
Every copied file is scanned again for stay/patient identifiers and size; results/<t>/ is replaced as a whole.
v0.6: then scripts/sanitize_results.py blanks every derivable small cell of results/ (the export package of v0.5
had ood_pct / % outside the range implying counts of 1–9), so a re-publication cannot bring them back.
Does not touch src/ or config/, so it never changes the frozen code or configuration hashes.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perisim.evaluation.export import ID_RE  # noqa: E402

ALLOWED = {".html", ".md", ".json", ".csv", ".png", ".svg"}
SKIP = re.compile(r"(^|/)(cv|temporal)/[^/]+/fold\d+\.json$|\.zip$")
MAX_BYTES = 2_000_000          # as scripts/check_public_safety.py


def publish(work: Path, exp: str) -> list[str]:
    src = work / "experiments" / exp / "export"
    if not src.is_dir():
        return []
    files = [(p, p.relative_to(src).as_posix()) for p in sorted(src.rglob("*")) if p.is_file()]
    root = work / "experiments" / exp
    docs = root / "docs"
    files += [(p, f"docs/{p.name}") for p in sorted(docs.glob("*.md"))] if docs.is_dir() else []
    # the exploratory post-hoc analyses (scripts/posthoc_*.py) run after the export stage: same layout
    files += [(p, p.relative_to(root).as_posix()) for d in sorted((root / "results").glob("posthoc*"))
              for p in sorted(d.rglob("*")) if p.is_file()]
    files += [(p, p.relative_to(root).as_posix()) for p in sorted((root / "reports").glob("posthoc_*.html")) +
              sorted((root / "artifacts" / "main").glob("posthoc_*.json"))]
    files = list({rel: (p, rel) for p, rel in files}.values())     # the experiment's own docs win over the export's
    dest = ROOT / "results" / "arc-metric" / (os.environ.get("PERISIM_ARM") or "arc") / exp   # never over the published results
    shutil.rmtree(dest, ignore_errors=True)
    copied, refused = [], []
    for p, rel in files:
        if SKIP.search(rel) or p.suffix.lower() not in ALLOWED or rel == "EXPORT_MANIFEST.md":
            continue
        if p.stat().st_size > MAX_BYTES:
            refused.append(f"{rel}: too large ({p.stat().st_size} bytes)")
            continue
        if ID_RE.search(p.read_text(encoding="utf-8", errors="ignore") if p.suffix != ".png" else ""):
            refused.append(f"{rel}: contains identifiers")
            continue
        q = dest / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        copied.append(rel)
    for r in refused:
        print(f"[publish] {exp}: NOT copied, {r}")
    return copied


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiments", nargs="*", default=["t2", "t1", "t3"])
    ap.add_argument("--workdir", default=str(ROOT / f"work_{os.environ.get('PERISIM_ARM') or 'arc'}"))
    a = ap.parse_args()
    for exp in a.experiments:
        out = publish(Path(a.workdir), exp)
        if out:
            print(f"[publish] {exp}: {len(out)} files in results/{exp}/")
    # v0.6: no derivable small cell may stay in what was copied (privacy rule: no count 1–9, direct or derived)
    sys.path.insert(0, str(ROOT / "scripts"))
    from sanitize_results import sanitize

    for f, log in sorted(sanitize(ROOT / "results").items()):
        print(f"[publish] small cells blanked in {f}: {len(log)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
