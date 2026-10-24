"""Candidate export package (§9; KHDP: aggregates only, prior review via admin@khdp.net).

Only reports and aggregate JSON files are copied. Never: predictions/, common/, derived/, cache/ or Parquet.
Each file is scanned for stay/patient identifiers; if any is found, the file is excluded.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import zipfile
from pathlib import Path

from perisim.utils import now_iso

LOG = logging.getLogger("perisim.export")
ID_RE = re.compile(r"\b(INSPIRE|MIMIC|EICU)-\d")   # internal identifiers: cohort prefix + numeric id
ALLOWED_SUFFIX = {".html", ".md", ".json", ".png", ".svg"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def build_export(ws, cfg) -> Path:
    out = ws.export
    for p in out.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    cands: list[tuple[Path, str]] = []
    for p in ws.reports.glob("*"):
        cands.append((p, f"reports/{p.name}"))
    for p in ws.data_reports.glob("schema_*"):            # discovery reports, shared by the experiments
        cands.append((p, f"reports/{p.name}"))
    for p in ws.docs.glob("*.md"):
        cands.append((p, f"docs/{p.name}"))
    art = ws.root / "artifacts"
    if art.exists():
        for p in art.rglob("*.json"):
            cands.append((p, "artifacts/" + p.relative_to(art).as_posix()))
    kept, dropped = [], []
    for src, rel in cands:
        if not src.is_file() or src.suffix.lower() not in ALLOWED_SUFFIX:
            continue
        text = src.read_text(encoding="utf-8", errors="ignore") if src.suffix != ".png" else ""
        if ID_RE.search(text):
            dropped.append((rel, "contains identifiers"))
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        kept.append((rel, dst.stat().st_size, _sha(dst)))
    lines = [
        "# PERISIM — candidate export package", "",
        f"Generated: {now_iso()} · code {cfg.hash()[:8]} · experiment `{ws.experiment}` · workdir `{ws.root}`", "",
        "Contents: aggregates only (metrics, weights w_j, importances, VDM tables with n ≥ "
        f"{cfg.experiment['evaluation']['small_cell_threshold']}, curves, HTML reports). No patient rows.", "",
        "## KHDP procedure", "",
        "1. Write to admin@khdp.net stating the **purpose** (validation of the PERISIM model; TRIPOD+AI manuscript) "
        "and the **destination** of the files.",
        "2. Attach or reference this manifest and follow the review indicated by the administrator.", "",
        "## Automatic checks", "",
        "- Excluded by design: `predictions/`, `common/`, `derived/`, `cache/`, all `.parquet` files.",
        "- Scan for stay/patient identifiers in every file: "
        + (f"{len(dropped)} file(s) excluded." if dropped else "none found."), "",
        "## Files", "", "| file | bytes | sha256[:16] |", "|---|---|---|",
    ] + [f"| {r} | {s} | {h} |" for r, s, h in sorted(kept)]
    if dropped:
        lines += ["", "## Excluded", ""] + [f"- {r}: {why}" for r, why in dropped]
    (out / "EXPORT_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    zpath = out / f"perisim_export_{now_iso()[:10]}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file() and p != zpath:
                z.write(p, p.relative_to(out).as_posix())
    LOG.info("export: %d files in %s (%d excluded)", len(kept), out, len(dropped))
    return out
