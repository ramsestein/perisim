"""Paquete de exportación candidato (§9; KHDP: solo agregados, revisión previa vía admin@khdp.net).

Se copian únicamente informes y JSON de agregados. Nunca: predictions/, common/, derived/, cache/ ni Parquet.
Cada fichero se escanea en busca de identificadores de estancia/paciente; si aparece alguno, se excluye.
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
ID_RE = re.compile(r"\b(INSPIRE|MIMIC|EICU)-\d")   # identificadores internos: prefijo de cohorte + id numérico
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
            dropped.append((rel, "contiene identificadores"))
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        kept.append((rel, dst.stat().st_size, _sha(dst)))
    lines = [
        "# PERISIM — paquete de exportación candidato", "",
        f"Generado: {now_iso()} · código {cfg.hash()[:8]} · workdir `{ws.root}`", "",
        "Contenido: solo agregados (métricas, pesos w_j, importancias, tablas VDM con n ≥ "
        f"{cfg.experiment['evaluation']['small_cell_threshold']}, curvas, informes HTML). Sin filas de pacientes.", "",
        "## Procedimiento KHDP", "",
        "1. Escribir a admin@khdp.net indicando el **propósito** (validación del modelo PERISIM; manuscrito TRIPOD+AI) "
        "y el **destino** de los ficheros.",
        "2. Adjuntar o referenciar este manifiesto y seguir la revisión indicada por el administrador.", "",
        "## Comprobaciones automáticas", "",
        "- Excluidos por diseño: `predictions/`, `common/`, `derived/`, `cache/`, todos los `.parquet`.",
        "- Escaneo de identificadores de estancia/paciente en cada fichero: "
        + (f"{len(dropped)} fichero(s) excluido(s)." if dropped else "ninguno encontrado."), "",
        "## Ficheros", "", "| fichero | bytes | sha256[:16] |", "|---|---|---|",
    ] + [f"| {r} | {s} | {h} |" for r, s, h in sorted(kept)]
    if dropped:
        lines += ["", "## Excluidos", ""] + [f"- {r}: {why}" for r, why in dropped]
    (out / "EXPORT_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    zpath = out / f"perisim_export_{now_iso()[:10]}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file() and p != zpath:
                z.write(p, p.relative_to(out).as_posix())
    LOG.info("export: %d ficheros en %s (%d excluidos)", len(kept), out, len(dropped))
    return out
