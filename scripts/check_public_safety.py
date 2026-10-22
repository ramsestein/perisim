"""Comprueba que lo que va a entrar en el repositorio PÚBLICO no contiene secretos ni datos.

Uso:  python scripts/check_public_safety.py            (ficheros preparados con git add)
      python scripts/check_public_safety.py --all      (todos los ficheros versionables)
Lo ejecuta también el hook .githooks/pre-commit (activar con: git config core.hooksPath .githooks).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    (r"sk-ant-[A-Za-z0-9_\-]{20,}", "clave de Anthropic"),
    (r"sk-proj-[A-Za-z0-9_\-]{20,}", "clave de OpenAI"),
    (r"\bsk-[A-Za-z0-9]{20,}", "clave tipo sk-"),
    (r"\bsk_[A-Za-z0-9_\-]{20,}", "clave tipo sk_"),
    (r"AIza[0-9A-Za-z_\-]{30,}", "clave de Google"),
    (r"khdp_pat_[A-Za-z0-9]{10,}", "token de KHDP"),
    (r"ghp_[A-Za-z0-9]{30,}", "token de GitHub"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "clave privada"),
    (r"(?i)\b(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]", "credencial literal"),
]
FORBIDDEN_NAMES = re.compile(r"(^|/)(\.env(\..*)?|KHDP_settings\.md)$")
FORBIDDEN_EXT = {".parquet", ".csv", ".gz", ".feather", ".arrow", ".npy", ".npz", ".pkl", ".joblib", ".zip"}
MAX_BYTES = 2_000_000


def _files(all_files: bool) -> list[str]:
    if all_files:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True).stdout
    else:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"], cwd=ROOT,
                             capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f.strip()]


def main() -> int:
    files = _files("--all" in sys.argv)
    problems = []
    for f in files:
        p = ROOT / f
        if FORBIDDEN_NAMES.search(f):
            problems.append(f"{f}: fichero prohibido en un repositorio público")
            continue
        if p.suffix.lower() in FORBIDDEN_EXT:
            problems.append(f"{f}: tipo de fichero de datos ({p.suffix})")
            continue
        if not p.is_file():
            continue
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"{f}: demasiado grande ({p.stat().st_size} bytes) — ¿datos o salidas?")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat, what in SECRET_PATTERNS:
            if f.endswith("check_public_safety.py"):
                break
            if re.search(pat, text):
                problems.append(f"{f}: posible {what}")
    if problems:
        print("✖ NO SUBIR — problemas encontrados:\n  " + "\n  ".join(problems))
        return 1
    print(f"✔ {len(files)} ficheros revisados: sin secretos ni datos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
