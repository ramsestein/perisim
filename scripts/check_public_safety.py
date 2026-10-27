"""Checks that whatever is about to enter the PUBLIC repository contains no secrets and no data.

Usage:  python scripts/check_public_safety.py            (files staged with git add)
        python scripts/check_public_safety.py --all      (all versionable files)
The .githooks/pre-commit hook runs it too (enable with: git config core.hooksPath .githooks).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    (r"sk-ant-[A-Za-z0-9_\-]{20,}", "Anthropic key"),
    (r"sk-proj-[A-Za-z0-9_\-]{20,}", "OpenAI key"),
    (r"\bsk-[A-Za-z0-9]{20,}", "sk- style key"),
    (r"\bsk_[A-Za-z0-9_\-]{20,}", "sk_ style key"),
    (r"AIza[0-9A-Za-z_\-]{30,}", "Google key"),
    (r"khdp_pat_[A-Za-z0-9]{10,}", "KHDP token"),
    (r"ghp_[A-Za-z0-9]{30,}", "GitHub token"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"(?i)(api[_-]?key|secret|password|passwd|token)\w*\s*[:=]\s*['\"][^'\"\s]{12,}['\"]", "hard-coded credential"),
    (r"\bhf_[A-Za-z0-9]{30,}", "Hugging Face token"),
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS access key"),
    (r"github_pat_[A-Za-z0-9_]{30,}", "GitHub fine-grained token"),
    (r"\b(INSPIRE|MIMIC|EICU)-\d{5,}", "patient/stay identifier"),
]
# Unquoted KEY=value credentials are only checked in non-code text files (env/ini/yaml/shell)
UNQUOTED_CRED = (r"(?im)^\s*(export\s+)?\w*(api[_-]?key|secret|password|passwd|token)\w*\s*[:=]\s*[^\s'\"#]{16,}",
                 "hard-coded credential (unquoted)")
TEXT_CONFIG_EXT = {".env", ".ini", ".cfg", ".yaml", ".yml", ".sh", ".txt", ".toml", ".conf", ""}
FORBIDDEN_NAMES = re.compile(r"(^|/)(\.env(\..*)?|[^/]*\.env|secrets?\.[a-z]+|KHDP_settings\.md)$")
FORBIDDEN_EXT = {".parquet", ".csv", ".tsv", ".gz", ".feather", ".arrow", ".npy", ".npz", ".pkl", ".joblib", ".zip",
                 ".xlsx", ".xls", ".sav", ".rds", ".rdata", ".sqlite", ".db", ".h5", ".hdf5", ".dta", ".sas7bdat"}
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
    try:  # never crash the git hook on a non-UTF-8 console (Windows cp1252)
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    files = _files("--all" in sys.argv)
    problems = []
    for f in files:
        p = ROOT / f
        if FORBIDDEN_NAMES.search(f):
            problems.append(f"{f}: file not allowed in a public repository")
            continue
        if p.suffix.lower() in FORBIDDEN_EXT and not (f.startswith("results/") and p.suffix.lower() == ".csv"):
            problems.append(f"{f}: data file type ({p.suffix})")   # results/: aggregate tables (publish_results.py)
            continue
        if not p.is_file():
            continue
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"{f}: too large ({p.stat().st_size} bytes) — data or outputs?")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        pats = SECRET_PATTERNS + ([UNQUOTED_CRED] if p.suffix.lower() in TEXT_CONFIG_EXT else [])
        for pat, what in pats:
            if f.endswith("check_public_safety.py"):
                break
            if re.search(pat, text):
                problems.append(f"{f}: possible {what}")
    if problems:
        print("✖ DO NOT PUSH — problems found:\n  " + "\n  ".join(problems))
        return 1
    print(f"✔ {len(files)} files checked: no secrets or data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
