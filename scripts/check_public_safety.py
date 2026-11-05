"""Checks that whatever is about to enter the PUBLIC repository contains no secrets and no data.

Usage:  python scripts/check_public_safety.py            (files staged with git add: their STAGED content)
        python scripts/check_public_safety.py --all      (all versionable files: working-tree content)
The .githooks/pre-commit hook runs it too (enable with: git config core.hooksPath .githooks).

Checks (v0.6, software audit 2026-11-05 item 2): forbidden names and data file types (tabular, serialized,
archives, backups); size; secrets; stay / patient identifiers (perisim.evaluation.privacy.ID_RE, as the export
package; source code: 5+ digits, its tests use short synthetic identifiers); and derivable small cells in every
CSV / JSON (and the Markdown / HTML reports under results/): person counts 1–9, complements n − count 1–9,
percentages / shares that imply a count or complement of 1–9 next to their n, bin shares whose hidden bin is
recoverable as 1 − Σ(shown). Staged files are read from the INDEX (what will be committed), not from the working
tree. CSV files are allowed only under results/ (aggregate tables).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perisim.evaluation import privacy as PV  # noqa: E402  (stdlib only: cheap to import)

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
]
# Unquoted KEY=value credentials are only checked in non-code text files (env/ini/yaml/shell)
UNQUOTED_CRED = (r"(?im)^\s*(export\s+)?\w*(api[_-]?key|secret|password|passwd|token)\w*\s*[:=]\s*[^\s'\"#]{16,}",
                 "hard-coded credential (unquoted)")
CRED_WORDS = ("api_key", "api-key", "apikey", "secret", "passw", "token")
TEXT_CONFIG_EXT = {".env", ".ini", ".cfg", ".yaml", ".yml", ".sh", ".txt", ".toml", ".conf", ""}
FORBIDDEN_NAMES = re.compile(r"(^|/)(\.env(\..*)?|[^/]*\.env|secrets?\.[a-z]+|KHDP_settings\.md)$")
# v0.6: + serialized objects (.pickle; .jsonl / .ndjson: one record per row), archives and backups
FORBIDDEN_EXT = {".parquet", ".csv", ".tsv", ".gz", ".feather", ".arrow", ".npy", ".npz", ".pkl", ".pickle",
                 ".joblib", ".zip", ".xlsx", ".xls", ".sav", ".rds", ".rdata", ".sqlite", ".db", ".h5", ".hdf5",
                 ".dta", ".sas7bdat", ".jsonl", ".ndjson", ".tar", ".tgz", ".xz", ".bz2", ".7z", ".rar", ".zst",
                 ".lz4", ".bak", ".orig", ".old", ".tmp", ".swp", ".orc", ".avro", ".mat", ".pt", ".pth", ".ckpt",
                 ".msgpack", ".dat"}
SMALL_CELL_ALWAYS = {".csv", ".json"}
SMALL_CELL_RESULTS = {".md", ".html"}
MAX_BYTES = 2_000_000


def _git(*args: str, inp: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, input=inp, check=False).stdout


def _files(all_files: bool) -> list[str]:
    if all_files:
        out = _git("ls-files", "--cached", "--others", "--exclude-standard")
    else:
        out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    return [f for f in out.decode("utf-8", errors="replace").splitlines() if f.strip()]


def _staged_blobs(files: list[str]) -> dict[str, bytes | None]:
    """Content of each file as it is in the INDEX (git cat-file --batch on ':path'); None if not in the index.
    v0.6: the size and content checks used to read the working tree, so a file staged large and then truncated
    on disk passed."""
    if not files:
        return {}
    out = _git("cat-file", "--batch", inp="".join(f":{f}\n" for f in files).encode("utf-8"))
    res, pos = {}, 0
    for f in files:
        nl = out.index(b"\n", pos)
        head = out[pos:nl].split()
        pos = nl + 1
        if len(head) < 3 or head[1] != b"blob":        # "<name> missing" or not a blob
            res[f] = None
            continue
        size = int(head[2])
        res[f] = out[pos:pos + size]
        pos += size + 1
    return res


def _worktree(files: list[str]) -> dict[str, bytes | None]:
    return {f: ((ROOT / f).read_bytes() if (ROOT / f).is_file() else None) for f in files}


def cohort_sizes(content: dict[str, bytes | None]) -> dict:
    """Analysis n of each cohort (last step of results/t2/flow/flow_<cohort>.json), the denominator of the
    per-variable percentages of a table without its own n."""
    out = {}
    for c in PV.COHORTS:
        f = f"results/t2/flow/flow_{c}.json"
        b = content.get(f)
        if b is None and (ROOT / f).is_file():
            b = (ROOT / f).read_bytes()
        try:
            steps = json.loads(b) if b else None
            if steps and isinstance(steps[-1].get("n_stays"), int):
                out[c] = steps[-1]["n_stays"]
        except (ValueError, AttributeError, TypeError):
            pass
    return out


def check(files: list[str], content: dict[str, bytes | None]) -> list[str]:
    problems = []
    cohort_n = cohort_sizes(content)
    for f in files:
        suf = Path(f).suffix.lower()
        if FORBIDDEN_NAMES.search(f):
            problems.append(f"{f}: file not allowed in a public repository")
            continue
        # v0.6: no blanket exemption; a CSV is allowed only as an aggregate table under results/ (scanned below)
        if suf in FORBIDDEN_EXT and not (suf == ".csv" and f.startswith("results/")):
            problems.append(f"{f}: data file type ({suf})")
            continue
        data = content.get(f)
        if data is None:
            continue
        if len(data) > MAX_BYTES:
            problems.append(f"{f}: too large ({len(data)} bytes) — data or outputs?")
            continue
        if f.endswith("check_public_safety.py"):
            continue
        text = data.decode("utf-8", errors="ignore")
        pats = SECRET_PATTERNS + ([UNQUOTED_CRED] if suf in TEXT_CONFIG_EXT else [])
        low = text.lower()
        cred_words = any(w in low for w in CRED_WORDS)
        for pat, what in pats:
            if what.startswith("hard-coded credential") and not cred_words:
                continue                                   # speed: the keyword cannot match
            if re.search(pat, text):
                problems.append(f"{f}: possible {what}")
        # v0.6: the export package's identifier regex (it used to require 5+ digits everywhere)
        if (PV.ID_RE_CODE if suf == ".py" else PV.ID_RE).search(text):
            problems.append(f"{f}: possible patient/stay identifier")
        # v0.6: small cells, in every CSV / JSON and in the reports under results/
        if suf in SMALL_CELL_ALWAYS or (suf in SMALL_CELL_RESULTS and f.startswith("results/")):
            cells = PV.scan_text(f, text, cohort_n)
            if cells:
                problems.append(f"{f}: {len(cells)} small cell(s) (< 10) published or derivable, e.g. {cells[0]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    try:  # never crash the git hook on a non-UTF-8 console (Windows cp1252)
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    argv = sys.argv[1:] if argv is None else argv
    all_files = "--all" in argv
    files = _files(all_files)
    content = _worktree(files) if all_files else _staged_blobs(files)
    problems = check(files, content)
    if problems:
        print("✖ DO NOT PUSH — problems found:\n  " + "\n  ".join(problems))
        return 1
    print(f"✔ {len(files)} files checked: no secrets, data or small cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
