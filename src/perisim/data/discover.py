"""Cohort discovery under the datasets root (run_pod.sh: <workspace>/data, else the KHDP mount
<workspace>/datasets) and schema checking.

Generates reports/schema_<cohort>.{md,json}: paths, expected vs. actual columns and item inventory
(with counts < minimum n suppressed). This is metadata, not patient data: it is used to fix
config/harmonization.yaml after the first run in the pod.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import polars as pl

from perisim.config import Config
from perisim.data.io import distinct_counts, find_table, read_header, resolve_columns

LOG = logging.getLogger("perisim.data.discover")
# Heavy non-tabular trees that never contain the cohorts. NOT "files": the standard PhysioNet layout is
# physionet.org/files/<database>/<version>/ (AUDIT A2).
_SKIP_DIRS = {"waveforms", "waves", "matched", "notes", "ecg", "cxr", "mimic-iv-ecg", "mimic-cxr", ".git",
              "__pycache__"}

# Tables whose item inventory is computed (cohort -> [(table, canonical column)])
INVENTORY = {
    "inspire": [("labs", "item"), ("ward_vitals", "item"), ("medications", "drug"), ("medications", "route"),
                ("operations", "department"), ("operations", "anesthesia_type")],
    "mimic": [("services", "curr_service"), ("admissions", "admission_type"), ("admissions", "discharge_location"),
              ("admissions", "admission_location")],
    "eicu": [("patient", "unit_admit_source"), ("patient", "unit_discharge_location"),
             ("patient", "hospital_discharge_location"), ("lab", "item"), ("infusiondrug", "drug"),
             ("respiratorycharting", "label")],
}


def _version_key(path: Path) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", str(path))]
    return tuple(nums)


def _marker_ok(d: Path, markers: list[str]) -> bool:
    return all(find_table(d, m) is not None for m in markers)


def _score(d: Path, markers: list[str]) -> tuple:
    p = find_table(d, markers[0])
    size = 0
    if p is not None:
        try:
            size = p.stat().st_size if p.is_file() else sum(x.stat().st_size for x in p.glob("*.parquet"))
        except OSError:
            pass
    demo = "demo" in str(d).lower()
    return (not demo, size, _version_key(d))


def discover_cohorts(cfg: Config) -> dict:
    """Returns {cohort: {"path": str|None, "candidates": [...], "source": "env"|"auto"|None}}."""
    paths = cfg.experiment["paths"]
    root = Path(paths["datasets_root"]).expanduser()
    max_depth = int(cfg.get("experiment.resources.discovery_max_depth", 7))
    budget = float(cfg.get("experiment.resources.discovery_time_budget_s", 300))
    result: dict[str, dict] = {}
    pending = {}
    for cohort in ("inspire", "mimic", "eicu"):
        explicit = str(paths.get(f"{cohort}_dir", "auto"))
        markers = cfg.harmonization["sources"][cohort]["marker_tables"]
        if explicit and explicit.lower() != "auto":
            p = Path(explicit).expanduser()
            ok = p.is_dir() and _marker_ok(p, markers)
            result[cohort] = {"path": str(p) if ok else None, "candidates": [str(p)], "source": "env",
                              "error": None if ok else f"{p} does not contain {markers}"}
        else:
            pending[cohort] = markers
            result[cohort] = {"path": None, "candidates": [], "source": "auto"}
    if pending and root.is_dir():
        t0 = time.time()
        found: dict[str, list[Path]] = {c: [] for c in pending}
        stack = [(root, 0)]
        while stack and time.time() - t0 < budget:
            d, depth = stack.pop()
            for cohort, markers in pending.items():
                if _marker_ok(d, markers):
                    found[cohort].append(d)
            if depth >= max_depth:
                continue
            try:
                subdirs = [e for e in os.scandir(d) if e.is_dir(follow_symlinks=True)]
            except OSError:
                continue
            for e in subdirs:
                if e.name.lower() not in _SKIP_DIRS and not e.name.startswith("."):
                    stack.append((Path(e.path), depth + 1))
        if stack:
            LOG.warning("Discovery stopped by time budget (%.0f s); use PERISIM_<COHORT>_DIR", budget)
        for cohort, cands in found.items():
            cands = sorted(set(cands), key=lambda d: _score(d, pending[cohort]), reverse=True)
            result[cohort]["candidates"] = [str(c) for c in cands]
            result[cohort]["path"] = str(cands[0]) if cands else None
    elif pending:
        LOG.warning("datasets_root=%s does not exist", root)
    try:
        top = sorted(e.name for e in os.scandir(root) if e.is_dir())[:200] if root.is_dir() else []
    except OSError:
        top = []
    result["_datasets_root"] = {"path": str(root), "top_level": top}
    for c in ("inspire", "mimic", "eicu"):
        LOG.info("Cohort %-7s → %s", c, result[c]["path"] or "NOT FOUND")
    return result


def schema_report(cfg: Config, cohort: str, base: Path, cache_dir: Path, min_n: int) -> dict:
    """Expected vs. actual columns per table + item inventory (counts < min_n suppressed)."""
    src = cfg.harmonization["sources"][cohort]
    rep: dict = {"cohort": cohort, "path": str(base), "tables": {}, "inventory": {}, "ok": True}
    for tname, tcfg in src["tables"].items():
        p = find_table(base, tcfg["file"])
        entry: dict = {"file": tcfg["file"], "found": p is not None}
        if p is not None:
            header = read_header(p)
            wanted = dict(tcfg.get("columns", {}))
            opt_wanted = {k: v for k, v in (tcfg.get("optional_columns") or {}).items() if v}
            for i, a in enumerate(tcfg.get("atc_columns", []) or []):   # the first ATC column is required, the rest optional
                (wanted if i == 0 else opt_wanted)[f"atc_{i}"] = a
            _, missing = resolve_columns(header, wanted)
            _, opt_missing = resolve_columns(header, opt_wanted)
            entry.update(path=str(p), columns=header, missing=missing, optional_missing=opt_missing,
                         size_mb=round((p.stat().st_size if p.is_file() else 0) / 1e6, 1))
            if missing:
                rep["ok"] = False
        rep["tables"][tname] = entry
    for tname, canon in INVENTORY.get(cohort, []):
        t = rep["tables"].get(tname, {})
        tc = src["tables"].get(tname, {})
        col = (tc.get("columns") or {}).get(canon) or (tc.get("optional_columns") or {}).get(canon)
        if not t.get("found") or not col or col in t.get("missing", []) + t.get("optional_missing", []):
            continue
        try:
            dc = distinct_counts(Path(t["path"]), col, cache_dir)
            rare = int((dc["n"] < min_n).sum())
            dc = dc.filter(pl.col("n") >= min_n)       # values seen < 10 times are not listed (may be free text, A6)
            rep["inventory"][f"{tname}.{col}"] = [{"value": r["value"], "n": int(r["n"])} for r in dc.to_dicts()] + \
                ([{"value": f"({rare} other values with n < {min_n})", "n": None}] if rare else [])
        except Exception as exc:  # inventory: informational, never blocking
            rep["inventory"][f"{tname}.{col}"] = [{"error": str(exc)}]
    if cohort == "mimic":
        rep["itemid_labels"] = _mimic_item_labels(cfg, base, cache_dir)
    return rep


def _mimic_item_labels(cfg: Config, base: Path, cache_dir: Path) -> dict:
    """Label of each configured itemid (d_items / d_labitems), to verify them at a glance."""
    from perisim.data.io import load_table

    src = cfg.harmonization["sources"]["mimic"]["tables"]
    labels: dict[str, str] = {}
    for t in ("d_items", "d_labitems"):
        p = find_table(base, src[t]["file"])
        if p is None:
            continue
        df = load_table(p, src[t]["columns"], None, cache_dir)
        labels.update({str(r["itemid"]).strip(): r["label"] for r in df.to_dicts()})
    out = {}
    for name, spec in cfg.harmonization["variables"].items():
        s = (spec.get("sources") or {}).get("mimic")
        if s:
            out[name] = {str(i): labels.get(str(i), "NOT FOUND?") for i in s.get("items", [])}
    for k, s in cfg.harmonization["sources"]["mimic"].get("support_items", {}).items():
        out[f"support_{k}"] = {str(i): labels.get(str(i), "NOT FOUND?") for i in s["items"]}
    return out


def schema_markdown(rep: dict) -> str:
    lines = [f"# Schema — {rep['cohort']}", "", f"Path: `{rep['path']}`", "",
             "Status: " + ("OK" if rep["ok"] else "**MISSING COLUMNS** (fix config/harmonization.yaml)"), ""]
    lines += ["| table | file | found | missing columns | missing optional |", "|---|---|---|---|---|"]
    for t, e in rep["tables"].items():
        lines.append(f"| {t} | {e['file']} | {'yes' if e['found'] else 'NO'} | {', '.join(e.get('missing', [])) or '—'} "
                     f"| {', '.join(e.get('optional_missing', [])) or '—'} |")
    lines.append("")
    for t, e in rep["tables"].items():
        if e.get("columns"):
            lines.append(f"- **{t}** ({e.get('size_mb', '?')} MB): {', '.join(e['columns'])}")
    if rep.get("itemid_labels"):
        lines += ["", "## Labels of the configured itemids (MIMIC-IV)", ""]
        for v, m in rep["itemid_labels"].items():
            lines.append(f"- {v}: " + "; ".join(f"{k} = {lab}" for k, lab in m.items()))
    for k, vals in rep.get("inventory", {}).items():
        lines += ["", f"## Inventory `{k}` (top {len(vals)})", ""]
        lines.append(", ".join(f"{v.get('value')} ({v.get('n')})" if "error" not in v else f"ERROR {v['error']}"
                               for v in vals))
    return "\n".join(lines) + "\n"
