"""Removes derivable small cells from the aggregates ALREADY published under results/ (v0.6; software audit
2026-11-05, item 1). The published files cannot be regenerated outside the pod (no patient data), so the leaking
values are blanked in place: a value that publishes, or lets derive, a count of 1–9 (directly, by subtraction from a
published total, or from a percentage / share next to its n) becomes null / NaN, together with whatever would let
it be recovered (its complement, the other bins of the same variable). No other number is changed; metrics are
never touched.

What is cleaned:
- every CSV / JSON record (CSV row, JSON object) flagged by perisim.evaluation.privacy.scan_record: person counts
  1–9, complements n − count of 1–9, percentages / shares that next to their n imply a count or a complement of 1–9
  (e.g. ood_pct of the per-hospital rows of transport analysis F, % outside INSPIRE's range of analysis E, whose
  denominator is the cohort's non-missing n); a nulled "rate" also nulls its "ci";
- bin shares that add up to 1 (t2_shift_shares, t2_drift_points): complementary suppression, as the fixed
  scripts/posthoc_transport.suppressed_shares does, with each group's n (cohort n × non-missing share); a bin hidden
  in t2_drift_points is hidden in t2_drift_missing_points too;
- transport.json accumulation_summary: the ood_pct mean / quantiles of every C group with a blanked row;
- Markdown / HTML report cells "P% (CI; n N)" whose percentage of N implies a count < 10 (the n is kept);
- flow diagrams of the completeness audit rounded from a small step on (the old privacy.flow): the rounded rows'
  counts and exclusions are blanked (with the exact count before them and the exact cohort n they let a small
  exclusion be recovered; the merged, exact flow is results/t2/flow/).

Usage:  python scripts/sanitize_results.py [--check] [--root results]
        --check: report only (exit 1 if anything would change)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perisim.evaluation import privacy as PV  # noqa: E402

MIN_N = 10


# --------------------------------------------------------------------------- helpers
def cohort_sizes(root: Path) -> dict:
    """Analysis n of each cohort: the last step of the published extraction flow (results/t2/flow)."""
    out = {}
    for c in PV.COHORTS:
        f = root / "t2" / "flow" / f"flow_{c}.json"
        if f.exists():
            steps = json.loads(f.read_text(encoding="utf-8"))
            if steps and isinstance(steps[-1].get("n_stays"), int):
                out[c] = steps[-1]["n_stays"]
    return out


def _read_csv(path: Path) -> tuple[list[str], list[list[str]], str]:
    text = path.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    return rows[0], rows[1:], text


def _write_csv(path: Path, head: list[str], body: list[list[str]], original: str) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n" if "\r\n" in original else "\n")
    w.writerow(head)
    w.writerows(body)
    out = buf.getvalue()
    if not original.endswith(("\n", "\r")):
        out = out.rstrip("\r\n")
    return out


def _null_token(text: str) -> str:
    return "NaN" if re.search(r"(^|,)NaN(,|$)", text, re.M) else ""


def _f(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("nan")


def infer_n(shares: list[float], candidates: list[float]) -> float | None:
    """The group n among `candidates` for which every shown share is an integer count."""
    shown = [s for s in shares if s == s]
    for c in candidates:
        if c and all(abs(s * c - round(s * c)) < 1e-6 * max(1.0, c / 1000) for s in shown):
            return c
    return None


def search_n(shares: list[float], n_max: int = 300_000) -> int | None:
    """Smallest n for which every share is an integer count (a complete group of bins: its n)."""
    import numpy as np
    sh = np.array([x for x in shares if x == x and x > 0])
    if sh.size == 0:
        return None
    for lo in range(1, n_max, 50_000):
        n = np.arange(lo, min(lo + 50_000, n_max), dtype=float)
        k = sh[None, :] * n[:, None]
        ok = np.all(np.abs(k - np.round(k)) < 1e-6 * np.maximum(1.0, n[:, None] / 1000), axis=1)
        if ok.any():
            return int(n[np.argmax(ok)])
    return None


def complementary(shares: dict, n: float | None) -> list:
    """Bins (keys of `shares`, value share or NaN) to hide on top of the hidden ones so that the hidden bins add up
    to ≥ MIN_N stays. With n unknown, the smallest shown bin (≥ MIN_N stays by construction) is hidden on top."""
    hidden = [k for k, v in shares.items() if v != v]
    shown = sorted(((v, k) for k, v in shares.items() if v == v), key=lambda t: t[0])
    mass = 1.0 - sum(v for v, _ in shown)
    extra = []
    if not hidden or mass <= 1e-9:
        return extra
    while shown:
        if n is not None:
            if mass * n >= MIN_N - 1e-6:
                break
        elif extra:                       # n unknown: one shown bin (≥ MIN_N stays) hidden on top is enough
            break
        v, k = shown.pop(0)
        extra.append(k)
        mass += v
    return extra


# --------------------------------------------------------------------------- generic records
def clean_record(rec: dict, cohort_n: dict, null) -> list[str]:
    """Blanks the flagged fields of one record in place; returns 'field: reason' strings."""
    done = []
    for k, why in PV.scan_record(rec, cohort_n, MIN_N):
        rec[k] = null
        done.append(f"{k}: {why}")
        if k in ("rate", "share", "prevalence", "obs") and "ci" in rec:   # its CI gives the count back
            rec["ci"] = None
            done.append("ci: CI of a blanked proportion")
    return done


def clean_csv(path: Path, cohort_n: dict, extra_n: dict | None = None) -> tuple[str | None, list[str], list]:
    head, body, text = _read_csv(path)
    null = _null_token(text)
    log, blanked = [], []
    for i, r in enumerate(body):
        if len(r) != len(head):
            continue
        rec = dict(zip(head, r))
        ch = clean_record(rec, cohort_n, null)
        if ch:
            body[i] = [rec[h] for h in head]
            log += [f"row {i + 2} {c}" for c in ch]
            blanked.append(rec)
    hidden_bins = []
    if "share" in head and "variable" in head:
        bin_col = next((c for c in ("value", "bin", "label") if c in head), None)
        gi = [head.index(c) for c in ("variable", "site") if c in head]
        bi, si = head.index(bin_col), head.index("share")
        groups: dict = {}
        for r in body:
            groups.setdefault(tuple(r[i] for i in gi), {})[r[bi]] = _f(r[si])
        site_n: dict = {}                 # n of each site from its complete groups (no hidden bin)
        for g, sh in groups.items():
            if len(g) > 1 and g[1] not in site_n and all(v == v for v in sh.values()) and len(sh) > 2:
                site_n[g[1]] = search_n(list(sh.values()))
        for g, sh in groups.items():
            site = g[1] if len(g) > 1 else None
            cands = [x for x in ((extra_n or {}).get(g), cohort_n.get(site), site_n.get(site)) if x]
            n = infer_n(list(sh.values()), cands)
            extra = complementary(sh, n)
            if extra:
                for r in body:
                    if tuple(r[i] for i in gi) == g and r[bi] in extra:
                        r[si] = null
                hidden_bins += [(g, b) for b in extra]
                log.append(f"group {'/'.join(g)}: bins {extra} hidden (complementary suppression; group n "
                           f"{'unknown' if n is None else int(round(n))})")
    if not log:
        return None, [], []
    return _write_csv(path, head, body, text), log, blanked


def _json_format(text: str, obj) -> tuple:
    for ind in (2, 1, None, 4):
        for ea in (False, True):
            for end in ("", "\n"):
                if json.dumps(obj, indent=ind, ensure_ascii=ea) + end == text:
                    return ind, ea, end
    return None


def clean_json(path: Path, cohort_n: dict) -> tuple[str | None, list[str], object]:
    text = path.read_text(encoding="utf-8")
    obj = PV.load_json_keep(text)
    fmt = _json_format(text, json.loads(text))
    log = []
    for p, d in PV.walk_json(obj):
        log += [f"{p or '.'} {c}" for c in clean_record(d, cohort_n, None)]
    if not log:
        return None, [], obj
    if fmt is None:
        raise SystemExit(f"{path}: unknown JSON layout; not rewritten")
    ind, ea, end = fmt
    return json.dumps(obj, indent=ind, ensure_ascii=ea) + end, log, obj


# --------------------------------------------------------------------------- specific files
def shift_share_n(root: Path, cohort_n: dict) -> dict:
    """{(variable, site): non-missing n} from t2_shift_variables (cohort n × (1 − missing %))."""
    f = root / "t2" / "results" / "posthoc" / "t2_shift_variables.csv"
    out = {}
    if not f.exists():
        return out
    head, body, _ = _read_csv(f)
    for r in body:
        rec = dict(zip(head, r))
        for c, n in cohort_n.items():
            mp = _f(rec.get(f"missing_pct_{c}", ""))
            if mp == mp:
                out[(rec["variable"], c)] = round(n * (1 - mp / 100.0))
    return out


def accumulation_groups(rows: list[dict]) -> set:
    return {(r["cohort"], r["order"], r["start"], str(r["gamma"]), str(int(float(r["n_local"])))) for r in rows}


def clean_accumulation_summary(obj, groups: set) -> list[str]:
    log = []
    for p, d in PV.walk_json(obj):
        if not isinstance(d, dict) or "ood_pct_mean" not in d or "n_local" not in d:
            continue
        key = (d.get("cohort"), d.get("order"), d.get("start"), str(d.get("gamma")), str(int(d.get("n_local"))))
        if key in groups:
            for k in ("ood_pct_mean", "ood_pct_low", "ood_pct_high"):
                if d.get(k) is not None:
                    d[k] = None
                    log.append(f"{p} {k}: summary of C rows whose ood_pct was blanked")
    return log


def clean_md_cells(text: str) -> tuple[str, list[str]]:
    log = []

    def sub(m):
        cell = m.group(0)
        if PV.scan_text_cells(cell, MIN_N):
            log.append(f"cell '{cell}' → percentage and CI suppressed")
            return f"suppressed (n {m.group(3)})"
        return cell
    return PV.PCT_N_CELL.sub(sub, text), log


def clean_completeness_flow(obj) -> list[str]:
    """Rows of a completeness flow rounded by the old privacy.flow: counts and exclusions blanked."""
    log = []
    fl = obj.get("flow") if isinstance(obj, dict) else None
    for i, r in enumerate(fl or []):
        if r.get("rounded"):
            for k in ("n_stays", "n_subjects", "excluded"):
                if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool):
                    r[k] = None
                    log.append(f".flow[{i}] {k}: row rounded from a small step on (recoverable by subtraction)")
    return log


def clean_completeness_html(text: str, steps: list[str]) -> tuple[str, list[str]]:
    """The flow table of a completeness report: the cells of the rounded steps → '—'."""
    log = []
    for st in steps:
        pat = re.compile(r"(<tr><td>" + re.escape(st.replace("<", "&lt;").replace(">", "&gt;")) +
                         r"</td>)(?=(?:<td>[^<]*</td>){3}</tr>)(?!(?:<td>—</td>){3})(<td>[^<]*</td>){3}(</tr>)")
        new, k = pat.subn(lambda m: m.group(1) + "<td>—</td>" * 3 + m.group(3), text, count=1)
        if k:
            text = new
            log.append(f"flow row '{st}': counts blanked")
    return text, log


# --------------------------------------------------------------------------- main
def sanitize(root: Path, write: bool = True) -> dict:
    cohort_n = cohort_sizes(root)
    changes: dict = {}
    pending: dict = {}

    def put(path: Path, text: str, log: list[str]):
        pending[path] = text
        changes.setdefault(str(path.relative_to(root.parent) if root.parent in path.parents else path), []).extend(log)

    share_n = shift_share_n(root, cohort_n)
    acc_groups: set = set()
    for f in sorted(root.rglob("*.csv")):
        new, log, blanked = clean_csv(f, cohort_n, share_n if f.name == "t2_shift_shares.csv" else None)
        if f.name.startswith("t2_C_accumulation"):
            acc_groups |= accumulation_groups(blanked)
        if new is not None:
            put(f, new, log)
    # a bin hidden in t2_drift_points: hidden in t2_drift_missing_points too
    dp = root / "t2" / "results" / "posthoc_drift" / "t2_drift_points.csv"
    dm = root / "t2" / "results" / "posthoc_drift_missing" / "t2_drift_missing_points.csv"
    if dp.exists() and dm.exists():
        def current(path):
            t = pending.get(path) or path.read_text(encoding="utf-8")
            rows = list(csv.reader(io.StringIO(t)))
            return rows[0], rows[1:], t
        h1, b1, _ = current(dp)
        hid = {(r[h1.index("site")], r[h1.index("variable")], r[h1.index("label")]) for r in b1
               if _f(r[h1.index("share")]) != _f(r[h1.index("share")])}
        h2, b2, t2 = current(dm)
        log = []
        for i, r in enumerate(b2):
            key = (r[h2.index("site")], r[h2.index("variable")], r[h2.index("label")])
            if key in hid and _f(r[h2.index("share")]) == _f(r[h2.index("share")]):
                r[h2.index("share")] = _null_token(t2)
                log.append(f"row {i + 2} share: bin hidden in t2_drift_points")
        if log:
            put(dm, _write_csv(dm, h2, b2, t2), log)
    rounded_steps: dict = {}
    for f in sorted(root.rglob("*.json")):
        new, log, obj = clean_json(f, cohort_n)
        extra = []
        if f.name == "transport.json" and acc_groups:
            extra += clean_accumulation_summary(obj, acc_groups)
        if f.name.startswith("completeness_"):
            fl = obj.get("flow") if isinstance(obj, dict) else None
            rounded_steps[f.stem] = [r["step"] for r in (fl or []) if r.get("rounded")]
            extra += clean_completeness_flow(obj)
        if extra:
            text = f.read_text(encoding="utf-8")
            ind, ea, end = _json_format(text, json.loads(text))
            new = json.dumps(obj, indent=ind, ensure_ascii=ea) + end
            log = log + extra
        if new is not None:
            put(f, new, log)
    for f in sorted(list(root.rglob("*.md")) + list(root.rglob("*.html"))):
        if f.name in ("REPORT.md", "README.md"):
            continue                                       # not edited here (rule: reports are the lead's)
        text = f.read_text(encoding="utf-8")
        new, log = clean_md_cells(text)
        if f.suffix == ".html" and f.stem in rounded_steps and rounded_steps[f.stem]:
            new, l2 = clean_completeness_html(new, rounded_steps[f.stem])
            log += l2
        if log:
            put(f, new, log)
    if write:
        for p, t in pending.items():
            p.write_text(t, encoding="utf-8", newline="")
    return changes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(ROOT / "results"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    ch = sanitize(Path(a.root), write=not a.check)
    for f, log in sorted(ch.items()):
        print(f"{f}: {len(log)} change(s)")
        for line in log[:5]:
            print(f"    {line}")
        if len(log) > 5:
            print(f"    … {len(log) - 5} more")
    print(f"{len(ch)} file(s) {'would change' if a.check else 'changed'}.")
    return 1 if (a.check and ch) else 0


if __name__ == "__main__":
    sys.exit(main())
