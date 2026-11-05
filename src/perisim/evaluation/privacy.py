"""Small-cell suppression for everything that may leave the pod (SPECS §9; KHDP export policy; AUDIT A6).

Rules applied consistently:
- a count 1–9 is replaced by "<10" (primary suppression);
- its complement is suppressed too when the total is known (n − count < 10 reveals a small group);
- a set of counts with a known total and exactly one suppressed cell gets the next-smallest cell suppressed as
  well (secondary suppression), so that no cell can be recovered by subtraction;
- prevalences / percentages are shown only when both the numerator and its complement are ≥ 10;
- flow diagrams (v0.6): a step that excludes < 10 stays, or loses < 10 patients, is merged with the next step, so no
  exclusion can be recovered by subtracting consecutive steps (rounding did not protect: see `flow`);
- (v0.6) a published table is scanned for derivable small cells before it leaves (`scan_record`, `scan_shares`,
  `scan_csv_text`, `scan_json_text`): person counts 1–9, complements n − count 1–9, and percentages / shares that,
  next to their denominator, imply a count (or a complement) of 1–9.
"""

from __future__ import annotations

import csv
import functools
import io
import json
import math
import re
from decimal import Decimal, InvalidOperation

SUPPRESSED = "<10"
# Internal stay / patient identifiers (cohort prefix + numeric id). v0.6: the one regex used by the export package,
# publish(), publish_results.py and scripts/check_public_safety.py (the latter used to require 5+ digits).
ID_RE = re.compile(r"\b(INSPIRE|MIMIC|EICU)-\d")
# The same identifiers with 5+ digits: the variant for source code, whose tests and docstrings use short synthetic
# identifiers ("MIMIC-11").
ID_RE_CODE = re.compile(ID_RE.pattern + r"\d{4}")


def cell(n: int | None, min_n: int = 10, total: int | None = None) -> int | str | None:
    """Primary suppression of a count (and of its complement when the total is known)."""
    if n is None:
        return None
    n = int(n)
    if 0 < n < min_n:
        return f"<{min_n}"
    if total is not None and 0 < int(total) - n < min_n:
        return f">{int(total) - min_n}"
    return n


def share(k: int | None, n: int | None, min_n: int = 10, digits: int = 1) -> float | None:
    """Percentage k/n, only when both k and n − k are ≥ min_n (or exactly 0)."""
    if k is None or not n:
        return None
    k, n = int(k), int(n)
    if (0 < k < min_n) or (0 < n - k < min_n):
        return None
    return round(100.0 * k / n, digits)


def counts(cells: dict, min_n: int = 10) -> dict:
    """Primary + secondary suppression of a set of counts that add up to a published total.

    v0.6: every suppressed cell gets the same plain label "<{min_n}" (the old secondary label "<{rest+1}" gave an
    upper bound that, with the published total, recovered the cells exactly), and the suppressed cells must add up
    to ≥ min_n (or be all the non-zero cells), so that neither a single cell nor the sum of the suppressed ones can
    be read off the total by subtraction. Further cells are suppressed, smallest first, as needed."""
    label = f"<{min_n}"
    out = {k: (v if v is None or not 0 < int(v) < min_n else label) for k, v in cells.items()}
    small = [k for k, v in out.items() if isinstance(v, str)]
    if small:
        hidden = sum(int(cells[k]) for k in small)
        rest = sorted(((int(v), k) for k, v in out.items() if not isinstance(v, str) and v), key=lambda t: t[0])
        # v0.6: complementary suppression until one cell alone cannot be recovered and the hidden total is ≥ min_n
        while rest and (len(small) == 1 or hidden < min_n):
            v, k = rest.pop(0)
            out[k] = label
            small.append(k)
            hidden += v
    return out


def flow(rows: list[dict], min_n: int = 10) -> list[dict]:
    """Exportable flow diagram (see the module docstring).

    v0.6: merge rule instead of rounding. Rounding from the first small step on did not protect: the exact count
    before it, the exclusions ≥ min_n shown exactly after it and the exact final n (published everywhere as the
    cohort n) recovered the small exclusion by subtraction. Now a step that excludes 1 … min_n − 1 stays, or whose
    number of PATIENTS falls by 1 … min_n − 1, is merged with the next step (a small last step with the previous
    one), so every published difference between consecutive rows is 0 or ≥ min_n. "rounded" is kept (always False)
    for the readers of earlier reports."""
    def small(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < min_n

    def drop(a, b, key):
        x, y = a.get(key), b.get(key)
        return x - y if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None

    merged, carry = [], None
    for r in rows:
        r = {"step": r["step"], "n_stays": r.get("n_stays"), "n_subjects": r.get("n_subjects"),
             "excluded": r.get("excluded")}
        if carry is not None:
            r["step"] = carry["step"] + "; " + r["step"]
            r["excluded"] = (carry["excluded"] or 0) + (r["excluded"] or 0)
            carry = None
        prev = merged[-1] if merged else None
        if small(r["excluded"]) or (prev is not None and (small(drop(prev, r, "n_stays"))
                                                          or small(drop(prev, r, "n_subjects")))):
            carry = r
            continue
        merged.append(r)
    if carry is not None:                       # a small last step: merge it backwards
        if merged:
            last = merged.pop()
            merged.append({"step": last["step"] + "; " + carry["step"], "n_stays": carry["n_stays"],
                           "n_subjects": carry["n_subjects"],
                           "excluded": (last["excluded"] or 0) + (carry["excluded"] or 0)})
        else:
            merged.append(carry)
    return [{"step": r["step"], "n_stays": cell(r["n_stays"], min_n), "n_subjects": cell(r["n_subjects"], min_n),
             "excluded": cell(r["excluded"], min_n), "rounded": False} for r in merged]


# --------------------------------------------------------------------------- v0.6: scan of published aggregates
# Person counts: n, n_<something>, <something>_n, stays, patients, subjects, excluded, count(s), totals.
COUNT_KEY = re.compile(r"^(n|N|count|counts|stays|patients|subjects|excluded|total)$|^n_|_n$|_count$|_stays$|"
                       r"_patients$|_subjects$|^patients_|^stays_|_pos$")
# ...that are not counts of people (design sizes, numbers of variables / folds / models / hospitals, …)
NOT_COUNT_KEY = re.compile(r"^n_(folds|models|models_run|features|estimators|dims|bootstrap|values|shaped|failing|"
                           r"passing|competitive|tests|attempts|accepted|anchors|items|max|min|hospitals|queries_differ|"
                           r"threads|jobs|reps|bins|classes|variables|blocks|splits|seeds|common|to_match_.*|"
                           r"full_within_.*)$|^n_cv$|threads|_k$|^k$")
PCT_KEY = re.compile(r"(^|_)(pct|percent)(_|$)")
FRAC_KEY = re.compile(r"(^|_)(share|prevalence|prev|fraction|proportion|ood_rate|rate|obs|observed)(_|$)")
NOT_FRAC_KEY = re.compile(r"share_of_.*gain|share_of_neighbours|learning_rate|acceptance_rate|_ci_|ci_low|ci_high|"
                          r"_p_two_sided|_minus_|shape_|^share_true$|alarm_share|trigger_share|share_cal_out|"
                          r"eligible_share|^d_|_rms")
# Denominators looked for in the same record, in this order.
DENOMS = ("n", "N", "n_test", "n_index", "n_index_stays", "n_stays", "n_patients", "total", "test_stays",
          "test_patients", "n_eval_landmarks", "n_landmarks", "patients", "stays")
COMPLEMENT_KEY = re.compile(r"pos|survivor|detected|with|excluded|count|events|^n_[a-z]+_only$")
COHORTS = ("inspire", "mimic", "eicu")


def _num(v):
    """(float value, decimals shown) of a scalar, or None. Strings are parsed as numbers (CSV cells)."""
    if isinstance(v, bool) or v is None:
        return None
    txt = v.txt if hasattr(v, "txt") else (v if isinstance(v, str) else repr(v))
    try:
        f = float(txt)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    t = txt.strip()
    if "e" not in t and "E" not in t:
        i = t.find(".")
        return f, (0 if i < 0 else len(t) - i - 1)
    try:
        e = Decimal(t).as_tuple().exponent
        d = max(0, -int(e)) if isinstance(e, int) else 0
    except InvalidOperation:
        d = 17
    return f, d


@functools.lru_cache(maxsize=4096)
def _relevant(k: str) -> bool:
    """A field the scan looks at: a count, a percentage / share, a denominator or a missing % (speed)."""
    return k in DENOMS or is_count_key(k) or _scale(k) is not None or k.startswith("missing_pct")


@functools.lru_cache(maxsize=4096)
def is_count_key(k: str) -> bool:
    return bool(COUNT_KEY.search(k)) and not NOT_COUNT_KEY.search(k) and not k.startswith("≥")


@functools.lru_cache(maxsize=4096)
def _scale(k: str) -> float | None:
    if NOT_FRAC_KEY.search(k):
        return None
    if PCT_KEY.search(k):
        return 100.0
    if FRAC_KEY.search(k):
        return 1.0
    return None


def implied_small(v: float, d: int, n: float, scale: float, min_n: int = 10) -> bool:
    """True when a percentage / share `v` (shown with `d` decimals) of a denominator `n` implies that the count, or
    its complement n − count, is 1 … min_n − 1: every integer count compatible with the shown value is small. For a
    full-precision value next to a denominator rounded to 10, the denominators n − 5 … n + 5 are tried too (an exact
    k/m reveals m)."""
    if n is None or n <= 0:
        return False
    cands = [float(n)]
    if d >= 8 and float(n).is_integer() and int(n) % 10 == 0:
        cands += [float(m) for m in range(int(n) - 5, int(n) + 6) if m != n and m > 0]
    half = 0.5 * 10.0 ** (-min(d, 15))
    for m in cands:
        lo, hi = (v - half) * m / scale, (v + half) * m / scale
        tol = 1e-6 * max(1.0, m / 1000)
        a, b = math.ceil(lo - tol), math.floor(hi + tol)
        if b < a:
            continue
        if b - a > 2 * min_n:
            continue
        ks = range(max(a, 0), min(b, int(m)) + 1)
        if not ks:
            continue
        if all(0 < k < min_n for k in ks) or all(0 < m - k < min_n for k in ks):
            return True
    return False


def scan_record(rec: dict, cohort_n: dict | None = None, min_n: int = 10) -> list[tuple[str, str]]:
    """Fields of one record (a CSV row; the scalar items of a JSON object) that publish or let derive a small cell.
    Returns [(field, reason)]. `cohort_n` ({cohort: n stays}) is the denominator of `*_pct_<cohort>` fields of a
    record without its own n (per-variable missingness: n; % outside a range: the non-missing n)."""
    out = []
    vals = {k: _num(v) for k, v in rec.items()
            if _relevant(k) and not isinstance(v, (dict, list))}
    vals = {k: v for k, v in vals.items() if v is not None}
    # {"n": count, "pct": share} (audit.py, completeness): there "n" is the numerator, not the denominator
    skip = {"n"} if any(k == "pct" or k.startswith("prev") for k in vals) else set()
    den_key = next((k for k in DENOMS if k in vals and vals[k][0] > 0 and k not in skip), None)
    den = vals[den_key][0] if den_key else None
    for k, (v, d) in vals.items():
        if is_count_key(k) and float(v).is_integer():
            if 0 < v < min_n:
                out.append((k, f"count {int(v)}"))
            elif den is not None and k != den_key and k not in DENOMS and COMPLEMENT_KEY.search(k) \
                    and 0 < den - v < min_n:
                out.append((k, f"complement {den_key} − {k} = {int(den - v)}"))
            continue
        sc = _scale(k)
        if sc is None:
            continue
        n_use = den
        if n_use is None and cohort_n:
            coh = next((c for c in COHORTS if k.endswith("_" + c)), None)
            if coh and cohort_n.get(coh):
                n_use = float(cohort_n[coh])
                if "outside" in k:                      # % of the non-missing values
                    mk = f"missing_pct_{coh}"
                    if mk not in vals:
                        n_use = None
                    else:
                        n_use = n_use * (1 - vals[mk][0] / 100.0)
                        n_use = float(round(n_use)) if abs(n_use - round(n_use)) < 1e-6 * max(n_use, 1) else n_use
        if n_use and implied_small(v, d, n_use, sc, min_n):
            out.append((k, f"{k}={v:g} of n={n_use:g} implies a count or complement < {min_n}"))
    return out


def scan_shares(groups: dict, n_ref: dict | None = None, min_n: int = 10) -> list[tuple[object, str]]:
    """Bin shares that add up to 1 within a group ({group: [share or NaN/None, …]}): a group whose hidden bins can
    be recovered by 1 − Σ(shown), i.e. exactly one hidden bin, or hidden bins adding up to < min_n stays when the
    group's n is known (`n_ref` {group: n}). Returns [(group, reason)]."""
    out = []
    for g, sh in groups.items():
        vals = [None if (x is None or (isinstance(x, float) and x != x)) else float(x) for x in sh]
        hid = sum(v is None for v in vals)
        if not hid or hid == len(vals):
            continue
        mass = 1.0 - sum(v for v in vals if v is not None)
        if mass <= 1e-9:
            continue
        if hid == 1:
            out.append((g, "one hidden bin recoverable as 1 − Σ(shown)"))
        elif n_ref and n_ref.get(g) and mass * n_ref[g] < min_n - 0.5:
            out.append((g, f"hidden bins add up to {mass * n_ref[g]:.1f} < {min_n} stays"))
    return out


class _F(float):
    """A JSON float that remembers its text (number of decimals shown)."""
    txt: str

    def __new__(cls, s):
        o = float.__new__(cls, s)
        o.txt = s
        return o


def load_json_keep(text: str):
    return json.loads(text, parse_float=_F)


def walk_json(o, path: str = ""):
    """(path, dict) for every JSON object."""
    if isinstance(o, dict):
        yield path, o
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                yield from walk_json(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk_json(v, f"{path}[{i}]")


def share_groups(header: list[str], rows: list[list[str]]) -> dict:
    """Groups of a long table of bin shares (columns share + value/bin + variable [+ site]); bins repeated over
    other columns (component, class) are counted once."""
    if "share" not in header or "variable" not in header:
        return {}
    bin_col = next((c for c in ("bin", "value", "label") if c in header), None)
    if bin_col is None:
        return {}
    gi = [header.index(c) for c in ("variable", "site") if c in header]
    bi, si = header.index(bin_col), header.index("share")
    g: dict = {}
    for r in rows:
        if len(r) != len(header):
            continue
        key = tuple(r[i] for i in gi)
        x = _num(r[si])
        g.setdefault(key, {})[r[bi]] = None if x is None else x[0]
    return {k: list(v.values()) for k, v in g.items()}


def scan_csv_text(text: str, cohort_n: dict | None = None, min_n: int = 10, full_shares: bool = True) -> list[str]:
    """Small-cell findings of a CSV table (see scan_record / scan_shares). `full_shares`: the share column of the
    table lists every bin of a variable (so the shares add up to 1)."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    head, body = rows[0], rows[1:]
    out = []
    for i, r in enumerate(body):
        for k, why in scan_record(dict(zip(head, r)), cohort_n, min_n):
            out.append(f"row {i + 2} {k}: {why}")
    if full_shares:
        for g, why in scan_shares(share_groups(head, body), None, min_n):
            out.append(f"group {'/'.join(g)}: {why}")
    return out


def scan_json_text(text: str, cohort_n: dict | None = None, min_n: int = 10) -> list[str]:
    try:
        o = load_json_keep(text)
    except ValueError:
        return []
    return [f"{p or '.'} {k}: {why}" for p, d in walk_json(o) for k, why in scan_record(d, cohort_n, min_n)]


# Markdown / HTML reports: "12.7% (7.4%–19.5%; n 200)" cells and simple tables.
PCT_N_CELL = re.compile(r"(\d+(?:\.\d+)?)\s?%\s*\(([^;()]*);\s*n\s*=?\s*([\d,]+)\)")


def _cell_value(c: str) -> str:
    c = re.sub(r"<[^>]+>", "", c).strip().replace(",", "").replace(" ", "")
    return c[:-1].strip() if c.endswith("%") else c


def _header_key(h: str) -> str:
    h = re.sub(r"<[^>]+>", "", h).strip().lower()
    if h in ("%", "pct", "percent") or h.startswith("% "):
        return "pct"
    return re.sub(r"[^a-z0-9≥%]+", "_", h).strip("_")


def md_tables(text: str):
    """(line index of the header, header keys, [(line index, cells)]) of every pipe table."""
    lines = text.split("\n")
    i = 0
    while i < len(lines) - 1:
        if lines[i].lstrip().startswith("|") and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            head = [_header_key(h) for h in lines[i].strip().strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append((j, [c for c in lines[j].strip().strip("|").split("|")]))
                j += 1
            yield i, head, rows
            i = j
        else:
            i += 1


def scan_text_cells(text: str, min_n: int = 10) -> list[tuple[str, str]]:
    """'P% (CI; n N)' cells whose percentage of N (N possibly rounded to 10) implies a count or complement < min_n.
    Returns [(matched text, reason)]."""
    out = []
    for m in PCT_N_CELL.finditer(text):
        v, d = _num(m.group(1))
        n = float(m.group(3).replace(",", ""))
        if implied_small(v, d, n, 100.0, min_n) or _rounded_n_small(v, d, n, min_n):
            out.append((m.group(0), f"{m.group(1)}% of n {m.group(3)} implies a count or complement < {min_n}"))
    return out


def _rounded_n_small(v: float, d: int, n: float, min_n: int) -> bool:
    """As implied_small for a rounded percentage next to an n rounded to the nearest 10."""
    if not (float(n).is_integer() and int(n) % 10 == 0):
        return False
    half = 0.5 * 10.0 ** (-d)
    ks = set()
    for m in range(max(int(n) - 5, 1), int(n) + 5):
        lo, hi = (v - half) * m / 100, (v + half) * m / 100
        ks |= {(k, m) for k in range(max(math.ceil(lo - 1e-9), 0), min(math.floor(hi + 1e-9), m) + 1)}
    return bool(ks) and (all(0 < k < min_n for k, _ in ks) or all(0 < m - k < min_n for k, m in ks))


def scan_md_text(text: str, cohort_n: dict | None = None, min_n: int = 10) -> list[str]:
    out = [f"cell '{c}': {why}" for c, why in scan_text_cells(text, min_n)]
    for _, head, rows in md_tables(text):
        for j, cells in rows:
            if len(cells) != len(head):
                continue
            rec = {h: _cell_value(c) for h, c in zip(head, cells) if h}
            for k, why in scan_record(rec, cohort_n, min_n):
                out.append(f"line {j + 1} {k}: {why}")
    return out


def html_tables(text: str):
    """(header keys, [cells]) of every <table> of a generated report (evaluation/html.py layout)."""
    for t in re.finditer(r"<table>(.*?)</table>", text, re.S):
        head = [_header_key(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", t.group(1), re.S)]
        body = re.search(r"<tbody>(.*?)</tbody>", t.group(1), re.S)
        rows = [re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
                for r in re.findall(r"<tr[^>]*>(.*?)</tr>", body.group(1) if body else "", re.S)]
        yield head, rows


def scan_html_text(text: str, cohort_n: dict | None = None, min_n: int = 10) -> list[str]:
    out = [f"cell '{c}': {why}" for c, why in scan_text_cells(re.sub(r"<[^>]+>", " ", text), min_n)]
    for head, rows in html_tables(text):
        for i, cells in enumerate(rows):
            if len(cells) != len(head):
                continue
            rec = {h: _cell_value(c) for h, c in zip(head, cells) if h}
            for k, why in scan_record(rec, cohort_n, min_n):
                out.append(f"table row {i + 1} {k}: {why}")
    return out


def scan_text(name: str, text: str, cohort_n: dict | None = None, min_n: int = 10) -> list[str]:
    """Small-cell findings of a published file, by type (.csv, .json, .md, .html); [] for other types."""
    suf = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if suf == "csv":
        return scan_csv_text(text, cohort_n, min_n)
    if suf == "json":
        return scan_json_text(text, cohort_n, min_n)
    if suf == "md":
        return scan_md_text(text, cohort_n, min_n)
    if suf in ("html", "htm"):
        return scan_html_text(text, cohort_n, min_n)
    return []
