"""Small-cell suppression for everything that may leave the pod (SPECS §9; KHDP export policy; AUDIT A6).

Rules applied consistently:
- a count 1–9 is replaced by "<10" (primary suppression);
- its complement is suppressed too when the total is known (n − count < 10 reveals a small group);
- a set of counts with a known total and exactly one suppressed cell gets the next-smallest cell suppressed as
  well (secondary suppression), so that no cell can be recovered by subtraction;
- prevalences / percentages are shown only when both the numerator and its complement are ≥ 10;
- flow diagrams: exclusions < 10 are suppressed and the remaining counts are rounded to the nearest 10 from that
  step on, so exclusions cannot be recovered by subtracting consecutive steps.
"""

from __future__ import annotations

SUPPRESSED = "<10"


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
    """Primary + secondary suppression of a set of counts that add up to a published total."""
    out = {k: (v if v is None or not 0 < int(v) < min_n else f"<{min_n}") for k, v in cells.items()}
    small = [k for k, v in out.items() if isinstance(v, str)]
    if len(small) == 1:
        rest = sorted(((int(v), k) for k, v in out.items() if not isinstance(v, str) and v), key=lambda t: t[0])
        if rest:
            out[rest[0][1]] = f"<{max(min_n, rest[0][0] + 1)}" if rest[0][0] < 2 * min_n else "suppressed"
    return out


def flow(rows: list[dict], min_n: int = 10) -> list[dict]:
    """Exportable flow diagram (see the module docstring)."""
    out, rounding = [], False
    for r in rows:
        ex = r.get("excluded")
        if ex is not None and 0 < ex < min_n:
            rounding = True
        rnd = (lambda v: None if v is None else ("<10" if 0 < v < min_n else int(round(v, -1)))) if rounding else \
            (lambda v: cell(v, min_n))
        out.append({"step": r["step"], "n_stays": rnd(r.get("n_stays")), "n_subjects": rnd(r.get("n_subjects")),
                    "excluded": (cell(ex, min_n) if not rounding or ex is None or ex >= min_n else f"<{min_n}"),
                    "rounded": rounding})
    return out
