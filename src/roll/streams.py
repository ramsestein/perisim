"""The streams (design §4–§5): the starting window, the evaluated patients in order, the segments, the evaluation
windows and the steps.

- Order: segments in calendar order (MIMIC-IV blocks, eICU years) or in the pair's order (cross-cohort); random
  inside a segment, drawn anew for each of the R orders. INSPIRE is one random segment.
- Evaluation windows (§8.1): W consecutive evaluated patients inside a segment; a remainder below W/2 joins the
  segment's last window, otherwise it is a window of its own. No window straddles a boundary.
- Steps (§5): B patients, never crossing a window's end (a window's last step may be shorter).
- The delay L in patients (§7.6): 3 months at the admission rate of the segment.

Each stream has a hash of its order, so every arm reads the same sequence.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field

import numpy as np

from hypo.data import concat

from .data import Cohort

POP = {"mimic": "mimic", "inspire": "inspire", "eicu": "eicu"}


def segments_of(seg: np.ndarray) -> list[tuple[str, int, int]]:
    out, s = [], 0
    for i in range(1, len(seg) + 1):
        if i == len(seg) or seg[i] != seg[s]:
            out.append((str(seg[s]), s, i))
            s = i
    return out


def windows_for(segments: list[tuple[str, int, int]], W: dict) -> list[dict]:
    """Evaluation windows of W[segment] patients inside each segment; a remainder below W/2 joins the last."""
    out = []
    for lab, a, b in segments:
        w = int(W[lab])
        K, r = divmod(b - a, w)
        ends = [a + w * (i + 1) for i in range(K)]
        if K == 0:
            ends = [b]
        elif r >= w / 2:
            ends.append(b)
        else:
            ends[-1] = b
        s = a
        for e in ends:
            out.append({"start": s, "end": e, "seg": lab, "index": len(out)})
            s = e
    return out


def steps_for(windows: list[dict], B: int) -> list[tuple[int, int, int]]:
    """(start, end, window) of each step, in evaluated positions: B patients, never crossing a window's end."""
    out = []
    for w in windows:
        for s in range(w["start"], w["end"], B):
            out.append((s, min(s + B, w["end"]), w["index"]))
    return out


@dataclass
class Pool:
    """The landmarks of a stream's cohorts, as one table (global rows and units)."""
    cohorts: list[Cohort]
    row_off: dict
    unit_off: dict

    def __post_init__(self):
        self.lm = self.cohorts[0].lm if len(self.cohorts) == 1 else concat([c.lm for c in self.cohorts])
        self.y = self.lm.y.astype(np.int64)
        subj = np.concatenate([c.subject[c.unit_of_row] for c in self.cohorts]).astype(str)
        _, self.pcode = np.unique(subj, return_inverse=True)            # the patient, for the exclusion
        self.gate_ok = np.concatenate([c.gate_ok for c in self.cohorts])
        self.unit_subject = np.concatenate([c.subject for c in self.cohorts]).astype(str)
        self.unit_cohort = np.concatenate([np.full(c.n_units, c.name, dtype=object) for c in self.cohorts])

    def rows(self, units: np.ndarray) -> np.ndarray:
        """Global rows of global units, in order."""
        out = []
        for u in np.asarray(units, dtype=np.int64):
            for c in self.cohorts:
                o = self.unit_off[c.name]
                if o <= u < o + c.n_units:
                    out.append(c.rows(int(u - o)) + self.row_off[c.name])
                    break
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def make_pool(cohorts: list[Cohort]) -> Pool:
    ro, uo, r, u = {}, {}, 0, 0
    for c in cohorts:
        ro[c.name], uo[c.name] = r, u
        r += len(c.lm)
        u += c.n_units
    return Pool(cohorts, ro, uo)


@dataclass
class Stream:
    name: str
    kind: str
    order: int
    N: int
    metric: str                       # the cohort excluded by the frozen metric's rotation
    pool: Pool
    start: np.ndarray                 # global units of the starting window, in entry order
    seq: np.ndarray                   # evaluated global units, in order
    seg: np.ndarray                   # segment label of each evaluated position
    pop: dict                         # segment → population ("mimic", "inspire", "eicu", "eicu_hospital")
    rate: dict                        # segment → patients per month
    hospital: str = ""
    internal: dict = field(default_factory=dict)   # segment → True where the metric saw the cohort
    windows: list = field(default_factory=list)

    @property
    def segments(self) -> list[tuple[str, int, int]]:
        return segments_of(self.seg)

    def make_windows(self, W: dict) -> None:
        self.windows = windows_for(self.segments, {lab: int(W[p]) for lab, p in self.pop.items()})

    def steps(self, B: int) -> list[tuple[int, int, int]]:
        return steps_for(self.windows, B)

    def delay(self, seg: str, months: float) -> int:
        return int(round(months * self.rate[seg]))

    def digest(self) -> str:
        h = hashlib.sha256()
        for part in (self.pool.unit_subject[self.start], self.pool.unit_subject[self.seq], self.seg.astype(str)):
            h.update("|".join(part).encode())
        return h.hexdigest()[:16]

    def describe(self) -> dict:
        return {"stream": self.name, "kind": self.kind, "order": self.order, "N": self.N, "metric": f"ext-{self.metric}",
                "hospital": self.hospital, "evaluated": int(len(self.seq)), "windows": len(self.windows),
                "segments": [{"segment": s, "patients": b - a, "internal": bool(self.internal.get(s, False)),
                              "patients_per_month": round(self.rate[s], 2)} for s, a, b in self.segments],
                "digest": self.digest()}


# --------------------------------------------------------------------------- hospitals
def hospital_labels(c: Cohort, min_patients: int = 300) -> dict[str, str]:
    """eICU hospital → H1, H2, …: `hypo`'s labels (hospitals with ≥ 300 patients, by size), then the rest by size."""
    from hypo.external import hospital_labels as hypo_labels

    import polars as pl

    lab = dict(hypo_labels(c.lm, min_patients))
    d = pl.DataFrame({"h": np.asarray(c.lm.hospital, dtype=object).astype(str), "s": c.lm.ft.subjects.astype(str)})
    n = d.unique().group_by("h").len().sort(["len", "h"], descending=[True, False])
    k = len(lab)
    for hh in n["h"].to_list():
        if hh not in lab:
            k += 1
            lab[hh] = f"H{k}"
    return lab


def eligible_hospitals(c: Cohort, years: list[int], min_each_year: int) -> list[str]:
    """Hospitals with at least `min_each_year` patients with landmarks in each year (Phase 0's count), by label."""
    h = np.asarray(c.lm.hospital, dtype=object).astype(str)
    s = c.lm.ft.subjects.astype(str)
    out = []
    for hh in np.unique(h):
        m = h == hh
        if all(np.unique(s[m & (c.row_year == y)]).size >= min_each_year for y in years):
            out.append(hh)
    lab = hospital_labels(c)
    return sorted(out, key=lambda x: int(lab[x][1:]))


# --------------------------------------------------------------------------- building
def _rng(seed: int, name: str, hospital: str, order: int, N: int) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(f"{name}|{hospital}".encode()), order, N])


def _perm(rng: np.random.Generator, units: np.ndarray) -> np.ndarray:
    return units[rng.permutation(units.size)]


def build(name: str, spec: dict, cohorts: dict[str, Cohort], cfg: dict, order: int, N: int,
          hospital: str = "") -> Stream | None:
    """One stream, one order, one N. None when the stream cannot start (too few patients for N)."""
    kind, mo = spec["kind"], cfg["months"]
    rng = _rng(int(cfg["seed"]), name, hospital, order, N)
    if kind in ("temporal", "random"):
        c = cohorts[spec["cohort"]]
        pool = make_pool([c])
        u = np.arange(c.n_units)
        if kind == "temporal":
            labs = sorted(set(c.block[u]))
            parts = [_perm(rng, u[c.block[u] == b]) for b in labs]
            rate = {b: (c.block == b).sum() / mo["mimic_block"] for b in labs}
        else:
            labs, parts = [c.name], [_perm(rng, u)]
            rate = {c.name: c.n_units / mo[f"{c.name}_cohort"]}
        if N >= parts[0].size:
            return None
        start, parts[0] = parts[0][:N], parts[0][N:]
        seg = np.concatenate([np.full(p.size, lab, dtype=object) for lab, p in zip(labs, parts)])
        return Stream(name, kind, order, N, spec["metric"], pool, start, np.concatenate(parts), seg,
                      {lab: POP[c.name] for lab in labs}, rate)
    if kind in ("within", "deploy"):
        c = cohorts[spec["cohort"]]
        pool = make_pool([c])
        years = list(cfg["eicu_hospitals"]["years"])
        u = np.arange(c.n_units)
        mine = u[c.hospital == hospital]
        labs = [str(y) for y in years]
        parts = [_perm(rng, mine[c.year[mine] == y]) for y in years]
        rate = {str(y): (c.year[mine] == y).sum() / mo["eicu_year"] for y in years}
        if kind == "within":
            if N >= parts[0].size:
                return None
            start, parts[0] = parts[0][:N], parts[0][N:]
        else:
            others = u[c.hospital != hospital]
            if N > others.size:
                return None
            start = _perm(rng, others)[:N]
        seg = np.concatenate([np.full(p.size, lab, dtype=object) for lab, p in zip(labs, parts)])
        return Stream(name, kind, order, N, spec["metric"], pool, start, np.concatenate(parts), seg,
                      {lab: "eicu_hospital" for lab in labs}, rate, hospital=hospital)
    if kind == "cross":
        a, b = cohorts[spec["first"]], cohorts[spec["second"]]
        pool = make_pool([a, b])
        ua = _perm(rng, np.arange(a.n_units)) + pool.unit_off[a.name]
        ub = _perm(rng, np.arange(b.n_units)) + pool.unit_off[b.name]
        na, nb = int(spec["first_n"]), int(spec["second_n"])
        if N + na > ua.size or nb > ub.size:
            return None
        start, sa, sb = ua[:N], ua[N:N + na], ub[:nb]
        seg = np.concatenate([np.full(na, a.name, dtype=object), np.full(nb, b.name, dtype=object)])
        rate = {a.name: a.n_units / mo[f"{a.name}_cohort"], b.name: b.n_units / mo[f"{b.name}_cohort"]}
        return Stream(name, kind, order, N, spec["metric"], pool, start, np.concatenate([sa, sb]), seg,
                      {a.name: POP[a.name], b.name: POP[b.name]}, rate,
                      internal={a.name: a.name != spec["metric"], b.name: b.name != spec["metric"]})
    raise ValueError(kind)
