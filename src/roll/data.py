"""The data of the streams: `hypo`'s analysis landmarks, grouped into units (patients), with their calendar fields.

- A unit is a patient: every landmark of every stay of the patient, entered at once (design §5). In eICU a patient
  belongs to the hospital holding most of their landmarks and to the first year in which they have one; the few
  patients seen in two hospitals are counted in the run log.
- Calendar fields: MIMIC-IV's `anchor_year_group` (patients.csv.gz), eICU's `hospitaldischargeyear` (patient.csv.gz)
  and `hospital` (from `hypo`'s landmarks). INSPIRE has none (Phase 0).
- The admission gate (design §5): duplicates (stay, L), values outside `hypo`'s bounds, impossible values. A refused
  landmark is still predicted but enters no arm's data.
- Embeddings (`hypo`'s frozen metric, weighted, float32) and the frozen comparators' base predictions are computed
  once per cohort and metric.

Patient-level throughout: nothing here is exported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from hypo.data import Landmarks
from hypo.data import load as load_lm

from .config import data_root

GATE_COLS = {"map": ["map_last", "map_mean_6h", "map_min_6h"], "hr": ["hr_last"], "spo2": ["spo2_last", "spo2_min_6h"],
             "rr": ["rr_last"], "temp": ["temp_last"], "hb": ["hb_last"], "creatinine": ["creatinine_last"],
             "lactate": ["lactate_last"]}


def trailing_int(s: np.ndarray) -> np.ndarray:
    """The original number of a prefixed identifier ("EICU-123", "mimic:MIMIC-45" → 123, 45)."""
    return pl.Series(np.asarray(s, dtype=str)).str.extract(r"(\d+)$").cast(pl.Int64).to_numpy()


@dataclass
class Cohort:
    name: str
    lm: Landmarks
    unit_of_row: np.ndarray                 # unit index of each landmark
    subject: np.ndarray                     # unit → patient id ("<cohort>:<subject>")
    block: np.ndarray                       # unit → MIMIC-IV block ("" elsewhere)
    year: np.ndarray                        # unit → eICU year (−1 elsewhere)
    hospital: np.ndarray                    # unit → eICU hospital ("" elsewhere)
    gate_ok: np.ndarray                     # landmark → passes the admission gate
    row_year: np.ndarray | None = None      # landmark → eICU year (the gate's hospital eligibility, as Phase 0)
    info: dict = field(default_factory=dict)
    _rows: list | None = None

    @property
    def n_units(self) -> int:
        return len(self.subject)

    def rows(self, u: int) -> np.ndarray:
        if self._rows is None:
            order = np.argsort(self.unit_of_row, kind="stable")
            cuts = np.searchsorted(self.unit_of_row[order], np.arange(self.n_units + 1))
            self._rows = [order[cuts[i]:cuts[i + 1]] for i in range(self.n_units)]
        return self._rows[u]


def admission_gate(lm: Landmarks, bounds: dict) -> np.ndarray:
    """True for the landmarks the gate admits."""
    ok = np.ones(len(lm), dtype=bool)
    for k, cols in GATE_COLS.items():
        lo, hi = bounds[k]
        for c in cols:
            if c in lm.ft.num:
                x = lm.ft.num[c]
                ok &= ~(np.isfinite(x) & ((x < lo) | (x > hi)))
    ids = lm.ft.ids.astype(str)
    _, first = np.unique(ids, return_index=True)
    dup = np.ones(len(lm), dtype=bool)
    dup[first] = False
    ok &= ~dup
    L = lm.frame["L"].cast(pl.Float64).to_numpy() if "L" in lm.frame.columns else lm.ft.num["L"]
    mn, mean = lm.ft.num["map_min_6h"], lm.ft.num["map_mean_6h"]
    ok &= ~((L < 0) | (lm.ft.num["age"] < 18) | (np.isfinite(mn) & np.isfinite(mean) & (mn > mean + 1e-9)))
    return ok


def build(name: str, lm: Landmarks, row_block: np.ndarray | None, row_year: np.ndarray | None,
          bounds: dict) -> Cohort:
    """Units of a cohort from its landmarks and the per-landmark calendar fields (None where the cohort has none)."""
    subj = lm.ft.subjects.astype(str)
    uniq, inv = np.unique(subj, return_inverse=True)
    n = uniq.size
    block = np.full(n, "", dtype=object)
    year = np.full(n, -1, dtype=np.int64)
    hosp = np.full(n, "", dtype=object)
    info = {"units": int(n), "landmarks": int(len(lm))}
    if row_block is not None:
        rb = np.asarray(row_block, dtype=object)
        d = pl.DataFrame({"u": inv, "b": rb.astype(str)}).group_by("u").agg(pl.col("b").min())
        block[d["u"].to_numpy()] = d["b"].to_numpy()
        info["units_without_block"] = int((block == "None").sum())
    if row_year is not None:
        h = np.asarray(lm.hospital, dtype=object).astype(str)
        d = pl.DataFrame({"u": inv, "y": np.asarray(row_year, dtype=np.int64), "h": h})
        yr = d.group_by("u").agg(pl.col("y").min())
        year[yr["u"].to_numpy()] = yr["y"].to_numpy()
        hc = d.group_by(["u", "h"]).agg(pl.len().alias("n")).sort(["u", "n", "h"], descending=[False, True, False])
        main = hc.unique("u", keep="first", maintain_order=True)
        hosp[main["u"].to_numpy()] = main["h"].to_numpy()
        info["units_in_two_hospitals"] = int((hc.group_by("u").len()["len"] > 1).sum())
    ok = admission_gate(lm, bounds)
    info["gate_refused_landmarks"] = int((~ok).sum())
    ry = None if row_year is None else np.asarray(row_year, dtype=np.int64)
    return Cohort(name, lm, inv, uniq.astype(object), block, year, hosp, ok, ry, info)


def real_cohort(name: str, hcfg: dict) -> Cohort:
    """A cohort from `hypo`'s landmarks and the calendar fields of its original tables."""
    import duckdb

    lm = load_lm(name, hcfg)
    con = duckdb.connect()
    rb = ry = None
    if name == "mimic":
        src = (data_root() / "mimic4" / "hosp" / "patients.csv.gz").as_posix()
        p = con.execute(f"SELECT subject_id, anchor_year_group FROM read_csv_auto('{src}')").pl()
        m = dict(zip(p["subject_id"].to_list(), p["anchor_year_group"].to_list()))
        rb = np.array([m.get(int(s)) for s in trailing_int(lm.ft.subjects)], dtype=object)
    if name == "eicu":
        src = (data_root() / "eicu" / "patient.csv.gz").as_posix()
        p = con.execute(f"SELECT patientunitstayid, hospitaldischargeyear FROM read_csv_auto('{src}')").pl()
        m = dict(zip(p["patientunitstayid"].to_list(), p["hospitaldischargeyear"].to_list()))
        stay = trailing_int(np.array([i.split(":")[0] for i in lm.ft.ids.astype(str)]))
        ry = np.array([m.get(int(s), -1) for s in stay], dtype=np.int64)
    return build(name, lm, rb, ry, hcfg["bounds"])


# --------------------------------------------------------------------------- frozen artefacts
def unwrap(comp):
    """A frozen `hypo` comparator (lambda q: pred(enc.transform_model(q.ft)[0])[:, 1]) → (enc, pred)."""
    names = comp.__code__.co_freevars
    if set(names) != {"enc", "pred"}:
        raise ValueError(f"not a hypo comparator closure: {names}")
    cells = dict(zip(names, [c.cell_contents for c in comp.__closure__]))
    return cells["enc"], cells["pred"]


class Frozen:
    """`hypo`'s frozen metric and comparators of each rotation (named by the cohort it excluded), and what the
    streams need from them, computed once per cohort: weighted embeddings, base predictions, model matrices."""

    def __init__(self, metrics: dict, comps: dict, infos: dict):
        self.metrics, self.comps, self.infos = metrics, comps, infos
        self._E, self._p, self._X = {}, {}, {}

    @classmethod
    def load(cls, families: list[str]) -> "Frozen":
        import json

        from hypo.frozen import final_dir, load_pickle

        d = final_dir()
        metrics, comps, infos = {}, {}, {}
        for x in ("inspire", "mimic", "eicu"):
            metrics[x] = load_pickle(d / f"metric_ext-{x}.pkl")
            for f in families:
                comps[(x, f)] = load_pickle(d / f"comp_ext-{x}_{f}.pkl")
                infos[(x, f)] = json.loads((d / f"comp_ext-{x}_{f}.json").read_text(encoding="utf-8"))
        return cls(metrics, comps, infos)

    def Ew(self, metric: str, c: Cohort) -> np.ndarray:
        key = (metric, c.name)
        if key not in self._E:
            m = self.metrics[metric]
            self._E[key] = np.ascontiguousarray(m.embed(c.lm) * m.sqrt_w_dims, dtype=np.float32)
        return self._E[key]

    def p0(self, metric: str, family: str, c: Cohort) -> np.ndarray:
        key = (metric, family, c.name)
        if key not in self._p:
            self._p[key] = np.asarray(self.comps[(metric, family)](c.lm), dtype=np.float64)
        return self._p[key]

    def X(self, metric: str, family: str, c: Cohort) -> np.ndarray:
        """Model matrix of the transported comparator's own encoder (the inputs a refit uses)."""
        key = (metric, family, c.name)
        if key not in self._X:
            enc, _ = unwrap(self.comps[(metric, family)])
            self._X[key] = enc.transform_model(c.lm.ft)[0]
        return self._X[key]
