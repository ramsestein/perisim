"""Landmark tables → the input table of perisim's encoder, and the learning sample of the metric.

One row per landmark of the analysis population (normotensive in the last hour, window observable). Landmark ids
are "<stay>:<L>"; patients are "<cohort>:<subject>", so patients of different cohorts never collide. The encoder
sees the inputs of `model.inputs` only; outcomes and strata stay beside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from perisim.features.build import FeatureSpec
from perisim.features.table import FeatureTable

from .config import workdir

KINDS = ("continuous", "binary", "nominal")


@dataclass
class Landmarks:
    ft: FeatureTable          # encoder inputs
    y: np.ndarray             # primary outcome (0 / 1)
    band: np.ndarray          # lowest hourly MAP of the next 6 h: 0 < 65, 1 65-75, 2 ≥ 75 (−1 unknown)
    site: np.ndarray          # cohort of each landmark
    hospital: np.ndarray      # eICU hospital (None elsewhere)
    frame: pl.DataFrame       # outcomes and strata carried by the neighbours (no inputs)

    def __len__(self) -> int:
        return len(self.y)

    def take(self, idx: np.ndarray) -> "Landmarks":
        idx = np.asarray(idx)
        return Landmarks(self.ft.take(idx), self.y[idx], self.band[idx], self.site[idx], self.hospital[idx],
                         self.frame[idx])


CARRIED = ["L", "art_line", "y_first", "map_min_next", "vaso_start", "icu_discharge_24h", "died_in_window"]


def specs(cfg: dict) -> list[FeatureSpec]:
    """Encoder specs of `model.inputs`; the feature-dropout group is the variable's family (map, hr, vaso…)."""
    inp = cfg["model"]["inputs"]
    return [FeatureSpec(name=v, type=k, group=v.split("_")[0], block="inputs") for k in KINDS for v in inp[k]]


def load(cohort: str, cfg: dict, frame: pl.DataFrame | None = None) -> Landmarks:
    """Analysis landmarks of a cohort (from <workdir>/landmarks/<cohort>.parquet unless `frame` is given)."""
    df = frame if frame is not None else pl.read_parquet(workdir() / "landmarks" / f"{cohort}.parquet")
    df = df.filter(pl.col("normotensive") & pl.col("known")).sort("stay_id", "L")
    sp = specs(cfg)
    num, cat = {}, {}
    for s in sp:
        if s.type == "nominal":
            cat[s.name] = df[s.name].cast(pl.Utf8).to_numpy().astype(object)
        else:
            num[s.name] = df[s.name].cast(pl.Float64).to_numpy().astype(np.float64)
    keys = df.select(pl.concat_str([pl.col("stay_id").cast(pl.Utf8), pl.lit(":"), pl.col("L").cast(pl.Utf8)])
                     .alias("id"), pl.concat_str([pl.lit(cohort + ":"), pl.col("subject_id").cast(pl.Utf8)]).alias("s"))
    ids, subjects = keys["id"].to_numpy().astype(object), keys["s"].to_numpy().astype(object)
    hosp = df["hospital"].cast(pl.Utf8).to_numpy().astype(object) if "hospital" in df.columns \
        else np.full(df.height, None, dtype=object)
    return Landmarks(FeatureTable(ids, subjects, sp, num, cat), df["y"].cast(pl.Int64).to_numpy(),
                     df["band"].fill_null(-1).cast(pl.Int64).to_numpy(), np.full(df.height, cohort, dtype=object),
                     hosp, df.select([c for c in CARRIED if c in df.columns]))


def concat(parts: list[Landmarks]) -> Landmarks:
    ft = FeatureTable(np.concatenate([p.ft.ids for p in parts]), np.concatenate([p.ft.subjects for p in parts]),
                      parts[0].ft.specs, {k: np.concatenate([p.ft.num[k] for p in parts]) for k in parts[0].ft.num},
                      {k: np.concatenate([p.ft.cat[k] for p in parts]) for k in parts[0].ft.cat})
    return Landmarks(ft, np.concatenate([p.y for p in parts]), np.concatenate([p.band for p in parts]),
                     np.concatenate([p.site for p in parts]), np.concatenate([p.hospital for p in parts]),
                     pl.concat([p.frame for p in parts], how="diagonal_relaxed"))


def learning_sample(lm: Landmarks, per_patient: int, per_site: int, seed: int) -> np.ndarray:
    """Rows of the metric's learning sample: at most `per_patient` random landmarks per patient, then the same
    number of landmarks from every site (min(per_site, what the smallest site offers)), drawn at random."""
    rng = np.random.default_rng(seed)
    key = rng.random(len(lm))
    _, inv = np.unique(lm.ft.subjects.astype(str), return_inverse=True)
    order = np.lexsort((key, inv))
    rank = np.empty(len(lm), dtype=np.int64)
    first = np.r_[True, inv[order][1:] != inv[order][:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(lm)), 0))
    rank[order] = np.arange(len(lm)) - start
    pool = {s: np.where((lm.site == s) & (rank < per_patient))[0] for s in np.unique(lm.site)}
    n = min([per_site] + [p.size for p in pool.values()])
    return np.sort(np.concatenate([rng.choice(p, size=n, replace=False) for p in pool.values()]))
