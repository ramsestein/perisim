"""In-memory (numpy) input table for modelling: numeric columns with NaN and categorical columns as text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from perisim.features.build import FeatureSpec
from perisim.utils import read_json


@dataclass
class FeatureTable:
    ids: np.ndarray
    subjects: np.ndarray
    specs: list[FeatureSpec]
    num: dict[str, np.ndarray]
    cat: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, idx: np.ndarray) -> "FeatureTable":
        idx = np.asarray(idx)
        return FeatureTable(self.ids[idx], self.subjects[idx], self.specs,
                            {k: v[idx] for k, v in self.num.items()}, {k: v[idx] for k, v in self.cat.items()})

    def select(self, names: list[str]) -> "FeatureTable":
        keep = set(names)
        specs = [s for s in self.specs if s.name in keep]
        return FeatureTable(self.ids, self.subjects, specs,
                            {k: v for k, v in self.num.items() if k in keep},
                            {k: v for k, v in self.cat.items() if k in keep})

    def spec(self, name: str) -> FeatureSpec:
        return next(s for s in self.specs if s.name == name)

    def column(self, name: str) -> np.ndarray:
        return self.num[name] if name in self.num else self.cat[name]

    @classmethod
    def from_frame(cls, X: pl.DataFrame, specs: list[FeatureSpec], subjects: pl.DataFrame) -> "FeatureTable":
        X = X.join(subjects.select("stay_id", "subject_id"), on="stay_id", how="left").sort("stay_id")
        num, cat = {}, {}
        for s in specs:
            if s.name not in X.columns:
                if s.type in ("nominal", "hier"):
                    cat[s.name] = np.full(X.height, None, dtype=object)
                else:
                    num[s.name] = np.full(X.height, np.nan)
                continue
            if s.type in ("nominal", "hier"):
                cat[s.name] = X[s.name].cast(pl.Utf8).to_numpy().astype(object)
            else:
                num[s.name] = X[s.name].cast(pl.Float64).to_numpy().astype(np.float64)
        return cls(X["stay_id"].to_numpy().astype(object), X["subject_id"].to_numpy().astype(object), specs, num, cat)

    @classmethod
    def load(cls, feat_dir: Path, stays: pl.DataFrame, names: list[str] | None = None) -> "FeatureTable":
        X = pl.read_parquet(feat_dir / "features.parquet")
        specs = [FeatureSpec(**d) for d in read_json(feat_dir / "feature_specs.json")]
        if names is not None:
            keep = set(names)
            specs = [s for s in specs if s.name in keep]
            for s in specs:
                if s.name not in X.columns:
                    X = X.with_columns(pl.lit(None, pl.Utf8 if s.type in ("nominal", "hier") else pl.Float64).alias(s.name))
            X = X.select(["stay_id"] + [s.name for s in specs])
        return cls.from_frame(X, specs, stays)
