"""Base class for the per-cohort extractors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Collection

import polars as pl

from perisim.config import Config
from perisim.data.io import find_table, load_table, read_header, resolve_columns


class BaseExtractor:
    cohort: str = ""
    prefix: str = ""

    def __init__(self, cfg: Config, src_dir: Path, cache_dir: Path):
        self.cfg = cfg
        self.h = cfg.harmonization
        self.src = self.h["sources"][self.cohort]
        self.ccfg = {**cfg.cohorts.get("common", {}), **cfg.cohorts.get(self.cohort, {})}
        self.src_dir = Path(src_dir)
        self.cache_dir = Path(cache_dir)
        self.log = logging.getLogger(f"perisim.data.{self.cohort}")
        self.followup_min = float(cfg.targets["window"]["followup_days"]) * 1440.0

    # ------------------------------------------------------------------ reading
    def has_table(self, name: str) -> bool:
        t = self.src["tables"].get(name)
        return bool(t) and find_table(self.src_dir, t["file"]) is not None

    def table(self, name: str, columns: list[str] | None = None,
              filters: dict[str, Collection] | None = None, optional: bool = False,
              extra: dict[str, str] | None = None) -> pl.DataFrame | None:
        tcfg = self.src["tables"][name]
        path = find_table(self.src_dir, tcfg["file"])
        if path is None:
            if optional:
                self.log.warning("Optional table not found: %s", tcfg["file"])
                return None
            raise FileNotFoundError(f"{self.cohort}: table {tcfg['file']} not found in {self.src_dir}")
        cols = {k: v for k, v in tcfg["columns"].items() if columns is None or k in columns}
        if extra:   # optional columns: used only if the real table has them
            present, absent = resolve_columns(read_header(path), extra)
            if absent:
                self.log.warning("%s: optional columns not present, ignored: %s", tcfg["file"], absent)
            cols.update({k: extra[k] for k in present})
        self.log.info("Reading %s (%s)", tcfg["file"], ", ".join(cols))
        return load_table(path, cols, filters, self.cache_dir)

    def optional_column_names(self, name: str) -> dict[str, str]:
        return {k: v for k, v in (self.src["tables"][name].get("optional_columns") or {}).items() if v}

    def sid(self, col: str) -> pl.Expr:
        """Identifier prefixed with the cohort (stays, subjects and hospitalizations)."""
        return pl.lit(self.prefix + "-") + pl.col(col).cast(pl.Utf8).str.strip_chars().str.replace(r"\.0+$", "")

    def raw(self, col: str) -> pl.Expr:
        """Inverse of `sid`: strips the cohort prefix."""
        return pl.col(col).str.slice(len(self.prefix) + 1)

    def raw_ids(self, stays: pl.DataFrame, col: str) -> list[str]:
        return stays.select(self.raw(col)).to_series().unique().drop_nulls().to_list()

    # ------------------------------------------------------------------ interface
    def extract_stays(self) -> pl.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    def extract_events(self, index: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:  # pragma: no cover
        raise NotImplementedError
