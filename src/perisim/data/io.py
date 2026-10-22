"""Lazy reading of original tables (§9.1, Memory / Storage).

- Parquet: `polars.scan_parquet` with column selection and filtering before loading.
- CSV / CSV.GZ: block-wise streaming read with pyarrow, filtering rows and columns per block,
  and writing the filtered extract to the cache as zstd Parquet. A full table is never loaded.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Collection

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

LOG = logging.getLogger("perisim.data.io")
EXTS = (".parquet", ".csv.gz", ".csv")


def _ci_child(parent: Path, name: str) -> Path | None:
    exact = parent / name
    if exact.exists():
        return exact
    try:
        for e in parent.iterdir():
            if e.name.lower() == name.lower():
                return e
    except OSError:
        return None
    return None


def find_table(base: Path | None, rel: str) -> Path | None:
    """Locates `rel` (e.g. 'hosp/admissions', 'vitalPeriodic') as .parquet, .csv.gz or .csv,
    case-insensitively; also accepts a directory of Parquet files."""
    if base is None:
        return None
    parts = rel.split("/")
    cur: Path | None = Path(base)
    for p in parts[:-1]:
        cur = _ci_child(cur, p)
        if cur is None:
            return None
    stem = parts[-1].lower()
    try:
        entries = list(cur.iterdir())
    except OSError:
        return None
    for ext in EXTS:
        for e in entries:
            if e.is_file() and e.name.lower() == stem + ext:
                return e
    for e in entries:
        if e.is_dir() and e.name.lower() == stem and any(x.suffix == ".parquet" for x in e.iterdir()):
            return e
    return None


def read_header(path: Path) -> list[str]:
    if path.is_dir():
        first = sorted(path.glob("*.parquet"))[0]
        return pq.read_schema(first).names
    if path.suffix == ".parquet":
        return pq.read_schema(path).names
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as fh:   # tolerate a BOM
        return next(csv.reader(fh))


_TRAILING_ZEROS = r"\.0+$"


def norm_id(v) -> str:
    """Identifier as comparable text: "123", 123, 123.0 and "123.0" all become "123" (AUDIT A3)."""
    return re.sub(_TRAILING_ZEROS, "", str(v).strip())


def resolve_columns(header: list[str], wanted: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """{canonical: real} -> ({canonical: resolved_real}, [missing]), case-insensitively."""
    lower = {h.lower(): h for h in header}
    found, missing = {}, []
    for canon, real in wanted.items():
        if real is None:
            continue
        hit = lower.get(str(real).lower())
        if hit is None:
            missing.append(str(real))
        else:
            found[canon] = hit
    return found, missing


def _cache_key(path: Path, cols: dict[str, str], filters: dict[str, Collection] | None) -> str:
    st = path.stat()
    filt = {k: sorted(map(str, v)) for k, v in (filters or {}).items()}
    blob = json.dumps([str(path), st.st_size, int(st.st_mtime), sorted(cols.items()), filt], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _stream_csv(path: Path, cols: dict[str, str], filters: dict[str, Collection] | None, dst: Path,
                block_mb: int = 64) -> None:
    real_cols = list(dict.fromkeys(cols.values()))
    conv = pacsv.ConvertOptions(include_columns=real_cols, column_types={c: pa.string() for c in real_cols},
                                strings_can_be_null=True)
    read_opts = pacsv.ReadOptions(block_size=block_mb << 20)
    parse = pacsv.ParseOptions(newlines_in_values=True)
    value_sets = {cols[k]: pa.array(sorted({norm_id(x) for x in v}), pa.string()) for k, v in (filters or {}).items()}
    tmp = dst.with_name(dst.name + ".tmp")
    writer = None
    n_in = n_out = 0
    stream = pa.input_stream(str(path), compression="detect")
    try:
        with pacsv.open_csv(stream, read_options=read_opts, parse_options=parse, convert_options=conv) as reader:
            schema = reader.schema
            for batch in reader:
                n_in += batch.num_rows
                tbl = pa.Table.from_batches([batch])
                if value_sets:
                    mask = None
                    for c, vs in value_sets.items():
                        col = pc.replace_substring_regex(pc.utf8_trim_whitespace(tbl[c]), pattern=_TRAILING_ZEROS,
                                                         replacement="")
                        m = pc.is_in(col, value_set=vs)
                        mask = m if mask is None else pc.and_(mask, m)
                    tbl = tbl.filter(mask)
                if tbl.num_rows:
                    if writer is None:
                        writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
                    writer.write_table(tbl)
                    n_out += tbl.num_rows
        if writer is None:
            pq.write_table(pa.Table.from_batches([], schema=schema), tmp, compression="zstd")
    finally:
        if writer is not None:
            writer.close()
    try:
        os.replace(tmp, dst)
    except OSError:
        shutil.copyfile(tmp, dst)
        tmp.unlink(missing_ok=True)
    LOG.info("  %s: %s rows read → %s filtered", path.name, f"{n_in:,}", f"{n_out:,}")


def load_table(path: Path, cols: dict[str, str], filters: dict[str, Collection] | None = None,
               cache_dir: Path | None = None) -> pl.DataFrame:
    """Reads `cols` ({canonical: real}) from `path`, filtering by text equality on `filters`
    ({canonical: values}). Returns a DataFrame with canonical names."""
    header = read_header(path)
    resolved, missing = resolve_columns(header, cols)
    if missing:
        raise KeyError(f"{path}: missing columns {missing}. Fix config/harmonization.yaml")
    filters = {k: v for k, v in (filters or {}).items() if k in resolved}
    if path.is_dir() or path.suffix == ".parquet":
        src = str(path / "*.parquet") if path.is_dir() else str(path)
        lf = pl.scan_parquet(src).select([pl.col(r).alias(c) for c, r in resolved.items()])
        for c, vals in filters.items():
            lf = lf.filter(pl.col(c).cast(pl.Utf8).str.strip_chars().str.replace(_TRAILING_ZEROS, "")
                           .is_in([norm_id(v) for v in vals]))
        return _warn_if_empty(lf.collect(), path, filters)
    if cache_dir is None:
        raise ValueError("cache_dir required for CSV")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{path.name.split('.')[0]}_{_cache_key(path, resolved, filters)}.parquet"
    if not dst.exists():
        _stream_csv(path, resolved, filters, dst)
    df = pl.read_parquet(dst)
    df = df.rename({r: c for c, r in resolved.items() if r in df.columns and r != c}) \
        .select([c for c in resolved]) if resolved else df
    return _warn_if_empty(df, path, filters)


def _warn_if_empty(df: pl.DataFrame, path: Path, filters: dict | None) -> pl.DataFrame:
    """A filtered read that returns nothing usually means an identifier/format mismatch, not absent data."""
    if filters and df.height == 0:
        LOG.warning("  %s: the filter on %s returned 0 rows — check identifier formats and item names "
                    "(config/harmonization.yaml)", path.name, ", ".join(filters))
    return df


def distinct_counts(path: Path, column: str, cache_dir: Path, top: int = 500) -> pl.DataFrame:
    """Distinct value counts of a column (for the `discover` item inventory)."""
    df = load_table(path, {"v": column}, None, cache_dir)
    return (df.group_by("v").len().sort("len", descending=True).head(top)
            .rename({"v": "value", "len": "n"}))
