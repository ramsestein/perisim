"""Cross-cutting utilities: logging, timer with memory usage, disk, atomic I/O and automatic notes."""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

LOG = logging.getLogger("perisim")


def setup_logging(logfile: Path | None = None, level: int = logging.INFO) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger("perisim")
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_perisim", False) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh._perisim = True  # type: ignore[attr-defined]
        root.addHandler(sh)
    if logfile is not None:
        try:
            fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError as exc:  # object storage without append: console only
            root.warning("Cannot write the log to %s (%s); console only", logfile, exc)
    root.propagate = False


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except Exception:  # pragma: no cover
        return float("nan")


def free_disk_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("nan")


def dir_size_gb(path: Path) -> float:
    total = 0
    for p in Path(path).rglob("*"):
        with contextlib.suppress(OSError):
            if p.is_file():
                total += p.stat().st_size
    return total / 1e9


@contextlib.contextmanager
def timed(label: str, log: logging.Logger = LOG, store: dict | None = None) -> Iterator[dict]:
    rec: dict[str, Any] = {"label": label}
    t0 = time.perf_counter()
    log.info("▶ %s", label)
    try:
        yield rec
    finally:
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        rec["rss_gb"] = round(rss_gb(), 2)
        log.info("■ %s (%.1f s, RSS %.2f GB)", label, rec["seconds"], rec["rss_gb"])
        if store is not None:
            store[label] = rec


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _clean(o: Any) -> Any:
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(json.loads(json.dumps(obj, cls=NumpyEncoder))), indent=2, ensure_ascii=False)
    atomic_write_text(path, text)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:   # e.g. truncated by an OOM kill
        LOG.warning("Unreadable JSON %s (%s): treated as missing", path, exc)
        return default


def write_jsonl(path: Path, records: list) -> Path:
    """One compact JSON object per line (per-patient outputs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "".join(json.dumps(_clean(json.loads(json.dumps(r, cls=NumpyEncoder))),
                                               ensure_ascii=False) + "\n" for r in records))
    return path


def read_jsonl(path: Path) -> list:
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        return []


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except OSError:  # some object-storage mounts do not support atomic rename
        shutil.copyfile(tmp, path)
        tmp.unlink(missing_ok=True)


def write_parquet(df, path: Path) -> Path:
    """Writes a polars DataFrame to zstd Parquet atomically (§9.1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    try:
        os.replace(tmp, path)
    except OSError:
        shutil.copyfile(tmp, path)
        tmp.unlink(missing_ok=True)
    return path


def append_note(path: Path, title: str, body: str) -> None:
    """Appends a dated entry to the workdir's automatic OPEN_QUESTIONS / DECISIONS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "" if path.exists() else f"# {path.stem.replace('_', ' ')} (generated in the pod)\n\n"
    entry = f"{head}## {now_iso()} — {title}\n\n{body.strip()}\n\n"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    atomic_write_text(path, old + entry)


def set_threads(n: int) -> None:
    """Sets BLAS/OpenMP/torch threads for a job (§9.1: 4 jobs × 4 threads)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:  # pragma: no cover
        pass
    try:
        import faiss

        faiss.omp_set_num_threads(n)
    except Exception:  # pragma: no cover
        pass
