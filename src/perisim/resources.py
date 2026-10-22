"""Real compute resources of the container and the processes × threads plan (SPECS §9.1).

Inside a container, `os.cpu_count()` and most libraries see every core of the host node, not the cores
assigned to the pod. The real limits come from cgroups (CPU quota and memory limit) and from the CPU
affinity mask. The plan honours SPECS §9.1: at most 4 concurrent jobs with 4 threads each, reduced when the
container has fewer cores or less memory.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

MAX_JOBS = 4             # SPECS §9.1: at most 4 concurrent jobs …
MAX_THREADS_PER_JOB = 4  # … with 4 threads each
MIN_GB_PER_JOB = 12.0    # memory budget per modelling job (one INSPIRE fold with all comparators)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cgroup_cpus() -> float | None:
    """CPU quota of the cgroup (v2 `cpu.max`, v1 `cfs_quota_us / cfs_period_us`); None if unlimited."""
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2:
        quota, _, period = v2.partition(" ")
        if quota != "max" and period:
            return int(quota) / int(period)
        return None
    q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") or _read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us")
    p = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us") or _read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
    if q and p and int(q) > 0:
        return int(q) / int(p)
    return None


def cgroup_memory_gb() -> float | None:
    """Memory limit of the cgroup in GB; None if unlimited."""
    raw = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if not raw or raw == "max":
        return None
    val = int(raw)
    return None if val >= 1 << 60 else val / 1e9   # v1 reports a huge number when unlimited


def available_cpus() -> int:
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):          # Windows / macOS
        affinity = os.cpu_count() or 1
    quota = cgroup_cpus()
    cpus = min(affinity, math.floor(quota)) if quota else affinity
    return max(1, int(cpus))


def available_memory_gb() -> float:
    try:
        import psutil

        total = psutil.virtual_memory().total / 1e9
    except Exception:  # pragma: no cover
        total = float("nan")
    lim = cgroup_memory_gb()
    return min(total, lim) if lim else total


def plan(resources_cfg: dict | None = None) -> dict:
    """Processes × threads plan. `max_parallel_jobs` / `threads_per_job` may be integers or "auto"."""
    rc = resources_cfg or {}
    cpus = available_cpus()
    mem = available_memory_gb()
    t_cfg, j_cfg = rc.get("threads_per_job", "auto"), rc.get("max_parallel_jobs", "auto")
    threads = min(MAX_THREADS_PER_JOB, cpus) if str(t_cfg) == "auto" else int(t_cfg)
    if str(j_cfg) == "auto":
        jobs = max(1, min(MAX_JOBS, cpus // max(threads, 1)))
        if math.isfinite(mem):
            jobs = max(1, min(jobs, int(mem // MIN_GB_PER_JOB)))
    else:
        jobs = int(j_cfg)
    return {"cpus": cpus, "memory_gb": round(mem, 1) if math.isfinite(mem) else None,
            "cgroup_cpu_quota": cgroup_cpus(), "cgroup_memory_gb": cgroup_memory_gb(),
            "jobs": int(jobs), "threads_per_job": int(threads)}


def apply_library_limits(cpus: int) -> None:
    """Cap the thread pools that would otherwise size themselves from the host's core count."""
    try:
        import pyarrow as pa

        pa.set_cpu_count(max(1, cpus))
        pa.set_io_thread_count(max(4, min(16, cpus)))
    except Exception:  # pragma: no cover
        pass
