"""§9.1 — Processes × threads plan from the container's real limits (never more than 4 × 4)."""

from __future__ import annotations

import pytest

from perisim import resources


@pytest.mark.parametrize("cpus,mem,jobs,threads", [
    (16, 128.0, 4, 4),   # the KHDP pod of SPECS §9.1
    (64, 512.0, 4, 4),   # big host: still capped at 4 × 4
    (8, 128.0, 2, 4),    # fewer cores than specified
    (16, 24.0, 2, 4),    # memory-bound: 24 GB / 12 GB per job
    (2, 8.0, 1, 2),      # tiny container
])
def test_plan_from_limits(monkeypatch, cpus, mem, jobs, threads):
    monkeypatch.setattr(resources, "available_cpus", lambda: cpus)
    monkeypatch.setattr(resources, "available_memory_gb", lambda: mem)
    p = resources.plan({"max_parallel_jobs": "auto", "threads_per_job": "auto"})
    assert (p["jobs"], p["threads_per_job"]) == (jobs, threads)
    assert p["jobs"] * p["threads_per_job"] <= max(cpus, threads)


def test_explicit_values_win(monkeypatch):
    monkeypatch.setattr(resources, "available_cpus", lambda: 16)
    monkeypatch.setattr(resources, "available_memory_gb", lambda: 128.0)
    p = resources.plan({"max_parallel_jobs": 2, "threads_per_job": 3})
    assert (p["jobs"], p["threads_per_job"]) == (2, 3)


def test_cgroup_v2_quota(monkeypatch):
    files = {"/sys/fs/cgroup/cpu.max": "800000 100000", "/sys/fs/cgroup/memory.max": "34359738368"}
    monkeypatch.setattr(resources, "_read", lambda p: files.get(p))
    assert resources.cgroup_cpus() == 8.0
    assert round(resources.cgroup_memory_gb(), 1) == 34.4
