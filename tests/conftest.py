"""Fixtures: small synthetic data (no real data) or, on the pod, the real workdir artifacts
(PERISIM_TEST_WORKDIR) for the anti-leakage and split tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@pytest.fixture(autouse=True)
def _arc_guard_off_for_synthetic_chains(request, monkeypatch):
    """Branch arc-metric: the end-to-end tests build synthetic work folders in tmp_path (with experiments/ inside)
    and call the command lines, which would refuse them as unmarked. The guard itself is tested in test_arc.py."""
    if request.module.__name__.endswith("test_arc"):
        return
    import perisim.arc_guard as G

    def _noop(path=None, arm=None):
        return Path(path if path is not None else os.environ.get("PERISIM_WORKDIR") or G.default_workdir_name(arm))

    monkeypatch.setattr(G, "ensure_arc_workdir", _noop)


@pytest.fixture(scope="session")
def cfg():
    from perisim.config import load_config

    return load_config(profile="smoke")


@pytest.fixture(scope="session")
def real_workdir():
    """(experiment root, variant). The shared data layer (common/) is in real_base."""
    wd = os.environ.get("PERISIM_TEST_WORKDIR")
    return (Path(wd), os.environ.get("PERISIM_TEST_VARIANT", "main")) if wd else None


@pytest.fixture(scope="session")
def real_base():
    b = os.environ.get("PERISIM_TEST_BASE") or os.environ.get("PERISIM_TEST_WORKDIR")
    return Path(b) if b else None


@pytest.fixture(scope="session")
def synthetic_inspire(tmp_path_factory, cfg):
    """Synthetic INSPIRE extracted to the common format (index stays + inputs_long + outcomes)."""
    from perisim.cohort.select import select_index_stays
    from perisim.data.inspire import InspireExtractor
    from perisim.data.synthetic import synth_inspire

    import numpy as np

    root = tmp_path_factory.mktemp("synth")
    synth_inspire(root, 160, np.random.default_rng(3))
    ex = InspireExtractor(cfg, root / "inspire", root / "cache")
    cand = ex.extract_stays()
    idx, flow = select_index_stays(cand, cfg.cohorts, "inspire")
    inp, outc = ex.extract_events(idx)
    return {"stays": idx, "inputs": inp, "outcomes": outc, "flow": flow, "candidates": cand}
