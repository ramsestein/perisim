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


@pytest.fixture(scope="session")
def cfg():
    from perisim.config import load_config

    return load_config(profile="smoke")


@pytest.fixture(scope="session")
def real_workdir():
    wd = os.environ.get("PERISIM_TEST_WORKDIR")
    return (Path(wd), os.environ.get("PERISIM_TEST_VARIANT", "main")) if wd else None


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
