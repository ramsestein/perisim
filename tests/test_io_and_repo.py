"""AUDIT A1/A2/A3/D16 — the published repo contains every module; identifier filters survive float ids;
CSV headers with a BOM; discovery walks the PhysioNet `files/` layout."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest

from perisim.data.io import load_table, read_header

ROOT = Path(__file__).resolve().parents[1]


def test_every_source_module_is_tracked_by_git():
    if not (ROOT / ".git").exists() or not shutil.which("git"):
        pytest.skip("not a git checkout")
    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", "--stdin"],
                             input="\n".join(str(p.relative_to(ROOT)).replace("\\", "/")
                                             for p in (ROOT / "src").rglob("*.py")),
                             capture_output=True, text=True).stdout.split()
    assert not ignored, f".gitignore excludes source files: {ignored[:5]}"


def test_float_ids_still_match(tmp_path):
    pq = tmp_path / "t.parquet"
    pl.DataFrame({"hadm_id": [20000001.0, 20000002.0], "v": [1, 2]}).write_parquet(pq)
    assert load_table(pq, {"hadm_id": "hadm_id", "v": "v"}, {"hadm_id": ["20000001"]}).height == 1
    csv = tmp_path / "t.csv"
    csv.write_text("hadm_id,v\n20000001.0,1\n20000002,2\n", encoding="utf-8")
    out = load_table(csv, {"hadm_id": "hadm_id", "v": "v"}, {"hadm_id": [20000001, "20000002.0"]}, tmp_path / "c")
    assert out.height == 2


def test_bom_header(tmp_path):
    p = tmp_path / "b.csv"
    p.write_bytes("﻿subject_id,x\n1,2\n".encode("utf-8"))
    assert read_header(p)[0] == "subject_id"


def test_discovery_walks_physionet_files_layout(tmp_path, cfg):
    from perisim.data.discover import discover_cohorts

    d = tmp_path / "physionet.org" / "files" / "inspire" / "1.4.2"
    d.mkdir(parents=True)
    (d / "operations.csv").write_text("op_id\n1\n", encoding="utf-8")
    c = cfg.with_overrides({"experiment": {"paths": {"datasets_root": str(tmp_path), "inspire_dir": "auto",
                                                     "mimic_dir": "auto", "eicu_dir": "auto"}}})
    assert discover_cohorts(c)["inspire"]["path"] == str(d)
