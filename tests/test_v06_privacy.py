"""v0.6 privacy and post hoc fixes (software audit 2026-11-05, items 1, 2, 9, 10 and export.publish)."""

import csv
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from perisim.evaluation import privacy as PV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- 1. complementary suppression
def test_suppressed_shares_cannot_be_recovered_by_subtraction():
    ph = _script("posthoc_transport")
    counts = np.array([7.0, 300, 45, 648])
    s = ph.suppressed_shares(counts)
    assert np.isnan(s[0]) and np.isnan(s[2])                 # the next-smallest bin is hidden too
    assert counts[np.isnan(s)].sum() >= 10
    s = ph.suppressed_shares(np.array([3.0, 4, 50, 45, 200]))  # two small bins adding up to 7: one more hidden
    assert np.isnan(s[:2]).all() and np.isnan(s[3]) and not np.isnan(s[2])
    s = ph.suppressed_shares(np.array([0.0, 0, 50, 45]))       # empty bins: nothing to protect
    assert np.isnan(s[:2]).all() and np.isclose(s[2], 50 / 95) and np.isclose(s[3], 45 / 95)
    s = ph.suppressed_shares(np.array([12.0, 30, 58]))         # nothing small: unchanged
    assert np.allclose(s, [0.12, 0.30, 0.58])


# --------------------------------------------------------------------------- 3. privacy.counts / flow
def test_counts_plain_labels_and_hidden_total():
    out = PV.counts({"a": 9, "b": 10, "c": 100})
    assert out == {"a": "<10", "b": "<10", "c": 100}           # no "<11" upper bound
    out = PV.counts({"a": 3, "b": 4, "c": 100, "d": 50})
    hidden = [k for k, v in out.items() if v == "<10"]
    assert sum({"a": 3, "b": 4, "c": 100, "d": 50}[k] for k in hidden) >= 10 and out["c"] == 100
    assert PV.counts({"a": 0, "b": 20, "c": None}) == {"a": 0, "b": 20, "c": None}


def test_flow_merges_small_steps_of_stays_and_patients():
    rows = [{"step": "all", "n_stays": 1000, "n_subjects": 900, "excluded": None},
            {"step": "s1", "n_stays": 994, "n_subjects": 895, "excluded": 6},       # 6 stays: merged
            {"step": "s2", "n_stays": 950, "n_subjects": 860, "excluded": 44},
            {"step": "s3", "n_stays": 930, "n_subjects": 855, "excluded": 20},      # 5 patients: merged
            {"step": "s4", "n_stays": 900, "n_subjects": 830, "excluded": 30},
            {"step": "s5", "n_stays": 897, "n_subjects": 829, "excluded": 3}]       # small last step: backwards
    out = PV.flow(rows)
    assert [r["step"] for r in out] == ["all", "s1; s2", "s3; s4; s5"]
    for a, b in zip(out, out[1:]):
        for k in ("n_stays", "n_subjects"):
            assert a[k] - b[k] == 0 or a[k] - b[k] >= 10
    assert all(not (isinstance(r["excluded"], int) and 0 < r["excluded"] < 10) for r in out)


# --------------------------------------------------------------------------- the scanner
def test_scan_record_rules():
    assert PV.scan_record({"n": 283, "ood_pct": "0.353357"})              # 1 of 283
    assert PV.scan_record({"n": 283, "ood_pct": "99.6466"})               # complement 1
    assert not PV.scan_record({"n": 283, "ood_pct": "10.6007"})           # 30 of 283
    assert not PV.scan_record({"n": 11, "pct": 0.05})                     # "n" is the numerator (audit.py)
    assert PV.scan_record({"n": 300, "n_pos": 295})                       # complement 5
    assert PV.scan_record({"stays": 4})
    assert not PV.scan_record({"n_folds": 5, "k": 3, "n": 0})
    # a full-precision rate next to an n rounded to 10 reveals the exact k / m
    assert PV.scan_record({"n": 340, "rate": 5 / 343})
    # per-variable % without n: the cohort n (missing) and the non-missing n (outside the range)
    rec = {"variable": "x", "missing_pct_mimic": "0.0", "outside_range_pct_mimic": str(100 * 3 / 24768)}
    assert PV.scan_record(rec, {"mimic": 24768})
    assert PV.scan_text_cells("1.5% (0.3%–2.8%; n 340)") and not PV.scan_text_cells("15.4% (15.0%–15.8%; n 65,000)")


def test_scan_shares_and_csv():
    text = "variable,site,bin,share\nv,a,0,0.5\nv,a,1,0.49\nv,a,2,NaN\nw,a,0,0.5\nw,a,1,0.5\n"
    assert any("v/a" in x for x in PV.scan_csv_text(text))
    assert not PV.scan_csv_text(text.replace("v,a,1,0.49", "v,a,1,NaN"))


# --------------------------------------------------------------------------- 2. sanitizer
def _w(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)


def test_sanitizer_blanks_only_leaking_values(tmp_path):
    S = _script("sanitize_results")
    root = tmp_path / "results"
    (root / "t2" / "flow").mkdir(parents=True)
    for c, n in (("inspire", 1000), ("mimic", 2000), ("eicu", 3000)):
        (root / "t2" / "flow" / f"flow_{c}.json").write_text(json.dumps([{"step": "x", "n_stays": n}]))
    f = root / "t2" / "results" / "transport" / "t2_F_hospitals_part1.csv"
    _w(f, [["cohort", "hospital", "n", "logloss", "ood_pct"],
           ["eicu", "H1", "283", "1.0037", str(100 / 283)],               # 1 OOD stay: blanked
           ["eicu", "H2", "283", "0.9", str(3000 / 283)]])                # 30: kept
    e = root / "t2" / "results" / "transport" / "t2_E_coverage_missingness.csv"
    _w(e, [["variable", "missing_pct_mimic", "outside_range_pct_mimic"],
           ["a", "50.0", str(100 * 4 / 1000)],                            # 4 of the 1000 non-missing: blanked
           ["b", "0.0", str(100 * 40 / 2000)]])
    sh = root / "t2" / "results" / "posthoc" / "t2_shift_shares.csv"
    _w(sh, [["variable", "site", "bin", "value", "share"],
            ["v", "mimic", "0", "<1", "NaN"], ["v", "mimic", "1", "1-2", str(20 / 1000)],
            ["v", "mimic", "2", ">2", str(973 / 1000)]])
    sv = root / "t2" / "results" / "posthoc" / "t2_shift_variables.csv"
    _w(sv, [["variable", "missing_pct_mimic"], ["v", "50.0"]])
    ch = S.sanitize(root)
    assert len(ch) == 3
    F = list(csv.reader(open(f, encoding="utf-8")))
    assert F[1][4] == "" and F[1][3] == "1.0037" and F[2][4] == str(3000 / 283)
    E = list(csv.reader(open(e, encoding="utf-8")))
    assert E[1][2] == "" and E[1][1] == "50.0" and E[2][2] == str(100 * 40 / 2000)
    SH = list(csv.reader(open(sh, encoding="utf-8")))
    assert SH[2][4] == "NaN" and SH[3][4] == str(973 / 1000)            # complementary suppression
    assert S.sanitize(root) == {}                                         # idempotent
    cn = S.cohort_sizes(root)
    for p in (f, e, sh):
        assert not PV.scan_text(p.name, p.read_text(encoding="utf-8"), cn)


def test_published_results_are_clean():
    """The repository's results/ pass the small-cell scan (CSV, JSON, Markdown and HTML reports)."""
    C = _script("check_public_safety")
    files = [p.relative_to(ROOT).as_posix() for p in (ROOT / "results").rglob("*")
             if p.suffix in (".csv", ".json", ".md", ".html")]
    content = {f: (ROOT / f).read_bytes() for f in files}
    cn = C.cohort_sizes(content)
    bad = [(f, PV.scan_text(f, content[f].decode("utf-8", "ignore"), cn)[:1]) for f in files
           if PV.scan_text(f, content[f].decode("utf-8", "ignore"), cn)]
    assert not bad


# --------------------------------------------------------------------------- 2. check_public_safety
def test_check_public_safety_reads_the_index(tmp_path, monkeypatch):
    C = _script("check_public_safety")
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(git + ["init", "-q"], cwd=repo, check=True)
    monkeypatch.setattr(C, "ROOT", repo)
    (repo / "big.txt").write_text("x" * (C.MAX_BYTES + 10))
    (repo / "results").mkdir()
    (repo / "results" / "t.csv").write_text("cohort,n,ood_pct\neicu,283,0.353357\n")
    (repo / "results" / "ok.csv").write_text("cohort,n,logloss\neicu,283,1.0\n")
    (repo / "dump.jsonl").write_text("{}\n")
    (repo / "data.csv").write_text("a\n1\n")
    (repo / "x.json").write_text('{"note": "stay EICU-12"}')
    subprocess.run(git + ["add", "-A"], cwd=repo, check=True)
    (repo / "big.txt").write_text("small now")                          # truncated in the working tree only
    files = C._files(False)
    probs = "\n".join(C.check(files, C._staged_blobs(files)))
    assert "big.txt: too large" in probs                                  # read from the index
    assert "results/t.csv" in probs and "small cell" in probs
    assert "results/ok.csv" not in probs
    assert "dump.jsonl: data file type" in probs and "data.csv: data file type" in probs
    assert "x.json: possible patient/stay identifier" in probs
    allp = "\n".join(C.check(C._files(True), C._worktree(C._files(True))))
    assert "big.txt" not in allp                                           # --all: the working tree


def test_export_publish_scans_identifiers(tmp_path):
    from perisim.evaluation.export import ID_RE, has_identifiers

    assert ID_RE is PV.ID_RE and PV.ID_RE_CODE.search("MIMIC-" + "12345") and not PV.ID_RE_CODE.search("MIMIC-11")
    (tmp_path / "a.json").write_text('{"id": "INSPIRE-3"}')
    (tmp_path / "b.json").write_text('{"n": 300}')
    with zipfile.ZipFile(tmp_path / "c.zip", "w") as z:
        z.writestr("x.md", "EICU-7")
    (tmp_path / "d.zip").write_bytes(b"PK")
    assert has_identifiers(tmp_path / "a.json") and not has_identifiers(tmp_path / "b.json")
    assert has_identifiers(tmp_path / "c.zip") and has_identifiers(tmp_path / "d.zip")


def test_audit_quantiles_need_values_beyond_them():
    from perisim.evaluation.audit import _quantiles

    q = _quantiles(np.arange(40.0), [0.05, 0.25, 0.5, 0.75, 0.95], 10)
    assert set(q) == {0.25, 0.5, 0.75}                                     # p5 / p95 need 200 values
    assert set(_quantiles(np.arange(200.0), [0.05, 0.95], 10)) == {0.05, 0.95}


# --------------------------------------------------------------------------- 8. post hoc scripts
def test_strata_shares_gated_by_their_denominator():
    st = _script("posthoc_hypo_strata")
    rng = np.random.default_rng(0)
    assert not st.evaluable(0.01, rng.normal(0.01, 0.02, 500))            # CI includes 0
    assert not st.evaluable(0.001, rng.normal(0.001, 0.0001, 500))        # tiny
    assert st.evaluable(0.02, rng.normal(0.02, 0.002, 500), positive=True)
    assert not st.evaluable(-0.02, rng.normal(-0.02, 0.002, 500), positive=True)
    r = st.ratio(1.0, 2.0, rng.normal(0.5, 0.1, 500), False, 0.001, rng.normal(0.001, 0.01, 500))
    assert r["est"] is None and r["verdict"] == "not evaluable" and st._r(r) == "not evaluable"
    assert st.ratio(1.0, 2.0, rng.normal(0.5, 0.1, 500), True, 0.02, rng.normal(0.02, 0.002, 500))["est"] == 0.5


def test_alignment_reproduction_gate_fails_on_missing_rows():
    al = _script("posthoc_alignment")
    p = np.array([0.1, 0.2, 0.3])
    assert al.reproduces(al.reproduction(p, p.copy()))
    assert not al.reproduces(al.reproduction(np.full(3, np.nan), p))      # all missing: used to pass
    assert not al.reproduces(al.reproduction(np.array([0.1, np.nan, 0.3]), p))
    assert not al.reproduces(al.reproduction(p + 1e-6, p))
    assert al.reproduces(None) and not al.reproduces(float("nan"))
    k1 = al.freeze_key({"hashes": {"code": "a", "config": "b"}})
    assert k1 != al.freeze_key({"hashes": {"code": "a", "config": "c"}})
    al._FREEZE["key"] = k1
    assert al.cache_dir().name == k1
    al._FREEZE["key"] = None


def test_frozen_src_and_t2_freeze_check(tmp_path, monkeypatch):
    fz = _script("_frozen")
    monkeypatch.delenv("PERISIM_SRC", raising=False)
    assert fz.src_dir() == ROOT / "src"
    src = tmp_path / "frozen_src"
    src.mkdir()
    monkeypatch.setenv("PERISIM_SRC", str(src))
    before = list(sys.path)
    try:
        assert fz.use_src() == src.resolve() and sys.path[0] == str(src.resolve())
    finally:
        sys.path[:] = before
    monkeypatch.setenv("PERISIM_SRC", str(tmp_path / "missing"))
    with pytest.raises(SystemExit):
        fz.use_src()
    monkeypatch.delenv("PERISIM_SRC")

    from perisim.config import code_hash

    class Cfg:
        def __init__(self, h):
            self.h = h

        def scientific_hash(self):
            return self.h

    class Ws:
        state = tmp_path / "state"

    Ws.state.mkdir()
    (Ws.state / "freeze.json").write_text(json.dumps({"tag": "freeze-t2", "code_hash": code_hash(),
                                                      "config_hash": "c1"}))
    assert fz.check_t2_freeze(Cfg("c1"), Ws)["tag"] == "freeze-t2"
    with pytest.raises(SystemExit, match="PERISIM_SRC=frozen/t2/src"):
        fz.check_t2_freeze(Cfg("c2"), Ws)
    (Ws.state / "freeze.json").write_text(json.dumps({"tag": "t", "code_hash": "old", "config_hash": "c1"}))
    with pytest.raises(SystemExit, match="code_hash"):
        fz.check_t2_freeze(Cfg("c1"), Ws)
