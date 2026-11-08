"""Copies into results/ the aggregate outputs that `publish_results.py` (T2's export package) and the hand copies of
`hypo` left out, so the repository holds every result the manuscript uses. Aggregates only; every file is checked,
BEFORE anything is written (v0.6), for size (≤ 2 MB, as scripts/check_public_safety.py), identifiers and small cells
(perisim.evaluation.privacy.scan_text, the scanner of check_public_safety.py); if any file fails, nothing is written.

- results/t2/flow/: the cohort flow of the extraction (work/common/<cohort>/flow.json). A step that excludes 1–9
  stays, or loses 1–9 patients (v0.6), is merged with the next step, so no count below 10 can be read or derived.
- results/t2/results/transport/: the two transport tables refused for size (t2_C_accumulation, t2_F_hospitals),
  values rounded to 6 significant digits and split into parts of < 2 MB. Hospitals are labelled H1, H2, … as in
  the export.
- results/feasibility/: the two feasibility studies (ICU course; hypotension) that preceded the choice of `hypo`.
- results/hypo/results/external_dryrun_alpha0/: the first dry run of the external stage, the one that found the
  α = 0 fault (DECISIONS 2026-10-29). It is kept for the record; it is not evidence.
- results/hypo/: the second round of post hoc analyses (scripts/posthoc_hypo_ablation.py), when it has run, and the
  third round's strata (scripts/posthoc_hypo_strata.py) and MAP–vasopressor test (scripts/posthoc_hypo_map_vaso.py).
- results/hypo/ and results/t2/: the transport equivalence (scripts/posthoc_equivalence.py), when it has run.
- results/alignment/: the neighbour alignment of `hypo` and T2 (scripts/posthoc_alignment.py), once its `hypo` part
  has run.

Usage:  python scripts/publish_supplement.py [--workdir <work folder>]
"""

from __future__ import annotations

import argparse
import os
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perisim.evaluation import privacy as PV  # noqa: E402

MAX_BYTES = 1_900_000


def sig(x: str, n: int = 6) -> str:
    try:
        v = float(x)
    except ValueError:
        return x
    return x if v != v else f"{v:.{n}g}"


def flow(work: Path, out: Path) -> dict[Path, bytes]:
    """{path: content} of the flow tables. v0.6: privacy.flow's merge rule, on stays AND patients (a step whose
    number of patients falls by 1–9 let the excluded patients be read off consecutive rows)."""
    made, md = {}, ["# Cohort flow of the extraction (T2)", "",
                    "Steps that exclude 1–9 stays or lose 1–9 patients are merged with the next step (no count below "
                    "10 can be read or derived). Counts: stays (patients).", ""]
    for c in ("inspire", "mimic", "eicu"):
        src = work / "common" / c / "flow.json"
        if not src.exists():
            continue
        merged = [{k: v for k, v in s.items() if k != "rounded"}
                  for s in PV.flow(json.loads(src.read_text(encoding="utf-8")))]
        assert all(not (isinstance(s.get(k), int) and 0 < s[k] < 10) for s in merged for k in ("n_stays", "n_subjects", "excluded"))
        made[out / f"flow_{c}.json"] = json.dumps(merged, indent=1, ensure_ascii=False).encode("utf-8")

        def fmt(v):
            return "" if v is None else (format(v, ",") if isinstance(v, int) else str(v))
        md += [f"## {c}", "", "| step | stays | patients | excluded stays |", "|---|---|---|---|"]
        md += [f"| {s['step']} | {fmt(s['n_stays'])} | {fmt(s['n_subjects'])} | {fmt(s['excluded'])} |" for s in merged]
        md.append("")
    made[out / "flow.md"] = "\n".join(md).encode("utf-8")
    return made


def split_csv(src: Path, out: Path, stem: str) -> dict[Path, bytes]:
    """{path: content} of the parts (< MAX_BYTES each) of a large table, values rounded to 6 significant digits."""
    rows = list(csv.reader(open(src, encoding="utf-8", newline="")))
    head, body = rows[0], [[sig(v) for v in r] for r in rows[1:]]
    parts, cur, size = [], [], 0
    for r in body:
        line = len(",".join(r)) + 1
        if cur and size + line > MAX_BYTES - 2000:
            parts.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += line
    parts.append(cur)
    made = {}
    for i, p in enumerate(parts):
        name = f"{stem}.csv" if len(parts) == 1 else f"{stem}_part{i + 1}.csv"
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(head)
        w.writerows(p)
        made[out / name] = buf.getvalue().encode("utf-8")
    return made


def copy_files(pairs: list[tuple[Path, Path]]) -> dict[Path, bytes]:
    return {d: s.read_bytes() for s, d in pairs if s.exists()}


def problems(files: dict[Path, bytes], cohort_n: dict) -> list[str]:
    """v0.6: everything is checked BEFORE anything is written: size (≤ 2 MB, as check_public_safety.py),
    identifiers and small cells (the same scanner as check_public_safety.py)."""
    out = []
    for p, b in files.items():
        rel = p.relative_to(ROOT).as_posix() if ROOT in p.parents else str(p)
        if len(b) > 2_000_000:
            out.append(f"{rel}: too large ({len(b)} bytes)")
            continue
        if p.suffix.lower() == ".png":
            continue
        text = b.decode("utf-8", errors="ignore")
        if PV.ID_RE.search(text):
            out.append(f"{rel}: contains identifiers")
        cells = PV.scan_text(p.name, text, cohort_n)
        if cells:
            out.append(f"{rel}: {len(cells)} small cell(s), e.g. {cells[0]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", default=str(ROOT / f"work_{os.environ.get('PERISIM_ARM') or 'arc'}"))
    a = ap.parse_args(argv)
    work = Path(a.workdir)
    res = ROOT / "results"
    made: dict[Path, bytes] = {}
    made |= flow(work, res / "t2" / "flow")
    tsrc = work / "experiments" / "t2" / "export" / "results" / "transport"
    tdst = res / "t2" / "results" / "transport"
    stale = []
    for stem in ("t2_C_accumulation", "t2_F_hospitals"):
        if (tsrc / f"{stem}.csv").exists():
            stale += list(tdst.glob(f"{stem}*.csv"))
            made |= split_csv(tsrc / f"{stem}.csv", tdst, stem)
    fz = work / "feasibility"
    made |= copy_files([(fz / "feasibility_icu_trajectory.md", res / "feasibility" / "icu_trajectory.md"),
                        (fz / "feasibility_icu_trajectory.json", res / "feasibility" / "icu_trajectory.json"),
                        (fz / "hypotension" / "feasibility_hypotension.md", res / "feasibility" / "hypotension.md"),
                        (fz / "hypotension" / "feasibility_hypotension.json", res / "feasibility" / "hypotension.json")])
    hy = work / "experiments" / "hypo"
    a0 = hy / "external_dryrun_alpha0"
    made |= copy_files([(f, res / "hypo" / "results" / "external_dryrun_alpha0" / f.name)
                        for f in sorted(a0.glob("*.json")) + sorted(a0.glob("*.md"))])
    ab = hy / "posthoc_ablation"
    made |= copy_files([(f, res / "hypo" / "results" / "posthoc_ablation" / f.name) for f in sorted(ab.glob("*.json"))])
    made |= copy_files([(hy / "reports" / "posthoc_ablation.md", res / "hypo" / "reports" / "posthoc_ablation.md")]
                       + [(f, res / "hypo" / "reports" / "posthoc_ablation" / f.name)
                          for f in sorted((hy / "reports" / "posthoc_ablation").glob("*.png"))])
    made |= copy_files([(hy / "posthoc_strata" / "strata.json", res / "hypo" / "results" / "posthoc_strata" / "strata.json"),
                        (hy / "reports" / "posthoc_strata.md", res / "hypo" / "reports" / "posthoc_strata.md")])
    made |= copy_files([(hy / "posthoc_map_vaso" / "map_vaso.json", res / "hypo" / "results" / "posthoc_map_vaso" / "map_vaso.json"),
                        (hy / "reports" / "posthoc_map_vaso.md", res / "hypo" / "reports" / "posthoc_map_vaso.md")]
                       + [(f, res / "hypo" / "reports" / "posthoc_map_vaso" / f.name)
                          for f in sorted((hy / "reports" / "posthoc_map_vaso").glob("*.png"))])
    eq_t2 = work / "experiments" / "t2"
    made |= copy_files([(hy / "posthoc_equivalence" / "equivalence.json",
                         res / "hypo" / "results" / "posthoc_equivalence" / "equivalence.json"),
                        (hy / "reports" / "posthoc_equivalence.md", res / "hypo" / "reports" / "posthoc_equivalence.md"),
                        (eq_t2 / "artifacts" / "main" / "posthoc_equivalence.json",
                         res / "t2" / "artifacts" / "main" / "posthoc_equivalence.json"),
                        (eq_t2 / "reports" / "posthoc_equivalence.md", res / "t2" / "reports" / "posthoc_equivalence.md")])
    al = work / "experiments" / "posthoc_alignment"
    if (al / "posthoc_alignment.json").exists() and json.loads((al / "posthoc_alignment.json").read_text(encoding="utf-8"))["hypo"]:
        made |= copy_files([(f, res / "alignment" / f.name) for f in sorted(al.glob("alignment_*.json"))]
                           + [(al / "posthoc_alignment.md", res / "alignment" / "posthoc_alignment.md")]
                           + [(f, res / "alignment" / f.name) for f in sorted(al.glob("posthoc_alignment*.png"))])
    cohort_n = {}
    for c in PV.COHORTS:
        f = res / "t2" / "flow" / f"flow_{c}.json"
        b = made.get(f) or (f.read_bytes() if f.exists() else None)
        if b:
            steps = json.loads(b)
            if steps and isinstance(steps[-1].get("n_stays"), int):
                cohort_n[c] = steps[-1]["n_stays"]
    bad = problems(made, cohort_n)
    if bad:                                        # v0.6: nothing is written when any file fails
        print("NOT WRITTEN — problems found:\n  " + "\n  ".join(bad))
        return 1
    for old in stale:
        if old not in made:
            old.unlink()
    for p, b in made.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        print(p.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
