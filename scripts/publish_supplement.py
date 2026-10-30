"""Copies into results/ the aggregate outputs that `publish_results.py` (T2's export package) and the hand copies of
`hypo` left out, so the repository holds every result the manuscript uses. Aggregates only; every file is checked
again for size (≤ 2 MB, as scripts/check_public_safety.py) and small cells.

- results/t2/flow/: the cohort flow of the extraction (work/common/<cohort>/flow.json). A step that excludes 1–9
  stays is merged with the next step, so no count below 10 can be read or derived.
- results/t2/results/transport/: the two transport tables refused for size (t2_C_accumulation, t2_F_hospitals),
  values rounded to 6 significant digits and split into parts of < 2 MB. Hospitals are labelled H1, H2, … as in
  the export.
- results/feasibility/: the two feasibility studies (ICU course; hypotension) that preceded the choice of `hypo`.
- results/hypo/results/external_dryrun_alpha0/: the first dry run of the external stage, the one that found the
  α = 0 fault (DECISIONS 2026-10-29). It is kept for the record; it is not evidence.
- results/hypo/: the second round of post hoc analyses (scripts/posthoc_hypo_ablation.py), when it has run.

Usage:  python scripts/publish_supplement.py [--workdir <work folder>]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1_900_000


def sig(x: str, n: int = 6) -> str:
    try:
        v = float(x)
    except ValueError:
        return x
    return x if v != v else f"{v:.{n}g}"


def flow(work: Path, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    made, md = [], ["# Cohort flow of the extraction (T2)", "",
                    "Steps that exclude 1–9 stays are merged with the next step (no count below 10 is shown). Counts: "
                    "stays (patients).", ""]
    for c in ("inspire", "mimic", "eicu"):
        src = work / "common" / c / "flow.json"
        if not src.exists():
            continue
        steps = json.loads(src.read_text(encoding="utf-8"))
        merged, carry = [], None
        for s in steps:
            s = dict(s)
            if carry is not None:
                s["step"] = carry["step"] + "; " + s["step"]
                s["excluded"] = (carry["excluded"] or 0) + (s["excluded"] or 0)
                carry = None
            if isinstance(s.get("excluded"), int) and 0 < s["excluded"] < 10:
                carry = s
                continue
            merged.append(s)
        if carry is not None:                       # a small last step: merge it backwards
            prev = merged.pop()
            merged.append({"step": prev["step"] + "; " + carry["step"], "n_stays": carry["n_stays"],
                           "n_subjects": carry["n_subjects"],
                           "excluded": (prev["excluded"] or 0) + (carry["excluded"] or 0)})
        assert all(not (isinstance(s.get(k), int) and 0 < s[k] < 10) for s in merged for k in ("n_stays", "n_subjects", "excluded"))
        (out / f"flow_{c}.json").write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")
        made.append(f"flow_{c}.json")
        md += [f"## {c}", "", "| step | stays | patients | excluded stays |", "|---|---|---|---|"]
        md += [f"| {s['step']} | {s['n_stays']:,} | {s['n_subjects']:,} | "
               f"{'' if s['excluded'] is None else format(s['excluded'], ',')} |" for s in merged]
        md.append("")
    (out / "flow.md").write_text("\n".join(md), encoding="utf-8")
    return made + ["flow.md"]


def split_csv(src: Path, out: Path, stem: str) -> list[str]:
    rows = list(csv.reader(open(src, encoding="utf-8", newline="")))
    head, body = rows[0], [[sig(v) for v in r] for r in rows[1:]]
    for old in out.glob(f"{stem}*.csv"):
        old.unlink()
    parts, cur, size = [], [], 0
    for r in body:
        line = len(",".join(r)) + 1
        if cur and size + line > MAX_BYTES - 2000:
            parts.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += line
    parts.append(cur)
    made = []
    for i, p in enumerate(parts):
        name = f"{stem}.csv" if len(parts) == 1 else f"{stem}_part{i + 1}.csv"
        with open(out / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(head)
            w.writerows(p)
        made.append(name)
    return made


def copy_files(pairs: list[tuple[Path, Path]]) -> list[str]:
    made = []
    for s, d in pairs:
        if not s.exists():
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
        made.append(str(d.relative_to(ROOT)))
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", default=str(ROOT / "work"))
    a = ap.parse_args(argv)
    work = Path(a.workdir)
    res = ROOT / "results"
    made = []
    made += [f"results/t2/flow/{f}" for f in flow(work, res / "t2" / "flow")]
    tsrc = work / "experiments" / "t2" / "export" / "results" / "transport"
    tdst = res / "t2" / "results" / "transport"
    for stem in ("t2_C_accumulation", "t2_F_hospitals"):
        if (tsrc / f"{stem}.csv").exists():
            made += [f"results/t2/results/transport/{f}" for f in split_csv(tsrc / f"{stem}.csv", tdst, stem)]
    fz = work / "feasibility"
    made += copy_files([(fz / "feasibility_icu_trajectory.md", res / "feasibility" / "icu_trajectory.md"),
                        (fz / "feasibility_icu_trajectory.json", res / "feasibility" / "icu_trajectory.json"),
                        (fz / "hypotension" / "feasibility_hypotension.md", res / "feasibility" / "hypotension.md"),
                        (fz / "hypotension" / "feasibility_hypotension.json", res / "feasibility" / "hypotension.json")])
    hy = work / "experiments" / "hypo"
    a0 = hy / "external_dryrun_alpha0"
    made += copy_files([(f, res / "hypo" / "results" / "external_dryrun_alpha0" / f.name)
                        for f in sorted(a0.glob("*.json")) + sorted(a0.glob("*.md"))])
    ab = hy / "posthoc_ablation"
    made += copy_files([(f, res / "hypo" / "results" / "posthoc_ablation" / f.name) for f in sorted(ab.glob("*.json"))])
    made += copy_files([(hy / "reports" / "posthoc_ablation.md", res / "hypo" / "reports" / "posthoc_ablation.md")]
                       + [(f, res / "hypo" / "reports" / "posthoc_ablation" / f.name)
                          for f in sorted((hy / "reports" / "posthoc_ablation").glob("*.png"))])
    big = [m for m in made if (ROOT / m).stat().st_size > 2_000_000]
    for m in made:
        print(m)
    if big:
        print("TOO LARGE:", big)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
