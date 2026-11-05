"""Does the similarity depend on the outcome? Comparison of the metrics learned by two experiments (v0.5).

Agreed for the T1 experiment (and T3) against the earlier ones: on the same INSPIRE patients and outer folds,
- neighbour overlap: Jaccard index between the top neighbours each metric retrieves for the same test patient;
- weight ranks: Spearman correlation of the mean weights w_j of the common variables;
- shapes: for each variable shaped in both, correlation between the 1-D summaries (first principal direction) of
  their fold-0 consensus curves (targets have different classes, so the curves are compared through that summary).
Aggregates only: the per-patient neighbour lists stay in predictions/ (never exported).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import spearmanr

from perisim.evaluation.html import box, page, table
from perisim.utils import read_json, read_jsonl, write_json

LOG = logging.getLogger("perisim.compare")
ORDER = ("t2", "t1", "t3")


def _summary_1d(curve: list[dict]) -> np.ndarray:
    G = np.array([v["g"] for v in curve], dtype=float)
    G = G - G.mean(0)
    if not np.isfinite(G).all() or np.allclose(G, 0):
        return np.zeros(len(curve))
    u, s, _ = np.linalg.svd(G, full_matrices=False)
    return u[:, 0] * s[0]


def compare_experiments(ws_a, ws_b, n_outer: int, min_n: int = 10) -> dict | None:
    """Metric of experiment A vs experiment B (Workspaces of the same base). None if either is incomplete."""
    ra, rb = read_json(ws_a.artifacts() / "results.json"), read_json(ws_b.artifacts() / "results.json")
    if not ra or not rb or not ra.get("targets") or not rb.get("targets"):
        return None
    ta, tb = next(iter(ra["targets"])), next(iter(rb["targets"]))
    A, B = ra["targets"][ta], rb["targets"][tb]
    out: dict = {"a": ws_a.experiment, "b": ws_b.experiment, "targets": [ta, tb]}
    wa, wb = A.get("weights_mean") or {}, B.get("weights_mean") or {}
    common = sorted(set(wa) & set(wb))
    if len(common) >= 3:
        rho = spearmanr([wa[v] for v in common], [wb[v] for v in common]).correlation
        out["weights"] = {"n_variables": len(common), "spearman": float(rho)}
    jac, per_fold = [], []
    for f in range(n_outer):
        pa = {r["patient_id"]: r for r in read_jsonl(ws_a.predictions("cv", "patient_output") / f"fold{f}.jsonl")}
        pb = {r["patient_id"]: r for r in read_jsonl(ws_b.predictions("cv", "patient_output") / f"fold{f}.jsonl")}
        js = []
        for pid in set(pa) & set(pb):
            na = {x["case_id"] for x in (pa[pid].get("neighbors") or [])}
            nb = {x["case_id"] for x in (pb[pid].get("neighbors") or [])}
            if na and nb:
                js.append(len(na & nb) / len(na | nb))
        if len(js) >= min_n:
            per_fold.append({"fold": f, "n_patients": len(js), "jaccard_mean": float(np.mean(js)),
                             "jaccard_median": float(np.median(js))})
            jac.extend(js)
    if jac:
        out["neighbours"] = {"n_patients": len(jac), "jaccard_mean": float(np.mean(jac)),
                             "jaccard_p10": float(np.percentile(jac, 10)), "jaccard_p90": float(np.percentile(jac, 90)),
                             "per_fold": per_fold}
    sa, sb = A.get("shapes") or {}, B.get("shapes") or {}
    rows = []
    for v in sorted(set(sa) & set(sb)):
        ca, cb = (sa[v] or {}).get("curve_fold0"), (sb[v] or {}).get("curve_fold0")
        if not ca or not cb or len(ca) != len(cb) or len(ca) < 3:
            continue
        xa, xb = _summary_1d(ca), _summary_1d(cb)
        if np.allclose(xa, 0) or np.allclose(xb, 0):
            continue
        rows.append({"variable": v, "abs_correlation": float(abs(np.corrcoef(xa, xb)[0, 1]))})
    if rows:
        out["shapes"] = {"n_variables": len(rows), "median_abs_correlation":
                         float(np.median([r["abs_correlation"] for r in rows])), "per_variable": rows}
    return out


def render_comparison(c: dict) -> str:
    sec = [box(f"Metric of experiment <b>{c['b']}</b> ({c['targets'][1]}) against experiment <b>{c['a']}</b> "
               f"({c['targets'][0]}), on the same INSPIRE patients and outer folds: does the similarity depend on the "
               "outcome? Jaccard 1 = the same neighbours; 0 = disjoint.")]
    if c.get("neighbours"):
        n = c["neighbours"]
        sec.append("<h2>Neighbour overlap</h2>" + table(["Patients", "Jaccard mean", "p10", "p90"],
                                                         [[n["n_patients"], n["jaccard_mean"], n["jaccard_p10"],
                                                           n["jaccard_p90"]]], nd=3))
        sec.append(table(["Fold", "Patients", "Jaccard mean", "Jaccard median"],
                         [[r["fold"], r["n_patients"], r["jaccard_mean"], r["jaccard_median"]] for r in n["per_fold"]],
                         nd=3))
    if c.get("weights"):
        w = c["weights"]
        sec.append("<h2>Weight ranks</h2>" + table(["Common variables", "Spearman ρ"], [[w["n_variables"], w["spearman"]]],
                                                    nd=3))
    if c.get("shapes"):
        s = c["shapes"]
        sec.append("<h2>Shapes (1-D summary of the fold-0 curves)</h2>" +
                   table(["Variable", "|correlation|"], [[r["variable"], r["abs_correlation"]] for r in s["per_variable"]],
                         nd=3))
    return page(f"PERISIM — similarity of {c['b']} vs {c['a']}", sec, "PERISIM · aggregates only")


def run_comparisons(pipe) -> None:
    """In the report of a later experiment, compare with every earlier experiment already reported."""
    me = pipe.ws.experiment
    if me not in ORDER or pipe.variant != "main":
        return
    for other in ORDER[: ORDER.index(me)]:
        ws_o = pipe.ws.__class__(pipe.ws.base, "main", other)
        try:
            c = compare_experiments(ws_o, pipe.ws, int(pipe.cfg.experiment["cv"]["outer_folds"]),
                                    int(pipe.cfg.experiment["evaluation"]["small_cell_threshold"]))
        except Exception as exc:          # never block the report
            LOG.warning("comparison %s vs %s failed: %s", me, other, exc)
            continue
        if c:
            write_json(pipe.ws.artifacts() / f"comparison_{other}_{me}.json", c)
            (pipe.ws.reports / f"comparison_{other}_{me}.html").write_text(render_comparison(c), encoding="utf-8")
            LOG.info("comparison with %s written (reports/comparison_%s_%s.html)", other, other, me)
