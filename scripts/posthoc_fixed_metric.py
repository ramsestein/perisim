"""Post-hoc, EXPLORATORY (`hypo`): a fixed mutual-information metric against the transported learned metric.

Plan, arms, gate and rule: docs/DECISIONS.md, 2026-11-06 ("a fixed mutual-information metric against the transported
learned metric"). Run with `hypo`'s frozen code (PERISIM_SRC=frozen/hypo/src); the freeze record is checked at the
start and the end, and nothing frozen is refitted.

Every arm uses site X's own out-of-fold case base, the external stage's five folds, k = 50, distinct patients and the
same vote; only the metric changes:
- NCA-transported, NCA-local, Euclidean: the stored predictions;
- MI-transported: Euclidean kNN on the Euclidean comparator's neutral axes, variables weighted by their mutual
  information with the outcome on the two development sites' landmarks;
- MI-local: the same, mutual information on the fold's case base;
- NCA-weights (neutral axes): the transported metric's weights without its shapes.

Usage:  python scripts/posthoc_fixed_metric.py [--sites inspire mimic eicu] [--threads 8]
Outputs (aggregates only): work/experiments/hypo/posthoc_fixed_metric/fixed_metric.json and
reports/posthoc_fixed_metric.md; per-fold predictions of the new arms in posthoc_fixed_metric/units/ (row-level,
git-ignored work folder, never published).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:                                    # v0.6 repository: $PERISIM_SRC (frozen code) first on sys.path
    sys.path.insert(0, str(ROOT / "scripts"))
    import _frozen  # noqa: E402
    _frozen.use_src()
except ImportError:                     # a v0.5 checkout: its own src is the frozen code
    sys.path.insert(0, str(ROOT / "src"))

LOG = logging.getLogger("posthoc.fixed_metric")
N_BOOT = 2000
SEED = 20261106
N_QBINS = 20          # quantile bins of a continuous / ordinal variable for the mutual information
MIN_CAT = 50          # categories with fewer landmarks pooled
MIN_DEN = 0.005       # nats: the smallest denominator for which a share is read
GATE_TOL = 1e-9
ARMS = ["euclidean", "nca_transported", "nca_local", "mi_transported", "mi_local", "nca_w_neutral"]


# --------------------------------------------------------------------------- mutual information
def _entropy(counts: np.ndarray) -> float:
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log(p)).sum())


def plugin_mi(code: np.ndarray, y: np.ndarray) -> float:
    """I(x; y) in nats, plug-in, for a discrete code (int ≥ 0) and a binary outcome."""
    if code.size == 0:
        return 0.0
    k = int(code.max()) + 1
    joint = np.bincount(code * 2 + y.astype(np.int64), minlength=2 * k).reshape(k, 2).astype(np.float64)
    n = joint.sum()
    hy = _entropy(joint.sum(axis=0))
    h_cond = sum(joint[i].sum() / n * _entropy(joint[i]) for i in range(k) if joint[i].sum() > 0)
    return max(hy - h_cond, 0.0)


def _cat_code(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes of the observed categories (rare ones pooled) and the observed mask."""
    s = np.asarray(v, dtype=object)
    obs = np.array([x is not None and not (isinstance(x, float) and np.isnan(x)) and str(x) != "" for x in s])
    vals = np.array([str(x) for x in s[obs]], dtype=object)
    uniq, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
    keep = cnt >= MIN_CAT
    remap = np.where(keep, np.cumsum(keep) - 1, int(keep.sum()))      # rare categories → one pooled code
    return remap[inv].astype(np.int64), obs


def block_codes(block, ft) -> tuple[np.ndarray, np.ndarray] | None:
    """(code, observed mask) of one encoder block, or None for a kind without a defined code."""
    kind, src = block.kind, block.source
    if kind in ("continuous", "ordinal"):
        x = np.asarray(ft.num[src], dtype=np.float64)
        obs = np.isfinite(x)
        if obs.sum() < 2:
            return None
        edges = np.unique(np.quantile(x[obs], np.linspace(0, 1, N_QBINS + 1)))
        return np.searchsorted(edges[1:-1], x[obs], side="right").astype(np.int64), obs
    if kind == "missing":
        x = np.asarray(ft.num[src], dtype=np.float64)
        return (~np.isfinite(x)).astype(np.int64), np.ones(x.size, bool)
    if kind == "binary":
        x = np.asarray(ft.num[src], dtype=np.float64)
        obs = np.isfinite(x)
        return (x[obs] > 0.5).astype(np.int64), obs
    if kind in ("nominal", "hier"):
        return _cat_code(ft.cat[src])
    return None


def mi_weights(enc, ft, y: np.ndarray) -> dict[str, float]:
    """Mutual information of every block of `enc` (by name) with y, on the landmarks `ft`."""
    out = {}
    for b in enc.blocks:
        c = block_codes(b, ft)
        out[b.name] = 0.0 if c is None or c[1].sum() == 0 else plugin_mi(c[0], np.asarray(y)[c[1]])
    return out


def weight_vector(enc, by_name: dict[str, float]) -> np.ndarray:
    """Weights in the order of enc's variables (missing names → 0), rescaled to sum to the number of variables."""
    w = np.array([max(float(by_name.get(n, 0.0)), 0.0) for n in enc.feature_names])
    if w.sum() <= 0:
        raise ValueError("every weight is 0")
    return w * (w.size / w.sum())


# --------------------------------------------------------------------------- statistics
def _boot(losses: dict[str, np.ndarray], subjects: np.ndarray, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Bootstrap replicates of every arm's mean loss (multinomial patient weights, shared by the arms)."""
    _, inv = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
    P = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=P).astype(np.float64)
    sums = {a: np.bincount(inv, weights=v, minlength=P) for a, v in losses.items()}
    rng = np.random.default_rng(seed)
    out = {a: [] for a in losses}
    for start in range(0, n_boot, 250):
        W = rng.multinomial(P, np.full(P, 1.0 / P), size=min(250, n_boot - start)).astype(np.float64)
        n = W @ cnt
        for a in losses:
            out[a].append((W @ sums[a]) / n)
    return {a: np.concatenate(v) for a, v in out.items()}


def _ci(b: np.ndarray, level: float = 0.95) -> list[float]:
    b = b[np.isfinite(b)]
    q = 100 * (1 - level) / 2
    return [float(np.percentile(b, q)), float(np.percentile(b, 100 - q))] if b.size else [float("nan")] * 2


def _diff(est: dict, B: dict, a: str, b: str) -> dict:
    d = B[a] - B[b]
    return {"est": float(est[a] - est[b]), "ci95": _ci(d)}


def _share(est: dict, B: dict, num: tuple[str, str], den: tuple[str, str]) -> dict:
    """(L_num0 − L_num1) / (L_den0 − L_den1), or not evaluable when the denominator is not readable."""
    dn, db = est[den[0]] - est[den[1]], B[den[0]] - B[den[1]]
    lo, _ = _ci(db)
    if not (lo > 0 and dn >= MIN_DEN):
        return {"status": "not evaluable", "denominator": {"est": float(dn), "ci95": _ci(db)}}
    nn, nb = est[num[0]] - est[num[1]], B[num[0]] - B[num[1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = nb / db
    return {"est": float(nn / dn), "ci90": _ci(r, 0.90), "ci95": _ci(r)}


def summarise(p: dict[str, np.ndarray], y: np.ndarray, subjects: np.ndarray, seed: int) -> dict:
    from hypo.stats import calibration_line_w, citl_w, logit, logloss_rows
    from sklearn.metrics import roc_auc_score

    losses = {a: logloss_rows(v, y) for a, v in p.items()}
    est = {a: float(v.mean()) for a, v in losses.items()}
    B = _boot(losses, subjects, N_BOOT, seed)
    arms = {}
    for a, v in p.items():
        x = logit(np.clip(v, 1e-6, 1 - 1e-6))
        arms[a] = {"logloss": est[a], "logloss_ci95": _ci(B[a]), "auroc": float(roc_auc_score(y, v)),
                   "citl": citl_w(x, y.astype(np.float64)), "slope": calibration_line_w(x, y.astype(np.float64))[1]}
    has = set(p)
    out = {"n": int(y.size), "patients": int(np.unique(np.asarray(subjects).astype(str)).size), "arms": arms}
    if {"mi_transported", "nca_transported"} <= has:
        d1 = _diff(est, B, "mi_transported", "nca_transported")
        d1["learned_metric_adds"] = bool(d1["ci95"][0] > 0)
        out["delta1"] = d1
        out["Q"] = _share(est, B, ("euclidean", "mi_transported"), ("euclidean", "nca_transported"))
        out["R_mi"] = _share(est, B, ("euclidean", "mi_transported"), ("euclidean", "nca_local"))
        out["R_nca"] = _share(est, B, ("euclidean", "nca_transported"), ("euclidean", "nca_local"))
    if {"mi_local", "nca_transported"} <= has:
        out["delta2"] = _diff(est, B, "mi_local", "nca_transported")
    if {"nca_w_neutral", "nca_transported"} <= has:
        out["delta3"] = _diff(est, B, "nca_w_neutral", "nca_transported")
        out["mi_vs_nca_weights"] = _diff(est, B, "mi_transported", "nca_w_neutral")
    return out


# --------------------------------------------------------------------------- arms
def knn_arm(S, X: str, f: int, enc_w, mcfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """kNN on the Euclidean comparator's geometry (encoder fitted on the fold's case base) with the weights returned
    by `enc_w(enc, tr)`: (test rows, p, info). Same seed as Phase 2's comparators."""
    from hypo.casebase import CaseBase
    from hypo.metric import C, Metric
    from perisim.metric.encoding import Encoder

    tr_idx, te_idx = S.split(X, f)
    lm = S.lm(X)
    tr, te = lm.take(tr_idx), lm.take(te_idx)
    seed = int(S.cfg["seed"]) + 100 * f
    enc = Encoder(geometry="shape").fit(tr.ft, tr.y, C)
    w = enc_w(enc, tr)
    cb = CaseBase(Metric(enc, w), tr, mcfg)
    info = cb.fit_alpha(seed)
    return te_idx, np.asarray(cb.predict(te)["p"], dtype=np.float64), {"alpha": info.get("alpha"), "n_case_base": int(tr_idx.size)}


def run_site(st, X: str, units: Path, gate_only: bool = False) -> dict:
    import polars as pl
    from hypo.external_report import _on, _phase2, _units

    S, mcfg = st.S, st.mcfg
    lm = S.lm(X)
    dev = [c for c in S.cfg["cohorts"] if c != X]
    t0 = time.perf_counter()

    # gate: the Euclidean comparator rebuilt with weights 1 on fold 0
    ref = _phase2(X, "euclidean")
    rows0, p0, _ = knn_arm(S, X, 0, lambda enc, tr: np.ones(enc.n_features), mcfg)
    refp = _on(ref, rows0, "p")
    gap = np.abs(refp - p0)
    if refp is None or not np.all(np.isfinite(gap)) or float(gap.max()) > GATE_TOL:
        raise RuntimeError(f"fixed metric {X}: the rebuilt Euclidean comparator does not reproduce Phase 2 on fold 0 "
                           f"(max |Δp| = {np.nanmax(gap) if gap.size else float('nan')})")
    LOG.info("fixed metric %s: gate passed (max |Δp| = %.2e)", X, float(gap.max()))
    if gate_only:
        return {}

    # mutual information on the development sites (all their landmarks), by variable name
    from perisim.metric.encoding import Encoder
    from hypo.metric import C
    dev_lm = [S.lm(c) for c in dev]
    mi_dev: dict[str, float] = {}
    enc_names = Encoder(geometry="shape").fit(lm.ft, lm.y, C)          # only to enumerate the variables
    pft, py = _pool(dev_lm)             # pooled development landmarks: common bins / categories
    for b in enc_names.blocks:
        c = block_codes(b, pft)
        mi_dev[b.name] = 0.0 if c is None or not c[1].any() else plugin_mi(c[0], py[c[1]])
    del pft, py
    nca = st.metric(X)
    nca_w = dict(zip(nca.enc.feature_names, np.asarray(nca.w, dtype=np.float64)))

    units.mkdir(parents=True, exist_ok=True)
    folds = sorted(set(S.fold(X).tolist()))
    info = {"mi_dev": mi_dev, "folds": {}}
    for f in folds:
        for arm, fn in (("mi_transported", lambda enc, tr: weight_vector(enc, mi_dev)),
                        ("mi_local", lambda enc, tr: weight_vector(enc, mi_weights(enc, tr.ft, tr.y))),
                        ("nca_w_neutral", lambda enc, tr: weight_vector(enc, nca_w))):
            path = units / f"{arm}_{X}_f{f}.parquet"
            if path.exists():
                continue
            r, p, inf = knn_arm(S, X, f, fn, mcfg)
            pl.DataFrame({"row": r, "p": p}).write_parquet(path)
            info["folds"][f"{arm}_f{f}"] = inf
            LOG.info("fixed metric %s fold %d %s: α %s", X, f, arm, inf["alpha"])

    T = _units(f"transported_{X}_f*.parquet")
    rows = T["row"].to_numpy()
    p = {"nca_transported": T["p"].to_numpy(), "nca_local": _on(_units(f"local_{X}_f*.parquet"), rows, "p"),
         "euclidean": _on(ref, rows, "p")}
    for arm in ("mi_transported", "mi_local", "nca_w_neutral"):
        fs = sorted(units.glob(f"{arm}_{X}_f*.parquet"))
        p[arm] = _on(pl.concat([pl.read_parquet(x) for x in fs]).sort("row"), rows, "p")
    ok = np.all([np.isfinite(v) for v in p.values()], axis=0)
    if ok.sum() != rows.size:
        raise RuntimeError(f"fixed metric {X}: {rows.size - ok.sum()} rows without a prediction in some arm")
    y = lm.y[rows]
    subj = lm.ft.subjects[rows]
    res = {"all": summarise(p, y, subj, SEED)}
    vaso = lm.ft.num["vaso_on"][rows]
    for v in (0, 1):
        m = vaso == v
        if 0 < m.sum() < m.size:
            r = summarise({a: q[m] for a, q in p.items()}, y[m], subj[m], SEED + 1 + v)
            r["share_of_landmarks"] = float(m.mean())
            res[f"vaso_on = {v}"] = r
    res["weights_top"] = {"mi_dev": _top(mi_dev), "nca": _top(nca_w)}
    res["seconds"] = time.perf_counter() - t0
    return res


def _pool(dev_lm):
    """The development sites' landmarks as one feature table, and their outcomes."""
    from perisim.features.table import FeatureTable
    ft = FeatureTable(np.concatenate([d.ft.ids for d in dev_lm]), np.concatenate([d.ft.subjects for d in dev_lm]),
                      dev_lm[0].ft.specs,
                      {k: np.concatenate([d.ft.num[k] for d in dev_lm]) for k in dev_lm[0].ft.num},
                      {k: np.concatenate([np.asarray(d.ft.cat[k], dtype=object) for d in dev_lm])
                       for k in dev_lm[0].ft.cat})
    return ft, np.concatenate([d.y for d in dev_lm])


def _top(w: dict[str, float], n: int = 10) -> list[list]:
    tot = sum(max(v, 0.0) for v in w.values()) or 1.0
    return [[k, round(max(v, 0.0) / tot, 4)] for k, v in sorted(w.items(), key=lambda kv: -kv[1])[:n]]


# --------------------------------------------------------------------------- report
def render(sites: dict) -> str:
    L = ["# A fixed mutual-information metric against the transported learned metric (`hypo`, exploratory)", "",
         "Plan fixed before running: docs/DECISIONS.md, 2026-11-06. Same case base, k, neighbours and vote in every "
         "arm; only the metric changes. Δ1 = log-loss(MI-transported) − log-loss(NCA-transported): the learned metric "
         "adds at a site when the lower limit of its 95 % CI is > 0. Q = share of the transported NCA metric's gain "
         f"over the Euclidean kNN already obtained by mutual-information weights. {N_BOOT:,} patient resamples.", "",
         "| site | landmarks | Euclidean | NCA-transported | NCA-local | MI-transported | MI-local | NCA weights, neutral |",
         "|---|---|---|---|---|---|---|---|"]
    for X, r in sites.items():
        a = r["all"]["arms"]
        L.append(f"| {X} | {r['all']['n']:,} | " + " | ".join(f"{a[k]['logloss']:.4f}" for k in ARMS) + " |")
    L += ["", "| site | stratum | Δ1 (95 % CI) | learned adds | Q (90 % CI) | R NCA | R MI | Δ2 MI-local − NCA | Δ3 shapes |",
          "|---|---|---|---|---|---|---|---|---|"]
    for X, r in sites.items():
        for s, u in r.items():
            if not isinstance(u, dict) or "delta1" not in u:
                continue
            d1, q = u["delta1"], u["Q"]
            qs = f"{q['est']:.2f} ({q['ci90'][0]:.2f}–{q['ci90'][1]:.2f})" if "est" in q else "not evaluable"
            rn = f"{u['R_nca']['est']:.2f}" if "est" in u["R_nca"] else "n.e."
            rm = f"{u['R_mi']['est']:.2f}" if "est" in u["R_mi"] else "n.e."
            L.append(f"| {X} | {s} | {d1['est']:+.4f} ({d1['ci95'][0]:+.4f} to {d1['ci95'][1]:+.4f}) | "
                     f"{'yes' if d1['learned_metric_adds'] else 'no'} | {qs} | {rn} | {rm} | "
                     f"{u['delta2']['est']:+.4f} | {u['delta3']['est']:+.4f} |")
    L += ["", "| site | arm | AUROC | CITL | slope |", "|---|---|---|---|---|"]
    for X, r in sites.items():
        for k in ARMS:
            a = r["all"]["arms"][k]
            L.append(f"| {X} | {k} | {a['auroc']:.4f} | {a['citl']:+.3f} | {a['slope']:.3f} |")
    verdicts = [r["all"]["delta1"]["learned_metric_adds"] for r in sites.values()]
    rule = ("supported (all sites)" if all(verdicts) else "not supported (no site)" if not any(verdicts)
            else "mixed: " + ", ".join(f"{X} {'yes' if v else 'no'}" for X, v in zip(sites, verdicts)))
    L += ["", f"**Rule (pre-specified):** the learned metric adds over a fixed mutual-information metric — {rule}.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="*", default=["inspire", "mimic", "eicu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--gate-only", action="store_true", help="only rebuild the Euclidean comparator on fold 0")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.external import Stage
    from hypo.frozen import check, external_dir

    rec = check()
    st = Stage(threads=a.threads)
    od = external_dir().with_name("posthoc_fixed_metric")
    res = {"freeze": rec.get("tag"), "sites": {}}
    for X in a.sites:
        res["sites"][X] = run_site(st, X, od / "units", gate_only=a.gate_only)
    check()
    if a.gate_only:
        LOG.info("fixed metric: gate passed for %s", ", ".join(a.sites))
        return 0
    (od / "fixed_metric.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    rep = external_dir().parent / "reports" / "posthoc_fixed_metric.md"
    rep.write_text(render(res["sites"]), encoding="utf-8")
    LOG.info("fixed metric: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
