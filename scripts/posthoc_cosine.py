"""Post-hoc, PRE-SPECIFIED: a cosine kNN against the transported learned metric (`hypo` targets and NWICU).

Plan, arms, gate and rule: docs/DECISIONS.md, 2026-11-08 ("a cosine kNN against the transported learned metric").
Run with `hypo`'s frozen code (PERISIM_SRC=frozen/hypo/src, or a v0.5 checkout); the freeze record is checked at the
start and the end, and nothing frozen is refitted.

Every arm uses the target's own out-of-fold case base, k = 50, distinct patients and the same vote; only the metric
changes:
- COS-neutral (primary comparator, the analogue of PRAM's cosine): cosine on the Euclidean comparator's neutral axes,
  centred on the fold's case base;
- COS-NCA: cosine on the transported NCA metric's weighted embedding, centred on the fold's case base;
- stored: NCA-transported, NCA-local, Euclidean and (hypo targets) MI-transported (§17).
The cosine is computed exactly as a Euclidean kNN on centred, row-normalised vectors (‖u − v‖² = 2 − 2 cos).

Usage:  python scripts/posthoc_cosine.py [--sites inspire mimic eicu nwicu] [--threads 8] [--gate-only]
Outputs (aggregates only): work/experiments/hypo/posthoc_cosine/cosine.json, reports/posthoc_cosine.md;
per-fold predictions in posthoc_cosine/units/ (row-level, git-ignored work folder, never published).
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
sys.path.insert(0, str(ROOT / "scripts"))
try:                                    # v0.6 repository: $PERISIM_SRC (frozen code) first on sys.path
    import _frozen  # noqa: E402
    _frozen.use_src()
except ImportError:                     # a v0.5 checkout: its own src is the frozen code
    sys.path.insert(0, str(ROOT / "src"))

import posthoc_fixed_metric as FM  # noqa: E402

LOG = logging.getLogger("posthoc.cosine")
N_BOOT = 2000
SEED = 20261108
GATE_TOL = 1e-9
HYPO_SITES = ("inspire", "mimic", "eicu")
ROTATIONS = ("inspire", "mimic", "eicu")


# --------------------------------------------------------------------------- the cosine metric
class CosineMetric:
    """Cosine distance on a base metric's weighted embedding, centred on a reference set: embed() returns the
    centred, row-normalised vectors and the weights are 1, so the case base's Euclidean search is a cosine search."""

    def __init__(self, base, ref, E_ref: np.ndarray | None = None):
        self.base = base
        self.sw = np.asarray(base.sqrt_w_dims, dtype=np.float64)
        E = base.embed(ref) if E_ref is None else E_ref
        self.mu = (np.asarray(E, dtype=np.float64) * self.sw).mean(axis=0)

    def unit(self, E: np.ndarray) -> np.ndarray:
        X = np.asarray(E, dtype=np.float64) * self.sw - self.mu
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return (X / np.where(n > 1e-12, n, 1.0)).astype(np.float32)   # a row at the centre: zero vector

    def embed(self, lm) -> np.ndarray:
        return self.unit(self.base.embed(lm))

    @property
    def sqrt_w_dims(self) -> np.ndarray:
        return np.ones(self.mu.size)


def neutral_metric(tr):
    """The Euclidean comparator's metric on a case base: neutral axes (encoder fitted on it), weights 1."""
    from hypo.metric import C, Metric
    from perisim.metric.encoding import Encoder

    enc = Encoder(geometry="shape").fit(tr.ft, tr.y, C)
    return Metric(enc, np.ones(enc.n_features))


def knn_predict(metric, tr, te, mcfg: dict, seed: int) -> tuple[np.ndarray, dict]:
    from hypo.casebase import CaseBase

    cb = CaseBase(metric, tr, mcfg)
    info = cb.fit_alpha(seed)
    return np.asarray(cb.predict(te)["p"], dtype=np.float64), {"alpha": info.get("alpha")}


def _write(path: Path, rows: np.ndarray, p: np.ndarray) -> None:
    import polars as pl
    pl.DataFrame({"row": rows, "p": p}).write_parquet(path)


def _gather(units: Path, arm: str, rows: np.ndarray) -> np.ndarray:
    import polars as pl
    from hypo.external_report import _on

    fs = sorted(units.glob(f"{arm}_f*.parquet"))
    if not fs:
        raise RuntimeError(f"cosine: no units for {arm}")
    return _on(pl.concat([pl.read_parquet(f) for f in fs]).sort("row"), rows, "p")


def _gate(p_rebuilt: np.ndarray, p_stored: np.ndarray | None, what: str) -> None:
    gap = np.abs(np.asarray(p_stored, dtype=np.float64) - p_rebuilt) if p_stored is not None else np.array([np.nan])
    if p_stored is None or not np.all(np.isfinite(gap)) or float(gap.max()) > GATE_TOL:
        raise RuntimeError(f"cosine {what}: the rebuilt Euclidean comparator does not reproduce the stored one on "
                           f"fold 0 (max |Δp| = {np.nanmax(gap) if gap.size else float('nan')})")
    LOG.info("cosine %s: gate passed (max |Δp| = %.2e)", what, float(gap.max()))


# --------------------------------------------------------------------------- statistics
def wauc(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted AUROC, ties counted one half."""
    u, g = np.unique(score, return_inverse=True)
    pos = np.bincount(g, weights=w * y, minlength=u.size)
    neg = np.bincount(g, weights=w * (1 - y), minlength=u.size)
    below = np.cumsum(neg) - neg
    den = pos.sum() * neg.sum()
    return float((pos * (below + 0.5 * neg)).sum() / den) if den > 0 else float("nan")


class _AUC:
    """wauc with the score grouping precomputed (one per arm), for the bootstrap."""

    def __init__(self, score: np.ndarray, y: np.ndarray):
        self.u, self.g = np.unique(score, return_inverse=True)
        self.y = y.astype(np.float64)

    def __call__(self, w: np.ndarray) -> float:
        pos = np.bincount(self.g, weights=w * self.y, minlength=self.u.size)
        neg = np.bincount(self.g, weights=w * (1 - self.y), minlength=self.u.size)
        den = pos.sum() * neg.sum()
        return float((pos * (np.cumsum(neg) - 0.5 * neg)).sum() / den) if den > 0 else float("nan")


def boot(p: dict[str, np.ndarray], y: np.ndarray, subjects: np.ndarray, n_boot: int, seed: int) -> dict:
    """Bootstrap replicates of every arm's mean log-loss and AUROC (multinomial patient weights shared by the arms)."""
    from hypo.stats import logloss_rows

    _, inv = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
    P = int(inv.max()) + 1
    loss = {a: logloss_rows(v, y) for a, v in p.items()}
    auc = {a: _AUC(v, y) for a, v in p.items()}
    rng = np.random.default_rng(seed)
    L = {a: np.empty(n_boot) for a in p}
    A = {a: np.empty(n_boot) for a in p}
    for b in range(n_boot):
        w = rng.multinomial(P, np.full(P, 1.0 / P)).astype(np.float64)[inv]
        sw = w.sum()
        for a in p:
            L[a][b] = float(w @ loss[a]) / sw
            A[a][b] = auc[a](w)
    est_l = {a: float(v.mean()) for a, v in loss.items()}
    est_a = {a: wauc(v, y.astype(np.float64), np.ones(y.size)) for a, v in p.items()}
    return {"L": L, "A": A, "est_l": est_l, "est_a": est_a}


def verdict(ci: list[float], pos: str, neg: str) -> str:
    return pos if ci[0] > 0 else neg if ci[1] < 0 else "no difference shown"


def summarise(p: dict[str, np.ndarray], y: np.ndarray, subjects: np.ndarray, nca_arms: list[str], seed: int,
              n_boot: int | None = None) -> dict:
    from hypo.stats import calibration_line_w, citl_w, logit

    B = boot(p, y, subjects, N_BOOT if n_boot is None else n_boot, seed)
    yf = y.astype(np.float64)
    arms = {}
    for a, v in p.items():
        x = logit(np.clip(v, 1e-6, 1 - 1e-6))
        arms[a] = {"logloss": B["est_l"][a], "logloss_ci95": FM._ci(B["L"][a]), "auroc": B["est_a"][a],
                   "auroc_ci95": FM._ci(B["A"][a]), "citl": citl_w(x, yf), "slope": calibration_line_w(x, yf)[1]}
    out = {"n": int(y.size), "patients": int(np.unique(np.asarray(subjects).astype(str)).size), "arms": arms,
           "primary": {}, "secondary": {}}
    for nca in nca_arms:
        dl = B["L"]["cos_neutral"] - B["L"][nca]
        da = B["A"][nca] - B["A"]["cos_neutral"]
        d = {"delta_c": {"est": B["est_l"]["cos_neutral"] - B["est_l"][nca], "ci95": FM._ci(dl)},
             "delta_auroc": {"est": B["est_a"][nca] - B["est_a"]["cos_neutral"], "ci95": FM._ci(da)}}
        d["verdict"] = verdict(d["delta_c"]["ci95"], "NCA beats cosine", "cosine beats NCA")
        d["verdict_auroc"] = verdict(d["delta_auroc"]["ci95"], "NCA beats cosine", "cosine beats NCA")
        out["primary"][nca] = d
    sec = {"cos_neutral − euclidean": ("cos_neutral", "euclidean")}
    for a in p:
        if a.startswith("cos_nca"):
            sec[f"{a} − {a.replace('cos_nca', 'nca_transported')}"] = (a, a.replace("cos_nca", "nca_transported"))
    if "mi_transported" in p:
        sec["cos_neutral − mi_transported"] = ("cos_neutral", "mi_transported")
    for k, (a, b) in sec.items():
        if a in p and b in p:
            out["secondary"][k] = {"logloss": {"est": B["est_l"][a] - B["est_l"][b], "ci95": FM._ci(B["L"][a] - B["L"][b])},
                                   "auroc": {"est": B["est_a"][a] - B["est_a"][b], "ci95": FM._ci(B["A"][a] - B["A"][b])}}
    return out


# --------------------------------------------------------------------------- hypo targets
def run_hypo(st, X: str, units: Path, gate_only: bool = False) -> dict:
    from hypo.external_report import _on, _phase2, _units

    S, mcfg = st.S, st.mcfg
    lm = S.lm(X)
    t0 = time.perf_counter()
    tr0, te0 = S.split(X, 0)
    seed0 = int(S.cfg["seed"])
    p0, _ = knn_predict(neutral_metric(lm.take(tr0)), lm.take(tr0), lm.take(te0), mcfg, seed0)
    ref = _phase2(X, "euclidean")
    _gate(p0, _on(ref, te0, "p"), X)
    if gate_only:
        return {}
    units.mkdir(parents=True, exist_ok=True)
    nca = st.metric(X)
    alphas = {}
    for f in sorted(set(S.fold(X).tolist())):
        tr_idx, te_idx = S.split(X, f)
        tr, te = lm.take(tr_idx), lm.take(te_idx)
        seed = seed0 + 100 * f
        for arm, base in (("cos_neutral", lambda: neutral_metric(tr)), ("cos_nca", lambda: nca)):
            path = units / f"{arm}_{X}_f{f}.parquet"
            if path.exists():
                continue
            p, inf = knn_predict(CosineMetric(base(), tr), tr, te, mcfg, seed)
            _write(path, te_idx, p)
            alphas[f"{arm}_f{f}"] = inf["alpha"]
            LOG.info("cosine %s fold %d %s: α %s", X, f, arm, inf["alpha"])
    T = _units(f"transported_{X}_f*.parquet")
    rows = T["row"].to_numpy()
    p = {"nca_transported": T["p"].to_numpy(), "nca_local": _on(_units(f"local_{X}_f*.parquet"), rows, "p"),
         "euclidean": _on(ref, rows, "p")}
    mi = sorted((st_dir(st) / "posthoc_fixed_metric" / "units").glob(f"mi_transported_{X}_f*.parquet"))
    if mi:
        import polars as pl
        p["mi_transported"] = _on(pl.concat([pl.read_parquet(x) for x in mi]).sort("row"), rows, "p")
    for arm in ("cos_neutral", "cos_nca"):
        p[arm] = _gather(units, f"{arm}_{X}", rows)
    _complete(p, X)
    res = summarise(p, lm.y[rows], lm.ft.subjects[rows], ["nca_transported"], SEED)
    res["alpha"], res["seconds"] = alphas, time.perf_counter() - t0
    return res


def st_dir(st) -> Path:
    from hypo.frozen import external_dir
    return external_dir().parent


def _complete(p: dict, X: str) -> None:
    for a, v in p.items():
        if v is None or not np.all(np.isfinite(v)):
            raise RuntimeError(f"cosine {X}: arm {a} has rows without a prediction")


# --------------------------------------------------------------------------- NWICU
def run_nwicu(units: Path, gate_only: bool = False, quick: bool = False, tag: str = "nwicu") -> dict:
    import nwicu_transport as NT
    from hypo.cv import outer_folds, settings
    from hypo.frozen import final_dir, load_pickle

    NT.TAG = tag
    cfg, mcfg, _ = settings(quick)
    lm = NT.landmarks(cfg)
    fold = outer_folds(lm, int(cfg["cv"]["outer_folds"]), int(cfg["seed"]))
    seed0 = int(cfg["seed"])
    nu = NT.work() / "units"
    t0 = time.perf_counter()
    tr0, te0 = np.where(fold != 0)[0], np.where(fold == 0)[0]
    p0, _ = knn_predict(neutral_metric(lm.take(tr0)), lm.take(tr0), lm.take(te0), mcfg, seed0)
    r_, v_ = NT.read(nu / "euclidean_f0.parquet")
    stored = np.full(te0.size, np.nan)
    pos = np.searchsorted(te0, r_)
    ok = (pos < te0.size) & (te0[np.minimum(pos, te0.size - 1)] == r_)
    stored[pos[ok]] = v_[ok]
    _gate(p0, stored, "nwicu")
    if gate_only:
        return {}
    units.mkdir(parents=True, exist_ok=True)
    metrics = {e: load_pickle(final_dir() / f"metric_ext-{e}.pkl") for e in ROTATIONS}
    E = {e: m.embed(lm) for e, m in metrics.items()}
    alphas = {}
    for f in sorted(set(fold.tolist())):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        ref, q = lm.take(tr), lm.take(te)
        seed = seed0 + 100 * f
        jobs = [("cos_neutral_nwicu", lambda: (CosineMetric(neutral_metric(ref), ref), None, None))]
        for e in ROTATIONS:
            jobs.append((f"cos_nca_{e}_nwicu", lambda e=e: (CosineMetric(metrics[e], ref, E_ref=E[e][tr]), e, None)))
        for arm, make in jobs:
            path = units / f"{arm}_f{f}.parquet"
            if path.exists():
                continue
            cm, e, _ = make()
            from hypo.casebase import CaseBase
            Eu = cm.unit(E[e]) if e else cm.embed(lm)
            cb = CaseBase(cm, ref, mcfg, E=Eu[tr])
            inf = cb.fit_alpha(seed)
            _write(path, te, np.asarray(cb.predict(q, E=Eu[te])["p"], dtype=np.float64))
            alphas[f"{arm}_f{f}"] = inf.get("alpha")
            LOG.info("cosine nwicu fold %d %s: α %s", f, arm, inf.get("alpha"))
    rows = np.arange(len(lm))
    P = NT.gather(nu, len(lm))
    p = {"euclidean": P["euclidean"], "nca_local": P["local"]}
    for e in ROTATIONS:
        p[f"nca_transported_{e}"] = P[f"knn_{e}"]
        p[f"cos_nca_{e}"] = _gather(units, f"cos_nca_{e}_nwicu", rows)
    p["cos_neutral"] = _gather(units, "cos_neutral_nwicu", rows)
    _complete(p, "nwicu")
    res = summarise(p, lm.y, lm.ft.subjects, [f"nca_transported_{e}" for e in ROTATIONS], SEED + 1)
    res["alpha"], res["seconds"] = alphas, time.perf_counter() - t0
    return res


# --------------------------------------------------------------------------- report
LABEL = {"nca_transported": "NCA, transported", "nca_local": "NCA, local", "euclidean": "Euclidean",
         "mi_transported": "MI-weighted, transported", "cos_neutral": "cosine (neutral axes)",
         "cos_nca": "cosine on the NCA embedding"}


def _label(a: str) -> str:
    for e in ROTATIONS:
        if a.endswith(f"_{e}"):
            other = "+".join(c for c in ROTATIONS if c != e)
            return f"{LABEL[a[:-len(e) - 1]]} ({other})"
    return LABEL.get(a, a)


def _ci(d: dict, f: str = "+.4f") -> str:
    return f"{d['est']:{f}} ({d['ci95'][0]:{f}} to {d['ci95'][1]:{f}})"


def render(sites: dict) -> str:
    L = ["# A cosine kNN against the transported learned metric (post hoc, pre-specified)", "",
         "Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-08. Same case base, k, neighbours and vote "
         "in every arm; only the metric changes. Δc = log-loss(cosine) − log-loss(NCA, transported): NCA beats cosine "
         "when its 95 % CI is above 0, cosine beats NCA when below. ΔAUROC = AUROC(NCA, transported) − "
         f"AUROC(cosine). {N_BOOT:,} patient resamples, shared by the arms of a target.", "",
         "## Primary comparison", "", "| target | NCA metric | Δc log-loss (95 % CI) | verdict | ΔAUROC (95 % CI) | verdict |",
         "|---|---|---|---|---|---|"]
    for X, r in sites.items():
        for nca, d in r["primary"].items():
            L.append(f"| {X} | {_label(nca)} | {_ci(d['delta_c'])} | {d['verdict']} | {_ci(d['delta_auroc'])} | "
                     f"{d['verdict_auroc']} |")
    L += ["", "## Every arm", "", "| target | arm | log-loss | AUROC | CITL | slope |", "|---|---|---|---|---|---|"]
    for X, r in sites.items():
        for a, v in r["arms"].items():
            L.append(f"| {X} | {_label(a)} | {v['logloss']:.4f} | {v['auroc']:.4f} | {v['citl']:+.3f} | {v['slope']:.3f} |")
    L += ["", "## Secondary contrasts (descriptive)", "", "| target | contrast | Δ log-loss (95 % CI) | Δ AUROC (95 % CI) |",
          "|---|---|---|---|"]
    for X, r in sites.items():
        for k, d in r["secondary"].items():
            L.append(f"| {X} | {k} | {_ci(d['logloss'])} | {_ci(d['auroc'])} |")
    v = [d["verdict"] for r in sites.values() for d in r["primary"].values()]
    L += ["", f"**Rule (pre-specified), log-loss:** NCA beats cosine in {v.count('NCA beats cosine')} of {len(v)} "
          f"comparisons; cosine beats NCA in {v.count('cosine beats NCA')}; no difference shown in "
          f"{v.count('no difference shown')}.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="*", default=list(HYPO_SITES) + ["nwicu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="quick frozen artefacts (tests)")
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.external import Stage
    from hypo.frozen import check, external_dir

    rec = check()
    od = external_dir().with_name("posthoc_cosine")
    res = {"freeze": rec.get("tag"), "sites": {}}
    st = Stage(quick=a.quick, threads=a.threads) if any(X in HYPO_SITES for X in a.sites) else None
    for X in a.sites:
        if X == "nwicu":
            res["sites"][X] = run_nwicu(od / "units", a.gate_only, a.quick)
        else:
            res["sites"][X] = run_hypo(st, X, od / "units", a.gate_only)
    check()
    if a.gate_only:
        LOG.info("cosine: gate passed for %s", ", ".join(a.sites))
        return 0
    (od / "cosine.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    rep = external_dir().parent / "reports" / "posthoc_cosine.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(render(res["sites"]), encoding="utf-8")
    LOG.info("cosine: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
