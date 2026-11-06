"""Post-hoc, EXPLORATORY (`hypo`): two repairs of the vasopressor failure of the metric transported to MIMIC-IV.

Plan, arms, gate and rule: docs/DECISIONS.md, 2026-11-06 ("two repairs of the vasopressor failure"). Run with `hypo`'s
frozen code (PERISIM_SRC=frozen/hypo/src); the freeze record is checked at the start and the end; frozen artefacts are
only read, and the metrics learned here are new artefacts of this analysis.

Every arm keeps the transported metric's encoder and shapes (the geometry) and its development pair's learning sample;
only the weights or the neighbourhood change:
- NCA-refit: the NCA refitted without prior (the reference arm);
- A-sel / A-strong: the NCA with a mutual-information prior, λ chosen inside the development pair / λ = 1;
- B0: the frozen metric with neighbours of the query's vasopressor state only;
- B: weights learned separately for vasopressor off / on, each with the same-state case base.

Usage:  python scripts/posthoc_vaso_repair.py [--sites inspire mimic eicu] [--threads 16] [--quick]
Outputs (aggregates only): work/experiments/hypo/posthoc_vaso_repair/vaso_repair.json and
reports/posthoc_vaso_repair.md; per-fold predictions and learned weights in posthoc_vaso_repair/units/ (work folder).
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

import posthoc_fixed_metric as FM  # noqa: E402  (mutual information, bootstrap helpers)

LOG = logging.getLogger("posthoc.vaso_repair")
SEED = 20261107
LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0]
LAM_STRONG = 1.0
VAL_SHARE = 0.20
MIN_STATE_ROWS = 1000
SAFE = 0.001          # nats: the largest tolerated overall loss against NCA-refit (upper 95 % limit)
GATE_TOL = 1e-9
REPAIRS = ["a_sel", "a_strong", "b0_stratified", "b_conditional"]


def vaso_state(lm) -> np.ndarray:
    v = np.asarray(lm.ft.num["vaso_on"], dtype=np.float64)
    return np.where(np.isfinite(v), v > 0.5, False)


# --------------------------------------------------------------------------- learning
def learn_weights(enc, sample, mcfg: dict, seed: int, threads: int, prior: np.ndarray | None = None,
                  lam: float = 0.0) -> np.ndarray:
    """Diagonal NCA weights on `sample` in the geometry of `enc` (the frozen settings), optionally with a prior."""
    import torch
    from hypo.metric import C, SiteNCA

    torch.set_num_threads(max(1, int(threads)))
    ncfg = mcfg["nca"]
    E = enc.transform(sample.ft)
    nca = SiteNCA(enc.dim_feature, enc.n_features, C, n_candidates=int(ncfg["n_candidates"]),
                  batch_size=int(ncfg["batch_size"]), lr=float(ncfg["lr"]), refresh_every=int(ncfg["refresh_every"]),
                  eps0=float(ncfg["eps0"]), label_smoothing=float(ncfg["label_smoothing"]), seed=seed,
                  sites=sample.site)
    nca.fit(E, sample.y, prior=prior, lam=float(lam), epochs=int(ncfg["epochs"]), subjects=sample.ft.subjects)
    return np.asarray(nca.weights, dtype=np.float64)


def mi_prior(enc, sample) -> np.ndarray:
    """Each variable's share of the mutual information with the outcome on the learning sample (enc's order)."""
    mi = FM.mi_weights(enc, sample.ft, sample.y)
    v = np.array([max(mi.get(n, 0.0), 0.0) for n in enc.feature_names])
    return v / v.sum() if v.sum() > 0 else np.full(v.size, 1.0 / v.size)


def kNN_loss(enc, w, train, val, mcfg: dict, seed: int) -> float:
    from hypo.casebase import CaseBase
    from hypo.metric import Metric
    from hypo.stats import logloss_rows

    cb = CaseBase(Metric(enc, w), train, mcfg)
    cb.fit_alpha(seed)
    return float(logloss_rows(np.asarray(cb.predict(val)["p"], dtype=np.float64), val.y).mean())


def select_lambda(enc, sample, prior, mcfg: dict, seed: int, threads: int) -> tuple[float, dict]:
    """λ with the lowest validation log-loss on a patient split of the learning sample (ties → smaller λ)."""
    subj = np.asarray(sample.ft.subjects).astype(str)
    uniq = np.unique(subj)
    rng = np.random.default_rng(seed)
    val_p = set(rng.choice(uniq, size=max(1, int(round(VAL_SHARE * uniq.size))), replace=False).tolist())
    is_val = np.array([s in val_p for s in subj])
    tr, va = sample.take(np.where(~is_val)[0]), sample.take(np.where(is_val)[0])
    scores = {}
    for lam in LAMBDAS:
        w = learn_weights(enc, tr, mcfg, seed, threads, prior=prior, lam=lam)
        scores[lam] = kNN_loss(enc, w, tr, va, mcfg, seed)
        LOG.info("vaso repair: λ %g → validation log-loss %.5f", lam, scores[lam])
    best = min(LAMBDAS, key=lambda l: (round(scores[l], 6), l))
    return best, {str(k): v for k, v in scores.items()}


# --------------------------------------------------------------------------- arms on the target site
def predict_fold(st, X: str, f: int, enc, w_by_state: dict, stratified: bool) -> tuple[np.ndarray, np.ndarray]:
    """(test rows, p) of the site's fold f: case base = the fold's training landmarks; with `stratified`, each
    vasopressor state queries the same-state part of the case base with its own weights."""
    from hypo.casebase import CaseBase
    from hypo.metric import Metric

    lm = st.S.lm(X)
    tr, te = st.S.split(X, f)
    seed = st.seed + 100 * f
    E = st.embed(X, X)                       # the frozen encoder's embedding of the site (cached by the stage)
    if not stratified:
        cb = CaseBase(Metric(enc, w_by_state[None]), lm.take(tr), st.mcfg, E=E[tr])
        cb.fit_alpha(seed)
        return te, np.asarray(cb.predict(lm.take(te), E=E[te])["p"], dtype=np.float64)
    state = vaso_state(lm)
    p = np.full(te.size, np.nan)
    for s in (False, True):
        q = te[state[te] == s]
        if q.size == 0:
            continue
        base = tr[state[tr] == s]
        if base.size < int(st.mcfg.get("k", 50)) + 1:      # too few same-state patients: whole case base
            base = tr
        cb = CaseBase(Metric(enc, w_by_state[s]), lm.take(base), st.mcfg, E=E[base])
        cb.fit_alpha(seed)
        p[np.isin(te, q)] = np.asarray(cb.predict(lm.take(q), E=E[q])["p"], dtype=np.float64)
    return te, p


def run_site(st, X: str, units: Path, threads: int) -> dict:
    import polars as pl
    from hypo.data import concat, learning_sample
    from hypo.external_report import _on, _phase2, _units

    t0 = time.perf_counter()
    S, mcfg = st.S, st.mcfg
    metric = st.metric(X)
    enc, w_frozen = metric.enc, np.asarray(metric.w, dtype=np.float64)
    names = list(enc.feature_names)
    vaso_names = [n for n in names if n in ("vaso_on", "vaso_24h", "vaso_hours")]

    # gate: the stored transported arm rebuilt from the frozen metric (fold 0)
    T = _units(f"transported_{X}_f*.parquet")
    r0, p0 = predict_fold(st, X, 0, enc, {None: w_frozen}, stratified=False)
    ref = _on(T, r0, "p")
    gap = np.abs(ref - p0)
    if not np.all(np.isfinite(gap)) or float(gap.max()) > GATE_TOL:
        raise RuntimeError(f"vaso repair {X}: the rebuilt transported arm does not reproduce the stored predictions "
                           f"(max |Δp| = {np.nanmax(gap) if gap.size else float('nan')})")
    LOG.info("vaso repair %s: gate passed (max |Δp| = %.2e)", X, float(gap.max()))

    # the development pair's learning sample, reproduced as the frozen artefacts drew it
    dev = [c for c in S.cfg["cohorts"] if c != X]
    fseed = int(S.cfg["seed"]) + int(st.x["final_seed_offset"])
    pool = concat([S.lm(c) for c in dev])
    rows = learning_sample(pool, int(mcfg["learn"]["per_patient"]), int(mcfg["learn"]["per_site"]), fseed)
    sample = pool.take(rows)
    del pool
    units.mkdir(parents=True, exist_ok=True)
    wfile = units / f"weights_{X}.json"
    W = json.loads(wfile.read_text(encoding="utf-8")) if wfile.exists() else {}

    def cached(key, fn):
        if key not in W:
            W[key] = [float(x) for x in fn()]
            wfile.write_text(json.dumps(W), encoding="utf-8")
        return np.asarray(W[key], dtype=np.float64)

    w_refit = cached("nca_refit", lambda: learn_weights(enc, sample, mcfg, fseed, threads))
    prior = mi_prior(enc, sample)
    if "lambda" not in W:
        lam, scores = select_lambda(enc, sample, prior, mcfg, fseed, threads)
        W["lambda"], W["lambda_scores"] = lam, scores
        wfile.write_text(json.dumps(W), encoding="utf-8")
    lam = float(W["lambda"])
    w_sel = cached("a_sel", lambda: learn_weights(enc, sample, mcfg, fseed, threads, prior=prior, lam=lam))
    w_strong = cached("a_strong", lambda: learn_weights(enc, sample, mcfg, fseed, threads, prior=prior,
                                                        lam=LAM_STRONG))
    sstate = vaso_state(sample)
    w_state = {}
    for s in (False, True):
        idx = np.where(sstate == s)[0]
        if idx.size >= MIN_STATE_ROWS and np.unique(sample.y[idx]).size == 2:
            w_state[s] = cached(f"b_state_{int(s)}", lambda idx=idx: learn_weights(enc, sample.take(idx), mcfg, fseed,
                                                                                    threads))
        else:
            w_state[s] = w_refit
    del sample

    arms = {"nca_refit": ({None: w_refit}, False), "a_sel": ({None: w_sel}, False),
            "a_strong": ({None: w_strong}, False), "b0_stratified": ({False: w_frozen, True: w_frozen}, True),
            "b_conditional": (w_state, True)}
    folds = sorted(set(S.fold(X).tolist()))
    for f in folds:
        for arm, (wd, strat) in arms.items():
            path = units / f"{arm}_{X}_f{f}.parquet"
            if path.exists():
                continue
            r, p = predict_fold(st, X, f, enc, wd, strat)
            pl.DataFrame({"row": r, "p": p}).write_parquet(path)
            LOG.info("vaso repair %s fold %d %s done", X, f, arm)

    rows_t = T["row"].to_numpy()
    p = {"nca_transported": T["p"].to_numpy(),
         "nca_local": _on(_units(f"local_{X}_f*.parquet"), rows_t, "p"),
         "euclidean": _on(_phase2(X, "euclidean"), rows_t, "p")}
    mi_dir = units.parent.with_name("posthoc_fixed_metric") / "units"
    mi_files = sorted(mi_dir.glob(f"mi_transported_{X}_f*.parquet"))
    if mi_files:
        p["mi_transported"] = _on(pl.concat([pl.read_parquet(x) for x in mi_files]).sort("row"), rows_t, "p")
    for arm in arms:
        fs = sorted(units.glob(f"{arm}_{X}_f*.parquet"))
        p[arm] = _on(pl.concat([pl.read_parquet(x) for x in fs]).sort("row"), rows_t, "p")
    ok = np.all([np.isfinite(v) for v in p.values()], axis=0)
    if ok.sum() != rows_t.size:
        raise RuntimeError(f"vaso repair {X}: {rows_t.size - ok.sum()} rows without a prediction in some arm")
    lm = S.lm(X)
    y, subj, state = lm.y[rows_t], lm.ft.subjects[rows_t], vaso_state(lm)[rows_t]
    res = {"all": summarise(p, y, subj, SEED)}
    for s in (False, True):
        m = state == s
        if 0 < m.sum() < m.size:
            r = summarise({a: q[m] for a, q in p.items()}, y[m], subj[m], SEED + 1 + int(s))
            r["share_of_landmarks"] = float(m.mean())
            res[f"vaso_on = {int(s)}"] = r

    def vshare(w):
        w = np.asarray(w, dtype=np.float64)
        return float(sum(w[names.index(n)] for n in vaso_names) / w.sum())

    res["weights"] = {"vaso_share": {"frozen": vshare(w_frozen), "nca_refit": vshare(w_refit), "a_sel": vshare(w_sel),
                                     "a_strong": vshare(w_strong), "b_state_0": vshare(w_state[False]),
                                     "b_state_1": vshare(w_state[True]), "mi_prior": vshare(prior)},
                      "refit_vs_frozen_logw_r": float(np.corrcoef(np.log(w_refit + 1e-12),
                                                                  np.log(w_frozen + 1e-12))[0, 1]),
                      "lambda": lam, "lambda_scores": W["lambda_scores"],
                      "b_state_own_weights": {str(int(s)): bool(w_state[s] is not w_refit) for s in (False, True)}}
    res["seconds"] = time.perf_counter() - t0
    return res


# --------------------------------------------------------------------------- statistics
def summarise(p: dict, y: np.ndarray, subjects: np.ndarray, seed: int) -> dict:
    from hypo.stats import calibration_line_w, citl_w, logit, logloss_rows
    from sklearn.metrics import roc_auc_score

    losses = {a: logloss_rows(v, y) for a, v in p.items()}
    est = {a: float(v.mean()) for a, v in losses.items()}
    B = FM._boot(losses, subjects, FM.N_BOOT, seed)
    arms = {}
    for a, v in p.items():
        x = logit(np.clip(v, 1e-6, 1 - 1e-6))
        auc = float(roc_auc_score(y, v)) if np.unique(y).size == 2 else float("nan")
        arms[a] = {"logloss": est[a], "auroc": auc, "citl": citl_w(x, y.astype(np.float64)),
                   "slope": calibration_line_w(x, y.astype(np.float64))[1]}
    out = {"n": int(y.size), "arms": arms, "vs_refit": {}, "vs_mi": {}}
    for a in REPAIRS + ["nca_transported"]:
        d = B["nca_refit"] - B[a]               # positive: the arm is better than NCA-refit
        out["vs_refit"][a] = {"est": float(est["nca_refit"] - est[a]), "ci95": FM._ci(d),
                              "loss_upper95": float(FM._ci(B[a] - B["nca_refit"])[1])}
        if "mi_transported" in B:
            out["vs_mi"][a] = {"est": float(est["mi_transported"] - est[a]), "ci95": FM._ci(B["mi_transported"] - B[a])}
    return out


def verdicts(sites: dict) -> dict:
    """The pre-specified rule: repairs (MIMIC-IV's vasopressor stratum) and safety (every site)."""
    out = {}
    for a in REPAIRS:
        st = sites.get("mimic", {}).get("vaso_on = 1")
        repairs = bool(st and st["vs_refit"][a]["ci95"][0] > 0)
        safe = all(r["all"]["vs_refit"][a]["loss_upper95"] < SAFE for r in sites.values())
        out[a] = {"repairs": repairs, "safe": safe, "works": repairs and safe}
    return out


# --------------------------------------------------------------------------- report
LABEL = {"nca_transported": "NCA transported (stored)", "nca_refit": "NCA refit (reference)",
         "a_sel": "A-sel: MI prior, λ chosen", "a_strong": "A-strong: MI prior, λ = 1",
         "b0_stratified": "B0: frozen metric, same-state neighbours", "b_conditional": "B: per-state metric",
         "mi_transported": "MI-transported (§17)", "euclidean": "Euclidean", "nca_local": "NCA local"}


def render(sites: dict, verdict: dict) -> str:
    L = ["# Two repairs of the vasopressor failure (`hypo`, exploratory)", "",
         "Plan fixed before running: docs/DECISIONS.md, 2026-11-06. Same encoder and shapes, case base, k and vote in "
         "every arm; only the weights or the neighbourhood change. Δ = L(NCA-refit) − L(arm): positive = the arm is "
         f"better. {FM.N_BOOT:,} patient resamples.", ""]
    for X, r in sites.items():
        L += [f"## {X}", "", "| arm | stratum | log-loss | Δ vs NCA-refit (95 % CI) | AUROC | CITL | slope |",
              "|---|---|---|---|---|---|---|"]
        for s, u in r.items():
            if not isinstance(u, dict) or "arms" not in u:
                continue
            for a, m in u["arms"].items():
                d = u["vs_refit"].get(a)
                ds = f"{d['est']:+.4f} ({d['ci95'][0]:+.4f} to {d['ci95'][1]:+.4f})" if d else "—"
                L.append(f"| {LABEL.get(a, a)} | {s} | {m['logloss']:.4f} | {ds} | {m['auroc']:.3f} | "
                         f"{m['citl']:+.3f} | {m['slope']:.2f} |")
        w = r["weights"]
        L += ["", f"λ chosen: {w['lambda']} (validation: {w['lambda_scores']}); log-weight correlation NCA-refit vs "
              f"frozen: {w['refit_vs_frozen_logw_r']:.3f}; vasopressor inputs' share of the weights: "
              + ", ".join(f"{k} {v:.2%}" for k, v in w["vaso_share"].items()), ""]
    L += ["## Rule (pre-specified)", "", "| arm | repairs MIMIC-IV's vasopressor stratum | safe at every site | works |",
          "|---|---|---|---|"]
    for a, v in verdict.items():
        L.append(f"| {LABEL[a]} | {'yes' if v['repairs'] else 'no'} | {'yes' if v['safe'] else 'no'} | "
                 f"{'**yes**' if v['works'] else 'no'} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="*", default=["inspire", "mimic", "eicu"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="quick frozen artefacts (tests)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.external import Stage
    from hypo.frozen import check, external_dir

    rec = check()
    st = Stage(quick=a.quick, threads=a.threads)
    od = external_dir().with_name("posthoc_vaso_repair")
    res = {"freeze": rec.get("tag"), "sites": {}}
    for X in a.sites:
        res["sites"][X] = run_site(st, X, od / "units", a.threads)
    check()
    res["rule"] = verdicts(res["sites"])
    (od / "vaso_repair.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    rep = external_dir().parent / "reports" / "posthoc_vaso_repair.md"
    rep.write_text(render(res["sites"], res["rule"]), encoding="utf-8")
    LOG.info("vaso repair: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
