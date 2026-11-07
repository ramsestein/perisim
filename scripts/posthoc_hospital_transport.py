"""Post-hoc, PRE-SPECIFIED transport of the metric's weights between hospitals inside eICU (`hypo`; plan:
docs/DECISIONS.md 2026-11-07).

The external stage moved the metric between health systems (INSPIRE, MIMIC-IV, eICU). Here the unit is the
hospital: for each of eICU's largest hospitals, NCA weights learned on eICU's OTHER hospitals are used with that
hospital's own case base, and compared with weights learned on that hospital itself and with the weights transported
from the other two systems. All arms share the frozen eICU-target metric's encoder (its shapes and axes); only the
weights differ, which is where §14.3 and §15.4 place what travels.

Arms (per target hospital h, on h's landmarks, eICU's five external-stage folds restricted to h; case base = h's
landmarks of the other folds, k = 50, the frozen vote):
  uniform        weights all 1 (the Euclidean reference of R in this geometry)
  cross_system   the frozen weights transported from INSPIRE + MIMIC-IV (the external stage's metric for eICU)
  other_hosp     NCA weights learned on eICU minus h (frozen NCA settings; learning sample ≤ 3 landmarks per patient,
                 ≤ 30,000 landmarks; candidates from the whole pool)
  local          NCA weights learned on h's own case-base landmarks of each fold (same settings)

Usage:  python scripts/posthoc_hospital_transport.py [--threads 8] [--quick]
Outputs (aggregates; hospitals anonymised by size rank): posthoc_hospital_transport/hospitals.json,
reports/posthoc_hospital_transport.md
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
try:
    import _frozen  # noqa: E402
    _frozen.use_src()
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

import posthoc_fixed_metric as FM  # noqa: E402
import posthoc_state_test as T  # noqa: E402

LOG = logging.getLogger("posthoc.hospitals")
SITE = "eicu"
MIN_PATIENTS = 500
MAX_HOSPITALS = 10
N_BOOT = 2000
SEED = 20261107
ARMS = ("uniform", "cross_system", "other_hosp", "local")


def nca_weights(enc, sample, mcfg: dict, seed: int, threads: int) -> np.ndarray:
    import torch
    from hypo.metric import C, SiteNCA

    torch.set_num_threads(max(1, int(threads)))
    n = mcfg["nca"]
    E = enc.transform(sample.ft)
    nca = SiteNCA(enc.dim_feature, enc.n_features, C, n_candidates=int(n["n_candidates"]),
                  batch_size=int(n["batch_size"]), lr=float(n["lr"]), refresh_every=int(n["refresh_every"]),
                  eps0=float(n["eps0"]), label_smoothing=float(n["label_smoothing"]), seed=seed, sites=None)
    nca.fit(E, sample.y, epochs=int(n["epochs"]), subjects=sample.ft.subjects)
    return np.asarray(nca.weights, dtype=np.float64)


def learn(st, lm, rows: np.ndarray, enc, seed: int, path: Path) -> np.ndarray:
    from hypo.data import learning_sample

    if path.exists():
        return np.asarray(json.loads(path.read_text())["w"], dtype=np.float64)
    pool = lm.take(rows)
    pool.site[:] = SITE                                    # one pool: the per-site balancing does not apply
    s = learning_sample(pool, int(st.mcfg["learn"]["per_patient"]), int(st.mcfg["learn"]["per_site"]), seed)
    w = nca_weights(enc, pool.take(s), st.mcfg, seed, st.threads)
    path.write_text(json.dumps({"w": w.tolist(), "n_learn": int(s.size)}))
    return w


def predict(st, lm, E, enc, w, base: np.ndarray, q: np.ndarray, seed: int) -> np.ndarray:
    from hypo.casebase import CaseBase
    from hypo.metric import Metric

    cb = CaseBase(Metric(enc, w), lm.take(base), st.mcfg, E=E[base])
    cb.fit_alpha(seed)
    return np.asarray(cb.predict(lm.take(q), E=E[q])["p"], dtype=np.float64)


def summarise(p: dict, y: np.ndarray, subj: np.ndarray, seed: int) -> dict:
    from hypo.stats import citl_w, logit, logloss_rows
    from sklearn.metrics import roc_auc_score

    losses = {a: logloss_rows(v, y) for a, v in p.items()}
    est = {a: float(v.mean()) for a, v in losses.items()}
    B = FM._boot(losses, subj, N_BOOT, seed)
    e = {"est": {("euclidean" if a == "uniform" else "nca_local" if a == "local" else a): v for a, v in est.items()},
         "B": {("euclidean" if a == "uniform" else "nca_local" if a == "local" else a): v for a, v in B.items()}}
    out = {"n_landmarks": int(y.size), "n_patients": int(np.unique(subj.astype(str)).size),
           "prevalence": float(y.mean()), "arms": {}}
    for a, v in p.items():
        x = logit(np.clip(v, 1e-6, 1 - 1e-6))
        out["arms"][a] = {"logloss": est[a], "auroc": float(roc_auc_score(y, v)), "citl": citl_w(x, y.astype(float))}
    out["R_other_hosp"] = T.ratio_r(e, "other_hosp")
    out["R_cross_system"] = T.ratio_r(e, "cross_system")
    d = e["B"]["cross_system"] - e["B"]["other_hosp"]
    out["cross_minus_other"] = {"est": est["cross_system"] - est["other_hosp"], "ci95": FM._ci(d)}
    return out


def run(st, units: Path) -> dict:
    lm = st.S.lm(SITE)
    hosp = lm.hospital.astype(str)
    subj = lm.ft.subjects.astype(str)
    sizes = {h: np.unique(subj[hosp == h]).size for h in np.unique(hosp)}
    targets = [h for h, n in sorted(sizes.items(), key=lambda t: -t[1]) if n >= MIN_PATIENTS][:MAX_HOSPITALS]
    LOG.info("hospitals: %d eligible targets", len(targets))
    metric = st.metric(SITE)
    enc, w_cross = metric.enc, np.asarray(metric.w, dtype=np.float64)
    E = st.embed(SITE, SITE)
    fold = st.S.fold(SITE)
    # gate: the frozen transported arm with the whole case base, fold 0, reproduces the stored predictions
    import polars as pl
    from hypo.external_report import _on
    from hypo.frozen import external_dir
    tr0, te0 = st.S.split(SITE, 0)
    ref = _on(pl.read_parquet(external_dir() / f"transported_{SITE}_f0.parquet"), te0, "p")
    gap = np.abs(predict(st, lm, E, enc, w_cross, tr0, te0, st.seed) - ref)
    if not np.all(np.isfinite(gap)) or gap.max() > 1e-9:
        raise RuntimeError("hospitals: the frozen transported arm does not reproduce its stored predictions")
    LOG.info("hospitals: gate passed (max |Δp| = %.2e)", gap.max())
    res, pooled = {"n_eligible": len(targets), "hospitals": {}}, {a: [] for a in ARMS}
    rows_all = []
    for r, h in enumerate(targets, start=1):
        t0 = time.perf_counter()
        in_h = np.where(hosp == h)[0]
        w_other = learn(st, lm, np.where(hosp != h)[0], enc, SEED + r, units / f"w_other_h{r}.json")
        P = {a: np.full(in_h.size, np.nan) for a in ARMS}
        for f in sorted(set(fold[in_h].tolist())):
            q, base = in_h[fold[in_h] == f], in_h[fold[in_h] != f]
            w_loc = learn(st, lm, base, enc, SEED + 100 * r + f, units / f"w_local_h{r}_f{f}.json")
            pos = np.isin(in_h, q)
            seed = st.seed + 100 * f
            for a, w in (("uniform", np.ones(enc.n_features)), ("cross_system", w_cross), ("other_hosp", w_other),
                         ("local", w_loc)):
                P[a][pos] = predict(st, lm, E, enc, w, base, q, seed)
        if not all(np.isfinite(v).all() for v in P.values()):
            raise RuntimeError(f"hospitals: missing predictions at target {r}")
        res["hospitals"][f"h{r}"] = summarise(P, lm.y[in_h], subj[in_h], SEED + r)
        for a in ARMS:
            pooled[a].append(P[a])
        rows_all.append(in_h)
        LOG.info("hospitals: target %d of %d done (%.0f s)", r, len(targets), time.perf_counter() - t0)
    rows = np.concatenate(rows_all)
    res["pooled"] = summarise({a: np.concatenate(v) for a, v in pooled.items()}, lm.y[rows], subj[rows], SEED)
    return res


def rule(res: dict) -> dict:
    """The pre-specified reading (DECISIONS 2026-11-07)."""
    P = res["pooled"]
    hs = res["hospitals"].values()
    ev = [h for h in hs if h["R_other_hosp"].get("est") is not None]
    return {
        "Q1_within_system_equivalent_pooled": P["R_other_hosp"]["verdict"],
        "Q2_cross_vs_other_pooled": ("other hospitals better" if P["cross_minus_other"]["ci95"][0] > 0 else
                                     "cross-system better" if P["cross_minus_other"]["ci95"][1] < 0 else "no difference"),
        "Q3_hospitals_evaluable": len(ev),
        "Q3_hospitals_equivalent": sum(h["R_other_hosp"]["verdict"] == "equivalent" for h in ev),
        "Q3_cross_system_equivalent": sum(h["R_cross_system"]["verdict"] == "equivalent" for h in hs
                                          if h["R_cross_system"].get("est") is not None),
    }


def render(res: dict) -> str:
    def r(d):
        return "not evaluable" if d.get("est") is None else \
            f"{d['est']:.2f} ({d['ci90'][0]:.2f}–{d['ci90'][1]:.2f})"
    L = ["# Transport of the weights between eICU hospitals", "",
         "Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. Each target hospital's own case base; "
         "same encoder (the frozen eICU-target metric's); R = (L_uniform − L_arm)/(L_uniform − L_local), 90 % CI. "
         "Hospitals anonymised by size rank.", "",
         "| hospital | patients | landmarks | prevalence | log-loss uniform / cross-system / other hospitals / local | "
         "R other hospitals | R cross-system | cross − other (95 % CI) |", "|---|---|---|---|---|---|---|---|"]
    for name, h in list(res["hospitals"].items()) + [("pooled", res["pooled"])]:
        a = h["arms"]
        c = h["cross_minus_other"]
        L.append(f"| {name} | {h['n_patients']:,} | {h['n_landmarks']:,} | {h['prevalence']:.1%} | "
                 + " / ".join(f"{a[x]['logloss']:.4f}" for x in ARMS)
                 + f" | {r(h['R_other_hosp'])} | {r(h['R_cross_system'])} | {c['est']:+.4f} "
                 f"({c['ci95'][0]:+.4f} to {c['ci95'][1]:+.4f}) |")
    L += ["", "AUROC and CITL by arm are in hospitals.json.", "", "## Rule (pre-specified)", "", "```",
          json.dumps(res["rule"], indent=1), "```", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.external import Stage
    from hypo.frozen import check, external_dir

    rec = check()
    st = Stage(quick=a.quick, threads=a.threads)
    od = external_dir().with_name("posthoc_hospital_transport")
    (od / "units").mkdir(parents=True, exist_ok=True)
    res = run(st, od / "units")
    check()
    res["freeze"] = rec.get("tag")
    res["rule"] = rule(res)
    (od / "hospitals.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    rep = external_dir().parent / "reports" / "posthoc_hospital_transport.md"
    rep.write_text(render(res), encoding="utf-8")
    LOG.info("hospitals: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
