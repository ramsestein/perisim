"""Distillation study (docs/prereg_distill.md): a PERISIM metric learned from a frozen teacher and a synthetic cohort
built from population aggregates, evaluated at the six primary units of docs/prereg_arc.md.

Usage:
    python scripts/distill_run.py <step> --source <work_value> [--out <work_distill>] [--registration <OSF ID>]
                                  [--threads 8] [--quick]

Steps, in order (each resumes: what exists is not redone):

- ``aggregates``  per rotation, the real learning sample of `hypo.frozen.final` and one aggregate file per source
                  site (`distill/aggregates/ext-<e>/<site>.json`). Needs ``--registration``: the OSF ID must also be
                  written in docs/prereg_distill.md (§7: nothing is generated from real aggregates before that).
- ``fit``         per rotation and arm (cop, ind): the synthetic cohort, labelled by the frozen LightGBM of arm
                  `value` (`final/comp_ext-<e>_lightgbm.pkl`), `fit_metric` on it, and the fidelity / DCR summaries.
- ``check``       integrity checks of §3.5 before any target: (1) M_REAL re-fitted for rotation eICU with this code
                  path reproduces arm `value`'s weights to 1e-6; (4) this code's case-base prediction reproduces the
                  published transported predictions of arm `value` (hypo eICU fold 0; NWICU rotation eICU fold 0).
- ``freeze``      the freeze record (`distill/freeze.json`): registration, prereg hash, code hash, arm `value`'s
                  freeze tag, and the hash of every aggregate file and distilled metric.
- ``evaluate``    the distilled metrics at the targets (hypo: external-stage folds; NWICU: its folds), per fold.
- ``dump``        per arm, `<arm>/arc_compare/<unit>.npz` in the format of scripts/arc_dump.py: the distilled metric
                  as "transported"; ids, subjects, y, local and Euclidean log-losses copied from arm `value`'s dump.
- ``compare``     H1–H3 and the descriptives (`distill/results.json`, `distill/results.md`; aggregates only).
- ``all``         every step in order.

The source folder (arm `value`'s work folder) is only read: PERISIM_WORKDIR points at it so that `hypo` finds its
landmarks, folds and frozen artefacts; everything this study writes goes to ``--out`` (default: work_distill next to
the source), which carries the marker DISTILL_WORKDIR. Patient-level files (synthetic cohorts are never written;
predictions and per-row log-losses are) stay in ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

LOG = logging.getLogger("distill")
PREREG = ROOT / "docs" / "prereg_distill.md"
MARKER = "DISTILL_WORKDIR"
OSF = re.compile(r"^10\.17605/OSF\.IO/[A-Z0-9]{4,}$")
ROTATIONS = ("inspire", "mimic", "eicu")
CHECK_TOL = 1e-6
STEPS = ("aggregates", "fit", "check", "freeze", "evaluate", "dump", "compare")


# ---------------------------------------------------------------------------- folders, hashes, registration
def _sha(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes().replace(b"\r\n", b"\n") if path.suffix in (".py", ".md", ".json") else path.read_bytes())
    return h.hexdigest()[:n]


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "src" / "distill").rglob("*.py")) + [Path(__file__).resolve()]:
        h.update(f.name.encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def guard_out(out: Path, source: Path) -> Path:
    out, source = out.resolve(), source.resolve()
    if out == source or source in out.parents or out in source.parents:
        raise SystemExit(f"[distill] --out must be outside the source folder ({source})")
    out.mkdir(parents=True, exist_ok=True)
    m = out / MARKER
    if not m.exists():
        if any(p.name != MARKER for p in out.iterdir()):
            raise SystemExit(f"[distill] {out} holds other files and no {MARKER} marker: use a new folder")
        m.write_text(json.dumps({"study": "distill", "prereg": "docs/prereg_distill.md", "source": str(source)}))
    return out


def registered(reg: str | None) -> str:
    if not reg or not OSF.match(reg):
        raise SystemExit("[distill] --registration <OSF ID> (10.17605/OSF.IO/XXXXX) is required before any synthetic "
                         "cohort is generated from real aggregates (prereg §5, §7)")
    if reg not in PREREG.read_text(encoding="utf-8"):
        raise SystemExit(f"[distill] {reg} is not written in docs/prereg_distill.md: record the registration first")
    return reg


def d(out: Path, *parts) -> Path:
    p = out / "distill"
    for x in parts:
        p = p / x
    p.mkdir(parents=True, exist_ok=True)
    return p


def wjson(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------- shared context
class Ctx:
    def __init__(self, a):
        from hypo.cv import settings
        from hypo.frozen import check, final_dir

        self.a = a
        self.out = a.out
        self.cfg, self.mcfg, self.space = settings(a.quick)
        self.seed = int(self.cfg["seed"])
        self.final_seed = self.seed + int(self.cfg["external"]["final_seed_offset"])
        self.final = final_dir()
        self.rec = check()                    # arm value's freeze: nothing it covers has changed
        self._lm = {}

    def lm(self, c: str):
        from hypo.data import load

        if c not in self._lm:
            self._lm[c] = load(c, self.cfg)
        return self._lm[c]

    def learn(self, e: str):
        from distill.study import learn_rows
        from hypo.data import concat

        pool = concat([self.lm(c) for c in self.cfg["cohorts"] if c != e])
        return pool, pool.take(learn_rows(pool, self.mcfg, self.final_seed))

    def aggs(self, e: str) -> dict:
        from distill.aggregates import SiteAggregates

        p = d(self.out, "aggregates", f"ext-{e}")
        files = sorted(p.glob("*.json"))
        if not files or not (d(self.out) / f"aggregates_ext-{e}.json").exists():
            raise SystemExit(f"[distill] no aggregates for rotation {e}: run the step `aggregates`")
        return {f.stem: SiteAggregates.from_json(json.loads(f.read_text(encoding="utf-8"))) for f in files}

    def teacher(self, e: str):
        from hypo.frozen import load_pickle

        return load_pickle(self.final / f"comp_ext-{e}_lightgbm.pkl")

    def metric_path(self, arm: str, e: str) -> Path:
        return d(self.out, "metrics") / f"metric_dist-{arm}_ext-{e}.pkl"


# ---------------------------------------------------------------------------- steps
def step_aggregates(c: Ctx) -> None:
    from distill import aggregates as A
    from distill.study import rotation_aggregates

    reg = registered(c.a.registration)
    done, bad = {}, []
    for i, e in enumerate(ROTATIONS):
        if (d(c.out) / f"aggregates_ext-{e}.json").exists():        # written last: the rotation is complete
            continue
        _, learn = c.learn(e)
        ag = rotation_aggregates(learn, c.final_seed + 1000 * (i + 1))
        for s, a in ag.items():
            bad += [f"ext-{e}/{s}: {x}" for x in A.small_cells(a)]
        done[e] = (learn, ag)
    if bad:                                                            # nothing is written while any cell is small
        raise SystemExit(f"[distill] released cells on fewer than {A.MIN_PATIENTS} patients (nothing written):\n  "
                         + "\n  ".join(bad))
    for e, (learn, ag) in done.items():
        p = d(c.out, "aggregates", f"ext-{e}")
        for s, a in ag.items():
            wjson(p / f"{s}.json", a.to_json())
        wjson(d(c.out) / f"aggregates_ext-{e}.json",
              {"rotation": e, "registration": reg, "n_learn": len(learn), "seed": c.final_seed,
               "sites": {s: {"n_rows": a.n_rows, "n_patients": a.n_patients, "dims": len(a.corr),
                             "min_patients_cell": A.released_min_patients(a)} for s, a in ag.items()}})
        LOG.info("aggregates ext-%s: %s", e, {s: a.n_rows for s, a in ag.items()})


def step_fit(c: Ctx) -> None:
    from distill.study import ARMS, dcr, fidelity, fit_on, synthetic_cohort, weight_agreement
    from hypo.frozen import dump, load_pickle

    for i, e in enumerate(ROTATIONS):
        ag = c.aggs(e)
        teacher = c.teacher(e)
        _, learn = c.learn(e)
        m_real = load_pickle(c.final / f"metric_ext-{e}.pkl")
        tpath = d(c.out) / f"teacher_ext-{e}.json"
        if not tpath.exists():                                   # §3.5.3: in sample (it saw these rows) and at the target
            from sklearn.metrics import roc_auc_score

            tgt = c.lm(e)
            wjson(tpath, {"rotation": e, "auroc_learn_in_sample": float(roc_auc_score(learn.y, teacher(learn))),
                          "auroc_target": float(roc_auc_score(tgt.y, teacher(tgt))),
                          "mean_p_learn": float(np.mean(teacher(learn))), "prevalence_learn": float(learn.y.mean())})
        for arm in ARMS:
            path = c.metric_path(arm, e)
            if path.exists():
                continue
            t = time.perf_counter()
            syn, info = synthetic_cohort(ag, teacher, learn.ft.specs, c.final_seed, i, arm)
            m = fit_on(syn, c.mcfg, c.space, c.final_seed, c.a.threads)
            rec = {"rotation": e, "arm": arm, "registration": c.a.registration, "seed": c.final_seed,
                   "code_hash": code_hash(), "seconds": time.perf_counter() - t, "labels": info,
                   "real_prevalence_learn": float(learn.y.mean()),
                   "fidelity": fidelity(syn, learn, ag, c.final_seed),
                   "dcr": dcr(syn, learn, c.final_seed),
                   "weights": weight_agreement(m, m_real),
                   "dropped": m.info.get("dropped_site_constant") if isinstance(m.info, dict) else None}
            wjson(path.with_suffix(".json"), rec)
            dump(m, path)
            LOG.info("metric %s ext-%s: %.0f s, Spearman with M_REAL %.3f", arm, e, rec["seconds"],
                     rec["weights"]["spearman"])


def _published_hypo(c: Ctx, X: str, f: int):
    import polars as pl
    from hypo.frozen import external_dir

    p = external_dir() / f"transported_{X}_f{f}.parquet"
    if not p.is_file():
        return None
    t = pl.read_parquet(p)
    return t["row"].to_numpy(), t["p"].to_numpy()


def step_check(c: Ctx) -> None:
    from distill.study import fit_on, predict_folds
    from hypo.cv import Sites, cv_dir
    from hypo.frozen import load_pickle

    out = d(c.out) / "checks.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if "refit_real_eicu" not in res:
        e = "eicu"
        _, learn = c.learn(e)
        t = time.perf_counter()
        m = fit_on(learn, c.mcfg, c.space, c.final_seed, c.a.threads)
        ref = load_pickle(c.final / f"metric_ext-{e}.pkl")
        a, b = m.weights_table(), ref.weights_table()
        diff = max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))
        res["refit_real_eicu"] = {"max_weight_diff": diff, "pass": diff <= CHECK_TOL, "seconds": time.perf_counter() - t}
        wjson(out, res)
    if "casebase_hypo_eicu_f0" not in res:
        S = Sites(c.cfg, cv_dir(False))
        lm, (tr, te) = S.lm("eicu"), S.split("eicu", 0)
        p = predict_folds(load_pickle(c.final / "metric_ext-eicu.pkl"), lm, [(0, tr, te)], c.mcfg, c.seed)
        pub = _published_hypo(c, "eicu", 0)
        diff = float(np.max(np.abs(p[pub[0]] - pub[1]))) if pub is not None else float("nan")
        res["casebase_hypo_eicu_f0"] = {"max_p_diff": diff, "pass": bool(diff <= 1e-9)}
        wjson(out, res)
    if "casebase_nwicu_eicu_f0" not in res:
        import nwicu_transport as NT
        from hypo.cv import outer_folds

        lm = NT.landmarks(c.cfg)
        fold = outer_folds(lm, int(c.cfg["cv"]["outer_folds"]), c.seed)
        tr, te = np.where(fold != 0)[0], np.where(fold == 0)[0]
        p = predict_folds(load_pickle(c.final / "metric_ext-eicu.pkl"), lm, [(0, tr, te)], c.mcfg, c.seed)
        r, v = NT.read(NT.work() / "units" / "knn_eicu_f0.parquet")
        diff = float(np.max(np.abs(p[r] - v)))
        res["casebase_nwicu_eicu_f0"] = {"max_p_diff": diff, "pass": bool(diff <= 1e-9)}
        wjson(out, res)
    bad = [k for k, v in res.items() if not v.get("pass")]
    LOG.info("checks: %s", {k: v for k, v in res.items()})
    if bad:
        raise SystemExit(f"[distill] integrity check failed: {bad}. The run stops (prereg §3.5); investigate and log "
                         "in docs/DECISIONS.md")


def _artefacts(out: Path) -> dict:
    files = sorted((out / "distill" / "aggregates").rglob("*.json")) + sorted((out / "distill" / "metrics").glob("*.pkl"))
    return {str(f.relative_to(out)).replace("\\", "/"): _sha(f) for f in files}


def step_freeze(c: Ctx) -> None:
    from distill.study import ARMS

    reg = registered(c.a.registration)
    missing = [f"{a}/{e}" for a in ARMS for e in ROTATIONS if not c.metric_path(a, e).exists()]
    if missing:
        raise SystemExit(f"[distill] cannot freeze: metrics missing {missing}")
    chk = d(c.out) / "checks.json"
    if not chk.exists() or not all(v.get("pass") for v in json.loads(chk.read_text(encoding="utf-8")).values()):
        raise SystemExit("[distill] cannot freeze before the integrity checks pass (step `check`)")
    p = d(c.out) / "freeze.json"
    rec = {"registration": reg, "prereg_sha": _sha(PREREG), "code_hash": code_hash(),
           "source_freeze": c.rec.get("tag"), "artefacts": _artefacts(c.out),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if {k: old[k] for k in ("registration", "prereg_sha", "artefacts")} != \
                {k: rec[k] for k in ("registration", "prereg_sha", "artefacts")}:
            raise SystemExit("[distill] a different freeze record exists: the artefacts or the plan changed")
        return
    wjson(p, rec)
    LOG.info("frozen: %d artefacts, registration %s", len(rec["artefacts"]), reg)


def frozen(c: Ctx) -> dict:
    p = d(c.out) / "freeze.json"
    if not p.exists():
        raise SystemExit("[distill] no freeze record: run the step `freeze` first")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec["artefacts"] != _artefacts(c.out) or rec["prereg_sha"] != _sha(PREREG):
        raise SystemExit("[distill] changed since the freeze (artefacts or docs/prereg_distill.md)")
    return rec


def _units_dir(c: Ctx) -> Path:
    return d(c.out, "units")


def _write_unit(path: Path, rows: np.ndarray, p: np.ndarray, info: dict) -> None:
    import polars as pl

    tmp = path.with_suffix(".tmp")
    pl.DataFrame({"row": rows, "p": p}).write_parquet(tmp)
    path.with_suffix(".json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


def step_evaluate(c: Ctx) -> None:
    import nwicu_transport as NT
    from distill.study import ARMS, predict_folds
    from hypo.cv import Sites, cv_dir, outer_folds
    from hypo.frozen import load_pickle

    frozen(c)
    U = _units_dir(c)
    S = Sites(c.cfg, cv_dir(False))
    n_f = int(c.cfg["cv"]["outer_folds"])
    for X in ROTATIONS:                                          # hypo: the target's own folds and case base
        lm = S.lm(X)
        for arm in ARMS:
            m = E = None
            for f in range(n_f):
                path = U / f"hypo_{X}_{arm}_f{f}.parquet"
                if path.exists():
                    continue
                if m is None:
                    m = load_pickle(c.metric_path(arm, X))
                    E = m.embed(lm)
                tr, te = S.split(X, f)
                t = time.perf_counter()
                p = predict_folds(m, lm, [(f, tr, te)], c.mcfg, c.seed, E=E)
                _write_unit(path, te, p[te], {"unit": f"hypo_{X}", "arm": arm, "fold": f,
                                              "seconds": time.perf_counter() - t})
                LOG.info("hypo %s %s fold %d", X, arm, f)
    nw = NT.landmarks(c.cfg)                                     # NWICU: its pre-registered folds and case base
    fold = outer_folds(nw, n_f, c.seed)
    for e in ROTATIONS:
        for arm in ARMS:
            m = E = None
            for f in sorted(set(fold.tolist())):
                path = U / f"nwicu_{e}_{arm}_f{f}.parquet"
                if path.exists():
                    continue
                if m is None:
                    m = load_pickle(c.metric_path(arm, e))
                    E = m.embed(nw)
                tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
                p = predict_folds(m, nw, [(f, tr, te)], c.mcfg, c.seed, E=E)
                _write_unit(path, te, p[te], {"unit": f"nwicu_{e}", "arm": arm, "fold": f})
                LOG.info("nwicu %s %s fold %d", e, arm, f)


def _gather(U: Path, unit: str, arm: str, n: int) -> np.ndarray:
    import polars as pl

    p = np.full(n, np.nan)
    files = sorted(U.glob(f"{unit}_{arm}_f*.parquet"))
    if not files:
        raise SystemExit(f"[distill] no predictions for {unit} {arm}: run the step `evaluate`")
    for f in files:
        t = pl.read_parquet(f)
        p[t["row"].to_numpy()] = t["p"].to_numpy()
    return p


def step_dump(c: Ctx) -> None:
    import nwicu_transport as NT
    from distill.study import ARMS
    from hypo.stats import logloss_rows

    frozen(c)
    U = _units_dir(c)
    src = c.a.source / "arc_compare"
    sets = {f"hypo_{X}": c.lm(X) for X in ROTATIONS}
    nw = NT.landmarks(c.cfg)
    sets.update({f"nwicu_{e}": nw for e in ROTATIONS})
    for unit, lm in sets.items():
        ref = src / f"{unit}.npz"
        if not ref.is_file():
            raise SystemExit(f"[distill] arm value's dump {ref} is missing (scripts/arc_dump.py, arm value)")
        with np.load(ref, allow_pickle=False) as z:
            v = {k: z[k] for k in z.files}
        pos = {s: i for i, s in enumerate(lm.ft.ids.astype(str))}
        rows = np.array([pos[s] for s in v["ids"]])
        if not np.array_equal(lm.y[rows].astype(v["y"].dtype), v["y"]):
            raise SystemExit(f"[distill] {unit}: labels do not match arm value's dump")
        for arm in ARMS:
            p = _gather(U, unit, arm, len(lm))[rows]
            if not np.isfinite(p).all():
                raise SystemExit(f"[distill] {unit} {arm}: rows of arm value's dump without a prediction")
            o = c.out / arm / "arc_compare"
            o.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(o / f"{unit}.npz", ids=v["ids"], subjects=v["subjects"], y=v["y"],
                                ll_transported=logloss_rows(p, v["y"]), ll_local=v["ll_local"],
                                ll_euclidean=v["ll_euclidean"])
            np.save(o / f"{unit}.p.npy", p)
        LOG.info("dump %s: %d rows", unit, rows.size)


# ---------------------------------------------------------------------------- the pre-specified readings
LOWER = 0.80


def _ni(r: dict) -> str:
    from arc_compare import _evaluable

    if not _evaluable(r["gain_value"]):
        return "not evaluable"
    return "non-inferior" if r["Q"]["ci90"][0] >= LOWER else "not shown"


def _r(r: dict) -> str:
    from arc_compare import _evaluable

    if not _evaluable(r["gain_local_arc"]):
        return "not evaluable"
    return "equivalent" if r["R_arc"]["ci90"][0] >= LOWER else "not equivalent"


def evaluate_hypotheses(real_cop: dict, real_ind: dict, cop_ind: dict) -> dict:
    units = sorted(real_cop["units"])
    h1 = {u: _ni(real_cop["units"][u]) for u in units}
    ev = [u for u in units if h1[u] != "not evaluable"]
    n_ne = len(units) - len(ev)
    H1 = bool(ev) and all(h1[u] == "non-inferior" for u in ev) and n_ne <= 1
    h2 = {u: cop_ind["units"][u]["D"]["ci95"][0] > 0 for u in units if u in cop_ind["units"]}
    h3 = {u: _r(real_cop["units"][u]) for u in units}
    return {"H1": {"holds": H1, "units": h1, "not_evaluable": n_ne},
            "H2": {"holds": sum(h2.values()) >= 4, "units": h2, "n_ind_worse": int(sum(h2.values()))},
            "H3": {"units": h3},
            "Q_IND": {u: _ni(real_ind["units"][u]) for u in sorted(real_ind["units"])}}


def descriptives(c: Ctx) -> dict:
    from sklearn.metrics import roc_auc_score
    from distill.study import ARMS
    from hypo.stats import calibration_line_w, citl_w, logit

    out = {}
    for arm in ARMS:
        o = c.out / arm / "arc_compare"
        for f in sorted(o.glob("*.npz")):
            with np.load(f, allow_pickle=False) as z:
                y = z["y"].astype(np.float64)
            p = np.clip(np.load(o / f"{f.stem}.p.npy"), 1e-6, 1 - 1e-6)
            x = logit(p)
            a, b = calibration_line_w(x, y)
            out.setdefault(f.stem, {})[arm] = {"citl": citl_w(x, y), "slope": b, "auroc": float(roc_auc_score(y, p))}
    for arm in ARMS:
        for e in ROTATIONS:
            j = json.loads(c.metric_path(arm, e).with_suffix(".json").read_text(encoding="utf-8"))
            out.setdefault("weights", {}).setdefault(e, {})[arm] = {"spearman_with_real": j["weights"]["spearman"],
                                                                     "l1_profile": j["weights"]["l1_profile"],
                                                                     "weights": j["weights"]["weights"]}
            out.setdefault("fidelity", {}).setdefault(e, {})[arm] = {"labels": j["labels"],
                                                                      "real_prevalence": j["real_prevalence_learn"],
                                                                      "marginals": j["fidelity"], "dcr": j["dcr"]}
    return out


def step_compare(c: Ctx) -> None:
    import arc_compare as AC

    rec = frozen(c)
    rc, ri, ci = AC.compare(c.a.source, c.out / "cop"), AC.compare(c.a.source, c.out / "ind"), \
        AC.compare(c.out / "cop", c.out / "ind")
    res = {"registration": rec["registration"], "freeze": {k: rec[k] for k in ("code_hash", "prereg_sha", "source_freeze")},
           "hypotheses": evaluate_hypotheses(rc, ri, ci), "real_vs_cop": rc["units"], "real_vs_ind": ri["units"],
           "cop_vs_ind": ci["units"], "descriptives": descriptives(c)}
    wjson(d(c.out) / "results.json", res)
    (d(c.out) / "results.md").write_text(render(res), encoding="utf-8")
    LOG.info("results: %s", d(c.out) / "results.md")


def render(res: dict) -> str:
    h = res["hypotheses"]
    L = ["# Distillation: a PERISIM metric learned from a teacher and population aggregates", "",
         f"Registration {res['registration']}; code {res['freeze']['code_hash']}; arm value freeze "
         f"{res['freeze']['source_freeze']}. Log-loss in nats; patient bootstrap, 2,000 replicates.", "",
         f"**H1** (Q_COP non-inferior in every evaluable unit): **{'holds' if h['H1']['holds'] else 'does not hold'}**. "
         f"**H2** (SYN-IND worse than SYN-COP in ≥ 4 of 6): **{'holds' if h['H2']['holds'] else 'does not hold'}** "
         f"({h['H2']['n_ind_worse']} of {len(h['H2']['units'])}).", "",
         "| unit | L_euclid | L_real | L_cop | L_ind | Q_COP (90 % CI) | H1 | D_COP (95 % CI) | Q_IND (90 % CI) | "
         "D_IND − D_COP (95 % CI) | R_COP (90 % CI) | H3 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(x, k="est"):
        return f"{x[k]:.3f}"

    def ci_(x, k):
        return f"{x['est']:.3f} ({x[k][0]:.3f}–{x[k][1]:.3f})"

    for u, rc in res["real_vs_cop"].items():
        ri, ci = res["real_vs_ind"].get(u), res["cop_vs_ind"].get(u)
        lg = rc["logloss"]
        L.append(f"| {u} | {lg['euclidean']:.4f} | {lg['value']:.4f} | {lg['arc']:.4f} | "
                 f"{ri['logloss']['arc']:.4f} | {ci_(rc['Q'], 'ci90')} | {h['H1']['units'][u]} | "
                 f"{rc['D']['est']:+.4f} ({rc['D']['ci95'][0]:+.4f} to {rc['D']['ci95'][1]:+.4f}) | "
                 f"{ci_(ri['Q'], 'ci90')} | {ci['D']['est']:+.4f} ({ci['D']['ci95'][0]:+.4f} to {ci['D']['ci95'][1]:+.4f}) | "
                 f"{ci_(rc['R_arc'], 'ci90')} | {h['H3']['units'][u]} |")
    L += ["", "## Descriptives", "", "| unit | arm | CITL | slope | AUROC |", "|---|---|---|---|---|"]
    for u, v in res["descriptives"].items():
        if u in ("weights", "fidelity"):
            continue
        for arm, s in v.items():
            L.append(f"| {u} | {arm} | {s['citl']:+.3f} | {s['slope']:.3f} | {s['auroc']:.3f} |")
    L += ["", "| rotation | arm | Spearman with M_REAL | L1 of weight profiles | mean p_T | real prevalence | "
          "DCR median syn→real | DCR median real→real | exact copies |", "|---|---|---|---|---|---|---|---|---|"]
    for e, v in res["descriptives"]["weights"].items():
        for arm, w in v.items():
            fi = res["descriptives"]["fidelity"][e][arm]
            L.append(f"| {e} | {arm} | {w['spearman_with_real']:.3f} | {w['l1_profile']:.3f} | "
                     f"{fi['labels']['mean_p_teacher']:.4f} | {fi['real_prevalence']:.4f} | "
                     f"{fi['dcr']['synthetic_to_real']['median']:.3f} | {fi['dcr']['real_to_real']['median']:.3f} | "
                     f"{fi['dcr']['synthetic_to_real']['share_exact']:.4f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=STEPS + ("all",))
    ap.add_argument("--source", type=Path, required=True, help="arm value's work folder (read only)")
    ap.add_argument("--out", type=Path, default=None, help="this study's work folder (default: work_distill)")
    ap.add_argument("--registration", default=os.environ.get("PERISIM_DISTILL_OSF"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="quick settings (tests)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    a.source = a.source.resolve()
    a.out = guard_out(a.out or a.source.with_name("work_distill"), a.source)
    os.environ["PERISIM_WORKDIR"] = str(a.source)
    os.environ["PERISIM_ARM"] = "value"
    from perisim.arc_guard import marker_arm

    if marker_arm(a.source) != "value":
        raise SystemExit(f"[distill] {a.source} is not arm value's work folder (marker)")
    c = Ctx(a)
    for s in (STEPS if a.step == "all" else (a.step,)):
        LOG.info("step %s", s)
        globals()[f"step_{s}"](c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
