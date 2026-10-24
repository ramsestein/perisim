"""Model-transfer validations (§6.2b, decided by the lead 2026-10-24) — after the external validation, on the frozen
pipeline.

In each external cohort the patients are split once, at patient level, into local cases L (80 %) and held-out queries
H (20 %). Every variant predicts the SAME queries H with the SAME labels, so all contrasts are paired:

- A_transport   the INSPIRE model as frozen (weights, shapes, scales) with the INSPIRE case base (mode A on H).
- T_local       the INSPIRE metric with the local cases L as case base: transfer the metric, use the local outcomes.
- T_pooled      the INSPIRE metric with INSPIRE ∪ L as case base: the local cases added to the system.
- L_perisim     PERISIM fitted on L alone (its own weights and shapes): a kNN learned in that dataset.
- L_euclidean   unsupervised Euclidean kNN (neutral axes) on L;   L_prevalence   the class prevalence of L.
Multi-source development ("retrained adding eICU", MIMIC-IV only, T2 only): PERISIM refitted on INSPIRE ∪ eICU,
- S_transport (case base INSPIRE ∪ eICU), S_local (case base L), S_pooled (INSPIRE ∪ eICU ∪ L).

Labels: the T2 / T3 cut-points of the target cohort are estimated on L only and applied to L and H; each development
cohort keeps its own cut-points (categories relative to each hospital); the absolute categories are a co-primary
analysis (every case base relabelled, same metrics). eICU: T1 reduced to severe / not severe, as in the external
validation. The kNN smoothing α of each case base is fitted on that case base (never on H); every kNN variant is
reported over the same k grid, the primary k for the contrasts.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing as mp
import pickle
import shutil
import traceback

import numpy as np
import polars as pl

from perisim.evaluation.html import box, page, table
from perisim.evaluation.report import eval_sections, evaluate_predictions, paired_between
from perisim.experiment import TARGET_CLASSES, fit_perisim, knn_predict, labels_for, load_cohort_data
from perisim.external import _eval_target_for, _pred_frame, absolute_labels
from perisim.features.table import FeatureTable
from perisim.knn.predict import KNNModel
from perisim.utils import set_threads, write_json, write_parquet

LOG = logging.getLogger("perisim.transfer")
_SEED_OFFSET = {"inspire": 0, "mimic": 1, "eicu": 2}
# (a, b): Δ log-loss a − b on the held-out queries at the primary k (negative = a better); missing ones are skipped
CONTRASTS = [("T_local", "L_perisim"), ("T_pooled", "T_local"), ("T_local", "A_transport"), ("T_pooled", "L_perisim"),
             ("L_perisim", "L_euclidean"), ("T_local", "L_prevalence"),
             ("S_local", "T_local"), ("S_pooled", "T_pooled"), ("S_transport", "A_transport"), ("S_local", "L_perisim")]


def patient_holdout(subjects: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """True for the stays of a random `fraction` of the patients (the held-out queries)."""
    u = np.unique(subjects.astype(str))
    perm = np.random.default_rng(seed).permutation(u.size)
    held = set(u[perm[:int(round(fraction * u.size))]].tolist())
    return np.array([s in held for s in subjects.astype(str)], dtype=bool)


def concat_ft(a: FeatureTable, b: FeatureTable) -> FeatureTable:
    """Stays of two cohorts in one table (same harmonized features; identifiers are prefixed by cohort)."""
    return FeatureTable(np.concatenate([a.ids, b.ids]), np.concatenate([a.subjects, b.subjects]), a.specs,
                        {k: np.concatenate([a.num[k], b.num[k]]) for k in a.num},
                        {k: np.concatenate([a.cat[k], b.cat[k]]) for k in a.cat})


def _labellings(target: str, et: str, X, fit_idx: np.ndarray, cfg, development: bool) -> dict[str, np.ndarray]:
    """{analysis: labels} for one cohort (−1 = not evaluable). Development cohorts predicting the reduced eICU target
    carry the reduced label (readmission or death, AUDIT E1); cut-points are estimated on `fit_idx` only."""
    if et == "t1_bin":
        own = (X.extra["t1_reduced"] if development else X.t1_severe).copy()
    else:
        own = labels_for(target, X, fit_idx, cfg.targets)[0]
    out = {"own": own}
    if target in ("t2", "t3"):
        out["absolute"] = absolute_labels(target, X, cfg.targets)
    return out


def _knn(E: np.ndarray, w_dims: np.ndarray, y: np.ndarray, C: int, subjects: np.ndarray, k_grid: list[int],
         kcfg: dict) -> KNNModel | None:
    """Case base with the labelled stays only and α fitted on it; None if unusable."""
    ok = np.where(y >= 0)[0]
    if ok.size < 20 or np.unique(y[ok]).size < 2:
        return None
    kn = KNNModel(E[ok], w_dims, y[ok], C, subjects[ok], k_grid)
    kn.fit_alpha_and_ood(kcfg["alpha_grid"], int(kcfg["k_primary"]), kcfg["ood_percentile"])
    return kn


# --------------------------------------------------------------------------- orchestration
def transfer_jobs(cfg, available: list[str], targets: list[str] | None = None) -> list[tuple[str, str, str]]:
    """(target, cohort, source) jobs; source = "inspire" (the frozen development model) or "multisource"."""
    tcfg = cfg.experiment.get("transfer") or {}
    reduced = set(cfg.targets["t1"].get("reduced_cohorts") or [])
    externals = [c for c in cfg.experiment["cohorts"]["external"] if c in available]
    ms = tcfg.get("multisource") or {}
    jobs = []
    for t in targets or cfg.targets_to_model:
        for c in externals:
            if t == "t3" and c in reduced:      # no full T1 in that cohort: T3 not evaluable (as externally)
                continue
            jobs.append((t, c, "inspire"))
        if t in (ms.get("targets") or []) and all(x in available for x in ms.get("development") or []):
            jobs += [(t, c, "multisource") for c in (ms.get("test") or []) if c in externals]
    return jobs


def run_transfer(pipe) -> None:
    from perisim.pipeline import _child_threads

    cfg, ws = pipe.cfg, pipe.ws
    tcfg = cfg.experiment.get("transfer") or {}
    if not tcfg.get("enabled", True):
        LOG.info("transfer: disabled (experiment.transfer.enabled)")
        return
    jobs = transfer_jobs(cfg, pipe.available_cohorts())
    missing = [t for t in {j[0] for j in jobs} if not (ws.predictions("external", t) / "final_model.pkl").exists()]
    if missing:
        raise RuntimeError(f"transfer: no saved final model for {missing}; rerun the external stage "
                           "(--stage external --force) with this code version")
    common = {"cfg": cfg.to_dict(), "profile": cfg.profile, "base": str(ws.base), "experiment": ws.experiment,
              "names": pipe.feature_names(), "d7": pipe.d7(), "threads": pipe.threads(),
              "dev": cfg.experiment["cohorts"]["development"]}
    args = [{**common, "target": t, "cohort": c, "source": s} for t, c, s in jobs]
    par = max(1, min(len(args), pipe.jobs()))
    LOG.info("transfer: %d job(s) %s in %d process(es) × %d threads", len(args), jobs, par, pipe.threads())
    if par == 1:
        results = [_transfer_job(a) for a in args]
    else:
        results = []
        with _child_threads(pipe.threads()), \
                cf.ProcessPoolExecutor(max_workers=par, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_transfer_job, a): a for a in args}
            for fu in cf.as_completed(futs):
                try:
                    results.append(fu.result())
                except Exception as exc:  # worker died (e.g. OOM)
                    a = futs[fu]
                    results.append({"target": a["target"], "cohort": a["cohort"], "source": a["source"],
                                    "error": f"process terminated ({exc!r})"})
    errors = [f"{r['target']}/{r['cohort']}/{r['source']}: {r['error']}" for r in results if r.get("error")]
    if errors:
        raise RuntimeError("Transfer validation failed:\n" + "\n".join(errors))
    out = {"targets": _aggregate(cfg, ws, results), "holdout_fraction": float(tcfg["holdout_fraction"]),
           "note": "Model-transfer validations on held-out patients of each external cohort (§6.2b)."}
    write_json(ws.artifacts() / "transfer.json", out)
    (ws.reports / "transfer.html").write_text(render_transfer(out, int(cfg.experiment["knn"]["k_primary"]),
                                                              int(cfg.experiment["evaluation"]["n_bootstrap"])),
                                              encoding="utf-8")


def _aggregate(cfg, ws, results: list[dict]) -> dict:
    kp = int(cfg.experiment["knn"]["k_primary"])
    nb = int(cfg.experiment["evaluation"]["n_bootstrap"])
    out: dict = {}
    for t in cfg.targets_to_model:
        for c in sorted({r["cohort"] for r in results if r["target"] == t}):
            rs = [r for r in results if r["target"] == t and r["cohort"] == c]
            et = _eval_target_for(t, c, cfg)
            res_c: dict = {"eval_target": et, "info": {r["source"]: r["info"] for r in rs}, "analyses": {}}
            for name in sorted({n for r in rs for n in r["files"]}, key=lambda n: (n != "own", n)):
                pred = pl.concat([pl.read_parquet(r["files"][name]) for r in rs if name in r["files"]],
                                 how="diagonal_relaxed")
                methods = set(pred["method"].unique().to_list())
                ref = "T_local" if "T_local" in methods else sorted(methods)[0]
                ev = evaluate_predictions(pred, et, kp, nb, cfg.seed, ref=(ref, kp))
                kof = {m: (0 if m.endswith("prevalence") else kp) for m in methods}
                con = {f"{a} − {b}": paired_between(pred, (a, kof[a]), (b, kof[b]), nb, cfg.seed)
                       for a, b in CONTRASTS if a in methods and b in methods}
                res_c["analyses"][name] = {"n_queries": int(pred["stay_id"].n_unique()), "evaluation": ev,
                                           "contrasts": {k: v for k, v in con.items() if v is not None}}
            out.setdefault(t, {})[c] = res_c
    return out


def _transfer_job(args: dict) -> dict:
    t, c, s = args["target"], args["cohort"], args["source"]
    try:
        from perisim.config import Config
        from perisim.utils import setup_logging
        from perisim.workspace import Workspace

        setup_logging()
        set_threads(int(args["threads"]))
        cfg = Config(**args["cfg"], profile=args["profile"], variant="main")
        ws = Workspace(args["base"], experiment=args["experiment"])
        files, info = _run_job(cfg, ws, t, c, s, args)
        return {"target": t, "cohort": c, "source": s, "files": files, "info": info}
    except Exception:
        return {"target": t, "cohort": c, "source": s, "error": traceback.format_exc()}


def _fit(cfg, ft: FeatureTable, y: np.ndarray, C: int, seed: int, args: dict, ck) -> object:
    """PERISIM with the same nested procedure and stopping rule as the final development model."""
    d7 = args["d7"]
    st = {k: d7[k] for k in ("W", "P", "n_min", "n_max", "min_delta", "tol_delta") if k in d7}
    ok = np.where(y >= 0)[0]
    fp = fit_perisim(cfg, ft.take(ok), y[ok], C, seed=seed, threads=int(args["threads"]), stopping_override=st,
                     exclude_families=set(d7.get("exclude_families", [])), early_stop=True,
                     need_comparator_weights=False, checkpoint_dir=ck)
    shutil.rmtree(ck, ignore_errors=True)
    return fp


def _run_job(cfg, ws, target: str, c: str, source: str, args: dict) -> tuple[dict, dict]:
    names, dev = args["names"], args["dev"]
    kcfg = cfg.experiment["knn"]
    kp, k_grid = int(kcfg["k_primary"]), [int(k) for k in kcfg["k_grid"]]
    tcfg = cfg.experiment["transfer"]
    et = _eval_target_for(target, c, cfg)
    C = 2 if et == "t1_bin" else TARGET_CLASSES[target]
    seed = int(cfg.seed) + 101 * _SEED_OFFSET.get(c, 9)
    X = load_cohort_data(ws.common(c), ws.derived(c), names)
    held = patient_holdout(X.ft.subjects, float(tcfg["holdout_fraction"]), seed)   # same split for every source
    iL, iH = np.where(~held)[0], np.where(held)[0]
    ftL, ftH = X.ft.take(iL), X.ft.take(iH)
    lab = _labellings(target, et, X, iL, cfg, development=False)       # cut-points from the local cases only
    D = load_cohort_data(ws.common(dev), ws.derived(dev), names)
    lab_D = _labellings(target, et, D, np.arange(len(D)), cfg, development=True)
    out_dir = ws.predictions("transfer", target)
    info: dict = {"n_local_stays": int(iL.size), "n_heldout_stays": int(iH.size),
                  "n_local_patients": int(np.unique(ftL.subjects).size),
                  "n_heldout_patients": int(np.unique(ftH.subjects).size)}

    if source == "inspire":
        with open(ws.predictions("external", target) / "final_model.pkl", "rb") as fh:
            fm = pickle.load(fh)
        enc, w = fm["enc"], fm["w"]
        wd = enc.weights_dims(w)
        E_D, E_L, E_H = enc.transform(D.ft), enc.transform(ftL), enc.transform(ftH)
        N_L, N_H = enc.transform_neutral(ftL), enc.transform_neutral(ftH)
        nd = enc.neutral_weights_dims(np.ones(enc.n_features))
        fpL = None
        if tcfg.get("local_model", True):
            LOG.info("transfer: PERISIM fitted on the local cases of %s (%s, n=%d)", c, et, iL.size)
            fpL = _fit(cfg, ftL, lab["own"][iL], C, seed, args, out_dir / f"{c}_local_ckpt")
            info["local_model"] = {"lambda": fpL.info["lambda"], "n_accepted": fpL.loop["n_accepted"],
                                   "stop_reason": fpL.loop["stop_reason"],
                                   "weights": dict(zip(fpL.enc.feature_names, np.round(fpL.w, 8).tolist()))}
            EL2, EH2, wd2 = fpL.enc.transform(ftL), fpL.enc.transform(ftH), fpL.enc.weights_dims(fpL.w)
        bases = lambda name: [  # noqa: E731  (method, case-base model, query embedding, k grid)
            ("A_transport", _knn(E_D, wd, lab_D[name], C, D.ft.subjects, k_grid, kcfg), E_H, k_grid),
            ("T_local", _knn(E_L, wd, lab[name][iL], C, ftL.subjects, k_grid, kcfg), E_H, k_grid),
            ("T_pooled", _knn(np.vstack([E_D, E_L]), wd, np.concatenate([lab_D[name], lab[name][iL]]), C,
                              np.concatenate([D.ft.subjects, ftL.subjects]), k_grid, kcfg), E_H, k_grid),
            ("L_euclidean", _knn(N_L, nd, lab[name][iL], C, ftL.subjects, [kp], kcfg), N_H, [kp]),
        ] + ([("L_perisim", _knn(EL2, wd2, lab[name][iL], C, ftL.subjects, k_grid, kcfg), EH2, k_grid)]
             if fpL is not None else [])
    else:   # multi-source development: PERISIM refitted on INSPIRE ∪ the other development cohorts
        others = [x for x in (tcfg.get("multisource") or {}).get("development", []) if x != dev]
        ft_S, lab_S = D.ft, {k: v for k, v in lab_D.items()}
        for o in others:
            O = load_cohort_data(ws.common(o), ws.derived(o), names)
            lo = _labellings(target, et, O, np.arange(len(O)), cfg, development=True)   # its own cut-points
            ft_S = concat_ft(ft_S, O.ft)
            lab_S = {k: np.concatenate([lab_S[k], lo[k]]) for k in lab_S}
        LOG.info("transfer: PERISIM refitted on %s (n=%d) for %s", [dev] + others, len(ft_S), c)
        fpS = _fit(cfg, ft_S, lab_S["own"], C, int(cfg.seed), args, out_dir / f"{c}_multisource_ckpt")
        info["multisource_model"] = {"development": [dev] + others, "n_train": int((lab_S["own"] >= 0).sum()),
                                     "lambda": fpS.info["lambda"], "n_accepted": fpS.loop["n_accepted"],
                                     "stop_reason": fpS.loop["stop_reason"],
                                     "weights": dict(zip(fpS.enc.feature_names, np.round(fpS.w, 8).tolist()))}
        wd = fpS.enc.weights_dims(fpS.w)
        E_S, E_L, E_H = fpS.enc.transform(ft_S), fpS.enc.transform(ftL), fpS.enc.transform(ftH)
        bases = lambda name: [  # noqa: E731
            ("S_transport", _knn(E_S, wd, lab_S[name], C, ft_S.subjects, k_grid, kcfg), E_H, k_grid),
            ("S_local", _knn(E_L, wd, lab[name][iL], C, ftL.subjects, k_grid, kcfg), E_H, k_grid),
            ("S_pooled", _knn(np.vstack([E_S, E_L]), wd, np.concatenate([lab_S[name], lab[name][iL]]), C,
                              np.concatenate([ft_S.subjects, ftL.subjects]), k_grid, kcfg), E_H, k_grid),
        ]

    files = {}
    for name, y in lab.items():
        yH = y[iH]
        q = np.where(yH >= 0)[0]
        if q.size < 20:
            LOG.warning("transfer: only %d evaluable held-out stays in %s for %s/%s → skipped", q.size, c, target, name)
            continue
        ids, sq, yq = ftH.ids[q], ftH.subjects[q], yH[q]
        frames = []
        for method, model, Eq, ks in bases(name):
            if model is None:
                LOG.warning("transfer: %s not usable in %s/%s/%s", method, c, target, name)
                continue
            P, _ = knn_predict(model, Eq[q], sq, ks)
            frames += [_pred_frame(ids, sq, yq, method, k, P[k]) for k in ks]
        if source == "inspire":
            yl = y[iL][y[iL] >= 0]
            prior = np.bincount(yl, minlength=C) / max(yl.size, 1)
            frames.append(_pred_frame(ids, sq, yq, "L_prevalence", 0, np.tile(prior, (q.size, 1))))
        if frames:
            path = out_dir / f"{c}_{source}_{name}.parquet"
            write_parquet(pl.concat(frames, how="diagonal_relaxed"), path)
            files[name] = str(path)
    return files, info


# --------------------------------------------------------------------------- report
_LEGEND = [
    ["A_transport", "INSPIRE model with the INSPIRE case base (full transport, mode A on the held-out patients)"],
    ["T_local", "INSPIRE weights and shapes with the local 80 % as case base (transfer of the metric)"],
    ["T_pooled", "INSPIRE weights and shapes with INSPIRE ∪ local 80 % as case base (local cases added)"],
    ["L_perisim", "PERISIM fitted on the local 80 % alone (a kNN learned in that dataset)"],
    ["L_euclidean", "unsupervised Euclidean kNN (neutral axes) on the local 80 %"],
    ["L_prevalence", "class prevalence of the local 80 %"],
    ["S_transport", "PERISIM refitted on INSPIRE ∪ eICU, with that case base (MIMIC-IV only)"],
    ["S_local", "INSPIRE ∪ eICU weights and shapes with the local 80 % as case base"],
    ["S_pooled", "INSPIRE ∪ eICU weights and shapes with INSPIRE ∪ eICU ∪ local 80 % as case base"],
]


def render_transfer(out: dict, kp: int, n_boot: int = 2000) -> str:
    f = out.get("holdout_fraction", 0.2)
    sec = [box(f"Model-transfer validations. In each external cohort {100 * (1 - f):.0f} % of the patients form the local "
               f"case base and {100 * f:.0f} % are held out; every variant predicts the same held-out stays with the same "
               "labels (target-cohort cut-points estimated on the local cases only). Contrasts: Δ log-loss a − b at the "
               f"primary k = {kp} (negative = a better), paired patient-level bootstrap."),
           table(["Variant", "Meaning"], _LEGEND)]
    for t, by_c in out["targets"].items():
        sec.append(f"<h2>{t.upper()}</h2>")
        for c, rc in by_c.items():
            for src, inf in rc["info"].items():
                rows = [[k, v] for k, v in inf.items() if not isinstance(v, dict)]
                for mk in ("local_model", "multisource_model"):
                    if isinstance(inf.get(mk), dict):
                        rows += [[f"{mk}.{k}", v] for k, v in inf[mk].items() if k != "weights"]
                sec.append(table([f"{c} — {src}", "value"], rows))
            for name, ra in rc["analyses"].items():
                sec.append(f"<h3>{c} — {name} labels (n={ra['n_queries']} held-out stays, target {rc['eval_target']})</h3>")
                sec.extend(eval_sections(ra["evaluation"], n_boot))
                if ra.get("contrasts"):
                    sec.append(table(["Contrast", "Δ log-loss", "CI low", "CI high", "p", "common stays"],
                                     [[k, v["mean"], v["ci_low"], v["ci_high"], v.get("p_two_sided"), v["n_common"]]
                                      for k, v in ra["contrasts"].items()], nd=4))
    return page("PERISIM model-transfer validations", sec, "PERISIM · §6.2b · aggregates only")
