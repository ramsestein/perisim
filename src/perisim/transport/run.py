"""Orchestration of the transport stage: frozen-artefact hashes, phase 1 (fits) and phase 2 (per-cohort analyses) in
parallel processes, E (between-site homogeneity of the shapes) and H6, and the outputs: CSV tables and figures in
<workdir>/experiments/<t>/results/transport/, artifacts/transport.json and reports/transport.html (aggregates only)."""

from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing as mp
import shutil
import traceback

import numpy as np
import polars as pl
from scipy.stats import chi2, fisher_exact

from perisim.evaluation.html import box, page, table
from perisim.transport.core import check_hashes, frozen_hashes, variable_class
from perisim.utils import set_threads, write_json

LOG = logging.getLogger("perisim.transport")


# --------------------------------------------------------------------------- jobs
def transport_jobs(cfg, available: list[str], targets: list[str] | None = None) -> dict[str, list[tuple]]:
    """Phase-1 fits and phase-2 analyses as (kind, target, cohort-or-site) jobs."""
    tc = cfg.experiment.get("transport") or {}
    dev = cfg.experiment["cohorts"]["development"]
    reduced = set(cfg.targets["t1"].get("reduced_cohorts") or [])
    externals = [c for c in cfg.experiment["cohorts"]["external"] if c in available]
    ms = tc.get("multisource") or {}
    fits, analyses = [], []
    for t in targets or cfg.targets_to_model:
        cohorts = [c for c in externals if not (t == "t3" and c in reduced)]   # no full T1 there: T3 not evaluable
        for c in cohorts:
            analyses.append(("analysis", t, c))
            if tc.get("local_model", True):
                fits.append(("local", t, c))
        if t in (ms.get("targets") or []) and all(x in available for x in ms.get("development") or []):
            fits += [("multisource", t, c) for c in (ms.get("test") or []) if c in cohorts]
        fits += [("shapes", t, s) for s in [dev] + [c for c in cohorts if not (t == "t1" and c in reduced)]]
    return {"fits": fits, "analyses": analyses}


def _job(args: dict) -> dict:
    kind, t, c = args["kind"], args["target"], args["cohort"]
    try:
        from perisim.config import Config
        from perisim.transport import analyses as A
        from perisim.utils import setup_logging
        from perisim.workspace import Workspace

        setup_logging()
        set_threads(int(args["threads"]))
        cfg = Config(**args["cfg"], profile=args["profile"], variant="main")
        ws = Workspace(args["base"], experiment=args["experiment"])
        fn = {"local": A.fit_local, "multisource": A.fit_multisource, "shapes": A.fit_site_shapes,
              "analysis": A.cohort_analyses}[kind]
        return {"kind": kind, "target": t, "cohort": c, "result": fn(cfg, ws, t, c, args)}
    except Exception:
        return {"kind": kind, "target": t, "cohort": c, "error": traceback.format_exc()}


def _run_jobs(pipe, common: dict, jobs: list[tuple]) -> list[dict]:
    from perisim.pipeline import _child_threads

    args = [{**common, "kind": k, "target": t, "cohort": c} for k, t, c in jobs]
    par = max(1, min(len(args), pipe.jobs()))
    LOG.info("transport: %d job(s) in %d process(es) × %d threads: %s", len(args), par, pipe.threads(), jobs)
    if par == 1:
        out = [_job(a) for a in args]
    else:
        out = []
        with _child_threads(pipe.threads()), \
                cf.ProcessPoolExecutor(max_workers=par, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_job, a): a for a in args}
            for fu in cf.as_completed(futs):
                try:
                    out.append(fu.result())
                except Exception as exc:  # worker died (e.g. OOM)
                    a = futs[fu]
                    out.append({"kind": a["kind"], "target": a["target"], "cohort": a["cohort"],
                                "error": f"process terminated ({exc!r})"})
    errors = [f"{r['kind']}/{r['target']}/{r['cohort']}: {r['error']}" for r in out if r.get("error")]
    if errors:
        raise RuntimeError("Transport analysis failed:\n" + "\n".join(errors))
    return out


# --------------------------------------------------------------------------- stage
def run_transport(pipe) -> None:
    cfg, ws = pipe.cfg, pipe.ws
    tc = cfg.experiment.get("transport") or {}
    if not tc.get("enabled", True):
        LOG.info("transport: disabled (experiment.transport.enabled)")
        return
    targets = list(cfg.targets_to_model)
    missing = [t for t in targets if not (ws.predictions("external", t) / "final_model.pkl").exists()]
    if missing:
        raise RuntimeError(f"transport: no saved final model for {missing}; rerun the external stage "
                           "(--stage external --force) with this code version")
    before = frozen_hashes(ws, targets)
    key = pipe.key("transport")
    for t in targets:          # phase-1 outputs of another stage key (other code or config) are never reused
        d = ws.predictions("transport", t)
        if (d / "KEY").exists() and (d / "KEY").read_text().strip() != key:
            shutil.rmtree(d, ignore_errors=True)
        ws.predictions("transport", t).mkdir(parents=True, exist_ok=True)
        (ws.predictions("transport", t) / "KEY").write_text(key)
    jobs = transport_jobs(cfg, pipe.available_cohorts(), targets)
    common = {"cfg": cfg.to_dict(), "profile": cfg.profile, "base": str(ws.base), "experiment": ws.experiment,
              "names": pipe.feature_names(), "d7": pipe.d7(), "threads": pipe.threads(),
              "dev": cfg.experiment["cohorts"]["development"],
              "externals": [c for c in cfg.experiment["cohorts"]["external"] if c in pipe.available_cohorts()],
              "eicu_src": (pipe.disc.get("eicu") or {}).get("path")}
    todo = [j for j in jobs["fits"] if not _fit_file(ws, j).exists()]
    if todo:
        _run_jobs(pipe, common, todo)
    res = _run_jobs(pipe, common, jobs["analyses"])
    out = _aggregate(cfg, ws, targets, {(r["target"], r["cohort"]): r["result"] for r in res}, common["names"])
    after = frozen_hashes(ws, targets)
    check_hashes(before, after)
    out["frozen_hashes"] = {"start": before, "end": after, "unchanged": True}
    write_json(ws.artifacts() / "transport.json", out)
    write_json(_results_dir(ws) / "frozen_hashes.json", out["frozen_hashes"])
    (ws.reports / "transport.html").write_text(render_transport(out), encoding="utf-8")
    LOG.info("transport: results in %s", _results_dir(ws))


def _fit_file(ws, job):
    kind, t, c = job
    name = {"local": f"{c}_local_model.pkl", "multisource": f"{c}_multisource_model.pkl",
            "shapes": f"shapes_{c}.pkl"}[kind]
    return ws.predictions("transport", t) / name


def _results_dir(ws):
    d = ws.root / "results" / "transport"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- aggregation
def _aggregate(cfg, ws, targets: list[str], res: dict, names: list[str]) -> dict:
    from perisim.transport.analyses import _load

    status = {t: ("confirmatory" if t == "t2" else "descriptive") for t in targets}
    out: dict = {"targets": {}, "status_of_hypotheses": status,
                 "note": "Transport analyses (SPECS §6.2b, amendment of 2026-10-24); A–F secondary, H1–H6 "
                         "confirmatory for T2 only."}
    rd = _results_dir(ws)
    for t in targets:
        per = {c: r for (tt, c), r in res.items() if tt == t}
        tables = {k: [] for k in ("A", "B", "C", "D", "F")}
        for c, r in per.items():
            for k in tables:
                tables[k] += [{"cohort": c, **row} for row in r.get(k, [])]
        for k, rows in tables.items():
            if rows:
                _csv(rows, rd / f"{t}_{k}_{_TABLE_NAMES[k]}.csv")
        sites = {}
        for f in sorted(ws.predictions("transport", t).glob("shapes_*.pkl")):
            s = _load(f)
            sites[s["site"]] = s
        E = shape_homogeneity(sites, cfg.experiment["cohorts"]["development"],
                              float(cfg.experiment["transport"]["homogeneity_alpha"]))
        cov = coverage_missingness(ws, cfg, names)
        if E["rows"]:
            _csv(E["rows"], rd / f"{t}_E_shape_homogeneity.csv")
        if cov:
            _csv(cov, rd / f"{t}_E_coverage_missingness.csv")
        H = {c: r["hypotheses"] for c, r in per.items()}
        H6 = h6(E)
        hyp_rows = _hypothesis_rows(t, status[t], H, H6)
        _csv(hyp_rows, rd / f"{t}_hypotheses.csv")
        het = hospital_heterogeneity(tables["F"], tables["B"])
        figs = _figures(t, per, tables, E, rd)
        out["targets"][t] = {"cohorts": {c: {k: v for k, v in r.items() if k not in ("C", "F")} for c, r in per.items()},
                             "accumulation_summary": _accumulation_summary(tables["C"]),
                             "shape_homogeneity": {k: v for k, v in E.items() if k != "rows"},
                             "shape_homogeneity_failing": [x for x in E["rows"] if x.get("fails")],
                             "hypotheses": hyp_rows, "H6": H6, "hospital_heterogeneity": het, "figures": figs}
    return out


_TABLE_NAMES = {"A": "decomposition", "B": "composition", "C": "accumulation", "D": "recalibration", "F": "hospitals"}


def _csv(rows: list[dict], path) -> None:
    flat = [{k: (v if not isinstance(v, (dict, list)) else str(v)) for k, v in r.items()} for r in rows]
    pl.DataFrame(flat, infer_schema_length=None, strict=False).write_csv(path)


def shape_homogeneity(sites: dict, dev: str, alpha: float = 0.05) -> dict:
    """E: between-site homogeneity of each shaped variable and class, on INSPIRE's value grid. Each site's curves are
    first divided by ONE site-level scale (the median over variables of the weighted least-squares slope of the site's
    curve on INSPIRE's): a uniform shrinkage of all effects does not change the metric, so it is not a failure to
    transport, and the median keeps a few non-transporting variables from biasing the scale. Then
    Cochran's Q over the sites, summed over the values (df = Σ (sites − 1)), I², Bonferroni over variables × classes.
    Only sites with the same classes as the development cohort enter."""
    if dev not in sites:
        return {"rows": [], "sites": list(sites), "note": "no development shapes"}
    ref = sites[dev]
    use = {s: v for s, v in sites.items() if v["C"] == ref["C"]}
    names = sorted(set.intersection(*[set(v["shapes"]) for v in use.values()])) if use else []
    scale = {dev: 1.0}
    for s, v in use.items():
        if s == dev:
            continue
        slopes = []
        for n in names:            # per-variable WLS slope against INSPIRE; the site scale is their median, so a few
            a, b = v["shapes"][n], ref["shapes"][n]       # variables that do not transport cannot bias it
            w = 1.0 / np.maximum(a["se2"] + b["se2"], 1e-12)
            den = float(np.nansum(w * b["mean"] ** 2))
            if den > 0:
                slopes.append(float(np.nansum(w * a["mean"] * b["mean"])) / den)
        scale[s] = float(np.median(slopes)) if slopes else 1.0
    rows = []
    n_tests = max(1, len(names) * max(ref["C"], 1))
    for n in names:
        for c in range(ref["C"]):
            Y = np.array([use[s]["shapes"][n]["mean"][:, c] / (scale[s] or 1.0) for s in use])
            V = np.array([use[s]["shapes"][n]["se2"][:, c] / (scale[s] or 1.0) ** 2 for s in use])
            Q = df = 0.0
            for j in range(Y.shape[1]):
                ok = np.isfinite(Y[:, j]) & np.isfinite(V[:, j])
                if ok.sum() < 2:
                    continue
                w = 1.0 / np.maximum(V[ok, j], 1e-12)
                m = float((w * Y[ok, j]).sum() / w.sum())
                Q += float((w * (Y[ok, j] - m) ** 2).sum())
                df += int(ok.sum() - 1)
            if df == 0:
                continue
            p = float(chi2.sf(Q, df))
            dev_site = {f"rms_dev_{s}": float(np.sqrt(np.nanmean((Y[i] - Y[list(use).index(dev)]) ** 2)))
                        for i, s in enumerate(use) if s != dev}
            rows.append({"variable": n, "class": c, "variable_class": variable_class(n), "Q": Q, "df": int(df),
                         "p": p, "I2": max(0.0, (Q - df) / Q) if Q > 0 else 0.0,
                         "fails": bool(p < alpha / n_tests), **dev_site})
    return {"rows": rows, "sites": list(use), "scale": scale, "n_tests": n_tests, "alpha": alpha,
            "excluded_sites": [s for s in sites if s not in use]}


def h6(E: dict) -> dict:
    """H6: among the variables failing the homogeneity test (any class), the practice + coding share is > 50 % and
    larger than among the variables that pass (one-sided Fisher exact test); physiological vs practice + coding."""
    by_var: dict[str, dict] = {}
    for r in E.get("rows", []):
        v = by_var.setdefault(r["variable"], {"cls": r["variable_class"], "fails": False})
        v["fails"] = v["fails"] or r["fails"]
    cnt = {(f, k): 0 for f in (True, False) for k in ("pc", "phys")}
    for v in by_var.values():
        if v["cls"] in ("practice", "coding"):
            cnt[(v["fails"], "pc")] += 1
        elif v["cls"] == "physiological":
            cnt[(v["fails"], "phys")] += 1
    nf = cnt[(True, "pc")] + cnt[(True, "phys")]
    npass = cnt[(False, "pc")] + cnt[(False, "phys")]
    if nf == 0:
        return {"holds": None, "note": "no variable fails the homogeneity test", "n_failing": 0,
                "n_passing": npass}
    _, p = fisher_exact([[cnt[(True, "pc")], cnt[(True, "phys")]], [cnt[(False, "pc")], cnt[(False, "phys")]]],
                        alternative="greater")
    share = cnt[(True, "pc")] / nf
    return {"holds": bool(share > 0.5 and p < 0.05), "share_practice_coding_failing": share,
            "share_practice_coding_passing": cnt[(False, "pc")] / npass if npass else None, "fisher_p": float(p),
            "n_failing": nf, "n_passing": npass}


def coverage_missingness(ws, cfg, names: list[str]) -> list[dict]:
    """E: per variable and external site, % of values outside INSPIRE's observed range (where every g_j is constant)
    and the missingness of the variable in each cohort."""
    from perisim.experiment import load_cohort_data

    dev = cfg.experiment["cohorts"]["development"]
    data = {}
    for c in [dev] + list(cfg.experiment["cohorts"]["external"]):
        try:
            data[c] = load_cohort_data(ws.common(c), ws.derived(c), names)
        except Exception:
            continue
    if dev not in data:
        return []
    ref = data[dev].ft
    rows = []
    for spec in ref.specs:
        n = spec.name
        r = {"variable": n, "variable_class": variable_class(n), "type": spec.type,
             f"missing_pct_{dev}": _missing(ref, n)}
        lo = hi = None
        if n in ref.num:
            v = ref.num[n][np.isfinite(ref.num[n])]
            lo, hi = (float(v.min()), float(v.max())) if v.size else (None, None)
        for c, d in data.items():
            if c == dev or n not in d.ft.num and n not in d.ft.cat:
                continue
            r[f"missing_pct_{c}"] = _missing(d.ft, n)
            if lo is not None and n in d.ft.num:
                x = d.ft.num[n][np.isfinite(d.ft.num[n])]
                r[f"outside_range_pct_{c}"] = float(((x < lo) | (x > hi)).mean() * 100) if x.size else None
        rows.append(r)
    return rows


def _missing(ft, n) -> float:
    if n in ft.num:
        return float((~np.isfinite(ft.num[n])).mean() * 100)
    v = ft.cat[n]
    return float(np.mean([x is None or (isinstance(x, float) and np.isnan(x)) for x in v]) * 100)


def hospital_heterogeneity(F: list[dict], B: list[dict]) -> dict:
    """F: spread across eICU hospitals (local case base and INSPIRE case base) vs the country-level difference."""
    out = {}
    for comp in ("local", "inspire_only"):
        rr = [r for r in F if r.get("analysis") == "B" and r.get("composition") == comp and r.get("gamma") == "1"]
        if len(rr) >= 2:
            out[comp] = {m: {"sd_across_hospitals": float(np.nanstd([r[m] for r in rr], ddof=1)),
                             "range": [float(np.nanmin([r[m] for r in rr])), float(np.nanmax([r[m] for r in rr]))]}
                         for m in ("logloss", "citl_ge1", "auroc_macro") if all(m in r for r in rr)}
            out[comp]["n_hospitals"] = len(rr)
    for c in sorted({r["cohort"] for r in B}):
        loc = [r for r in B if r["cohort"] == c and r["labels"] == "own" and r["composition"] == "local"]
        ins = [r for r in B if r["cohort"] == c and r["labels"] == "own" and r["composition"] == "inspire_only"]
        if loc and ins:
            out[f"country_{c}"] = {"logloss_modeA_minus_local": ins[0]["logloss"] - loc[0]["logloss"],
                                   "citl_ge1_modeA_minus_local": ins[0].get("citl_ge1", np.nan) - loc[0].get("citl_ge1",
                                                                                                             np.nan)}
    return out


def _accumulation_summary(C: list[dict]) -> list[dict]:
    """Mean and 2.5–97.5 % interval over the repetitions of the main metrics, per curve point."""
    if not C:
        return []
    df = pl.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in C],
                      infer_schema_length=None, strict=False)
    keys = ["cohort", "order", "start", "gamma", "n_local"]
    mets = [m for m in ("logloss", "brier", "citl_ge1", "slope_ge1", "auroc_worst", "ood_pct") if m in df.columns]
    agg = []
    for m in mets:
        agg += [pl.col(m).mean().alias(f"{m}_mean"), pl.col(m).quantile(0.025).alias(f"{m}_low"),
                pl.col(m).quantile(0.975).alias(f"{m}_high")]
    return df.group_by(keys).agg(agg + [pl.len().alias("reps")]).sort(keys).to_dicts()


def _hypothesis_rows(t: str, status: str, H: dict, H6: dict) -> list[dict]:
    rows = []
    for c, h in H.items():
        for k in ("H1", "H2", "H3", "H4", "H5"):
            if k in h:
                v = h[k]
                rows.append({"target": t, "hypothesis": k, "cohort": c, "status": status, "holds": v.get("holds"),
                             "summary": _short(k, v)})
    rows.append({"target": t, "hypothesis": "H6", "cohort": "all sites", "status": status, "holds": H6.get("holds"),
                 "summary": str({k: v for k, v in H6.items() if k != "holds"})})
    return rows


def _short(k: str, v: dict) -> str:
    if k == "H1":
        return f"local CITL inside ±{v['margin']}: {v['local_inside_margin']}; mode A beyond: {v['modeA_beyond_margin']}"
    if k == "H2":
        return "slope CI low (local): " + ", ".join(f"{s['ci_low']:.3f}" for s in v["slopes_local"])
    if k == "H3":
        d = v["auroc_difference_frozen_minus_local"]
        return f"AUROC frozen − local = {d['mean']:.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}], margin −{v['margin']}"
    if k == "H4":
        d = v["ood_modeA_minus_local"]
        return f"OOD mode A − local = {d['mean']:.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"
    if k == "H5":
        return f"crossing at n_local = {v.get('crossing_n_local')}; schedule − empty: {v.get('schedule_minus_empty')}"
    return ""


# --------------------------------------------------------------------------- figures and report
def _figures(t: str, per: dict, tables: dict, E: dict, rd) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    summ = _accumulation_summary(tables["C"])
    for c in sorted({r["cohort"] for r in summ}):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for start, gamma, lab in (("empty", "1", "empty start"), ("inspire", "1", "INSPIRE start, γ = 1"),
                                  ("inspire", "schedule", "INSPIRE start, γ(n)"),
                                  ("inspire+other", "1", "INSPIRE ∪ other, γ = 1")):
            rr = sorted([r for r in summ if r["cohort"] == c and r["order"] == "random" and r["start"] == start
                         and r["gamma"] == gamma and r["n_local"] > 0], key=lambda r: r["n_local"])
            if rr:
                x = [r["n_local"] for r in rr]
                ax.plot(x, [r["logloss_mean"] for r in rr], marker="o", label=lab)
                ax.fill_between(x, [r["logloss_low"] for r in rr], [r["logloss_high"] for r in rr], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("local patients in the case base")
        ax.set_ylabel("log-loss (test)")
        ax.set_title(f"{c}: accumulation (random order)")
        ax.legend(fontsize=8)
        figs[f"accumulation_{c}"] = _save_fig(fig, rd / f"{t}_fig_accumulation_{c}.png")
    A = [r for r in tables["A"] if not r["variant"].startswith("Δ")]
    if A:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        cs = sorted({r["cohort"] for r in A})
        vs = list(dict.fromkeys(r["variant"] for r in A))
        wdt = 0.8 / max(len(vs), 1)
        for i, v in enumerate(vs):
            rr = [next((r for r in A if r["cohort"] == c and r["variant"] == v), None) for c in cs]
            xs = [j + i * wdt for j, r in enumerate(rr) if r]
            ys = [r["logloss"] for r in rr if r]
            err = [[r["logloss"] - r["logloss_ci_low"] for r in rr if r], [r["logloss_ci_high"] - r["logloss"] for r in rr if r]]
            ax.bar(xs, ys, width=wdt, yerr=err, label=v, capsize=2)
        ax.set_xticks([j + 0.4 - wdt / 2 for j in range(len(cs))], cs)
        ax.set_ylabel("log-loss (test)")
        ax.set_title("Metric vs case base: decomposition of the transport loss")
        ax.legend(fontsize=7)
        figs["decomposition"] = _save_fig(fig, rd / f"{t}_fig_decomposition.png")
    if E.get("rows"):
        devcols = sorted({k for r in E["rows"] for k in r if k.startswith("rms_dev_")})
        names = sorted({r["variable"] for r in E["rows"]})
        M_ = np.full((len(names), len(devcols)), np.nan)
        fails = {r["variable"] for r in E["rows"] if r["fails"]}
        for r in E["rows"]:
            for j, k in enumerate(devcols):
                i = names.index(r["variable"])
                M_[i, j] = np.nanmax([M_[i, j], r.get(k, np.nan)])
        fig, ax = plt.subplots(figsize=(3 + 1.2 * len(devcols), 0.25 * len(names) + 1.5))
        im = ax.imshow(M_, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(names)), [("✗ " if n in fails else "") + n for n in names], fontsize=7)
        ax.set_xticks(range(len(devcols)), [k.replace("rms_dev_", "") for k in devcols])
        fig.colorbar(im, ax=ax, label="RMS deviation from INSPIRE (scaled)")
        ax.set_title("Shape homogeneity (✗ = fails)")
        figs["shape_homogeneity"] = _save_fig(fig, rd / f"{t}_fig_shape_homogeneity.png")
    for c, r in per.items():
        cc = r.get("calibration_curves") or {}
        if not cc:
            continue
        thr = [x["threshold"] for x in cc.get("local", [])]
        fig, axes = plt.subplots(1, len(thr), figsize=(3.2 * len(thr), 3.2), squeeze=False)
        for ax, th in zip(axes[0], thr):
            for mode, lab in (("local", "local case base"), ("mode_A", "INSPIRE case base")):
                pts = next((x["points"] for x in cc.get(mode, []) if x["threshold"] == th), [])
                if pts:
                    ax.plot([p["p_mean"] for p in pts], [p["obs"] for p in pts], marker="o", label=lab)
            ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
            ax.set_title(f"P(Y ≥ {th})")
            ax.set_xlabel("predicted")
            ax.set_ylabel("observed")
        if axes[0][0].get_legend_handles_labels()[0]:          # no curve has enough events in a small cohort
            axes[0][0].legend(fontsize=7)
        fig.suptitle(f"{c}: calibration by mode")
        figs[f"calibration_{c}"] = _save_fig(fig, rd / f"{t}_fig_calibration_{c}.png")
    return figs


def _save_fig(fig, path) -> str:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def render_transport(out: dict) -> str:
    sec = [box("Transport analyses (SPECS §6.2b, pre-registration amendment of 2026-10-24). The frozen geometry (g_j, "
               "w_j) of the INSPIRE model is used with case bases of different composition and size, on a fixed "
               "patient-level test set of each external cohort. H1–H6 are confirmatory for T2 and descriptive for "
               "T1 / T3; A–F are secondary. Frozen artefacts: hashes identical at the start and the end.")]
    for t, r in out["targets"].items():
        sec.append(f"<h2>{t.upper()}</h2>")
        sec.append("<h3>Hypotheses</h3>" + table(["Hypothesis", "Cohort", "Status", "Holds", "Summary"],
                                                  [[h["hypothesis"], h["cohort"], h["status"], h["holds"], h["summary"]]
                                                   for h in r["hypotheses"]]))
        for c, rc in r["cohorts"].items():
            sec.append(f"<h3>{c} (target {rc['eval_target']})</h3>")
            sec.append(table(["Set", "stays", "patients"], [[k, v["stays"], v["patients"]] for k, v in rc["sizes"].items()]))
            if rc.get("A"):
                sec.append("<h4>A. Decomposition of the transport loss</h4>" + table(
                    ["Variant", "log-loss", "CI low", "CI high", "AUROC macro", "CITL P(Y≥1)", "slope P(Y≥1)"],
                    [[a["variant"], a.get("logloss"), a.get("logloss_ci_low"), a.get("logloss_ci_high"),
                      a.get("auroc_macro"), a.get("citl_ge1"), a.get("slope_ge1")] for a in rc["A"]], nd=4))
            b = [x for x in rc.get("B", []) if x["labels"] == "own"]
            if b:
                sec.append("<h4>B. Case-base composition and site weight γ (cross-fitted tertiles)</h4>" + table(
                    ["Composition", "γ", "case base", "log-loss", "Brier", "AUROC macro", "CITL P(Y≥1)", "slope P(Y≥1)",
                     "% out of support"],
                    [[x["composition"], x["gamma"], x["n_case_base"], x["logloss"], x["brier"], x["auroc_macro"],
                      x.get("citl_ge1"), x.get("slope_ge1"), x.get("ood_pct")] for x in b], nd=4))
            if rc.get("D"):
                sec.append("<h4>D. Minimal recalibration</h4>" + table(
                    ["Method", "tuning patients", "k", "α", "log-loss", "CITL P(Y≥1)", "slope P(Y≥1)"],
                    [[d["method"], d["tuning_patients"], d["k"], d["alpha"], d["logloss"], d.get("citl_ge1"),
                      d.get("slope_ge1")] for d in rc["D"]], nd=4))
        for name, f in (r.get("figures") or {}).items():
            sec.append(f"<p>Figure <code>{f}</code> (results/transport/)</p>")
        fails = r.get("shape_homogeneity_failing") or []
        sec.append("<h3>E. Shape homogeneity between sites</h3>" + box(
            f"Sites: {r['shape_homogeneity'].get('sites')}; site scale vs INSPIRE: {r['shape_homogeneity'].get('scale')}; "
            f"{len(fails)} variable × class tests fail (Bonferroni over {r['shape_homogeneity'].get('n_tests')})."))
        if fails:
            sec.append(table(["Variable", "class", "variable class", "Q", "df", "p", "I²"],
                             [[x["variable"], x["class"], x["variable_class"], x["Q"], x["df"], x["p"], x["I2"]]
                              for x in fails], nd=4))
        if r.get("hospital_heterogeneity"):
            sec.append("<h3>F. eICU by hospital vs countries</h3>" + box(str(r["hospital_heterogeneity"])))
    return page("PERISIM transport analyses", sec, "PERISIM · §6.2b · aggregates only")
