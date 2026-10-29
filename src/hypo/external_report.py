"""Aggregate report of the external stage: <workdir>/reports/external.md / .json (docs/prereg_hypo.md §6–§7).

H1–H5 with their pre-registered decision rules (2,000 patient-level bootstrap resamples, the same for every
statistic of a site), then the secondary analyses (500 resamples). Aggregates only: hospitals labelled H1, H2, … by
size; cells of fewer than `min_cell` landmarks are not shown.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from .config import workdir
from .cv import Sites, cv_dir
from .external import hospital_labels, xsettings
from .frozen import check, external_dir
from .report import auc_prep, auc_weighted
from .stats import PatientBootstrap, calibration_line_w, ci, citl_w, logit, logloss_rows

LOG = logging.getLogger("hypo.external_report")


_SRC = {"dir": None}          # the units' folder: external/ (default) or the dry run's


def _dir():
    return _SRC["dir"] or external_dir()


def _units(pattern: str) -> pl.DataFrame | None:
    fs = sorted(_dir().glob(pattern))
    parts = [pl.read_parquet(f) for f in fs]
    parts = [p for p in parts if p.height]
    return pl.concat(parts, how="diagonal_relaxed").sort("row") if parts else None


def _phase2(c: str, name: str) -> pl.DataFrame | None:
    fs = sorted(cv_dir(False).glob(f"comp_{c}_f*_{name}.parquet"))
    return pl.concat([pl.read_parquet(f) for f in fs]).sort("row") if fs else None


def _on(df: pl.DataFrame | None, rows: np.ndarray, col: str) -> np.ndarray | None:
    """Column `col` of a unit table on the landmark rows `rows` (NaN where the unit has no prediction)."""
    if df is None or col not in df.columns:
        return None
    out = np.full(rows.size, np.nan)
    r = df["row"].to_numpy()
    pos = np.searchsorted(rows, r)
    ok = (pos < rows.size) & (rows[np.minimum(pos, rows.size - 1)] == r)
    out[pos[ok]] = df[col].to_numpy().astype(np.float64)[ok]
    return out


def _json_units(pattern: str) -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(_dir().glob(pattern))]


class Boot:
    """Statistics of one site on the same patient resamples: log-losses (per-row losses), AUROCs, calibration."""

    def __init__(self, subjects: np.ndarray, n_boot: int, seed: int):
        self.b = PatientBootstrap(subjects, n_boot, seed)

    def run(self, idx: np.ndarray, y: np.ndarray, *, loss: dict | None = None, auc: dict | None = None,
            calib: dict | None = None, d_loss: list | None = None, d_auc: list | None = None,
            level: float = 0.95) -> dict:
        loss, auc, calib = loss or {}, auc or {}, calib or {}
        d_loss, d_auc = d_loss or [], d_auc or []
        prep = {m: auc_prep(p) for m, p in auc.items()}
        x = {m: logit(p) for m, p in calib.items()}
        yf = y.astype(np.float64)
        one = np.ones(len(y))

        def stats(w):
            s = {f"ll:{m}": float((w * v).sum() / w.sum()) for m, v in loss.items()}
            s.update({f"auc:{m}": auc_weighted(y, w, prep[m]) for m in auc})
            for m in calib:
                s[f"citl:{m}"] = citl_w(x[m], yf, w)
                s[f"slope:{m}"] = calibration_line_w(x[m], yf, w)[1]
            return s

        point = stats(one)
        B = {k: [] for k in point}
        for wp in self.b:
            for k, v in stats(self.b.rows(wp, idx)).items():
                B[k].append(v)
        B = {k: np.asarray(v) for k, v in B.items()}
        out = {k: {"est": v, "ci": ci(B[k], level)} for k, v in point.items()}
        for kind, pairs in (("ll", d_loss), ("auc", d_auc)):
            for a, b in pairs:
                out[f"{kind}:{a} − {b}"] = {"est": point[f"{kind}:{a}"] - point[f"{kind}:{b}"],
                                            "ci": ci(B[f"{kind}:{a}"] - B[f"{kind}:{b}"], level)}
        return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(p)
    return auc_weighted(y[ok], np.ones(int(ok.sum())), auc_prep(p[ok])) if ok.any() else float("nan")


def site_report(X: str, S: Sites, cfg: dict, x: dict) -> dict | None:
    T = _units(f"transported_{X}_f*.parquet")
    if T is None:
        return None
    lm = S.lm(X)
    rows = T["row"].to_numpy()
    y = lm.y[rows]
    subj = lm.ft.subjects[rows]
    seed = int(cfg["seed"]) + 77 * cfg["cohorts"].index(X)
    out = {"n_landmarks": int(rows.size), "n_patients": int(np.unique(subj.astype(str)).size),
           "prevalence": float(y.mean()), "folds": sorted({int(f) for f in S.fold(X)[rows]})}
    pT = T["p"].to_numpy()
    Lu = _units(f"local_{X}_f*.parquet")
    pL = _on(Lu, rows, "p")
    pLM = _on(_phase2(X, "last_map"), rows, "p")
    # ---------------------------------------------------------------- H1–H3
    ok = np.isfinite(pT) & (np.isfinite(pL) if pL is not None else False) & \
        (np.isfinite(pLM) if pLM is not None else False)
    prim = {}
    if np.any(ok):
        idx = np.where(ok)[0]
        bp = Boot(subj, int(x["n_boot_primary"]), seed)
        r = bp.run(idx, y[idx], loss={"T": logloss_rows(pT[idx], y[idx]), "LM": logloss_rows(pLM[idx], y[idx])},
                   auc={"T": pT[idx], "L": pL[idx]}, calib={"T": pT[idx]}, d_loss=[("T", "LM")], d_auc=[("T", "L")])
        h1, h3 = r["auc:T − L"], r["ll:T − LM"]
        citl, slope = r["citl:T"], r["slope:T"]
        prim = {"n": int(idx.size),
                "H1": {**h1, "auc_transported": r["auc:T"], "auc_local": r["auc:L"],
                       "holds": bool(h1["ci"][0] > -float(x["h1_margin"]))},
                "H2": {"citl": citl, "slope": slope,
                       "holds": bool(-float(x["h2_citl"]) < citl["ci"][0] and citl["ci"][1] < float(x["h2_citl"])
                                     and slope["ci"][0] >= float(x["h2_slope"]))},
                "H3": {**h3, "holds": bool(h3["ci"][1] < 0)}}
        # H5 (sites with hospitals)
        gam = [float(g) for g in x["gamma"]]
        if all(f"p_gamma{g:g}" in T.columns for g in gam) and "p_gamma1" in T.columns:
            p1 = T["p_gamma1"].to_numpy()
            idx_all = np.arange(rows.size)
            loss = {"g1": logloss_rows(p1, y)} | {f"g{g:g}": logloss_rows(T[f"p_gamma{g:g}"].to_numpy(), y)
                                                  for g in gam}
            lvl = 1 - 0.05 / len(gam)
            r5 = bp.run(idx_all, y, loss=loss, d_loss=[(f"g{g:g}", "g1") for g in gam], level=lvl)
            prim["H5"] = {"level": lvl, "gamma": {f"{g:g}": r5[f"ll:g{g:g} − g1"] for g in gam},
                          "holds": bool(any(r5[f"ll:g{g:g} − g1"]["ci"][1] < 0 for g in gam))}
        # H4
        Hs = _units(f"hospitals_{X}_f*.parquet")
        if Hs is not None:
            pos = {int(rw): i for i, rw in enumerate(rows)}
            res4 = {}
            for kind in ("hospital", "control"):
                h = Hs.filter(pl.col("kind") == kind)
                ii = np.array([pos[int(v)] for v in h["row"].to_numpy()])
                res4[kind] = (ii, (h["ll_own"] - h["ll_other"]).to_numpy())
            # Δ_hosp, Δ_ctrl and their difference on the same resamples
            iH, dH = res4["hospital"]
            iC, dC = res4["control"]
            Bh, Bc = [], []
            for wp in bp.b:
                wh, wc = bp.b.rows(wp, iH), bp.b.rows(wp, iC)
                Bh.append(float((wh * dH).sum() / wh.sum()))
                Bc.append(float((wc * dC).sum() / wc.sum()))
            Bh, Bc = np.asarray(Bh), np.asarray(Bc)
            dh, dc = {"est": float(dH.mean()), "ci": ci(Bh)}, {"est": float(dC.mean()), "ci": ci(Bc)}
            dd = {"est": dh["est"] - dc["est"], "ci": ci(Bh - Bc)}
            prim["H4"] = {"delta_hospital": dh, "delta_control": dc, "difference": dd,
                          "n_hospital_landmarks": int(iH.size), "n_control_landmarks": int(iC.size),
                          "holds": bool(dh["ci"][1] < 0 and dd["ci"][1] < 0)}
            out["hospital_rows"] = Hs
    out["primary"] = prim
    # ---------------------------------------------------------------- secondary
    bs = Boot(subj, int(x["n_boot"]), seed + 1)
    sec = {}
    C = _units(f"comps_{X}_f*.parquet")
    methods = {"PERISIM transported": pT}
    if pL is not None:
        methods["PERISIM local"] = pL
    for name in x["comparators"]:
        v = _on(C, rows, f"p_{name}")
        if v is not None:
            methods[f"{name} transported"] = v
    for name in cfg["cv"]["comparators"]:
        v = _on(_phase2(X, name), rows, "p")
        if v is not None:
            methods[f"{name} local"] = v
    for e in cfg["cohorts"]:                                     # S3: the Phase 2 metric that included X
        if e == X:
            continue
        fs = sorted(cv_dir(False).glob(f"perisim_ext-{e}_f*.parquet"))
        parts = [pl.read_parquet(f).filter(pl.col("site") == X) for f in fs]
        parts = [p for p in parts if p.height]
        if parts:
            partner = [c for c in cfg["cohorts"] if c not in (e, X)][0]
            methods[f"PERISIM with {partner} (Phase 2)"] = _on(pl.concat(parts).sort("row"), rows, "p")
    common = np.all([np.isfinite(v) for v in methods.values()], axis=0)
    idx = np.where(common)[0]
    loss = {m: logloss_rows(v[idx], y[idx]) for m, v in methods.items()}
    auc = {m: v[idx] for m, v in methods.items() if not m.startswith("prevalence")}
    calib = {m: v[idx] for m, v in methods.items() if not m.startswith("prevalence")}
    others = [m for m in methods if m != "PERISIM transported"]
    r = bs.run(idx, y[idx], loss=loss, auc=auc, calib=calib, d_loss=[("PERISIM transported", m) for m in others],
               d_auc=[("PERISIM transported", m) for m in others if m in auc])
    sec["S2"] = {"n": int(idx.size), "methods": {m: {"logloss": r[f"ll:{m}"], "auroc": r.get(f"auc:{m}"),
                                                       "citl": r.get(f"citl:{m}"), "slope": r.get(f"slope:{m}"),
                                                       "mean_p": float(np.mean(methods[m][idx]))}
                                                   for m in methods},
                 "paired": {m: {"logloss": r[f"ll:PERISIM transported − {m}"],
                                "auroc": r.get(f"auc:PERISIM transported − {m}")} for m in others}}
    # recalibrated comparators and S1, as mean over the draws
    Cu = _units(f"curve_{X}_f*.parquet")
    rec_loss, curve_loss, curve_auc = {}, {}, {}
    if C is not None:
        for name in x["comparators"]:
            for n in x["recalibration_n"]:
                cols = [c for c in C.columns if c.startswith(f"p_{name}_n{n}_r")]
                if cols:
                    L_ = np.mean([logloss_rows(_on(C, rows, c), y) for c in cols], axis=0)
                    rec_loss[f"{name} n={n}"] = L_
    if Cu is not None:
        for n in x["curve"]["n"]:
            cols = [c for c in Cu.columns if c.startswith(f"p_n{n}_r")]
            if cols:
                ps = [_on(Cu, rows, c) for c in cols]
                curve_loss[int(n)] = np.mean([logloss_rows(p, y) for p in ps], axis=0)
                curve_auc[int(n)] = float(np.mean([_auc(y, p) for p in ps]))
    ok_c = np.all([np.isfinite(v) for v in list(rec_loss.values()) + list(curve_loss.values())] or
                  [np.ones(rows.size, bool)], axis=0)
    idx_c = np.where(ok_c & np.isfinite(pT))[0]
    if curve_loss or rec_loss:
        lossc = {"whole pool": logloss_rows(pT[idx_c], y[idx_c])} | \
            {f"n={n}": v[idx_c] for n, v in curve_loss.items()} | {k: v[idx_c] for k, v in rec_loss.items()}
        pairs = [(f"n={n}", "whole pool") for n in curve_loss] + \
            [(f"n={n}", f"{name} n={n}") for name in x["comparators"] for n in x["recalibration_n"]
             if n in curve_loss and f"{name} n={n}" in rec_loss]
        rc = bs.run(idx_c, y[idx_c], loss=lossc, d_loss=pairs)
        whole = rc["ll:whole pool"]["est"]
        within = {}
        for m in (0.02, 0.01):
            hit = [n for n in curve_loss if rc[f"ll:n={n}"]["est"] - whole <= m]
            within[str(m)] = min(hit) if hit else None
        sec["S1"] = {"n": int(idx_c.size), "whole_pool": rc["ll:whole pool"],
                     "curve": {str(n): {"logloss": rc[f"ll:n={n}"], "vs_whole": rc[f"ll:n={n} − whole pool"],
                                        "auroc_mean_over_draws": curve_auc[n]} for n in curve_loss},
                     "recalibrated": {k: rc[f"ll:{k}"] for k in rec_loss},
                     "curve_vs_recalibrated": {f"{a} vs {b}": rc[f"ll:{a} − {b}"] for a, b in pairs
                                               if b != "whole pool"},
                     "smallest_n_within": within}
    # S7: arterial-line strata, one reading per hour, bands
    art = lm.frame["art_line"].cast(pl.Float64).to_numpy()[rows]
    s7 = {}
    if np.nanmax(art) > 0:
        strata = {}
        for lab, sel in (("arterial line", art == 1), ("non-invasive only", art == 0)):
            ii = np.where(sel & common)[0]
            if ii.size < int(x["min_cell"]):
                continue
            ms = {m: methods[m][ii] for m in methods if not m.startswith(("prevalence", "PERISIM with"))}
            rr = bs.run(ii, y[ii], loss={m: logloss_rows(v, y[ii]) for m, v in ms.items()}, auc=ms)
            strata[lab] = {"n": int(ii.size), "prevalence": float(y[ii].mean()),
                           "neighbours_same_stratum": float(T["art_share"].to_numpy()[ii].mean()),
                           "methods": {m: {"logloss": rr[f"ll:{m}"], "auroc": rr[f"auc:{m}"]} for m in ms}}
        s7["arterial_line"] = strata
    yf = lm.frame["y_first"].cast(pl.Float64).to_numpy()[rows]
    okf = np.isfinite(yf)
    if okf.any():
        ii = np.where(okf)[0]
        rr = bs.run(ii, yf[ii].astype(int), loss={"first": logloss_rows(T["p_first"].to_numpy()[ii], yf[ii])},
                    auc={"first": T["p_first"].to_numpy()[ii]}, calib={"first": T["p_first"].to_numpy()[ii]})
        s7["one_reading_per_hour"] = {"n": int(ii.size), "prevalence": float(yf[ii].mean()),
                                      **{k.split(":")[0]: v for k, v in rr.items()}}
    band = lm.band[rows]
    okb = band >= 0
    if okb.any() and Lu is not None:
        PbT = np.column_stack([T[f"p_band{b}"].to_numpy() for b in range(3)])
        PbL = np.column_stack([_on(Lu, rows, f"p_band{b}") for b in range(3)])
        ii = np.where(okb & np.isfinite(PbL).all(1))[0]
        llT = -np.log(np.clip(PbT[ii, band[ii]], 1e-15, 1))
        llL = -np.log(np.clip(PbL[ii, band[ii]], 1e-15, 1))
        rr = bs.run(ii, y[ii], loss={"T": llT, "L": llL}, d_loss=[("T", "L")])
        mac = {}
        for nm, Pb in (("transported", PbT), ("local", PbL)):
            mac[nm] = float(np.mean([_auc((band[ii] == b).astype(int), Pb[ii, b]) for b in range(3)]))
        s7["bands"] = {"n": int(ii.size), "logloss_transported": rr["ll:T"], "logloss_local": rr["ll:L"],
                       "difference": rr["ll:T − L"], "macro_auroc": mac,
                       "shares": [float((band[ii] == b).mean()) for b in range(3)]}
    sec["S7"] = s7
    # S8: explanation aggregates, carried outcomes, out of support
    infos = _json_units(f"transported_{X}_f*.json")
    if infos:
        ex = infos[0]["explain"]
        names = ex["variables"]
        sh = np.sum([np.asarray(i["explain"]["share_sum"]) for i in infos], 0)
        shn = np.sum([np.asarray(i["explain"]["share_n"]) for i in infos], 0)
        pu = np.sum([np.asarray(i["explain"]["pull_sum"]) for i in infos], 0)
        pun = np.sum([np.asarray(i["explain"]["pull_n"]) for i in infos], 0)
        share = sh.sum(0) / max(shn.sum(), 1)
        pull = pu.sum(0) / max(pun.sum(), 1)
        top = pu[-1] / max(pun[-1], 1)
        sec["S8"] = {"distance_share": dict(zip(names, share.tolist())), "pull": dict(zip(names, pull.tolist())),
                     "pull_top_decile": dict(zip(names, top.tolist())),
                     "alpha_per_fold": [i["alpha"] for i in infos]}
        car = {}
        for col, nb, kind in (("vaso_start", "nb_vaso_start", "auroc"), ("icu_discharge_24h", "nb_discharge_24h", "auroc"),
                              ("map_min_next", "nb_map_min_next", "spearman")):
            if col not in lm.frame.columns:
                continue
            a = lm.frame[col].cast(pl.Float64).to_numpy()[rows]
            b = T[nb].to_numpy()
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < int(x["min_cell"]):
                continue
            car[col] = {"n": int(m.sum()), kind: _auc(a[m].astype(int), b[m]) if kind == "auroc"
                        else float(spearmanr(a[m], b[m]).statistic)}
        sec["S8"]["carried"] = car
        if Lu is not None:
            oT, oL = T["ood"].cast(pl.Float64).to_numpy(), _on(Lu, rows, "ood")
            ii = np.where(np.isfinite(oL))[0]
            rr = bs.run(ii, y[ii], loss={"T": oT[ii], "L": oL[ii]}, d_loss=[("T", "L")])
            sec["S8"]["out_of_support"] = {"transported": rr["ll:T"], "local": rr["ll:L"], "difference": rr["ll:T − L"]}
    # S6: by hospital
    if "hospital_rows" in out:
        Hs = out.pop("hospital_rows")
        labels = hospital_labels(lm, int(x["hospitals"]["min_patients"]))
        hosp = lm.hospital.astype(str)[rows]
        pos = {int(rw): i for i, rw in enumerate(rows)}
        per = {}
        for h, lab in labels.items():
            ii = np.where(hosp == h)[0]
            if ii.size < int(x["min_cell"]):
                continue
            hh = Hs.filter((pl.col("kind") == "hospital") & (pl.col("group") == lab))
            jj = np.array([pos[int(v)] for v in hh["row"].to_numpy()])
            d = (hh["ll_own"] - hh["ll_other"]).to_numpy()
            Bd = [float((bs.b.rows(wp, jj) * d).sum() / bs.b.rows(wp, jj).sum()) for wp in bs.b] if jj.size else []
            row = {"n_test": int(ii.size), "prevalence": float(y[ii].mean()), "auroc_transported": _auc(y[ii], pT[ii]),
                   "logloss_transported": float(logloss_rows(pT[ii], y[ii]).mean()),
                   "own_vs_other": {"est": float(d.mean()) if d.size else float("nan"), "ci": ci(np.asarray(Bd))},
                   "own_neighbour_share": float(T["own_share"].to_numpy()[ii].mean())
                   if "own_share" in T.columns else float("nan"),
                   "case_base_share": float((lm.hospital.astype(str) == h).mean())}
            for g in x["gamma"]:
                c = f"p_gamma{float(g):g}"
                if c in T.columns:
                    row[f"gamma_{g}"] = float(logloss_rows(T[c].to_numpy()[ii], y[ii]).mean() -
                                              logloss_rows(T["p_gamma1"].to_numpy()[ii], y[ii]).mean())
            per[lab] = row
        sec["S6"] = {"hospitals": per}
        dh = _json_units(f"drift-hospitals_ext-{X}.json")
        if dh:
            sec["S6"]["drift"] = {k: {"est": v["delta"], "ci": v["ci"], "n_test": v["n_test"],
                                      "frozen_logloss": v["frozen_logloss"]} for k, v in dh[0]["hospitals"].items()}
    out["secondary"] = sec
    return out


def drift_summary() -> dict:
    res = {}
    for d in _json_units("drift_ext-*_*.json"):
        v = sorted(d["variables"], key=lambda r: r["delta"])
        res[f"{d['rotation']}:{d['site']}"] = {
            "role": d["role"], "frozen_logloss": d["frozen_logloss"],
            "joint": {"est": d["joint"]["delta"], "ci": d["joint"]["ci"]},
            "missing": {"est": d["missing"]["delta"], "ci": d["missing"]["ci"]},
            "lookup_max_abs_error": d["lookup_max_abs_error"],
            "variables": [{k: r[k] for k in ("name", "delta", "ci", "eff_weight_frozen", "eff_weight_adapted",
                                             "shape_change", "split_half_r")} for r in v]}
    return res


def shapes_summary() -> dict:
    res = {}
    for d in _json_units("shapes_*.json"):
        res[d["site"]] = {"thresholds": d["thresholds"], "vs_transported": d["vs_transported"],
                          "hospitals": {k: {"thresholds": v["thresholds"], "n_learn": v["n_learn"]}
                                        for k, v in d["hospitals"].items()}}
    return res


def report(quick: bool = False, dry: bool = False) -> dict:
    """`dry`: the rehearsal's units (external_dryrun/, in-sample by construction; no freeze needed)."""
    from .config import code_hash
    from .external import dry_dir

    cfg, _, _, x = xsettings(quick)
    if dry:
        rec = {"tag": "DRY RUN — in-sample, not the external stage", "at": "—", "registration": "—",
               "hashes": {"code": code_hash()}}
        _SRC["dir"] = dry_dir()
    else:
        rec = check()
        _SRC["dir"] = None
    S = Sites(cfg, cv_dir(False))
    res = {"freeze": {k: rec[k] for k in ("tag", "at", "registration")}, "code": rec["hashes"]["code"],
           "dry": dry, "sites": {}, "drift": drift_summary(), "shapes": shapes_summary()}
    for X in cfg["cohorts"]:
        r = site_report(X, S, cfg, x)
        if r is not None:
            res["sites"][X] = r
            LOG.info("external report: %s done", X)
    _SRC["dir"] = None
    rep = workdir() / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    name = ("external_dryrun" if dry else "external") + ("_quick" if quick else "")
    (rep / f"{name}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (rep / f"{name}.md").write_text(render(res, x), encoding="utf-8")
    LOG.info("report: %s", rep / f"{name}.md")
    return res


# ---------------------------------------------------------------------------- markdown
def _f(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _e(d, nd=3):
    """est (lo to hi)"""
    return "—" if d is None else f"{_f(d['est'], nd)} ({_f(d['ci'][0], nd)} to {_f(d['ci'][1], nd)})"


def render(res: dict, x: dict) -> str:
    L = ["# `hypo` external stage" + (" — DRY RUN (each site served by a metric that included it: in-sample, "
                                      "for the mechanics only; not a result)" if res.get("dry") else ""), "",
         f"Freeze `{res['freeze']['tag']}` ({res['freeze']['at']}; registration {res['freeze']['registration']}); "
         f"code {res['code']}. Pre-registration: docs/prereg_hypo.md. Out-of-fold predictions on each external site's "
         "Phase 2 folds; 95 % CIs by patient bootstrap.", "",
         "## Hypotheses", "",
         "| site | H1 Δ AUROC transported − local (> −%s) | H2 CITL (±%s) · slope (≥ %s) | H3 Δ log-loss vs last MAP (< 0) |"
         % (x["h1_margin"], x["h2_citl"], x["h2_slope"]), "|---|---|---|---|"]
    yes = {True: "**holds**", False: "does not hold"}
    for c, s in res["sites"].items():
        p = s["primary"]
        if not p:
            continue
        L.append(f"| {c} | {_e(p['H1'])} — {yes[p['H1']['holds']]} | {_e(p['H2']['citl'])} · {_e(p['H2']['slope'])} — "
                 f"{yes[p['H2']['holds']]} | {_e(p['H3'], 4)} — {yes[p['H3']['holds']]} |")
    for c, s in res["sites"].items():
        p = s["primary"]
        if "H4" in p:
            h = p["H4"]
            L += ["", f"**H4 ({c})**: own hospital − other hospitals (base rates corrected) Δ_hosp "
                  f"{_e(h['delta_hospital'], 4)}; random-group control Δ_ctrl {_e(h['delta_control'], 4)}; difference "
                  f"{_e(h['difference'], 4)} — {yes[h['holds']]}."]
        if "H5" in p:
            h = p["H5"]
            L += ["", f"**H5 ({c})**: Δ log-loss γ − (γ = 1), {100 * h['level']:.2f} % CIs: " +
                  "; ".join(f"γ = {g}: {_e(d, 4)}" for g, d in h["gamma"].items()) + f" — {yes[h['holds']]}."]
    for c, s in res["sites"].items():
        sec = s["secondary"]
        L += ["", f"## {c}: {s['n_landmarks']:,} landmarks, {s['n_patients']:,} patients, prevalence "
              f"{s['prevalence']:.1%}", ""]
        if "S2" in sec:
            L += ["S2: methods on the common rows ({:,}).".format(sec["S2"]["n"]), "",
                  "| method | AUROC | log-loss | CITL | slope | mean p | Δ log-loss transported − method |",
                  "|---|---|---|---|---|---|---|"]
            for m, r in sec["S2"]["methods"].items():
                d = sec["S2"]["paired"].get(m, {}).get("logloss")
                L.append(f"| {m} | {_e(r['auroc'])} | {_e(r['logloss'], 4)} | {_f(r['citl']['est']) if r['citl'] else '—'} "
                         f"| {_f(r['slope']['est']) if r['slope'] else '—'} | {_f(r['mean_p'])} | "
                         f"{_e(d, 4) if d else '—'} |")
        if "S1" in sec:
            s1 = sec["S1"]
            L += ["", f"S1: local patients needed (frozen metric; mean over draws). Whole pool: "
                  f"{_e(s1['whole_pool'], 4)}; smallest n within 0.02 / 0.01: {s1['smallest_n_within']['0.02']} / "
                  f"{s1['smallest_n_within']['0.01']}.", "", "| n patients | log-loss | − whole pool | AUROC |",
                  "|---|---|---|---|"]
            for n, r in s1["curve"].items():
                L.append(f"| {n} | {_e(r['logloss'], 4)} | {_e(r['vs_whole'], 4)} | {_f(r['auroc_mean_over_draws'])} |")
            if s1["curve_vs_recalibrated"]:
                L += ["", "| PERISIM curve vs recalibrated comparator (same patients) | Δ log-loss |", "|---|---|"]
                for k, r in s1["curve_vs_recalibrated"].items():
                    L.append(f"| {k} | {_e(r, 4)} |")
        s7 = sec.get("S7", {})
        if s7.get("arterial_line"):
            L += ["", "S7: arterial-line strata (AUROC / log-loss).", "",
                  "| stratum | n | prevalence | neighbours in the same stratum | " +
                  " | ".join(next(iter(s7["arterial_line"].values()))["methods"]) + " |",
                  "|---|---|---|---|" + "---|" * len(next(iter(s7["arterial_line"].values()))["methods"])]
            for lab, r in s7["arterial_line"].items():
                L.append(f"| {lab} | {r['n']:,} | {r['prevalence']:.1%} | {r['neighbours_same_stratum']:.1%} | " +
                         " | ".join(f"{_f(m['auroc']['est'])} / {_f(m['logloss']['est'], 4)}" for m in r["methods"].values())
                         + " |")
        if "one_reading_per_hour" in s7:
            r = s7["one_reading_per_hour"]
            L += ["", f"S7: outcome read with one reading per hour (prevalence {r['prevalence']:.1%}): AUROC "
                  f"{_e(r['auc'])}, log-loss {_e(r['ll'], 4)}, CITL {_e(r['citl'])}, slope {_e(r['slope'])}."]
        if "bands" in s7:
            r = s7["bands"]
            L += ["", f"S7: three MAP bands (shares {', '.join(f'{v:.0%}' for v in r['shares'])}): log-loss transported "
                  f"{_e(r['logloss_transported'], 4)}, local {_e(r['logloss_local'], 4)}, Δ {_e(r['difference'], 4)}; "
                  f"macro AUROC {r['macro_auroc']['transported']:.3f} / {r['macro_auroc']['local']:.3f}."]
        if "S8" in sec:
            s8 = sec["S8"]
            top = sorted(s8["distance_share"].items(), key=lambda t: -t[1])[:8]
            pull = sorted(s8["pull"].items(), key=lambda t: -t[1])[:8]
            L += ["", "S8: share of the query–neighbour distance: " + ", ".join(f"{n} {v:.1%}" for n, v in top) + ".",
                  "", "S8: pull towards the event neighbours: " + ", ".join(f"{n} {v:.3g}" for n, v in pull) + "."]
            if s8.get("carried"):
                L += ["", "S8: carried outcomes: " + "; ".join(
                    f"{k} {'AUROC' if 'auroc' in v else 'Spearman'} {_f(v.get('auroc', v.get('spearman')))}"
                    for k, v in s8["carried"].items()) + "."]
            if "out_of_support" in s8:
                o = s8["out_of_support"]
                L += ["", f"S8: out of support: transported {_e(o['transported'])}, local {_e(o['local'])}, "
                      f"Δ {_e(o['difference'])}."]
        if "S6" in sec:
            L += ["", "S6: by hospital (H1, H2, … by size).", "",
                  "| hospital | n test | prevalence | AUROC | own − other | own-hospital neighbours / case-base share | "
                  + " | ".join(f"Δ γ={g}" for g in x["gamma"]) + " | joint re-deformation Δ |",
                  "|---|---|---|---|---|---|" + "---|" * len(x["gamma"]) + "---|"]
            dr = sec["S6"].get("drift", {})
            for lab, r in sec["S6"]["hospitals"].items():
                L.append(f"| {lab} | {r['n_test']:,} | {r['prevalence']:.1%} | {_f(r['auroc_transported'])} | "
                         f"{_e(r['own_vs_other'], 4)} | {r['own_neighbour_share']:.1%} / {r['case_base_share']:.1%} | "
                         + " | ".join(_f(r.get(f'gamma_{g}'), 4) for g in x["gamma"]) +
                         f" | {_e(dr[lab], 4) if lab in dr else '—'} |")
    if res["drift"]:
        L += ["", "## S4: drift (fold 0)", "",
              "| rotation : site | role | frozen log-loss | all variables re-deformed Δ | missing points only Δ | "
              "largest one-variable gains |", "|---|---|---|---|---|---|"]
        for k, d in res["drift"].items():
            g = ", ".join(f"{v['name']} {_f(v['delta'], 4)}" for v in d["variables"][:4])
            L.append(f"| {k} | {d['role']} | {_f(d['frozen_logloss'], 4)} | {_e(d['joint'], 4)} | "
                     f"{_e(d['missing'], 4)} | {g} |")
    if res["shapes"]:
        L += ["", "## S5: MAP thresholds (odds of hypotension double relative to %s mmHg; median and 10–90 %% over "
              "the accepted models)" % x["threshold"]["reference"], "", "| site / hospital | " +
              " | ".join(x["threshold"]["variables"]) + " |", "|---|" + "---|" * len(x["threshold"]["variables"])]

        def cells(th):
            return " | ".join(f"{_f(th[v]['median'], 1)} ({_f(th[v]['p10'], 1)}–{_f(th[v]['p90'], 1)})"
                              if v in th else "—" for v in x["threshold"]["variables"])
        for c, d in res["shapes"].items():
            L.append(f"| {c} | {cells(d['thresholds'])} |")
            for lab, h in d["hospitals"].items():
                L.append(f"| {c} {lab} | {cells(h['thresholds'])} |")
    return "\n".join(L) + "\n"
