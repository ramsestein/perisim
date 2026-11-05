"""Endpoints and statistics (design §8–§10), from the local predictions and logs of a stream's R orders.

- Per window and arm: the metrics of §8.2, and the within-window sampling variance of AUROC, AUPRC, AUPRC −
  prevalence and CITL from `windows.boot` patient resamples (the same for every arm). A window counts if it holds
  ≥ 30 event landmarks and ≥ 10 patients with an event; otherwise it is dropped for every arm of that order.
- Per arm (each statistic averaged over the R orders): level (mean, median, SD), excess SD, the 10th percentile,
  shares below the clinical threshold and the local reference, calibration (mean and SD of CITL and O/E, the share
  outside the band, the excess calibration error), the share of checks in which the primary trigger fires (H4b).
- Paired contrasts, PERISIM-FIFO against every other arm, with 95 % CIs from a moving-block bootstrap of the
  windows (Künsch, 1989), stratified by segment, block length ⌈N / W⌉ (at most half a segment's windows, so every
  segment has two blocks), the same resampled positions for every arm and order.
- By stretch: the same on the control windows and on the post-boundary windows (the first ⌈N / W⌉ of a segment).
- Recovery (§8.4), saw-tooth (§8.3), PERISIM's D1 per window and the trigger to re-learn the metric (§8.6).
- eICU: per-hospital contrasts pooled by DerSimonian–Laird, H6 per hospital (§9.1).
Aggregates only in the outputs (windows hold ≥ 20 patients).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl

from . import metrics as Mx
from .monitor import citl_se

NO_RANKING = ("prevalence",)
STATS = ("auroc_mean", "auroc_median", "auroc_sd", "auroc_excess_var", "auroc_p10", "auprc_mean", "auprc_p10",
         "amp_mean", "amp_excess_var", "aop_mean", "citl_mean", "citl_sd", "oe_mean", "oe_sd", "share_cal_out",
         "ece", "below_clinical", "below_reference", "trigger_share", "logloss_mean", "brier_mean", "slope_mean")


# --------------------------------------------------------------------------- one order: the window table
def window_table(frame: pl.DataFrame, windows: list[dict], arms: list[str], cfg: dict, checks: list[dict],
                 seed: int, workers: int = 8) -> dict:
    wc = cfg["windows"]
    R = int(wc["boot"])
    y_all = frame["y"].to_numpy().astype(np.float64)
    code_all = frame["patient"].to_numpy()
    wid = frame["window"].to_numpy()
    ood = frame["ood"].to_numpy()
    P = {a: frame[f"p_{a}"].to_numpy() for a in arms}
    K = len(windows)
    T = {"valid": np.zeros(K, bool), "seg": np.array([w["seg"] for w in windows], dtype=object),
         "patients": np.zeros(K, int), "events": np.zeros(K, int), "start": np.array([w["start"] for w in windows]),
         "d1_rate": np.full(K, np.nan), "d1_z": np.full(K, np.nan),
         "m": {a: {} for a in arms}, "v": {a: {} for a in arms}, "citl_z": {a: np.full(K, np.nan) for a in arms}}
    names = ("auroc", "auprc", "auprc_minus_prev", "auprc_over_prev", "citl", "oe", "slope", "logloss", "brier",
             "prevalence")
    for a in arms:
        T["m"][a] = {k: np.full(K, np.nan) for k in names}
        T["v"][a] = {k: np.full(K, np.nan) for k in Mx.BOOT_METRICS}
    nom = float(cfg["diagnostics"]["ood_nominal"])
    order = np.argsort(wid, kind="stable")
    cuts = np.searchsorted(wid[order], np.arange(K + 1))

    def one_window(k: int) -> dict:
        r = order[cuts[k]:cuts[k + 1]]
        y, code = y_all[r], code_all[r]
        o = ood[r].astype(np.float64)
        _, inv = np.unique(code, return_inverse=True)
        se = math.sqrt(float((np.bincount(inv, weights=o - nom) ** 2).sum())) / len(o)
        out = {"patients": np.unique(code).size, "events": int(y.sum()), "d1_rate": o.mean(),
               "d1_z": (o.mean() - nom) / se if se > 0 else np.nan,
               "valid": y.sum() >= int(wc["min_event_landmarks"])
               and np.unique(code[y == 1]).size >= int(wc["min_event_patients"]), "arms": {}}
        if not out["valid"]:
            return out
        Wb = Mx.boot_weights(code, R, seed + k)
        one = np.ones(len(y))
        for a in arms:
            p = P[a][r]
            pt = Mx.all_metrics(p, y, one)
            bt = Mx.boot_metrics(p, y, Wb)
            if a in NO_RANKING:                  # a rate shared by every patient of a step: nothing to rank
                for name in ("auroc", "auprc", "auprc_minus_prev", "auprc_over_prev", "slope"):
                    pt[name] = np.nan
                    bt.pop(name, None)
            c, s_ = citl_se(p, y, code)
            out["arms"][a] = ({n: float(v) for n, v in pt.items()}, {n: float(np.nanvar(v, ddof=1)) for n, v in bt.items()},
                              c / s_ if s_ > 0 else np.nan)
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:          # windows are independent: the same result
        done = list(ex.map(one_window, range(K)))
    for k, o in enumerate(done):
        for key in ("patients", "events", "d1_rate", "d1_z", "valid"):
            T[key][k] = o[key]
        for a, (pt, bt, z) in o["arms"].items():
            for name, v in pt.items():
                T["m"][a][name][k] = v
            for name, v in bt.items():
                T["v"][a][name][k] = v
            T["citl_z"][a][k] = z
    fires = {a: np.full(K, np.nan) for a in arms}
    for ch in checks:
        for a in arms:
            if a in ch:
                fires[a][ch["window"]] = float(ch[a])
    T["fires"] = fires
    return T


# --------------------------------------------------------------------------- statistics of one order
def _excess(m: np.ndarray, v: np.ndarray) -> float:
    return float(np.var(m, ddof=1) - np.mean(v)) if m.size > 1 else np.nan


def arm_stats(T: dict, a: str, idx: np.ndarray, cfg: dict, cohort_ref: float) -> dict:
    ok = T["valid"][idx]
    j = idx[ok]
    if j.size == 0:
        return {k: np.nan for k in STATS}
    m, v = T["m"][a], T["v"][a]
    e = cfg["endpoints"]
    au, ap, amp, ct, oe = m["auroc"][j], m["auprc"][j], m["auprc_minus_prev"][j], m["citl"][j], m["oe"][j]
    tp = cfg["triggers"]["primary"]
    if not np.isfinite(au).all():                # the prevalence arm: calibration and scores only
        return {**{k: np.nan for k in STATS},
                "citl_mean": ct.mean(), "citl_sd": float(np.std(ct, ddof=1)) if j.size > 1 else np.nan,
                "oe_mean": oe.mean(), "oe_sd": float(np.std(oe, ddof=1)) if j.size > 1 else np.nan,
                "share_cal_out": float(np.mean((np.abs(ct) > tp["citl_band"]) | (oe < tp["oe_low"]) | (oe > tp["oe_high"]))),
                "ece": math.sqrt(max(0.0, float(np.mean(ct ** 2) - np.mean(v["citl"][j])))),
                "trigger_share": float(np.nanmean(T["fires"][a][idx])) if np.isfinite(T["fires"][a][idx]).any() else np.nan,
                "logloss_mean": m["logloss"][j].mean(), "brier_mean": m["brier"][j].mean()}
    return {"auroc_mean": au.mean(), "auroc_median": float(np.median(au)), "auroc_sd": float(np.std(au, ddof=1)) if j.size > 1 else np.nan,
            "auroc_excess_var": _excess(au, v["auroc"][j]), "auroc_p10": float(np.percentile(au, 10)),
            "auprc_mean": ap.mean(), "auprc_p10": float(np.percentile(ap, 10)), "amp_mean": amp.mean(),
            "amp_excess_var": _excess(amp, v["auprc_minus_prev"][j]), "aop_mean": m["auprc_over_prev"][j].mean(),
            "citl_mean": ct.mean(), "citl_sd": float(np.std(ct, ddof=1)) if j.size > 1 else np.nan,
            "oe_mean": oe.mean(), "oe_sd": float(np.std(oe, ddof=1)) if j.size > 1 else np.nan,
            "share_cal_out": float(np.mean((np.abs(ct) > tp["citl_band"]) | (oe < tp["oe_low"]) | (oe > tp["oe_high"]))),
            "ece": math.sqrt(max(0.0, float(np.mean(ct ** 2) - np.mean(v["citl"][j])))),
            "below_clinical": float(np.mean(au < e["clinical_auroc"])),
            "below_reference": float(np.mean(au < cohort_ref - e["margins"]["auroc"])),
            "trigger_share": float(np.nanmean(T["fires"][a][idx])) if np.isfinite(T["fires"][a][idx]).any() else np.nan,
            "logloss_mean": m["logloss"][j].mean(), "brier_mean": m["brier"][j].mean(), "slope_mean": m["slope"][j].mean()}


def mean_stats(Ts: list[dict], a: str, idx: np.ndarray, cfg: dict, ref: float) -> dict:
    per = [arm_stats(T, a, idx, cfg, ref) for T in Ts]
    return {k: float(np.nanmean([p[k] for p in per])) if np.isfinite([p[k] for p in per]).any() else np.nan
            for k in STATS}


def vec_stats(Ts: list[dict], a: str, IDX: np.ndarray, M: np.ndarray, cfg: dict, ref: float) -> dict:
    """`mean_stats` for many resampled sets of windows at once: IDX (B × K) window positions, M (B × K) the positions
    kept (a stretch). The same statistics as `arm_stats`, averaged over the orders (tested equal)."""
    import warnings

    e, tp = cfg["endpoints"], cfg["triggers"]["primary"]
    per = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for T in Ts:
            valid = T["valid"][IDX] & M
            n = valid.sum(1)

            def take(x, keep=valid):
                X = np.asarray(x, dtype=np.float64)[IDX]
                X[~keep] = np.nan
                return X

            m, v = T["m"][a], T["v"][a]
            au, ap, amp, ct, oe = (take(m[k]) for k in ("auroc", "auprc", "auprc_minus_prev", "citl", "oe"))
            two = n > 1

            def sd(X):
                return np.where(two, np.nanstd(X, 1, ddof=1), np.nan)

            def exv(X, V):
                return np.where(two, np.nanvar(X, 1, ddof=1) - np.nanmean(take(V), 1), np.nan)

            def share(cond, X):
                return np.nanmean(np.where(np.isnan(X), np.nan, cond.astype(float)), 1)

            out_cal = (np.abs(ct) > tp["citl_band"]) | (oe < tp["oe_low"]) | (oe > tp["oe_high"])
            F = take(T["fires"][a], keep=M)
            per.append({"auroc_mean": np.nanmean(au, 1), "auroc_median": np.nanmedian(au, 1), "auroc_sd": sd(au),
                        "auroc_excess_var": exv(au, v.get("auroc", np.full(len(T["valid"]), np.nan))),
                        "auroc_p10": np.nanpercentile(au, 10, 1), "auprc_mean": np.nanmean(ap, 1),
                        "auprc_p10": np.nanpercentile(ap, 10, 1), "amp_mean": np.nanmean(amp, 1),
                        "amp_excess_var": exv(amp, v.get("auprc_minus_prev", np.full(len(T["valid"]), np.nan))),
                        "aop_mean": np.nanmean(take(m["auprc_over_prev"]), 1),
                        "citl_mean": np.nanmean(ct, 1), "citl_sd": sd(ct), "oe_mean": np.nanmean(oe, 1), "oe_sd": sd(oe),
                        "share_cal_out": share(out_cal, ct),
                        "ece": np.sqrt(np.maximum(0.0, np.nanmean(ct ** 2, 1) - np.nanmean(take(v["citl"]), 1))),
                        "below_clinical": share(au < e["clinical_auroc"], au),
                        "below_reference": share(au < ref - e["margins"]["auroc"], au),
                        "trigger_share": np.where(n > 0, np.nanmean(F, 1), np.nan),
                        "logloss_mean": np.nanmean(take(m["logloss"]), 1),
                        "brier_mean": np.nanmean(take(m["brier"]), 1), "slope_mean": np.nanmean(take(m["slope"]), 1)})
        return {k: np.nanmean(np.stack([p[k] for p in per]), 0) for k in STATS}


def contrast(sf: dict, sc: dict, floor: float) -> dict:
    """FIFO minus a comparator (H2: the ratio of excess SDs, each variance floored). Scalars or arrays."""
    def ratio(vf, vc):
        return np.sqrt((np.maximum(vf, 0.0) + floor ** 2) / (np.maximum(vc, 0.0) + floor ** 2))

    return {"H1_auroc": sf["auroc_mean"] - sc["auroc_mean"], "H1_amp": sf["amp_mean"] - sc["amp_mean"],
            "H2_auroc_ratio": ratio(sf["auroc_excess_var"], sc["auroc_excess_var"]),
            "H2_amp_ratio": ratio(sf["amp_excess_var"], sc["amp_excess_var"]),
            "H3_auroc_p10": sf["auroc_p10"] - sc["auroc_p10"], "H3_auprc_p10": sf["auprc_p10"] - sc["auprc_p10"],
            "H4_ece": sf["ece"] - sc["ece"], "H4_trigger_share": sf["trigger_share"] - sc["trigger_share"],
            "logloss": sf["logloss_mean"] - sc["logloss_mean"], "brier": sf["brier_mean"] - sc["brier_mean"]}


# --------------------------------------------------------------------------- the moving-block bootstrap
def mbb(seg: np.ndarray, ell: int, B: int, seed: int) -> list[np.ndarray]:
    """B resampled sets of window positions: moving blocks inside each segment (blocks of at most half the
    segment's windows), each segment resampled to its own size."""
    rng = np.random.default_rng(seed)
    parts = []
    s = 0
    for i in range(1, len(seg) + 1):
        if i == len(seg) or seg[i] != seg[s]:
            parts.append(np.arange(s, i))
            s = i
    out = []
    for _ in range(B):
        idx = []
        for p in parts:
            K = p.size
            L = max(1, min(ell, K // 2))
            nb = math.ceil(K / L)
            st = rng.integers(0, K - L + 1, size=nb)
            idx.append(np.concatenate([p[x:x + L] for x in st])[:K])
        out.append(np.concatenate(idx))
    return out


def ci95(b: np.ndarray) -> list[float]:
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b.size else [np.nan, np.nan]


# --------------------------------------------------------------------------- recovery and saw-tooth
def recovery(T: dict, windows: list[dict], a: str, metric: str, delta: float, consecutive: int,
             start_is_boundary: bool = False) -> list[dict]:
    out = []
    labs = []
    for w in windows:
        if not labs or labs[-1][0] != w["seg"]:
            labs.append((w["seg"], []))
        labs[-1][1].append(w["index"])
    for lab, idx in (labs if start_is_boundary else labs[1:]):
        m = T["m"][a][metric]
        vals = np.array([m[i] if T["valid"][i] else np.nan for i in idx])
        half = vals[len(vals) // 2:]
        stable = float(np.nanmean(half)) if np.isfinite(half).any() else np.nan
        rec = None
        for j in range(len(idx) - consecutive + 1):
            seg = vals[j:j + consecutive]
            if np.isfinite(seg).all() and (np.abs(seg - stable) <= delta).all():
                rec = windows[idx[j]]["start"] - windows[idx[0]]["start"]
                break
        size = windows[idx[-1]]["end"] - windows[idx[0]]["start"]
        out.append({"segment": lab, "stable": stable, "patients": rec if rec is not None else size,
                    "censored": rec is None})
    return out


def sawtooth(m: np.ndarray, valid: np.ndarray, windows: list[dict], log: list[dict]) -> dict:
    """For each cycle between two deployments: the window after the first minus the window before the next."""
    dep = sorted(e["pos"] for e in log if e.get("event") == "deploy")
    starts = np.array([w["start"] for w in windows])
    ends = np.array([w["end"] for w in windows])
    drops = []
    for d0, d1 in zip(dep, dep[1:]):
        after = np.flatnonzero(starts >= d0)
        before = np.flatnonzero(ends <= d1)
        if after.size and before.size and before[-1] > after[0]:
            i, j = after[0], before[-1]
            if valid[i] and valid[j]:
                drops.append(m[i] - m[j])
    return {"cycles": len(drops), "mean_amplitude": float(np.mean(drops)) if drops else np.nan}


# --------------------------------------------------------------------------- one stream
def rewindow(frame: pl.DataFrame, log: dict, scale: float) -> tuple[pl.DataFrame, dict]:
    """The same predictions in evaluation windows `scale` times as large (the W sensitivity): the arms' runs, their
    checks and updates are unchanged; only the windows the endpoints are read on change (H4b is not re-read)."""
    from .streams import windows_for

    segs, a = [], 0
    for s in log["stream"]["segments"]:
        segs.append((s["segment"], a, a + int(s["patients"])))
        a += int(s["patients"])
    size = {}
    for w in log["windows"]:
        size.setdefault(w["seg"], []).append(w["end"] - w["start"])
    W = {lab: max(1, int(round(np.median(v) * scale))) for lab, v in size.items()}
    wins = windows_for(segs, W)
    ends = np.array([w["end"] for w in wins])
    wid = np.searchsorted(ends, frame["pos"].to_numpy(), side="right")
    return frame.with_columns(pl.Series("window", wid)), {**log, "windows": wins, "checks": []}


def stream_endpoints(runs: list[tuple[pl.DataFrame, dict]], cfg: dict, seed: int, light: bool = False,
                     arms: list[str] | None = None) -> dict:
    """Endpoints of one stream (one N, one hospital) from its R orders: [(predictions, log), …]. `light`: the whole
    stream's arms and contrasts only (the sensitivity analyses)."""
    logs = [lg for _, lg in runs]
    windows = logs[0]["windows"]
    arms = list(logs[0]["arms"]) if arms is None else arms
    K = len(windows)
    if any(len(lg["windows"]) != K for lg in logs):
        raise ValueError("the orders of a stream must have the same windows")
    Ts = [window_table(fr, lg["windows"], arms, cfg, lg["checks"], seed + 1000 * i) for i, (fr, lg) in enumerate(runs)]
    seg = Ts[0]["seg"]
    st = logs[0]["stream"]
    N, W0 = int(st["N"]), max(1, round(np.median([w["end"] - w["start"] for w in windows])))
    ell = max(1, math.ceil(N / W0))
    cohort = _cohort_of(st, cfg)
    ref = float(cfg["endpoints"]["local_reference_auroc"].get(cohort, np.nan))
    floor = float(cfg["endpoints"]["h2_floor_sd"])
    first = {}
    for w in windows:
        first.setdefault(w["seg"], w["index"])
    deploy = st.get("kind") == "deploy"                  # stream (b): its start is a boundary (other hospitals → own)
    post = set()
    for s_, i0 in first.items():
        if i0 != 0 or deploy:
            post.update(range(i0, min(i0 + ell, K)))
    sets = {"all": np.arange(K), "control": np.array([i for i in range(K) if i not in post], dtype=int),
            "post_boundary": np.array(sorted(post), dtype=int)}
    if light:
        sets = {"all": sets["all"]}
    boot = mbb(seg, ell, int(cfg["endpoints"]["boot"]), seed)
    res = {"stream": st, "orders": len(runs), "windows": K, "block_length": ell,
           "windows_dropped": [int((~T["valid"]).sum()) for T in Ts], "arms": {}, "contrasts": {}, "stretch": {}}
    IDXB = np.stack(boot)
    IDX0 = np.arange(K)[None, :]
    for name, idx in sets.items():
        if idx.size == 0:
            continue
        S = {a: {k: float(x[0]) for k, x in vec_stats(Ts, a, IDX0, np.isin(IDX0, idx), cfg, ref).items()} for a in arms}
        if name == "all":
            res["arms"] = S
        MB = np.isin(IDXB, idx)
        BS = {a: vec_stats(Ts, a, IDXB, MB, cfg, ref) for a in arms}
        con = {}
        for c in arms:
            if c == "perisim_fifo":
                continue
            pt = contrast(S["perisim_fifo"], S[c], floor)
            bs = contrast(BS["perisim_fifo"], BS[c], floor)
            con[c] = {k: {"estimate": float(pt[k]), "ci": ci95(bs[k]),
                          "se": float(np.nanstd(bs[k], ddof=1)) if np.isfinite(bs[k]).sum() > 1 else np.nan}
                      for k in pt}
        if name == "all":
            res["contrasts"] = con
            res["arms_ci"] = {a: {k: ci95(BS[a][k]) for k in STATS} for a in arms}
        else:
            res["stretch"][name] = {"windows": int(idx.size), "arms": S, "contrasts": con}
    if light:
        return res
    rc = cfg["endpoints"]["recovery"]
    res["recovery"] = {a: {m: [recovery(T, windows, a, m, float(rc[m]), int(rc["consecutive"]), deploy) for T in Ts]
                           for m in ("auroc", "citl")} for a in arms}
    res["psi"] = {}
    for name, idx in sets.items():
        per = [[c.get("psi_mean", np.nan) for c in lg["checks"] if c["window"] in set(idx.tolist())] for lg in logs]
        vals = [np.nanmean(p) for p in per if np.isfinite(p).any()]
        res["psi"][name] = float(np.mean(vals)) if vals else np.nan
    res["sawtooth"] = {}
    for a in arms:
        if not ("_realistic" in a or a.endswith("_ideal")):
            continue
        out = {}
        for m in ("auroc", "citl"):
            per = [sawtooth(T["m"][a][m], T["valid"], windows, lg["arms"][a]) for T, lg in zip(Ts, logs)]
            amp = [p["mean_amplitude"] for p in per]
            out[m] = {"cycles": float(np.mean([p["cycles"] for p in per])),
                      "mean_amplitude": float(np.nanmean(amp)) if np.isfinite(amp).any() else np.nan}
        res["sawtooth"][a] = out
    res["updates"] = {a: _updates([lg["arms"][a] for lg in logs]) for a in arms
                      if a not in ("perisim_fifo", "perisim_cumulative", "perisim_static", "prevalence")}
    res["d1"] = _d1(Ts, windows, sets, cfg)
    res["redeform"] = [r for lg in logs for r in lg.get("redeform", [])]
    res["relearn"] = _relearn(Ts, res["redeform"], cfg)
    return res


def _cohort_of(st: dict, cfg: dict) -> str:
    spec = cfg["streams"][st["stream"]]
    return spec.get("cohort") or spec.get("second")


def _updates(per_order: list) -> dict:
    n_trig = [sum(1 for e in lg if e.get("event") == "trigger") for lg in per_order]
    n_dep = [sum(1 for e in lg if e.get("event") == "deploy" and e.get("phase") in ("update", "ideal")) for lg in per_order]
    choices = {}
    for lg in per_order:
        for e in lg:
            if e.get("event") == "deploy" and e.get("phase") in ("update", "ideal"):
                choices[e["choice"]] = choices.get(e["choice"], 0) + 1
    golive = [e.get("choice") for lg in per_order for e in lg if e.get("phase") == "go-live"]
    secs = [e.get("seconds", 0.0) for lg in per_order for e in lg if e.get("event") == "deploy"]
    par = [e.get("parameters_changed", 0) for lg in per_order for e in lg if e.get("event") == "deploy"]
    return {"triggers_mean": float(np.mean(n_trig)), "deployments_mean": float(np.mean(n_dep)), "choices": choices,
            "golive_choices": {c: golive.count(c) for c in set(golive)},
            "seconds_mean": float(np.mean(secs)) if secs else 0.0,
            "parameters_changed_mean": float(np.mean(par)) if par else 0.0}


def _d1(Ts: list[dict], windows: list[dict], sets: dict, cfg: dict) -> dict:
    z = float(cfg["diagnostics"]["ood_alarm_z"])
    out = {}
    for name, idx in sets.items():
        if idx.size == 0:
            continue
        rates = [float(np.nanmean(T["d1_rate"][idx])) for T in Ts]
        alarms = [float(np.nanmean(T["d1_z"][idx] > z)) for T in Ts]
        out[name] = {"rate": float(np.mean(rates)), "alarm_share": float(np.mean(alarms))}
    out["by_window"] = {"rate": np.nanmean([T["d1_rate"] for T in Ts], 0).tolist(),
                        "segment": [w["seg"] for w in windows]}
    return out


def _relearn(Ts: list[dict], redeform: list[dict], cfg: dict) -> dict:
    rc = cfg["diagnostics"]["relearn"]
    first = []
    for T in Ts:
        hit = T["d1_rate"] >= float(rc["ood_rate"])
        run, at = 0, None
        for i, h in enumerate(hit):
            run = run + 1 if h else 0
            if run >= int(rc["consecutive"]):
                at = i
                break
        first.append(at)
    gains = [r for r in redeform if r["gain"] <= float(rc["gain"]) and r["ci"][1] < 0]
    return {"ood_first_window_by_order": first, "redeform_hits": [{"eval_window": r["eval_window"],
                                                                   "gain": r["gain"]} for r in gains]}


# --------------------------------------------------------------------------- eICU: hospitals
def dersimonian_laird(y: np.ndarray, se: np.ndarray) -> dict:
    y, se = np.asarray(y, float), np.asarray(se, float)
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, se = y[ok], se[ok]
    k = y.size
    if k == 0:
        return {"k": 0}
    w = 1 / se ** 2
    mu_f = float((w * y).sum() / w.sum())
    Q = float((w * (y - mu_f) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (Q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    ws = 1 / (se ** 2 + tau2)
    mu = float((ws * y).sum() / ws.sum())
    s = float(math.sqrt(1 / ws.sum()))
    return {"k": int(k), "estimate": mu, "ci": [mu - 1.959964 * s, mu + 1.959964 * s], "se": s, "tau2": tau2,
            "I2": max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0, "Q": Q}


H_KEYS = ("H1_auroc", "H1_amp", "H2_auroc_ratio", "H2_amp_ratio", "H3_auroc_p10", "H3_auprc_p10", "H4_ece",
          "H4_trigger_share")


def pool_hospitals(per_hospital: dict[str, dict], comparator: str) -> dict:
    """Random-effects pooling of each contrast over hospitals (H2 on the log of the ratio)."""
    out = {}
    for key in H_KEYS:
        ys, ses, labs = [], [], []
        for h, r in per_hospital.items():
            c = r["contrasts"].get(comparator, {}).get(key)
            if c is None:
                continue
            est, se = c["estimate"], c["se"]
            if key.startswith("H2"):
                lo, hi = c["ci"]
                if not (np.isfinite(est) and est > 0 and lo > 0 and hi > 0):
                    continue
                est, se = math.log(est), (math.log(hi) - math.log(lo)) / (2 * 1.959964)
            ys.append(est)
            ses.append(se)
            labs.append(h)
        pooled = dersimonian_laird(np.array(ys), np.array(ses))
        if key.startswith("H2") and pooled.get("k"):
            pooled = {**pooled, "ratio": math.exp(pooled["estimate"]), "ratio_ci": [math.exp(x) for x in pooled["ci"]]}
        out[key] = {"pooled": pooled, "hospitals": {h: {"estimate": y_, "se": s_} for h, y_, s_ in zip(labs, ys, ses)}}
    return out


def h6_hospital(runs: list[tuple[pl.DataFrame, dict]], cfg: dict, ml_arm: str, seed: int) -> dict:
    """H6 in one hospital of stream (a): the D1 rate in the first window after the change of year minus its mean in
    the other windows; the share of D1 alarms in those control windows; the same for the ML arm's matched test
    (CITL = 0, two-sided 1 %). Averaged over orders; SE from the orders."""
    z1, zml = float(cfg["diagnostics"]["ood_alarm_z"]), float(cfg["endpoints"]["h6"]["ml_citl_z"])
    sig, false_p, sig_ml, false_ml = [], [], [], []
    for i, (fr, lg) in enumerate(runs):
        T = window_table(fr, lg["windows"], [ml_arm, "perisim_fifo"], cfg, [], seed + i)
        first = {}
        for w in lg["windows"]:
            first.setdefault(w["seg"], w["index"])
        b = [i0 for i0 in first.values() if i0 != 0]
        if not b:
            continue
        ctrl = np.array([k for k in range(len(lg["windows"])) if k not in b])
        sig.append(float(T["d1_rate"][b[0]] - np.nanmean(T["d1_rate"][ctrl])))
        false_p.append(float(np.nanmean(T["d1_z"][ctrl] > z1)))
        a_ml = np.abs(T["citl_z"][ml_arm]) > zml
        sig_ml.append(float(a_ml[b[0]]) - float(np.nanmean(a_ml[ctrl])))
        false_ml.append(float(np.nanmean(a_ml[ctrl])))

    def m_se(x):
        x = np.asarray(x, float)
        return float(np.nanmean(x)), float(np.nanstd(x, ddof=1) / math.sqrt(max(np.isfinite(x).sum(), 1))) if x.size > 1 else np.nan

    return {"signal": m_se(sig), "false_alarms": m_se(false_p), "ml_signal": m_se(sig_ml), "ml_false_alarms": m_se(false_ml)}


def pool_h6(per_hospital: dict[str, dict], B: int, seed: int) -> dict:
    labs = list(per_hospital)
    sig = dersimonian_laird(np.array([per_hospital[h]["signal"][0] for h in labs]),
                            np.array([per_hospital[h]["signal"][1] for h in labs]))
    rng = np.random.default_rng(seed)

    def boot(key):
        x = np.array([per_hospital[h][key][0] for h in labs], float)
        bs = [np.nanmean(x[rng.integers(0, x.size, x.size)]) for _ in range(B)]
        return {"estimate": float(np.nanmean(x)), "ci": ci95(np.array(bs))}

    return {"signal": sig, "false_alarms": boot("false_alarms"), "ml_signal": boot("ml_signal"),
            "ml_false_alarms": boot("ml_false_alarms"), "hospitals": len(labs)}
