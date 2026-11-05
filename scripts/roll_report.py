"""Phase 5 of `roll`: the aggregate exports behind results/roll/REPORT.md, from the single run's frozen outputs.

Reads (on the workstation, <workdir>/experiments/roll): endpoints.json, endpoints_deviation.json (the declared
deviation, scripts/roll_deviation_pooling.py), t2/endpoints.json, runs/governability.json, runs/stage.json, every
unit's log.json (compute time only) and state/freeze.json. Computes nothing new: it selects and tabulates.

Writes, under results/roll/ (aggregates only; eICU hospitals as H1, H2, … by size, never by their identifier):
  reports/endpoints.md         the tables of every endpoint the report cites
  results/streams/<name>.json  the full endpoints of the MIMIC-IV, cross-cohort and INSPIRE streams (every N)
  results/eicu_pooled.json, results/eicu_pooled_deviation.json   eICU's pooled results: frozen, then the deviation
  results/eicu_<within|deploy>_hospitals.json                     a compact table per hospital
  results/t2.json, results/governability.json, results/run.json, results/freeze.json
  figures/forest_eicu_<a|b>.png, figures/h7_by_N.png

Usage:  python scripts/roll_report.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roll import run as R  # noqa: E402
from roll.config import workdir  # noqa: E402
from roll.frozen import check, state_dir  # noqa: E402

OUT = ROOT / "results" / "roll"
ARM5 = "lightgbm_realistic"
RULES = {"H1_auroc": ("lo", -0.02), "H1_amp": ("lo", -0.03), "H2_auroc_ratio": ("hi", 1.0), "H2_amp_ratio": ("hi", 1.0),
         "H3_auroc_p10": ("lo", 0.0), "H3_auprc_p10": ("lo", 0.0), "H4_ece": ("hi", 0.0), "H4_trigger_share": ("hi", 0.0)}
NAMES = {"H1_auroc": "H1 AUROC", "H1_amp": "H1 AUPRC − prevalence", "H2_auroc_ratio": "H2 excess SD ratio, AUROC",
         "H2_amp_ratio": "H2 excess SD ratio, AUPRC − prev.", "H3_auroc_p10": "H3 P10 of the AUROC",
         "H3_auprc_p10": "H3 P10 of the AUPRC", "H4_ece": "H4a excess calibration error",
         "H4_trigger_share": "H4b trigger share"}
RULE_TXT = {"lo": "lower bound > {}", "hi": "upper bound < {}"}
COMPS = ["lightgbm_ideal", "lightgbm_realistic_L0", "lightgbm_realistic_auroc5", "lightgbm_realistic_auroc10",
         "lightgbm_realistic_logloss10", "lightgbm_realistic_davis", "lightgbm_realistic_window", "lightgbm_golive",
         "lightgbm_frozen", "logreg_realistic", "logreg_realistic_L0", "logreg_realistic_window", "logreg_ideal",
         "perisim_cumulative", "perisim_static", "prevalence"]
LEVEL_ARMS = ["perisim_fifo", "perisim_cumulative", "perisim_static", "prevalence", "lightgbm_frozen",
              "lightgbm_golive", ARM5, "lightgbm_realistic_L0", "lightgbm_realistic_window", "lightgbm_ideal",
              "logreg_frozen", "logreg_realistic", "logreg_ideal"]
LEVEL_COLS = [("auroc_mean", "AUROC"), ("amp_mean", "AUPRC − prev."), ("citl_mean", "CITL"), ("oe_mean", "O/E"),
              ("slope_mean", "slope"), ("ece", "excess cal. error"), ("trigger_share", "trigger share"),
              ("logloss_mean", "log-loss"), ("brier_mean", "Brier")]
HOSP_COMPS = [ARM5, "lightgbm_ideal", "lightgbm_realistic_L0", "lightgbm_realistic_window", "logreg_realistic",
              "perisim_cumulative", "perisim_static"]     # per hospital (file size); every comparator is pooled
PLAIN = ["mimic_temporal", "inspire_mimic", "eicu_mimic", "mimic_inspire", "inspire_control"]
CONF = [("MIMIC-IV temporal, N 500", "s", "mimic_temporal|all|N500"), ("eICU (a), N 100", "e", "eicu_within|N100"),
        ("eICU (b), N 250", "e", "eicu_deploy|N250")]
# the reference palette's categorical slots, in order (light mode; validated with the dataviz skill's script)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def ok(k, lo, hi):
    side, t = RULES[k]
    return lo > t if side == "lo" else hi < t


def fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def ci_txt(est, lo, hi, d=3):
    return f"{fmt(est, d)} ({fmt(lo, d)} to {fmt(hi, d)})"


def clean_stream(st: dict) -> dict:
    return {k: v for k, v in st.items() if k != "hospital"}


def from_stream(S, key, comp=ARM5):
    c = S[key]["contrasts"][comp]
    return {k: (c[k]["estimate"], *c[k]["ci"]) for k in RULES}


def from_pool(P):
    out = {}
    for k in RULES:
        q = P[k]["pooled"]
        if not q.get("k"):
            out[k] = (math.nan, math.nan, math.nan)
        elif k.startswith("H2"):
            out[k] = (q["ratio"], *q["ratio_ci"])
        else:
            out[k] = (q["estimate"], *q["ci"])
    return out


def row_conf(ep, dev, kind, key, comp=ARM5, deviation=False):
    if kind == "s":
        return from_stream(ep["streams"], key, comp)
    r = from_pool(ep["eicu"][key]["pooled"][comp])
    if deviation:
        r = from_pool(dev["eicu"][key]["pooled"][comp])
    return r


def table(lines, head, rows):
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines.append("")


def hospital_table(ep, name, N):
    keys = sorted((k for k in ep["streams"] if k.startswith(f"{name}|") and k.endswith(f"|N{N}")),
                  key=lambda k: int(k.split("|")[1][1:]))
    out = {}
    for k in keys:
        s = ep["streams"][k]
        h = k.split("|")[1]
        st = clean_stream(s["stream"])
        out[h] = {"patients_evaluated": st.get("evaluated"), "windows": s["windows"],
                  "segments": [{x: seg[x] for x in ("segment", "patients")} for seg in st.get("segments", [])],
                  "windows_dropped": s["windows_dropped"],
                  "levels": {a: {c: s["arms"][a][c] for c, _ in LEVEL_COLS} for a in LEVEL_ARMS if a in s["arms"]},
                  "contrasts": {c: {kk: {"estimate": v["estimate"], "ci": v["ci"], "se": v["se"]}
                                    for kk, v in s["contrasts"][c].items()} for c in HOSP_COMPS if c in s["contrasts"]},
                  "updates": {a: s["updates"][a] for a in (ARM5, "lightgbm_ideal") if a in s["updates"]},
                  "d1": {kk: vv for kk, vv in s["d1"].items() if kk != "by_window"}, "relearn": s["relearn"],
                  "h6": s.get("h6")}
    return out


def forest(ep, dev, key, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("H1_auroc", "H1: window AUROC, PERISIM-FIFO − arm 5", -0.02),
              ("H4_ece", "H4a: excess calibration error, PERISIM-FIFO − arm 5", 0.0)]
    P = ep["eicu"][key]["pooled"][ARM5]
    hs = sorted(P["H1_auroc"]["hospitals"], key=lambda h: int(h[1:]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.24 * len(hs) + 1.6), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (k, lab, margin) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        hosp = P[k]["hospitals"]
        ys = list(range(len(hs), 0, -1))
        for y, h in zip(ys, hs):
            e, s = hosp[h]["estimate"], hosp[h]["se"]
            ax.plot([e - 1.96 * s, e + 1.96 * s], [y, y], color=SERIES[0], lw=2, solid_capstyle="round")
            ax.plot(e, y, "o", color=SERIES[0], ms=5, mec="#fcfcfb", mew=1)
        q = P[k]["pooled"]
        ax.fill([q["ci"][0], q["estimate"], q["ci"][1], q["estimate"]], [-0.5, -0.1, -0.5, -0.9], color=INK)
        ax.axvline(0, color=INK2, lw=1)
        if margin:
            ax.axvline(margin, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.text(margin, len(hs) + 1.2, f"margin {margin}", color=INK2, fontsize=8, ha="center")
        ax.set_title(f"{lab}\npooled {q['estimate']:.3f} ({q['ci'][0]:.3f} to {q['ci'][1]:.3f}); "
                     f"I² {q['I2']:.2f}, τ² {q['tau2']:.1e}", fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
    axes[0].set_yticks(list(range(len(hs), 0, -1)) + [-0.5])
    axes[0].set_yticklabels(hs + ["pooled (DL)"])
    axes[0].set_ylim(-1.3, len(hs) + 1.8)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Per hospital: estimate ± 1.96 SE (moving-block bootstrap, 10 orders). Pooled: "
                          "DerSimonian–Laird random effects. Negative = PERISIM-FIFO lower.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def h7_figure(ep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("MIMIC-IV temporal", "s", "mimic_temporal"), ("INSPIRE → MIMIC-IV", "s", "inspire_mimic"),
              ("eICU → MIMIC-IV", "s", "eicu_mimic"), ("MIMIC-IV → INSPIRE", "s", "mimic_inspire"),
              ("INSPIRE control", "s", "inspire_control"), ("eICU (a), pooled", "e", "eicu_within"),
              ("eICU (b), pooled", "e", "eicu_deploy")]
    panels = [("H1_auroc", "H1: mean window AUROC, PERISIM-FIFO − arm 5", -0.02),
              ("H4_ece", "H4a: excess calibration error, PERISIM-FIFO − arm 5", 0.0)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (k, lab, margin) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for i, (name, kind, st) in enumerate(series):
            xs, es, los, his = [], [], [], []
            for N in (100, 250, 500, 1000, 2000):
                if kind == "s":
                    key = f"{st}|all|N{N}"
                    if key not in ep["streams"]:
                        continue
                    c = ep["streams"][key]["contrasts"][ARM5][k]
                    e, lo, hi = c["estimate"], *c["ci"]
                else:
                    key = f"{st}|N{N}"
                    if key not in ep["eicu"]:
                        continue
                    q = ep["eicu"][key]["pooled"][ARM5][k]["pooled"]
                    e, lo, hi = q["estimate"], *q["ci"]
                xs.append(N * (1 + 0.025 * (i - 3)))  # dodged, so that the CIs at one N do not overlap
                es.append(e)
                los.append(max(0.0, e - lo))          # a percentile CI need not contain the point estimate
                his.append(max(0.0, hi - e))
            ax.errorbar(xs, es, yerr=[los, his], color=SERIES[i], lw=2, marker=MARKERS[i], ms=6, mec="#fcfcfb",
                        capsize=0, elinewidth=1.2, label=name)
        ax.axhline(0, color=INK2, lw=1)
        if margin:
            ax.axhline(margin, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_xticks([100, 250, 500, 1000, 2000])
        ax.set_xticklabels(["100", "250", "500", "1,000", "2,000"])
        ax.set_xlabel("case-base window N (patients)", fontsize=8.5, color=INK2)
        ax.set_title(lab, fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8)
    h, lab_ = axes[0].get_legend_handles_labels()
    fig.legend(h, lab_, fontsize=8, frameon=False, loc="lower center", ncol=4, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.045))
    fig.text(0.01, 0.01, "Estimates with 95 % CIs (moving-block bootstrap; eICU: DerSimonian–Laird over hospitals). "
                         "Dashed: H1's margin. Primary N: 500 (MIMIC-IV, cross-cohort, control), 100 (a), 250 (b).",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    cfg, _, _, _, cs = R._real()
    rec = check(cfg, cs)
    W = workdir()
    ep = json.loads((W / "endpoints.json").read_text(encoding="utf-8"))
    dev = json.loads((W / "endpoints_deviation.json").read_text(encoding="utf-8"))
    t2 = json.loads((W / "t2" / "endpoints.json").read_text(encoding="utf-8"))
    gov = json.loads((W / "runs" / "governability.json").read_text(encoding="utf-8"))
    stage = json.loads((W / "runs" / "stage.json").read_text(encoding="utf-8"))
    S, E = ep["streams"], ep["eicu"]
    for d in ("reports", "results/streams", "figures"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # --- the run: units and compute time
    secs, units = {}, {}
    for lg in (W / "runs").glob("*/*/*/*/log.json"):
        name = lg.relative_to(W / "runs").parts[0]
        x = json.loads(lg.read_text(encoding="utf-8"))
        secs[name] = secs.get(name, 0.0) + float(x.get("seconds", 0.0))
        units[name] = units.get(name, 0) + 1
    run = {"tag": rec["tag"], "started": stage["started"], "code_hash": stage["code_hash"], "units": units,
           "units_total": sum(units.values()), "compute_hours": {k: round(v / 3600, 2) for k, v in secs.items()},
           "windows_total": sum(v["windows"] * v["orders"] for v in S.values()),
           "windows_dropped": sum(sum(v["windows_dropped"]) for v in S.values()),
           "w_sensitivity_errors": [k for k, v in S.items() if any("error" in x for x in v.get("sensitivity_W", {}).values())]}

    # --- JSON exports
    for name in PLAIN:
        part = {}
        for k, v in S.items():
            if k.startswith(f"{name}|"):
                v = dict(v)
                v["stream"] = clean_stream(v["stream"])
                v["d1"] = {kk: vv for kk, vv in v["d1"].items() if kk != "by_window"}
                part[k] = v
        (OUT / "results" / "streams" / f"{name}.json").write_text(json.dumps(part, indent=1), encoding="utf-8")
    compact = {"separators": (",", ":")}
    (OUT / "results" / "eicu_pooled.json").write_text(json.dumps(
        {"note": "the frozen output: DerSimonian–Laird over hospitals, with each hospital's estimate and SE", **E},
        **compact), encoding="utf-8")
    (OUT / "results" / "eicu_pooled_deviation.json").write_text(json.dumps(dev, **compact), encoding="utf-8")
    for n in ("eicu_within", "eicu_deploy"):
        (OUT / "results" / f"{n}_hospitals.json").write_text(json.dumps(
            {"note": f"per hospital, H1, H2, … by size; contrasts against {', '.join(HOSP_COMPS)}",
             **{f"N{N}": hospital_table(ep, n, N) for N in (100, 250)}}, **compact), encoding="utf-8")
    (OUT / "results" / "t2.json").write_text(json.dumps(t2, indent=1), encoding="utf-8")
    (OUT / "results" / "governability.json").write_text(json.dumps(gov, indent=1), encoding="utf-8")
    (OUT / "results" / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    (OUT / "results" / "freeze.json").write_text((state_dir() / "freeze.json").read_text(encoding="utf-8"), encoding="utf-8")

    # --- figures
    forest(ep, dev, "eicu_within|N100", "eICU (a), within each hospital, N = 100: per hospital and pooled",
           OUT / "figures" / "forest_eicu_a.png")
    forest(ep, dev, "eicu_deploy|N250", "eICU (b), deployment, N = 250: per hospital and pooled",
           OUT / "figures" / "forest_eicu_b.png")
    h7_figure(ep, OUT / "figures" / "h7_by_N.png")

    # --- the tables
    L = ["# `roll` — the endpoints (aggregates)", "",
         f"Generated by `scripts/roll_report.py` from the single run (tag `{rec['tag']}`). Every contrast is "
         "PERISIM-FIFO minus the comparator, paired on the same windows, with its 95 % CI (moving-block bootstrap; "
         "eICU: DerSimonian–Laird over hospitals). Negative = PERISIM-FIFO lower. ✔ = the pre-registered rule is met "
         "(prereg §7).", ""]
    L += ["## 1. The run", ""]
    table(L, ["stream", "units", "compute (h)"], [[k, units[k], fmt(secs[k] / 3600, 1)] for k in sorted(units)])
    L += [f"Total {run['units_total']} units; {run['windows_dropped']} of {run['windows_total']} windows dropped "
          f"(fewer than 30 event landmarks or 10 patients with an event); W-sensitivity errors: "
          f"{len(run['w_sensitivity_errors'])}.", "",
          f"Governability ({gov['unit']}): digest with 6 threads `{gov['digest_run']}`, with 1 thread "
          f"`{gov['digest_one_thread']}` (identical: {gov['identical']}); FIFO window rebuilt from the log alone: "
          f"{gov['reconstruction']['mismatches']} mismatches in {gov['reconstruction']['steps']} steps.", ""]

    L += ["## 2. Confirmatory: PERISIM-FIFO − arm 5 (LightGBM realistic, primary trigger, L = 3 months)", ""]
    rows = []
    rc = {t: row_conf(ep, dev, kd, key) for t, kd, key in CONF}
    rd = {t: row_conf(ep, dev, kd, key, deviation=True) for t, kd, key in CONF if kd == "e"}
    for k in RULES:
        r = [NAMES[k]]
        for t, kd, key in CONF:
            e, lo, hi = rc[t][k]
            cell = ci_txt(e, lo, hi, 3) + (" ✔" if ok(k, lo, hi) else " ✘")
            if kd == "e" and k == "H4_trigger_share":
                e2, lo2, hi2 = rd[t][k]
                cell += f"; deviation: {ci_txt(e2, lo2, hi2, 3)}" + (" ✔" if ok(k, lo2, hi2) else " ✘")
            r.append(cell)
        r.append(RULE_TXT[RULES[k][0]].format(RULES[k][1]))
        rows.append(r)
    table(L, ["endpoint"] + [t for t, _, _ in CONF] + ["rule"], rows)
    L += ["Hospitals pooled (k) for each endpoint:", ""]
    table(L, ["endpoint"] + [t for t, kd, _ in CONF if kd == "e"],
          [[NAMES[k]] + [f"{E[key]['pooled'][ARM5][k]['pooled'].get('k')} (I² {fmt(E[key]['pooled'][ARM5][k]['pooled'].get('I2', math.nan), 2)})"
                         + (f"; deviation {dev['eicu'][key]['pooled'][ARM5][k]['pooled'].get('k')}" if k == "H4_trigger_share" else "")
                         for t, kd, key in CONF if kd == "e"] for k in RULES])
    h2_ties = {key: {k: len([1 for x in E[key]["pooled"][ARM5][k]["hospitals"].values() if x["se"] == 0])
                     for k in ("H2_auroc_ratio", "H2_amp_ratio")} for _, kd, key in CONF if kd == "e"}
    L += [f"H2: hospitals left out by the frozen filter because neither arm has any excess variance (ratio exactly 1): "
          f"{json.dumps(h2_ties)}. Hospitals without a ratio (no valid estimate) are also left out.", ""]
    L += ["Intersection–union (every endpoint in every confirmatory stream):", ""]
    hyp = {"H1": ["H1_auroc", "H1_amp"], "H2": ["H2_auroc_ratio", "H2_amp_ratio"], "H3": ["H3_auroc_p10", "H3_auprc_p10"],
           "H4": ["H4_ece", "H4_trigger_share"]}
    table(L, ["hypothesis"] + [t for t, _, _ in CONF] + ["overall"],
          [[h] + ["✔" if all(ok(k, *rc[t][k][1:]) for k in ks) else "✘" for t, _, _ in CONF]
           + ["**holds**" if all(all(ok(k, *rc[t][k][1:]) for k in ks) for t, _, _ in CONF) else "does not hold"]
           for h, ks in hyp.items()])
    h6 = E["eicu_within|N100"]["h6"]
    L += ["**H6** (eICU (a), N 100):", ""]
    table(L, ["quantity", "estimate (95 % CI)", "rule"],
          [["(a) out-of-support rate, first window after the change of year − other windows (DL)",
            ci_txt(h6["signal"]["estimate"], *h6["signal"]["ci"], 4) + (" ✔" if h6["signal"]["ci"][0] > 0 else " ✘"), "lower bound > 0"],
           ["(b) share of out-of-support alarms in the other windows (hospital bootstrap)",
            ci_txt(h6["false_alarms"]["estimate"], *h6["false_alarms"]["ci"], 4) + (" ✔" if h6["false_alarms"]["ci"][1] <= 0.05 else " ✘"), "upper bound ≤ 0.05"],
           ["(c) ML monitor, CITL = 0 at two-sided 1 %: signal", ci_txt(h6["ml_signal"]["estimate"], *h6["ml_signal"]["ci"], 4), "reported"],
           ["(c) ML monitor: alarms in the other windows", ci_txt(h6["ml_false_alarms"]["estimate"], *h6["ml_false_alarms"]["ci"], 4), "reported"]])
    L += [f"{h6['hospitals']} hospitals.", ""]

    L += ["## 3. The arms' levels", "", "Means over windows (and orders). eICU: the mean over hospitals of each "
          "hospital's value (unweighted; descriptive, not pooled). The prevalence arm has no ranking metrics.", ""]
    for t, kd, key in CONF:
        keys = [key] if kd == "s" else [k for k in S if k.startswith(key.split("|")[0] + "|") and k.endswith("|" + key.split("|")[1])]
        L += [f"**{t}** ({len(keys)} unit{'s' if len(keys) > 1 else ''})", ""]
        rows = []
        for a in LEVEL_ARMS:
            vals = []
            for c, _ in LEVEL_COLS:
                xs = [S[k]["arms"][a][c] for k in keys if math.isfinite(S[k]["arms"][a][c])]
                vals.append(fmt(sum(xs) / len(xs)) if xs else "—")
            rows.append([a] + vals)
        table(L, ["arm"] + [n for _, n in LEVEL_COLS], rows)

    L += ["## 4. Against the other arms (reported with the same rules, not decided)", "",
          "eICU's H4b with the deviation in brackets where it changes.", ""]
    for t, kd, key in CONF:
        L += [f"**{t}**", ""]
        rows = []
        for c in COMPS:
            r = row_conf(ep, dev, kd, key, c)
            cells = [ci_txt(*r[k], 3) + (" ✔" if ok(k, *r[k][1:]) else "") for k in ("H1_auroc", "H1_amp", "H4_ece", "H4_trigger_share")]
            if kd == "e":
                r2 = row_conf(ep, dev, kd, key, c, deviation=True)
                if [key, c, "H4_trigger_share"] in dev["changed"]:
                    cells[3] += f" [{ci_txt(*r2['H4_trigger_share'], 3)}" + (" ✔]" if ok("H4_trigger_share", *r2["H4_trigger_share"][1:]) else "]")
            rows.append([c] + cells)
        table(L, ["comparator", NAMES["H1_auroc"], NAMES["H1_amp"], NAMES["H4_ece"], NAMES["H4_trigger_share"]], rows)

    s = S["mimic_temporal|all|N500"]
    L += ["## 5. MIMIC-IV temporal (N 500): by stretch and with W halved and doubled", ""]
    rows = []
    for name, sv in s["stretch"].items():
        c = sv["contrasts"][ARM5]
        rows.append([f"{name} ({sv['windows']} windows)"] + [ci_txt(c[k]["estimate"], *c[k]["ci"], 3) for k in ("H1_auroc", "H1_amp", "H4_ece", "H4_trigger_share")])
    for sc, v in s["sensitivity_W"].items():
        c = v["contrasts"][ARM5]
        rows.append([f"W × {sc} ({v['windows']} windows)"] + [ci_txt(c[k]["estimate"], *c[k]["ci"], 3) for k in ("H1_auroc", "H1_amp", "H4_ece")] + ["not re-read"])
    table(L, ["", NAMES["H1_auroc"], NAMES["H1_amp"], NAMES["H4_ece"], NAMES["H4_trigger_share"]], rows)

    L += ["## 6. Cross-cohort streams and the INSPIRE control (N 500; estimates, no decision rule)", ""]
    rows = []
    for st in PLAIN[1:]:
        r = from_stream(S, f"{st}|all|N500")
        rows.append([st] + [ci_txt(*r[k], 3) + (" ✔" if ok(k, *r[k][1:]) else "") for k in RULES])
    table(L, ["stream"] + [NAMES[k] for k in RULES], rows)
    L += ["Against the ideal arm (ceiling):", ""]
    rows = []
    for st in PLAIN[1:]:
        r = from_stream(S, f"{st}|all|N500", "lightgbm_ideal")
        rows.append([st] + [ci_txt(*r[k], 3) for k in ("H1_auroc", "H1_amp", "H4_ece", "H4_trigger_share")])
    table(L, ["stream", NAMES["H1_auroc"], NAMES["H1_amp"], NAMES["H4_ece"], NAMES["H4_trigger_share"]], rows)
    L += ["Post-boundary stretch (the first ⌈N/W⌉ windows after each boundary) against arm 5:", ""]
    rows = []
    for st in PLAIN[:4]:
        sv = S[f"{st}|all|N500"]["stretch"].get("post_boundary")
        if sv:
            c = sv["contrasts"][ARM5]
            rows.append([f"{st} ({sv['windows']} windows)"] + [ci_txt(c[k]["estimate"], *c[k]["ci"], 3) for k in ("H1_auroc", "H1_amp", "H4_ece")])
    table(L, ["stream", NAMES["H1_auroc"], NAMES["H1_amp"], NAMES["H4_ece"]], rows)

    L += ["**H5, recovery** (patients from a boundary to the first two consecutive windows within δ of the arm's stable "
          "level of the next segment; mean over boundaries and orders; censored = never recovered):", ""]
    rows = []
    for st in PLAIN[:4]:
        rec_ = S[f"{st}|all|N500"]["recovery"]
        cells = []
        for a in ("perisim_fifo", ARM5, "lightgbm_ideal"):
            for m in ("auroc", "citl"):
                pts = [r["patients"] for o in rec_[a][m] for r in o]
                cen = sum(r["censored"] for o in rec_[a][m] for r in o)
                cells.append(f"{sum(pts) / len(pts):.0f}" + (f" ({cen} cens.)" if cen else "") if pts else "—")
        rows.append([st] + cells)
    table(L, ["stream", "FIFO AUROC", "FIFO CITL", "arm 5 AUROC", "arm 5 CITL", "ideal AUROC", "ideal CITL"], rows)
    L += ["**Saw-tooth** (arm 5; per cycle between deployments: the window after the first minus the window before the next):", ""]
    rows = []
    for st in PLAIN:
        sw = S[f"{st}|all|N500"]["sawtooth"][ARM5]
        rows.append([st, fmt(sw["auroc"]["cycles"], 1), fmt(sw["auroc"]["mean_amplitude"], 4), fmt(sw["citl"]["mean_amplitude"], 3)])
    table(L, ["stream", "cycles per order", "AUROC amplitude", "CITL amplitude"], rows)

    L += ["## 7. H7: the window N", "", "H1's and H4's rules against arm 5 at every N (✔ both endpoints met).", ""]
    rows = []
    for st in PLAIN:
        for N in (250, 500, 1000, 2000):
            r = from_stream(S, f"{st}|all|N{N}")
            rows.append([st, N, "✔" if ok("H1_auroc", *r["H1_auroc"][1:]) and ok("H1_amp", *r["H1_amp"][1:]) else "✘",
                         "✔" if ok("H4_ece", *r["H4_ece"][1:]) and ok("H4_trigger_share", *r["H4_trigger_share"][1:]) else "✘",
                         ci_txt(*r["H1_auroc"], 3), ci_txt(*r["H4_ece"], 3)])
    for key in E:
        r = row_conf(ep, dev, "e", key)
        r2 = row_conf(ep, dev, "e", key, deviation=True)
        rows.append([key.split("|")[0] + " (pooled)", key.split("|")[1][1:],
                     "✔" if ok("H1_auroc", *r["H1_auroc"][1:]) and ok("H1_amp", *r["H1_amp"][1:]) else "✘",
                     "✔" if ok("H4_ece", *r["H4_ece"][1:]) and ok("H4_trigger_share", *r2["H4_trigger_share"][1:]) else "✘",
                     ci_txt(*r["H1_auroc"], 3) + f" (k {E[key]['hospitals']})", ci_txt(*r["H4_ece"], 3)])
    table(L, ["stream", "N", "H1", "H4 (eICU: H4b with the deviation)", NAMES["H1_auroc"], NAMES["H4_ece"]], rows)

    L += ["## 8. The update logs (mean per order)", ""]
    rows = []
    groups = [("MIMIC-IV temporal", ["mimic_temporal|all|N500"])] + \
             [(st, [f"{st}|all|N500"]) for st in PLAIN[1:]] + \
             [("eICU (a), N 100", [k for k in S if k.startswith("eicu_within|") and k.endswith("|N100")]),
              ("eICU (b), N 250", [k for k in S if k.startswith("eicu_deploy|") and k.endswith("|N250")])]
    for name, keys in groups:
        for a in (ARM5, "lightgbm_ideal"):
            u = [S[k]["updates"][a] for k in keys]
            ch, gl = {}, {}
            for x in u:
                for c_, n in x["choices"].items():
                    ch[c_] = ch.get(c_, 0) + n
                for c_, n in x["golive_choices"].items():
                    gl[c_] = gl.get(c_, 0) + n
            rows.append([name, a, fmt(sum(x["triggers_mean"] for x in u) / len(u), 2),
                         fmt(sum(x["deployments_mean"] for x in u) / len(u), 2),
                         ", ".join(f"{c_} {n}" for c_, n in sorted(ch.items(), key=lambda z: -z[1])),
                         ", ".join(f"{c_} {n}" for c_, n in sorted(gl.items(), key=lambda z: -z[1])),
                         fmt(sum(x["parameters_changed_mean"] for x in u) / len(u), 2),
                         fmt(sum(x["seconds_mean"] for x in u) / len(u), 2)])
    table(L, ["stream", "arm", "triggers", "deployments", "closed-test choices (all orders)", "go-live choices",
              "parameters changed", "seconds per update"], rows)

    L += ["## 9. PERISIM's diagnostics and the ML monitor's PSI", ""]
    rows = []
    for name, keys in groups:
        d1 = [S[k]["d1"] for k in keys]
        rate = sum(d["all"]["rate"] for d in d1) / len(d1)
        alarm = sum(d["all"]["alarm_share"] for d in d1) / len(d1)
        post = [d["post_boundary"]["rate"] for d in d1 if "post_boundary" in d]
        rl = [S[k]["relearn"] for k in keys]
        ood = sum(1 for r in rl for x in r["ood_first_window_by_order"] if x is not None)
        d2 = sum(len(r["redeform_hits"]) for r in rl)
        psi = [S[k]["psi"]["all"] for k in keys if math.isfinite(S[k]["psi"]["all"])]
        rows.append([name, fmt(rate, 4), fmt(sum(post) / len(post), 4) if post else "—", fmt(alarm, 3), ood, d2,
                     fmt(sum(psi) / len(psi), 3) if psi else "—"])
    table(L, ["stream", "D1 rate", "D1 rate, post-boundary", "D1 alarm share", "re-learn by D1 (orders)",
              "re-learn by D2 (windows)", "PSI (mean over inputs)"], rows)
    rd2 = S["mimic_temporal|all|N500"]["redeform"]
    if rd2:
        L += [f"D2 in MIMIC-IV temporal (order 0): {len(rd2)} re-deformations; gain (nats) median "
              f"{sorted(r['gain'] for r in rd2)[len(rd2) // 2]:.4f}, min {min(r['gain'] for r in rd2):.4f}, "
              f"max {max(r['gain'] for r in rd2):.4f}.", ""]

    L += ["## 10. T2 (secondary, descriptive): MIMIC-IV temporal with T2's frozen method, N 2,000, W 250", ""]
    rows = []
    for m in ("logloss", "rps"):
        for a, v in t2["arms"][m].items():
            c = t2["contrasts"][m].get(a)
            rows.append([m, a, f"{v['mean']:.4f} (SD {v['sd']:.4f})", ci_txt(c["estimate"], *c["ci"], 4) if c else "—"])
    table(L, ["score", "arm", "mean over windows", "PERISIM-FIFO − arm (95 % CI)"], rows)
    L += [f"{t2['orders']} orders, {t2['windows']} windows.", ""]

    (OUT / "reports" / "endpoints.md").write_text("\n".join(L), encoding="utf-8")
    check(cfg, cs)
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
