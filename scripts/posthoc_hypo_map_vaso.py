"""Post-hoc, EXPLORATORY follow-up of `hypo`: does the MAP–risk relation change with a vasopressor running? (not
pre-registered; plan fixed before running, docs/DECISIONS.md 2026-11-04). Reads only; the freeze record is checked at
the start and the end.

On fold 0's training landmarks of MIMIC-IV (primary) and eICU (descriptive), split by `vaso_on`:
1. observed rates of hypotension in the next 6 h by 5-mmHg bins of the last MAP and the 6-h minimum, per stratum, with
   patient-resample CIs;
2. the consensus shapes of each stratum by S5's procedure (hypo.drift._shapes: the 40-model shape loop), on a learning
   sample of at most 3 landmarks per patient; the no-vasopressor patients split into two random halves whose samples
   are drawn to the vasopressor sample's size. Per MAP summary, S5's comparison (weighted r and slope of one log-odds
   curve on the other) for vaso–no-vaso a, vaso–no-vaso b and no-vaso a–b (the noise); the MAP block of §13.2.

Usage:  python scripts/posthoc_hypo_map_vaso.py [--sites mimic eicu] [--threads 16]     (run_local.ps1 hypo-map-vaso)

Fits cached in work/experiments/hypo/posthoc_map_vaso/fits.pkl (never exported); outputs (aggregates only):
posthoc_map_vaso/map_vaso.json, reports/posthoc_map_vaso.md and reports/posthoc_map_vaso/<site>.png.
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
import _frozen  # noqa: E402

_frozen.use_src()      # v0.6: $PERISIM_SRC (a frozen experiment's src) first on sys.path, else ROOT/src

from hypo import drift as D  # noqa: E402
from hypo.data import learning_sample  # noqa: E402
from hypo.external import Stage  # noqa: E402
from hypo.frozen import check, curve_table, dump, external_dir, load_pickle, thresholds  # noqa: E402
from hypo.stats import PatientBootstrap, ci  # noqa: E402

LOG = logging.getLogger("hypo.map_vaso")
MAPS = ["map_last", "map_mean_6h", "map_min_6h", "map_sd_6h", "map_slope_6h"]
RATE_VARS = ["map_last", "map_min_6h"]
EDGES = np.r_[-np.inf, np.arange(55.0, 100.0001, 5.0), np.inf]
MIN_BIN = 100
GROUPS = ("vaso", "novaso_a", "novaso_b")
PAIRS = [("vaso", "novaso_a"), ("vaso", "novaso_b"), ("novaso_a", "novaso_b")]
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"vaso": "#eb6834", "novaso_a": "#2a78d6", "novaso_b": "#2a78d6"}
LABEL = {"vaso": "vasopressor running", "novaso_a": "no vasopressor (half a)", "novaso_b": "no vasopressor (half b)"}


def out_dir() -> Path:
    return external_dir().with_name("posthoc_map_vaso")


def report_dir() -> Path:
    return external_dir().parent / "reports"


def samples(st: Stage, X: str, tr: np.ndarray, vaso: np.ndarray) -> dict[str, np.ndarray]:
    """Rows (positions in the site's landmarks) of the three learning samples."""
    lm = st.S.lm(X)
    per_p, per_s = int(st.mcfg["learn"]["per_patient"]), int(st.mcfg["learn"]["per_site"])
    seed = st.seed + 300 + st.ci(X)
    r1 = tr[vaso[tr] == 1]
    out = {"vaso": r1[learning_sample(lm.take(r1), per_p, per_s, seed)]}
    n = out["vaso"].size
    r0 = tr[vaso[tr] == 0]
    s0 = lm.ft.subjects[r0].astype(str)
    pats = np.random.default_rng([st.seed, 53, st.ci(X)]).permutation(np.unique(s0))
    for h, part in (("novaso_a", pats[: pats.size // 2]), ("novaso_b", pats[pats.size // 2:])):
        rh = r0[np.isin(s0, part)]
        rows = rh[learning_sample(lm.take(rh), per_p, per_s, seed + (1 if h.endswith("a") else 2))]
        pick = np.random.default_rng([st.seed, 59, st.ci(X), len(h)]).choice(rows.size, size=min(n, rows.size),
                                                                             replace=False)
        out[h] = np.sort(rows[pick])
    return out


def rates(st: Stage, X: str, tr: np.ndarray, vaso: np.ndarray) -> dict:
    """Observed rate of the outcome by bin of the last MAP and the 6-h minimum, per stratum (patient resamples)."""
    lm = st.S.lm(X)
    out = {}
    for s, val in (("vaso", 1), ("novaso", 0)):
        r = tr[vaso[tr] == val]
        y = lm.y[r].astype(float)
        boot = PatientBootstrap(lm.ft.subjects[r], int(st.x["n_boot"]), st.seed + 61 + st.ci(X) + 7 * val)
        Wp = np.array([wp for wp in boot])
        out[s] = {"n": int(r.size), "rate": float(y.mean()), "by": {}}
        for v in RATE_VARS:
            x = lm.ft.num[v][r]
            b = np.digitize(x, EDGES[1:-1])
            cells = []
            for i in range(EDGES.size - 1):
                m = (b == i) & np.isfinite(x)
                if m.sum() < MIN_BIN:
                    continue
                num = Wp @ np.bincount(boot.inv[m], weights=y[m], minlength=boot.uniq.size)
                den = Wp @ np.bincount(boot.inv[m], minlength=boot.uniq.size).astype(float)
                lo, hi = (float(e) if np.isfinite(e) else None for e in (EDGES[i], EDGES[i + 1]))
                cells.append({"lo": lo, "hi": hi, "n": int(round(m.sum(), -1)),
                              "rate": float(y[m].mean()), "ci": ci(num / den)})
            out[s]["by"][v] = cells
    return out


def site_part(st: Stage, X: str, fits: dict, cache: Path) -> dict:
    import posthoc_hypo as PH

    t = time.perf_counter()
    lm = st.S.lm(X)
    tr, _ = st.S.split(X, int(st.x["drift"]["fold"]))
    vaso = lm.ft.num["vaso_on"]
    S = samples(st, X, tr, vaso)
    tcfg, min_cell = st.x["threshold"], int(st.x["min_cell"])
    encs, res = {}, {"site": X, "fold": int(st.x["drift"]["fold"]), "rates": rates(st, X, tr, vaso), "groups": {}}
    for g in GROUPS:
        key = f"{X}:{g}"
        if key not in fits:
            LOG.info("%s: shapes of %s (%d landmarks)", X, g, S[g].size)
            enc, loop, dropped = D._shapes(st, lm.take(S[g]), st.seed + 400 + 10 * st.ci(X) + GROUPS.index(g))
            fits[key] = {"enc": enc, "curves": loop["curves"], "n_accepted": loop["n_accepted"], "dropped": dropped}
            dump(fits, cache)
        f = fits[key]
        encs[g] = f["enc"]
        fi = f["enc"].feature_index()
        res["groups"][g] = {
            "n_learn": int(S[g].size), "n_patients": int(np.unique(lm.ft.subjects[S[g]].astype(str)).size),
            "prevalence": float(lm.y[S[g]].mean()), "n_accepted": f["n_accepted"], "dropped_constant": f["dropped"],
            "thresholds": thresholds(f["enc"], f["curves"], None, tcfg),
            "block": PH.map_block(f["enc"], f["curves"], tcfg),
            "shapes": {v: curve_table(f["enc"], v, f["enc"].blocks[fi[v]].shape, min_cell)
                       for v in MAPS if v in fi and f["enc"].blocks[fi[v]].shape is not None}}
    res["pairs"] = {f"{a}|{b}": {v: c for v, c in D._compare(encs[a], encs[b], min_cell).items() if v in MAPS}
                    for a, b in PAIRS}
    res["seconds"] = time.perf_counter() - t
    return res


# --------------------------------------------------------------------------- figure and report
def figure(r: dict, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), facecolor=SURFACE)
    ax = axs[0]
    for s, col, lab in (("novaso", COL["novaso_a"], "no vasopressor"), ("vaso", COL["vaso"], "vasopressor running")):
        cells = r["rates"][s]["by"]["map_last"]
        x = [c["lo"] + 2.5 if c["lo"] is not None else c["hi"] - 2.5 for c in cells]
        yv = [100 * c["rate"] for c in cells]
        lo = [100 * max(0.0, c["rate"] - c["ci"][0]) for c in cells]
        hi = [100 * max(0.0, c["ci"][1] - c["rate"]) for c in cells]
        ax.errorbar(x, yv, yerr=[lo, hi], color=col, lw=2, marker="o", ms=4, capsize=0, label=lab)
    ax.set_title("observed: hypotension in 6 h by last MAP", fontsize=9)
    ax.set_xlabel("last MAP (mmHg, 5-mmHg bins)")
    ax.set_ylabel("% of landmarks")
    ax.legend(frameon=False, fontsize=8)
    for ax, v in zip(axs[1:], ["map_last", "map_mean_6h", "map_min_6h"]):
        for g in GROUPS:
            tab = r["groups"][g]["shapes"].get(v, [])
            if not tab:
                continue
            ax.plot([c["x"] for c in tab], [c["logodds"] for c in tab], color=COL[g], lw=2,
                    ls="--" if g == "novaso_b" else "-", label=LABEL[g])
        ax.set_title(f"shape of {v} (centred log-odds)", fontsize=9)
        ax.set_xlabel(f"{v} (mmHg)")
        ax.set_xlim(45, 110)
    axs[1].legend(frameon=False, fontsize=8)
    for ax in axs:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID_C, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{r['site']}: the MAP–risk relation with and without a vasopressor running (fold 0, exploratory)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def _cmp(c: dict | None) -> str:
    return f"{c['r']:.2f} / {c['slope']:.2f}" if c else "—"


def _blk(b: dict, k: str, d: int = 1) -> str:
    if "consensus" not in b:
        return "—"
    m = b.get("models", {}).get(k)
    tail = f" ({m['median']:.{d}f}; {m['p10']:.{d}f}–{m['p90']:.{d}f})" if m else ""
    return f"{b['consensus'][k]:.{d}f}{tail}"


def render(R: dict) -> str:
    L = ["# `hypo` post hoc: the MAP–risk relation with and without a vasopressor running (exploratory)", "",
         "Plan: docs/DECISIONS.md 2026-11-04. Fold 0's training landmarks. Shapes by S5's procedure on equal-size "
         "learning samples; r / slope = weighted correlation and slope of the first curve on the second (S5's "
         "comparison). Block: threshold (mmHg) and odds ratios of a steady MAP against 85 mmHg, consensus (median; "
         "10–90 % over the 40 models).", ""]
    for X, r in R.items():
        L += [f"## {X}", "", f"![{X}](posthoc_map_vaso/{X}.png)", "",
              "| group | landmarks (patients) | prevalence | block threshold | OR 70 vs 85 | OR 65 vs 85 |",
              "|---|---|---|---|---|---|"]
        for g in GROUPS:
            x = r["groups"][g]
            L.append(f"| {LABEL[g]} | {x['n_learn']:,} ({x['n_patients']:,}) | {x['prevalence']:.1%} | "
                     f"{_blk(x['block'], 'threshold')} | {_blk(x['block'], 'or_70')} | {_blk(x['block'], 'or_65')} |")
        L += ["", "| MAP summary | vaso – no-vaso a | vaso – no-vaso b | no-vaso a – b (noise) |", "|---|---|---|---|"]
        for v in MAPS:
            L.append(f"| {v} | " + " | ".join(_cmp(r["pairs"][f"{a}|{b}"].get(v)) for a, b in PAIRS) + " |")
        L += ["", "Per-variable thresholds (consensus; median and 10–90 % over the models):", "",
              "| group | " + " | ".join(r["groups"]["vaso"]["thresholds"]) + " |",
              "|---|" + "---|" * len(r["groups"]["vaso"]["thresholds"])]
        for g in GROUPS:
            th = r["groups"][g]["thresholds"]
            L.append(f"| {LABEL[g]} | " + " | ".join(
                f"{t['consensus']:.1f} ({t['median']:.1f}; {t['p10']:.1f}–{t['p90']:.1f})" for t in th.values()) + " |")
        for v in RATE_VARS:
            L += ["", f"Observed rate of hypotension in the next 6 h by {v} (95 % CI):", "",
                  "| bin (mmHg) | no vasopressor | vasopressor running |", "|---|---|---|"]
            cells = {s: {(c["lo"], c["hi"]): c for c in r["rates"][s]["by"][v]} for s in ("novaso", "vaso")}
            for key in sorted(set(cells["novaso"]) | set(cells["vaso"]), key=lambda k: -1e9 if k[0] is None else k[0]):
                lab = f"< {key[1]:.0f}" if key[0] is None else (f"≥ {key[0]:.0f}" if key[1] is None
                                                                  else f"{key[0]:.0f}–{key[1]:.0f}")
                row = []
                for s in ("novaso", "vaso"):
                    c = cells[s].get(key)
                    row.append(f"{c['rate']:.1%} ({c['ci'][0]:.1%}–{c['ci'][1]:.1%}; n {c['n']:,})" if c else "—")
                L.append(f"| {lab} | " + " | ".join(row) + " |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="*", default=["mimic", "eicu"])
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    rec = check()
    st = Stage(threads=a.threads)
    out_dir().mkdir(parents=True, exist_ok=True)
    cache = out_dir() / "fits.pkl"
    fits = load_pickle(cache) if cache.exists() else {}
    R = {X: site_part(st, X, fits, cache) for X in a.sites}
    check()
    for X, r in R.items():
        r["figure"] = figure(r, report_dir() / "posthoc_map_vaso" / f"{X}.png")
    (out_dir() / "map_vaso.json").write_text(json.dumps({"freeze": rec["tag"], "sites": R}, indent=1, default=float),
                                             encoding="utf-8")
    rep = report_dir() / "posthoc_map_vaso.md"
    rep.write_text(render(R), encoding="utf-8")
    LOG.info("MAP–vasopressor report: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
