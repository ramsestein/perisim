"""Post-hoc, PRE-SPECIFIED sensitivity of the intervention strata to v0.5's landmark leak (`hypo` REPORT §19; plan:
docs/DECISIONS.md 2026-11-07).

In v0.5's landmark builder, charted support points were merged into runs over the whole stay before the cut at L, so
a record after L could mark `vaso_on`, `sedation_on` or `vent_on` at L (audit 2026-11-05, item 1). v0.6 rebuilds
the supports known at L from the records before L only. This script recomputes the three states that way from the
v0.5 extract's supports, and re-runs §19's test (scripts/posthoc_state_test.py) with the strata and the same-state
neighbours (B0) defined by them. Everything else is as in §19: the frozen metric's predictions, its inputs (where
`vaso_on` weighs 0.0008–0.02) and the comparators are v0.5's and unchanged.

v0.6 also reads eICU's rate-0 infusion records as stop markers; the v0.5 extract does not carry them, so that part
of the fix is not applied here (a limit, stated in the plan).

On at L (v0.6's rule, `landmarks.SupportsBefore` with `support_lookback_h` = 2): some support record of that kind
with h < L whose coverage clipped at L, floor(min(h_end, L)), reaches hour max(L − 2, 0). For charted points
(h_end = h) the run they belong to does not matter: the latest hour a run covers, as known at L, is its last point
before L.

Usage:  python scripts/posthoc_leak_sensitivity.py [--sites inspire mimic eicu] [--threads 8]
Outputs: work/experiments/hypo/posthoc_leak_sensitivity/leak.json, reports/posthoc_leak_sensitivity.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import _frozen  # noqa: E402
    _frozen.use_src()
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

import posthoc_state_test as T  # noqa: E402

LOG = logging.getLogger("posthoc.leak")
KIND = {"vaso_on": "vaso", "sedation_on": "sedation", "vent_on": "vent"}


def states_before(supports, keys, look: int = 2) -> dict:
    """{var: bool array in `keys` order}: the support running at L as known at L. `supports`: polars frame with
    stay_id, kind, h, h_end; `keys`: polars frame with stay_id, L (the landmarks' order)."""
    import polars as pl

    s = supports.select(pl.col("stay_id").cast(pl.Utf8), "kind", pl.col("h").cast(pl.Float64),
                        pl.col("h_end").cast(pl.Float64))
    k = keys.select(pl.col("stay_id").cast(pl.Utf8), pl.col("L").cast(pl.Int64)).with_row_index("pos")
    out = {v: np.zeros(k.height, dtype=bool) for v in KIND}
    for L in k["L"].unique().sort().to_list():
        sub = k.filter(pl.col("L") == L)
        cov = (s.filter(pl.col("h") < L)
               .with_columns(pl.min_horizontal("h_end", pl.lit(float(L))).floor().alias("last_h"))
               .filter(pl.col("last_h") >= max(L - look, 0))
               .select("stay_id", "kind").unique())
        for var, kind in KIND.items():
            on = cov.filter(pl.col("kind") == kind).select("stay_id").with_columns(pl.lit(True).alias("on"))
            j = sub.join(on, on="stay_id", how="left")
            out[var][j["pos"].to_numpy()] = j["on"].fill_null(False).to_numpy()
    return out


def site_states(X: str, lm, cfg: dict) -> dict:
    """v0.6 states for the analysis landmarks of site X, in `lm`'s row order (checked against lm's ids)."""
    import polars as pl
    from hypo.data import workdir

    wd = workdir()
    df = pl.read_parquet(wd / "landmarks" / f"{X}.parquet").filter(pl.col("normotensive") & pl.col("known")) \
        .sort("stay_id", "L")
    ids = df.select(pl.concat_str([pl.col("stay_id").cast(pl.Utf8), pl.lit(":"), pl.col("L").cast(pl.Utf8)]))
    if not np.array_equal(ids.to_series().to_numpy().astype(str), lm.ft.ids.astype(str)):
        raise RuntimeError(f"leak {X}: the landmarks file does not match the analysis landmarks")
    sup = pl.read_parquet(wd / "extract" / f"{X}_supports.parquet")
    look = int(cfg["landmarks"]["support_lookback_h"])
    st = states_before(sup, df.select("stay_id", "L"), look)
    for var in KIND:                                        # inputs the source cannot record stay "off"
        v05 = np.asarray(lm.ft.num[var], dtype=np.float64)
        st[var] = np.where(np.isfinite(v05), st[var], False)
    return st


def reclass(v05: np.ndarray, v06: np.ndarray) -> dict:
    return {"share_on_v05": float(v05.mean()), "share_on_v06": float(v06.mean()),
            "on_to_off": float((v05 & ~v06).mean()), "off_to_on": float((~v05 & v06).mean())}


def compare(old: dict, new: dict) -> dict:
    """Pre-specified reading: each (site, state) verdict of §19 unchanged?"""
    out = {}
    for X, r in new.items():
        for var, s in r["states"].items():
            o = old.get(X, {}).get("states", {}).get(var)
            if o is None:
                out[f"{X}|{var}"] = {"v05": None, "v06": s["verdict"], "unchanged": False}
                continue
            out[f"{X}|{var}"] = {"v05": o["verdict"], "v06": s["verdict"], "unchanged": o["verdict"] == s["verdict"]}
    for X, r in old.items():                  # a cell that drops out under the leak-free strata is a change too
        for var, o in r.get("states", {}).items():
            if f"{X}|{var}" not in out:
                out[f"{X}|{var}"] = {"v05": o["verdict"], "v06": None, "unchanged": False}
    return {"per_cell": out, "all_unchanged": bool(out) and all(v["unchanged"] for v in out.values())}


def render(res: dict, base: str) -> str:
    L = ["# §19 with the intervention strata rebuilt without the landmark leak", "",
         "Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. States at L from records before L only "
         "(v0.6's rule; eICU's rate-0 stop markers not applied). Predictions and comparators unchanged.", "",
         "## Reclassification of the states", "", "| site | state | on, v0.5 | on, v0.6 rule | on → off | off → on |",
         "|---|---|---|---|---|---|"]
    for X, d in res["reclass"].items():
        for var, r in d.items():
            L.append(f"| {X} | {var} | {r['share_on_v05']:.1%} | {r['share_on_v06']:.1%} | {r['on_to_off']:.1%} | "
                     f"{r['off_to_on']:.1%} |")
    L += ["", "## Verdicts, v0.5 strata against leak-free strata", "", "| site · state | v0.5 | leak-free | unchanged |",
          "|---|---|---|---|"]
    f = lambda v: "absent" if v is None else f"failure {'yes' if v['failure'] else 'no'}, repairs {'yes' if v['repairs'] else 'no'}, " \
                  f"safe {'yes' if v['safe'] else 'no'}"  # noqa: E731
    for k, v in res["compare"]["per_cell"].items():
        L.append(f"| {k} | {f(v['v05'])} | {f(v['v06'])} | {'yes' if v['unchanged'] else '**no**'} |")
    L += ["", f"All verdicts unchanged: **{'yes' if res['compare']['all_unchanged'] else 'no'}**", "",
          "## Full test with the leak-free strata", "", base]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="*", default=["inspire", "mimic", "eicu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        stream=sys.stdout)
    from hypo.external import Stage
    from hypo.frozen import check, external_dir

    rec = check()
    st = Stage(quick=a.quick, threads=a.threads)
    od = external_dir().with_name("posthoc_leak_sensitivity")
    (od / "units").mkdir(parents=True, exist_ok=True)
    v06 = {X: site_states(X, st.S.lm(X), st.S.cfg) for X in a.sites}
    res = {"freeze": rec.get("tag"), "reclass": {}, "sites": {}}
    for X in a.sites:
        lm = st.S.lm(X)
        res["reclass"][X] = {var: reclass(T.state_of(lm, var), v06[X][var]) for var in KIND
                             if np.isfinite(np.asarray(lm.ft.num[var], dtype=float)).any()}
    orig = T.state_of
    T.state_of = lambda lm, var: v06[lm.site[0]][var]        # the leak-free strata for §19's code
    try:
        for X in a.sites:
            res["sites"][X] = T.run_site(st, X, od / "units", list(T.STATES))
    finally:
        T.state_of = orig
    check()
    res["weights"] = {var: {X: T.weight_share(st.metric(X).enc, st.metric(X).w, var) for X in a.sites}
                      for var in T.STATES}
    res["rule_19"] = T.rule(res["sites"], res["weights"])
    old_path = external_dir().with_name("posthoc_state_test") / "state_test.json"
    if not old_path.exists():                 # audit 2026-11-07: no vacuous "all unchanged"
        raise RuntimeError(f"leak: {old_path} (§19's run) is needed for the comparison")
    old = json.loads(old_path.read_text(encoding="utf-8"))["sites"]
    res["compare"] = compare(old, res["sites"])
    (od / "leak.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    base = T.render({"sites": res["sites"], "rule": res["rule_19"], "weights": res["weights"]})
    rep = external_dir().parent / "reports" / "posthoc_leak_sensitivity.md"
    rep.write_text(render(res, base), encoding="utf-8")
    LOG.info("leak sensitivity: %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
