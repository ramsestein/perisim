"""Declared deviation of `roll` (decided by the lead on 2026-10-31, after the single run; docs/DECISIONS.md): the
eICU pooling with standard errors below a numerical tolerance treated as zero.

The frozen DerSimonian–Laird pooling (`roll.endpoints.dersimonian_laird`) leaves out hospitals whose bootstrap SE is
zero (`se > 0`). Floating-point noise can leave an SE of ~1e-16 where every resample gives the same value. Such a
hospital passes the filter and gets an almost infinite weight, so the pooled estimate becomes that one hospital's,
with a CI of zero width (H4 (b) in eICU (a) and (b)). Here every SE ≤ TOL is set to zero before the same frozen
function is called, so those hospitals are left out as the frozen code intends. Nothing else changes: the
per-hospital estimates and SEs are read from the frozen output `endpoints.json`.

Usage:  python scripts/roll_deviation_pooling.py

Outside src/roll on purpose (the frozen code hash covers it). Reads and writes aggregates only:
<workdir>/experiments/roll/endpoints_deviation.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roll import run as R  # noqa: E402
from roll.config import workdir  # noqa: E402
from roll.endpoints import H_KEYS, dersimonian_laird  # noqa: E402
from roll.frozen import check  # noqa: E402

TOL = 1e-12


def repool(hospitals: dict, h2: bool) -> tuple[dict, list[str]]:
    labs = list(hospitals)
    y = np.array([hospitals[h]["estimate"] for h in labs], float)
    se = np.array([hospitals[h]["se"] for h in labs], float)
    tiny = [h for h, s in zip(labs, se) if np.isfinite(s) and 0 < s <= TOL]
    se = np.where(np.isfinite(se) & (se <= TOL), 0.0, se)
    p = dersimonian_laird(y, se)
    if h2 and p.get("k"):
        p = {**p, "ratio": math.exp(p["estimate"]), "ratio_ci": [math.exp(x) for x in p["ci"]]}
    return p, tiny


def same(a: dict, b: dict) -> bool:
    if a.get("k") != b.get("k"):
        return False
    return all(math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-12) for k in ("estimate", "se") if k in a)


def main() -> int:
    cfg, _, _, _, cs = R._real()
    rec = check(cfg, cs)
    root = workdir()
    src = root / "endpoints.json"
    ep = json.loads(src.read_text(encoding="utf-8"))
    out = {"note": "declared deviation (DECISIONS 2026-10-31): eICU pooling with SE <= tolerance treated as zero",
           "tolerance": TOL, "tag": rec["tag"], "source": src.name, "computed": time.strftime("%Y-%m-%d %H:%M:%S"),
           "eicu": {}, "changed": []}
    for key, v in ep["eicu"].items():
        o = {"hospitals": v["hospitals"], "pooled": {}}
        for comp, P in v["pooled"].items():
            o["pooled"][comp] = {}
            for hk in H_KEYS:
                p, tiny = repool(P[hk]["hospitals"], hk.startswith("H2"))
                o["pooled"][comp][hk] = {"pooled": p, "left_out": tiny}
                if not same(p, P[hk]["pooled"]):
                    out["changed"].append([key, comp, hk])
        if "h6" in v:
            name, N = key.split("|")
            per = {k.split("|")[1]: s["h6"] for k, s in ep["streams"].items()
                   if k.startswith(f"{name}|") and k.endswith(f"|{N}") and "h6" in s}
            hs = {h: {"estimate": x["signal"][0], "se": x["signal"][1]} for h, x in per.items()}
            p, tiny = repool(hs, False)
            o["h6_signal"] = {"pooled": p, "left_out": tiny}
            if not same(p, v["h6"]["signal"]):
                out["changed"].append([key, "", "h6_signal"])
        out["eicu"][key] = o
    check(cfg, cs)
    dst = root / "endpoints_deviation.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(dst)
    print(f"{len(out['changed'])} pooled results change:")
    def f(q):
        est, ci = q.get("ratio", q.get("estimate", math.nan)), q.get("ratio_ci", q.get("ci", [math.nan] * 2))
        return f"{est:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] k={q.get('k')}"

    for key, comp, hk in out["changed"]:
        if hk == "h6_signal":
            a, b = ep["eicu"][key]["h6"]["signal"], out["eicu"][key]["h6_signal"]
        else:
            a, b = ep["eicu"][key]["pooled"][comp][hk]["pooled"], out["eicu"][key]["pooled"][comp][hk]
        print(f"  {key} {comp} {hk}: frozen {f(a)}  ->  {f(b['pooled'])}  (left out {b['left_out']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
