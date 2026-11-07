"""Two checks raised by the audit of 2026-11-07 (`hypo`; frozen code; aggregates only).

1. Effective weight shares. The raw NCA weight share w_j / Σw is not comparable across variables or metrics when the
   axes' scales differ between the development data and the target. The effective share of variable j in a metric
   applied to site X is its share of the weighted squared distance between random pairs of X's analysis landmarks:
   E[w_j Δ_j²] / E[Σ_i w_i Δ_i²]. Reported for every input of the three transported metrics, at their targets.
2. eICU patients in several hospitals. For the ten hospitals of REPORT §20: how many target-hospital patients also
   have landmarks in another eICU hospital (those landmarks entered the other-hospital NCA learning pool).

Usage:  python scripts/posthoc_audit_checks.py
Output: work/experiments/hypo/reports/posthoc_audit_checks.md
"""

from __future__ import annotations

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

N_PAIRS = 200_000
SEED = 20261107


def effective_shares(st, X: str) -> dict:
    m = st.metric(X)
    E = np.asarray(st.embed(X, X), dtype=np.float64)
    sw = np.asarray(m.sqrt_w_dims, dtype=np.float64)
    df = np.asarray(m.enc.dim_feature)
    rng = np.random.default_rng(SEED)
    a, b = rng.integers(0, E.shape[0], N_PAIRS), rng.integers(0, E.shape[0], N_PAIRS)
    D = ((E[a] - E[b]) * sw) ** 2
    per_dim = D.mean(0)
    names = list(m.enc.feature_names)
    eff = np.zeros(len(names))
    np.add.at(eff, df, per_dim)
    eff = eff / eff.sum()
    raw = np.asarray(m.w, dtype=np.float64) / np.sum(m.w)
    return {n: (float(raw[i]), float(eff[i])) for i, n in enumerate(names)}


def multi_hospital(st) -> dict:
    import json
    lm = st.S.lm("eicu")
    hosp, subj = lm.hospital.astype(str), lm.ft.subjects.astype(str)
    sizes = {h: np.unique(subj[hosp == h]).size for h in np.unique(hosp)}
    targets = [h for h, n in sorted(sizes.items(), key=lambda t: -t[1]) if n >= 500][:10]
    out = {}
    for r, h in enumerate(targets, start=1):
        s_in = set(np.unique(subj[hosp == h]))
        s_out = set(np.unique(subj[hosp != h]))
        both = len(s_in & s_out)
        out[f"h{r}"] = {"patients": len(s_in), "also_elsewhere": both if both == 0 or both >= 10 else "1-9"}
    return out


def main() -> int:
    from hypo.external import Stage
    from hypo.frozen import check, external_dir

    check()
    st = Stage(quick=False, threads=8)
    L = ["# Audit checks (2026-11-07)", "", "## 1. Effective weight shares at each target (random pairs of analysis landmarks)",
         "", "| input | " + " | ".join(f"{X}: raw / effective" for X in ("inspire", "mimic", "eicu")) + " |",
         "|---|---|---|---|"]
    sh = {X: effective_shares(st, X) for X in ("inspire", "mimic", "eicu")}
    names = list(dict.fromkeys(n for X in sh for n in sh[X]))
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{sh[X][n][0]:.2%} / {sh[X][n][1]:.2%}" if n in sh[X] else "—"
                                          for X in ("inspire", "mimic", "eicu")) + " |")
    grp = {"vasopressor (vaso_on, vaso_24h, vaso_hours)": ["vaso_on", "vaso_24h", "vaso_hours"],
           "sedation (sedation_on, sedation_24h)": ["sedation_on", "sedation_24h"], "ventilation (vent_on)": ["vent_on"]}
    L += ["", "| group | " + " | ".join(f"{X}: raw / effective" for X in ("inspire", "mimic", "eicu")) + " |", "|---|---|---|---|"]
    for g, vs in grp.items():
        L.append(f"| {g} | " + " | ".join(
            f"{sum(sh[X][v][0] for v in vs if v in sh[X]):.2%} / {sum(sh[X][v][1] for v in vs if v in sh[X]):.2%}"
            for X in ("inspire", "mimic", "eicu")) + " |")
    L += ["", "## 2. eICU target hospitals (REPORT §20): patients with landmarks in another eICU hospital", "",
          "| hospital | patients | also in another hospital |", "|---|---|---|"]
    for h, d in multi_hospital(st).items():
        L.append(f"| {h} | {d['patients']:,} | {d['also_elsewhere']} |")
    check()
    rep = external_dir().parent / "reports" / "posthoc_audit_checks.md"
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"audit checks: {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
