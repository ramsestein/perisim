"""PM-kNN vs the implementation it was ported from (PERISIM's `perisim` and `hypo` packages), on synthetic data.

    python tools/compare_with_perisim.py <path to a PERISIM checkout (frozen hypo-era code, src/perisim + src/hypo)>

Prints, for the same data, seed and settings: the models' matrix, the models drawn and their validation log-loss, the
consensus curves and the NCA weights of both (A–C, one row per patient, one site), then the same with repeated
patients and two sites (D: group-level splits, site nuisance columns, inputs constant within every site, within-site
NCA candidates, and the case base: own-patient exclusion, distinct patients, α, out-of-support threshold).
Needs the PERISIM checkout's dependencies (torch among them).

Intended differences since pmknn 0.4.0 (PERISIM v0.6 method fixes), kept out of the comparison and reported apart:
- a discrete top value (≥ `min_bin` rows equal to the maximum) gets its own bin: the synthetic inputs are continuous
  with no ties at the maximum, so the bins are the same (checked in A and D);
- the flatness test of a continuous / ordinal curve reads its first differences: the curves are compared with the
  0.4.0 test switched off (`local_test=False`, the frozen test), and the decisions of the new test on the same
  models' curves are printed separately ("flatness, 0.4.0 test").
"""
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

if len(sys.argv) != 2:
    sys.exit(__doc__)
REPO = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(HERE / "src"), str(HERE / "tests")]

import numpy as np
import yaml

from perisim.features.build import FeatureSpec
from perisim.features.table import FeatureTable
from perisim.importance.shapes import consensus_shapes as p_consensus_shapes
from perisim.metric.encoding import Encoder as PEncoder
from perisim.metric.nca import DiagonalNCA as PNCA
from hypo.casebase import CaseBase as PCaseBase
from hypo.metric import Metric as PMetric, SiteNCA, _site_constant as p_site_constant, shape_loop

import pmknn
import pmknn.rashomon as c_rashomon
from pmknn.core import _site_constant as c_site_constant
from pmknn.encoding import Encoder as CEncoder
from pmknn.knn import CaseBase as CCaseBase
from pmknn.nca import DiagonalNCA as CNCA
from pmknn.rashomon import ShapeSettings, fit_rashomon
from pmknn.schema import Schema, Table
from pmknn.shapes import consensus_shapes as c_consensus_shapes
from synthetic import make_data

# the frozen flatness test inside the loop, for a like-for-like comparison of the curves
c_rashomon.consensus_shapes = partial(c_consensus_shapes, local_test=False)

INPUTS = {"u_shape": "continuous", "linear": "continuous", "noise": "continuous", "kind": "nominal",
          "flag": "binary", "lab": "continuous"}
space = yaml.safe_load(open(REPO / "config" / "model_space.yaml", encoding="utf-8"))
space = {"families": {k: v for k, v in space["families"].items() if k in ("logreg", "spline_logreg", "random_forest")}}
c_space = pmknn.ModelSpace.default(include=["logreg", "spline_logreg", "random_forest"])
scfg = dict(n_models=12, max_attempts_factor=3, epsilon=0.05, augment_copies=2, max_train=3000, max_val=1500,
            n_explain=1000, subsample=600, flat_alpha=0.05, corr_cluster_threshold=0.8, site_adjusted=True,
            n_reference_pairs=4000)
nca_kw = dict(n_candidates=200, batch_size=256, lr=0.05, refresh_every=5, label_smoothing=1e-3, seed=0)


def tables(df, group: str | None, site: str | None):
    sch = Schema.build(INPUTS, "y", group=group, site=site)
    sch.resolve(df)
    y = sch.labels(df)
    ct = Table.from_frame(df, sch)
    specs = [FeatureSpec(n, t, n, "x") for n, t in INPUTS.items()]
    num = {n: df[n].to_numpy(dtype=float) for n, t in INPUTS.items() if t != "nominal"}
    cat = {n: df[n].to_numpy(dtype=object) for n, t in INPUTS.items() if t == "nominal"}
    ids = np.arange(len(df)).astype(str)
    subjects = df[group].astype(str).to_numpy() if group else ids.copy()
    return y, ct, FeatureTable(ids, subjects, specs, num, cat), subjects


def no_ties_at_max(df) -> bool:
    return all(int((df[n] == df[n].max()).sum()) < 50 for n, t in INPUTS.items() if t == "continuous")


def compare_loop(tag, pe, ce, pft, ct, y, sites):
    """Shape loop of both (frozen flatness test) and the 0.4.0 flatness decisions on the same curves."""
    lm = SimpleNamespace(ft=pft, y=y, site=np.zeros(len(y), dtype=object) if sites is None else sites)
    rp = shape_loop(pe, lm, scfg, space, seed=3, threads=4)
    rc = fit_rashomon(ce, ct, y, c_space, ShapeSettings(**scfg), seed=3, n_jobs=4)
    vp = [r["val_logloss"] for r in rp["models"]]
    vc = [r["val_logloss"] for r in rc.attempts]
    print(f"{tag}. site nuisance columns: pmknn {rc.site_levels}")
    print(f"{tag}. same models drawn:", [r["family"] for r in rp["models"]] == [r["family"] for r in rc.attempts],
          "| val log-loss max |diff|:", float(np.max(np.abs(np.array(vp) - np.array(vc)))))
    for j in pe.shapeable_blocks():
        gp, gc = rp["shapes"].get(j), rc.shapes.get(j)
        name = pe.blocks[j].name
        if gp is None or gc is None:
            print(f"{tag}. {name}: shaped perisim={gp is not None} pmknn={gc is not None}")
        else:
            print(f"{tag}. {name}: g max |diff| = {float(np.abs(gp - gc).max()):.2e}")
    # flatness: the frozen test vs the 0.4.0 test (first differences) on the same competitive curves
    kept = {ce.feature_index()[n]: c for n, c in rc.curves.items()}
    new, _ = c_consensus_shapes(kept, ce, 0.05)
    for j in sorted(kept):
        b = ce.blocks[j]
        if b.kind in ("continuous", "ordinal"):
            print(f"{tag}. flatness, 0.4.0 test: {b.name}: shaped frozen={rc.shapes.get(j) is not None} "
                  f"0.4.0={new.get(j) is not None}")
    return rp, rc


# ------------------------------------------------------------------ A–C: one row per patient, one site
df = make_data(6000, seed=11)
y, ct, pft, ids = tables(df, None, None)
pe = PEncoder(geometry="shape", max_bins=20, min_bin=50).fit(pft, y, 2)
ce = CEncoder(max_bins=20, min_bin=50).fit(ct, y, 2)
print("A. no ties at the maximum (discrete top value not triggered):", no_ties_at_max(df))
print("A. blocks equal:", pe.feature_names == ce.feature_names)
Xp, op = pe.transform_model(pft)
Xc, oc = ce.transform_model(ct)
print("A. model matrix max |diff|:", float(np.abs(Xp - Xc).max()), "owners equal:", bool((op == oc).all()))
En_p, En_c = pe.transform(pft), ce.transform(ct)
print("A. neutral embedding max |diff| (ECDF approximation):", round(float(np.abs(En_p - En_c).max()), 4))

rp, rc = compare_loop("B", pe, ce, pft, ct, y, None)

pe2 = pe.with_shapes(rp["shapes"], pft)
E = pe2.transform(pft)
wp = PNCA(pe2.dim_feature, pe2.n_features, 2, eps0=1e-4, **nca_kw).fit(E, y, epochs=10, subjects=ids).weights
wc = CNCA(pe2.dim_feature, pe2.n_features, 2, **nca_kw).fit(E, y, epochs=10, groups=ids).weights
sp, sc = wp / wp.sum(), wc / wc.sum()
print("C. NCA weight shares perisim:", np.round(sp, 3))
print("C. NCA weight shares pmknn:  ", np.round(sc, 3))
print("C. max |share diff|:", round(float(np.abs(sp - sc).max()), 4),
      "| corr of log w:", round(float(np.corrcoef(np.log(wp), np.log(wc))[0, 1]), 4))

# ------------------------------------------------------------------ D: repeated patients, two sites
# 3 rows per patient; each patient belongs to one site; site s1 has a higher base rate (a difference the site
# nuisance columns must absorb) and a source gap: `lab` is never measured there, so its missing indicator is
# constant within every site and leaves the metric.
df = make_data(6000, seed=12, rows_per_patient=3)
rng = np.random.default_rng(12)
site_of = {p: ("s0" if rng.uniform() < 0.5 else "s1") for p in df["pid"].unique()}
df["site"] = df["pid"].map(site_of)
s1 = (df["site"] == "s1").to_numpy()
df.loc[s1 & (rng.uniform(size=len(df)) < 0.15), "y"] = 1
df.loc[s1, "lab"] = np.nan
df.loc[~s1, "lab"] = df.loc[~s1, "lab"].fillna(0.0)
sites = df["site"].to_numpy(dtype=object)
y, ct, pft, pid = tables(df, "pid", "site")
print("D. rows", len(df), "| patients", df["pid"].nunique(), "| sites", sorted(set(sites)),
      "| no ties at the maximum:", no_ties_at_max(df))
pe = PEncoder(geometry="shape", max_bins=20, min_bin=50).fit(pft, y, 2)
ce = CEncoder(max_bins=20, min_bin=50).fit(ct, y, 2)
dp = [pe.blocks[j].name for j in p_site_constant(pe, pe.transform(pft), sites)]
dc = c_site_constant(ce, ct)
print("D. constant within every site: perisim", dp, "pmknn", dc)
pe.blocks = [b for b in pe.blocks if b.name not in dp]
pe._index()
ce.drop(dc)
print("D. blocks equal:", pe.feature_names == ce.feature_names)
Xp, op = pe.transform_model(pft)
Xc, oc = ce.transform_model(ct)
print("D. model matrix max |diff|:", float(np.abs(Xp - Xc).max()))

rp, rc = compare_loop("D", pe, ce, pft, ct, y, sites)

pe2 = pe.with_shapes(rp["shapes"], pft)
E = pe2.transform(pft)
wp = SiteNCA(pe2.dim_feature, pe2.n_features, 2, eps0=1e-4, sites=sites, **nca_kw).fit(E, y, epochs=10,
                                                                                         subjects=pid).weights
wc = CNCA(pe2.dim_feature, pe2.n_features, 2, **nca_kw).fit(E, y, epochs=10, groups=pid, sites=sites).weights
sp, sc = wp / wp.sum(), wc / wc.sum()
print("D. NCA (within-site candidates, own patient excluded) max |share diff|:", round(float(np.abs(sp - sc).max()), 4),
      "| corr of log w:", round(float(np.corrcoef(np.log(wp), np.log(wc))[0, 1]), 4))

# case base of one site (s0), queried by its own rows (own patient excluded) and by the other site's rows
mcfg = {"k": 30, "distinct_patients": True, "fetch_factor": 6, "alpha_queries": 1500,
        "alpha_grid": [0.1, 0.5, 1, 2, 5, 10], "ood_percentile": 99.0}
m0 = sites == "s0"
idx0 = np.where(m0)[0]
lm0 = SimpleNamespace(ft=pft.take(idx0), y=y[idx0])
pcb = PCaseBase(PMetric(pe2, wp), lm0, mcfg, E=E[idx0])
fp = pcb.fit_alpha(seed=0)
ccb = CCaseBase(E[idx0], pe2.weights_dims(wp), y[idx0], 2, pid[idx0], k=30, distinct=True, fetch=6)
fc = ccb.fit_alpha(mcfg["alpha_grid"], k_grid=None, n_queries=1500, ood_percentile=99.0, seed=0)
print("D. case base α perisim / pmknn:", fp["alpha"], fc["alpha"], "| LOO log-loss |diff|:",
      abs(fp["loo_logloss"] - fc["loo_logloss"]), "| OOD threshold |diff|:",
      abs(fp["ood_threshold"] - fc["ood_threshold"]))
for tag, q in (("own site, own patients excluded", idx0[:500]), ("other site", np.where(~m0)[0][:500])):
    Dp, Ip, vp_ = pcb.search(E[q], pid[q])
    Dc, Ic, vc_ = ccb.search(E[q], pid[q])
    same_nb = bool(((Ip == Ic) | ~vp_).all() and (vp_ == vc_).all())
    own = bool((pid[idx0][Ic] == pid[q][:, None])[vc_].any())
    dup = max(len(set(pid[idx0][Ic[i][vc_[i]]])) != int(vc_[i].sum()) for i in range(len(q)))
    pp = pcb.probs(Ip, vp_, 30)
    pc = ccb.probs(Ic, vc_, 30)
    print(f"D. queries ({tag}): same neighbours {same_nb} | own patient among neighbours {own} | "
          f"a patient twice {bool(dup)} | p max |diff| {float(np.abs(pp - pc).max()):.2e}")
