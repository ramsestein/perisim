"""CE-kNN vs the implementation it was ported from (PERISIM's `perisim` and `hypo` packages), on synthetic data.

    python tools/compare_with_perisim.py <path to a PERISIM checkout (branch main)>

Prints, for the same data, seed and settings: the models' matrix, the models drawn and their validation log-loss, the
consensus curves and the NCA weights of both. Needs the PERISIM checkout's dependencies (torch among them).
"""
import sys
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
from perisim.metric.encoding import Encoder as PEncoder
from perisim.metric.nca import DiagonalNCA as PNCA
from hypo.metric import shape_loop

import ceknn
from ceknn.encoding import Encoder as CEncoder
from ceknn.nca import DiagonalNCA as CNCA
from ceknn.rashomon import ShapeSettings, fit_rashomon
from ceknn.schema import Schema, Table
from synthetic import make_data

df = make_data(6000, seed=11)
inputs = {"u_shape": "continuous", "linear": "continuous", "noise": "continuous", "kind": "nominal",
          "flag": "binary", "lab": "continuous"}
sch = Schema.build(inputs, "y")
sch.resolve(df)
y = sch.labels(df)
ct = Table.from_frame(df, sch)
specs = [FeatureSpec(n, t, n, "x") for n, t in inputs.items()]
num = {n: df[n].to_numpy(dtype=float) for n, t in inputs.items() if t != "nominal"}
cat = {n: df[n].to_numpy(dtype=object) for n, t in inputs.items() if t == "nominal"}
ids = np.arange(len(df)).astype(str)
pft = FeatureTable(ids, ids.copy(), specs, num, cat)

# A. encoder: blocks and the models' matrix
pe = PEncoder(geometry="shape", max_bins=20, min_bin=50).fit(pft, y, 2)
ce = CEncoder(max_bins=20, min_bin=50).fit(ct, y, 2)
print("A. blocks equal:", pe.feature_names == ce.feature_names)
Xp, op = pe.transform_model(pft)
Xc, oc = ce.transform_model(ct)
print("A. model matrix max |diff|:", float(np.abs(Xp - Xc).max()), "owners equal:", bool((op == oc).all()))
En_p, En_c = pe.transform(pft), ce.transform(ct)
print("A. neutral embedding max |diff| (ECDF approximation):", round(float(np.abs(En_p - En_c).max()), 4))

# B. Rashomon loop and consensus shapes, same seed and settings
space = yaml.safe_load(open(REPO / "config" / "model_space.yaml", encoding="utf-8"))
space = {"families": {k: v for k, v in space["families"].items() if k in ("logreg", "spline_logreg", "random_forest")}}
scfg = dict(n_models=12, max_attempts_factor=3, epsilon=0.05, augment_copies=2, max_train=3000, max_val=1500,
            n_explain=1000, subsample=600, flat_alpha=0.05, corr_cluster_threshold=0.8, site_adjusted=True,
            n_reference_pairs=4000)
lm = SimpleNamespace(ft=pft, y=y, site=np.zeros(len(y), dtype=object))
rp = shape_loop(pe, lm, scfg, space, seed=3, threads=4)
rc = fit_rashomon(ce, ct, y, ceknn.ModelSpace.default(include=["logreg", "spline_logreg", "random_forest"]),
                  ShapeSettings(**scfg), seed=3, n_jobs=4)
vp = [r["val_logloss"] for r in rp["models"]]
vc = [r["val_logloss"] for r in rc.attempts]
print("B. same models drawn:", [r["family"] for r in rp["models"]] == [r["family"] for r in rc.attempts],
      "| val log-loss max |diff|:", float(np.max(np.abs(np.array(vp) - np.array(vc)))))
for j in pe.shapeable_blocks():
    gp, gc = rp["shapes"].get(j), rc.shapes.get(j)
    name = pe.blocks[j].name
    if gp is None or gc is None:
        print(f"B. {name}: shaped perisim={gp is not None} ceknn={gc is not None}")
    else:
        print(f"B. {name}: g max |diff| = {float(np.abs(gp - gc).max()):.2e}")

# C. NCA on the same embedding (perisim's, shapes installed)
pe2 = pe.with_shapes(rp["shapes"], pft)
E = pe2.transform(pft)
kw = dict(n_candidates=200, batch_size=256, lr=0.05, refresh_every=5, label_smoothing=1e-3, seed=0)
wp = PNCA(pe2.dim_feature, pe2.n_features, 2, eps0=1e-4, **kw).fit(E, y, epochs=10, subjects=ids).weights
wc = CNCA(pe2.dim_feature, pe2.n_features, 2, **kw).fit(E, y, epochs=10, groups=ids).weights
sp, sc = wp / wp.sum(), wc / wc.sum()
print("C. NCA weight shares perisim:", np.round(sp, 3))
print("C. NCA weight shares ceknn:  ", np.round(sc, 3))
print("C. max |share diff|:", round(float(np.abs(sp - sc).max()), 4),
      "| corr of log w:", round(float(np.corrcoef(np.log(wp), np.log(wc))[0, 1]), 4))
