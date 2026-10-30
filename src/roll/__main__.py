"""Command line of experiment `roll` (docs/design_roll.md).

  python -m roll check                    the freezes of T2 and `hypo` (unchanged?)
  python -m roll streams                  the real streams (order 0, primary N): sizes, windows, digests (aggregates)
  python -m roll dry [--threads 4]        Phase 2 dry run on synthetic data: every arm, trigger and endpoint
  python -m roll t2-check                 T2's frozen artefacts and MIMIC-IV stays load (counts only, nothing fitted)

The single real run (Phase 4) is added with the freeze (Phase 3): it runs only on the frozen version.
Patient-level outputs stay in <workdir>/experiments/roll; aggregates only are ever exported.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import frozen_guard, workdir

LOG = logging.getLogger("roll")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m roll", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("streams")
    s.add_argument("--streams", nargs="*")
    d = sub.add_parser("dry")
    d.add_argument("--threads", type=int, default=4)
    d.add_argument("--streams", nargs="*")
    sub.add_parser("t2-check")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if a.cmd == "check":
        print(json.dumps(frozen_guard(), indent=1))
        return 0
    if a.cmd == "dry":
        from .run import dry

        out = dry(a.threads, a.streams)
        print((out / "summary.md").read_text(encoding="utf-8"))
        return 0
    if a.cmd == "t2-check":
        import numpy as np

        from . import t2

        start = frozen_guard()
        d = t2.load()
        blocks, counts = np.unique(d["block"], return_counts=True)
        print(json.dumps({"freeze_start": start, "stays": d["n_stays"], "unlabelled": d["n_unlabelled"],
                          "patients": int(np.unique(d["subject"]).size), "k": d["k"], "alpha_grid": d["alpha_grid"],
                          "stays_per_block": dict(zip(blocks.tolist(), counts.tolist())),
                          "embedding_dims": int(d["Ew"].shape[1]), "freeze_end": frozen_guard()}, indent=1))
        return 0
    if a.cmd == "streams":
        from hypo.config import load as load_hypo

        from .data import real_cohort
        from .run import jobs, settings
        from .streams import build

        start = frozen_guard()
        cfg = settings(False)[0]
        cs = {c: real_cohort(c, load_hypo()) for c in ("inspire", "mimic", "eicu")}
        out = {"freeze_start": start, "cohorts": {c: v.info for c, v in cs.items()}, "streams": []}
        for j in jobs(cfg, cs, a.streams, primary_only=True):
            st = build(j["stream"], j["spec"], cs, cfg, 0, j["N"], j["hospital"])
            if st is None:
                out["streams"].append({"stream": j["stream"], "label": j["label"], "N": j["N"], "skipped": True})
                continue
            st.make_windows(cfg["windows"]["W"])
            out["streams"].append({**st.describe(), "label": j["label"], "hospital": ""})
        out["freeze_end"] = frozen_guard()
        p = workdir() / "streams.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print(p)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
