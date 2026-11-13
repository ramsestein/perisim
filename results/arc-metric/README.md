# results/arc-metric

Results of branch `arc-metric`: every PERISIM experiment re-run in two arms that differ only in how a shaped
continuous or ordinal value is placed on its axis.

- **`value`** is the published placement, the value of the consensus curve g_j at the value's bin.
- **`arc`** is the arc length of g_j up to the value.

The pre-registration is [docs/prereg_arc.md](../../docs/prereg_arc.md) (OSF
[10.17605/OSF.IO/NAQ2T](https://doi.org/10.17605/OSF.IO/NAQ2T)).

- [REPORT.md](REPORT.md): the results, readings and deviations.
- [arc_compare.md](arc_compare.md) and [compare.json](compare.json): the paired contrast (`scripts/arc_compare.py`).
- [value/](value/) and [arc/](arc/): each arm's aggregate outputs, by experiment (`t2`, `hypo`, `nwicu`, `roll`).
  - Copied with the same rules as `scripts/publish_results.py`: aggregates only, ≤ 2 MB, no identifiers, small cells
    blanked.
  - Two T2 CSVs and roll's `endpoints.json` exceed 2 MB and are not included. Roll's confirmatory hypotheses are
    extracted to `roll/results/hypotheses.json`.

The published results of v0.5 are in the other folders of `results/`. They are not changed by this branch.
