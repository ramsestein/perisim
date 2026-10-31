"""Experiment `roll`: prequential evaluation with a rotating case base (docs/design_roll.md).

A pre-registered retrospective evaluation on data already examined. A separate package: it reads `hypo`'s frozen
metrics and comparators and T2's frozen targets, and changes neither `perisim` nor `hypo` (their freezes are checked
at the start and the end of every stage)."""
