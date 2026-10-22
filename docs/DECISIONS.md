# DECISIONS — dated log

Implementation decisions taken by the lead or derived from SPECS.md. Those that the pipeline takes in
the pod by declared rules (D7 from the pilot, chosen λ, Δ̃ plateaus, MLP dropped for cost,
temporal validation omitted…) are logged automatically in `<workdir>/docs/DECISIONS.md`.

## 2026-10-21 — Execution infrastructure on KHDP

- Code in a public GitHub repository, cloned in the pod through `http://github.khdp-forward-server`.
  The repository contains no data, secrets or outputs (`.gitignore` + `scripts/check_public_safety.py`).
- Single entry point `run_pod.sh`: virtual environment on the container's local disk (`~/.perisim/venv`),
  dependencies from the KHDP PyPI mirror and torch from the PyTorch mirror (CPU index, cu126 fallback).
- Read-only data in `/home/khdp-user/workspace/datasets` (automatic discovery of the three cohorts);
  work and artefacts in `updown/perisim/<user>/work`.
- Pipeline of idempotent stages with markers (hash of configuration + code): an OOM or a pod restart
  resumes from the last completed stage (or fold).
- Parallelism: at most 4 processes × 4 threads (§9.1); FAISS `IndexFlatL2` on CPU.

## 2026-10-21 — Interpretations of SPECS (see OPEN_QUESTIONS for confirmation)

- Normalisation of δ_j: each block is divided by s_j, with s_j² = E[δ_j²] over random training pairs,
  so that the mean square of each dissimilarity is 1.
- ICD-10-PCS levels: one-hot per level (prefixes of 1 to 5 characters); codes with < 10 cases in
  training → zero vector.
- `carried` is built by the targets stage from the common format (not by the extraction).
- The first postoperative stay of each hospitalisation is the unit of analysis: if it fails a later
  criterion (e.g. < 4 h), the hospitalisation is excluded.
