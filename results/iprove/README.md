# INSPIRE → iPROVE (pre-specified, 2026-11-07)

Plan and rule: [docs/DECISIONS.md](../../docs/DECISIONS.md), 2026-11-07, commit `2b6b386`. Scripts:
`scripts/iprove_build_inspire.py`, `scripts/iprove_transport.py`. Tables: [transport.md](transport.md),
[cohort.md](cohort.md). Patient-level data stay on the lead's workstation.

**Task.** Intraoperative vasoactive drugs from 19 preoperative variables coded as in iPROVE.
- Source: INSPIRE abdominal surgery, 16,748 patients, 48.6 % with vasoactives.
- Target: iPROVE, 4,273 patients, 38.1 %.

**Readings (rule fixed before running).**
- **Q1, R of the transported NCA kNN: not evaluable.** NCA weights learned on iPROVE's own patients do not beat
  uniform weights there (log-loss 0.6115 against 0.6061), so there is no local gain to retain.
  - The transported NCA weights do worse than both (0.6208).
  - Mutual-information weights do best among the kNN arms (0.6048).
- **Q2, transported NCA kNN against INSPIRE's models recalibrated on iPROVE: kNN better.**
  - Against recalibrated logistic regression: −0.0155 (−0.0232 to −0.0082).
  - Against recalibrated LightGBM: −0.0224 (−0.0304 to −0.0147).
  - Every kNN arm with iPROVE's case base discriminates better than the transported models (AUROC 0.66–0.69
    against 0.62–0.64).
  - A LightGBM trained on iPROVE is best overall (0.5832, AUROC 0.71).
- **Q3, calibration with no recalibration:** CITL +0.03, slope 0.91 for the transported NCA kNN; +0.09 to +0.10 for
  the uniform and MI kNN. Unrecalibrated, INSPIRE's models are far off (CITL −0.84, −0.87).

**Reading.**
- **The case base travels; the learned weights did not.** With iPROVE's patients as the case base, a kNN adapts to a
  new country, design and population better than INSPIRE's parametric models do with a two-parameter
  recalibration.
- **The NCA weights learned in INSPIRE are worse than no learning.** They put weight on variables that INSPIRE
  measures differently or not at all:
  - INSPIRE's comorbidities come from mostly outpatient ICD codes: hypertension 7 % against 36 % in iPROVE; smoking,
    alcohol and dyslipidaemia never coded;
  - COPD takes 17 % of the weight on 1.2 % prevalence;
  - the variables constant in INSPIRE keep their starting weight.

  This is the vasopressor lesson again (`hypo` §16–19): learned weights travel only for variables that both sources
  measure the same way.

**Limits.**
- Bins are inferred from iPROVE's midpoints, and the SpO2 threshold is assumed (64 % of INSPIRE patients have no
  preoperative ward SpO2).
- The kNN arms use iPROVE's labels through the case base; recalibration uses them only for two parameters.
- One outcome, chosen after PPC proved impossible in INSPIRE.
