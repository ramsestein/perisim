# OPEN_QUESTIONS — PERISIM

Questions the code does NOT decide silently (SPECS.md, preamble). Each entry states what has been implemented
as a v0 proposal and what needs to be confirmed. Issues detected in the pod are added automatically to
`<workdir>/docs/OPEN_QUESTIONS.md` (exported with the aggregates); carry the decisions taken over to this file.

## Schemas and data [VERIFY on the first run: `reports/schema_<cohort>.md`]

1. **INSPIRE — column and item names.** `config/harmonization.yaml` uses the names from the publication
   (operations: `icuin_time`, `emop`, `antype`, `icd10_pcs`…; labs/ward_vitals: `item_name`, `value`).
   The `item_name` inventory in the schema report will show what to rename.
2. **INSPIRE — devices in `ward_vitals`.** Ventilation/RRT/ECMO/IABP are assumed to appear as "active" records
   (`vent`, `crrt`…). If they are only start markers, change `sources.inspire.devices.mode`.
3. **INSPIRE — ICU readmission (D1).** Implemented as a new ICU admission linked to another surgery in the same
   hospitalisation. Non-surgical readmissions are not observable this way; the audit quantifies this.
   If that is not enough: severe T1 in INSPIRE without readmission and a declared sensitivity analysis.
4. **INSPIRE — optional columns.** Donors (v1.4.2+), year (temporal validation), discharge destination and
   limitation of life-sustaining treatment: `optional_columns` is set to `null` until the actual schema has been seen.
5. **INSPIRE — ATC medication.** `atc_code`, `atc_code2`, `atc_code3` are used as alternative codes for the same
   administration. Confirm their meaning (different levels of the same drug?).
6. **INSPIRE — red blood cell transfusion.** Searched for by drug name (regex). Confirm whether it is recorded
   on the ward.
7. **MIMIC-IV — diagnoses.** `diagnoses_icd` has no timestamp (billing at discharge): it is excluded as an
   input by the anti-leakage rule. Consequence: diagnoses are not part of the `core` set (intersection).
8. **MIMIC-IV — postoperative ICU rule (D10).** Surgical service active at ICU admission **or** an ICD-10-PCS
   section 0 procedure with an open/percutaneous endoscopic approach (or ICD-9 01–86) on that day or the day
   before. Review with the audit.
9. **MIMIC-IV — medication.** No ATC: name → ATC mapping by regex (`harmonization.medications.name_to_atc`).
   Transfusion is observable only inside the ICU (inputevents) unless eMAR records it.
10. **eICU — sources.** Ventilation (`respiratoryCare`, `treatment`), vasopressors (`infusionDrug`), GCS
    (`nurseCharting`), FiO2 (`respiratoryCharting`). Medication = orders, not administrations.
11. **Format on KHDP.** `.csv`, `.csv.gz`, `.parquet` and Parquet folders are supported. If KHDP serves the
    databases as a database (Postgres/BigQuery), a new reader will be needed in `src/perisim/data/io.py`.

## Method

12. **NCA prior and bandwidth.** `L(w) = LogLoss + λ Σ (log w_j − log c − log(imp_j + ε₀))²` with a free
    global scale `c`: the prior regularises the *profile* of the importances and leaves the bandwidth free
    (otherwise the prior would also fix the kernel scale). Confirm this reading of §5.4.
13. **λ inside the stopping loop.** The §5.3.2 criterion recomputes w with the §5.4 conversion; a fixed λ is used
    (`importance.lambda_loop = 1`) because the final λ is chosen afterwards by inner CV.
14. **λ selection with the outer fold's SAGE.** SAGE is computed once per outer fold (with inner split 0)
    and reused to choose λ in the 3 inner folds. Recomputing SAGE per inner fold would triple the cost.
15. **kNN α.** Chosen per k with patient-level LOO on the training set of the outer fold (the same for all kNN
    comparators), not in the inner folds.
16. **Metric for the clinical output (§5.6).** One metric per target. The per-patient output uses the neighbours
    from the `patient_output_target` metric (T1 by default). Should it be T3 (the integrated objective)?
17. **Distribution of importance within a group.** Large blocks (dx/ATC > 50): ∝ prevalence × mutual
    information (univariate SAGE with the Bayes-optimal model). Everything else (value + indicator, PCS levels,
    clusters with |ρ| > 0.8): equal shares.
18. **D7 rule from the pilot.** `min_delta = 0.05 × Δ̃` from the pilot at iteration n_min (rule declared in
    `experiment.yaml` before execution). `tol_Δ = 0.01`.
19. **SWIFT.** The source of ICU admission is approximated by the source of hospital admission; missing
    values score 0. Logistic recalibration in the training fold.
20. **T3.** "Early discharge" = stay ≤ first tertile and "prolonged" = stay > second tertile, computed in the
    training fold over the patients without complication.

## Open decisions from SPECS §10 (v0 values in config)

D1 (INSPIRE readmission), D2 (2 h; 4 h in sensitivity), D3 (≥ 4 h), D4 (AKI 1–2, antibiotic, transfusion),
D5 (tertiles), D6 (0.5%), D7 (from the pilot), D8 (disabled), D9 (OSF: `freeze --registration`),
D10 (MIMIC/eICU rule), D11 (k = 50).
