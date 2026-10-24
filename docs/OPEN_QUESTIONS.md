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
3. **INSPIRE — ICU readmission (D1).** *Decided 2026-10-22 (AUDIT D10):* a new ICU admission linked to another
   surgery OR a new ventilation / RRT / ECMO / IABP run after t0 in `ward_vitals` (`targets.yaml:
   icu_readmission_proxy`). **To check in the first audit:** the proxy prevalence and the split by source
   (`completeness_inspire.html`) against MIMIC-IV / eICU readmission rates, before acknowledging the audit.
4. **INSPIRE — optional columns.** Donors (v1.4.2+), year (temporal validation), discharge destination and
   limitation of life-sustaining treatment: `optional_columns` is set to `null` until the actual schema has been seen.
5. **INSPIRE — ATC medication.** `atc_code`, `atc_code2`, `atc_code3` are used as alternative codes for the same
   administration. Confirm their meaning (different levels of the same drug?).
6. **INSPIRE — red blood cell transfusion.** Searched for by drug name (regex). Confirm whether it is recorded
   on the ward. *Decided 2026-10-24 (lead): not recorded; removed from T1 (see 36).*
7. **MIMIC-IV — diagnoses.** `diagnoses_icd` has no timestamp (billing at discharge): it is excluded as an
   input by the anti-leakage rule. Consequence: diagnoses are not part of the `core` set (intersection).
8. **MIMIC-IV — postoperative ICU rule (D10).** *Decided 2026-10-22 (AUDIT D1/D8):* OR/PACU transfer in
   `hosp.transfers` within 24 h before the ICU admission; procedure codes only for hospitalisations without any
   OR/PACU transfer. **To verify:** (a) the OR-related `careunit` values (`cohorts.yaml: or_careunits`, "PACU"
   assumed) in `schema_mimic.md`; (b) the fallback uses a declared approximation of the HCUP "major therapeutic"
   class until the HCUP Procedure Classes files are added (`cohorts.yaml: hcup_pclass_files`); (c) the
   "service_only" count in the audit (sensitivity cohort).
9. **MIMIC-IV — medication.** No ATC: name → ATC mapping by regex (`harmonization.medications.name_to_atc`).
   Transfusion is observable only inside the ICU (inputevents) unless eMAR records it.
10. **eICU — sources.** Ventilation (`respiratoryCare`, `treatment`), vasopressors (`infusionDrug`), GCS
    (`nurseCharting`), FiO2 (`respiratoryCharting`). Medication = orders, not administrations.
11. **Format on KHDP.** `.csv`, `.csv.gz`, `.parquet` and Parquet folders are supported. If KHDP serves the
    databases as a database (Postgres/BigQuery), a new reader will be needed in `src/perisim/data/io.py`.

## Method

12. **NCA prior and bandwidth.** *Resolved 2026-10-22 (B2/B4, SPECS v0.4 §5.4):* `log w = log c + log(p + ε₀) + u`
    with c free and the penalty on the SAGE group totals (mean over groups).
13. **λ inside the stopping loop.** *Resolved 2026-10-22 (B3, SPECS v0.4 §5.3.2):* Δ is measured on the
    prior-implied metric; no NCA (and no λ) inside the loop.
14. **λ selection with the outer fold's SAGE.** *Resolved 2026-10-22 (C3):* SAGE is recomputed in each inner fold
    (`metric.lambda_prior: inner_fold_sage`); the pilot extrapolates the extra cost.
15. **kNN α.** Chosen per k with patient-level LOO on the training set of the outer fold (the same for all kNN
    comparators), not in the inner folds. *Kept and declared 2026-10-22 (C6).*
16. **Metric for the clinical output (§5.6).** One metric per target. The per-patient output uses the neighbours
    from the `patient_output_target` metric (T1 by default). Should it be T3 (the integrated objective)?
17. **Distribution of importance within a group.** Large blocks (dx/ATC > 50): ∝ prevalence × mutual
    information (univariate SAGE with the Bayes-optimal model). Everything else (value + indicator, PCS levels,
    clusters with |ρ| > 0.8): equal shares. *Since 2026-10-22 (B4) this split is only the starting point and the
    stopping-rule metric: the NCA prior acts on group totals and the NCA splits within groups.*
18. **D7 rule from the pilot.** `min_delta = 0.05 × Δ̃` from the pilot at iteration n_min (rule declared in
    `experiment.yaml` before execution). `tol_Δ = 0.01`.
19. **SWIFT.** The source of ICU admission is approximated by the source of hospital admission; missing
    values score 0. Logistic recalibration in the training fold.
20. **T3.** "Early discharge" = stay ≤ first tertile and "prolonged" = stay > second tertile, computed in the
    training fold over the patients without complication.

## Introduced by the decisions of 2026-10-22 [VERIFY on the first run]

21. **MIMIC-IV — Code Status values (D12).** Limitation = a Code Status (itemid 223758) value matching
    `harmonization.yaml: sources.mimic.limitation.value_regex` (DNR / DNI / comfort…); the first such record ≤ the
    cutoff excludes the stay. Check the value inventory.
22. **eICU — care limitation (D12).** `carePlanGeneral` rows of the "Care Limitation" group other than "Full
    Therapy". Check the `cplgroup` / `cplitemvalue` inventory.
23. **MIMIC-IV — blood-gas specimen type (D13).** pH / pCO₂ / pO₂ are kept only for specimens whose itemid 52033
    value starts with "ART". Check the values (and that `labevents` has `specimen_id` and `value`).
24. **Chronic dialysis markers (D5).** INSPIRE: ICD-10 `N18.5/N18.6/Z99.2/Z49.x` and RRT device records before the
    surgery; MIMIC-IV: dialysis procedures (ICD-9 39.95, ICD-10-PCS 5A1D…) dated before the ICU admission day and
    RRT in an earlier ICU stay of the hospitalisation. MIMIC-IV has no timestamped ESRD diagnosis, so detection is
    less sensitive there (declared).
25. **eICU — medication coverage (D6 rule).** The decision concerned MIMIC-IV eMAR; the same coverage rule is
    applied to eICU medication orders (inputs only, since eICU T1 is reduced). Set
    `sources.eicu.medication_coverage: null` to revert.
26. **Sensitivity variants and the cohort.** The limitation exclusion uses the main cutoff (t0 − 2 h) for every
    variant, because the index cohort is extracted once (the 4 h variant keeps the same stays).

## Introduced by methodology v0.5 (2026-10-24)

27. **T3 metric.** Learned anew (implemented) or combined from the T1 and T2 metrics? To decide after T1.
28. **Flatness rule of the shapes.** Implemented as a homogeneity F test at 5 % (Bonferroni over classes) on top of the
    decided shrinkage, because "τ² = 0" alone keeps about half of the pure-noise curves (see DECISIONS.md).
29. **Shapes in the target cohort.** Modes A and B transport INSPIRE's shapes; re-estimating them in MIMIC-IV / eICU
    (the analogue of the B-VDM variant) would require training the consensus models there — not done (declared).
30. **Missing values in the shape geometry.** Kept as median imputation + missingness indicator (the indicator is a
    binary axis); a "missing" position on each continuous axis was discussed but not adopted.
31. **Relabelled sensitivity variants** assess the results under another outcome definition with the main metric;
    whether the metric itself would differ is not assessed (declared).

## From the first discover in the pod (2026-10-24)

32. **INSPIRE department `OT`.** *Decided 2026-10-24 (lead):* ophthalmology, in the same group as EY
    (plastic_other). RAD, IM, AN, PED and RO (few operations) stay in "other".
33. **INSPIRE `CTS` = cardiothoracic.** *Decided 2026-10-24 (lead):* split by the ICD-10-PCS code of the operation —
    02 (heart and great vessels, incl. thoracic aorta) or 5A (extracorporeal support) → cardiac; 03–06 (arteries,
    veins) → vascular; any other code → thoracic; no code → cardiac.
34. **INSPIRE MAP.** *Decided 2026-10-24 (lead):* `nibp_mbp` is charted about a third as often as `nibp_sbp` /
    `nibp_dbp`, so MAP = (SBP + 2·DBP) / 3 at the time points with SBP and DBP but no charted MAP.
35. **INSPIRE transfusion.** *Decided 2026-10-24 (lead):* no red-cell product among the most frequent drug names; the
    completeness audit decides — if `transfusion_rbc` has no events, the component is reviewed there (T1).

36. **INSPIRE transfusion — completeness audit (first real run, T2 experiment).** The audit stopped with "inspire:
    component transfusion with prevalence < 0.5 %". *Decided 2026-10-24 (lead):* transfusion is removed from T1 in
    every cohort (one harmonized outcome definition for development and external validation; mild = AKI 1–2 or a
    new antibiotic course); it remains a carried variable.

## Open decisions from SPECS §10 (v0 values in config)

D1 (INSPIRE readmission), D2 (2 h; 4 h in sensitivity), D3 (≥ 4 h), D4 (AKI 1–2, antibiotic; transfusion removed 2026-10-24),
D5 (tertiles), D6 (0.5%), D7 (from the pilot), D8 (disabled), D9 (OSF: `freeze --registration`),
D10 (MIMIC/eICU rule), D11 (k = 50).
