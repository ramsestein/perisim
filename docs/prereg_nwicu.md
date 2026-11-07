# Pre-registration: transport of the frozen `hypo` metrics to NWICU

**Status.** This document fixes the plan before any NWICU outcome is built: before landmarks, MAP windows or
hypotension labels exist. It is meant to be uploaded to OSF as written, and is also committed to the repository.
- Before writing it, the lead's team looked at NWICU only through its data dictionary, table headers, the number of
  stays per care unit and admission type, how densely blood pressure is charted, and which drug names and event
  labels the eMAR contains.
- No MAP value, outcome or model output has been computed on NWICU.

**Question.** Do the patient-similarity metrics learned and frozen in the `hypo` experiment keep their gain in a
fourth, independent health system? Each was learned on two of INSPIRE, MIMIC-IV and eICU. They are used in NWICU
with NWICU's own patients as the case base and nothing refitted.

## Data

NWICU (Northwestern ICU database v0.1.0, PhysioNet; Northwestern Medicine, Chicago; stays mostly 2020–2022), tables
`nw_icu` and `nw_hosp`.

**Population.** The analogue of `hypo`'s postoperative ICU population, chosen by the lead before any outcome:
- the first stay of each hospitalisation whose first care unit is a surgical ICU (`SICU` or `CTICU`);
- adults (anchor age ≥ 18).

NWICU has no surgery table and no surgery times, so this is a proxy: it includes some patients who were not operated
on. Organ donors cannot be identified, and no limitation-of-treatment times exist.

**Landmarks, outcome and inputs: `hypo`'s definitions with v0.6's landmark code.** Landmarks every 2 h from 2 to
48 h after ICU admission. The outcome is an hour with median MAP < 65 mmHg in the next 6 h, or death in the ICU.

Deviations forced by the data:
1. **MAP is computed**, (SBP + 2·DBP)/3, from systolic and diastolic values charted at the same time:
   - arterial line items 320050/320051 → invasive MAP;
   - cuff items 320179/320180 → non-invasive MAP.

   NWICU charts no measured MAP. This is the rule `hypo` already applies to INSPIRE when only SBP/DBP were charted.
2. **Vasopressor and sedation states come from eMAR events** (`Confirmed` or `Applied`), as charted points. A point
   is a charting of an infusion or dose; NWICU has no infusion intervals. They are merged into runs while gaps are
   < 6 h, from the records before L only (v0.6's rule, the eICU treatment).
   - Vasopressors: norepinephrine, epinephrine, phenylephrine, vasopressin, dopamine. Local-anaesthetic and
     nebulised epinephrine are excluded.
   - Sedation: propofol, dexmedetomidine, midazolam.
3. **Mechanical ventilation is not recorded.** `vent_on` is a source-missing input, as INSPIRE's vasopressor inputs.
4. **Surgical group is unknown** (one level, "unknown") and the emergency flag is from `admission_type`
   (EMERGENCY, URGENT, TRAUMA).
5. **Labs** are hemoglobin (100007), creatinine (100002) and lactate (100031), timed when the result is available
   (`storetime`, else `charttime`). Temperature is converted from °F.

## Arms

NWICU patients are split into five patient-level folds, stratified by the outcome, with `hypo`'s seed. For each fold:
queries are that fold's landmarks; the case base is the other folds' landmarks.

- **Transported metric, three replicates.** The frozen metrics of `hypo`'s three rotations (`metric_ext-inspire`,
  `metric_ext-mimic`, `metric_ext-eicu`), with the frozen vote and α fit. Nothing is refitted.
- **Local metric.** `hypo`'s frozen procedure (shapes and NCA, the frozen settings) fitted on the case-base
  landmarks of each fold.
- **Euclidean kNN.** `hypo`'s Euclidean comparator fitted on the case-base landmarks of each fold; R's reference.
- **LightGBM transported.** The three frozen LightGBM comparators applied as they are, and with logistic
  recalibration (intercept and slope) fitted on the case-base landmarks of each fold.
- **LightGBM local.** `hypo`'s comparator trained on the case-base landmarks of each fold; the ceiling.

Code: `hypo`'s frozen code (v0.5; the freeze record is checked at start and end), except the NWICU extraction and
landmark builder (v0.6's landmark code). Scripts: `scripts/nwicu_build.py`, `scripts/nwicu_transport.py`.

## Statistics and pre-specified readings

Patient bootstrap, 2,000 resamples; log-loss is primary. The secondary metrics are AUROC, Brier, CITL and the
calibration slope.

- **Primary.** For each transported metric, R = (L_euclid − L_transported)/(L_euclid − L_local) with its 90 % CI.
  - It is *equivalent* if the lower limit is ≥ 0.80.
  - It is *not evaluable* if L_euclid − L_local < 0.005 nats or the 95 % CI of that difference includes 0.
  - The replication is **confirmed** if all three transported metrics are equivalent, **partial** if one or two are,
    and **not confirmed** if none is. Non-evaluable metrics are reported as such and count as not equivalent.
- **Secondary 1.** Each transported kNN against the recalibrated LightGBM from the same rotation (log-loss difference,
  95 % CI): better, worse, or no difference.
- **Secondary 2.** Calibration of each transported kNN with no recalibration (CITL, slope), described.
- **Secondary 3.** Local LightGBM against the best transported kNN, described as the cost of not refitting.
- **Descriptive.** The vasopressor stratum (`hypo` §19): the transport cost on and off vasopressors, and the
  same-state rule (B0) by vasopressor state.
  - This is reported, not tested. The NWICU vasopressor state is reconstructed from eMAR charting and is not
    comparable to MIMIC-IV's intervals.

## Limits declared in advance

- The surgical-ICU proxy for the postoperative population.
- MAP computed from SBP/DBP.
- Supports from eMAR points.
- No ventilation.
- One system, 2020–2022, with COVID-era case mix.
- The analysts know the `hypo` results; the readings above are fixed so that knowledge cannot steer them.
