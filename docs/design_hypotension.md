# Next use case: hypotension in the next 6 h in the ICU after surgery

Status: **defaults D1–D5 approved by the lead on 2026-10-28; D7 resolved in Phase 0; D6 and D8 open.** Branch `hypotension`, package `src/hypo/`; Phases 0–2 done (landmarks and audit, model, internal validation `reports/cv.md`); Phase 3 in progress: draft pre-registration [prereg_hypo.md](prereg_hypo.md) and the external stage's code, to freeze once the lead approves it; nothing registered yet. T2 was the deliberately hard case
(little information at ICU discharge) used to learn whether the shape functions carry real information, where the
method breaks and what it is good for. This experiment applies that, end to end, to a case with dense data and a
frequent outcome, again centred on **transfer and explainability**. Feasibility measured on the original tables of the
three cohorts: [feasibility_hypotension.md](feasibility_hypotension.md) (`scripts/feasibility_hypotension.py`,
aggregates only). Decisions for the lead: **D1–D8** (§7).

## 1. The use case

Every 2 h during the first 48 h of a postoperative ICU stay, for a patient who is normotensive now, PERISIM returns:
the probability of hypotension in the next 6 h (an hour with median MAP < 65 mmHg); the k most similar moments of
other patients of the same hospital, with their MAP course and what was done (vasopressors started, fluids); and why,
read from the metric (which variables put the patient near the patients who became hypotensive, and with what shape).
The metric is learned once and shared; the case base stays in each hospital. The same machinery tells each hospital
where its patients' physiology, measurement or practice differs (the drift report).

## 2. What T2 taught, and what it sets here

| T2 finding | Here |
|---|---|
| The rare class fails (~0.5 events per neighbourhood at k = 50) | Outcome measured at 13–31 %: 6–15 events per neighbourhood |
| The case base is knowledge, not recalibration; needs thousands of patients; −0.011 to −0.022 log-loss per doubling | 12,000–34,000 patients and 150,000–485,000 landmarks per site |
| Relations are hospital-specific in eICU (own hospital −0.041 vs others at equal size, base rates corrected) | eICU per hospital (37 with ≥ 300 patients); hospital-aware case base |
| SAGE prior adds nothing (λ ≈ 0) | Shapes + diagonal NCA, no SAGE stage |
| A two-site metric transports like a local one | Leave-one-site-out: every site is external once |
| Drift sits in case-mix / coding and in what a missing value means | Inputs that every site measures densely; measurement modality and missing values typed and analysed |
| The post-hoc drift analyses explained the transport loss | Re-deformation, missing-only re-deformation, ICE and the per-hospital drift atlas are pre-registered |

## 3. Feasibility (original tables; postoperative adult ICU stays, first of the hospitalization, deaths included)

| | INSPIRE | MIMIC-IV | eICU |
|---|---|---|---|
| stays (patients) | 12,893 (12,002) | 28,163 (26,083) | 35,282 (33,759) |
| landmarks, normotensive in the last hour (patients) | 149,514 (12,470) | 395,934 (26,453) | 485,293 (34,069) |
| **hypotension in the next 6 h, normotensive now** | **13.2 %** | **30.7 %** | **19.3 %** |
| … all landmarks (includes persisting hypotension) | 17.2 % | 38.8 % | 27.5 % |
| … off vasopressors / on vasopressors | 13.1 / 27.2 % | 26.7 / 45.7 % | 18.5 / 30.9 % |
| landmarks on vasopressors | 0.9 % | 22.6 % | 7.9 % |
| MAP readings per hour (median) | 1 | 1 | 12 |
| MAP hours with an arterial (invasive) value | 0 % | 63 % | 54 % |
| ICU hours with a MAP, first 48 h | 89 % | 88 % | 93 % |
| heart rate / SpO₂ in the last hour; ≥ 3 hourly MAPs in 6 h | 98 / 95 / 92 % | 100 / 98 / 95 % | 100 / 97 / 94 % |
| lowest hourly MAP of the next 6 h, tertile cuts | 71.5, 83.0 | 65.0, 73.0 | 68.5, 78.0 |

- **Frequent and stable**: the rate barely moves over the 48 h (INSPIRE 18 → 11 %, MIMIC-IV 31 → 28 %, eICU 22 →
  17 %); death inside the window ≤ 0.2 %; starting a vasopressor hides little hypotension (the composite
  "hypotension or vasopressor started" adds ≤ 2 points).
- **The definition depends on the measurement density.** eICU charts ~12 MAPs per hour (periodic monitor feed),
  MIMIC-IV and INSPIRE ~1. In eICU, "any reading < 65" gives 38 % against 19 % with the hourly median and 24 % with
  one reading per hour (MIMIC-IV 35 / 31 / 31 %, INSPIRE 18 / 13 / 15 %). The outcome is defined on the hourly median
  in every site, with "one reading per hour" as a sensitivity analysis.
- **Measurement modality differs**: INSPIRE has only non-invasive MAP in the ICU (charted, or (SBP + 2·DBP)/3);
  MIMIC-IV and eICU have an arterial line in ~70 % of the stays. A real transfer difference, not a bug: it is part of
  what the drift analysis should see.
- **Practice differs**: vasopressors at 0.9 % of INSPIRE's landmarks against 22.6 % in MIMIC-IV (INSPIRE's
  medication table has the ICU administrations, but cannot tell whether an infusion is running: **D7**, resolved); INSPIRE's patients run higher MAPs (tertile cuts).
- **eICU**: 197 hospitals with landmarks, 37 with ≥ 300 patients, 2 with ≥ 1,000; the hospitals' rate spreads from
  11.5 % (p10) to 25.3 % (p90).

## 4. Proposed design

**Population and landmarks.** Postoperative adult ICU stays (T2's rule), the first of each hospitalization, deaths
included. Landmarks every 2 h from 2 h to 48 h after the ICU admission, in the ICU and alive, with a MAP in the last
hour and that hour's median MAP ≥ 65 (normotensive now). Every landmark of a patient is in the case base; a patient's
own landmarks are never its neighbours (patient-level exclusion), and all splits and bootstraps are by patient.

**Outcome (primary).** Hypotension in the next 6 h: an hour in [L, L + 6 h) with median MAP < 65 mmHg, while in the
ICU; death in the window counts as hypotension. Secondary: the lowest hourly MAP of the next 6 h in three absolute
bands (< 65 / 65–75 / ≥ 75 mmHg, **D2**); the composite with vasopressor start; carried (neighbour distributions):
vasopressor started, MAP at 6 h, ICU discharge in 24 h.

**Inputs at L** (all before L; the list is `model.inputs` of `config/hypo.yaml`): MAP last, mean, minimum, slope and
variability over 6 h (hourly medians); heart rate last and slope; SpO₂ last and minimum; respiratory rate and
temperature; vasopressor running now, in the last 24 h and hours on it; sedation running now and in the last 24 h;
ventilation; hours since ICU admission; surgical group (unmapped → "unknown") and emergency; age and sex; last Hb,
creatinine, lactate. A missing value is its own value; INSPIRE cannot time infusions, so "running now" and
vasopressor hours are a declared source gap there (D7, Phase 0). BMI, surgery duration and the time since the
surgery are left out (absent in MIMIC-IV / eICU). The arterial-line flag is not an input (**D3**); it stratifies
the drift analysis.

**Metric (Phase 1: `src/hypo/metric.py`, `casebase.py`; `metric` in `config/hypo.yaml`).** Consensus shape
functions g_j (T2's procedure) from 40 accepted models of T2's model space; without SAGE the loop stops at that
fixed number. The development site enters those models as a nuisance column, not the metric, so each g_j is a
within-site effect and not a difference of base rates. Diagonal-NCA weights (λ = 0), with each anchor's candidates
searched in its own site, as the metric is used. An input constant within every development site (the missing
indicator of a source gap is the site itself) leaves the metric. Learning sample: at most 3 random landmarks per
patient and the same number from each development site (≤ 30,000). Case base: every landmark of the site's
patients. The k = 50 neighbours are distinct patients, each at its most similar moment, never the query's own. α by
patient-level LOO on 20,000 of the case base's landmarks; out of support as T2. Nested patient-level CV (5 × 3) in
the development sites: the inner level tunes the comparators (PERISIM has nothing left to tune but α).

**Sites (D1).** Leave-one-site-out: the metric learned on two cohorts, frozen, applied to the third with its own case
base; three rotations. eICU's rotation also per hospital (37 with ≥ 300 patients).

**Comparators.** "Last MAP" (logistic regression on the last MAP and its slope: the honest bar for an outcome so
close to an input); logistic regression and LightGBM on the same inputs (transported, recalibrated on 100 / 250 /
500 local patients, refitted locally); Euclidean kNN; prevalence.

## 5. Analyses (to register before the external stage)

| | Hypothesis | Decision rule |
|---|---|---|
| H1 | The frozen two-site metric with a local case base discriminates as well as PERISIM refitted locally | paired AUROC difference, lower 95 % CI > −0.02, in each external site |
| H2 | With a local case base the predictions are calibrated | calibration in the large inside ±0.10 log-odds and slope lower 95 % CI ≥ 0.90, each external site |
| H3 | PERISIM beats the last-MAP model | paired log-loss difference, 95 % CI < 0, each external site |
| H4 | Local relations are hospital-specific | eICU: own hospital − other hospitals at equal size, base rates corrected: 95 % CI < 0; random-group control ≈ 0 |
| H5 | A hospital-aware case base beats the plain pooled one | eICU: own-hospital weight γ ∈ {2, 4, 8} pre-declared vs γ = 1: 95 % CI < 0 |

Secondary (estimates with CIs, no decision rule): local patients needed (dense accumulation curve); drift per site
and per eICU hospital (ICE; re-deformation one variable at a time and all at once; missing-only re-deformation; the
development sites as the in-sample control), with the MAP threshold where the risk rises read from g_j per site and
per hospital; arterial line vs non-invasive strata; the "one reading per hour" outcome; LightGBM and logistic
regression transported / recalibrated / refitted.

## 6. Running it end to end

1. A branch and a new experiment id (`hypo`); T2 reproducible from its freeze tag (changes in `src/` move the code
   hash).
2. Extraction (landmark builder on the original tables: hourly vitals, infusions, labs, surgery) → completeness audit
   → internal CV per rotation → freeze + registration (**D6**) → one external stage: transport, case-base curves,
   hospital analyses, drift → report.
3. Compute: no SAGE; kNN over ≤ 500,000 landmarks per site with FAISS; landmarks thinned to every 4 h in the case base
   if the CV needs it (**D4**).

## 7. Decisions for the lead

- **D1** Leave-one-site-out (proposed) or a fixed development pair.
- **D2** Secondary outcome: absolute MAP bands (proposed) or each site's tertiles.
- **D3** Arterial-line flag: stratifier only (proposed) or an input.
- **D4** Case-base density: every 2 h (proposed) or thinned to 4 h.
- **D5** Horizon and threshold: 6 h and MAP < 65 (proposed); 65 is the usual target, 60 or 70 as sensitivity.
- **D6** Its own OSF registration (T2's real OSF ID is still pending).
- **D7** Resolved in Phase 0: INSPIRE's ICU vasopressor use is genuinely low; "running now" and vasopressor hours are
  a declared source gap there, and "in the last 24 h" is added for every cohort (DECISIONS.md).
- **D8** T1 and T3 of the original plan: dropped, postponed, or run on T2's frozen pipeline.
