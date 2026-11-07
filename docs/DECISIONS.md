# DECISIONS — dated log

Implementation decisions taken by the lead or derived from SPECS.md. Those that the pipeline takes in
the pod by declared rules (D7 from the pilot, chosen λ, Δ̃ plateaus, MLP dropped for cost,
temporal validation omitted…) are logged automatically in `<workdir>/docs/DECISIONS.md`.


## 2026-11-08 — Result: a cosine kNN against the transported learned metric (REPORT §21)

Run on the lead's workstation with `hypo`'s frozen code (v0.5), plan commit `3f52b4d`; the gate passed at all four
targets (max |Δp| = 0). Report: `results/hypo/reports/posthoc_cosine.md`.
- **Rule:** NCA beats cosine in 6 of 6 comparisons (INSPIRE, MIMIC-IV, eICU, and NWICU with each of the three frozen
  metrics), in log-loss (+0.0077 to +0.0146) and AUROC (+0.0067 to +0.0190).
- Cosine ≈ Euclidean on the neutral axes; the cosine on the NCA embedding is close to the NCA distance.
- Fixed mutual-information weights (§17) also beat cosine at the three `hypo` targets.
- **Reading, as pre-specified:** in this setting the learned metric beats cosine, unlike PRAM's S13. This is stated as
  a difference between settings, not as an error in PRAM: even fixed MI weights beat cosine here, while in PRAM they
  match it, so the difference is not specific to NCA.
- **Incident.** While launching this run, `run_audit.bat` (the row above `run_cosine.bat`) was started by mistake. It
  only re-summarises cached units: its outputs (NWICU transport, IV-only sensitivity, audit checks) were checked and
  are identical to the published ones.

## 2026-11-08 — Post hoc, pre-specified: a cosine kNN against the transported learned metric (`hypo` and NWICU; lead; plan fixed before running)

**Why.** PRAM v2 (Jeong et al., 2026, medRxiv), supplementary Table S13, reports in eICU that a neighbourhood-components
metric, transferred from development or fitted on the target bank, is below cosine distance (ΔAUROC over the frozen
logistic model: cosine +0.0197 / +0.0358, transferred NCA +0.0105 / +0.0268, for AKI / 168 h mortality). PERISIM never
ran a cosine kNN, so it cannot say whether its NCA metric beats cosine in its own setting. This analysis adds that arm.
It cannot show that PRAM's NCA was suboptimal: their variables (42 static first-24 h aggregates), outcomes and NCA
implementation differ, and any disagreement is attributable to the setting as a whole, not to one design choice.

**Status.** Post hoc, pre-specified, run with `hypo`'s frozen code (v0.5; freeze record checked at the start and the
end; nothing frozen is refitted). The cosine arms have not been computed when this plan is fixed. Code:
`scripts/posthoc_cosine.py`.

**Arms** (each target's own out-of-fold case base; the external stage's five folds for INSPIRE, MIMIC-IV and eICU, the
pre-registered NWICU folds for NWICU; k = 50, distinct patients, the same smoothed vote; α tuned by leave-one-out on
the case base with the comparators' seed):
- **COS-neutral** (primary comparator; the analogue of PRAM's M0): cosine distance on the Euclidean comparator's
  neutral axes (encoder fitted on the fold's case base, no outcome information), each column centred on the fold's
  case-base mean. On equal-scale axes this is the cosine between standardised (rank-based) feature vectors, as cosine
  on standardised features in PRAM. Implemented exactly as a Euclidean kNN on the centred, row-normalised vectors,
  since ‖u − v‖² = 2 − 2 cos(u, v) for unit vectors (same neighbours, same order).
- **COS-NCA** (secondary): the cosine on the transported NCA metric's weighted embedding (√w per column, shapes
  installed), centred on the fold's case base: does the angle instead of the distance matter once weights are learned?
  In NWICU, one per frozen metric.
- Comparators, stored and unchanged: NCA-transported, NCA-local, Euclidean (all targets); MI-transported (§17; the
  analogue of PRAM's M1; three `hypo` targets only).

**Gate.** As in §17: the Euclidean comparator rebuilt with this script's code (weights 1) on fold 0 must reproduce the
stored Euclidean predictions to 1e-9 at every target (Phase 2 for the `hypo` targets; `euclidean_f0` for NWICU);
otherwise the script stops.

**Primary endpoint and rule.** At each of the four targets, Δc = log-loss(COS-neutral) − log-loss(NCA-transported),
with a 95 % patient-bootstrap CI (2,000 multinomial resamples of patients, shared by all arms of a target). In NWICU
the NCA-transported arm is each of the three frozen metrics, each read separately. Per target:
- **NCA beats cosine** if the lower 95 % limit of Δc is > 0;
- **cosine beats NCA** if the upper limit is < 0;
- otherwise **no difference shown**.

Secondary, read the same way and reported because PRAM reports AUROC: ΔAUROC = AUROC(NCA-transported) −
AUROC(COS-neutral) with the same resamples (weighted AUROC, ties counted one half). Also descriptive: COS-neutral
against Euclidean (does the angle matter on neutral axes), COS-NCA against NCA-transported, and calibration in the
large and slope of every arm.

**Pre-specified reading.**
- NCA beats cosine at every target: in this setting the learned metric beats cosine, unlike PRAM's S13; stated as a
  difference between settings, not as an error in PRAM.
- Cosine ≥ NCA at some target: PRAM's finding holds in our setting there, and FINDINGS and the article say so.
- Mixed: reported per target, without pooling.

**Privacy.** Aggregates only; per-fold predictions stay in the git-ignored work folder.

## 2026-11-08 — Repository made presentable; old tags and branches deleted (lead)

**Why.** The lead plans to write to PRAM's authors, possibly to join efforts, and asked for a clean, understandable
repository in English. The repository was already in English, so the work was structure and presentation.

**Done** (commit `618f338`). Code and results were not moved: the frozen paths and their hashes stay valid.
- **README** rewritten as a front page: the question, key results with their level of evidence, the experiments, a
  repository map, reproduction, privacy, history and naming, licence and citation.
- **`docs/FINDINGS.md`**: every result in plain language, graded as pre-registered, post hoc pre-specified or
  exploratory, with a section on what is ours and what is not. It cites PRAM v2's supplementary Table S13, in which
  a transferred NCA performs about like one fitted on the target bank, both below cosine distance.
- **`docs/README.md`**: an index that separates records from living documents.
- **`docs/RUNNING.md`**: the operational guide, moved from the README unchanged.
- **`LICENSE`** (MIT, as `pyproject.toml` already declared) and **`CITATION.cff`**.

**Deleted on GitHub** at the lead's request. Done from the lead's workstation with the lead's own git (a .bat; this
session cannot push deletions):
- the three freeze tags (T2, `hypo`, `roll`), which pointed into the pre-snapshot history;
- twelve stale branches.

`main` and `pmknn` remain. The frozen code is in `frozen/` with its hashes. The references of PRs #1 and #2 can only
be removed by GitHub support.

## 2026-11-07 — Results of the audit follow-ups

- **Effective weight shares** (`results/hypo/reports/posthoc_audit_checks.md`):
  - vasopressor inputs: raw → effective 3.64 % → 0.29 % (INSPIRE target, inputs absent), 0.08 % → 0.47 % (MIMIC-IV),
    1.75 % → 1.61 % (eICU);
  - sedation: 0.41 % → 0.23 %, 0.54 % → 1.46 %, 2.36 % → 1.67 %.

  P1 stays false. The MIMIC-IV metric still gives the vasopressor the least weight among the sites that record it.
  REPORT §16, §18 and §19 are annotated.
- **eICU patients in several hospitals:** 0 to 17 per target hospital, at most about 1.5 %. Negligible (REPORT §20,
  note).
- **NWICU B0 (descriptive):** on the vasopressor stratum, +0.0036 for the INSPIRE + eICU metric, −0.0020 and
  −0.0024 for the other two.
- **NWICU, intravenous forms only:** fewer than ten records removed; every figure unchanged.

## 2026-11-07 — Audit of the day's analyses (lead asked for a full check; plan before running the follow-ups)

**What was checked.**
- The whole test suite, with the v0.6 code (260 passed, 6 skipped) and the frozen `hypo` code (26 passed).
- Two independent reviews, which had not seen the conclusions, of the day's scripts: the NWICU build and transport,
  the state test, the leak sensitivity, the hospital transport and the alignment robustness.
- Every figure quoted in REPORT §15.4, §19 and §20 and in `results/nwicu/README.md`, traced back to its report file.
- Privacy and the indexes.

**Clean.** No leakage was found anywhere:
- folds are patient-level;
- case bases, α, every metric fit, recalibration and the local LightGBM see training folds only;
- frozen comparators and embeddings are used as the frozen code uses them;
- `states_before` matches v0.6's `SupportsBefore` (0 mismatches in 86,400 randomised cells);
- every quoted figure matches its source.

**Findings and what is done about them.**
1. **R2's reference pairs included the ablation variants**, which are transported metrics too (§15.4). Re-checked
   on the 55 pairs of genuinely local metrics only (local; n500–half, a and b). Transported–local stays inside the
   prediction interval at all three sites:

   | site | observed r | predicted r | 95 % PI |
   |---|---|---|---|
   | INSPIRE | 0.781 | 0.784 | 0.744–0.824 |
   | MIMIC-IV | 0.906 | 0.913 | 0.898–0.928 |
   | eICU | 0.895 | 0.891 | 0.868–0.913 |

   The "generic" reading stands; it is noted in §15.4.
2. **Raw weight shares are not comparable across metrics** (the share is w_j/Σw on axes scaled on each rotation's
   development data). This is used in P1 (§19) and in the vasopressor narrative (0.08 %, 1.75 %, 3.64 %). To be
   recomputed as effective shares: the share of the weighted squared distance between random pairs of the target's
   landmarks (`scripts/posthoc_audit_checks.py`). Statements are to be corrected to whatever that shows.
3. **eICU patients can appear in more than one hospital**, and their other-hospital landmarks entered §20's
   other-hospital NCA pool. This is leakage into weight learning only. To be counted (same script).
4. **NWICU, pre-registered B0 by vasopressor state (descriptive) was not computed.** It is added now to
   `nwicu_transport.py`, late and labelled so; the units of the primary run are reused.
5. **NWICU's vasopressor rule let in non-intravenous forms**: nasal, rectal, ophthalmic, oral, the auto-injector.
   They account for about 250 of about 280,000 vasopressor events. This touches inputs only, never the outcome.
   - Post hoc sensitivity: rebuild with intravenous forms only (`nwicu_build.py --iv-only`) and transport again
     (`--tag nwicu_ivonly`).
   - The primary result stays as pre-registered.
6. **NWICU README overstatements**: "recalibrated LightGBM beats local LightGBM" had no CI, and "no failure" in a
   descriptive stratum had no criterion. Softened.
7. **Robustness of the code.**
   - The leak sensitivity now refuses to run without §19's results, so it can no longer report a vacuous
     "all unchanged", and it flags cells that drop out.
   - NWICU units carry a run stamp, so they cannot be mixed across runs; the primary run started from an empty
     folder.
   - §15.4: INSPIRE's 4,000-item setting was skipped silently because it has 2,288 queries.

## 2026-11-07 — Result: NWICU (pre-registered, OSF 10.17605/OSF.IO/P5CA3; `results/nwicu/`)

Read by the pre-registered rule, unchanged.
- **Primary: confirmed.** All three frozen metrics are equivalent: R 1.46, 1.53 and 1.64; every lower 90 % limit
  ≥ 1.10.
- **Secondary 1: kNN worse** than the recalibrated LightGBM at all three rotations (+0.010 to +0.013 nats).
- **Secondary 2:** CITL +0.07 to +0.10, slope 1.00–1.06.
- **Secondary 3:** local LightGBM beats the best transported kNN by 0.0095.
- **Vasopressor stratum (descriptive):** no failure.

## 2026-11-07 — NWICU: fourth system, pre-registered (lead; plan fixed before any NWICU outcome exists)

**Why.** An independent, shareable replication of the main result: the frozen metrics keep their gain in a new
system, with that system's patients as the case base. NWICU is credentialed but open on PhysioNet, so anyone can
re-run it. The INSPIRE → iPROVE experiment was withdrawn by the lead the same day, before this entry: variable
concordance problems, suspected leakage, and data that cannot be shared.

**Feasibility, without outcomes.** Dictionary, headers, stay counts, charting density and eMAR drug names only:
- no measured MAP, so MAP is computed from SBP and DBP charted together;
- vasopressors and sedation appear only as eMAR events (Confirmed / Applied / Not Given), with no intervals;
- no ventilation anywhere;
- no surgery table.

The lead chose the surgical-ICU proxy (SICU, CTICU) for the postoperative population.

**Plan.** [docs/prereg_nwicu.md](prereg_nwicu.md), registered on OSF by the lead before the landmarks were built: https://doi.org/10.17605/OSF.IO/P5CA3 (2026-11-07).
- Three frozen metrics transported, with the local metric and Euclidean kNN as R's references.
- R per metric, read with the bioequivalence rule: confirmed if all three are equivalent, partial if one or two.
- Secondary comparisons against recalibrated and local LightGBM.
- The vasopressor stratum descriptive only.

Scripts: `scripts/nwicu_build.py` (v0.6's landmark code), `scripts/nwicu_transport.py` (frozen `hypo` code).

## 2026-11-07 — Results: transport between eICU hospitals (REPORT §20) and §19 without the landmark leak

Both run on the lead's workstation with `hypo`'s frozen code; gates exact; freeze record checked at start and end.
- **A, hospitals.** Q1 equivalent (pooled R of other-hospital weights 1.35, 90 % CI 1.24–1.46). Q2 no difference
  (cross-system − other hospitals −0.0004, −0.0013 to +0.0005). Q3: 8 of 10 evaluable, all 8 equivalent for both
  sources. R > 1 because one hospital's patients learn weaker weights than many hospitals' patients: the local
  reference is data-limited.
- **B, leak.** Not all verdicts unchanged. eICU's vasopressor failure disappears (contrast +0.0022, −0.0001 to
  +0.0048), and INSPIRE's B0 by ventilation becomes safe, at the threshold. MIMIC-IV's vasopressor failure and repair,
  P1–P3 and every other cell are unchanged. The vasopressor finding is MIMIC-IV's alone.

## 2026-11-07 — Post hoc, pre-specified: transport between eICU hospitals, and §19 without the landmark leak (`hypo`; lead; plans and rules fixed before running)

### A. Transport of the weights between hospitals inside eICU

**Why.** The external stage moved the metric between health systems; reviewers will ask whether the hospital is
the unit that matters (the review suggested eICU's hospitals as units). §6 used eICU's hospitals only for the case
base, never to learn the metric.

**Design** (`scripts/posthoc_hospital_transport.py`):
- **Targets.** eICU hospitals with ≥ 500 patients in the analysis landmarks, at most the 10 largest; anonymised by
  size rank.
- **Data per target h.** The external stage's five folds restricted to h; queries are h's landmarks in fold f, and
  the case base is h's landmarks of the other folds (k = 50, the frozen vote and α fit).
- **Arms.** All share the frozen eICU-target metric's encoder (shapes and axes); only the weights change:
  - uniform: all 1, the reference of R in this geometry;
  - cross_system: the frozen weights from INSPIRE + MIMIC-IV;
  - other_hosp: NCA weights learned on eICU minus h;
  - local: NCA weights learned on h's case-base landmarks of each fold.

  The NCA settings are the frozen ones (≤ 3 landmarks per patient, ≤ 30,000 landmarks, candidates from the whole
  pool).
- **Gate.** The frozen weights with eICU's whole case base rebuild the stored transported predictions on fold 0 to
  1e-9.
- **Statistics.** Patient bootstrap, 2,000 resamples. R_arm = (L_uniform − L_arm)/(L_uniform − L_local) with 90 %
  CI and §16's denominator rule. Also cross_system − other_hosp in log-loss with 95 % CI. All per hospital and
  pooled over the targets.

**Pre-specified readings.**
- Q1: pooled R of other_hosp — equivalent (lower 90 % limit ≥ 0.80) or not.
- Q2: pooled cross_system − other_hosp — other hospitals better, cross-system better, or no difference.
- Q3: number of evaluable hospitals whose R of other_hosp, and of cross_system, is equivalent.

No direction is predicted for Q2. A local metric learned on one hospital's patients may be weak, which R's
denominator rule handles: a hospital whose local gain over uniform is unclear is "not evaluable".

### B. §19 with the intervention strata rebuilt without the landmark leak

**Why.** The audit (2026-11-05, item 1) found that v0.5's states at L could be set by records after L, above all in
eICU. §19's verdicts, including eICU's new vasopressor failure, rest on those strata.

**Design** (`scripts/posthoc_leak_sensitivity.py`). The three states are recomputed with v0.6's rule from the v0.5
extract's supports. A state is on at L when some record of that kind with h < L has its coverage, clipped at L,
reaching hour max(L − 2, 0). §19's code is then re-run with those strata and with B0 by those states; predictions,
inputs and comparators are unchanged. Not applied: v0.6's eICU rate-0 stop markers, which the v0.5 extract lacks.

**Reading.** Report the share of landmarks reclassified per site and state. §19 is *robust to the leak* if every
(site, state) verdict (failure, repairs, safe) is unchanged; the cells that change are named.

## 2026-11-07 — Result: robustness of the neighbour alignment (`hypo`; REPORT §15.4)

Run on the lead's workstation (numpy, Linux shell of the workstation) from §15.3's caches. By the rule fixed below,
unchanged:
- **C1 (convergence) holds** at every setting and site.
- **C2 fails at one setting**: MIMIC-IV, case base, k = 5 (−0.003, −0.007 to +0.002); it holds at the other 3 case-base
  k's and at every within-sample setting. §15.3 is "not robust" by the rule's letter; its MIMIC-IV reading is
  restricted to k ≥ 10.
- C3: INSPIRE's "no closer than Euclidean" is a near-tie that depends on k; MIMIC-IV and eICU are closer at every k.
- **R2: generic at all three sites.** Transported–local's agreement is what any two metrics with its overlap show.
- **Consequence for the article.** "Many neighbourhoods, the same risk" is reported as a property of kNN over a
  redundant case base, not as a transport finding nor as evidence of a shared geometry.

## 2026-11-07 — Post hoc, pre-specified: robustness of the neighbour alignment (`hypo` §15.3; lead; plan and rule fixed before running)

**Why.** The second novelty review (2026-11-07) raised two objections to §15.3:
1. **k and set size.** Mutual-kNN alignment depends on k and on how dense the set of candidates is (Koepke et al.,
   2026). By construction, §15.3's case-base measure uses the same case base for every metric, and its within-sample
   measure the same 1,000 items, so the n-axis does not change the gallery. But the readings were shown at k = 10
   and 50 only.
2. **"Other neighbours, the same vote" may be generic.** Transported–local shares 14 % of its 50 neighbours with
   logits correlated 0.91. Whether any two metrics with that little overlap agree as much was not measured.

**What has been seen.** §15.3's tables at k = 10 (within) and k = 10, 50 (case base), and all 300 pairs' overlap
and logit correlation in `alignment_hypo_*.json` were computed, but the overlap–agreement relation over pairs was
never examined.

**Data.** Only `posthoc_alignment`'s per-metric caches (fold 0, one query per test patient, 25 metrics per site).
Numpy only; nothing is refitted.

**R1 (k and set size).**
- Case base at k ∈ {5, 10, 20, 50}.
- Within a sample (Huh et al.) at 1,000 items, k ∈ {5, 10, 20, 50, 100}, with the analytic chance k/(n − 1).
- Set sizes {500, 1,000, 2,000, 4,000} items at k = 10 and at k = n/100.
- Query (= patient) bootstrap, 1,000 resamples.

Readings checked at every setting:
- C1, convergence: halves > n2,000 pair > n500 pair;
- C2, transported–local below the two halves: case base, upper 95 % limit of the difference < 0; within a sample,
  point estimate;
- C3, against Euclidean–local: described; §15.3 found INSPIRE not closer and MIMIC-IV and eICU closer.

**Rule R1.** §15.3 is *robust* if C1 and C2 hold at every setting at all three sites; otherwise the settings where
they fail are reported and §15.3's readings are restricted accordingly.

**R2 (overlap against agreement).** Over the 300 metric pairs of each site, fit by least squares
r_logit = a + b × overlap (case base, k = 50) on the pairs without the transported metric. Then compare the
transported–local pair with the 95 % prediction interval at its overlap. Readings:
- *generic*: inside the interval;
- *transport agrees more* than its overlap predicts: above it;
- *transport agrees less*: below it.

The local–local pairs (500 to half) are reported beside it. The pairs are not independent, so the interval is
descriptive.

Script: `scripts/posthoc_alignment_robustness.py`. Outputs: `posthoc_alignment/robustness.{json,md}`. Result below and
as an addendum to §15.3, whatever it is.

## 2026-11-07 — Result: the intervention-state test (`hypo`; REPORT §19)

Run on the lead's workstation with v0.5 (= `hypo`'s frozen code); gate exact at every site; freeze record checked at
start and end. Read by the rule fixed below, unchanged:
- **P1 (weights): false.** Sedation's share is 0.41 % in the metric learned without INSPIRE, 0.54 % and 2.36 % in the
  two learned with it.
- **P2 (sedation): no failure to repair** at MIMIC-IV or eICU; B0 by sedation safe overall. Not a confirmation.
- **P3 (ventilation control): holds.** No failure, no repair anywhere; B0 by ventilation not safe at INSPIRE (upper
  limit +0.0010, at the threshold).
- Reference: the vasopressor stratum fails at MIMIC-IV (repaired by B0, +0.0040) and, under this definition, at eICU
  (not repaired, −0.0002). LightGBM transported pays far more on vasopressor landmarks than off them at both sites.
- **Consequence for the article.** The mechanism's general form is not supported; the same-state rule is reported as
  a site-specific repair for the vasopressor, and the vasopressor stratum as where transport costs most for every
  model. The claim "a source-missing input is under-weighted and breaks transport" is withdrawn.

## 2026-11-07 — Post hoc, pre-specified: does the intervention-state mechanism predict sedation? (`hypo`; lead; plan and rule fixed before running)

**Why.** The second novelty review (2026-11-07) ranks the vasopressor finding (REPORT §15.1, §18) as the article's most
original piece, but it is post hoc, at one site, with one intervention. The proposed mechanism makes predictions that
can be tested on data already in hand. **INSPIRE records neither vasopressors nor sedation** (`config/hypo.yaml`,
inputs missing at INSPIRE: `vaso_on`, `sedation_on`, `vaso_hours`), so the mechanism predicts for sedation what was
seen for the vasopressor. Mechanical ventilation (`vent_on`) is recorded at all three sites and is the negative
control. This partly replaces an external confirmation in NWICU, which lacks `inputevents` and may lack the MAP.

**What has been seen before this plan.** The weight tables (§7.4, §14.2) include the sedation and ventilation inputs.
No performance by sedation or ventilation stratum has ever been computed. The vasopressor results are known; the
vasopressor is re-run only as the reference case.

**The mechanism, stated so it can fail.** An input constant within one development source gets little weight in the
jointly learned diagonal metric. Where the intervention changes the physiology in the target, the transported metric
pays. Neighbours drawn from the query's own intervention state (B0, as in §18) repair that without refitting.

**Arms** (all on the external stage's five folds; nothing refitted):
- stored: NCA transported, NCA local, Euclidean kNN (Phase 2), LightGBM transported (`comps`, uncalibrated), LightGBM
  local (Phase 2), MI transported (§17);
- new: B0 for each state `s` = the frozen transported metric, each query voting among case-base landmarks of its own
  state `s` (whole case base if fewer than k + 1 same-state landmarks). Gate: the frozen metric rebuilds the stored
  transported predictions on fold 0 to 1e-9.

**Quantities** (patient bootstrap, 2,000 resamples, the same patient weights for every stratum and arm):
- cost by stratum = L(transported) − L(local), in `s` = 1 and `s` = 0, and their contrast (on − off);
- repair = L(transported) − L(B0) in `s` = 1; safety = upper 95 % limit of L(B0) − L(transported) over all landmarks;
- R = (L_euclid − L_arm)/(L_euclid − L_local) for the transported arm and for B0, with 90 % CI and the denominator
  rule of §16 (not evaluable below 0.005 nats or with a CI including 0);
- descriptive: each state's weight share in each transported metric; LightGBM's transport cost by stratum.

**Per site and state:** failure = cost in `s` = 1 with lower 95 % limit > 0 **and** contrast with lower 95 % limit > 0;
repairs = repair ≥ **0.001 nats** (a minimum effect, the A-sel lesson) with lower 95 % limit > 0; safe = safety < 0.001.

**Pre-specified predictions** (sedation can be tested only where it is recorded: MIMIC-IV and eICU as targets; INSPIRE
as a target has no sedated stratum):
- **P1 (weights).** Sedation's weight share is lower in the metrics transported to MIMIC-IV and to eICU (learned with
  INSPIRE) than in the metric transported to INSPIRE (learned on MIMIC-IV and eICU).
- **P2 (sedation).** Wherever the sedated stratum fails, B0 repairs it, and B0 is safe at every site where sedation is
  recorded. Reading: *confirmed* (both hold), *not confirmed*, or *no failure to repair*. The mechanism does not say
  that sedation must fail: a failure needs the intervention to change the physiology at the target (eICU showed no
  vasopressor failure). "No failure" is reported as such and is not a confirmation of P2.
- **P3 (ventilation, control).** At no site does the ventilated stratum fail, and B0 by ventilation does not repair.

**Limits fixed in advance.**
- The landmark leak of v0.5 (audit 2026-11-05, item 1) also touches `sedation_on` and `vent_on`, above all in eICU,
  whose supports are charted points; MIMIC-IV's sedation comes from `inputevents` intervals and is hardly affected.
  The strata are the frozen v0.5 landmarks'. A sensitivity analysis on v0.6 landmarks is a later item.
- One analyst knows the vasopressor result; the predictions follow from it, which is the point of the test.
- No correction across the three states: P1–P3 are read jointly, as stated, not as separate significance tests.

Script: `scripts/posthoc_state_test.py` (frozen hypo code via PERISIM_SRC). Outputs: `posthoc_state_test/`,
`reports/posthoc_state_test.md`. Result to be added below this entry and as REPORT §19, whatever it is.

## 2026-11-06 — Post hoc: a fixed mutual-information metric against the transported learned metric (`hypo`; lead; plan fixed before running)

**Why.** The literature review for the article (2026-11-06) found that the closest prior work, PRAM (Jeong et al.,
2026, medRxiv), already swaps a local case bank under a FIXED distance (cosine, or Euclidean weighted by mutual
information), and names learned NCA-type metrics as open future work. If the transported NCA metric does not beat a
fixed mutual-information metric used the same way, the novelty reduces to "a kNN with a local case base", which PRAM
partly covers. This is the decisive comparison for the framing "transport the metric, not the model".

**Status.** Exploratory, post hoc, run with `hypo`'s frozen code (`PERISIM_SRC=frozen/hypo/src`; the freeze record is
checked at the start and the end; nothing frozen is refitted). The transported NCA, local NCA and Euclidean
predictions are already known (REPORT §4, §16); the mutual-information arms have not been computed when this plan is
fixed. Code: `scripts/posthoc_fixed_metric.py`.

**Arms** (every arm: each site X's own out-of-fold case base, the external stage's five folds, k = 50, distinct
patients, the same smoothed vote; α tuned by leave-one-out on the case base with the Euclidean comparator's seed):
- NCA-transported, NCA-local, Euclidean: the stored predictions (external stage; Phase 2's `euclidean`).
- **MI-transported** (primary comparator): Euclidean kNN on the neutral axes of the Euclidean comparator (encoder
  fitted on the fold's case base), each variable weighted by its mutual information with the outcome computed on all
  landmarks of the two development sites (the transported metric's development data).
- MI-local: the same, with mutual information computed on the fold's case base.
- NCA-weights, neutral axes: the transported metric's weights on the neutral axes (no shapes): isolates the weights.
- Mutual information: plug-in estimator, natural log; continuous and ordinal variables on 20 quantile bins of their
  observed values; binary and missing-indicator variables as 0 / 1; nominal and procedure variables as categories,
  those with < 50 landmarks pooled; observed rows only. Weights = mutual information (0 → variable dropped), rescaled
  to sum to the number of variables (the scale does not change the neighbours).

**Gate.** Rebuilding the Euclidean comparator with the script's own code (weights 1) on fold 0 of every site must
reproduce Phase 2's predictions to 1e-9 on the same rows; otherwise the script stops.

**Primary endpoint and rule.** At each site, Δ1 = log-loss(MI-transported) − log-loss(NCA-transported), with a 95 %
patient-bootstrap CI (2,000 resamples, the same for every arm of a site). The learned metric adds over a fixed
mutual-information metric at a site when the lower limit is > 0.
- Supported if it holds at all three sites; not supported if it holds at none (the article's novelty claim then
  reduces to the local case base and must be reframed); otherwise reported site by site.

**Secondary (descriptive).**
- Q = (L_euclidean − L_MI-transported) / (L_euclidean − L_NCA-transported): the share of the transported NCA metric's
  gain over the Euclidean kNN already obtained by mutual-information weights (90 % and 95 % CIs; not evaluable when
  the denominator's 95 % CI includes 0 or it is below 0.005 nats).
- R for MI-transported (§16's ratio, relative to the local NCA metric), to set beside the NCA metric's R.
- Δ2 = L(MI-local) − L(NCA-transported); Δ3 = L(NCA-weights, neutral axes) − L(NCA-transported) (the shapes);
  L(MI-transported) − L(NCA-weights, neutral axes) (weights against weights on the same geometry).
- AUROC, calibration in the large and slope of every arm; Δ1 by `vaso_on` stratum (v0.5's landmarks: the strata are
  partly defined with information after L, REPORT §15, so only descriptive).

## 2026-11-06 — Result: two repairs of the vasopressor failure (`hypo`)

Run on the lead's workstation with `hypo`'s frozen code (freeze record checked at the start and the end; the gate
reproduced the stored transported arm exactly at every site; the NCA refit reproduced the frozen weights exactly).
By the pre-specified rule, B0 (the frozen metric with neighbours of the query's vasopressor state) and A-sel work;
A-strong is not safe; B neither repairs nor is safe. **A-sel is not read as a repair**: λ = 0 was chosen at every
site, and its effect (+0.0003 nats, lower limit +0.00001) is negligible with the stratum's calibration unchanged;
the rule lacked a minimum effect size, a fault of the plan recorded here rather than corrected after the fact.
Details: results/hypo/REPORT.md §18.

## 2026-11-06 — Post hoc: two repairs of the vasopressor failure — an NCA regularised toward mutual information, and a vasopressor-conditioned metric (`hypo`; lead; plan fixed before running)

**Why.** §17: the metric transported to MIMIC-IV (learned on INSPIRE and eICU) gives the three vasopressor inputs 0.08 %
of its weight and under-predicts in the vasopressor stratum (calibration in the large +0.189), while a fixed
mutual-information metric does not fail. Two repairs, learned on the development pair only (no information from the
target site), test whether the failure can be avoided: (A) keep the learned weights near a univariate
mutual-information profile, so a rare but informative input cannot be discarded; (B) let the metric depend on the
vasopressor state, the simplest non-separable deformation (two physiologies, two metrics).

**Status.** Exploratory, post hoc, `hypo`'s frozen code (`PERISIM_SRC=frozen/hypo/src`; freeze record checked at the
start and the end; the frozen artefacts are only read; the new metrics are new artefacts of this analysis). None of
the new arms has been computed when this plan is fixed. Code: `scripts/posthoc_vaso_repair.py`.

**Common to every arm.** The development pair's learning sample is reproduced exactly (`learning_sample`, the frozen
artefacts' seed, 30,000 landmarks per site); the transported metric's encoder with its shapes is kept (the geometry
does not change, only the weights); the NCA settings and seed are the frozen ones; the case base is the site's own
out-of-fold patients, k = 50, α by leave-one-out on the case base with the external stage's seed. The vasopressor
state is `vaso_on` (missing → 0, not running).

**Reference arm.** NCA-refit: the NCA refitted on the reproduced sample without prior, same procedure as the repairs,
so that refit noise is not read as an effect. The stored transported predictions and the mutual-information arm of
§17 are reported beside it.

**Gate.** The transported arm rebuilt by the script from the frozen metric must reproduce the stored predictions on
fold 0 of every site (|Δp| ≤ 1e-9); otherwise the script stops. How closely NCA-refit reproduces the frozen weights
(correlation of log weights) is reported, not gated.

**Arms.**
- A-sel: NCA with a mutual-information prior (each variable's share of the mutual information on the learning
  sample; one group per variable; the existing λ penalty of SPECS §5.4). λ chosen in {0, 0.01, 0.1, 1, 10} inside the
  development pair: an 80 / 20 patient split of the learning sample, fit on 80 %, log-loss of the kNN predictions on
  the 20 % with the 80 % as case base; the lowest wins (ties → the smaller λ); then refitted on the whole sample.
- A-strong (secondary): the same with λ = 1 fixed.
- B0, stratified: the frozen transported metric with neighbours restricted to the query's vasopressor state (the
  case base split by state, α per part).
- B, conditional: NCA weights learned separately on the vasopressor-off and vasopressor-on rows of the learning
  sample (no prior), each used with the same-state part of the case base. A state with fewer than 1,000 learning
  rows uses NCA-refit's weights.

**Primary endpoint and rule.** For each repair arm, in MIMIC-IV's vasopressor stratum: Δ = L(NCA-refit) − L(arm),
95 % patient-bootstrap CI (2,000 resamples, the same for every arm of a site); the arm repairs when the lower limit
is > 0. Safety: at every site, the upper 95 % limit of L(arm) − L(NCA-refit) over all landmarks is < 0.001 nats. An
arm works when it repairs and is safe at all three sites.

**Secondary (descriptive).** Overall and stratified log-loss differences against NCA-refit and against the
mutual-information arm; AUROC, calibration in the large and slope by stratum; λ chosen; the vasopressor inputs'
share of the weights. The strata use v0.5's landmarks (supports partly defined after the landmark, REPORT §15): the
stratified reading, and B0/B's use of `vaso_on` at inference, inherit that limitation.

## 2026-11-06 — Result: fixed mutual-information metric against the transported learned metric (`hypo`)

Run on the lead's workstation with `hypo`'s frozen code (freeze record checked at the start and the end; the gate
reproduced the Euclidean comparator exactly at every site). The pre-specified rule came out **mixed**: the learned
metric adds over a fixed mutual-information metric in INSPIRE and eICU, not in MIMIC-IV, where the deficit sits in the
landmarks with a vasopressor running (a calibration failure of the transported NCA metric in that stratum). Details
and limits: results/hypo/REPORT.md §17, which also revises §16's reading of MIMIC-IV (a weight-learning failure, not
an unobservable input). Not changed by this result: the plan, the arms and the rule of the entry below.

## 2026-11-05 — v0.6: the defects of the 2026-11-05 audit fixed; v0.5 kept in frozen/; history condensed (lead)

**What happened.** A software audit of the whole repository (six independent reviews, each finding reproduced on
synthetic data or on the published aggregates) found the defects below. The lead decided to fix all of them now, as
v0.6, to keep v0.5's frozen code in `frozen/` (see frozen/README.md; code and configuration hashes recomputed from
those files match the three freeze records), and to condense the repository's history into one snapshot, because
some published files had leaked derivable small cells.

**What does not change.** Every published result (T2, `hypo`, `roll`, the post hoc analyses) is v0.5's and is not
regenerated. No finding changes a confirmatory verdict (T2 H1–H6, `hypo` H1–H5, `roll` H1–H7). Where a finding bears
on the reading of a result, the report says so (hypo REPORT §15 and §16, roll REPORT §10). v0.6's code hash differs
from every v0.5 freeze, so v0.5's stages and post hoc scripts run only with `PERISIM_SRC=frozen/<exp>/src`.

**Privacy (published files corrected in place; `scripts/sanitize_results.py`, re-runnable with `--check`).**
- Derivable counts of 1–9 were blanked, and nothing else was changed: the suppressed shift and drift shares
  (complementary suppression), out-of-support percentages next to an n in the T2 transport tables (C, F and
  transport.json), out-of-range and missingness percentages (E, posthoc_transport), three eICU vasopressor rates in
  posthoc_map_vaso, and an INSPIRE flow step recoverable from the rounded flow.
- Code: `suppressed_shares` adds complementary suppression; the transport tables route percentages through
  `privacy.share`; `privacy.counts` labels every suppressed cell "<10" with a suppressed total ≥ 10; `privacy.flow`
  merges small steps instead of rounding; `audit.py` checks complements and shows quantiles only with ≥ 10 values
  beyond them; `shape_tables` labels the outer bins as open intervals (no training minimum or maximum);
  `export.publish` runs the identifier scan.
- `check_public_safety.py` reads staged content from the index, forbids more file types, and scans every CSV, JSON
  and report under results/ for small cells; `publish_supplement.py` and `publish_results.py` check before writing.

**Extraction (T2; T1/T3 labels).**
- KDIGO thresholds compared with a 1e-9 tolerance: an exact boundary rise (0.9 → 1.2 mg/dL) now counts. Affects T1
  and T3, which had not run; T2 does not use it.
- Ids and codes normalised on both sides of every comparison (float itemids no longer drop events silently);
  medication windows use max(end, start); MIMIC GCS needs three distinct components and INSPIRE's duplicates are not
  summed; tied "last values" resolved deterministically; empty inputs and null hadm_id handled; INSPIRE's limitation
  flag has its own value list; the metamorphic leak test also strips post-t0 stay columns; `conform()` on an empty
  frame no longer adds a null row.

**Method.**
- A frequent maximum (≥ 50 rows) of a discrete input gets its own bin (ASA 4 and 5, GCS 15 were merged with the bin
  below).
- The flatness test of the consensus shapes is run on the first differences of the curves (the local effects)
  against zero; on cumulative ALE noise its false-positive rate falls from about 0.35 to 0.10.
- α is looked up for the requested k (no silent fallback to 1.0); an unseen PCS code maps to its parent pool or to
  the neutral 0; T2 cut-points are refitted on each inner training part; LightGBM runs deterministically.

**Pipeline and freeze.**
- The transport stage runs the same freeze check as the external stage (code and scientific configuration against
  freeze.json). Post hoc checks compare the live hashes with the freeze record and fail on missing files.
- Package versions are compared with the freeze record (stop, unless `PERISIM_ALLOW_ENV_DRIFT=1`, recorded as a
  deviation); a disabled freeze gate is recorded and marks the results as descriptive; stale transport tables are
  cleared; `--force` refits phase 1; the relabel cache key covers evaluation and code.

**`hypo`.**
- Supports "running now" (vasopressor, sedation, ventilation) use only records before the landmark; in v0.5 charted
  points were merged into runs over the whole stay before the cut, so a record after L could set them at L (eICU's
  infusions most). eICU infusion stops end a run; non-numeric rates are not taken as running.
- The α leave-one-out prior leaves out the query's own patient (also in H5's γ vote); MIMIC labs are timed at their
  result (storetime); `died_in_window` counts only deaths in the ICU; the freeze hashes the landmark files and Phase
  2 predictions and re-checks the pre-registration.

**`roll`.**
- H4 (b) is computed, for each arm, on the patients predicted by the model in service (v0.5 counted the old model
  for a few checks after each deployment, which favoured PERISIM); the window AUPRC is the expected average
  precision under random tie-breaking (v0.5 penalised PERISIM's tied votes by about 0.007); H6 (c) excludes untested
  windows; H6 (a)'s SE comes from a patient bootstrap; the W sensitivity uses the configured W; a cross-cohort
  stretch's deployment delay uses the stretch's own admission rate (a deviation from design §7.6).

**pmknn 0.4.0** (branch `pmknn`): group and site columns optional at prediction and for a local case base; no
individual value in the exported JSON (bin edges and ECDF knots backed by ≥ min_bin rows; counts rounded, 1–9
null); NCA with a single site; the two method fixes above; the validation against the frozen code now covers
repeated patients and two sites.

**Not fixed (documented).** `hypo`'s drift analysis (S4) still includes the query's patient in its single LOO prior;
`roll`'s block length uses the median window size; three of `roll`'s H6 SEs (not used for pooling) still come from
the orders.

**History.** The repository's history, branches and tags were replaced by a single snapshot of this state. The v0.5
freeze tags (`perisim-freeze-t2-2026-10-27-141040`, `perisim-freeze-hypo-2026-10-29-120652`,
`perisim-freeze-roll-2026-10-30-173256`) no longer exist; their code is in `frozen/` with the original commit ids.

## 2026-11-05 — joblib pinned below 1.6 (environment only; no frozen code or configuration changed)

- joblib 1.6 removed `joblib.externals.cloudpickle`. `src/hypo/frozen.py` imports it to save and load `hypo`'s frozen
  artefacts, so in a fresh environment `hypo`, `roll`, the `hypo` post hoc scripts and three test modules
  (`test_hypo_ablation`, `test_hypo_external`, `test_hypo_posthoc`) failed at import.
- The import is not changed: `src/hypo` is under `hypo`'s and `roll`'s code hashes, and the artefacts already saved
  reference the vendored module when unpickled, so a standalone `cloudpickle` would not read them reliably anyway.
- Instead `requirements.txt` pins `joblib>=1.3,<1.6` (lock: 1.5.3). Dependency ranges are not part of any freeze
  (README, Paths and resources), so the T2, `hypo` and `roll` freezes are unaffected. With joblib 1.5.3 the whole
  suite passes (158 passed, 5 skipped).
- A later, unfrozen version may move to a standalone `cloudpickle` once the artefacts are re-saved with it.

## 2026-11-05 — The technique renamed: portable-metric kNN, PM-kNN (lead)

- **CE-kNN (consensus effect-space kNN, 2026-10-30) becomes PM-kNN (portable-metric kNN).** The ablations of `hypo`
  (REPORT §14) and T2 (PERISIM@50 against diagonal NCA@50: Δ log-loss 0.0004, CI −0.003 to 0.003; λ ≈ 0 in every
  fold) put the accuracy in the learned diagonal-NCA weights, not in the consensus of the curves. The name now says
  what the method offers, a metric learned at some sites and used unchanged at another with the local case base,
  rather than the component that adds least to prediction. The shapes stay as the layer for reading the metric and
  for drift.
- **Nothing in the method changes.** The frozen code and configuration are untouched, and so are the documents whose
  hashes the freezes record or that are records of their date (SPECS, pre-registrations, design and freeze records,
  reports, earlier DECISIONS entries): they keep "CE-kNN". The READMEs use the new name and point to it.
- **The library** moves from the branch `ceknn` (package `ceknn` 0.2.0) to the branch `pmknn` (package `pmknn` 0.3.0,
  class `PMKNN`). Metric files written by `ceknn` load unchanged.

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

## 2026-10-24 — Pod layout (lead)

- The datasets are the lead's own PhysioNet copies, unzipped in `<workspace>/data/{inspire,mimic4,eicu}` outside
  `updown`, for speed; `run_pod.sh` uses that folder when it exists and the KHDP mount `<workspace>/datasets`
  otherwise. Replaces the data path of 2026-10-21.
- Python environment in `<workspace>/venv` (the lead may create it beforehand; an existing environment is used as
  it is and never deleted) and run files in `<workspace>/perisim_run`, replacing `~/.perisim`: no hidden folders.
  Pip and matplotlib caches are not written to `~/.cache`; pytest writes no `.pytest_cache`.
- The repository is cloned in the workspace root (`<workspace>/perisim`) and everything is run from the root:
  `bash perisim/run_pod.sh`; the hints in messages and reports follow how the script was called.

## 2026-10-24 — Work folder on the local disk (lead)

- The first run with the three cohorts failed in `extract`: in `updown` (shared object storage) the temporary cache
  files written by the parallel extractions vanished before their rename. The work folder moves to the workspace's
  local disk (`<workspace>/perisim_work`, `PERISIM_WORKDIR`), like the data and the environment.
- After every run (also after a stop) the reports, the pod's docs and the export package — aggregates only — are
  copied to `updown/perisim/<user>/results/` (`PERISIM_PUBLISH_DIR`, empty = no copy); never predictions, extracts
  or cache. The local work folder remains the source of truth.

## 2026-10-24 — Completeness audit with the three cohorts (lead)

- MIMIC-IV `rrt_start` (prevalence < 0.5 %) accepted: RRT after the ICU exit is recorded only back in the ICU, as in
  INSPIRE; same definition, rare component.
- MIMIC-IV medication: the hospitalizations without any eMAR record (early years) use `prescriptions` (orders,
  start–stop, systemic routes only, the same name → ATC mapping), with their own D6 coverage marker; eMAR keeps
  priority where it exists (administrations rather than orders). An order running into the window counts from the
  window start (ongoing therapy, never a new course).
- The non-systemic routes of `prescriptions` are taken from its route inventory (eyes, ears, topical, inhaled,
  nasal, intradermal, catheter lock / dwell / flush, intravitreal, nerve blocks…); every configured MIMIC-IV
  item matched its d_items / d_labitems label (marked VERIFIED).
- The antibiotic-course rule now reads every record as an interval [start, end]: an order covers its days, an eMAR
  administration is one instant (identical result for point records).

## 2026-11-05 — Post hoc: T2's local–local reference for the neighbour alignment (lead; plan fixed before running)

Asked for by the lead after `hypo` REPORT §15.3. In `hypo`, two local metrics learned on disjoint patients share more
neighbours as n grows. INSPIRE, the site with the least information, converges least. The lead's reading: with less
information there is less common structure to converge to, so T2, with little information at ICU discharge, should
converge less than `hypo`. The 2026-11-04 plan made no local–local reference for T2, so its curve is unknown. This
adds it.

It is exploratory, not pre-registered, and has no decision rule. It runs in `scripts/posthoc_alignment.py`, outside
the frozen code. T2's and `hypo`'s freeze records are checked at the start and at the end. The plan is fixed before
any T2 draw is fitted. The 2026-11-04 results are only read, and `hypo` is not rerun.

- **The new metrics:**
  - **Local metrics learned on n patients** of the site's pool: n = 500, 1,000, 2,000, 4,000 and half the pool
    (MIMIC-IV 9,015, eICU 8,935), two disjoint draws at each n. The sizes, the draw rule and the seed rule are
    `hypo`'s.
    - Each one is fitted with the transport stage's own A(iii) procedure (`_fit`): the nested fit with D7's stopping
      parameters, on the drawn patients' labelled pool stays, with the site's cross-fitted tertile labels.
    - 4 BLAS threads per fit (DECISIONS 2026-11-04, the thread note), five fits side by side.
    - Each draw's case base is the whole pool, as for `hypo`'s draws. Its α is chosen as the method chooses it, by
      patient-level leave-one-out on that case base.
    - A draw whose fit fails is reported as failed, not replaced.
  - **The Euclidean kNN as the floor:** the external stage's Euclidean comparator (the neutral axes, uniform weights),
    as `hypo`'s z-score variant. Its predictions use the frozen α, and serve only the agreement column.
  - The frozen and local metrics and the re-deformations stay the 2026-11-04 run's.
- **Unchanged:**
  - the split, the queries and the case base (the pool's labelled stays, k = 50);
  - Huh et al.'s within-sample recipe, with the same 1,000 items, draws and permutations as the 2026-11-04 run.
- **What is read** (descriptive):
  - **T2's two-draw alignment against n,** beside `hypo`'s. The n-axis figure gains a panel per T2 site.
  - **Where T2's frozen–local alignment falls on that curve** (0.30 in MIMIC-IV, 0.26 in eICU), as for `hypo`.
  - **The same alignment against competence.** n does not mean the same in both experiments: a T2 patient gives one
    stay to learn on, and a `hypo` patient up to three landmarks. So each draw's competence is also reported, as the
    relative log-loss gain over the case base's prevalence (1 − log-loss / log-loss of the prevalence, on the test
    rows), and the learning rows beside the patients. The alignment against competence puts both experiments on one
    axis.
- **How it would read** (no rule):
  - T2's two-draw curve below `hypo`'s at the same n would mean that T2 converges less.
  - If T2 then lies on `hypo`'s alignment–competence relation, the difference goes with competence. That fits the
    lead's reading: less information, less competence, less convergence.
  - If T2 lies below `hypo` even at the same competence, something other than competence separates them.
  - The two experiments' competence ranges may not overlap. In the dry run (draws of 100 and 200 patients, a check of
    the mechanics only), T2's whole-pool local metric reached 0.10–0.12, while `hypo`'s draws span 0.14–0.19. Where
    they do not overlap, "at the same competence" is an extrapolation of `hypo`'s relation, and is reported as one.
- **Limits:**
  - The experiments differ in more than information: the outcome (three ordered classes against one binary), the
    variables, the case base (stays against landmarks) and the method's details (T2 has its SAGE stage; λ ≈ 0 in every
    fold). The comparison cannot isolate information.
  - Two draws per n, one split.
- **Cost:** 20 fits. The stage's two local fits on the whole pools took about 3.7 h side by side, so about 3–5 h in
  all. The fits are cached and the run is resumable.

## 2026-11-05 — Transport equivalence of the learned metric, by the bioequivalence convention (lead; criterion fixed with the point estimates known)

**Status: exploratory, not pre-registered.** The criterion below was fixed after the point estimates of `hypo` and T2
were known (listed at the end). It is taken from an external convention, not tuned to them, and it is reported with
that label. The intervals (90 % and 95 %) had not been computed when it was fixed.

- **The estimand.** On the same test rows, the same local case base and the same k,
  R = (L_euclidean − L_transported) / (L_euclidean − L_local): the share of the local metric's gain over the
  Euclidean kNN that the transported metric keeps (1 = all of it, 0 = none).
- **The criterion: bioequivalence.** The transported metric is read as a generic of the local one. It is
  **equivalent** when the lower limit of R's **90 %** CI is **≥ 0.80** (FDA, 2001; EMA, 2010: 90 % CI of the ratio
  within 0.80–1.25, two one-sided tests at α = 0.05). The upper limit does not apply: R > 1 means the transported
  metric does better than the local one.
- **Not evaluable** when the denominator (the local gain over the Euclidean kNN) has a 95 % CI that includes 0 or is
  below 0.005 nats. A ratio with a small denominator is unstable (Buyse & Molenberghs, 1998, on the proportion of the
  treatment effect explained, whose lower-limit rules of 0.5 / 0.75 (Freedman et al., 1992) are the nearest
  analogue).
- **Reported with it:** the 95 % CI of R, the denominator, and the absolute cost (Δ log-loss transported − local),
  since a small R can go with a negligible cost.
- **Units.** `hypo`: INSPIRE, MIMIC-IV and eICU, all landmarks, and by `vaso_on` where both strata exist (the
  external stage's predictions over the five folds, Phase 2's Euclidean kNN, as 2026-11-04). T2: MIMIC-IV and eICU,
  the transport stage's split and own labels (A(i), A(iii) and the local Euclidean kNN, rebuilt from the frozen
  artefacts; their log-loss must reproduce transport.json). 2,000 patient-level resamples, the same for every
  quantity of a unit. No decision rule across units: each gets its verdict.
- **Known when the criterion was fixed** (point estimates; 95 % CIs where published):
  - `hypo`: INSPIRE 0.91; MIMIC-IV 0.74 (0.67–0.81); eICU 0.96 (0.92–1.00); MIMIC-IV with a vasopressor running
    0.44 (0.25–0.59), without 0.84 (0.76–0.91);
  - T2: MIMIC-IV 0.49; eICU 0.72 (from transport.json's aggregates).
- **Reading planned:** the vasopressor strata of MIMIC-IV are post hoc (2026-11-04). They can locate a
  non-equivalence; they do not confirm its cause.
- Code: `scripts/posthoc_equivalence.py` (`run_local.ps1 equivalence`), outside the frozen code; the freeze records are
  checked at the start and the end.

## 2026-11-04 — `hypo` post hoc: does the MAP–risk relation change with a vasopressor running? (lead; plan fixed before running)

Asked for by the lead after §15.1: half of MIMIC-IV's transport cost sits in the landmarks with a vasopressor
running. Is the MAP physiology different there, with a MAP held up by the drug carrying another risk? It is
exploratory, not pre-registered, and has no decision rule. It runs in `scripts/posthoc_hypo_map_vaso.py`
(`run_local.ps1 hypo-map-vaso`), which reads only, and checks the freeze record at the start and at the end. The plan
below is fixed before any curve is computed.

- **Data:** MIMIC-IV (primary) and eICU (descriptive), the landmarks of fold 0's training patients, as S5. Strata:
  `vaso_on` 1 and 0.
- **Observed rates:** the share of landmarks followed by hypotension in the next 6 h, by 5-mmHg bins of the last MAP
  and of the 6-h minimum, in each stratum. 95 % CIs from 500 patient resamples; bins of fewer than 100 landmarks are
  not shown. This shows the level, which the centred shapes do not.
- **The consensus shapes in each stratum**, by S5's procedure (`hypo`'s 40-model shape loop and settings) on a
  learning sample of the stratum, at most 3 landmarks per patient:
  - **Equal sizes:** the no-vasopressor patients are split into two random halves (a and b). Each half's sample is
    drawn to the size of the vasopressor sample.
  - **Pairs:**
    - vasopressor – no vasopressor a, and vasopressor – no vasopressor b: the contrast;
    - no vasopressor a – b: the noise of learning the shapes at that size.
  - **Per MAP summary** (last, 6-h mean, 6-h minimum, 6-h SD, 6-h slope): the weighted correlation r and the slope of
    one log-odds curve on the other (S5's comparison).
  - **The MAP block (§13.2):** the block threshold, and the odds ratios of a steady 70 and 65 mmHg against 85.
    Reported for the consensus, and as the median and 10–90 % over the 40 models.
- **Reading** (descriptive): a difference between the strata beyond the difference between the two no-vasopressor
  halves would show a MAP–risk relation changed by the drug.
- **Limit:** the stratum is not randomised. Patients on vasopressors are sicker in ways the inputs may not capture.

## 2026-11-04 — `hypo` post hoc: where MIMIC-IV's small transport cost sits (lead; plan fixed before running)

Asked for by the lead after an external read. In MIMIC-IV the transported metric costs 0.0028 nats against the local
one; it keeps 74 % of the local metric's gain over the Euclidean kNN, against 91 % in INSPIRE and 96 % in eICU. The
arterial line does not explain it: S7 gives the same cost and the same 74 % with and without a line. The two
candidates left are weights of §7.4:
- vasopressors running now (`vaso_on`): INSPIRE cannot see an infusion, and the swap was the one single-variable
  gain in §13.1;
- time since ICU admission (`L`).

It is exploratory, not pre-registered, and has no decision rule. It runs in `scripts/posthoc_hypo_strata.py`, which
reads only saved predictions and the landmarks, and checks the freeze record at the start and at the end. The plan
below is fixed before any stratum is computed.

- **Cost by stratum, MIMIC-IV, the five folds** (the external stage's own predictions, nothing refitted):
  - **Strata:** `vaso_on` 0 / 1, and L in [0, 6), [6, 12), [12, 24) and [24, 48] h.
  - **Per stratum:**
    - landmarks and prevalence;
    - Δ log-loss transported − local (positive = transport cost), with 500 patient-level resamples;
    - the stratum's share of the total cost;
    - the share of the local metric's gain over the Euclidean kNN that the transported one keeps;
    - AUROC of each.
  - **Contrasts:** the difference of Δ between `vaso_on` 1 and 0, and between the last and the first L bin, on the same
    resamples.
  - **eICU,** where the cost is 0.0005 nats, is the same table, descriptive.
- **The weight swaps of §13.1 by stratum** (fold 0, S4's test sample, their saved per-landmark log-loss):
  - the gain of the `vaso_on` swap, by `vaso_on` stratum;
  - the gain of the L swap, and of all weights local, by L bin.
  - A gain that sits in its own stratum (the `vaso_on` swap on landmarks with an infusion) reads as a site
    difference in that input; a gain spread evenly does not.

## 2026-11-04 — Post hoc: do metrics learned apart retrieve the same neighbours? (lead; plan fixed before running)

Asked for by the lead after reading the transport and drift results against the Platonic Representation
Hypothesis (Huh et al., 2024): models trained apart converge towards one representation, measured by how many
nearest neighbours they share. It is exploratory, not pre-registered, and has no decision rule. It runs in
`scripts/posthoc_alignment.py`, outside the frozen code. The script only reads the frozen artefacts and checks
`hypo`'s and T2's freeze records at the start and at the end. The plan below is fixed before any alignment is
computed.

- **The measure:** the mutual k-nearest-neighbour alignment of Huh et al. For each query, the share of its k nearest
  neighbours that two metrics have in common, averaged over the queries (1 = the same neighbours). Two versions:
  - **In the case base**, the neighbours CE-kNN votes with: each query's k nearest cases in the site's case base.
    - k = 50 (the method's) and k = 10 (Huh et al.'s). The primary is k = 50, a neighbour identified by its landmark
      (`hypo`) or stay (T2); identified by its patient is secondary.
    - The chance level is k / the number of cases.
    - 95 % CIs from 500 resamples of the queries (one per patient), the same resamples for every pair.
  - **Within the sample**, Huh et al.'s recipe: 1,000 queries, the neighbours among themselves, k = 10, 20 random
    draws (mean and SD).
    - Each pair gets a permutation null (Gröger et al., 2026): 200 permutations of the correspondence between the two
      metrics' items. Reported: the null's mean, and the calibrated score (alignment − null) / (1 − null).
    - It has the same n and k in both experiments, so it is the version that compares `hypo` with T2.
- **Beside each pair:** the agreement of the predictions on the same queries (correlation of the logits, mean |Δp|).
  It tells "the same neighbours" from "other neighbours, the same vote".
- **`hypo`**, fold 0 of each external site, as the external stage: case base = the training folds, queries = one
  random landmark per test patient, neighbours = distinct patients.
  - **Transport:** the transported (frozen) metric against the local one. The local one is the H1 arm's fold-0
    metric, refitted with its seed; it must reproduce the stage's predictions exactly.
  - **Learning noise and n:** local metrics learned on n = 500, 1,000, 2,000 and 4,000 training patients, and on half
    of them, with two disjoint draws at each n.
    - The alignment of the two draws: does it rise with n?
    - The alignment of each draw with the transported and the local metric.
  - **Architecture:** the 14 variants of the ablation (§14.3: geometry × weights, transported role). Each must
    reproduce its fold-0 predictions.
    - Every pair, and each variant against the local metric.
    - Competence is the log-loss on all the fold's test landmarks. Does a variant align more with the others the more
      competent it is? (Spearman, Huh et al.'s competence–alignment relation.)
    - The standardised Euclidean variant (z-score, uniform weights) is the floor.
- **T2**, MIMIC-IV and eICU, with the transport stage's split: case base = the pool's labelled stays, queries = the
  test stays, k = 50, no distinct-patient rule (as T2).
  - **The metrics:**
    - the frozen INSPIRE metric (A(i));
    - the local refit (A(iii));
    - the joint and the missing-only re-deformations of the drift analyses (their cached fits).
  - **The pairs:** frozen–local, joint–local, missing–local and frozen–joint. Does re-deforming move the
    neighbourhoods towards the local metric's?
  - **No local–local reference:** T2 has none without refitting its local metric (about 4 h), and none is made.
- **What would read as convergence** (descriptive, no rule): a transported–local alignment near the two-draw
  alignment at the largest n, and well above the z-score floor and the chance level.
- **Two notes from the run** (2026-11-04, written before the eICU part was rerun):
  - **Threads.** The first run was launched outside `run_local.ps1`, and INSPIRE's local refit did not reproduce the
    stage (max |Δp| 0.074). The NCA's result depends on the BLAS thread count, and the stage ran with 4 threads per
    job. The check stopped the run. Its caches were deleted and it was relaunched through `run_local.ps1`
    (OMP/MKL/OPENBLAS/NUMEXPR threads = 4). INSPIRE's and MIMIC-IV's local refits then reproduced exactly (0.0).
  - **A tie in eICU.** The check stopped the run again at eICU: the local refit differs from the stage at 1 of 6,431
    queries, by one vote (|Δp| = 1/54).
    - The cause is a tie, not the metric. For that query, the 50th and 51st neighbours are at the same squared
      distance to float32 precision in the stage's search, which embeds all the fold's test landmarks in one batch
      (0.79595757 both). Searched with the queries alone they are 4 × 10⁻⁶ apart (relative). BLAS rounds differently
      with the batch size.
    - **The check is relaxed, for every metric.** A metric reproduces if its predictions match to 10⁻⁹, except at
      queries where they differ by at most one vote and the k-th and (k+1)-th neighbours are within 10⁻⁵ relative
      distance (a tie at float32 precision). Such queries are counted and reported.
    - The alignment uses this run's neighbours. The tie changes one neighbour of one query.

## 2026-10-31 — `roll`: a declared deviation in the eICU pooling, after the single run (lead's decision)

- **The single run ended** on 2026-10-30 at 23:09, with all 1,190 units and no failure. The freeze was checked at the
  end of each of the four parts and of `endpoints` and `t2`.
  - Determinism: the first unit, rerun with 1 thread, gave the same predictions digest.
  - Reconstruction: rebuilding the FIFO window from the log gave 0 mismatches in 159 steps.
  - The interim numbers of MIMIC-IV temporal (previous entry) are identical to the final ones.
- **The defect.**
  - The frozen DerSimonian–Laird pooling (`roll.endpoints.dersimonian_laird`) leaves out hospitals whose bootstrap SE
    is zero (`se > 0`).
  - In a few hospitals every resample gives the same trigger share, so the SE is zero up to floating-point noise
    (~1e-16). Such a hospital passes the filter and gets an almost infinite weight.
  - The pooled H4 (b) then becomes that one hospital's value, with a CI of zero width: −0.900 [−0.900, −0.900] in
    eICU (a) (hospital H28) and −0.700 [−0.700, −0.700] in eICU (b) (H36).
- **The deviation** (prereg §9: a change after results exist is reported as one, and the original output is kept).
  - Every SE ≤ 1e-12 is treated as zero before the same frozen function is called. Those hospitals are then left out,
    as the frozen code intends.
  - It is done by `scripts/roll_deviation_pooling.py`, outside `src/roll`. It reads the frozen per-hospital estimates
    and SEs from `endpoints.json` and writes `endpoints_deviation.json`. The freeze is checked at its start and end.
  - The report gives both outputs, the frozen one first.
- **What changes:** 30 pooled results, all of them H4 (b) (one to two hospitals left out each). H6 (a) and every other
  endpoint are unchanged.
  - Confirmatory, PERISIM-FIFO minus arm 5: −0.54 [−0.61, −0.47] in eICU (a) and −0.53 [−0.58, −0.48] in eICU (b). The
    rule (upper bound < 0) is met either way, so the verdict on H4 does not change.
  - Against arm 5′ in eICU (a), reported and not tested: +0.20 [0.20, 0.20] frozen, −0.21 [−0.31, −0.11] corrected.
- **Seen in the same function, not changed:** in H2, a hospital where neither arm has any excess variance (a ratio
  of exactly 1, SE exactly 0) is left out by the frozen filter. H2 fails in every confirmatory stream, and a tie at 1
  cannot meet its rule (upper bound < 1), so this cannot change the verdict. The report counts these hospitals.

## 2026-10-30 — `roll` Phase 4: an interim look at MIMIC-IV temporal, N = 500, during the single run (lead's decision)

- **When.** The single run started at 17:55 in four parts. Part 1 (MIMIC-IV temporal, N 250 and 500) had finished,
  and 797 of the 1,190 units were done. The lead asked to see that stream's results before the run ended.
- **What was computed.**
  - The frozen `roll.run.endpoints`, called on the job `mimic_temporal|all|N500` only, from a script outside the
    repository. No code was changed.
  - The freeze was checked before and after. The computation finished at 21:54 and took 28 min, beside the three
    running parts.
  - The seed is fixed per stream, so `python -m roll endpoints` will give the same numbers for this stream.
  - Output: `<workdir>/experiments/roll/interim_mimic_temporal_N500.json`, an aggregate kept on the workstation.
- **What was read:** every endpoint of that stream.
- **What it cannot change.**
  - The code, the configuration and the streams are frozen, and the remaining parts go on unchanged.
  - Nothing was tuned, and no stream, hospital, N or window was added or dropped (prereg §9).
  - The numbers are reported with all the others in the report; they are not repeated here.

## 2026-10-30 — `roll` Phase 2: the implementation, its tests and the synthetic dry run (analyst; for the lead)

- **The code.**
  - `src/roll/` and `config/roll.yaml`;
  - the command `python -m roll` (`check`, `streams`, `dry`, `t2-check`), also `run_local.ps1 roll`.

  It reads `hypo`'s frozen metrics and comparators and T2's frozen artefacts, and changes nothing of either. Both
  freezes are checked at the start and the end of every command. The single real run is added with the freeze
  (Phase 3).
- **The tests** (`tests/test_roll.py`, synthetic data):
  - each patient is evaluated once and before entering; FIFO; exclusion by patient; the same sequence for every arm;
  - the trigger, the delay and the refractory period work as declared;
  - determinism;
  - a `hypo` case base is reproduced exactly;
  - the closed test picks the needed update; ADWIN; the gate; the metrics against scikit-learn; the block bootstrap;
    DerSimonian–Laird; T2's scores.

  On the real data (`PERISIM_TEST_ROLL_REAL=1`), a window equal to `hypo`'s fold-0 INSPIRE case base reproduces
  `hypo`'s transported predictions bit for bit.
- **The real streams, built** (`python -m roll streams`; counts only, nothing fitted or predicted):
  - 35 hospitals in each eICU stream, as decided at the gate;
  - 240 evaluation windows in MIMIC-IV, 74 in INSPIRE;
  - the admission gate refuses no landmark;
  - 142 eICU patients have landmarks in two hospitals.

  T2's artefacts load: 24,768 stays, 23,162 patients, all labelled.
- **Corrections the dry run found** (design updated):
  - the order of the closed test (design §7.5): the approved order, from the refit down, kept a badly miscalibrated
    model every time. Now: the nested recalibrations top-down, then the refit;
  - ADWIN in the variance form of its cut (§7.4): the Hoeffding cut missed a tripling of the calibration error;
  - the prevalence arm gets no discrimination metrics (its rate is the same for every patient of a step).
- **Implementation choices, declared in the design:**
  - a step never crosses an evaluation window's end (§5);
  - an eICU patient belongs to the hospital holding most of their landmarks and to their first year (§5);
  - a trigger whose closed test keeps the model still completes its cycle (§7.2);
  - the delay in a cross-cohort stretch uses the cohort's rate (§7.6);
  - the bootstrap's blocks are capped at half a segment (§10);
  - D2's anchors and schedule (§8.6);
  - T2's artefacts (§12).
- **Seen on synthetic data only: a risk for the real run.**
  - With a window of 120 patients and k = 50, PERISIM-FIFO under-predicted: O/E 1.17–1.28, calibration slope about
    1.5. With 60 patients (most of them neighbours), O/E was 1.9–2.3. The cumulative arm was close to calibrated
    (O/E 0.96–1.30).
  - The neighbours are distinct patients, each at its closest landmark. When k is a large share of the window's
    patients, the vote shrinks.
  - eICU's stream (a) has N = 100 with k = 50. This could count against PERISIM in H4 there.

  This is not an analysis, and neither k nor N is changed. Whether it happens on the real data could be read,
  before the freeze, from `hypo`'s saved S1 predictions with case bases of 100 and 250 patients. That would be a new
  aggregate of examined data, so it is the lead's decision. (Measured: next entry.)

## 2026-10-30 — `roll`: the rehearsal at real size before the freeze (lead's decision)

- **What ran** (`python -m roll rehearse --threads 16`, in the lead's terminal; 61 min):
  - one unit of each stream kind at the primary N, order 0: MIMIC-IV temporal; eICU (a) and (b) in the
    median-sized eligible hospital (H18); INSPIRE → MIMIC-IV; the INSPIRE control;
  - all 24 arms;
  - in-sample artefacts: MIMIC-IV served by ext-eicu, eICU by ext-inspire, INSPIRE by ext-mimic, the cross-cohort
    stream by ext-eicu.

  Nothing of the single run was computed.
- **What was read**: time per unit and per arm, the endpoints' time, determinism and reconstruction. No metric was
  read; the outputs stay in `work/experiments/roll/rehearsal/` and are not used.
- **Findings**:
  - no fault;
  - predictions identical with 1 and 16 threads;
  - the FIFO window rebuilt from the log at all 44 steps, with no mismatch;
  - the freezes of T2 and `hypo` unchanged.
- **Time**:

  | unit | run | slowest arms |
  |---|---|---|
  | MIMIC-IV temporal | 35 min | PERISIM cumulative 375 s, FIFO 209 s, LightGBM ideal 112 s |
  | INSPIRE control | 9 min | |
  | INSPIRE → MIMIC-IV | 8 min | |
  | an eICU hospital | 0.2–0.3 min | |

  Scaled to the design, the runs need about 27 hours in one process. The endpoints needed about 18 hours.
- **Changes, for cost only** (the results are the same; tested):
  - the bootstrap over windows vectorised;
  - the per-window metrics computed in parallel threads (one window per thread);
  - the W sensitivity read for PERISIM-FIFO and arm 5 only, as the pre-registration reports it.

  The endpoints now need about 2 hours.
- **Running the single run in parts.** `roll run --streams … --N …` selects a part. Parts never write the same unit,
  so several can run side by side in separate terminals.

## 2026-10-30 — `roll` Phase 3: the pre-registration draft and the freeze's code (analyst; for the lead's approval)

- **The draft**: [prereg_roll.md](prereg_roll.md). It holds:
  - the question;
  - the data;
  - what was done and seen before the freeze;
  - the frozen artefacts;
  - streams, arms, hypotheses and rules;
  - the expectations (H1 expected to fail against LightGBM; little drift; the calibration of eICU (a));
  - the limitations;
  - the registration.
- **The freeze's code** (`src/roll/frozen.py`; `python -m roll freeze --registration <ID>`). The record holds:
  - `roll`'s code and configuration hashes;
  - `hypo`'s and T2's freezes, and T2's final model;
  - the digest of all 1,190 stream-orders of the design;
  - the git tag and commit, and the hash of the pre-registration.

  It refuses to run once real results exist, or with uncommitted changes. The single run (`python -m roll run`,
  `endpoints`, `t2`) checks the record at its start and its end.
- **Added so that the draft describes what runs:**
  - the ML monitor's PSI (a reference);
  - the W sensitivity, the same predictions read in windows of W/2 and 2W;
  - stream (b)'s start counted as a boundary (recovery, post-boundary windows);
  - the determinism and reconstruction check of the first unit (`runs/governability.json`).
- **Approved by the lead the same day**:
  - the pre-registration as drafted;
  - the local registration `LOCAL-PREREG-ROLL-2026-10-30`;
  - a mechanics rehearsal at real size before the freeze, with in-sample artefacts and without looking at results;
  - pushing the branch `roll`.

## 2026-10-30 — `roll`: the calibration of `hypo`'s small case bases, measured before the pre-registration (lead)

- **The lead's decision.** Read the calibration of `hypo`'s S1 case bases (n random patients of the training folds,
  every fold and draw; external stage). This is a new aggregate of predictions already examined, read before the
  pre-registration, and declared there. Script: `scripts/feasibility_roll.py s1cal`. Freezes unchanged.
- **What it shows** (mean over the 25 case bases of each n, and their range):

  | n patients | O/E (INSPIRE / MIMIC-IV / eICU) | calibration slope | CITL range over case bases |
  |---|---|---|---|
  | 100 | 1.06 / 0.94 / 0.98 | 1.40 / 1.45 / 1.43 | −0.38 to +0.40 |
  | 250 | 1.03 / 0.97 / 1.01 | 1.20 / 1.15 / 1.17 | −0.22 to +0.26 |
  | 500 | 1.04 / 0.99 / 1.01 | 1.11 / 1.09 / 1.11 | −0.20 to +0.27 |
  | the whole pool | 1.02 / 1.01 / 1.01 | 1.02 / 1.01 / 1.03 | −0.02 to +0.05 |

- **Reading.**
  - The synthetic under-prediction does not appear on the real data: O/E is within 0.94–1.06 at every n.
  - Small case bases are under-confident: a slope of about 1.4 at 100 patients, 1.2 at 250.
  - Their calibration in the large depends on which patients they hold: ±0.4 at 100 patients.
  - In eICU's stream (a), PERISIM-FIFO's window CITL will vary from window to window for this reason too. That
    variation is the model's, not sampling noise, so it enters the excess calibration error (H4a) against PERISIM.
- **What is kept.** N = 100 in stream (a), as approved. Raising it after this reading would favour PERISIM with a
  choice made on seen data. The expectation is declared in the pre-registration, and H7 shows the curve against N.

## 2026-10-30 — `roll` Phase 0 gate decided; Phase 1, the design (lead; analyst's proposals for the lead's approval)

- **Phase 0 gate (lead).** Every proposal of feasibility_roll.md accepted:
  - MIMIC-IV with five blocks, 2008–2010 to 2020–2022 (four boundaries);
  - eICU hospitals with ≥ 150 patients in each of 2014 and 2015 (35);
  - INSPIRE as a random-order control and a starting cohort;
  - no sensitivity analysis ordering eICU by `patientunitstayid`;
  - cross-cohort pairs INSPIRE → MIMIC-IV, eICU → MIMIC-IV, MIMIC-IV → INSPIRE;
  - confirmatory weight on MIMIC-IV and eICU (H6 in eICU only); cross-cohort streams secondary.
- **Phase 1** ([design_roll.md](design_roll.md)). Two measurements were added to `scripts/feasibility_roll.py`
  (`phase1`, aggregates only):
  - T2's analysis stays per block, year and hospital;
  - how often each proposed ML trigger fires on noise alone, in random windows of M patients.

  The freezes of T2 and `hypo` were unchanged at the start and the end.
- **What the measurements changed in the proposals:**
  - the CITL rule of the primary trigger is read as "the CI entirely outside ±0.10": the other reading fires on every
    check;
  - the confirmatory calibration endpoint is the excess calibration error: the share of windows with CITL outside
    ±0.10 is 40–60 % for a calibrated model at these window sizes;
  - in an eICU hospital the monitor fires on noise in about 1 check in 10 (M = 200). This is declared as the limit of
    monitoring a small hospital.
- **Declared before the freeze.** `hypo`'s own numbers predict that H1 fails against LightGBM at the primary N in every
  confirmatory stream. The proposal keeps LightGBM, the margin and the brief's N.
- **Fairness.** α fixed on the starting window; a go-live update for the realistic ML arm; H6 against the ML monitor at
  a matched nominal level (design §14).
- **Approved by the lead the same day**: design §16 as proposed:
  - windows, monitor, triggers, closed testing, comparator and margins;
  - H4 co-primary, on the excess calibration error;
  - α fixed on the starting window;
  - T2 in the MIMIC-IV stream only, descriptive;
  - the delay scenarios L = 0 and 3 months (no 6-month sensitivity).

## 2026-10-30 — New experiment `roll`: prequential evaluation with a rotating case base; Phase 0 (lead's brief)

- **The experiment and its thesis (lead).** The thesis put to the test: "CE-kNN is not less accurate than ML managed
  as a real hospital manages it, and it is more governable". More governable means:
  - deterministic inference, reproducible by hash;
  - an update without retraining: new patients enter the case base and the oldest leave it (FIFO);
  - stable performance along the stream;
  - diagnostics that flag real changes.

  The experiment is designed so that it can fail. Primary outcome: `hypo`'s, with the same definitions, inputs and
  cutoff. Secondary, descriptive only: T2. Metrics: `hypo`'s frozen metrics, the rotation that excluded the
  evaluated cohort; no new metric is learned.
- **What `roll` is.** A pre-registered retrospective evaluation on data already examined, not a prospective
  validation. The three cohorts were used with this outcome in `hypo`. What is new is the performance along a
  prequential stream (Gama et al., 2014).
- **Where it lives.** Branch `roll`, in its own worktree. T2's and `hypo`'s code and artefacts are only read. Their
  freezes are checked at the start and the end of every stage (`scripts/feasibility_roll.py`: unchanged).
- **Phase 0** ([feasibility_roll.md](feasibility_roll.md)):
  - MIMIC-IV has five `anchor_year_group` blocks (2008–2010 … 2020–2022);
  - 16 % of stays fall ≥ 3 years after their patient's anchor year;
  - the eMAR → prescriptions change reaches no `hypo` input;
  - eICU has 196 hospitals, 35 with ≥ 150 patients in each of 2014 and 2015;
  - `patientunitstayid` carries no calendar order;
  - INSPIRE has no calendar field;
  - the admission gate refuses nothing in these data;
  - a window needs about 100 patients (MIMIC-IV, eICU) or 150 (INSPIRE) for an AUROC within ±0.05.
- **The gate.** The decisions listed at the end of feasibility_roll.md are the lead's.

## 2026-10-30 — `hypo` post hoc, second round: ablation and the missing metrics (lead; plan fixed before running)

Asked for after an external read of the work (ablation and TRIPOD+AI metrics). It is exploratory, not
pre-registered, and runs in `scripts/posthoc_hypo_ablation.py`, outside `src/hypo`. The script only reads the frozen
artefacts and checks the freeze record at the start and at the end. The plan below is fixed before any result is
seen.

- **Metrics of the external stage's own predictions**, no refit:
  - **Methods:** as in S2. Those are PERISIM transported and local; last MAP, logistic regression and LightGBM,
    transported and local; prevalence, Euclidean kNN, PERISIM of Phase 2. Added are the transported comparators
    recalibrated on 500 local patients (draw 0).
  - **Statistics:** AUPRC, Brier score, log-loss, AUROC, calibration in the large and slope. CIs come from 500
    patient-level resamples, with paired differences against PERISIM transported.
  - **Calibration curves:** 10 equal-count bins of each method's prediction.
  - **Decision curves:** net benefit for thresholds 0.01–0.60, against treat-all and treat-none. CIs of the
    difference against PERISIM transported at 0.05, 0.10, …, 0.50.
  - **Level:** all of this is per landmark, on the same rows as S2.
- **Ablation, in the transported role** (each rotation's development pair, its learning sample and seed; each
  external site's five folds, case base = training folds, α by leave-one-out, as the external stage). 14 variants,
  geometry × weights:
  - **Geometries:**
    - z-score (standardised values, one-hot nominal, every block scaled to unit mean square): the standardised
      Euclidean kNN;
    - neutral (rank / one-hot);
    - HVDM (standardised + smoothed VDM, m = 5);
    - shapes (the frozen g_j).
  - **Weights:**
    - uniform;
    - SAGE: aggregated over the competitive models of the frozen Rashomon set, refitted with its seed and checked
      model by model against the frozen record;
    - mean |SHAP| of a LightGBM (T2's comparator 2 recipe);
    - permutation importance of the same LightGBM (log-loss, 3 repeats, held-out third of the learning sample);
    - diagonal NCA (hypo's settings).
  - **Grid:** z-score and HVDM × {uniform, NCA}; neutral and shapes × all five weights. Shapes × NCA is the frozen
    metric, and it must reproduce the external stage's predictions exactly.
  - **Contrasts** (log-loss and AUROC, paired, 500 resamples):
    - shapes − neutral at the same weights;
    - SAGE − uniform, and NCA − uniform, at the same geometry;
    - SHAP and permutation − SAGE;
    - NCA − SAGE;
    - HVDM − neutral;
    - every variant − z-score uniform (the standardised Euclidean kNN);
    - every variant − the frozen metric.

## 2026-10-30 — Name of the technique: consensus effect-space kNN, CE-kNN (lead)

- The technique is named **consensus effect-space kNN (CE-kNN)**. The project and the software keep the name
  PERISIM.
- What the name describes:
  - each variable is carried to its consensus shape function g_j (centred log-odds, the consensus of a Rashomon set
    of competitive models);
  - each variable is weighted by w_j;
  - the prediction is the vote of a local case base.
- Prior work reviewed, the closest first:
  - VDM / HVDM (Stanfill & Waltz 1986; Wilson & Martinez 1997): per-value class probabilities, one variable at a
    time;
  - SHAP's explanation space (Lundberg et al. 2020): supervised clustering, with one model;
  - KAAM (Almodóvar et al. 2025, preprint): nearest patients in one additive model's logit-contribution space;
  - metric learning (NCA, LMNN, and LSML for patient similarity);
  - the prognostic score (Hansen 2008).
- None of them combines the four pieces of CE-kNN: the multi-model consensus of the curves, the SAGE + NCA weights,
  a transported metric with a local case base, and re-deforming the g_j to measure drift. The name points at the
  consensus, which the closest work lacks.
- Some names were avoided because they are taken: CEM (Coarsened Exact Matching; Iacus, King & Porro 2012), MVDM
  (Modified VDM; Cost & Salzberg 1993), EDM (Elucidated Diffusion Models).
- PERISIM alone was set aside as the technique's name for two reasons: "peri-" ties it to perioperative care, and
  PeriopSim, a perioperative training simulator, already exists.

## 2026-10-30 — `hypo` reported, followed up post hoc, merged into `main` (lead)

- Phase 5 (2026-10-29): the report, [results/hypo/REPORT.md](../results/hypo/REPORT.md), with the aggregates of every
  phase in `results/hypo/`.
- After the report the lead asked for three post hoc follow-ups. They are exploratory, not pre-registered, and are
  reported in REPORT §13:
  - the H1 arm's local weights swapped into the transported metric;
  - a threshold for the MAP block as a whole;
  - S4's drift on folds 1–4.

  They run in `scripts/posthoc_hypo.py`, outside `src/hypo`. The script only reads the frozen artefacts and checks
  the freeze record at the start and at the end.
- The branch `hypotension` is merged into `main` as a separate experiment. Its files keep their paths, because the
  freeze record, the pre-registration and the tag refer to them. README.md and results/README.md describe it next
  to T2.

## 2026-10-29 — `hypo` frozen; external stage started (lead)

- The lead approved [prereg_hypo.md](prereg_hypo.md) as drafted, including the analyst's interpretations of H1–H5
  (entry below) and the α grid without 0.
- **D6**: a local registration ID, `LOCAL-PREREG-HYPO-2026-10-29`, with no OSF filing. The pushed tag is the
  timestamp.
- Freeze: tag `perisim-freeze-hypo-2026-10-29-120652` on commit `b0c9a02`, code `cf0d0b80c70a`, configuration
  `772940e4ed77`. The branch and the tag were pushed before the external stage started (12:07:25). Record:
  [freeze_hypo.md](freeze_hypo.md).

## 2026-10-29 — `hypo` Phase 3: pre-registration draft, frozen artefacts and the external stage's code (analyst; for the lead's approval before the freeze)

- Phase 2 finished (`reports/cv.md`, code `11a7ad77e519`; summary in docs/prereg_hypo.md §3); nothing tuned on it.
- **Draft pre-registration**: [prereg_hypo.md](prereg_hypo.md). H1–H5 of the design §5 made operational, the
  secondary analyses S1–S8, rules and limitations. Analyst's choices inside the design's wording, for the lead to
  confirm or change before the freeze:
  - H1 "PERISIM refitted locally" = a metric learned on the external site's training folds only, with the same
    case base;
  - H2 "inside ±0.10" read as T2 did, the 95 % CI inside the margin;
  - H3 against the last-MAP model refitted locally (the harder bar);
  - H4's control as a decision condition (Δ_hosp − Δ_ctrl < 0);
  - H5 with a Bonferroni over the three pre-declared γ;
  - 2,000 bootstrap resamples for H1–H5 and 500 for the rest;
  - the external stage on the Phase 2 folds (case base = the training folds), so every landmark of the external
    site is predicted once;
  - drift and shapes on fold 0 only.
- **The code of the external stage is written and frozen before it runs**, as T2's was:
  - `frozen.py`: `python -m hypo final` fits the three transported metrics on all patients of each development
    pair, and the transported comparators. `python -m hypo freeze` creates the tag and the hash record and refuses
    when external results exist or the tree is dirty. `check` is the gate.
  - `external.py`: the units per site and fold.
  - `drift.py`: S4–S6, T2's post hoc drift analyses re-implemented for a binary outcome and distinct-patient
    neighbours; candidates of the one-variable fits from the frozen metric, shared by all variables.
  - `external_report.py`: H1–H5 and the secondary tables.
  - `stats.py`: perisim's calibration fits with weights.
  - The external stage checks the freeze at its start and its end.
- Small changes to Phase 1 code, behaviour unchanged (tested):
  - the distinct-patient search is a function reusable by the drift fits;
  - the case base keeps its LOO neighbours, so H5 can re-choose α;
  - the shape loop returns the accepted models' curves, so S5 can read thresholds per model;
  - `hypo`'s code hash now normalises line endings as perisim's does, so the frozen hash is the same on any checkout
    (the hash itself changes from Phase 2's `11a7ad77e519`).
- A frozen-metric input that the external site cannot measure (INSPIRE's infusion timing) is the same value for all
  of that site's landmarks, so it adds nothing to any distance in its case base (tested). It needs no rule.
- **Dry run on real data** (`python -m hypo external --dry-run`, `external_dryrun/`, `reports/external_dryrun.md`):
  - eICU fold 0 is served by the rotation whose development pair includes eICU, so it is in-sample and touches no
    external result; it checks the mechanics only;
  - every unit ran at full size in 2 h 11 min;
  - it found one fault. In small case bases (H4's hospital contrasts, the S1 curve at n ≤ 500) the LOO chose α = 0,
    and unanimous neighbourhoods then predicted exactly 0 or 1. Four hospital test rows reached the log-loss clip
    (34.5), and up to 1,909 curve predictions were 0 or 1;
  - fix: α = 0 is removed from `metric.alpha_grid` (now 0.1–10). Phase 2 never chose α = 0, so its results are
    unchanged. The grid is used only by case bases, so the frozen artefacts are unchanged too. The affected
    dry-run units (hospitals, curve, hospital drift) were rerun; no prediction is 0 or 1 any more, and the largest
    log-loss of a row is 7.1;
  - the fix changes two in-sample dry-run figures, recorded in prereg §3. H4 goes from "does not hold" to "holds":
    the own-hospital arm had carried the clipped rows. S1's "within 0.01" goes from 500 to 250 patients. The old
    outputs are kept in `external_dryrun_alpha0/`.
- Order:
  1. `final` (done; code `cf0d0b80c70a`);
  2. dry run (done);
  3. the lead approves prereg_hypo.md and decides D6;
  4. commit the code;
  5. `freeze` and push the tag;
  6. `external` (about 8 h).

## 2026-10-28 — `hypo` Phase 2: internal validation by rotation (analyst's design; Phase 1 confirmed by the lead)

- The lead confirmed Phase 1's defaults (entry below). `src/hypo/cv.py`, `compare.py`, `report.py`; `cv` in
  `config/hypo.yaml`; `python -m hypo cv` (resumable: finished units are kept) and `python -m hypo cv-report`.
- **Folds fixed per site**: 5 patient-level folds, stratified by the outcome, the same in every rotation, so a
  site's test fold is predicted by the metric learned with each of the other two sites (one line per partner).
- **PERISIM per rotation and fold**: metric learned on the learning sample of both development sites' training
  patients; each site queried with its own case base (its training landmarks); α by LOO in that case base.
- **Comparators are local** (one site's training landmarks): prevalence, last MAP + slope (logistic), logistic on
  every input (C tuned), T2's LightGBM (grid × early stopping, temperature), Euclidean kNN (neutral axes, uniform
  weights, the same case base and vote). Settings are T2's (`model_space.yaml` `baselines`), tuned in 3 inner
  patient-level folds. They do not depend on the rotation, so each runs once per site and fold, and they are the
  external stage's "refitted locally" arm too. Transported comparators (trained on the development sites, applied
  to the external one) belong to the external stage.
- Report: AUROC, log-loss, Brier, calibration in the large and slope per site and method, and PERISIM − comparator
  paired in log-loss and AUROC, with 95 % CIs from 500 patient-level bootstrap resamples (the same for every method
  of a site; weighted AUROC). Descriptive: Phase 2 has no decision rule; the hypotheses H1–H5 are for the external
  stage, after the freeze.
- Chain checked on real data in quick mode (8 models, 10 epochs; rotation external-eICU, fold 0; `cv_quick/`,
  kept apart from the run): PERISIM unit 137 s; local comparators per site and fold 35 s (INSPIRE) to 90 s
  (MIMIC-IV). The full run is estimated at 2.5–3 h.

## 2026-10-28 — `hypo` Phase 1: the metric and the case base (analyst's defaults, confirmed by the lead)

- `src/hypo/data.py` (landmarks → perisim's encoder input; learning sample), `metric.py` (shapes and weights),
  `casebase.py` (neighbours and probabilities); settings in `metric` of `config/hypo.yaml`. perisim's encoder, model
  space, surrogates, curve reading, consensus, NCA and smoothed vote are used unchanged.
- **No SAGE stage** (T2: λ ≈ 0). The weights are a free diagonal NCA; the shape loop stops at a fixed 40 accepted
  models (T2's loops accepted 31–57), with the stability trace Δ of the distance (uniform weights) recorded only.
- **Several development sites.** The site is a nuisance column of the ensemble's models, not part of the metric:
  each g_j is a within-site effect, not a base-rate difference between sites. The NCA searches each anchor's
  candidates within its own site, as the metric is used (each site queries its own case base). Each development
  site gives the learning sample the same number of landmarks (≤ 30,000; ≤ 3 random landmarks per patient).
- **Inputs constant within every development site leave the metric**: the missing indicator of a source gap is the
  site itself (INSPIRE's "vasopressor running now" and vasopressor hours when INSPIRE is a development site).
- **Neighbours are distinct patients.** A patient's consecutive landmarks are near-copies; each other patient enters
  the k = 50 once, at its most similar moment, and the query's own patient never enters. α by patient-level LOO on
  20,000 of the case base's own landmarks; out of support as T2 (99th percentile of the k-th distance).
- The secondary bands are the same neighbours' smoothed vote over three classes; carried outcomes (vasopressor
  start, ICU discharge in 24 h, MAP at 6 h) are their shares / medians.
- Smoke run (`python -m hypo smoke --external eicu`: development INSPIRE + MIMIC-IV, 80 / 20 patient split, mechanics
  and cost only): 411 s in all (shapes 174 s, NCA 187 s); 40 accepted models in 46 attempts, Δ ≈ 0.002–0.003 from
  the 20th; 18 of 27 variables shaped; no neighbourhood short of 50 distinct patients; held-out mean prediction
  equal to each site's prevalence (13 / 31 %), AUROC 0.78 at both. A quick version (8 models, 10 epochs) gave the
  same numbers. Comparators are Phase 2.

## 2026-10-28 — New experiment `hypo`: hypotension in the next 6 h in the ICU after surgery (lead)

- T2 was the deliberately hard case (little information at ICU discharge), run to learn whether the shape functions
  carry real information, where the method breaks (rare classes, small case bases) and what it is good for (transfer
  with a local case base, drift diagnosis). The next experiment applies that to a case with dense data and a frequent
  outcome, centred again on transfer and explainability. Candidates measured on the original tables; rare outcomes
  (vasopressor start, 2–9 % per landmark) discarded; the lead chose hypotension in the next 6 h.
- Design: `docs/design_hypotension.md`; feasibility: `docs/feasibility_hypotension.md`. **Approved defaults:** D1
  leave-one-site-out; D2 secondary outcome in absolute MAP bands (< 65 / 65–75 / ≥ 75); D3 arterial line as a
  stratum, not an input; D4 landmarks every 2 h; D5 MAP < 65 mmHg on the hourly median, 6-h horizon. Open: D6 OSF
  registration, D7 verify INSPIRE's ICU vasopressor infusions, D8 T1 / T3 of the original plan.
- Code: a separate package `src/hypo/` (config `config/hypo.yaml`) on the branch `hypotension`. `perisim` is used as a
  library and not changed, so T2's frozen code and configuration hashes stay valid on this branch; `hypo` has its own
  code hash (its files plus perisim's hash). Population from T2's `stays_candidates` (postoperative adult ICU stays,
  first of the hospitalization, deaths kept); vitals, labs and supports read again from the original tables, because
  T2's common format keeps only T2's index stays and, in MIMIC-IV / eICU, the vitals of a window before the ICU
  discharge.
- **Phase 0 (data), approved by the lead.** First landmark build: INSPIRE 148,293 landmarks (13.2 % hypotension),
  MIMIC-IV 387,159 (30.7 %), eICU 466,385 (19.3 %). Fixed before the model:
  - eICU invasive ventilation was 1.7 % of the landmarks (respiratoryCare intervals alone): now also the ventilator
    settings charted in respiratoryCharting (set rate / volume, plateau / peak pressure, mode, …; PEEP / CPAP / EPAP
    left out, non-invasive ventilation charts them) and `treatment` (as T2); eICU temperature also from
    nurseCharting ("Temperature (C)"), vitalPeriodic having it for 16 %.
  - Charted points of a support merge into runs while the gaps are < 6 h (T2's `support_gap_hours`); new inputs
    "vasopressor / sedation in the last 24 h", comparable across sites.
  - **D7 resolved**: INSPIRE's medication table does hold the ICU (records for 99.8 % of the stays, median 19 per
    stay); vasopressors in the ICU 8.5 % of the stays (norepinephrine 4.0 %, dopamine 4.1 %), sedation 16.3 % — a
    practice difference, not missing data. But its records are administrations, not infusion intervals, so
    "running now" (vaso_on, sedation_on) and vasopressor hours are a declared `source` missing value in INSPIRE.
  - Left out of the metric: BMI (absent in MIMIC-IV), surgery duration and time since the surgery (no surgery times
    in MIMIC-IV / eICU). Unmapped surgical group → its own category "unknown" (MIMIC-IV 25 %).
  - A completeness audit (`python -m hypo audit`) is the gate before the model: outcome prevalence ≥ 10 % per
    cohort, every input ≥ 50 % coverage unless declared a gap.
  - Ventilation checked at the first landmark: INSPIRE stayed at 61–64 % up to 24 h in every surgical group because
    its `vent` item is 0 / 1 and `hypo` counted the 0 records (39 % of them) as ventilation; now as T2's extraction
    (`value != 0`, which T2 already applied). MIMIC-IV rose from 37 % at 2 h to 52 % at 6 h (the ventilation
    procedure is charted hours after the arrival): the ventilator settings charted hourly in chartevents are added
    (mode, set volume / rate, peak / plateau / mean airway pressure; PEEP left out), as in eICU.
  - eICU lactate (23 % of the landmarks, drawn selectively) is a declared `measurement` gap: kept with its missing
    value as its own value (design §4).
  - After the fixes the audit passes with no flags. Ventilation at L = 2 / 24 h: INSPIRE 37 / 11 %, MIMIC-IV
    45 / 26 %, eICU 40 / 12 %. MIMIC-IV's rise to 60 % at 6 h is composition, not charting: in the same stays it falls
    (45 → 43 %); the stays without a MAP in the first ICU hour (8,733, 82 % ventilated, mostly cardiac) enter the
    landmarks only from L = 4–6. The earliest landmarks under-represent the busiest arrivals; a property of the
    design (a MAP in the last hour is required), reported, not corrected.

## 2026-10-28 — Post-hoc case-base analyses: patients needed, and hospital-specific relations in eICU (lead)

- Two questions of the lead after the drift analysis. (1) With the frozen INSPIRE metric (the "scaffold"), how many
  local patients, added at random, does it need to recover what the whole local pool gives? The pre-registered
  accumulation curves (analysis C) jump from 2,000 patients to the whole pool (~18,000) and are still far from it at
  2,000. (2) The lead's hypothesis: the case base carries physiology and local relations, not just a recalibration,
  and eICU transports worse because it is several hospitals; are the relations hospital-specific?
- **Exploratory, post hoc, not pre-registered, no decision rule.** Frozen metric, transport stage's split, labels, test
  sets, k and α.
- (1) Dense accumulation: n ∈ {50, …, 16,000} patients drawn at random from the pool, 20 draws per n; INSPIRE (its
  own pool) as the reference. Reported: log-loss, macro AUROC, distance to the whole pool, share of its gain over the
  local prevalence, change per doubling of n, the n that comes within 0.02 / 0.01 of the whole pool and the n that
  matches the transported, recalibrated comparators of the post-hoc comparison. A first scratch run reproduced
  analysis C (n = 500: 1.116 / 1.116; 2,000: 1.086 / 1.086, MIMIC-IV).
- (2) eICU hospitals with ≥ 300 labelled index stays (analysis F's set and labels H1, H2, …), their test stays of the
  transport test set. Same case-base size: A = the hospital's own pool; B = as many patients drawn from the other
  eICU hospitals (10 draws); B' = B with its probabilities moved to the hospital's class shares (label-shift
  correction), so that A − B' measures relations beyond base rates. Control: the same with random groups of patients
  of the same sizes. Also the enrichment of own-hospital neighbours in the whole eICU pool and the k-th neighbour
  distance. Patient bootstrap of the paired per-stay differences.
- `scripts/posthoc_case_base.py` (`run_local.ps1 case-base`), outside `src/perisim`; frozen artefacts' SHA-256 checked
  at the start and the end. Outputs: `results/posthoc_case_base/`, `reports/posthoc_case_base.html`,
  `artifacts/<variant>/posthoc_case_base.json`; hospitals only by their size rank, never by eICU identifier.

## 2026-10-28 — Post-hoc drift follow-up: only the missing values re-deformed (lead)

- The drift analysis suggested that much of eICU's drift is in what a missing value means (blood gases, GCS). The
  lead asked for the direct test: with the frozen INSPIRE metric and each site's case base, free ONLY the points of
  the missing-data values, all at once, and see how much of the drift (all points re-deformed) and of the gap to the
  local refit A(iii) that recovers.
- **Exploratory, post hoc, not pre-registered, no decision rule.** Same lookup, split, labels, k, smoothed kNN,
  ridge and evaluation as the drift analysis; INSPIRE again the (in-sample) control.
- Freed: the "missing" value of every continuous variable and missingness indicator (it starts at the median's
  point) and the "unobserved" value of every binary (at INSPIRE's prevalence); nothing else moves. A 10⁻³ jitter
  starts each freed point off the observed point it coincides with (there the gradient of those pairs is zero).
- Read in three groups, each evaluated with only its own adapted points (no refit): **measurement** (a value not
  recorded), **structural** (hours since a support that was never used: a clinical state, not a gap) and **source**
  (a binary whose source was not recorded, e.g. the medication tables). Each move is also fitted on the drift
  analysis' two patient halves (agreement in direction).
- `scripts/posthoc_drift_missing.py` (`run_local.ps1 drift-missing`), outside `src/perisim`; frozen artefacts'
  SHA-256 checked at the start and the end. Outputs: `results/posthoc_drift_missing/`,
  `reports/posthoc_drift_missing.html`, `artifacts/<variant>/posthoc_drift_missing.json`; fits cached in
  `predictions/<variant>/posthoc/` (never exported); moves of values with < 10 stays at a site are not exported.

## 2026-10-28 — Post-hoc drift analysis: ICE and re-deformation of the frozen metric (lead)

- The lead stated the purpose of the transport work: PERISIM is meant as a tool to study drift, not to beat the ML
  comparators (its loss against them was expected). With the frozen INSPIRE metric and each site's own case base
  (analysis A(i)), which variables would have to change their shape function g_j, or their importance, to fit the
  site, and how?
- **Exploratory, post hoc, not pre-registered, no decision rule.** Same split, labels and k as the transport stage.
- The metric is read as a lookup: each variable's values (INSPIRE's bins, categories, 0 / 1) are points
  √w_j·g_j/s_j; continuous variables get an extra "missing" value at the median bin's point (INSPIRE imputes the
  median), so informative missingness can appear. The lookup reproduces the frozen embedding exactly (checked per
  site; MIMIC-IV test log-loss 1.0494, as in the post-hoc comparison).
- ICE: 1000 test stays moved along every value through the frozen geometry, hard kNN with the site's case base
  (blind where g_j is flat). Re-deformation: one variable's points freed for the case base and the queries, fitted by
  L-BFGS on a smoothed version of the same kNN (votes σ((r_a − d²)/τ_a) around each anchor's k-th distance, frozen α),
  patient-level leave-one-out over the case base, ridge 20 towards INSPIRE's points; evaluated with the real kNN on
  the test set (paired patient bootstrap). Per variable: Δ log-loss, effective weight at the site with INSPIRE's and
  with the adapted points, shape change (weighted Procrustes residual), split-half reliability. Then all variables
  at once, against the local refit A(iii).
- INSPIRE is run as the control with the same split rule and fits (in-sample: the metric was read from all of it).
- Outside `src/perisim` (`scripts/posthoc_drift.py`, `run_local.ps1 drift`); frozen artefacts' SHA-256 checked at
  the start and the end. Outputs: `results/posthoc_drift/`, `reports/posthoc_drift.html`,
  `artifacts/<variant>/posthoc_drift.json`; fitted points cached in `predictions/<variant>/posthoc/` (resumable,
  never exported); adapted points of values with < 10 stays at a site are not exported.

## 2026-10-28 — Post-hoc transport comparison with the parametric comparators (lead)

- After the T2 external validation and transport analyses, the lead asked how much PERISIM loses when transported
  compared with the ML comparators of the CV, with and without recalibration, and where the sites differ (the shape
  of g_j between sites, against the distribution of each variable). The external stage had transported only the
  logistic regression, so LightGBM had never been evaluated outside INSPIRE.
- **Exploratory, post hoc: added after the external and transport results had been seen; not part of the
  pre-registration and without a decision rule** (stated in the report). Nothing confirmatory changes.
- Same split (fixed test set, nested tuning subsets of analysis D, local pool), labels (cross-fitted tertiles) and k
  as the transport stage. Each method — PERISIM, LightGBM (temperature), logistic regression, Euclidean kNN,
  prevalence — trained on INSPIRE and applied as it is; recalibrated with 100 / 250 / 500 local patients (multinomial
  p'_c ∝ exp(a_c + b·log p_c), intercepts only or intercepts + slope, penalised ML with a N(0, 2²) prior on a and
  log b); refitted on the local pool (the ceiling; PERISIM = A(iii) of the transport stage). PERISIM also with
  INSPIRE's case base plus the same n local patients (γ(n)). LightGBM and logistic regression nested as in the CV.
  Paired patient-bootstrap CIs (2000).
- Shapes: the per-site consensus of analysis E on INSPIRE's grid, divided by the site scale, drawn per variable and
  class above the share of each site's stays in INSPIRE's bins (bins with < 10 stays suppressed; the outer bin edges,
  single patients' extreme values, are not shown). Per variable: PSI over INSPIRE's bins, % outside INSPIRE's range
  and missingness (a different population) next to the shape deviation (a different relationship) and w_j.
- **Outside `src/perisim`** (`scripts/posthoc_transport.py`, `run_local.ps1 posthoc`): the frozen code hash covers
  all of `src/perisim`. It only reads the frozen artefacts and checks their SHA-256 at the start and the end. Outputs:
  `results/posthoc/`, `reports/posthoc_transport.html`, `artifacts/<variant>/posthoc_transport.json` (aggregates);
  per-patient predictions cached in `predictions/<variant>/posthoc/` (never exported).

## 2026-10-27 — T2 frozen; aggregate results versioned in the repository (lead)

- T2 frozen on the lead's workstation: tag `perisim-freeze-t2-2026-10-27-141040` (commit `c4cc8b7`, pushed to
  GitHub before the external stage ran), code hash `77b25fe8f629`, scientific configuration hash `1f6d4f5b89d4`. The
  OSF registration was not filed yet: the freeze records the local ID `LOCAL-PREREG-T2-20261027` (not an OSF ID);
  the OSF ID will be added next to it. The public commit and tag are the verifiable timestamp.
- The "no outputs in the repository" rule came from the KHDP pod. On the lead's own computer, aggregate results are
  versioned in `results/<experiment>/` (`scripts/publish_results.py`: the export package minus the zip and the bulky
  per-fold CV artifacts, scanned again for identifiers). Never in the repository: `data/` and the rest of `work/`
  (extractions, features, per-patient predictions), which are restricted patient-level data. `.gitignore` and
  `scripts/check_public_safety.py` allow CSV tables under `results/` only.

## 2026-10-25 — Secondary k sweep of the internal CV (lead)

- The lead asked how much the choice of k contributes. The CV evaluates k ∈ {10, 20, 50, 100, 200}; on the
  provisional folds the macro and death AUROC of every kNN method were still rising at k = 200, so the grid does not
  show where the curves turn.
- **Secondary and descriptive, added after the provisional results of the first CV folds had been seen** (stated in
  the report). The primary k (D11, k = 50) and the confirmatory comparisons are unchanged.
- Each outer fold's metric, saved by the CV, is queried again over k = 1…2000 (at most 20 % of the smallest training
  case base), with α refitted per k by the same patient-level LOO and nothing else refitted. Reported: the curve of
  every kNN method (log-loss, Brier, RPS, AUROC, calibration slope), the paired log-loss difference of each k against
  k = 50, every comparator against PERISIM at the same k, PERISIM against LightGBM and logistic regression, and k
  chosen inside the training fold (argmin of the LOO log-loss that fits α) against k = 50. The best k pooled on the
  test folds is shown as optimistic.
- Comparators: re-queried with the weights the CV stored (w/Σw, 6 decimals, floor 1e-9); the HVDM encoder refitted as
  in `fit_perisim`; RF-GAP refitted with the fold seed. A reproduction check compares the sweep with the CV's own
  predictions on the CV grid.
- **Not a pipeline stage:** a new stage changes `pipeline.py`, which is in the key of `harmonize` and `pilot`; through
  the upstream keys that would re-run the pilot, the CV and the sensitivity variants (about 9 h here) to reproduce the
  same results. Command `ksweep` (`run_local.ps1 ksweep`, `run_pod.sh ksweep`), keyed by the CV it was computed from;
  `experiment.k_sweep` is in no stage's key (tested). Outputs: `results/ksweep/`, `reports/ksweep.html`,
  `artifacts/<variant>/ksweep_<t>.json` (aggregates, exported); patient-level predictions in
  `predictions/<variant>/ksweep/` (never exported).

## 2026-10-25 — Local runs on the lead's workstation; deviation from the pilot's cost rule (lead)

- The experiments run on the lead's workstation (Intel Core Ultra 9 275HX, 24 cores, 64 GB RAM, NVMe) with the lead's
  own PhysioNet copies, through `run_local.ps1` and the configuration profile `config/profiles/local.yaml` (only what
  differs from the KHDP configuration; recorded by the freeze). Patient-level data stay on the workstation; only logs
  and aggregates are read back.
- **Deviation from the declared pilot rule (decided on cost only, before any cross-validation result):** the rule
  "a family costing > 3 × the median of the others is excluded" (§9.1, sized for the pod's budget) excluded random
  forest at 3.1 × (32 s vs a median of 10 s per accepted model). With the whole CV estimated at about 1 h here, the
  local profile disables the cost exclusion (`pilot.family_cost_ratio_exclude: null`) and the pilot is re-run, so the
  consensus keeps every family of the model space. The pilot's fold-0 performance was not used for this decision.

## 2026-10-25 — Transport analyses approved and implemented (lead), SPECS §6.2b

- The lead approved the map, the pre-registration amendment and the eight decisions of
  [transport_review.md](transport_review.md) §8 as proposed, margins included (H1 ±0.10, H2 0.90, H3 0.02). The stage
  `transport` replaces `transfer` (its variants are contained in A and B).
- Implementation choices (declared in the amendment): the site scale of E is the **median** over variables of the
  per-variable WLS slope (a global WLS fit was biased by a single non-transporting variable: unit test); E runs the
  reduced consensus in all three sites, INSPIRE included, so the three are estimated the same way; A, C and D use the
  primary labelling (cross-fitted tertiles), B both; in A the frozen metric keeps INSPIRE's α, the locally refitted
  metrics fit their α on the local pool; the % out of support of C is computed in the first 5 repetitions; H1's
  "beyond the margin" means a CI entirely outside ±0.10; F uses the same γ values with random order only.
- The eICU hospital is read from the raw `patient` table inside the module (not added to the extraction, which would
  re-key the extraction and the development stages).
- Also fixed: `cohorts.allow_missing_cohorts` had been left under the `transfer` block of experiment.yaml when that
  block was inserted (read only when a cohort is missing, so it had not failed).

## 2026-10-24 — Model-transfer validations (lead), SPECS §6.2b

- New stage `transfer`, after `external` (so on the frozen pipeline). In MIMIC-IV and eICU: one patient-level split,
  80 % local cases / 20 % held-out queries, the same for every variant. INSPIRE weights and shapes with the local
  cases (T_local), with INSPIRE ∪ local cases (T_pooled), against PERISIM fitted on the local cases (L_perisim),
  an unsupervised Euclidean kNN and the local prevalence, and the full transport on the same queries (A_transport).
- Multi-source development: PERISIM refitted on INSPIRE ∪ eICU, evaluated on MIMIC-IV only, T2 only (eICU has no full
  T1): with its own case base, the local cases, or both.
- Implementation choices (declared): the target cohort's cut-points come from the local cases only; each development
  cohort keeps its own cut-points; α of every case base is fitted on that case base; the local and multi-source fits
  use the same nested procedure and stopping rule as the final INSPIRE model; the final INSPIRE model is saved by the
  external stage (inside the pod, `predictions/`) and reused, not refitted.

## 2026-10-24 — Harmonization from the real INSPIRE and eICU schemas (first discover in the pod)

- INSPIRE `medications` ships one ATC column (`atc_code`); `atc_code2` / `atc_code3` (data dictionary) are optional and
  used when present. It also has `route` (po / iv / ex): `ex` (external, e.g. topical epinephrine) feeds neither the
  ATC inputs nor the vasopressor support — the INSPIRE counterpart of excluding local products in MIMIC-IV / eICU
  (AUDIT D4).
- INSPIRE item names fixed from the inventory: `lacate` (sic), `paco2`, `pao2`, `nibp_mbp`; GCS is charted by
  component (`gcs_e`, `gcs_m`, `gcs_v`) and the total is E + M + V charted at the same time, the same rule as in
  MIMIC-IV (a time point without all three gives no total). Devices `vent`, `crrt`, `ecmo`, `iabp` confirmed.
- eICU: every configured table and column exists; lab names, FiO2 labels, vasopressor names and discharge locations
  confirmed against the inventory.
- Completeness audit of the first real run (OPEN_QUESTIONS 36): red blood cell transfusion is removed from T1 in
  every cohort — INSPIRE does not record it (prevalence < 0.5 %), and development and external validation must share
  one outcome definition. Mild = AKI 1–2 or a new antibiotic course; transfusion stays a carried variable. The audit
  no longer raises issues for components listed in `t1.exclude_components`.
- Lead's answers to OPEN_QUESTIONS 32–35: department OT = ophthalmology (group of EY); CTS split by the ICD-10-PCS
  code of the operation (02 / 5A cardiac, 03–06 vascular, other codes thoracic, no code cardiac); INSPIRE MAP =
  (SBP + 2·DBP) / 3 where SBP and DBP are charted without a MAP; transfusion left to the completeness audit.

## 2026-10-22 — Scientific decisions after the full audit (lead)

Decided by the lead on the options of `docs/DECISIONS_PENDING.md` (IDs from `docs/AUDIT.md`).
Status: **decided and implemented** (2026-10-22; SPECS v0.4, tests in `tests/test_decisions.py`).
Implementation notes: the D6 coverage rule also applies to eICU medication orders (inputs only); the HCUP
procedure-class files are optional (until provided, the MIMIC-IV fallback uses a declared approximation);
the INSPIRE readmission proxy ignores ward dialysis of chronic-RRT patients; binary event inputs are now explicit
0 where the source is observed, so availability = observability (also closes AUDIT C10).
Two readings to confirm: (1) D5 "not observable" is implemented as **not applicable** — AKI and new RRT simply do
not count for chronic-RRT patients and T1 is decided by the other components; (2) D6 makes T1 (and T3)
**undefined** for stays without medication coverage, whatever their other outcomes (excluding only the "none"
ones would be an outcome-dependent selection); severe / not severe and T2 stay defined for them
(`targets.yaml: unobserved_mild_policy: reduced_definition` would instead let the other components decide).

**Cohort and targets**
- **D1 / D8 — surgery in MIMIC-IV:** timestamped operating-room evidence — a PACU/OR transfer in
  `hosp.transfers` within 24 h before the ICU admission; reoperation = a later PACU/OR transfer after t0
  (minute precision). HCUP "major therapeutic" codes only as a fallback without a PACU transfer. A surgical
  service alone no longer qualifies.
- **C5 — t0 must be a discharge to the ward:** exclude index stays followed within 6 h by OR / PACU / IR /
  another ICU (MIMIC `transfers`; INSPIRE: next operation start within 6 h of `icuout_time`); reported in the flow.
- **D10 — ICU readmission in INSPIRE (SPECS D1):** proxy = a new invasive ventilation / CRRT / ECMO / IABP run
  after t0 in `ward_vitals`, plus ICU entry after another surgery; its prevalence is validated in the audit
  before use.
- **D5 — chronic dialysis / advanced CKD:** AKI and new RRT are not observable for patients with RRT before the
  index surgery; KDIGO ≥ 4.0 mg/dL only with an acute rise and baseline < 4.0; no baseline measured during RRT.
- **D7 — KDIGO timing:** rolling windows (≥ 1.5× the minimum of the previous 7 days or ≥ 0.3 mg/dL above the
  minimum of the previous 48 h), always also relative to the SPECS baseline; stage by the maximum ratio.
- **D6 — MIMIC-IV eMAR coverage:** hospitalisations without any eMAR record are unobservable for the mild
  medication components and medication inputs; eMAR coverage by `anchor_year_group` reported in the audit.
- **D12 — limitation of life-sustaining treatment:** implemented in MIMIC-IV ("Code Status") and eICU
  (`carePlanGeneral`) with records timestamped ≤ cutoff.
- **D13 — blood gases in MIMIC-IV:** arterial samples only (specimen type "ART.").
- **D14 — INSPIRE clipping:** deferred until the first completeness audit on the real data.

**Method**
- **B2:** reparameterise the regularised NCA as log w = log c + log(imp + ε₀) + u with penalty λ‖u‖² and c free;
  λ grid re-spaced.
- **B3:** the stopping criterion Δ is measured on the metric implied by the prior (w = c·imp, within-group split
  used only for Δ), not on the NCA-converted weights. **This amends SPECS §5.3.2** ("recompute w^(n) with the
  conversion of §5.4"): the SPECS text must be updated accordingly when implemented.
- **B4:** the SAGE prior acts on group totals (λ Σ_g (log Σ_{j∈g} w_j − log c − log imp_g)²); the NCA splits within
  groups.
- **C3:** SAGE is recomputed inside each inner fold for the selection of λ (≈ 3× the loop cost; the pilot will
  quantify it before the full CV).
- **B6 / B7:** explicit "other/missing" dimension per ICD-10-PCS level (cap revised at levels 4–5); unseen VDM
  categories → P(y) (SPECS §5.2).
- **E13:** the LightGBM comparator is tuned in the inner CV (small num_leaves × learning-rate grid, early
  stopping on the inner folds).
- **C6:** α remains selected by patient-level LOO on the training fold (declared deviation from §5.5; revisit
  only if the audit shows many small nominal categories).

**External validation and reporting**
- **C4:** external T2/T3 tertiles by patient-level 2-fold cross-fitting within the external cohort, plus
  absolute categories (≤ 3 / 4–7 / > 7 d) as a co-primary external analysis.
- **E14:** the §5.6 per-patient JSON is generated inside the pod for every test-fold and external query
  (`predictions/`, never exported), with a data-free schema example in the report.
- **E10 / E11:** per-fold metrics and cut-points next to the pooled ones; SAGE dispersion reported per variable.

## 2026-10-24 — Methodology v0.5 (discussion with the lead)

Status: **decided and implemented** (SPECS v0.5, tests in `tests/test_v05.py`). Options and reasoning: section 4 of
`docs/DECISIONS_PENDING.md`.

- **Metric geometry:** every continuous / ordinal / nominal / PCS variable lives on its consensus shape function
  g_j (centred log-odds, one component per class), read from the competitive models of the SAGE loop (ALE for
  ordered variables, local effect against the reference category for nominal / PCS), fitted by least squares with a
  per-model scale, shrunk per value (DerSimonian–Laird), and replaced by a neutral axis (rank / one-hot) when it is
  not distinguishable from flat. Weights w_j (SAGE prior + NCA) unchanged. The v0.4 HVDM geometry is a comparator.
- **Flatness rule (implementation note):** "τ² = 0" alone keeps about half of the pure-noise curves (DerSimonian–
  Laird τ² > 0 whenever Q exceeds its degrees of freedom), so a curve is kept only if a homogeneity F test (between-
  model variance pooled over the values) rejects "flat" at 5 % with Bonferroni over the classes; on simulated noise
  this neutralises ≈ 93 % of the curves and keeps every weak real shape tested. The shrinkage is as decided.
- **Stopping:** Δ on the prior-implied metric over the CURRENT shapes (B3 extended); the loop is checkpointed after
  each accepted model; early accepted models outside (1+ε)·best leave the consensus (AUDIT B8).
- **Model space:** logistic regression with sampled penalty (L2 / LASSO / elastic-net), penalised spline logistic
  regression (replaces the MLP), random forest, LightGBM, XGBoost. The pilot excludes any family costing > 3× the
  median of the others.
- **T2:** short / medium / long (survivors' tertiles) / in-hospital death, ordinal with death worst.
- **Experiments:** one per target (`PERISIM_EXPERIMENT=t2|t1|t3`, default t2), each with its own freeze (git tag
  `perisim-freeze-<target>-…`) and work folder; extraction shared.
- **Comparators:** trivial prevalence and prevalence by surgical group (internal), INSPIRE prevalence and the target
  cohort's LOO prevalence (external), Euclidean kNN on neutral axes, HVDM geometry with the same prior and λ, plus
  the v0.4 list.
- **Sensitivity:** outcome-definition variants by relabelling the main metric's case base; input variants refitted
  for the experiment's target only.
- **Cost:** curve subsample 1,000 per model; pilot budget gate with the pre-declared lever order (subsample →
  shapes only for 95 % of the importance → C3 prior from the outer fold).
- **Not decided yet:** the T3 metric (learned anew, as implemented now, or combined from T1 and T2).
