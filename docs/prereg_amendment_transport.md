# Pre-registration amendment — transportability and incremental case base (draft)

Status: **approved by the lead on 2026-10-25, margins included**; written before any analysis has been run on
MIMIC-IV or eICU and before the freeze of experiment T2. It is registered with the freeze (OSF) and the analyses are
confirmatory for T2. Design details: [transport_review.md](transport_review.md) §6; implementation:
`src/perisim/transport/`.

## Rationale

The PERISIM distance depends only on differences of the centred consensus shapes g_j, weighted by w_j, and the kNN
only on the order of the neighbours. A shift of baseline risk, or a uniform shrinkage of every effect, between sites
therefore leaves the metric unchanged. We hypothesise that the **geometry transports** and that the **calibration is
supplied by the local case base**: with a local case base the predictions should be calibrated in the large, and a
degraded metric should show as compressed predictions (calibration slope > 1, lower discrimination) rather than as a
global calibration error. We also evaluate a deployment-like use: a hospital starts from an existing case base and
adds its own patients without retraining.

## Frozen artefacts

g_j, w_j, the encoder, α and λ of the final INSPIRE model (`final_model.pkl`), the freeze record and the frozen
configuration. Their SHA-256 hashes are recorded at the start of the transport stage and checked at the end; the
analyses never refit or modify them (the local refits of A(ii) / A(iii) and E are separate objects).

## Data, splits and labels

Per external cohort (MIMIC-IV, eICU; eICU also per hospital with ≥ 300 index stays), patient-level and
seeded: a fixed test set (20 % of the patients), a tuning set (≤ 500 patients) and the local pool for case bases,
disjoint by patient. T2 labels: tertiles of the target cohort cross-fitted between two patient halves (each half
labelled with the other's cut-points), the same labels for every case-base size and composition; absolute categories
(≤ 3 / 4–7 / > 7 days) as the co-primary labelling. Patient-level bootstrap (2,000 resamples) for every CI.

**Primary contrast set.** "Local mode" = frozen INSPIRE metric with the whole local pool as case base (variant
T_local); "mode A" = frozen INSPIRE metric with the INSPIRE case base; both evaluated on the same test set, at the
primary k (50) with INSPIRE's α.

## Hypotheses (confirmatory for T2, each external cohort separately)

| | Hypothesis | Estimand and decision rule | Margin |
|---|---|---|---|
| **H1** | In local mode the calibration in the large is ≈ 0; in mode A it is not | calibration in the large (intercept with slope fixed at 1) of every cumulative threshold P(Y ≥ c), c ∈ {medium, long, death}: H1 holds if the 95 % CI is inside ±δ₁ for every threshold in local mode, and outside ±δ₁ for at least one threshold in mode A | δ₁ = 0.10 log-odds |
| **H2** | In local mode the calibration slope is ≥ 1 (compression, not over-confidence) | slope of every cumulative threshold: the lower 95 % CI bound ≥ 1 − δ₂ | δ₂ = 0.10 |
| **H3** | With the same case base, the INSPIRE metric discriminates almost as well as a locally learned one | paired difference of macro one-vs-rest AUROC, A(i) frozen INSPIRE metric − A(iii) local PERISIM, same local case base and test: lower 95 % CI bound > −δ₃ (non-inferiority) | δ₃ = 0.02 |
| **H4** | The out-of-support rate is higher in mode A than in local mode | paired difference of the % of test stays whose k-th neighbour distance exceeds the case base's own 99th percentile: 95 % CI > 0 | — |
| **H5** | The accumulation curve has a crossing point, before which an external start helps | (a) with random order, the mean log-loss of the empty start falls to or below that of the INSPIRE start (γ = 1) at some n_local of the grid; (b) at n_local ∈ {50, 100}, the INSPIRE start with the declared schedule γ(n) = 500 / (500 + n_local) has a lower log-loss than the empty start, 95 % interval over the R = 20 repetitions excluding 0 | — |
| **H6** | Variables whose shapes do not transport are mostly practice-dependent, not physiological | among variables failing the between-site homogeneity test (Cochran Q over the three sites, Bonferroni over variables × classes), the share in the practice + coding classes is > 50 % and greater than among the variables that pass (one-sided Fisher exact test, α = 0.05); classes pre-declared in the table below | — |

Variable classes for H6 (declared now): **physiological** — laboratory values and vital signs (`last_*`,
`slope_*`, `preop_*`); **practice** — organ supports, medications (ATC), ICU length of stay before discharge, surgery
duration, anaesthesia type, admission source, emergency status; **coding** — diagnoses, ICD-10-PCS levels, surgical
group; **patient** — age, sex, BMI, ASA (reported, not counted in H6).

No adjustment for multiplicity across H1–H6 (each answers a distinct question; estimates with CIs are the main
output); the Bonferroni correction inside H6 is part of its test.

## Secondary and exploratory analyses (estimates with CIs, no decision rule)

- **A.** Decomposition of the transport loss: A(i) frozen metric, A(ii) local NCA with prior at INSPIRE's w_j
  (INSPIRE's λ, g_j frozen), A(iii) full local PERISIM, same case base and test.
- **B.** Case-base composition (local; INSPIRE ∪ local; INSPIRE ∪ other external ∪ local; INSPIRE only) and site weight
  γ ∈ {1, 0.5, 0.25, 0} on non-local votes and prior counts, plus the schedule γ(n).
- **C.** Accumulation curves (n_local ∈ {0, 50, 100, 250, 500, 1000, 2000, all}; starts empty / INSPIRE / INSPIRE ∪
  other external; every γ) in time-block order (MIMIC-IV anchor-year groups, eICU discharge year; random inside the
  blocks, R = 20) and in random order; k_eff = min(k, case-base size).
- **D.** α (and k) tuned on 100 / 250 / 500 local patients versus none; logistic recalibration (sensitivity only);
  cumulative-threshold calibration for T2.
- **E.** Shapes re-estimated per external site with a reduced consensus (40 models, same procedure, INSPIRE's value
  grid); I² per variable and class; range coverage and missingness differences per variable.
- **F.** eICU by hospital: B and C per hospital; between-hospital vs between-country heterogeneity.
- T1 and T3 experiments: the same analyses, **descriptive** only.

## Implementation details (declared with the approval, 2026-10-25)

- H1: "outside ±δ₁" in mode A means a 95 % CI entirely beyond the margin (demonstrated miscalibration).
- E: each site's curves are divided by one site-level scale, the median over variables of the per-variable weighted
  least-squares slope against INSPIRE (robust to a few non-transporting variables), before Cochran's Q; the reduced
  consensus is run in all three sites, INSPIRE included. Sites whose target has other classes (eICU reduced T1) do
  not enter E.
- A: the frozen metric keeps INSPIRE's α; the locally refitted metrics (A(ii), A(iii)) fit their α on the local pool.
- A, C and D use the primary labelling; B uses both labellings. The % out of support of C is computed in the first
  5 repetitions (cost). F uses random order only.

## Known limitations, declared in advance

- No reliable calendar order: MIMIC-IV dates are shifted per patient (only 3-year anchor-year groups are real, and
  they coincide with the eMAR → prescriptions change of the medication source), eICU has two discharge years, INSPIRE
  none. The "temporal" accumulation is therefore block-wise with random order inside; a fully random order is
  reported alongside.
- The per-site shapes of E come from a reduced ensemble and from the external cohorts' own outcomes, so their
  homogeneity test has less power than the development consensus.
- γ = 0 keeps the composed neighbour search (non-local neighbours occupy slots but do not vote); "local only" is a
  separate composition.
