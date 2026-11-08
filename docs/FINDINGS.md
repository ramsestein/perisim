# What PERISIM found

A plain summary of every result, with the strength of evidence behind each, how it relates to prior work, and where
to find the full analysis. All figures are aggregates; the reports they come from are linked.

## The question

A kNN predicts a patient's risk from the outcomes of the most similar patients. "Similar" is a distance with one
weight per variable, learned by a diagonal neighbourhood components analysis (NCA; Goldberger et al., 2005). The
algorithm is not new, and PERISIM does not present it as a method.

The project studies **transport**: a metric learned at some hospitals is frozen and used at another, voting among
that hospital's own patients (its *case base*). The question is what travels and what is lost. Nothing is refitted at
the target.

## How the evidence is graded

| label | meaning |
|---|---|
| **pre-registered** | plan and decision rule fixed before the outcome data were analysed. `hypo` and `roll` were pre-registered locally (freeze records in `docs/`); NWICU on OSF ([10.17605/OSF.IO/P5CA3](https://doi.org/10.17605/OSF.IO/P5CA3)) |
| **post hoc, pre-specified** | an analysis added after the main results, whose plan and rule were committed (`docs/DECISIONS.md`) before it was run |
| **exploratory** | added after results without a fixed rule; hypothesis-generating |

## Main result: the learned metric travels

**Setting.**
- The outcome is hypotension (an hour with median MAP < 65 mmHg, or death in the ICU) in the next 6 h, predicted
  every 2 h during the first 48 h after surgery.
- The data are four ICU systems: INSPIRE (Seoul National University Hospital), MIMIC-IV (Boston), eICU (208 US
  hospitals) and NWICU (Chicago).

**Rotation across three systems** (`hypo`, pre-registered; the equivalence margin was fixed post hoc).
- A metric learned on two systems and used on the third discriminates almost as well as one learned on the third:
  ΔAUROC −0.001 to −0.003.
- To summarise this we use R, the share of a locally learned metric's gain over a Euclidean kNN that the transported
  metric keeps. By a bioequivalence convention, R is "equivalent" when the lower limit of its 90 % CI is ≥ 0.80.

| target | R (90 % CI) | reading |
|---|---|---|
| INSPIRE | 0.91 (0.83–0.99) | equivalent |
| eICU | 0.96 (0.93–1.00) | equivalent |
| MIMIC-IV | 0.74 (0.68–0.80) | not shown |

Sources: [hypo report](../results/hypo/REPORT.md) §3–§4 and §16.

**A fourth, independent system** (NWICU, pre-registered on OSF). All three frozen metrics are equivalent to a metric
learned on NWICU's own patients, and in fact better than it: R 1.46, 1.53 and 1.64, with every lower 90 % limit
≥ 1.10. This is the confirmatory replication. Source: [results/nwicu](../results/nwicu/README.md).

**Hospital by hospital** (eICU's ten largest hospitals, post hoc, pre-specified).
- Weights learned at other hospitals, or in the two other systems abroad, do better than weights learned on the
  hospital's own patients: pooled R 1.35 and 1.38, equivalent in all 8 evaluable hospitals.
- It makes no difference whether the weights come from the same system or from abroad: −0.0004 nats, not
  significant.

Source: [hypo report](../results/hypo/REPORT.md) §20.

**Reading.**
- A learned similarity metric keeps most or all of its gain when moved between health systems.
- For a hospital of ordinary size, borrowing the metric is better than learning it locally, because one hospital has
  too few patients to learn the weights well.
- An R above 1 means "the local reference is data-limited", not "transport beats the truth".

## What does the work: the case base calibrates, the weights discriminate

- **Calibration comes from the local case base, whatever the metric.** Any outcome-weighted metric (NCA, mutual
  information, Euclidean) voting among local patients is calibrated: calibration in the large 0.008–0.023, slope
  1.01–1.03. Transported parametric models without recalibration are far off, because base rates differ between
  sites (13 %, 31 %, 19 %). [hypo report](../results/hypo/REPORT.md) §4, §17.
- **Jointly learned weights beat marginal ones, but only slightly.** The comparison is with a fixed metric weighted by
  mutual information (as in PRAM) on the same geometry: +0.0048 nats in INSPIRE, +0.0040 in eICU, −0.0008 in
  MIMIC-IV. [hypo report](../results/hypo/REPORT.md) §17.
- **Learned weights clearly beat a cosine kNN** (post hoc, pre-specified), at all four targets: +0.008 to +0.015
  nats and +0.007 to +0.019 AUROC. Cosine does no better than Euclidean, and fixed mutual-information weights also
  beat it. So what pays is weighting the variables, not the choice between angle and distance.
  [hypo report](../results/hypo/REPORT.md) §21.
- **The per-variable shape functions add little.** What travels is mainly the weights.
- **Few local patients are needed.** A case base of 250 patients is within 0.02 nats of the whole site, and 500
  within 0.01. [hypo report](../results/hypo/REPORT.md) §5.

## What converges: relevance structure, not neighbours

These analyses are exploratory, with a pre-specified robustness check, and use the mutual-kNN alignment of Huh et al.
(2024) with the permutation null of Gröger et al. (2026).
- Two metrics learned on disjoint patients of one site agree more as data grow. This is robust to k and to the set
  size.
- The transported metric stays below that within-site agreement: each site has structure of its own.
- Sharing weights aligns neighbourhoods more than sharing the per-variable geometry.
- "Many different neighbourhoods give the same risk" turned out to be **generic**: any two of these metrics with
  that little overlap agree as much. It is not a transport finding. For case-based explanation it means that the
  "similar patients" shown are not unique even when the risk is.

Source: [hypo report](../results/hypo/REPORT.md) §15.3–15.4.

## Where transport fails, and what we learned

- **MIMIC-IV's shortfall sits in patients on vasopressors.** INSPIRE does not record vasopressors, and the metric
  learned with it gives them little weight (0.47 % effective share). Restricting the neighbours to patients in the
  same vasopressor state repairs MIMIC-IV without refitting (post hoc, pre-specified).
- **The general mechanism was tested and not supported.** The mechanism was: an input missing in one source is
  under-weighted, and transport breaks where it matters. The pre-specified test used sedation, also absent in
  INSPIRE: it was neither under-weighted nor did its stratum fail. Ventilation, the negative control, behaved as
  predicted.
- **The repair is local, not general.** eICU's apparent vasopressor failure disappeared once a landmark-timing leak
  was corrected. In NWICU the same-state rule helps only the same metric (descriptive).
- **The vasopressor stratum is hard for every transported model.** A transported LightGBM loses 4–9 times more there
  than the kNN, but that LightGBM was not recalibrated, so this is descriptive.

Source: [hypo report](../results/hypo/REPORT.md) §15.1, §17–§19.

## Against standard models

- **A transported LightGBM with a two-parameter logistic recalibration predicts better than the transported kNN**:
  +0.010 to +0.013 nats in NWICU. A LightGBM trained locally is also better: +0.011 to +0.016 nats in `hypo`, and
  +0.0095 against the best transported kNN in NWICU.
- **The kNN's case for use is therefore not accuracy.**
  - It needs no refitting, not even recalibration.
  - It is calibrated out of the box.
  - Each prediction can be traced to the patients that produced it.
  - It is updated by data only, by changing the case base.
- **Along a stream of patients** (`roll`, pre-registered), a rotating case base with the frozen metric:
  - is better calibrated than a LightGBM managed as a hospital would manage it, in all three streams;
  - is non-inferior in discrimination only in MIMIC-IV.

  Source: [roll report](../results/roll/REPORT.md).

## The first experiment, T2

T2 asked the same transport question at ICU discharge (length of stay and death). It was the hard case:
- the signal was weak;
- the transported metric was not equivalent (R 0.49 in MIMIC-IV, 0.72 in eICU);
- the hospital mattered about ten times more than in `hypo`.

Source: [results/t2](../results/t2/).

## Relation to prior work: what is ours and what is not

**Not ours.** These are cited and not claimed:
- **Adapting a frozen model without retraining by swapping a local retrieval bank, with a fixed metric:** PRAM (Jeong
  et al., 2026, medRxiv, v2). PRAM's supplementary Table S13 also reports, in one target (eICU) and as a sensitivity
  analysis, a neighbourhood-components metric transferred from development that performs similarly to one fitted on
  the target bank, both below cosine distance. In our setting the order is reversed (§21): NCA beats cosine, and so do
  fixed mutual-information weights, which in PRAM's S13 match cosine. The two results are compatible: weighting the
  variables pays in our setting (landmarks every 2 h, hypotension in 6 h) and hardly pays in theirs (42 static
  first-24 h aggregates, AKI and mortality).

  A pre-registered follow-up (OSF [10.17605/OSF.IO/3SBFG](https://doi.org/10.17605/OSF.IO/3SBFG);
  [results/pram](../results/pram/README.md)) rebuilt PRAM's setting and reproduced S13's ordering. It found that the
  reversal is explained mainly by **how the neighbours are used together with the NCA implementation**: a mixture
  with a frozen model and a full-matrix NCA erases the learned metric's advantage, also in our data; a standalone
  diagonal NCA restores it in PRAM's setting for AKI. The vote plays no part, and for 168 h mortality weighting gains
  little in either study.
- **Pooled or external models matching locally retrained ones when local data are scarce:** Johnson et al. (2018),
  Wiens et al. (2014), and the anchor-regression and foundation-model work of Londschien, Burger and colleagues.
- **Transport failing mainly through calibration, and simple recalibration correcting it:** Van Calster et al.
  (2019), Vergouwe et al. (2017), Yao et al. (2026).
- **NCA itself** (Goldberger et al., 2005), the mutual-kNN alignment (Huh et al., 2024) and its calibrated null
  (Gröger et al., 2026).

**What we can claim, to our knowledge.**
- The retention of a learned similarity metric's gain under transport, quantified as an estimand (R) with an
  equivalence criterion, across three health systems by rotation and ten hospitals, and confirmed in a
  pre-registered fourth system.
- That for hospitals of ordinary size borrowed metric weights are at least as good as locally learned ones, whether
  borrowed from the same system or from abroad.
- That, in this setting, a learned metric clearly beats a cosine kNN at four targets, and that the gain comes from
  weighting the variables, not from the choice between angle and distance.
- That the disagreement with PRAM about learned metrics against cosine is mainly methodological (mixture plus full-matrix
  NCA against a standalone diagonal NCA), shown by a pre-registered factorial reconstruction of both settings.
- That what converges between sites is the relevance structure (the weights), not the neighbours.
- A pre-specified, negative test of the "missing-in-source" failure mechanism, with a site-specific repair.

## Limits

- **Pre-registration.** `hypo` and `roll` were pre-registered locally; only NWICU is on OSF. R's 0.80 margin was
  fixed after the first rotation's point estimates were known.
- **Reading R.** R is unstable when the local gain over Euclidean is small (as in NWICU), and its bootstrap does not
  carry the variance of refitting the local metric.
- **Data.** Four systems, all ICUs, mostly surgical. NWICU's population is a surgical-ICU proxy, with MAP computed
  from SBP and DBP and no ventilation record.
- **Accuracy.** The kNN does not beat a recalibrated gradient-boosting model.

The full chronology of decisions, including errors found and corrected, is in [DECISIONS.md](DECISIONS.md).
