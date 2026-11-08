# Why a learned metric beats cosine in PERISIM but not in PRAM (pre-registered)

**Pre-registration:** [docs/prereg_pram_factors.md](../../docs/prereg_pram_factors.md), OSF
[10.17605/OSF.IO/3SBFG](https://doi.org/10.17605/OSF.IO/3SBFG). Deviations and the run log are dated in
[docs/DECISIONS.md](../../docs/DECISIONS.md) (2026-11-08). Full tables: [tier1.md](tier1.md) and [tier2.md](tier2.md),
with JSON beside them.

## The question

PERISIM's transported diagonal NCA metric beats a cosine kNN (REPORT §21). PRAM's Table S13 finds the opposite in eICU
(Jeong et al., 2026). Four candidate factors, each with a PERISIM level and a PRAM level:
- **U, use:** a standalone kNN, or a mixture with a frozen model, p = (1 − a)·p_base + a·p_kNN, a ≤ 0.5.
- **V, vote:** PERISIM's smoothed vote, or 1/(d + 1e-6).
- **I, NCA implementation:** diagonal, or scikit-learn's full matrix.
- **S, setting:** PERISIM's landmarks and hypotension, or PRAM's 42 static features with acute kidney injury (AKI)
  and 168 h mortality.

The two tiers:
- **Tier 1** crosses U, V and I on PERISIM's data (INSPIRE, MIMIC-IV, eICU, and NWICU with three frozen metrics).
- **Tier 2** rebuilds PRAM's setting (MIMIC-IV → eICU) and crosses the same factors.

Throughout, Δ = AUROC(NCA) − AUROC(cosine), with 95 % bootstrap CIs.

## Integrity
- **Tier 1:** all 22 gates reproduced the stored predictions (max |Δp| ≤ 2.2e-16). The home cell reproduces §21 to
  the fourth decimal.
- **Tier 2:** the cohort was checked before any model was fitted. Two extraction errors were corrected (MIMIC-IV
  weight; eICU GCS components) and a KDIGO 48 h sensitivity outcome was added; all three are dated deviations.
- **Fidelity limits:** about 30 % more stays than PRAM, and about twice its AKI prevalence (17.6 % against 6.1 % in
  eICU). Our frozen logistic model is also stronger than PRAM's (mortality AUROC 0.848 against 0.774).

## R0: PRAM's Table S13 is reproduced
At PRAM's levels (mixture with the frozen logistic model, 1/d vote, full-matrix NCA transferred), cosine beats NCA for
both outcomes:

| outcome | Δ (95 % CI) | PRAM |
|---|---|---|
| AKI | −0.0018 (−0.0033 to −0.0003) | −0.0092 |
| mortality | −0.0041 (−0.0057 to −0.0024) | −0.0090 |

The effect is smaller than PRAM's, in the same direction, and reproduced with confidence.

## Hypotheses (pre-specified rule)

| hypothesis | Tier 1 (6 units) | Tier 2 (2 outcomes) | verdict |
|---|---|---|---|
| **H-U:** the NCA advantage is larger standalone than mixed | larger in all 6 (effect +0.006 to +0.017) | AKI +0.0032 (CI > 0); mortality **−0.0136 (CI < 0)** | **not supported** |
| **H-V:** the vote matters | ≤ +0.0007; 3 of 6 CIs include 0 | ≤ +0.0002; AKI includes 0 | **inconclusive**; negligible |
| **H-I:** diagonal NCA beats full-matrix NCA | 5 of 6 CIs > 0 (+0.005 to +0.015); one includes 0 | AKI +0.0084, mortality +0.0019 (both CI > 0) | **inconclusive** by the rule (7 of 8 units support it) |
| **H-S:** in PRAM's setting, PERISIM's levels show no NCA advantage | — | AKI: NCA beats cosine (+0.034); mortality: no difference | **not supported** |

## Attribution (pre-specified rule, §6)

**No single factor is sufficient in both tiers.**
- In Tier 2, U or I alone reverses the ordering for AKI, but nothing alone does for mortality.
- In Tier 1, U alone is sufficient in 3 of 6 units and I alone in 1.
- V is never sufficient.

**Pairs: use and implementation together explain the reversal.**
- Tier 1: switching both U and I to PRAM's levels removes the NCA advantage in 5 of 6 units (all but eICU, where NCA
  still beats cosine by +0.0025).
- Tier 2: switching both to PERISIM's levels makes NCA beat cosine for AKI (+0.034) and removes cosine's advantage
  for mortality.

In other words, **PRAM's pipeline (a mixture with a full-matrix NCA) erases a learned metric's advantage also in
PERISIM's data, and PERISIM's pipeline (a standalone diagonal NCA) restores it in PRAM's setting for AKI.**

**The vote plays no part.** The setting alone does not explain the reversal (H-S is not supported), but the outcome
matters: for 168 h mortality, no metric gains from weighting variables. In PRAM's S13, as in ours, the
mutual-information metric ties with cosine there.

## What the factors do (descriptive)
- **Full-matrix NCA is the weak metric.**
  - In Tier 2 it never beats cosine, in any cell: alone or mixed, fitted on 2,000, 5,000 or 10,000 stays, or on the
    target bank.
  - Standalone, the diagonal NCA is much better: AKI AUROC 0.763 against 0.730.
- **The mixture shrinks every difference between metrics.** With a ≤ 0.5 and a base model carrying most of the
  signal, the kNN contributes a correction, and metric differences of 0.01–0.03 become 0.000–0.006.
- **Using the neighbours alone does not always favour NCA.** For mortality, full-matrix NCA alone is far below cosine
  (−0.018), and the mixture hides that. This is why H-U fails.

## Reading
The reversal between the two studies is mainly methodological. A learned metric's advantage over cosine shows when:
- the kNN predicts on its own;
- the metric is a diagonal weighting of variables.

It largely vanishes when the neighbours only correct a frozen model and the metric is a full-matrix NCA. Two
qualifications:
- The outcome matters: weighting gains little for 168 h mortality in either study.
- One unit (eICU in Tier 1) keeps an NCA advantage even under PRAM's pipeline.

This does not make PRAM's choice wrong for its purpose. In a retrieval-augmented mixture, cosine is a reasonable,
cheaper default. The learned metric matters for a standalone, case-based kNN.

## Limits
- PRAM's code is not available. Tier 2 is a reconstruction; it reproduces S13's direction with a smaller effect, and
  its cohort and AKI prevalence differ from PRAM's.
- That PRAM used scikit-learn's full-matrix NCA is an assumption.
- The sufficiency rule turns CIs into categories, so a switch can be "sufficient" by moving a CI just across 0.
- The CIs carry test-set sampling only, not the variance of refitting.

## References
Jeong, I., Lee, T., Kim, B., Park, J.-H., Kim, Y., & Lee, H. (2026). *PRAM: Post-hoc retrieval augmentation adapts
intensive care prediction models without retraining and can be shared across hospitals* (Version 2) [Preprint].
medRxiv. https://doi.org/10.64898/2026.04.03.26350132
