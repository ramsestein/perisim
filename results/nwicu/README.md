# NWICU: transport of the frozen `hypo` metrics to a fourth system (pre-registered)

Pre-registration: [docs/prereg_nwicu.md](../../docs/prereg_nwicu.md), OSF
[10.17605/OSF.IO/P5CA3](https://doi.org/10.17605/OSF.IO/P5CA3), registered before the landmarks were built. Run on
the lead's workstation on 2026-11-07, with `hypo`'s frozen code; the freeze record was checked at start and end.
Tables: [transport.md](transport.md), [build.md](build.md).

**Data.** NWICU (Northwestern Medicine, 2020–2022), first surgical-ICU stay (SICU or CTICU) per hospitalisation:
2,001 patients, 30,833 analysis landmarks, outcome 12.9 %.

**Primary (pre-registered): confirmed.** All three frozen metrics are equivalent to a metric learned in NWICU, and
better than it:

| metric learned on | R (90 % CI) |
|---|---|
| MIMIC-IV + eICU | 1.46 (1.10–2.08) |
| INSPIRE + eICU | 1.53 (1.14–2.21) |
| INSPIRE + MIMIC-IV | 1.64 (1.25–2.34) |

Log-loss is 0.3180–0.3193 for the transported metrics, 0.3225 for the local metric and 0.3295 for Euclidean. As in
eICU's hospitals (`hypo` §20), R > 1 means NWICU's own patients learn a weaker metric than the larger development
sites did: the local reference is data-limited.

**Secondary.**
- **Recalibrated LightGBM beats the transported kNN**: by 0.0104 to 0.0127 nats (95 % CIs exclude 0), at all three
  rotations. A LightGBM learned elsewhere, with two recalibration parameters fitted in NWICU (log-loss
  0.3066–0.3077), is nominally below a LightGBM trained in NWICU (0.3086); that difference was not tested.
- **Calibration without recalibration**: the transported kNN has CITL +0.07 to +0.10 and slope 1.00–1.06. The raw
  transported LightGBM has CITL −0.48 to −0.89.
- **Cost of not refitting**: local LightGBM beats the best transported kNN by 0.0095 (0.0068–0.0122).
- **Vasopressor stratum (descriptive, no inferential criterion)**: the transported metrics' loss is below the local
  metric's on and off vasopressors.

**Added after the audit of 2026-11-07** (DECISIONS, same date):
- **B0 by vasopressor state.** It was pre-registered as descriptive but not computed in the first run; it was added
  with the primary run's units unchanged. On the vasopressor stratum, same-state neighbours help only the metric
  learned on INSPIRE + eICU (+0.0036; positive = B0 better). This is the metric that gives the vasopressor the least
  weight, and the same one that failed in MIMIC-IV. For the other two metrics B0 costs 0.0020 and 0.0024. Over all
  landmarks B0 costs 0.0003 to 0.0008. These are point estimates, with no test.
- **Intravenous vasopressor forms only (post hoc sensitivity).** The primary rule also counted nasal, rectal,
  ophthalmic and oral phenylephrine and the epinephrine auto-injector. Inside the cohort they are 8 of 29,254
  vasopressor records. With them removed, every figure of the table is unchanged to the fourth decimal
  ([ivonly/transport.md](ivonly/transport.md)).

**Reading.**
- **The main result replicates in an independent, shareable system, as pre-registered.** A similarity metric learned
  in two other systems, frozen and used with NWICU's own patients, keeps all of its gain; R's 90 % CIs lie above 1.
- **Read R with its reference in mind.** The local metric gains only 0.007 nats over Euclidean, just above the
  0.005 rule, and comes from one fit per fold. The bootstrap holds predictions fixed, so it does not carry the
  variance of refitting. "Equivalent" here means "at least as good as a data-limited local metric".
- **Two things do not favour the kNN.** In this system the strongest option is a transported parametric model with
  a two-parameter recalibration. And the kNN, calibrated without any recalibration, slightly underpredicts.

**Limits.**
- The surgical-ICU proxy for the postoperative population.
- MAP computed from SBP and DBP.
- Supports from eMAR points, and no ventilation.
- One system, mostly 2020–2022.
