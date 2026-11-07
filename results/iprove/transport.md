# INSPIRE → iPROVE: intraoperative vasoactive drugs from preoperative variables

Plan and rule fixed before running: docs/DECISIONS.md, 2026-11-07. Source INSPIRE (16,748 patients, 48.6% with vasoactives); target iPROVE (4,273 patients, 38.1%). Five folds in iPROVE; 2,000 patient resamples.

| arm | refitted on iPROVE | log-loss (95 % CI) | AUROC | AUPRC | Brier | CITL | slope |
|---|---|---|---|---|---|---|---|
| knn_uniform | case base only | 0.6061 (0.5938–0.6186) | 0.684 | 0.521 | 0.2119 | +0.100 | 0.93 |
| knn_mi | case base only | 0.6048 (0.5928–0.6172) | 0.689 | 0.527 | 0.2108 | +0.085 | 0.94 |
| knn_nca | case base only | 0.6208 (0.6100–0.6318) | 0.659 | 0.494 | 0.2171 | +0.027 | 0.91 |
| knn_nca_local | weights + case base | 0.6115 (0.6003–0.6223) | 0.676 | 0.517 | 0.2139 | +0.041 | 0.95 |
| lr_raw | nothing | 0.7168 (0.7079–0.7260) | 0.644 | 0.502 | 0.2606 | -0.835 | 1.05 |
| lgbm_raw | nothing | 0.7309 (0.7214–0.7406) | 0.621 | 0.481 | 0.2673 | -0.870 | 0.85 |
| lr_recal | recalibration | 0.6363 (0.6263–0.6461) | 0.643 | 0.501 | 0.2227 | -0.000 | 0.99 |
| lgbm_recal | recalibration | 0.6432 (0.6340–0.6519) | 0.620 | 0.481 | 0.2261 | -0.000 | 1.00 |
| lr_local | everything | 0.6199 (0.6091–0.6303) | 0.682 | 0.507 | 0.2164 | +0.002 | 1.01 |
| lgbm_local | everything | 0.5832 (0.5703–0.5956) | 0.710 | 0.536 | 0.2039 | +0.005 | 0.93 |

- R, NCA transported: not evaluable
- R, MI weights: not evaluable
- knn_nca − lr_recal: -0.0155 (-0.0232 to -0.0082)
- knn_nca − lgbm_recal: -0.0224 (-0.0304 to -0.0147)
- knn_nca − knn_mi: +0.0160 (+0.0085 to +0.0232)
- knn_nca − lgbm_local: +0.0376 (+0.0287 to +0.0469)

## Weights learned in INSPIRE (share)

| variable | NCA | MI |
|---|---|---|
| edad_d | 0.310 | 0.260 |
| genero | 0.000 | 0.003 |
| altura_d | 0.016 | 0.030 |
| peso_d | 0.004 | 0.014 |
| IMC_d | 0.000 | 0.009 |
| cirugiaAbdominalProgramada | 0.000 | 0.001 |
| esCirugiaOncologica | 0.225 | 0.149 |
| ASA | 0.163 | 0.289 |
| spO2pre_b | 0.000 | 0.044 |
| hbPreoperatoria_d | 0.061 | 0.113 |
| infeccionRespiratoriaUltimoMes | 0.000 | 0.001 |
| hipertensionArterial | 0.000 | 0.017 |
| DM | 0.000 | 0.033 |
| cardiopatiaIsquemica | 0.000 | 0.016 |
| fumador | 0.017 | 0.000 |
| consumoAlcohol | 0.017 | 0.000 |
| dislipemia | 0.017 | 0.000 |
| ePOC | 0.169 | 0.007 |
| insuficienciaRenal | 0.000 | 0.014 |

## Rule (pre-specified)

```
{
 "Q1_R_knn_nca": "not evaluable",
 "Q2_vs_lr_recal": "kNN better",
 "Q2_vs_lgbm_recal": "kNN better",
 "Q3_knn_nca_calibration": {
  "citl": 0.026805861926964786,
  "slope": 0.9137872661458356
 }
}
```
