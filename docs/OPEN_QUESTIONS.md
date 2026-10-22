# OPEN_QUESTIONS — PERISIM

Dudas que el código NO decide en silencio (SPECS.md, preámbulo). Cada entrada indica qué se ha implementado
como propuesta v0 y qué hay que confirmar. Las incidencias detectadas en el pod se añaden automáticamente a
`<workdir>/docs/OPEN_QUESTIONS.md` (se exportan con los agregados); trasladar aquí las decisiones tomadas.

## Esquemas y datos [VERIFICAR en la primera ejecución: `reports/schema_<cohorte>.md`]

1. **INSPIRE — nombres de columnas e ítems.** `config/harmonization.yaml` usa los nombres de la publicación
   (operations: `icuin_time`, `emop`, `antype`, `icd10_pcs`…; labs/ward_vitals: `item_name`, `value`).
   El inventario de `item_name` del informe de esquema dirá qué renombrar.
2. **INSPIRE — dispositivos en `ward_vitals`.** Se asume que ventilación/TRS/ECMO/BCIA aparecen como registros
   de "activo" (`vent`, `crrt`…). Si son solo marcas de inicio, cambiar `sources.inspire.devices.mode`.
3. **INSPIRE — reingreso en UCI (D1).** Implementado como nueva entrada en UCI ligada a otra cirugía de la misma
   hospitalización. Los reingresos no quirúrgicos no son observables por esta vía; la auditoría lo cuantifica.
   Si no es suficiente: T1 grave en INSPIRE sin reingreso y sensibilidad declarada.
4. **INSPIRE — columnas opcionales.** Donantes (v1.4.2+), año (validación temporal), destino al alta y
   limitación del esfuerzo: `optional_columns` está a `null` hasta ver el esquema real.
5. **INSPIRE — medicación ATC.** Se usan `atc_code`, `atc_code2`, `atc_code3` como códigos alternativos de una
   misma administración. Confirmar su significado (¿niveles distintos del mismo fármaco?).
6. **INSPIRE — transfusión de hematíes.** Se busca por nombre de fármaco (regex). Confirmar si existe registro
   en planta.
7. **MIMIC-IV — diagnósticos.** `diagnoses_icd` no tiene marca temporal (facturación al alta): se excluye como
   input por la regla anti-fuga. Consecuencia: los diagnósticos no entran en el conjunto `core` (intersección).
8. **MIMIC-IV — regla de UCI postoperatoria (D10).** Servicio quirúrgico vigente al ingreso en UCI **o**
   procedimiento ICD-10-PCS sección 0 con abordaje abierto/percutáneo endoscópico (o ICD-9 01–86) ese día o el
   anterior. Revisar con la auditoría.
9. **MIMIC-IV — medicación.** Sin ATC: mapeo nombre → ATC por regex (`harmonization.medications.name_to_atc`).
   La transfusión solo es observable dentro de UCI (inputevents) salvo que eMAR la recoja.
10. **eICU — fuentes.** Ventilación (`respiratoryCare`, `treatment`), vasopresores (`infusionDrug`), GCS
    (`nurseCharting`), FiO2 (`respiratoryCharting`). Medicación = órdenes, no administraciones.
11. **Formato en KHDP.** Se admiten `.csv`, `.csv.gz`, `.parquet` y carpetas de Parquet. Si KHDP sirve las bases
    como base de datos (Postgres/BigQuery), hará falta un lector nuevo en `src/perisim/data/io.py`.

## Método

12. **Prior del NCA y ancho de banda.** `L(w) = LogLoss + λ Σ (log w_j − log c − log(imp_j + ε₀))²` con una
    escala global `c` libre: el prior regulariza el *perfil* de importancias y deja libre el ancho de banda
    (si no, el prior fijaría también la escala del kernel). Confirmar esta lectura de §5.4.
13. **λ dentro del bucle de parada.** El criterio §5.3.2 recalcula w con la conversión §5.4; se usa un λ fijo
    (`importance.lambda_loop = 1`) porque el λ final se elige después por CV interna.
14. **Selección de λ con SAGE del fold externo.** SAGE se calcula una vez por fold externo (con el split interno
    0) y se reutiliza para elegir λ en los 3 folds internos. Recalcular SAGE por fold interno triplicaría el coste.
15. **α del kNN.** Se elige por k con LOO por paciente en el entrenamiento del fold externo (igual para todos los
    comparadores kNN), no en los folds internos.
16. **Métrica de la salida clínica (§5.6).** Una métrica por target. La salida por paciente usa los vecinos de la
    métrica de `patient_output_target` (T1 por defecto). ¿Debe ser T3 (objetivo integrado)?
17. **Reparto de importancia dentro de un grupo.** Bloques grandes (dx/ATC > 50): ∝ prevalencia × información
    mutua (SAGE univariante con modelo bayesiano óptimo). Resto (valor + indicador, niveles PCS, clústeres
    |ρ| > 0,8): a partes iguales.
18. **Regla de D7 desde el piloto.** `min_delta = 0,05 × Δ̃` del piloto en la iteración n_min (regla declarada en
    `experiment.yaml` antes de ejecutar). `tol_Δ = 0,01`.
19. **SWIFT.** El origen del ingreso en UCI se aproxima con el origen del ingreso hospitalario; los valores
    ausentes puntúan 0. Recalibración logística en el fold de entrenamiento.
20. **T3.** "Alta precoz" = estancia ≤ primer tercil y "prolongada" > segundo tercil, calculados en el fold de
    entrenamiento sobre los pacientes sin complicación.

## Decisiones abiertas de SPECS §10 (valores v0 en config)

D1 (reingreso INSPIRE), D2 (2 h; 4 h en sensibilidad), D3 (≥ 4 h), D4 (LRA 1–2, antibiótico, transfusión),
D5 (terciles), D6 (0,5 %), D7 (desde el piloto), D8 (desactivado), D9 (OSF: `freeze --registration`),
D10 (regla MIMIC/eICU), D11 (k = 50).
