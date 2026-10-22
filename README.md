# PERISIM — Perioperative Similarity

kNN con métrica de distancia aprendida (SAGE consensuado entre muchos modelos + NCA diagonal regularizado)
para pacientes postoperatorios en el alta de UCI. Desarrollo en INSPIRE, validación externa en MIMIC-IV y eICU.
La especificación completa está en [SPECS.md](SPECS.md).

> **Repositorio público.** Solo código y configuración: nunca datos, salidas ni secretos. Antes de cada
> `git push`: `python scripts/check_public_safety.py --all` (lo hace también el hook `.githooks/pre-commit`).

## Ejecución en el pod de KHDP

En la terminal de VSCode del pod:

```bash
mkdir -p ~/workspace/updown/perisim/$(id -un) && cd ~/workspace/updown/perisim/$(id -un)
git clone http://github.khdp-forward-server/ramsestein/perisim.git
cd perisim
bash run_pod.sh
```

`run_pod.sh` lo hace todo solo:

1. Busca Python ≥ 3.11 (o crea uno con conda, que en KHDP ya apunta al mirror) y crea un entorno virtual en
   `~/.perisim/venv`, reutilizando los paquetes del sistema.
2. Instala las dependencias desde el mirror PyPI de KHDP y torch desde el mirror de PyTorch
   (índice CPU y, si no responde, cu126).
3. La primera vez que corre cada versión del código, lanza un **smoke test con datos sintéticos**: si falla,
   no toca los datos reales.
4. Lanza el pipeline **en segundo plano**, de forma que sobrevive al cierre de la terminal, y muestra el log.
   Ctrl-C deja de seguir el log, pero no detiene el pipeline.

| Orden | Para qué |
|---|---|
| `bash run_pod.sh` | preparar el entorno y lanzar / **reanudar** el pipeline |
| `bash run_pod.sh status` | qué etapas están hechas y dónde se ha parado |
| `bash run_pod.sh logs` | seguir el log de la ejecución en marcha |
| `bash run_pod.sh stop` | detenerla |
| `bash run_pod.sh fg --until audit` | ejecutar en primer plano hasta una etapa concreta |
| `bash run_pod.sh test` | pytest, incluidos los tests anti-fuga sobre los datos reales |
| `bash run_pod.sh freeze --registration <ID>` | congelación §6.3 (habilita la validación externa) |

Para actualizar el código: `git pull` y `bash run_pod.sh`. Solo se rehacen las etapas cuyo código o
configuración han cambiado, porque cada etapa guarda un marcador con su hash. Tras un OOM o un reinicio del
pod basta con relanzar: se retoma desde la última etapa (o el último fold) completado.

### Rutas (configurables con variables de entorno)

| Qué | Por defecto | Variable |
|---|---|---|
| Datos (solo lectura) | `/home/khdp-user/workspace/datasets` | `PERISIM_DATASETS` |
| INSPIRE / MIMIC-IV / eICU | descubrimiento automático bajo datasets | `PERISIM_INSPIRE_DIR`, `PERISIM_MIMIC_DIR`, `PERISIM_EICU_DIR` |
| Trabajo y resultados | `~/workspace/updown/perisim/<usuario>/work` | `PERISIM_WORKDIR` |
| Entorno y logs de la terminal | `~/.perisim` | `PERISIM_LOCAL` |

## Qué hace el pipeline y dónde se para

```
env → discover → extract → targets → features → leakage → audit ⛔ → harmonize ⛔ → pilot → cv → temporal
    → report → sensitivity → external ⛔ → export
```

| Etapa | Salida (en el workdir) | Puerta |
|---|---|---|
| `discover` | `reports/schema_<cohorte>.md`: columnas e ítems reales frente a `config/harmonization.yaml` | para si no encuentra INSPIRE o su esquema no coincide |
| `extract` | `common/<cohorte>/`: `stays`, `inputs_long`, `outcomes` y diagrama de flujo (§8.3) | — |
| `targets`, `features` | `derived/…`: T1/T2/T3, variables arrastradas e inputs con corte t0 − 2 h | — |
| `leakage` | test fila a fila de que ningún input supera el corte, más pytest | **para** si hay fuga (§4.4) |
| `audit` | `reports/completeness_<cohorte>.html` (§7) | **para** si algún componente de T1 tiene prevalencia < 0,5 % o ausencia > 30 % |
| `harmonize` | conjuntos de inputs `core` / `extended` / `full` | **para** si falta alguna cohorte montada |
| `pilot` | `reports/pilot.html`: tiempos, memoria y coste estimado; fija D7 con una regla declarada | — |
| `cv` | CV anidada 5 × 3 por paciente, PERISIM + 10 comparadores, k ∈ {10…200} | — |
| `report` | `reports/results_main.html` (TRIPOD+AI) | — |
| `sensitivity` | variantes de `experiment.yaml: sensitivity` | — |
| `external` | modos A y B en MIMIC-IV y eICU (§6.2) | **espera** a `freeze` (§6.3) |
| `export` | `export/`: solo agregados, con manifiesto y escaneo de identificadores | — |

**Primera ejecución esperable:** casi seguro se para en `discover` o en `audit`. En ese caso:

1. Revisa `reports/schema_*.md` y `reports/completeness_*.html` dentro del workdir.
2. Corrige `config/harmonization.yaml` en tu equipo, haz push y luego `git pull` en el pod.
3. Anota las decisiones en `docs/OPEN_QUESTIONS.md` y `docs/DECISIONS.md`.
4. Para continuar tras la auditoría: `PERISIM_ACK_AUDIT=true bash run_pod.sh`.

**Validación externa:** cuando la validación interna esté cerrada, confirma el estado del repositorio y
congela:

```bash
bash run_pod.sh freeze --registration <ID_OSF>
```

Después relanza `bash run_pod.sh`. A partir de ese momento, cualquier cambio de código o configuración bloquea
la etapa `external`, porque §6.3 prohíbe iterar mirando resultados externos.

## Exportación (KHDP §3.2)

`<workdir>/export/` contiene solo agregados: informes HTML, métricas, pesos w_j, importancias y tablas VDM con
celdas de n ≥ 10. Incluye `EXPORT_MANIFEST.md` y un zip. Para sacarlo del pod hay que escribir a
`admin@khdp.net` indicando el propósito y el destino, y seguir su revisión. Las predicciones por paciente
(`predictions/`) y los extractos (`common/`, `derived/`, `cache/`) **nunca** se exportan.

## Desarrollo local (sin datos reales)

```bash
pip install -r requirements.txt torch
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m perisim smoke
```

`smoke` genera datos sintéticos con el esquema de las tres cohortes y ejecuta el pipeline completo, incluidas
la congelación y la validación externa, con el perfil `config/profiles/smoke.yaml`.

## Estructura

```
config/            cohorts, harmonization (único sitio con nombres originales), targets, model_space, experiment
src/perisim/
  data/            lectura perezosa (Parquet zstd), descubrimiento, extractores INSPIRE / MIMIC-IV / eICU, sintéticos
  cohort/          inclusión, exclusión, t0 y corte
  targets/         T1 (KDIGO, curso de antibiótico, reingreso…), T2, T3, variables arrastradas
  features/        inputs con procedencia temporal (test anti-fuga) y tipos métricos
  importance/      sustitutos con feature dropout, SAGE por permutaciones, agregación y parada (§5.3)
  metric/          δ_j (continuas, ordinales, binarias, VDM, jerárquicas) y NCA diagonal regularizado (§5.4)
  knn/             FAISS, probabilidades suavizadas, fuera de soporte y salida por paciente (§5.5–5.6)
  baselines/       comparadores de §6.4
  evaluation/      métricas, bootstrap por paciente, auditoría, informes y exportación
  pipeline.py      etapas, puertas y paralelismo (4 procesos × 4 hilos)
tests/             anti-fuga, partición por paciente, VDM en el fold, métrica y targets
docs/              OPEN_QUESTIONS.md y DECISIONS.md
```
