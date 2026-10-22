# DECISIONS — registro fechado

Decisiones de implementación tomadas por el responsable o derivadas de SPECS.md. Las que el pipeline toma en
el pod por reglas declaradas (D7 desde el piloto, λ elegido, mesetas de Δ̃, exclusión de la MLP por coste,
validación temporal omitida…) se registran automáticamente en `<workdir>/docs/DECISIONS.md`.

## 2026-10-21 — Infraestructura de ejecución en KHDP

- Código en repositorio público de GitHub, clonado en el pod a través de `http://github.khdp-forward-server`.
  El repositorio no contiene datos, secretos ni salidas (`.gitignore` + `scripts/check_public_safety.py`).
- Punto de entrada único `run_pod.sh`: entorno virtual en disco local del contenedor (`~/.perisim/venv`),
  dependencias desde el mirror PyPI de KHDP y torch desde el mirror de PyTorch (índice CPU, reserva cu126).
- Datos de solo lectura en `/home/khdp-user/workspace/datasets` (descubrimiento automático de las tres cohortes);
  trabajo y artefactos en `updown/perisim/<usuario>/work`.
- Pipeline por etapas idempotentes con marcadores (hash de configuración + código): un OOM o un reinicio del pod
  reanuda desde la última etapa (o el último fold) completado.
- Paralelismo: como mucho 4 procesos × 4 hilos (§9.1); FAISS `IndexFlatL2` en CPU.

## 2026-10-21 — Interpretaciones de SPECS (ver OPEN_QUESTIONS para confirmar)

- Normalización de δ_j: cada bloque se divide por s_j con s_j² = E[δ_j²] entre pares aleatorios del
  entrenamiento, de modo que el cuadrado medio de cada disimilitud es 1.
- Niveles ICD-10-PCS: one-hot por nivel (prefijos de 1 a 5 caracteres); códigos con < 10 casos en el
  entrenamiento → vector nulo.
- `carried` lo construye la etapa de targets a partir del formato común (no la extracción).
- La primera estancia postoperatoria de cada hospitalización es la unidad de análisis: si no cumple un criterio
  posterior (p. ej. < 4 h), la hospitalización queda fuera.
