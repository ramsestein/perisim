#!/usr/bin/env bash
# =============================================================================
# PERISIM — punto de entrada único en el pod de KHDP.
#
#   bash run_pod.sh            # prepara el entorno y lanza TODO el pipeline en 2.º plano
#   bash run_pod.sh fg         # igual, en primer plano
#   bash run_pod.sh status     # estado de cada etapa y si hay una ejecución en marcha
#   bash run_pod.sh logs       # seguir el log de la última ejecución (Ctrl-C no la detiene)
#   bash run_pod.sh stop       # detener la ejecución en marcha
#   bash run_pod.sh smoke      # prueba de punta a punta con datos sintéticos
#   bash run_pod.sh test       # pytest (incluye los tests anti-fuga sobre los datos reales)
#   bash run_pod.sh freeze --registration <ID_OSF>   # congelación §6.3 (habilita la validación externa)
#   bash run_pod.sh setup      # solo instalar / comprobar el entorno
#
# Cualquier argumento extra tras start/fg se pasa a `python -m perisim run`
# (p. ej. `bash run_pod.sh fg --until audit`, `--stage features --force`).
#
# Variables de entorno (opcionales):
#   PERISIM_WORKDIR     carpeta de trabajo (defecto: updown/perisim/<usuario>/work)
#   PERISIM_DATASETS    raíz de datasets (defecto: /home/khdp-user/workspace/datasets)
#   PERISIM_INSPIRE_DIR / PERISIM_MIMIC_DIR / PERISIM_EICU_DIR  rutas explícitas (si no, auto)
#   PERISIM_ACK_AUDIT=true   continuar tras revisar la auditoría de completitud (§7)
#   PERISIM_SKIP_SMOKE=1     no ejecutar el smoke test antes de la primera ejecución real
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-start}"
[ $# -gt 0 ] && shift || true

: "${PERISIM_WORKSPACE:=/home/khdp-user/workspace}"
: "${PERISIM_DATASETS:=$PERISIM_WORKSPACE/datasets}"
_user="$(id -un 2>/dev/null || echo "${USER:-khdp-user}")"
if [ -d "$PERISIM_WORKSPACE/updown" ]; then
  : "${PERISIM_WORKDIR:=$PERISIM_WORKSPACE/updown/perisim/$_user/work}"
else
  : "${PERISIM_WORKDIR:=$REPO_DIR/work}"
fi
# Entorno y logs en disco local del contenedor (updown es almacenamiento de objetos compartido).
: "${PERISIM_LOCAL:=$HOME/.perisim}"
: "${PERISIM_VENV:=$PERISIM_LOCAL/venv}"
: "${PERISIM_CONDA_ENV:=$PERISIM_LOCAL/conda-py311}"
: "${PERISIM_PIP_INDEX:=http://pypi.khdp-forward-server/simple}"
: "${PERISIM_TORCH_INDEXES:=http://torch-download.khdp-forward-server/whl/cpu http://torch-download.khdp-forward-server/whl/cu126}"
: "${PERISIM_THREADS_PER_JOB:=4}"
export PERISIM_WORKSPACE PERISIM_DATASETS PERISIM_WORKDIR PERISIM_LOCAL

PIDFILE="$PERISIM_LOCAL/run.pid"
LOGDIR="$PERISIM_LOCAL/logs"
mkdir -p "$PERISIM_LOCAL" "$LOGDIR"

say()  { printf '\033[1;34m[perisim]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[perisim]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[perisim]\033[0m %s\n' "$*" >&2; exit 1; }

# --- hilos: como mucho 4 trabajos × 4 hilos (§9.1); nunca sobresuscribir ----------------
export OMP_NUM_THREADS="$PERISIM_THREADS_PER_JOB"
export MKL_NUM_THREADS="$PERISIM_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$PERISIM_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$PERISIM_THREADS_PER_JOB"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-16}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# OpenMP para LightGBM si el sistema no trae libgomp (ver fix_openmp)
[ -e "$PERISIM_LOCAL/lib/libgomp.so.1" ] && export LD_LIBRARY_PATH="$PERISIM_LOCAL/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

_version_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; }

find_python() {
  local c
  for c in ${PERISIM_PYTHON:-} python3.12 python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && _version_ok "$c"; then command -v "$c"; return 0; fi
  done
  if [ -x "$PERISIM_CONDA_ENV/bin/python" ] && _version_ok "$PERISIM_CONDA_ENV/bin/python"; then
    echo "$PERISIM_CONDA_ENV/bin/python"; return 0
  fi
  return 1
}

_reachable() { "$1" - "$2" <<'PY' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(sys.argv[1], timeout=8)
PY
}

_host_of() { echo "$1" | sed -E 's#^[a-z]+://([^/:]+).*#\1#'; }

setup_env() {
  local base_py vpy req_hash
  base_py="$(find_python || true)"
  if [ -z "$base_py" ]; then
    command -v conda >/dev/null 2>&1 || die "No hay Python >= 3.11 ni conda. Instala Python 3.11 o define PERISIM_PYTHON."
    say "No hay Python >= 3.11: creando entorno conda en $PERISIM_CONDA_ENV (mirror conda-forge de KHDP)"
    conda create -y -p "$PERISIM_CONDA_ENV" python=3.11 >/dev/null
    base_py="$PERISIM_CONDA_ENV/bin/python"
  fi
  say "Python base: $base_py ($("$base_py" -V 2>&1))"

  # Entorno: 1) venv (reutiliza paquetes del sistema) → 2) entorno conda (mirror de KHDP) → 3) pip --user
  local user_mode=0
  if [ ! -x "$PERISIM_VENV/bin/python" ]; then
    say "Creando entorno virtual en $PERISIM_VENV (reutiliza paquetes del sistema)"
    "$base_py" -m venv --system-site-packages "$PERISIM_VENV" >/dev/null 2>&1 || true
  fi
  if [ -x "$PERISIM_VENV/bin/python" ] && "$PERISIM_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    vpy="$PERISIM_VENV/bin/python"
  else
    rm -rf "$PERISIM_VENV"   # venv sin pip (falta ensurepip): no sirve
    if [ -x "$PERISIM_CONDA_ENV/bin/python" ] && "$PERISIM_CONDA_ENV/bin/python" -m pip --version >/dev/null 2>&1; then
      vpy="$PERISIM_CONDA_ENV/bin/python"
    elif command -v conda >/dev/null 2>&1; then
      warn "venv no disponible: creando entorno conda en $PERISIM_CONDA_ENV"
      conda create -y -q -p "$PERISIM_CONDA_ENV" python=3.11 pip >/dev/null
      vpy="$PERISIM_CONDA_ENV/bin/python"
    elif "$base_py" -m pip --version >/dev/null 2>&1; then
      warn "venv y conda no disponibles: instalación con pip --user"
      vpy="$base_py"; user_mode=1
    else
      die "No hay forma de crear un entorno (sin venv/ensurepip, conda ni pip). Define PERISIM_PYTHON=<python con pip>."
    fi
  fi

  # Índice de paquetes: mirror interno de KHDP si responde; si no, la configuración de pip por defecto.
  local pip_args=()
  if _reachable "$vpy" "$PERISIM_PIP_INDEX/"; then
    pip_args=(--index-url "$PERISIM_PIP_INDEX" --trusted-host "$(_host_of "$PERISIM_PIP_INDEX")")
    say "Usando el mirror PyPI de KHDP: $PERISIM_PIP_INDEX"
  else
    warn "Mirror de KHDP no accesible: se usa la configuración de pip por defecto"
  fi

  local pip_base=(--disable-pip-version-check -q)
  [ "$user_mode" = 1 ] && pip_base+=(--user)
  req_hash="$(cat "$REPO_DIR/requirements.txt" "$REPO_DIR/requirements-optional.txt" | sha256sum | cut -c1-16)"
  local marker="$PERISIM_LOCAL/req_$(printf '%s' "$vpy" | sha256sum | cut -c1-8)"
  if [ "$(cat "$marker" 2>/dev/null || true)" != "$req_hash" ]; then
    say "Instalando dependencias (requirements.txt)"
    "$vpy" -m pip install "${pip_base[@]}" ${pip_args[@]+"${pip_args[@]}"} -r "$REPO_DIR/requirements.txt"
    if ! "$vpy" -c "import torch" >/dev/null 2>&1; then
      local idx ok=0
      for idx in $PERISIM_TORCH_INDEXES; do
        if _reachable "$vpy" "$idx/torch/"; then
          say "Instalando torch desde $idx"
          if "$vpy" -m pip install "${pip_base[@]}" torch --index-url "$idx" \
               --trusted-host "$(_host_of "$idx")"; then ok=1; break; fi
        fi
      done
      if [ "$ok" = 0 ]; then
        warn "Índices de PyTorch no accesibles: se instala torch desde el índice general"
        "$vpy" -m pip install "${pip_base[@]}" ${pip_args[@]+"${pip_args[@]}"} torch
      fi
    fi
    "$vpy" -m pip install "${pip_base[@]}" ${pip_args[@]+"${pip_args[@]}"} -r "$REPO_DIR/requirements-optional.txt" \
      || warn "Dependencias opcionales no instaladas (no bloquea el pipeline)"
    echo "$req_hash" > "$marker"
  fi
  fix_openmp "$vpy"
  "$vpy" -m perisim env-check || die "La comprobación del entorno ha fallado (ver mensaje anterior)"
  VPY="$vpy"
  echo "$vpy" > "$PERISIM_LOCAL/python_path"
}

# LightGBM enlaza contra libgomp.so.1 del sistema. Sin root no se puede instalar con apt, pero las ruedas de
# scikit-learn / torch (o conda) ya traen una copia: se enlaza en $PERISIM_LOCAL/lib y se añade a LD_LIBRARY_PATH.
fix_openmp() {
  local py="$1" cand
  "$py" -c "import lightgbm" >/dev/null 2>&1 && return 0
  cand="$("$py" - <<'PY' 2>/dev/null
import glob, os, site, sys
dirs = list(site.getsitepackages()) + [site.getusersitepackages()]
pats = []
for sp in dirs:
    pats += sorted(glob.glob(os.path.join(sp, "scikit_learn.libs", "libgomp*.so*")))
    pats += sorted(glob.glob(os.path.join(sp, "torch", "lib", "libgomp*.so*")))
for d in (os.environ.get("CONDA_PREFIX", ""), sys.base_prefix, "/opt/conda"):
    if d:
        pats += sorted(glob.glob(os.path.join(d, "lib", "libgomp.so*")))
print(pats[0] if pats else "")
PY
)"
  if [ -z "$cand" ]; then
    warn "LightGBM no encuentra libgomp.so.1 y no hay copia local. Prueba: conda install -c conda-forge libgomp"
    return 0
  fi
  mkdir -p "$PERISIM_LOCAL/lib"
  ln -sf "$cand" "$PERISIM_LOCAL/lib/libgomp.so.1"
  export LD_LIBRARY_PATH="$PERISIM_LOCAL/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if "$py" -c "import lightgbm" >/dev/null 2>&1; then
    say "OpenMP: se usa $cand para LightGBM"
  else
    warn "LightGBM sigue sin cargar con $cand"
  fi
}

is_running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

maybe_smoke() {
  [ "${PERISIM_SKIP_SMOKE:-0}" = 1 ] && return 0
  local key; key="$("$VPY" -m perisim code-hash)"
  if [ -f "$PERISIM_LOCAL/smoke_ok_$key" ]; then return 0; fi
  say "Primera ejecución de esta versión del código: smoke test con datos sintéticos (unos minutos)"
  "$VPY" -m perisim smoke --workdir "$PERISIM_LOCAL/smoke" || die "El smoke test ha fallado: no se lanza el pipeline real."
  touch "$PERISIM_LOCAL/smoke_ok_$key"
}

launch_bg() {
  is_running && die "Ya hay una ejecución en marcha (PID $(cat "$PIDFILE")). Usa: bash run_pod.sh logs | stop"
  local log="$LOGDIR/run_$(date +%Y%m%d_%H%M%S).log"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$VPY" -m perisim run "$@" >"$log" 2>&1 </dev/null &
  else
    nohup "$VPY" -m perisim run "$@" >"$log" 2>&1 </dev/null &
  fi
  echo $! > "$PIDFILE"
  echo "$log" > "$LOGDIR/latest.txt"
  say "Pipeline lanzado en segundo plano (PID $!). Log: $log"
  say "Ctrl-C deja de seguir el log, pero NO detiene el pipeline. 'bash run_pod.sh stop' para pararlo."
  sleep 1
  tail -n +1 -F "$log"
}

case "$CMD" in
  start|run)
    setup_env; maybe_smoke; launch_bg "$@" ;;
  fg)
    setup_env; maybe_smoke; exec "$VPY" -m perisim run "$@" ;;
  setup)
    setup_env; say "Entorno listo: $VPY" ;;
  smoke)
    setup_env; exec "$VPY" -m perisim smoke --workdir "$PERISIM_LOCAL/smoke" "$@" ;;
  test)
    setup_env; cd "$REPO_DIR"; exec "$VPY" -m pytest "$@" ;;
  status)
    if is_running; then say "En marcha (PID $(cat "$PIDFILE"))"; else say "No hay ninguna ejecución en marcha"; fi
    py="$(cat "$PERISIM_LOCAL/python_path" 2>/dev/null || true)"
    [ -n "$py" ] && [ -x "$py" ] && "$py" -m perisim status || true ;;
  logs)
    [ -f "$LOGDIR/latest.txt" ] || die "No hay logs todavía"
    tail -n 200 -F "$(cat "$LOGDIR/latest.txt")" ;;
  stop)
    if is_running; then
      pid="$(cat "$PIDFILE")"; kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"; say "Detenido (PID $pid)"
    else say "No hay ninguna ejecución en marcha"; fi ;;
  freeze)
    setup_env; exec "$VPY" -m perisim freeze "$@" ;;
  *)
    die "Comando desconocido: $CMD (start | fg | status | logs | stop | smoke | test | freeze | setup)" ;;
esac
