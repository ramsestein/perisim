#!/usr/bin/env bash
# =============================================================================
# PERISIM — single entry point in the KHDP pod.
#
#   bash run_pod.sh            # set up the environment and launch the WHOLE pipeline in the background
#   bash run_pod.sh fg         # same, in the foreground
#   bash run_pod.sh status     # status of every stage and whether a run is in progress
#   bash run_pod.sh logs       # follow the log of the latest run (Ctrl-C does not stop it)
#   bash run_pod.sh stop       # stop the run in progress
#   bash run_pod.sh smoke      # end-to-end test on synthetic data
#   bash run_pod.sh test       # pytest (includes the anti-leakage tests on the real data)
#   bash run_pod.sh freeze --registration <OSF_ID>   # §6.3 freeze (enables the external validation)
#   bash run_pod.sh resources  # real CPU/memory limits of the container and the jobs × threads plan
#   bash run_pod.sh setup      # only install / check the environment
#
# Any extra argument after start/fg is passed to `python -m perisim run`
# (e.g. `bash run_pod.sh fg --until audit`, `--stage features --force`).
#
# Environment variables (optional):
#   PERISIM_WORKDIR     working folder (default: updown/perisim/<user>/work)
#   PERISIM_DATASETS    datasets root (default: /home/khdp-user/workspace/datasets)
#   PERISIM_INSPIRE_DIR / PERISIM_MIMIC_DIR / PERISIM_EICU_DIR  explicit paths (otherwise auto)
#   PERISIM_JOBS / PERISIM_THREADS  force jobs × threads (default: auto from the container limits)
#   PERISIM_ACK_AUDIT=true   continue after reviewing the completeness audit (§7)
#   PERISIM_SKIP_SMOKE=1     do not run the smoke test before the first real run
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
# Environment and terminal logs on the container's local disk (updown is shared object storage).
: "${PERISIM_LOCAL:=$HOME/.perisim}"
: "${PERISIM_VENV:=$PERISIM_LOCAL/venv}"
: "${PERISIM_CONDA_ENV:=$PERISIM_LOCAL/conda-py311}"
: "${PERISIM_PIP_INDEX:=http://pypi.khdp-forward-server/simple}"
: "${PERISIM_TORCH_INDEXES:=http://torch-download.khdp-forward-server/whl/cpu http://torch-download.khdp-forward-server/whl/cu126}"
export PERISIM_WORKSPACE PERISIM_DATASETS PERISIM_WORKDIR PERISIM_LOCAL

PIDFILE="$PERISIM_LOCAL/run.pid"
LOGDIR="$PERISIM_LOCAL/logs"
mkdir -p "$PERISIM_LOCAL" "$LOGDIR"

say()  { printf '\033[1;34m[perisim]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[perisim]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[perisim]\033[0m %s\n' "$*" >&2; exit 1; }

# --- threads: conservative defaults until the real container limits are known (size_threads) -------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# OpenMP for LightGBM when the system has no libgomp (see fix_openmp)
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
    command -v conda >/dev/null 2>&1 || die "No Python >= 3.11 and no conda. Install Python 3.11 or set PERISIM_PYTHON."
    say "No Python >= 3.11: creating a conda environment in $PERISIM_CONDA_ENV (KHDP conda-forge mirror)"
    conda create -y -p "$PERISIM_CONDA_ENV" python=3.11 >/dev/null
    base_py="$PERISIM_CONDA_ENV/bin/python"
  fi
  say "Base Python: $base_py ($("$base_py" -V 2>&1))"

  # Environment: 1) venv (reuses system packages) → 2) conda environment (KHDP mirror) → 3) pip --user
  local user_mode=0
  if [ ! -x "$PERISIM_VENV/bin/python" ]; then
    say "Creating a virtual environment in $PERISIM_VENV (reuses system packages)"
    "$base_py" -m venv --system-site-packages "$PERISIM_VENV" >/dev/null 2>&1 || true
  fi
  if [ -x "$PERISIM_VENV/bin/python" ] && "$PERISIM_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    vpy="$PERISIM_VENV/bin/python"
  else
    rm -rf "$PERISIM_VENV"   # venv without pip (ensurepip missing): unusable
    if [ -x "$PERISIM_CONDA_ENV/bin/python" ] && "$PERISIM_CONDA_ENV/bin/python" -m pip --version >/dev/null 2>&1; then
      vpy="$PERISIM_CONDA_ENV/bin/python"
    elif command -v conda >/dev/null 2>&1; then
      warn "venv not available: creating a conda environment in $PERISIM_CONDA_ENV"
      conda create -y -q -p "$PERISIM_CONDA_ENV" python=3.11 pip >/dev/null
      vpy="$PERISIM_CONDA_ENV/bin/python"
    elif "$base_py" -m pip --version >/dev/null 2>&1; then
      warn "venv and conda not available: installing with pip --user"
      vpy="$base_py"; user_mode=1
    else
      die "Cannot create an environment (no venv/ensurepip, conda or pip). Set PERISIM_PYTHON=<python with pip>."
    fi
  fi

  # Package index: KHDP internal mirror if it answers; otherwise pip's default configuration.
  local pip_args=()
  if _reachable "$vpy" "$PERISIM_PIP_INDEX/"; then
    pip_args=(--index-url "$PERISIM_PIP_INDEX" --trusted-host "$(_host_of "$PERISIM_PIP_INDEX")")
    say "Using the KHDP PyPI mirror: $PERISIM_PIP_INDEX"
  else
    warn "KHDP mirror not reachable: using pip's default configuration"
  fi

  local pip_base=(--disable-pip-version-check -q)
  [ "$user_mode" = 1 ] && pip_base+=(--user)
  req_hash="$(cat "$REPO_DIR/requirements.txt" "$REPO_DIR/requirements-optional.txt" | sha256sum | cut -c1-16)"
  local marker="$PERISIM_LOCAL/req_$(printf '%s' "$vpy" | sha256sum | cut -c1-8)"
  if [ "$(cat "$marker" 2>/dev/null || true)" != "$req_hash" ]; then
    say "Installing dependencies (requirements.txt)"
    "$vpy" -m pip install "${pip_base[@]}" ${pip_args[@]+"${pip_args[@]}"} -r "$REPO_DIR/requirements.txt"
    if ! "$vpy" -c "import torch" >/dev/null 2>&1; then
      local idx ok=0
      for idx in $PERISIM_TORCH_INDEXES; do
        if _reachable "$vpy" "$idx/torch/"; then
          say "Installing torch from $idx"
          if "$vpy" -m pip install "${pip_base[@]}" torch --index-url "$idx" \
               --trusted-host "$(_host_of "$idx")"; then ok=1; break; fi
        fi
      done
      if [ "$ok" = 0 ]; then
        warn "PyTorch indexes not reachable: installing torch from the general index"
        "$vpy" -m pip install "${pip_base[@]}" ${pip_args[@]+"${pip_args[@]}"} torch
      fi
    fi
    "$vpy" -m pip install "${pip_base[@]}" ${pip_args[@]+"${pip_args[@]}"} -r "$REPO_DIR/requirements-optional.txt" \
      || warn "Optional dependencies not installed (does not block the pipeline)"
    echo "$req_hash" > "$marker"
  fi
  fix_openmp "$vpy"
  "$vpy" -m perisim env-check || die "The environment check failed (see the message above)"
  VPY="$vpy"
  echo "$vpy" > "$PERISIM_LOCAL/python_path"
  size_threads
}

# LightGBM links against the system's libgomp.so.1. Without root it cannot be installed with apt, but the
# scikit-learn / torch wheels (or conda) already ship a copy: it is linked in $PERISIM_LOCAL/lib and added to
# LD_LIBRARY_PATH.
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
    warn "LightGBM cannot find libgomp.so.1 and there is no local copy. Try: conda install -c conda-forge libgomp"
    return 0
  fi
  mkdir -p "$PERISIM_LOCAL/lib"
  ln -sf "$cand" "$PERISIM_LOCAL/lib/libgomp.so.1"
  export LD_LIBRARY_PATH="$PERISIM_LOCAL/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if "$py" -c "import lightgbm" >/dev/null 2>&1; then
    say "OpenMP: using $cand for LightGBM"
  else
    warn "LightGBM still does not load with $cand"
  fi
}

# Threads from the container's REAL limits (cgroup quota / affinity), not from the host's core count that
# os.cpu_count() reports inside a container (§9.1: at most 4 jobs × 4 threads).
size_threads() {
  eval "$("$VPY" -m perisim resources --shell)"
  export OMP_NUM_THREADS="$PERISIM_THREADS_PER_JOB" MKL_NUM_THREADS="$PERISIM_THREADS_PER_JOB"
  export OPENBLAS_NUM_THREADS="$PERISIM_THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$PERISIM_THREADS_PER_JOB"
  export POLARS_MAX_THREADS
  say "Resources: $PERISIM_CPUS usable CPUs → $PERISIM_THREADS_PER_JOB threads per job (details: bash run_pod.sh resources)"
}

is_running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

maybe_smoke() {
  [ "${PERISIM_SKIP_SMOKE:-0}" = 1 ] && return 0
  local key; key="$("$VPY" -m perisim code-hash)"
  if [ -f "$PERISIM_LOCAL/smoke_ok_$key" ]; then return 0; fi
  say "First run of this code version: smoke test on synthetic data (a few minutes)"
  "$VPY" -m perisim smoke --workdir "$PERISIM_LOCAL/smoke" || die "The smoke test failed: the real pipeline is not launched."
  touch "$PERISIM_LOCAL/smoke_ok_$key"
}

launch_bg() {
  is_running && die "A run is already in progress (PID $(cat "$PIDFILE")). Use: bash run_pod.sh logs | stop"
  local log="$LOGDIR/run_$(date +%Y%m%d_%H%M%S).log"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$VPY" -m perisim run "$@" >"$log" 2>&1 </dev/null &
  else
    nohup "$VPY" -m perisim run "$@" >"$log" 2>&1 </dev/null &
  fi
  echo $! > "$PIDFILE"
  echo "$log" > "$LOGDIR/latest.txt"
  say "Pipeline launched in the background (PID $!). Log: $log"
  say "Ctrl-C stops following the log but does NOT stop the pipeline. Use 'bash run_pod.sh stop' to stop it."
  sleep 1
  tail -n +1 -F "$log"
}

case "$CMD" in
  start|run)
    setup_env; maybe_smoke; launch_bg "$@" ;;
  fg)
    setup_env; maybe_smoke; exec "$VPY" -m perisim run "$@" ;;
  setup)
    setup_env; say "Environment ready: $VPY" ;;
  smoke)
    setup_env; exec "$VPY" -m perisim smoke --workdir "$PERISIM_LOCAL/smoke" "$@" ;;
  test)
    setup_env; cd "$REPO_DIR"; exec "$VPY" -m pytest "$@" ;;
  resources)
    setup_env; exec "$VPY" -m perisim resources ;;
  status)
    if is_running; then say "Running (PID $(cat "$PIDFILE"))"; else say "No run in progress"; fi
    py="$(cat "$PERISIM_LOCAL/python_path" 2>/dev/null || true)"
    [ -n "$py" ] && [ -x "$py" ] && "$py" -m perisim status || true ;;
  logs)
    [ -f "$LOGDIR/latest.txt" ] || die "No logs yet"
    tail -n 200 -F "$(cat "$LOGDIR/latest.txt")" ;;
  stop)
    if is_running; then
      pid="$(cat "$PIDFILE")"; kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"; say "Stopped (PID $pid)"
    else say "No run in progress"; fi ;;
  freeze)
    setup_env; exec "$VPY" -m perisim freeze "$@" ;;
  *)
    die "Unknown command: $CMD (start | fg | status | logs | stop | smoke | test | freeze | resources | setup)" ;;
esac
