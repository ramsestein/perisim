#!/usr/bin/env bash
# =============================================================================
# PERISIM — single entry point in the KHDP pod. Run it from the workspace root (~/workspace, where perisim/,
# data/ and venv/ live); from any other folder it works the same.
#
#   bash perisim/run_pod.sh            # set up the environment and launch the WHOLE pipeline in the background
#   bash perisim/run_pod.sh fg         # same, in the foreground
#   bash perisim/run_pod.sh status     # status of every stage and whether a run is in progress
#   bash perisim/run_pod.sh logs       # follow the log of the latest run (Ctrl-C does not stop it)
#   bash perisim/run_pod.sh stop       # stop the run in progress
#   bash perisim/run_pod.sh smoke      # end-to-end test on synthetic data
#   bash perisim/run_pod.sh test       # pytest (includes the anti-leakage tests on the real data)
#   bash perisim/run_pod.sh freeze --registration <OSF_ID>   # §6.3 freeze (enables the external validation)
#   bash perisim/run_pod.sh resources  # real CPU/memory limits of the container and the jobs × threads plan
#   bash perisim/run_pod.sh setup      # only install / check the environment
#
# Any extra argument after start/fg is passed to `python -m perisim run`
# (e.g. `bash perisim/run_pod.sh fg --until audit`, `--stage features --force`).
#
# Environment variables (optional):
#   PERISIM_EXPERIMENT  t2 (default, primary target) | t1 | t3 — one experiment per target, each with its own freeze;
#                       the three share the extraction (e.g. `PERISIM_EXPERIMENT=t1 bash perisim/run_pod.sh`)
#   PERISIM_WORKDIR     working folder (default: <workspace>/perisim_work, local disk)
#   PERISIM_PUBLISH_DIR where every run copies its reports and export package (default: updown/perisim/<user>/results;
#                       empty = no copy)
#   PERISIM_WORKSPACE   KHDP workspace (default: the first of ~/workspace, /home/khdp-user/workspace or the folders
#                       above the repository that contains data/, datasets/ or updown/)
#   PERISIM_DATASETS    datasets root (default: <workspace>/data, one folder per cohort, if it exists; otherwise the
#                       KHDP mount <workspace>/datasets)
#   PERISIM_VENV        Python environment (default: <workspace>/venv; one created by the user is used as it is)
#   PERISIM_LOCAL       logs, PID and smoke-test files of the runs (default: <workspace>/perisim_run)
#   PERISIM_INSPIRE_DIR / PERISIM_MIMIC_DIR / PERISIM_EICU_DIR  explicit paths (otherwise auto)
#   PERISIM_JOBS / PERISIM_THREADS  force jobs × threads (default: auto from the container limits)
#   PERISIM_ACK_AUDIT=true   continue after reviewing the completeness audit (§7)
#   PERISIM_SKIP_SMOKE=1     do not run the smoke test before the first real run
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# How the user calls this script from where they work, for the hints in messages and reports
# (`bash perisim/run_pod.sh` from the workspace root, `bash run_pod.sh` inside the repository).
case "$REPO_DIR" in
  "$PWD") PERISIM_RUN="bash run_pod.sh" ;;
  "$PWD"/*) PERISIM_RUN="bash ${REPO_DIR#"$PWD"/}/run_pod.sh" ;;
  *) PERISIM_RUN="bash $REPO_DIR/run_pod.sh" ;;
esac
export PERISIM_RUN
CMD="${1:-start}"
[ $# -gt 0 ] && shift || true

# Workspace: the KHDP guide says /home/khdp-user/workspace, but the pod user may differ (e.g. coder): use the first
# candidate that has data/, datasets/ or updown/ (the user's home, the guide's path, or the folders above the repository).
if [ -z "${PERISIM_WORKSPACE:-}" ]; then
  for _cand in "$HOME/workspace" /home/khdp-user/workspace "$(dirname "$REPO_DIR")" "$(dirname "$(dirname "$REPO_DIR")")"; do
    if [ -d "$_cand/data" ] || [ -d "$_cand/datasets" ] || [ -d "$_cand/updown" ]; then PERISIM_WORKSPACE="$_cand"; break; fi
  done
  : "${PERISIM_WORKSPACE:=/home/khdp-user/workspace}"
fi
# Datasets: the user's own copy in <workspace>/data (unzipped outside updown, which is slow object storage) if it
# exists; otherwise the KHDP mount <workspace>/datasets.
if [ -z "${PERISIM_DATASETS:-}" ]; then
  if [ -d "$PERISIM_WORKSPACE/data" ]; then PERISIM_DATASETS="$PERISIM_WORKSPACE/data"
  else PERISIM_DATASETS="$PERISIM_WORKSPACE/datasets"; fi
fi
_user="$(id -un 2>/dev/null || echo "${USER:-khdp-user}")"
# Work on the workspace's local disk: updown (shared object storage) loses files that are being written and renamed
# by several processes (first real run, 2026-10-24). Every run then copies its reports and export package (aggregates
# only) to updown, which is persistent and shared.
if [ -d "$PERISIM_WORKSPACE" ]; then
  : "${PERISIM_WORKDIR:=$PERISIM_WORKSPACE/perisim_work}"
else
  : "${PERISIM_WORKDIR:=$REPO_DIR/work}"
fi
if [ -d "$PERISIM_WORKSPACE/updown" ]; then
  : "${PERISIM_PUBLISH_DIR=$PERISIM_WORKSPACE/updown/perisim/$_user/results}"
fi
# Environment and run files in the workspace, outside updown (shared object storage, slow for the many small files
# of an environment) and without hidden folders. Without a workspace (local development): next to the repository.
_rundir="$PERISIM_WORKSPACE"; [ -d "$_rundir" ] || _rundir="$REPO_DIR"
: "${PERISIM_LOCAL:=$_rundir/perisim_run}"
: "${PERISIM_VENV:=$_rundir/venv}"
: "${PERISIM_CONDA_ENV:=$PERISIM_LOCAL/conda-py311}"
: "${PERISIM_PIP_INDEX:=http://pypi.khdp-forward-server/simple}"
: "${PERISIM_TORCH_INDEXES:=http://torch-download.khdp-forward-server/whl/cpu http://torch-download.khdp-forward-server/whl/cu126}"
: "${PERISIM_EXPERIMENT:=t2}"
case "$PERISIM_EXPERIMENT" in t1|t2|t3) ;; *) echo "PERISIM_EXPERIMENT must be t1, t2 or t3" >&2; exit 1 ;; esac
PERISIM_PUBLISH_DIR="${PERISIM_PUBLISH_DIR:-}"
export PERISIM_WORKSPACE PERISIM_DATASETS PERISIM_WORKDIR PERISIM_LOCAL PERISIM_EXPERIMENT PERISIM_PUBLISH_DIR
# The KHDP forward servers (PyPI, PyTorch, conda-forge, GitHub mirrors) resolve inside the pod but not in the pod's
# HTTP proxy (squid answers 503 ERR_DNS_FAIL): they are always reached directly.
_np="${no_proxy:-${NO_PROXY:-}}"
case ",$_np," in *",.khdp-forward-server,"*) ;; *) _np="${_np:+$_np,}.khdp-forward-server" ;; esac
export no_proxy="$_np" NO_PROXY="$_np"

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PERISIM_LOCAL/matplotlib}"   # matplotlib's cache, not in ~/.cache
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$PERISIM_LOCAL/cuda_cache}"   # CUDA's JIT cache (torch), not in ~/.nv

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

# No environment yet: 1) venv (reuses system packages) → 2) conda environment (KHDP mirror) → 3) pip --user.
# Sets `vpy` and `user_mode` of the caller (setup_env).
_create_env() {
  local base_py created=0
  base_py="$(find_python || true)"
  if [ -z "$base_py" ]; then
    command -v conda >/dev/null 2>&1 || die "No Python >= 3.11 and no conda. Install Python 3.11 or set PERISIM_PYTHON."
    say "No Python >= 3.11: creating a conda environment in $PERISIM_CONDA_ENV (KHDP conda-forge mirror)"
    conda create -y -p "$PERISIM_CONDA_ENV" python=3.11 >/dev/null
    base_py="$PERISIM_CONDA_ENV/bin/python"
  fi
  say "Base Python: $base_py ($("$base_py" -V 2>&1))"
  [ -e "$PERISIM_VENV" ] || created=1
  say "Creating a virtual environment in $PERISIM_VENV (reuses system packages)"
  "$base_py" -m venv --system-site-packages "$PERISIM_VENV" >/dev/null 2>&1 || true
  if [ -x "$PERISIM_VENV/bin/python" ] && "$PERISIM_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    vpy="$PERISIM_VENV/bin/python"
    return 0
  fi
  if [ "$created" = 1 ]; then rm -rf "$PERISIM_VENV"; fi   # venv without pip (ensurepip missing): unusable
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
}

setup_env() {
  local vpy req_hash user_mode=0
  if [ -x "$PERISIM_VENV/bin/python" ]; then
    # An existing environment (e.g. created by the user, venv or conda) is used as it is: never recreated or deleted.
    vpy="$PERISIM_VENV/bin/python"
    _version_ok "$vpy" || die "$PERISIM_VENV has $("$vpy" -V 2>&1); PERISIM needs Python >= 3.11. Recreate it with Python 3.11+."
    "$vpy" -m pip --version >/dev/null 2>&1 || die "$PERISIM_VENV has no pip. Add it ($vpy -m ensurepip) or recreate the environment."
    say "Environment: $PERISIM_VENV ($("$vpy" -V 2>&1))"
  else
    _create_env
  fi

  # Package index: KHDP internal mirror if it answers; otherwise pip's default configuration.
  local pip_args=()
  if _reachable "$vpy" "$PERISIM_PIP_INDEX/"; then
    pip_args=(--index-url "$PERISIM_PIP_INDEX" --trusted-host "$(_host_of "$PERISIM_PIP_INDEX")")
    say "Using the KHDP PyPI mirror: $PERISIM_PIP_INDEX"
  else
    warn "KHDP mirror not reachable: using pip's default configuration"
  fi

  local pip_base=(--disable-pip-version-check --no-cache-dir -q)   # no cache in ~/.cache
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
          # the torch index may not host torch's own dependencies: they come from the PyPI mirror
          local extra=()
          [ ${#pip_args[@]} -gt 0 ] && extra=(--extra-index-url "$PERISIM_PIP_INDEX" --trusted-host "$(_host_of "$PERISIM_PIP_INDEX")")
          if "$vpy" -m pip install "${pip_base[@]}" torch --index-url "$idx" --trusted-host "$(_host_of "$idx")" \
               ${extra[@]+"${extra[@]}"}; then ok=1; break; fi
        fi
      done
      if [ "$ok" = 0 ]; then
        warn "PyTorch indexes not reachable: installing torch from the general index (on Linux this may pull ~3 GB of CUDA wheels)"
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
  say "Resources: $PERISIM_CPUS usable CPUs → $PERISIM_THREADS_PER_JOB threads per job (details: $PERISIM_RUN resources)"
}

# A PID file is trusted only if that PID is alive AND is a perisim process (PIDs are reused after a restart)
is_running() {
  [ -f "$PIDFILE" ] || return 1
  local pid; pid="$(cat "$PIDFILE")"
  kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o args= 2>/dev/null | grep -q "perisim" || return 1
}
refuse_if_running() {
  if is_running; then die "A run is already in progress (PID $(cat "$PIDFILE")). Use: $PERISIM_RUN logs | stop"; fi
  return 0
}

maybe_smoke() {
  [ "${PERISIM_SKIP_SMOKE:-0}" = 1 ] && return 0
  local key; key="$("$VPY" -m perisim code-hash)"
  if [ -f "$PERISIM_LOCAL/smoke_ok_$key" ]; then return 0; fi
  say "First run of this code version: smoke test on synthetic data (a few minutes)"
  "$VPY" -m perisim smoke --workdir "$PERISIM_LOCAL/smoke" || die "The smoke test failed: the real pipeline is not launched."
  touch "$PERISIM_LOCAL/smoke_ok_$key"
}

launch_bg() {
  refuse_if_running
  local log="$LOGDIR/run_$(date +%Y%m%d_%H%M%S).log"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$VPY" -m perisim run "$@" >"$log" 2>&1 </dev/null &
  else
    nohup "$VPY" -m perisim run "$@" >"$log" 2>&1 </dev/null &
  fi
  echo $! > "$PIDFILE"
  echo "$log" > "$LOGDIR/latest.txt"
  say "Pipeline launched in the background (experiment $PERISIM_EXPERIMENT, PID $!). Log: $log"
  say "Ctrl-C stops following the log but does NOT stop the pipeline. Use '$PERISIM_RUN stop' to stop it."
  sleep 1
  tail -n +1 -F "$log"
}

case "$CMD" in
  start|run)
    refuse_if_running; setup_env; maybe_smoke; launch_bg "$@" ;;
  fg)
    refuse_if_running; setup_env; maybe_smoke; echo $$ > "$PIDFILE"; exec "$VPY" -m perisim run "$@" ;;
  setup)
    setup_env; say "Environment ready: $VPY" ;;
  smoke)
    setup_env; exec "$VPY" -m perisim smoke --workdir "$PERISIM_LOCAL/smoke" "$@" ;;
  test)
    setup_env; cd "$REPO_DIR"; export PERISIM_TEST_WORKDIR="$PERISIM_WORKDIR/experiments/$PERISIM_EXPERIMENT"
    export PERISIM_TEST_BASE="$PERISIM_WORKDIR"; exec "$VPY" -m pytest -p no:cacheprovider "$@" ;;
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
      pid="$(cat "$PIDFILE")"; kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"; rm -f "$PIDFILE"; say "Stopped (PID $pid)"
    else rm -f "$PIDFILE"; say "No run in progress"; fi ;;
  freeze)
    setup_env; exec "$VPY" -m perisim freeze "$@" ;;
  *)
    die "Unknown command: $CMD (start | fg | status | logs | stop | smoke | test | freeze | resources | setup)" ;;
esac
