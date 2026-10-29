<#
PERISIM - local entry point on Windows (the KHDP pod uses run_pod.sh).

  powershell -ExecutionPolicy Bypass -File run_local.ps1 [start|fg|status|logs|stop|smoke|test|ksweep|freeze|posthoc|drift|drift-missing|case-base|hypo|resources] [args]

  start      smoke test the first time a code version runs, then the WHOLE pipeline in the background
  fg         the same in the foreground
  status     stages done, and whether a run is in progress
  logs       follow the log of the latest run (Ctrl-C does not stop it)
  stop       stop the run in progress (the whole process tree)
  smoke      end-to-end test on synthetic data
  test       pytest
  ksweep     secondary k sweep of the finished cross-validation (minutes; aggregates in results\ksweep)
  freeze     S6.3 freeze (e.g. freeze --registration <OSF_ID>)
  posthoc    exploratory, after transport: PERISIM vs LightGBM / logistic regression transported, recalibrated and
             refitted locally, and the shape functions by site (scripts\posthoc_transport.py; --force refits)
  drift      exploratory, after transport: ICE and re-deformation of every variable of the frozen INSPIRE metric
             with each site's case base (scripts\posthoc_drift.py; about 2 h, resumable; --force refits)
  drift-missing  exploratory, after drift: only the points of the missing values re-deformed, all at once
             (scripts\posthoc_drift_missing.py; resumable; --force refits)
  case-base  exploratory: how many local patients the frozen metric needs (dense accumulation), and whether the
             eICU neighbours of the same hospital carry relations the other hospitals' do not (scripts\posthoc_case_base.py)
  hypo       experiment `hypo` (hypotension in the next 6 h, docs\design_hypotension.md): python -m hypo <command>,
             e.g. `hypo landmarks` builds the landmark tables from the original tables, `hypo smoke --external eicu`
             runs the whole model on the other two sites (Phase 1), `hypo cv` the internal validation by
             rotation (Phase 2, resumable; `hypo cv-report` rebuilds its report), `hypo final` the frozen
             artefacts and `hypo freeze --registration <ID>` the freeze (Phase 3), `hypo external` the external
             stage on the frozen version (Phase 4, resumable; `hypo external-report` rebuilds its report)
Any extra argument after start / fg goes to `python -m perisim run` (e.g. `fg --until audit`).

The data stay on this computer (PhysioNet data use agreement): everything lives in the project folder, in folders
git ignores (the repository is public), and outside OneDrive; the script refuses otherwise.
Defaults, overridable with the same variables as run_pod.sh:
  PERISIM_DATASETS    <repo>\data             (one folder per cohort; found by their tables)
  PERISIM_WORKDIR     <repo>\work             (work and results)
  PERISIM_LOCAL       <repo>\perisim_run      (logs, PID, smoke test)
  PERISIM_EXPERIMENT  t2
  PERISIM_JOBS / PERISIM_THREADS  5 / 4       (the 5 outer folds in one round)
  PERISIM_PYTHON      python
  PERISIM_PROFILE     local                   (config\profiles\local.yaml: what differs from the pod; "" = none)
#>
param([string]$Command = "start", [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @())
$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Set-Default([string]$Name, [string]$Value) {
    if (-not [Environment]::GetEnvironmentVariable($Name, "Process")) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}
function Say([string]$Text) { Write-Host "[perisim] $Text" -ForegroundColor Cyan }
function Die([string]$Text) { Write-Host "[perisim] $Text" -ForegroundColor Red; exit 1 }

Set-Default "PERISIM_DATASETS" (Join-Path $RepoDir "data")          # all three git-ignored (/data/, /work/, /perisim_run/)
Set-Default "PERISIM_WORKDIR" (Join-Path $RepoDir "work")
Set-Default "PERISIM_LOCAL" (Join-Path $RepoDir "perisim_run")
Set-Default "PERISIM_EXPERIMENT" "t2"
Set-Default "PERISIM_JOBS" "5"
Set-Default "PERISIM_THREADS" "4"
Set-Default "PERISIM_PYTHON" "python"
if ($null -eq [Environment]::GetEnvironmentVariable("PERISIM_PROFILE", "Process")) { $env:PERISIM_PROFILE = "local" }
$ProfArgs = @()
if ($env:PERISIM_PROFILE) { $ProfArgs = @("--profile", $env:PERISIM_PROFILE) }   # run, status and freeze alike
if ("t1", "t2", "t3" -notcontains $env:PERISIM_EXPERIMENT) { Die "PERISIM_EXPERIMENT must be t1, t2 or t3" }

# git matches directory patterns (/work/, /perisim_run/) only for folders that exist: create them before checking
foreach ($p in @($env:PERISIM_WORKDIR, $env:PERISIM_LOCAL)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
# Data governance: never inside the public repository nor inside a cloud-synchronised folder
foreach ($p in @($env:PERISIM_DATASETS, $env:PERISIM_WORKDIR, $env:PERISIM_LOCAL)) {
    $full = [IO.Path]::GetFullPath($p)
    if ($full.StartsWith($RepoDir, [StringComparison]::OrdinalIgnoreCase)) {   # only a git-ignored folder (e.g. data\)
        & git -C $RepoDir check-ignore -q -- $full 2>$null
        if ($LASTEXITCODE -ne 0) { Die "$full is inside the repository (public) and not ignored by git: use a folder outside it" }
    }
    foreach ($cloud in @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial)) {
        if ($cloud -and $full.StartsWith($cloud, [StringComparison]::OrdinalIgnoreCase)) { Die "$full is inside OneDrive ($cloud): the data must not be synchronised" }
    }
}

$env:PERISIM_PUBLISH_DIR = ""                 # no shared storage here: the work folder is the only copy
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"; $env:PYTHONUNBUFFERED = "1"
$env:MPLBACKEND = "Agg"; $env:MPLCONFIGDIR = Join-Path $env:PERISIM_LOCAL "matplotlib"
$env:PYTHONPATH = Join-Path $RepoDir "src"
$env:PERISIM_RUN = "powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`""
foreach ($v in "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS") {
    [Environment]::SetEnvironmentVariable($v, $env:PERISIM_THREADS, "Process")
}
$env:POLARS_MAX_THREADS = [string]([int]$env:PERISIM_JOBS * [int]$env:PERISIM_THREADS)

$Py = $env:PERISIM_PYTHON
$LogDir = Join-Path $env:PERISIM_LOCAL "logs"
$PidFile = Join-Path $env:PERISIM_LOCAL "run.pid"
New-Item -ItemType Directory -Force -Path $env:PERISIM_LOCAL, $LogDir, $env:PERISIM_WORKDIR | Out-Null

function Test-Running {
    if (-not (Test-Path $PidFile)) { return $false }
    $id = [int](Get-Content $PidFile -Raw)
    return [bool](Get-Process -Id $id -ErrorAction SilentlyContinue)
}
function Assert-NotRunning {
    if (Test-Running) { Die "A run is already in progress (PID $(Get-Content $PidFile -Raw)). Use: logs | stop" }
}
function Invoke-Setup {
    & $Py -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) { Die "Python >= 3.11 is required (set PERISIM_PYTHON)" }
    & $Py -m perisim env-check
    if ($LASTEXITCODE -ne 0) { Die "The environment check failed (pip install -r requirements.txt)" }
}
function Invoke-SmokeIfNew {
    if ($env:PERISIM_SKIP_SMOKE -eq "1") { return }
    $key = (& $Py -m perisim code-hash).Trim()
    $marker = Join-Path $env:PERISIM_LOCAL "smoke_ok_$key"
    if (Test-Path $marker) { return }
    Say "First run of this code version: smoke test on synthetic data (a few minutes)"
    & $Py -m perisim smoke --workdir (Join-Path $env:PERISIM_LOCAL "smoke")
    if ($LASTEXITCODE -ne 0) { Die "The smoke test failed: the real pipeline is not launched." }
    New-Item -ItemType File -Force -Path $marker | Out-Null
}

Say "experiment $env:PERISIM_EXPERIMENT | data $env:PERISIM_DATASETS | work $env:PERISIM_WORKDIR | $env:PERISIM_JOBS jobs x $env:PERISIM_THREADS threads"
switch ($Command) {
    { $_ -in "start", "run" } {
        Assert-NotRunning; Invoke-Setup; Invoke-SmokeIfNew
        $log = Join-Path $LogDir ("run_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
        # Python directly, not through cmd.exe: cmd cannot open paths with non-ASCII characters (e.g. C:\Users\Ramses
        # with an accent). stdout = the pipeline log; stderr (tracebacks of a crash, warnings) next to it.
        $p = Start-Process -FilePath $Py -ArgumentList (@("-m", "perisim", "run") + $ProfArgs + $Rest) -WindowStyle Hidden `
            -RedirectStandardOutput $log -RedirectStandardError ($log -replace "\.log$", ".err.log") -PassThru
        Set-Content -Path $PidFile -Value $p.Id
        Set-Content -Path (Join-Path $LogDir "latest.txt") -Value $log
        Start-Sleep -Seconds 3
        if ($p.HasExited) { Die "The pipeline exited at once (code $($p.ExitCode)): see $($log -replace '\.log$', '.err.log')" }
        Say "Pipeline launched in the background (PID $($p.Id)). Log: $log"
        Say "Follow it with: logs   |   stop it with: stop   (closing this window does not stop it)"
    }
    "fg" { Assert-NotRunning; Invoke-Setup; Invoke-SmokeIfNew; & $Py -m perisim run @ProfArgs @Rest; exit $LASTEXITCODE }
    "status" {
        if (Test-Running) { Say "Running (PID $(Get-Content $PidFile -Raw))" } else { Say "No run in progress" }
        & $Py -m perisim status @ProfArgs
    }
    "logs" {
        $latest = Join-Path $LogDir "latest.txt"
        if (-not (Test-Path $latest)) { Die "No logs yet" }
        Get-Content -Path (Get-Content $latest -Raw).Trim() -Tail 200 -Wait
    }
    "stop" {
        if (Test-Running) {
            $id = [int](Get-Content $PidFile -Raw)
            & taskkill.exe /PID $id /T /F | Out-Null
            Remove-Item $PidFile -Force; Say "Stopped (PID $id)"
        } else { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue; Say "No run in progress" }
    }
    "smoke" { Invoke-Setup; & $Py -m perisim smoke --workdir (Join-Path $env:PERISIM_LOCAL "smoke") @Rest; exit $LASTEXITCODE }
    "test" {
        Invoke-Setup
        $env:PERISIM_TEST_WORKDIR = Join-Path $env:PERISIM_WORKDIR "experiments\$env:PERISIM_EXPERIMENT"
        $env:PERISIM_TEST_BASE = $env:PERISIM_WORKDIR
        Push-Location $RepoDir; & $Py -m pytest -p no:cacheprovider @Rest; $rc = $LASTEXITCODE; Pop-Location; exit $rc
    }
    "ksweep" { Invoke-Setup; & $Py -m perisim ksweep @ProfArgs @Rest; exit $LASTEXITCODE }
    "posthoc" { Invoke-Setup; & $Py (Join-Path $RepoDir "scripts\posthoc_transport.py") @ProfArgs @Rest; exit $LASTEXITCODE }
    "drift" { Invoke-Setup; & $Py (Join-Path $RepoDir "scripts\posthoc_drift.py") @ProfArgs @Rest; exit $LASTEXITCODE }
    "drift-missing" { Invoke-Setup; & $Py (Join-Path $RepoDir "scripts\posthoc_drift_missing.py") @ProfArgs @Rest; exit $LASTEXITCODE }
    "case-base" { Invoke-Setup; & $Py (Join-Path $RepoDir "scripts\posthoc_case_base.py") @ProfArgs @Rest; exit $LASTEXITCODE }
    "hypo" { Invoke-Setup; & $Py -m hypo @Rest; exit $LASTEXITCODE }
    "freeze" { Invoke-Setup; & $Py -m perisim freeze @ProfArgs @Rest; exit $LASTEXITCODE }
    "resources" { & $Py -m perisim resources; exit $LASTEXITCODE }
    default { Die "Unknown command: $Command (start | fg | status | logs | stop | smoke | test | ksweep | freeze | posthoc | drift | drift-missing | case-base | hypo | resources)" }
}
