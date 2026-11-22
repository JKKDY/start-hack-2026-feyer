# Benchmark the optimization milestones in milestones.txt natively on Windows.
#
# Same method as run_linux.sh: LF export of each commit, the ORIGINAL
# benchmark.py from 3b538f0, RUNS fresh processes per milestone. benchmark.py's
# --cpus needs Linux, so each process is pinned to logical CPU 0 with
# `start /affinity 1` instead.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File benchmarks\history\run_windows.ps1 `
#       -Python <python.exe> [-Only m0,m1] [-Work <dir>] [-Runs 2]
#
# kind plain  : runs as is.
# kind native : uses the differometor-native already installed in -Python's
#               environment unless -BuildNative is given (then installs
#               native\ with `uv pip install`, needs MSVC + CMake).
# kind aot    : builds the extension with `python setup.py build_ext --inplace`
#               (needs setuptools and MSVC).
#
# Memory: the original dense solver (m0, m1) peaks at about 14 GB. On a 16 GB
# machine, shut down WSL (`wsl --shutdown`) and close large applications
# first, otherwise Windows pages it out and a run takes far longer than the
# timeout (a run that times out is killed).
param(
  [Parameter(Mandatory = $true)][string]$Python,
  [string[]]$Only = @(),
  [string]$Work = "$env:TEMP\dfm-history",
  [int]$Runs = 2,
  [int]$TimeoutMinutes = 25,
  [switch]$BuildNative
)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (git -C $here rev-parse --show-toplevel)
$results = Join-Path $Work "results"
New-Item -ItemType Directory -Force $results | Out-Null
git -C $repo -c core.autocrlf=false archive --format=tar -o "$Work\orig.tar" 3b538f0 benchmark.py
$lastNative = ""

foreach ($line in Get-Content "$here\milestones.txt") {
  if ($line.Trim() -eq "" -or $line.TrimStart().StartsWith("#")) { continue }
  $f = $line.Split("|") | ForEach-Object { $_.Trim() }
  $label, $sha, $kind, $envs = $f[0], $f[1], $f[2], $f[3]
  $script = if ($f.Count -gt 5 -and $f[5]) { $f[5] } else { "benchmark.py" }
  if ($Only.Count -gt 0 -and -not ($Only | Where-Object { $label.StartsWith($_) })) { continue }
  $dir = Join-Path $Work $label
  if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
  New-Item -ItemType Directory -Force $dir | Out-Null
  git -C $repo -c core.autocrlf=false archive --format=tar -o "$Work\$label.tar" $sha
  tar -xf "$Work\$label.tar" -C $dir
  tar -xf "$Work\orig.tar" -C $dir   # original benchmark.py
  "=== $label ($sha, $kind, $script) env=[$envs]"
  if ($kind -eq "aot") {
    Push-Location $dir; & $Python setup.py build_ext --inplace *> "$results\$label.build.log"; $ok = $?; Pop-Location
    if (-not $ok) { "  build failed (see $results\$label.build.log)"; continue }
  }
  if ($kind -eq "native" -and $BuildNative -and $lastNative -ne $sha) {
    uv pip install --python $Python --reinstall-package differometor-native "$dir\native" *> "$results\$label.build.log"
    if ($LASTEXITCODE -ne 0) { "  native build failed"; continue }
    $lastNative = $sha
  }
  $setEnv = ($envs.Split(" ", [StringSplitOptions]::RemoveEmptyEntries) | ForEach-Object { "set `"$_`" && " }) -join ""
  foreach ($run in 1..$Runs) {
    $log = "$results\$($label)_run$run.log"
    $cmd = "$setEnv start /affinity 1 /wait /b `"`" `"$Python`" $script --repeats 3 --output $results\$($label)_run$run.json > `"$log`" 2>&1"
    $p = Start-Process -FilePath "cmd.exe" -WorkingDirectory $dir -ArgumentList "/c", $cmd -NoNewWindow -PassThru
    if (-not $p.WaitForExit($TimeoutMinutes * 60000)) {
      taskkill /T /F /PID $p.Id | Out-Null   # cmd.exe and the python it started
      "  run $run TIMEOUT (killed)"; continue
    }
    $l = (Get-Content $log | Select-String "first-run total|PASS|FAIL|Error" | ForEach-Object { $_.Line }) -join " "
    "  run $run`: $l"
  }
}
"Reports in $results. Table: $Python $here\make_table.py $results"
