#!/bin/bash
# Benchmark the optimization milestones in milestones.txt on Linux or WSL.
#
# Every milestone is exported from Git with LF line endings (fixture
# checksums must match), gets the ORIGINAL benchmark.py from 3b538f0, is
# built if needed, and is run RUNS times in fresh processes with
#   python <script> --cpus 1 --repeats 3 --output <results>/<label>_run<k>.json
#
# Usage:  benchmarks/history/run_linux.sh [label-prefix ...]
# Environment:
#   PYTHON   Python with the repository requirements (jax/jaxlib 0.9.0, numpy, ...)
#            and, for native milestones, pip + a C++20 compiler + CMake + Ninja.
#            Default: python3
#   WORK     where trees are exported (default: ~/dfm-history)
#   RESULTS  where reports go (default: $WORK/results)
#   RUNS     fresh runs per milestone (default: 2)
#
# Notes:
#   - native milestones pip-install native/ into $PYTHON's environment,
#     replacing any installed differometor-native. Run them last, or reinstall
#     the version you need afterwards.
#   - The original dense solver (m0, m1) needs about 14 GB of RAM, m2 several
#     GB. A default WSL VM (half the host RAM) can be too small; those runs
#     then die with "Killed". See docs/optimization-history.md.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
PYTHON=${PYTHON:-python3}
WORK=${WORK:-$HOME/dfm-history}
RESULTS=${RESULTS:-$WORK/results}
RUNS=${RUNS:-2}
mkdir -p "$WORK" "$RESULTS"
g() { git -c safe.directory='*' -c core.autocrlf=false -C "$REPO" "$@"; }

g show 3b538f0:benchmark.py > "$WORK/benchmark.orig.py"
ORIG_SHA=$(sha256sum "$WORK/benchmark.orig.py" | cut -d' ' -f1)
INC=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
EXT=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
LAST_NATIVE=""

selected() {
  [ $# -eq 0 ] && return 0
  for p in "$@"; do [[ "$label" == "$p"* ]] && return 0; done
  return 1
}

while IFS='|' read -r label sha kind envs desc script; do
  label=$(echo $label); sha=$(echo $sha); kind=$(echo $kind); envs=$(echo $envs); script=$(echo $script)
  script=${script:-benchmark.py}
  [ -z "$label" ] || [[ "$label" == \#* ]] && continue
  selected "$@" || continue
  dir="$WORK/$label"
  rm -rf "$dir" && mkdir -p "$dir"
  g archive "$sha" | tar -x -C "$dir"
  cp "$WORK/benchmark.orig.py" "$dir/benchmark.py"
  [ "$(sha256sum "$dir/benchmark.py" | cut -d' ' -f1)" = "$ORIG_SHA" ] || { echo "benchmark.py mismatch"; exit 1; }
  echo "=== $label ($sha, $kind, $script) env=[$envs]"
  case "$kind" in
    aot)  # same flags as the branch's setup.py on Linux; no setuptools needed
      g++ -O3 -shared -fPIC -ffp-contract=off -I"$INC" \
          -o "$dir/differometor/_native_components$EXT" "$dir/differometor/_native_components.cpp" \
          > "$RESULTS/$label.build.log" 2>&1 || { echo "  build failed (see $RESULTS/$label.build.log)"; continue; } ;;
    native)
      if [ "$LAST_NATIVE" != "$sha" ]; then
        (cd "$dir/native" && "$PYTHON" -m pip install -q --no-deps .) > "$RESULTS/$label.build.log" 2>&1 \
          || { echo "  native build failed (see $RESULTS/$label.build.log)"; continue; }
        LAST_NATIVE=$sha
      fi ;;
  esac
  for run in $(seq 1 "$RUNS"); do
    log="$RESULTS/${label}_run$run.log"
    (cd "$dir" && env $envs "$PYTHON" "$script" --cpus 1 --repeats 3 --output "$RESULTS/${label}_run$run.json" > "$log" 2>&1)
    echo "  run $run: $(grep -a -E 'Complete first-run total|PASS|FAIL' "$log" | tr '\n' ' ')"
  done
done < "$HERE/milestones.txt"
echo "Reports in $RESULTS. Table: $PYTHON $HERE/make_table.py $RESULTS"
