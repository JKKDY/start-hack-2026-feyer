---
name: verify-change
description: Verify and time a Differometor simulator change. Use after any edit to differometor/, before committing or reporting a speedup, and whenever asked to benchmark, test correctness, or compare branches.
---

# Verify a simulator change

The full reference is in docs/correctness-harness.md. Read it if anything below is unclear or fails.

## Rules

- Never modify `benchmark.py`. `git diff 3b538f0 -- benchmark.py` must be empty. For a variant, copy it to a new file such as `benchmark_<purpose>.py`.
- What the scored run needs must be the `differometor/` default. The original benchmark calls `run_build_step()` without solver options.
- Don't regenerate `tests/reference/` to make a failure go away. Run `--save` only when the cases, the fuzz rules (`FUZZ_VERSION`) or the reference ref change, and commit the result with that change.

## Steps

Use one Python 3.12 environment for all of them. On this Windows machine, that is `.venv/benchmark-local/windows-env/Scripts/python.exe`.

1. Correctness on all 13 setups, nominal plus fuzzed points, compared with committed references from the unmodified simulator. This takes about 3–4 minutes with the packed solver.
   ```
   python tests/check_general.py --suite all --output benchmark-harness-<name>.json
   ```
   It must end with `ALL PASS`. For a quick loop, run the default suite, which is the scored fixture only. Then run `--suite all` before committing.
2. Official check and scored time:
   ```
   python benchmark.py --cpus 1 --output benchmark-<name>.json
   ```
   It must print `PASS`. `--cpus` needs Linux or WSL. On native Windows, omit it; the run is then unpinned and its times are not comparable with pinned runs.
3. Compare timings only with earlier runs on the same machine. Use the benchmark's complete first-run total for the main comparison, with the harness report's `first_run_total` and `warm_median` as a per-setup view. `--candidate-ref <branch>` times another commit on the same machine without checking it out.
4. Report the results: pass or fail, the worst sensitivity and gradient errors, and first-run and warm times. Record lasting findings in the doc for that topic (docs/performance.md or docs/correctness-harness.md).

## Reading a failure

A failure line names the case, the point (0 is nominal, 1–4 are fuzzed), the quantity, its error and its tolerance. Errors on fuzzed points only usually mean the change depends on the fixture's parameter values. A crash or non-finite output on a case other than `uifo3_scored` usually means a code path the scored fixture never exercises: modulation, multiple detectors, carrier sweeps, or empty index arrays.
