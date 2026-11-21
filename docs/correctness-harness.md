# Correctness and timing harness

`benchmark.py` checks only the scored fixture at its saved parameters, against one public sensitivity curve. [`tests/check_general.py`](../tests/check_general.py) is broader. It compares every physical output and the gradient against references computed by the **unmodified simulator**. It does this at each setup's saved parameters and at seeded **fuzzed** parameter sets. It also records first-run and warm timings for each case, so every run doubles as a benchmark against the reference code.

## Verifying a change

The references are committed in `tests/reference/`, so a fresh checkout can verify a change at once. Nobody needs to run the slow original simulator again.

1. Run `python tests/check_general.py --suite all`. It tests `run_build_step()`'s default solver, which is the path the original `benchmark.py` scores. Add `--signal-solver dense` to test the dense path as well.
2. Require `ALL PASS`. A failure prints the case, the point (0 is nominal, 1–4 are fuzzed), the quantity, its error and its tolerance. The JSON report has every value.
3. For speed, compare the report's `first_run_total` and `warm_median` with an earlier report from the same machine. The stored reference timings come from the laptop described below. The printed speedups are exact only there.
4. Also run `python benchmark.py --cpus 1` for the official check and scored time.

Run `--save` again only when the cases, the fuzz rules (`FUZZ_VERSION`), the sample settings or the reference ref change. Then commit the new `tests/reference/` with that change.

Agents also get these steps as the project skill `verify-change`. Claude Code reads it from [`.claude/skills/verify-change/SKILL.md`](../.claude/skills/verify-change/SKILL.md) and Codex from [`.agents/skills/verify-change/SKILL.md`](../.agents/skills/verify-change/SKILL.md). The two files must stay identical, so edit both together. The skill points back to this document.

## Commands

The Windows experiments used the Python 3.12 environment in `.venv/benchmark-local/windows-env`. On Linux or WSL, use the environment from the [challenge README](../README.md).

```bash
python tests/check_general.py --save                   # regenerate references from git ref `main` (already committed)
python tests/check_general.py                          # scored fixture only (default suite)
python tests/check_general.py --suite all              # all 13 setups
python tests/check_general.py --cases aligo_am,uifo3_finesse
python tests/check_general.py --suite all --output benchmark-harness-<name>.json
python tests/check_general.py --candidate-ref <branch>  # another commit's simulator, without checking it out
```

A passing run ends with `ALL PASS` and exit code 0. Any tolerance failure, crash, non-finite output or all-zero gradient fails that case and exits with 1.

| Option | Meaning |
| --- | --- |
| `--suite scored` | Only `uifo3_scored`, the benchmark workload. Default in check mode. |
| `--suite all` | Every case in [`tests/cases.py`](../tests/cases.py). Default in save mode. |
| `--cases a,b` | An explicit subset. Overrides `--suite`. |
| `--reference-ref REF` | Save mode only. The git ref whose `differometor/` produces the references. Default `main`. |
| `--candidate-ref REF` | Check mode only. Tests `differometor/` from a git ref instead of the working tree. |
| `--samples N`, `--seed S`, `--scale X` | Save mode only. Fuzzed parameter sets per case (default 4), RNG seed (0), and a multiplier on every fuzz width (1). |
| `--signal-solver NAME` | Check mode only. Passes `signal_solver=NAME` to `run_build_step`, for example `dense`, to test another solver path on this branch. |
| `--output PATH` | The check-mode JSON report. Default `benchmark-harness.json`, which `.gitignore` already excludes. |
| `--no-pin` | Do not pin workers to one logical CPU. Timings are then not comparable. |

## How it works

**Reference simulator.** `--save` exports `differometor/` from `--reference-ref` with `git archive` into `tests/.tmp/simulator-<sha>/`. It then runs each case with that code. The harness and the case definitions always come from the working tree, so the reference and candidate code see the same setups. The worker refuses to run if `differometor` was imported from anywhere else. On this checkout, local `main` is `3b538f0` and `origin/main` is `c10e851`. Their `differometor/` trees are identical, since `c10e851` only adds a PDF.

**Fuzzing.** For each case, [`cases.fuzz_parameters`](../tests/cases.py) draws `--samples` parameter sets around the nominal point. Every optimised parameter moves independently in each sample, so a scored-fixture sample changes all 193 design variables at once.

| Property | Rule at scale 1 |
| --- | --- |
| `reflectivity` | `r ** exp(U(-0.1, 0.1))`. Stays in [0, 1] and keeps exact 0 and 1. |
| `tuning` | `+ U(-1, 1)` degrees |
| `angle` | `+ U(-5, 5)` degrees |
| `db` | `+ U(-1, 1)`, kept at 0 or above |
| `power`, `mass` | `× exp(U(-0.2, 0.2))` |
| `length` | `× exp(U(-0.05, 0.05))`. Tied grid lengths share one value, so they stay tied. |
| anything else | `× exp(U(-0.05, 0.05))` |

The parameter sets are stored in each reference file, and check mode replays exactly those points. Results therefore do not depend on NumPy's RNG staying stable across versions. If the rules change, bump `FUZZ_VERSION`. Check mode refuses references with a different version.

**Tolerances from round-off floors.** In save mode, every point, nominal and fuzzed, is evaluated a second time with all non-length parameters nudged by 1e-15 relative, with random signs. This roughly matches the backward error a stable solver injects. The resulting output change is that point's floor: how far a legitimately different but stable algorithm may move each quantity. In check mode the tolerance per quantity and point is `clamp(10 × floor, MIN_TOL, MAX_TOL)`. A quantity whose floor already exceeds `MAX_TOL` is ill-conditioned at that point. It gets a loose `10 × floor` gate, and the run prints a flag for it.

| Quantity | Error measure | MIN_TOL | MAX_TOL |
| --- | --- | ---: | ---: |
| carrier, signal | max normwise relative error per sweep column | 1e-9, 1e-8 | 1e-8, 1e-6 |
| powers | max relative error where power exceeds 1e-12 × max | 1e-9 | 1e-8 |
| noise, sensitivity | max abs. difference of natural log | 1e-8 | 1e-7 |
| loss | absolute | 1e-9 | 1e-7 |
| gradient | normwise relative | 1e-6 | 1e-4 |

The sensitivity cap of 1e-7 is 100× tighter than the benchmark's 1e-5 limit. Lengths are excluded from the nudge because a relative change on a 4000 m length is a physical phase change, not round-off. An earlier 1e-13 nudge overestimated the floors about 100× and flagged nearly everything as ill-conditioned.

**Execution and timing.** Each case runs in a fresh subprocess, pinned to one logical CPU, with the compilation cache disabled. It uses the same three stages as `benchmark.py`: `jit(vjp(simulate))`, then the jitted readout VJP, then the jitted backward pass. Custom VJPs and pytree changes are therefore tested as they are scored. Each worker records:

- `import`: JAX and simulator import
- `build`: the case setup and `run_build_step`
- `compile`: lowering plus compilation of all three stages
- `first_evaluation`: execution of the first point
- `first_run_total`: everything from worker start to the first point's results on the host, like the benchmark's complete first-run total
- `warm_evaluations`, `warm_median`: every later point with compiled code

The references store the same timings for the reference simulator. Check mode prints first-run and warm speedups per case. Speedups are meaningful only when both were measured on the same machine.

## Cases

| Case | What it covers that the scored fixture doesn't |
| --- | --- |
| `uifo3_scored` | The benchmark workload itself, 101 frequencies. |
| `uifo3_jitter` | All optimisable parameters perturbed; 37 frequencies, 500–4000 Hz. |
| `uifo3_finesse` | 5× lower mirror transmissions, so worse conditioning; 23 frequencies. |
| `uifo3_20_5000` | The other trained UIFO: 2 signal detectors (balanced homodyne), 2 noise detectors, a qhd phase, 7 squeezers. |
| `uifo2_rand_{space,am,fm}` | Random 2×2 topologies; laser amplitude and frequency modulation. |
| `aligo_{space,am,fm}`, `voyager_{space,fm}` | Classic detectors, all modulation modes, 2 lasers. |
| `cavity_tuning` | A carrier-changing sweep over 61 tunings, 3 power detectors, no noise detector. |

The loss is `mean(log10(sensitivity))` when the case has a signal detector and a frequency sweep. Otherwise it is the sum of detected powers.

## Files and report

| Path | Content |
| --- | --- |
| `tests/reference/<case>.npz` | `parameters` (points × P) and the stacked outputs for every point. Committed. |
| `tests/reference/meta.json` | Reference commit, fuzz settings, per-point floors, and per-case reference timings. Committed. |
| `tests/.tmp/` | Exported reference simulators and worker scratch files. Git-ignored. |
| `--output` JSON | Per case: pass or fail, failures, worst error per quantity, every point's error, tolerance and flag, timings, reference timings, and speedups. Also the working-tree commit, branch, whether `differometor/` was dirty, and the reference settings. |

The references are deterministic for a given simulator, case definition and seed. `--candidate-ref main` against references from `main` gave errors of exactly 0 for `cavity_tuning`, `aligo_space`, `uifo2_rand_fm` and `voyager_fm`.

## Reference results

References were generated on 21 November 2026 from `main` (`3b538f0`): 4 fuzz samples, seed 0, scale 1, so 5 points per case. The machine was an Intel Core i7-11370H laptop with 16 GB RAM, running native Windows, Python 3.12.10 and JAX 0.9.0, with each worker pinned to one logical CPU. The full `--save` took about 14 minutes, 6.3 of them for `uifo3_scored`. The 13 reference files take up 13 MB.

Floors are the output change from the 1e-15 nudge. The ill-conditioned column lists quantities whose floor exceeds `MAX_TOL` at one or more points, so they get the loose gate.

| Case | Sensitivity floor, nominal / worst | Gradient floor, nominal / worst | Ill-conditioned at some point | Reference first run (s) | Reference warm (s) |
| --- | --- | --- | --- | ---: | ---: |
| `uifo3_scored` | 1.7e-7 / 1.7e-7 | 2.0e-6 / 1.3e-4 | powers, noise, sensitivity, gradient | 107.6 | 28.374 |
| `uifo3_jitter` | 4.8e-8 / 3.5e-7 | 3.0e-6 / 3.4e-6 | powers, sensitivity | 32.5 | 5.146 |
| `uifo3_finesse` | 6.6e-7 / 1.1e-6 | 1.5e-5 / 1.6e-5 | powers, noise, sensitivity, loss | 34.0 | 3.143 |
| `uifo3_20_5000` | 3.8e-7 / 1.9e-6 | 1.1 / 1.1 | signal, powers, noise, sensitivity, gradient | 33.0 | 4.013 |
| `uifo2_rand_space` | 1.5e-11 / 1.1e-10 | 2.8e-11 / 2.1e-10 | none | 25.7 | 0.440 |
| `uifo2_rand_am` | 1.1e-14 / 1.1e-14 | 6.0e-15 / 7.0e-15 | none | 24.5 | 0.369 |
| `uifo2_rand_fm` | 9.8e-10 / 2.6e-9 | 2.5e-10 / 2.5e-10 | none | 23.3 | 0.416 |
| `aligo_space` | 4.9e-10 / 4.9e-10 | 4.4e-9 / 4.4e-9 | none | 25.5 | 0.135 |
| `aligo_am` | 1.2e-7 / 1.2e-7 | 1.1e-7 / 1.1e-7 | sensitivity | 24.5 | 0.050 |
| `aligo_fm` | 4.6e-8 / 4.6e-8 | 1.7e-9 / 1.7e-9 | none | 24.8 | 0.049 |
| `voyager_space` | 9.3e-12 / 9.3e-12 | 1.7e-4 / 1.7e-4 | gradient | 25.3 | 0.236 |
| `voyager_fm` | 2.7 / 2.7 | 0.90 / 0.90 | sensitivity, loss, gradient | 24.5 | 0.221 |
| `cavity_tuning` | 0 / 0 | 2.1e-13 / 2.5e-13 | none | 25.9 | 0.009 |

What the floors show:

- **The scored setup is about one order of magnitude from round-off deciding the answer.** Its nominal sensitivity floor is 1.7e-7, and one fuzzed point's gradient floor reaches 1.3e-4. Higher-finesse variants are already beyond that. Keep pivoted factorizations and float64.
- **`uifo3_20_5000`'s gradient is essentially undetermined at round-off level**, with a floor of 1.1.
- **`voyager_fm`'s detected FM signal is almost zero**, so its sensitivity has no meaning. The case serves as a crash and NaN test only.
- **Power floors reach 4.4e-4 on the trained UIFOs**, so the powers comparison on those cases is mostly under the loose gate.
- Reference first-run times include each worker's JAX import, about 2.5 s on this machine.

## Results

Results for each build, with benchmark times, are in [Performance experiments](performance.md#correctness-harness-results). Record new harness results there.
