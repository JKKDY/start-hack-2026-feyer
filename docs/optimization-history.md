# Optimization history: what produced what

This page benchmarks the main optimization steps of 21–22 November 2026, from the unoptimized original solver to the current native C++ engine. Each step was run with the **original, unchanged `benchmark.py`** from `3b538f0`, so the rows are comparable. Two exceptions are marked: the packed-sparse row runs its commit's own `benchmark.py` (see its footnote), and the JAX-free rows run `benchmark_native.py`. It is a coarse map: one row per group of related changes, not one per commit. For the detailed record of each change, see [performance.md](performance.md).

## Results (WSL, 1 CPU)

Measured on 22 November 2026. Machine: Intel Core i7-11370H laptop (16 GB RAM), WSL2 Ubuntu 26.04 with 7 GB RAM, Python 3.14.4, gcc 15, JAX and JAXLIB 0.9.0, NumPy 2.5.5. Every run used `benchmark.py --cpus 1 --repeats 3` (the JAX-free rows: `benchmark_native.py` with the same options), in a fresh process, from a clean LF export of the commit. Each milestone ran twice. The table shows the run with the lower first-run total, and the range covers both runs. Times are in seconds.

| Milestone | Commit | Setup | Compile | Forward | Sensitivity | Gradient | Full first run (range) | Without setup | Warm median | Check (max log error) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original dense JAX solver | `3b538f0` | not run in WSL (memory), see the Windows table | | | | | | | | |
| Dense JAX: one fewer noise solve, grouped components, detector selections without tiling | `089afa2` | not run in WSL (memory), see the Windows table | | | | | | | | |
| Packed sparse signal + noise solver (JAX JIT)† | `47cc00a` | 1.790 | 10.771 | 0.120 | 0.0007 | 0.103 | **12.786** (1 run) | 10.996 | 0.175 | PASS (5.37e-06) |
| + static index arrays, refined sparse solves, NumPy build step, folded fixed parameters | `2960b9e` | 0.359 | 7.278 | 0.154 | 0.0009 | 0.129 | **7.921** (7.92–8.83) | 7.562 | 0.243 | PASS (5.37e-06) |
| AOT branch: C++ component kernels, JAX glue | `2e61add` | 0.374 | 0.548 | 0.121 | 0.0010 | 0.153 | **1.198** (1.20–1.70) | 0.823 | 0.242 | PASS (5.37e-06) |
| AOT branch: + XLA optimization level 0, LU reuse in reverse mode | `8072817` | 0.362 | 0.371 | 0.132 | 0.0011 | 0.087 | **0.954** (0.95–1.07) | 0.592 | 0.200 | PASS (5.37e-06) |
| Native C++ engine via JAX bridge, XLA flags disabled | `b8bc5ef` | 0.291 | 0.370 | 0.143 | 0.0008 | 0.052 | **0.857** (0.86–0.97) | 0.566 | 0.177 | PASS (5.37e-06) |
| Native C++ engine via JAX bridge (default XLA flags) | `b8bc5ef` | 0.300 | 0.133 | 0.143 | 0.0008 | 0.055 | **0.631** (0.63–0.64) | 0.332 | 0.175 | PASS (5.37e-06) |
| + native left-looking sparse LU, one tape entry per frequency for signal + noise | `8d2f69f` | 0.290 | 0.133 | 0.082 | 0.0008 | 0.034 | **0.539** (0.54–0.62) | 0.249 | 0.089 | PASS (5.37e-06) |
| + one iterative-refinement step for signal solves | `f7be1dc` | 0.287 | 0.132 | 0.077 | 0.0008 | 0.028 | **0.524** (0.52–0.54) | 0.238 | 0.082 | PASS (5.37e-06) |
| JAX-free `benchmark_native.py`*: native C++ engine | `b8bc5ef` | 0.081 | 0 | 0.143 | 0.0004 | 0.055 | **0.279** (0.28–0.41) | 0.198 | 0.180 | PASS (5.37e-06) |
| JAX-free `benchmark_native.py`*: + native LU, one tape entry per frequency | `8d2f69f` | 0.114 | 0 | 0.096 | 0.0005 | 0.037 | **0.248** (0.25–0.27) | 0.134 | 0.107 | PASS (5.37e-06) |
| JAX-free `benchmark_native.py`*: + one refinement step | `f7be1dc` | 0.095 | 0 | 0.085 | 0.0004 | 0.026 | **0.206** (0.21–0.21) | 0.111 | 0.081 | PASS (5.37e-06) |

Columns:

- **Setup**: imports, CPU initialisation, input loading and setup build. About 0.22–0.25 s of it is `import jax`, which `benchmark.py` does itself.
- **Compile**: all trace/lower and compile phases (forward, sensitivity, gradient).
- **Forward, Sensitivity, Gradient**: execution of each stage, without its compilation.
- **Full first run**: `benchmark.py`'s complete first-run total, the prize metric.
- **Without setup**: full first run minus setup.
- **Warm median**: median of the three warm repeats.

† At `47cc00a` the packed solver was selected by that commit's own `benchmark.py` (a new `--signal-solver packed` default). With the original `benchmark.py` the library still used the dense solver there, and a run measured dense timings (27.7 s first run, 11.8 s warm on Windows). The library itself defaulted to packed only from the next milestone on. This row therefore runs the commit's own `benchmark.py`, which has the same workload and validation.

\* The last three rows are not the scored benchmark. They run `benchmark_native.py` from that commit's own tree: a copy of `benchmark.py` that never imports JAX, calls the native engine directly, and computes the readout and its VJP in NumPy. It keeps the same validation and report format, with compile phases recorded as 0 s. At `f7be1dc` it is 0.32 s faster than the scored benchmark: about 0.19 s of setup (almost all `import jax`) and 0.13 s of compilation. Forward and gradient run the same native code in both.

The raw reports are in [`benchmarks/history/results-wsl/`](../benchmarks/history/results-wsl/).

### Results on native Windows (1 CPU), including the original solver

The original dense solver peaks at about 14 GB, more than the 7 GB WSL VM, so the Linux OOM killer stopped it there. These rows ran natively on Windows on the same laptop (Windows 10, Python 3.12.10, JAX and JAXLIB 0.9.0), with WSL shut down and other applications closed. Each ran once with `run_windows.ps1`: pinned with `start /affinity 1`, numerical libraries limited to one thread, and `benchmark.py --repeats 3` (Windows cannot use `--cpus`). The last two rows overlap with the WSL table so the platforms can be compared: Windows is slower mainly in setup (imports and build, about 1 s against 0.3 s).

| Milestone | Commit | Setup | Compile | Forward | Sensitivity | Gradient | Full first run | Without setup | Warm median | Check (max log error) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original dense JAX solver | `3b538f0` | 2.025 | 26.049 | 13.938 | 0.0033 | 18.305 | **60.338** | 58.313 | 15.119 | PASS (5.38e-06) |
| Dense JAX: one fewer noise solve, grouped components, detector selections without tiling | `089afa2` | 1.951 | 14.123 | 9.597 | 0.0020 | 2.287 | **27.960** | 26.009 | 11.987 | PASS (5.38e-06) |
| Packed sparse signal + noise solver (JAX JIT)† | `47cc00a` | 2.041 | 12.287 | 0.120 | 0.0008 | 0.107 | **14.556** | 12.515 | 0.228 | PASS (5.37e-06) |
| + static index arrays, refined sparse solves, NumPy build step, folded fixed parameters | `2960b9e` | 1.117 | 7.908 | 0.150 | 0.0009 | 0.135 | **9.311** | 8.194 | 0.296 | PASS (5.37e-06) |
| Current native C++ engine (`f7be1dc`, as in the WSL table's `m9` row) | `f7be1dc` | 1.046 | 0.167 | 0.096 | 0.0007 | 0.024 | **1.334** | 0.288 | 0.126 | PASS (5.37e-06) |

The raw reports are in [`benchmarks/history/results-windows/`](../benchmarks/history/results-windows/). For reference, the official baseline of the original solver on the scoring machine (Xeon Platinum 8468, 1 CPU, [cpu-baseline.json](../benchmarks/cpu-baseline.json)) took 44.204 s: setup 3.661, compile 22.918, forward 13.288, gradient 4.336, warm 17.679.

Two earlier Windows attempts of the original solver did not finish. With the WSL VM and other applications running, only about 5.5 GB of its 13.3 GB fit in RAM and it was paged out. A run without the one-thread limits oversubscribed the pinned CPU. Both are why `run_windows.ps1` now sets the thread limits and kills runs that time out.

## What produced what

1. **Dense JAX tweaks** (`4a2c3d0`, `3e030f4`, `089afa2`): one fewer quantum-noise solve, and component functions evaluated per function instead of `vmap(switch)` over all 19 branches. On Windows the first run went from 60.3 to 28.0 s: compilation 26.0 → 14.1 s and gradient execution 18.3 → 2.3 s.
2. **Packed sparse solver** (`47cc00a`): signal and noise matrices solved as sparse systems instead of dense 751 × 751 LUs. Forward execution went from 9.6 to 0.12 s and warm time from 12.0 to 0.23 s (Windows). Compilation still dominated: 12.3 s of the 14.6 s first run.
3. **Baseline-optimized** (`2960b9e`): static index arrays, iterative refinement after each sparse solve, a NumPy build step and folded fixed parameters. First run 7.9 s in WSL, of which compilation was 7.3 s.
4. **Ahead-of-time instead of JIT** (`2e61add`): component kernels compiled in C++ ahead of time, so XLA no longer compiles the simulator. Compilation fell from 7.3 to 0.55 s and the first run from 7.9 to 1.2 s. This was the largest single step.
5. **XLA optimization level 0 and LU reuse** (`8072817`): what is left to compile is small, so XLA's optimizer costs more than it saves. Compilation 0.55 → 0.37 s, gradient 0.15 → 0.09 s, first run 0.95 s.
6. **Native C++ engine with a JAX bridge** (`b8bc5ef`): planner, assembly, solves and reverse mode all in C++ (CoDiPack + SuperLU), called from JAX through `pure_callback`. With XLA flags disabled the first run was 0.86 s. The default flags (`--xla_backend_optimization_level=0 --xla_cpu_use_fusion_emitters=false`) cut compilation from 0.37 to 0.13 s, giving 0.63 s.
7. **Native sparse LU and one tape entry per frequency** (`8d2f69f`): a left-looking LU replaced SuperLU for signal matrices (101 factorisations 55 → 20 ms), and the signal and noise solves of a frequency share one tape entry. Forward 0.143 → 0.082 s, gradient 0.055 → 0.034 s, warm 0.175 → 0.089 s.
8. **One refinement step** (`f7be1dc`): instead of two, for signal solves, with unchanged harness errors. First run 0.52 s, warm 0.082 s.

Of the current 0.52 s, about 0.29 s is setup (mostly `import jax`) and 0.13 s is JAX compiling the benchmark's own readout and gradient wrapper. Neither can be reduced without changing `benchmark.py`. The native simulation itself (forward, sensitivity and gradient) takes about 0.11 s.

Not included: the Rust branches (`experiment/rust-packed-solver`, `experiment/cold-input-setup`) need a Rust toolchain, which this machine does not have.

## How to reproduce

The scripts are in [`benchmarks/history/`](../benchmarks/history/):

- `milestones.txt`: the milestone list (label, commit, kind, extra environment, description, and optionally a script). Rows without a script run the original `benchmark.py`; `m2` runs its commit's own `benchmark.py` (`benchmark_commit.py`), and the `j*` rows run that commit's `benchmark_native.py`. Edit it to add or remove rows.
- `run_linux.sh`: Linux or WSL runner; the WSL table above came from this method.
- `run_windows.ps1`: native-Windows runner (pins with `start /affinity 1`, since `--cpus` needs Linux).
- `make_table.py`: turns the JSON reports into the table above.

For each milestone, the runners:

1. export the commit with `git -c core.autocrlf=false archive`, because the fixture SHA-256 checks need LF line endings;
2. keep the commit's own `benchmark.py` as `benchmark_commit.py`, then replace `benchmark.py` with the one from `3b538f0` and check its hash;
3. build what the milestone needs:
   - `aot` milestones compile `differometor/_native_components.cpp` in place with `g++ -O3 -shared -fPIC -ffp-contract=off`, the same flags as the branch's `setup.py`;
   - `native` milestones `pip install` the commit's `native/` package (C++20 compiler, CMake and Ninja needed);
4. run `benchmark.py` (or the row's script) with `--cpus 1 --repeats 3` `RUNS` times (default 2) in fresh processes, with the milestone's extra environment:
   - `DIFFEROMETOR_NATIVE=1` turns on the AOT branch's backend;
   - `XLA_FLAGS=--xla_backend_optimization_level=0` is that branch's recommended flag;
   - `DIFFEROMETOR_XLA_FLAGS=0` turns off the native bridge's automatic XLA flags.

On Linux or WSL:

```bash
PYTHON=~/dfm-venv/bin/python benchmarks/history/run_linux.sh            # all milestones
PYTHON=~/dfm-venv/bin/python benchmarks/history/run_linux.sh m3 m9      # only some
~/dfm-venv/bin/python benchmarks/history/make_table.py ~/dfm-history/results
```

`PYTHON` must have the repository requirements installed (`pip install -r requirements.txt`). Each benchmark run takes under a second from `m4` on, but every `native` commit is rebuilt from scratch with `pip install` (a few minutes each), so the native rows take most of the wall time. Native milestones replace the `differometor-native` installed in that environment, so run them last or reinstall afterwards. To run `m0` and `m1` in WSL, give the VM at least 16 GB (`memory=` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`).

On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File benchmarks\history\run_windows.ps1 -Python D:\Dev\start-hack-munich-2026\.venv\benchmark-local\windows-env\Scripts\python.exe -Only m0,m1,m2,m3,m9 -Runs 1
```

Native milestones use the `differometor-native` already installed in that environment unless `-BuildNative` is given; that option builds with MSVC via `uv pip install`. For `m0` and `m1` on a 16 GB machine, run `wsl --shutdown` and close large applications first. The runner limits OpenMP, OpenBLAS, MKL and NumExpr to one thread, as `benchmark.py --cpus 1` does on Linux. A run that exceeds `-TimeoutMinutes` (default 25) is killed.

Timings depend on the machine, the file cache (the first run after a reboot or WSL restart is slower, mostly in imports) and thermal state. Compare rows from the same machine and session.
