import os


def _configure_xla():
    """Compile the CPU programs with LLVM optimisation level 0.

    The simulator's compiled programs run for ~0.1 s but took seconds to compile, mostly
    inside LLVM's machine-code optimiser. At -O0 compilation is ~40 % faster while runtime
    barely changes: the heavy linear algebra runs in LAPACK / SuperLU, which are precompiled
    and unaffected by this flag. Numerical results are unchanged (same float64 operations).

    XLA reads XLA_FLAGS when the CPU backend is first initialised, which happens after this
    package is imported in the normal workflow. An optimisation level chosen by the user in
    XLA_FLAGS is respected; set DIFFEROMETOR_KEEP_XLA_FLAGS=1 to leave XLA_FLAGS untouched.
    (Allowed by the challenge organisers, 2026-11-21.)
    """
    if os.environ.get("DIFFEROMETOR_KEEP_XLA_FLAGS"):
        return
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_backend_optimization_level" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} --xla_backend_optimization_level=0".strip()


_configure_xla()

from differometor.setups import Setup  # noqa: E402
from differometor.simulate import run, run_build_step, run_simulation_step, simulate  # noqa: E402
from differometor.components import power_detector, signal_detector  # noqa: E402


def __getattr__(name):
    # Plotly is only needed for visualisation: import it on first use, not at package import.
    if name == "visualize_setup":
        from differometor.plot import visualize_setup

        return visualize_setup
    raise AttributeError(f"module 'differometor' has no attribute {name!r}")
