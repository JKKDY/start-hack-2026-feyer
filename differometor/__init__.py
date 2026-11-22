import importlib.util as _importlib_util
import os as _os

# XLA compile-time flags for the native backend. With the native engine
# (differometor/native_bridge.py) XLA only compiles small glue programs:
# callers' readouts written in jax.numpy and the host callbacks. Lower
# optimisation effort roughly halves their compile time while the simulator
# itself runs as ahead-of-time compiled C++. These flags would slow the JAX
# backend (where XLA runs the whole simulation), so they are only set when the
# native backend is selected and importable. XLA reads XLA_FLAGS when its CPU
# backend initialises, so this must run before the first JAX computation.
# Flags the caller already set are kept; DIFFEROMETOR_XLA_FLAGS=0 disables this.
_NATIVE_XLA_FLAGS = ("--xla_backend_optimization_level=0", "--xla_cpu_use_fusion_emitters=false")


def _apply_native_xla_flags():
    if _os.environ.get("DIFFEROMETOR_XLA_FLAGS", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    if _os.environ.get("DIFFEROMETOR_BACKEND", "native").strip().lower() != "native":
        return
    if _importlib_util.find_spec("differometor_native") is None:
        return
    existing = _os.environ.get("XLA_FLAGS", "")
    present = {flag.split("=", 1)[0] for flag in existing.split()}
    extra = [flag for flag in _NATIVE_XLA_FLAGS if flag.split("=", 1)[0] not in present]
    if extra:
        _os.environ["XLA_FLAGS"] = " ".join(filter(None, [existing, *extra]))


_apply_native_xla_flags()

from differometor.setups import Setup  # noqa: E402
from differometor.simulate import run, run_build_step, run_simulation_step, simulate  # noqa: E402
from differometor.components import power_detector, signal_detector  # noqa: E402


def __getattr__(name):
    # Plotly is only needed for visualisation: import it on first use, not at package import.
    if name == "visualize_setup":
        from differometor.plot import visualize_setup

        return visualize_setup
    raise AttributeError(f"module 'differometor' has no attribute {name!r}")
