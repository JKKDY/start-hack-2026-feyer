"""Native Differometor: upstream-compatible Python API over a C++ engine.

Top-level exports match upstream ``differometor/__init__.py``:

    Setup, run, run_build_step, run_simulation_step, simulate,
    visualize_setup, power_detector, signal_detector

Submodules mirror upstream: ``setups``, ``simulate``, ``components``,
``utils`` and ``plot``. The new native interface (plans, simulators,
forward/backward) lives in ``differometor_native.native``.
"""

from differometor_native._native import version as _native_version
from differometor_native.components import power_detector, signal_detector
from differometor_native.setups import Setup
from differometor_native.simulate import run, run_build_step, run_simulation_step, simulate

__version__ = _native_version()


try:  # upstream imports plot eagerly; keep that when Plotly is installed
    from differometor_native.plot import visualize_setup
except ImportError:  # pragma: no cover - depends on the environment

    def visualize_setup(*args, **kwargs):
        """Plotly HTML rendering of a setup (upstream `plot.visualize_setup`).

        Plotly is not installed, so the real function (with upstream's exact
        signature) cannot be imported. Calling this raises ImportError.
        """
        from differometor_native.plot import visualize_setup as _visualize_setup

        return _visualize_setup(*args, **kwargs)


__all__ = [
    "Setup",
    "run",
    "run_build_step",
    "run_simulation_step",
    "simulate",
    "visualize_setup",
    "power_detector",
    "signal_detector",
]
