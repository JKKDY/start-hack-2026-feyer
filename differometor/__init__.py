from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector


def __getattr__(name):
    # Plotly is only needed for visualisation: import it on first use, not at package import.
    if name == "visualize_setup":
        from differometor.plot import visualize_setup

        return visualize_setup
    raise AttributeError(f"module 'differometor' has no attribute {name!r}")
