"""Python entry point for the native Differometor engine.

The compiled extension lives in ``dfm_native._dfm_native``. This module only
re-exports its public names, so user code never imports the private module.
"""

from ._dfm_native import ComplexSparseLU, complex_gradient_smoke, version

__all__ = ["ComplexSparseLU", "complex_gradient_smoke", "version"]
