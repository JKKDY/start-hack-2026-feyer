"""Make the vendored upstream reference package importable in tests."""

import pathlib
import sys

_REFERENCE = pathlib.Path(__file__).resolve().parents[1] / "reference"
if str(_REFERENCE) not in sys.path:
    sys.path.insert(0, str(_REFERENCE))
