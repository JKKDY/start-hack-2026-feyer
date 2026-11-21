"""Run the standard CPU benchmark with a lighter LLVM optimization level."""

import os
from pathlib import Path
import sys


if __name__ == "__main__":
    benchmark = Path(__file__).resolve().with_name("benchmark.py")
    flags = os.environ.get("XLA_FLAGS", "")
    if not any(flag.startswith("--xla_backend_optimization_level=") for flag in flags.split()):
        os.environ["XLA_FLAGS"] = f"{flags} --xla_backend_optimization_level=1".strip()
    os.execv(sys.executable, [sys.executable, str(benchmark), *sys.argv[1:]])
