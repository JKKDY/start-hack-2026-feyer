"""Location of the optional, host-built CPU coefficient kernel."""
import hashlib
import os
from pathlib import Path
import platform
import sys

import jaxlib

_ROOT = Path(__file__).resolve().parent


def kernel_path():
    cache = Path(os.environ.get("DIFFEROMETOR_NATIVE_CACHE", _ROOT / ".native"))
    digest = hashlib.sha256()
    for filename in (
        "Cargo.toml",
        "Cargo.lock",
        "generate.py",
        "bridge.cc",
        "src/lib.rs",
        "src/sparse.rs",
        "sparse_bridge.cc",
        "klu_bridge.cc",
        "../components.py",
        "../native_coefficients.py",
        "../_rust_sweep.py",
        "build.py",
    ):
        digest.update((_ROOT / filename).read_bytes())
    tag = f"{sys.platform}-{platform.machine()}-{jaxlib.__version__}-{digest.hexdigest()[:20]}"
    return cache / (tag + (".dylib" if sys.platform == "darwin" else ".so"))


def installed_kernel():
    if (
        sys.platform not in ("darwin", "linux")
        or os.environ.get("DIFFEROMETOR_DISABLE_NATIVE") == "1"
    ):
        return None
    cache = Path(os.environ.get("DIFFEROMETOR_NATIVE_CACHE", _ROOT / ".native"))
    if not cache.is_dir():
        return None
    path = kernel_path()
    return path if path.is_file() else None
