"""Build the optional CPU kernel with Cargo and a C++17 compiler on this host."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import jax

from . import kernel_path


def build_sparse(source, work, destination, cargo, compiler):
    """Build an optional Rust cdylib linked against the system SuperLU."""
    candidates = (
        os.environ.get("SUPERLU_INCLUDE_DIR", "/usr/include/superlu"),
        "/opt/homebrew/opt/superlu/include", "/usr/local/opt/superlu/include",
    )
    include = next((Path(p) for p in candidates if (Path(p) / "slu_zdefs.h").is_file()), None)
    if include is None:
        print("SuperLU headers unavailable; using the SciPy sparse solver.")
        return
    obj = work / "sparse_bridge.o"
    subprocess.run([
        compiler, "-O2", "-std=c++17", "-fPIC", "-I" + str(include),
        "-c", str(source / "sparse_bridge.cc"), "-o", str(obj),
    ], check=True)
    prefix = include.parent.parent if include.parent.name == "include" else include.parent
    libdir = os.environ.get("SUPERLU_LIBRARY_DIR", str(prefix / "lib"))
    cpp = "-lc++" if sys.platform == "darwin" else "-lstdc++"
    subprocess.run([
        cargo, "rustc", "--example", "sparse", "--release", "--locked",
        "--target-dir", str(work / "target"), "--",
        "-C", "link-arg=" + str(obj), "-C", "link-arg=-L" + libdir,
        "-C", "link-arg=-lsuperlu", "-C", "link-arg=" + cpp,
    ], cwd=work, check=True)
    binary = work / "target/release/examples" / ("libsparse" + destination.suffix)
    binary.replace(destination.with_name("sparse-" + destination.name))


def main():
    if sys.platform not in ("darwin", "linux"):
        raise SystemExit("The optional CPU kernel supports macOS and Linux.")
    start = perf_counter()
    cargo = shutil.which("cargo")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not cargo or not compiler:
        raise SystemExit(
            "Install Cargo and a C++17 compiler before building the CPU kernel."
        )
    source = Path(__file__).resolve().parent
    destination = kernel_path()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        work = Path(tmp)
        (work / "src").mkdir()
        for name in ("Cargo.toml", "Cargo.lock", "src/lib.rs", "src/sparse.rs"):
            shutil.copyfile(source / name, work / name)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            (str(source.parents[1]), env.get("PYTHONPATH", ""))
        )
        subprocess.run(
            [sys.executable, str(source / "generate.py")], cwd=work, env=env, check=True
        )
        subprocess.run(
            [
                cargo,
                "build",
                "--release",
                "--locked",
                "--target-dir",
                str(work / "target"),
            ],
            cwd=work,
            check=True,
        )
        link = (
            ["-dynamiclib"]
            if sys.platform == "darwin"
            else ["-shared", "-ldl", "-lpthread", "-lm"]
        )
        binary = work / destination.name
        subprocess.run(
            [
                compiler,
                "-O2",
                "-std=c++17",
                "-fPIC",
                "-I" + jax.ffi.include_dir(),
                str(source / "bridge.cc"),
                str(work / "target/release/liboptical_coefficients.a"),
                *link,
                "-o",
                str(binary),
            ],
            check=True,
        )
        build_sparse(source, work, destination, cargo, compiler)
        binary.replace(destination)
    print(f"Built {destination} in {perf_counter()-start:.3f} s")


if __name__ == "__main__":
    main()
