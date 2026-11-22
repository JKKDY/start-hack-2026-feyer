"""Build the optional CPU kernel with Cargo on this host."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


from . import kernel_path


def main():
    if sys.platform not in ("darwin", "linux"):
        raise SystemExit("The optional CPU kernel supports macOS and Linux.")
    start = perf_counter()
    cargo = shutil.which("cargo")
    if not cargo:
        raise SystemExit(
            "Install Cargo before building the CPU kernel."
        )
    source = Path(__file__).resolve().parent
    destination = kernel_path()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        work = Path(tmp)
        (work / "src").mkdir()
        for name in ("Cargo.toml", "Cargo.lock", "src/lib.rs"):
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
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        (work / "target/release" / ("liboptical_coefficients" + suffix)).replace(destination)
    print(f"Built {destination} in {perf_counter()-start:.3f} s")


if __name__ == "__main__":
    main()
