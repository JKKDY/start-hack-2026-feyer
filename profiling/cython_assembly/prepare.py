"""Build the Cython assembly experiment in a fresh, isolated checkout."""

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SETUP_SCRIPT = """
import sys
from setuptools import Extension, setup
from Cython.Build import cythonize

fp_args = ["/fp:strict"] if sys.platform == "win32" else ["-ffp-contract=off"]
setup(ext_modules=cythonize([
    Extension("differometor._aot_assembly_kernel", ["differometor/_aot_assembly_kernel.pyx"],
              extra_compile_args=fp_args),
], compiler_directives={"language_level": 3}))
"""


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New directory for the candidate snapshot")
    parser.add_argument("--ref", default="HEAD", help="Source revision to stage before applying the spike")
    args = parser.parse_args()
    destination = args.destination.resolve()
    revision = git("rev-parse", "--verify", f"{args.ref}^{{commit}}").decode().strip()
    archive = git("-c", "core.autocrlf=false", "archive", "--format=tar", revision)
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as snapshot:
        snapshot.extractall(destination, filter="data")
    if (destination / "benchmark.py").read_bytes() != git("show", "3b538f0:benchmark.py"):
        raise ValueError("The selected commit must carry the original benchmark.py")

    package = destination / "differometor"
    simulator = package / "simulate.py"
    code = simulator.read_text(encoding="utf-8")
    begin = code.index("        coefficients = jnp.broadcast_to(packed_signal.template,")
    end = code.index("        nnz = signal_sparsity[0].size", begin)
    assembly = "        coefficients = _cython_assemble(packed_signal, updates, carrier_coupling)\n"
    code = code[:begin] + assembly + code[end:]
    anchor = "from differometor.static import StaticArray, make_static, unwrap_static"
    if code.count(anchor) != 1:
        raise ValueError("The selected simulator does not match the spike's assembly boundary")
    code = code.replace(anchor, anchor + "\nfrom differometor._aot_assembly_spike import assemble as _cython_assemble")
    simulator.write_text(code, encoding="utf-8")
    shutil.copyfile(HERE / "assembly.pyx", package / "_aot_assembly_kernel.pyx")
    shutil.copyfile(HERE / "bridge.py", package / "_aot_assembly_spike.py")
    (destination / "build_assembly_spike.py").write_text(SETUP_SCRIPT, encoding="utf-8")
    subprocess.run([sys.executable, "build_assembly_spike.py", "build_ext", "--inplace"],
                   cwd=destination, check=True)


if __name__ == "__main__":
    main()
