import sys
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    "differometor._native_components",
    ["differometor/_native_components.cpp"],
    depends=["differometor/_native_components.inc"],
    extra_compile_args=["/fp:strict"] if sys.platform == "win32" else ["-ffp-contract=off"],
    optional=True,
)])
