import platform
import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

args = ["/O2"] if sys.platform == "win32" else ["-O3"]
if platform.machine().lower() in ("x86_64", "amd64") and sys.platform != "win32":
    args.append("-mavx2")  # AVX2 kernels in hnsw.hpp (remove for very old CPUs)

ext = Pybind11Extension(
    "vsearch", ["src/bindings.cpp"], include_dirs=["include"],
    cxx_std=17, extra_compile_args=args,
)
setup(name="vsearch", version="0.1.0", ext_modules=[ext], cmdclass={"build_ext": build_ext},
      description="HNSW vector search engine in C++ with Python bindings", python_requires=">=3.8")
