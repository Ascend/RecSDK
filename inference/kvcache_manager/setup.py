import os
import shutil
import subprocess
import sys
from pathlib import Path
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext

library_name = "recsys_kvcache_manager_npu"
root_path: Path = Path(__file__).resolve().parent


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        cmake_path = shutil.which("cmake")
        if cmake_path is None:
            raise RuntimeError("CMake must be installed to build this extension")
        subprocess.check_output([cmake_path, "--version"])
        super().run()

    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        os.makedirs(extdir, exist_ok=True)

        # The defaults for SOC_VERSION and ASCEND_CANN_PACKAGE_PATH live in
        # CMakeLists.txt, which reads the very same environment variables. Only
        # forward them when the caller actually set one, so the default value is
        # defined in exactly one place.
        soc_version = os.environ.get("SOC_VERSION")
        ascend_home = os.environ.get("ASCEND_CANN_PACKAGE_PATH", os.environ.get("ASCEND_HOME"))
        max_jobs = os.environ.get("MAX_COMPILE_THREADS", "8")

        try:
            import pybind11

            pybind11_dir = pybind11.get_cmake_dir()
        except ImportError:
            raise RuntimeError("pybind11 is required: pip install pybind11")

        npu_inc, npu_lib = _torch_npu_dirs()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-Dpybind11_DIR={pybind11_dir}",
            f"-DCMAKE_PREFIX_PATH={pybind11_dir}",
        ]
        if soc_version:
            cmake_args.append(f"-DSOC_VERSION={soc_version}")
        if ascend_home:
            cmake_args.append(f"-DASCEND_CANN_PACKAGE_PATH={ascend_home}")
        if npu_inc:
            cmake_args.append(f"-DTORCH_NPU_INCLUDE_DIR={npu_inc}")
        if npu_lib:
            cmake_args.append(f"-DTORCH_NPU_LIB_DIR={npu_lib}")

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        cmake_path = shutil.which("cmake")
        if cmake_path is None:
            raise RuntimeError("CMake must be installed to build this extension")
        subprocess.check_call([cmake_path, ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(
            [cmake_path, "--build", ".", "--parallel", max_jobs, "--verbose"],
            cwd=build_temp,
        )


def get_version():
    try:
        git_path = shutil.which("git")
        if git_path is None:
            raise RuntimeError("Git is not available")
        git_sha = subprocess.check_output([git_path, "rev-parse", "HEAD"], cwd=str(root_path)).decode("ascii").strip()
    except Exception:
        git_sha = "Unknown"
    return os.environ.get("BUILD_VERSION", "0.3.0"), git_sha


def _torch_npu_dirs():
    try:
        import torch_npu
    except ImportError:
        return None, None
    npu_root = os.path.dirname(os.path.abspath(torch_npu.__file__))
    inc = os.path.join(npu_root, "include")
    lib = os.path.join(npu_root, "lib")
    return (inc if os.path.isdir(inc) else None, lib if os.path.isdir(lib) else None)


package = find_packages(exclude=("*test",))
version, sha = get_version()

setup(
    name=library_name,
    version=version,
    author="Recsys Team",
    description="Ascend A5 KVCache Manager (phase1: native_host)",
    packages=package,
    ext_modules=[CMakeExtension("kvcache_npu_cpp", str(root_path))],
    cmdclass={"build_ext": CMakeBuild},
    license="Apache-2.0",
    python_requires=">=3.9",
    install_requires=["torch", "torch_npu", "pybind11"],
)
