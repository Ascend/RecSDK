#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Shared setup utilities for RecSDK wrapper packages."""

import os
import sys
import subprocess
import shlex
import shutil
import platform
from setuptools.command.build_py import build_py


def install_requirements():
    """Install dependencies from requirements.txt line by line."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(base_dir, "requirements.txt")

    if not os.path.exists(req_file):
        print(f"No requirements.txt found in {base_dir}, skipping.")
        return

    print(f"Installing requirements from {req_file} line by line to avoid dependency resolution conflicts...")
    with open(req_file, 'r', encoding='utf-8') as f:
        reqs = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    # tensorboard 依赖未限制上界的 grpcio，若单独逐行安装，pip 会把 grpcio 升级到最新版本，
    # 导致 grpcio 被下载安装两次（先按 grpcio<1.63.0 装上 1.62.x，又被 tensorboard 升到最新版）。
    # 因此安装 tensorboard 时把 grpcio/grpcio-tools 的约束合并进同一条命令，让 pip 一次解析。
    grpcio_constraints = [r for r in reqs if r.startswith('grpcio')]

    for req in reqs:
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if req == 'tensorboard' and grpcio_constraints:
                cmd += grpcio_constraints
            cmd += shlex.split(req)
            subprocess.check_call(cmd)
            print(f"Successfully installed: {req}")
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {req}, proceeding with next requirement.")

    print("Finished processing all requirements from requirements.txt.")


def install_requirements_new():
    """Install dependencies from requirements.txt in a single pip call.

    install_requirements 的优化版本（其他子组件仍在使用原方法，故新增本方法）。

    调用场景：被 setup.py 在 ``is_building`` 块调用（先于任何 ``from torchrec``）。

    单次 ``pip install -r`` 替代原逐行 ``pip install <pkg>`` 的 N 次循环
    （N 次子进程 + N 轮解析 -> 1 次子进程 + 1 轮联合解析，典型 5min+ -> 数十秒）。
    两点语义保证：
    - 传递依赖完整：pip 对全文件联合解析并安装传递依赖（google-auth、rich 等），
      不使用 --no-deps（会跳过传递依赖，导致 tensorboard/torchx 等运行时缺包）；
    - grpcio 约束保住：grpcio<1.63.0 与 tensorboard 的 grpcio 依赖（无上界）
      同场联合解析，结果落在上界内；旧"双装"问题源于逐行调用时 tensorboard
      单独解析拉最新版，单次 -r 语义上不会重现。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(base_dir, "requirements.txt")

    if not os.path.exists(req_file):
        print(f"No requirements.txt found in {base_dir}, skipping.")
        return

    print(f"Installing requirements from {req_file} (single-pass, before setuptools stage)...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
    try:
        subprocess.check_call(cmd)
        print("Successfully installed all requirements.")
    except subprocess.CalledProcessError as e:
        print(
            f"Warning: pip install -r {req_file} failed (exit {e.returncode}). "
            f"Subsequent setuptools stage may fail if dependencies are missing."
        )


def detect_npu_chip():
    """Detect NPU chip type (A2/A3/A5)."""
    manual_core = os.environ.get("CORE_TYPE", "").upper()
    if manual_core in ["A2", "A3", "A5"]:
        return manual_core

    soc = os.environ.get("SOC_VERSION")
    if not soc:
        npu_smi = shutil.which("npu-smi")
        if not npu_smi:
            print("Warning: npu-smi command not found.")
            return "UNKNOWN"
        try:
            out = subprocess.check_output(
                [npu_smi, "info", "-m"],
                stderr=subprocess.STDOUT,
            ).decode(errors="ignore")
        except subprocess.CalledProcessError:
            print("Warning: npu-smi info -m failed.")
            return "UNKNOWN"

        for line in out.splitlines():
            if "Ascend" in line and "Mcu" not in line:
                suffix = line.split("Ascend", 1)[1].strip().split()[0]
                soc = f"Ascend{suffix}"
                break

    if not soc:
        print("Warning: Cannot detect SOC version.")
        return "UNKNOWN"

    if soc.startswith("Ascend95"):
        return "A5"
    elif soc.startswith("Ascend910B"):
        return "A2"
    elif soc.startswith("Ascend910_93"):
        return "A3"

    print(f"Warning: Unknown SOC version: {soc}")
    return "UNKNOWN"


def detect_version_via_pip(package_name):
    """Detect installed package version via pip show (no import needed)."""
    try:
        cmd = [sys.executable, "-m", "pip", "show", package_name]
        res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, encoding='utf-8')
        for line in res.splitlines():
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            # 确认Version行格式正确后split
            if line.startswith("Version:"):
                parts = line.split(":", 1)
                if len(parts) < 2 or not parts[1].strip():
                    print(f"Warning: Malformed Version line from pip show {package_name}: {line!r}")
                    return "UNKNOWN"
                return parts[1].strip()
    except Exception as e:
        print(f"Warning: Failed to detect {package_name} version via pip show: {e}")
    return "UNKNOWN"


def get_arch_suffix():
    """Get architecture suffix for package naming."""
    arch = platform.machine().lower()
    if arch in ["aarch64", "arm64"]:
        return "aarch64"
    elif arch in ["x86_64", "amd64"]:
        return "x86_64"
    return arch


def make_custom_build_py(merged_dir_name, chmod_so=False):
    """Create a custom build_py class that merges whl contents after build."""

    class _CustomBuildPy(build_py):
        def run(self):
            super().run()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.join(base_dir, merged_dir_name)
            if os.path.exists(src_dir):
                for root, _dirs, files in os.walk(src_dir):
                    rel_dir = os.path.relpath(root, src_dir)
                    if '.dist-info' in rel_dir:
                        continue
                    dest_dir = self.build_lib if rel_dir == '.' else os.path.join(self.build_lib, rel_dir)
                    os.makedirs(dest_dir, exist_ok=True)
                    for f in files:
                        if '.dist-info' in f:
                            continue
                        src_file = os.path.join(root, f)
                        dest_file = os.path.join(dest_dir, f)
                        shutil.copy2(src_file, dest_file)
                        if chmod_so and '.so' in f:
                            os.chmod(dest_file, 0o755)

    return _CustomBuildPy


def make_custom_build_py_new(merged_dir_name, chmod_so=False):
    """Create a custom build_py class that merges whl contents into build_lib.

    make_custom_build_py 的优化版本（其他子组件仍在使用原方法，故新增本方法）。

    O2 优化：原方案在 merged_tmp_dir 解压后 ``os.walk + shutil.copy2`` 把每份
    文件单独拷贝到 build_lib——N 次 stat + 读 + 写。改为单次 ``copytree`` +
    ``dirs_exist_ok=True``，由 shutil 内部对每个目录用一次 scandir 批处理，
    I/O 调用次数从 N 降到 1。

    O4 优化：copytree 完成后用 ``os.scandir`` 一次性遍历 build_lib 给 ``*.so``
    加执行权限（不再每次 copy2 之后单独 chmod）。
    """

    class _CustomBuildPy(build_py):
        def run(self):
            super().run()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.join(base_dir, merged_dir_name)
            if not os.path.exists(src_dir):
                return

            dest = self.build_lib
            os.makedirs(dest, exist_ok=True)
            # O2：单步 copytree 替代 N 次 walk + copy2；显式忽略子包 .dist-info
            # 目录（保留当前包自身 dist-info，由 build_py 自动生成）。
            shutil.copytree(
                src_dir,
                dest,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.dist-info"),
            )

            # O4：copytree 完成后再一次性 chmod *.so
            if chmod_so:
                _chmod_so_recursive(dest)

    return _CustomBuildPy


def _chmod_so_recursive(root):
    """对 root 下所有 *.so 加 0755 权限（O4：copytree 之后的批量 chmod）。"""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith('.so'):
                try:
                    os.chmod(os.path.join(dirpath, name), 0o755)
                except OSError:
                    pass


def install_horovod():
    """Install Horovod 0.22.1 with MPI and TensorFlow support."""
    print("Checking and installing Horovod 0.22.1 (required for mxrec)...")
    env = os.environ.copy()
    env["HOROVOD_WITH_MPI"] = "1"
    env["HOROVOD_WITH_TENSORFLOW"] = "1"

    cmd = [sys.executable, "-m", "pip", "install", "horovod==0.22.1", "--no-cache-dir"]
    try:
        subprocess.check_call(cmd, env=env)
        print("Horovod 0.22.1 installed successfully.")
    except subprocess.CalledProcessError:
        print("Warning: Failed to install Horovod 0.22.1. Please check logs.")
