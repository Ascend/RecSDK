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

    for req in reqs:
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + shlex.split(req)
            subprocess.check_call(cmd)
            print(f"Successfully installed: {req}")
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {req}, proceeding with next requirement.")

    print("Finished processing all requirements from requirements.txt.")


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
