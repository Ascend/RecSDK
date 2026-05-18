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

import os
import sys
import zipfile
import subprocess
import glob
import shutil
import shlex
import platform
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.build_py import build_py

SDK_VERSION = os.environ.get("RECSDK_VERSION", "26.1.0")
class custom_build_py(build_py):
    def run(self):
        super().run()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(current_dir, "_mxrec_merged_src")
        if os.path.exists(src_dir):
            for root, dirs, files in os.walk(src_dir):
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

def install_requirements():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(current_dir, "requirements.txt")
    
    if not os.path.exists(req_file):
        print(f"No requirements.txt found in {current_dir}, skipping.")
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

def install_horovod():
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

def detect_tf_version():
    try:
        cmd = [sys.executable, "-c", "import tensorflow; print(tensorflow.__version__)"]
        res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, encoding='utf-8').strip()
        return res
    except:
        try:
            res = subprocess.check_output(["python3", "-c", "import tensorflow; print(tensorflow.__version__)"],
                                            stderr=subprocess.DEVNULL, encoding='utf-8').strip()
        except:
            print(f"Error: Cannot detect tf version...")
            return "UNKNOWN"

def detect_npu_chip():
    # 优先检查手动指定的内核类型
    manual_core = os.environ.get("CORE_TYPE", "").upper()
    if manual_core in ["A2", "A3", "A5"]:
        return manual_core

    # 优先检查环境变量
    soc = os.environ.get("SOC_VERSION")
    if not soc:
        # 通过 npu-smi info -m 检测
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

    # 映射 SOC 到设备类型
    if soc.startswith("Ascend95"):
        return "A5"
    elif soc.startswith("Ascend910B"):
        return "A2"
    elif soc.startswith("Ascend910_93"):
        return "A3"

    print(f"Warning: Unknown SOC version: {soc}")
    return "UNKNOWN"

current_dir = os.path.dirname(os.path.abspath(__file__))
packages = find_packages()
package_dir = {}
data_files = []
merged_tmp_dir = os.path.join(current_dir, "_mxrec_merged_src")

is_building = any(arg in sys.argv for arg in ['install', 'bdist_wheel', 'build', 'develop', 'egg_info'])

if is_building:
    install_requirements()
    tf_version = detect_tf_version()

    if tf_version.startswith("1."):
        whl_dir = "mindxsdk-mxrec/tf1_whl"
    elif tf_version.startswith("2."):
        whl_dir = "mindxsdk-mxrec/tf2_whl"
    else:
        sys.exit(f"Error: Unsupported tensorflow version: {tf_version}.")

    full_whl_dir = os.path.join(current_dir, whl_dir)

    if not os.path.exists(full_whl_dir):
        sys.exit(f"Error: Cannot find directory of whl_file: {full_whl_dir}.")

    whl_paths = glob.glob(os.path.join(full_whl_dir, "*.whl"))
    if not whl_paths:
        sys.exit(f"Error: Cannot find any whl file.")
    
    print(f"Debugging full_whl_dir = {full_whl_dir}")
    print(f"Debugging whl_paths = {whl_paths}")

    if not os.path.exists(merged_tmp_dir):
        os.makedirs(merged_tmp_dir)
    
    for idx, whl_path in enumerate(whl_paths):
        print(f"  -> [{idx+1}/{len(whl_paths)}] unzipping: {os.path.basename(whl_path)}")
        with zipfile.ZipFile(whl_path, 'r') as zip_ref:
            zip_ref.extractall(merged_tmp_dir)

    extracted_pkgs = find_packages(where=merged_tmp_dir)
    for pkg in extracted_pkgs:
        if pkg not in packages:
            packages.append(pkg)
            top_level = pkg.split('.')[0]
            if top_level not in package_dir:
                package_dir[top_level] = os.path.join("_mxrec_merged_src", top_level)
                print(f"  -> Linked Python package: {top_level} (and its subpackages)")

class CustomInstallCommand(install):
    def run(self):
        super().run()
        install_horovod()
        chip_type = detect_npu_chip()
        
        if chip_type == "UNKNOWN":
            print("Warning: Matched UNKNOWN chip type, skipped ops installing.")
            return

        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_ops_dir = os.path.join(current_dir, "npu-ops", chip_type, "recsdk-npu-ops", "recsdk_ops")
        
        if not os.path.exists(target_ops_dir):
            print("Warning: Skipped ops installing.")
            return
            
        matched_ops = glob.glob(os.path.join(target_ops_dir, "*.run"))
        if not matched_ops:
            print(f"Warning: Matched chip type: {chip_type}, but no op in {target_ops_dir}")
            return
            
        print(f"Ready for installing ops for {chip_type}.")
        for op_path in matched_ops:
            os.chmod(op_path, 0o755)
            try:
                subprocess.check_call(["bash", op_path])
            except subprocess.CalledProcessError:
                print(f"Error: failed to install op {os.path.basename(op_path)}")
            
arch = platform.machine().lower()
if arch in ["aarch64", "arm64"]:
    arch_suffix = "aarch64"
elif arch in ["x86_64", "amd64"]:
    arch_suffix = "x86_64"
else:
    arch_suffix = f"{arch}"
package_version = SDK_VERSION

setup(
    name='tf_rec_v1',
    version=package_version,
    description='mx_rec wrapper package with dynamic TF version detection',
    install_requires=[],
    cmdclass={
        'install': CustomInstallCommand,
        'build_py': custom_build_py,
    },
    packages=packages,
    package_dir=package_dir,
    data_files=data_files,
    include_package_data=True,
    package_data={'': ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*', '*/*/*/*/*/*']}
)