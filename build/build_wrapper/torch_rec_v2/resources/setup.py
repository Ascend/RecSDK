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
        src_dir = os.path.join(current_dir, "_torchrec_merged_src")
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
                    if '.so' in f:
                        os.chmod(dest_file, 0o755)

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

def detect_pt_version():
    try:
        cmd = [sys.executable, "-c", "import torch; print(torch.__version__)"]
        res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, encoding='utf-8').strip()
        return res
    except:
        try:
            res = subprocess.check_output(["python3", "-c", "import torch; print(torch.__version__)"],
                                            stderr=subprocess.DEVNULL, encoding='utf-8').strip()
        except:
            print(f"Error: Cannot detect pt version...")
            return "UNKNOWN"

current_dir = os.path.dirname(os.path.abspath(__file__))
packages = find_packages()
package_dir = {}
data_files = []
merged_tmp_dir = os.path.join(current_dir, "_torchrec_merged_src")

is_building = any(arg in sys.argv for arg in ['install', 'bdist_wheel', 'build', 'develop', 'egg_info'])

if is_building:
    install_requirements()
    torch_version = detect_pt_version()

    if torch_version.startswith("2.7."):
        whl_dir = "mindxsdk-torchrec/pt2.7_whl"   
    else:
        # 兼容性处理，如果版本不匹配但目录下有对应 whl 也可以尝试
        print(f"Warning: Detected PyTorch version {torch_version} might be incompatible. Defaulting to pt2.7_whl.")
        whl_dir = "mindxsdk-torchrec/pt2.7_whl"

    full_whl_dir = os.path.join(current_dir, whl_dir)

    if not os.path.exists(full_whl_dir):
        print(f"Warning: Cannot find directory of whl_file: {full_whl_dir}. Skipping source injection.")
    else:
        whl_paths = glob.glob(os.path.join(full_whl_dir, "*.whl"))
        if not whl_paths:
            print(f"Warning: No .whl files found in {full_whl_dir}")
        else:
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
                        package_dir[top_level] = os.path.join("_torchrec_merged_src", top_level)
                        print(f"  -> Linked Python package: {top_level} (and its subpackages)")

class CustomInstallCommand(install):
    def run(self):
        super().run()
        # 后续可在此添加 SDK 专用的安装后钩子

arch = platform.machine().lower()
if arch in ["aarch64", "arm64"]:
    arch_suffix = "aarch64"
elif arch in ["x86_64", "amd64"]:
    arch_suffix = "x86_64"
else:
    arch_suffix = f"{arch}"
package_version = SDK_VERSION

setup(
    name='torch_rec_v2',
    version=package_version,
    description='torchrec wrapper package with dynamic PyTorch version detection',
    install_requires=[],
    cmdclass={
        'install': CustomInstallCommand,
        'build_py': custom_build_py,
    },
    packages=packages,
    package_dir=package_dir,
    data_files=data_files,
    include_package_data=True,
    package_data={'': ['*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*', '*/*/*/*/*/*']}
)