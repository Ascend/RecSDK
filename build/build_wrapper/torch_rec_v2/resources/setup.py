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
# pylint: disable=duplicate-code

import os
import sys
import zipfile
import glob
from setuptools import setup, find_packages
from setuptools.command.install import install

import _setup_common

SDK_VERSION = os.environ.get("RECSDK_VERSION", "26.1.0")

custom_build_py = _setup_common.make_custom_build_py("_torchrec_merged_src", chmod_so=True)


class CustomInstallCommand(install):
    pass


def detect_pt_version():
    return _setup_common.detect_version_via_pip("torch")


current_dir = os.path.dirname(os.path.abspath(__file__))
packages = find_packages()
package_dir = {}
data_files = []
merged_tmp_dir = os.path.join(current_dir, "_torchrec_merged_src")

is_building = any(arg in sys.argv for arg in ['install', 'bdist_wheel', 'build', 'develop', 'egg_info'])

if is_building:
    _setup_common.install_requirements()
    torch_version = detect_pt_version()

    if torch_version.startswith("2.7."):
        whl_dir = "mindxsdk-torchrec/pt2.7_whl"
    else:
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
                print(f"  -> [{idx + 1}/{len(whl_paths)}] unzipping: {os.path.basename(whl_path)}")
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


setup(
    name='torch_rec_v2',
    version=SDK_VERSION,
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
    package_data={'': ['*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*', '*/*/*/*/*/*']},
)
