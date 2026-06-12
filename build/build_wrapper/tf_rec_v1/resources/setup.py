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
import subprocess
import glob
from setuptools import setup, find_packages
from setuptools.command.install import install

import _setup_common

SDK_VERSION = os.environ.get("RECSDK_VERSION", "26.1.0")

custom_build_py = _setup_common.make_custom_build_py("_mxrec_merged_src")


class CustomInstallCommand(install):
    def run(self):
        super().run()
        _setup_common.install_horovod()
        chip_type = _setup_common.detect_npu_chip()

        if chip_type == "UNKNOWN":
            print("Warning: Matched UNKNOWN chip type, skipped ops installing.")
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_ops_dir = os.path.join(base_dir, "npu-ops", chip_type, "recsdk-npu-ops", "recsdk_ops")

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


def detect_tf_version():
    return _setup_common.detect_version_via_pip("tensorflow")


current_dir = os.path.dirname(os.path.abspath(__file__))
packages = find_packages()
package_dir = {}
data_files = []
merged_tmp_dir = os.path.join(current_dir, "_mxrec_merged_src")

is_building = any(arg in sys.argv for arg in ['install', 'bdist_wheel', 'build', 'develop', 'egg_info'])

if is_building:
    _setup_common.install_requirements()
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
        sys.exit("Error: Cannot find any whl file.")

    print(f"Debugging full_whl_dir = {full_whl_dir}")
    print(f"Debugging whl_paths = {whl_paths}")

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
                package_dir[top_level] = os.path.join("_mxrec_merged_src", top_level)
                print(f"  -> Linked Python package: {top_level} (and its subpackages)")


setup(
    name='tf_rec_v1',
    version=SDK_VERSION,
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
    package_data={'': ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*', '*/*/*/*/*/*']},
)
