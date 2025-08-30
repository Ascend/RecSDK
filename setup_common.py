#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
import re
import stat
import subprocess
import shutil
from pathlib import Path

from setuptools import setup, find_packages
import pkg_resources
from setuptools.extern.packaging import version as packaging_version


# Patch Version class to preserve original version string
class NoNormalizeVersion(packaging_version.Version):
    def __init__(self, version):
        self._orig_version = version
        super().__init__(version)

    def __str__(self):
        return self._orig_version


def safe_version(v):
    return v


def run_setup(build_script_name, build_type):
    script_path = Path(__file__).parent.absolute()

    packaging_version.Version = NoNormalizeVersion
    # Patch safe_version() to prevent version normalization
    pkg_resources.safe_version = safe_version

    try:
        with open("README.md") as file:
            LONG_DESCRIPTION = file.read()
    except IOError:
        LONG_DESCRIPTION = ""

    env_version = os.getenv("VERSION")
    if env_version and re.match(r'^[0-9]+\.[0-9]+\.[A-Za-z]+[0-9]+$', env_version):
        VERSION = env_version
    else:
        VERSION = "7.2.RC1"

    INIT_FILE = "training/tf_rec_v1/python/__init__.py"
    with open(INIT_FILE, 'r') as file:
        lines = file.readlines()

    for idx, line in enumerate(lines):
        if "__version__ = " not in line:
            continue
        lines[idx] = f"__version__ = '{VERSION}'\n"
        break

    FLAG = os.O_WRONLY | os.O_TRUNC
    MODE = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(INIT_FILE, FLAG, MODE), 'w') as out:
        out.writelines(lines)

    # compile so files
    build_script = os.path.join(script_path, build_script_name)
    res = subprocess.run([build_script], shell=False)
    if res.returncode:
        raise RuntimeError("compile so files failed!")

    common_dir = os.path.join(script_path, "training/common")
    subprocess.run(["python3", "setup.py", "bdist_wheel", f"--version={VERSION}",
                    f"--discription={LONG_DESCRIPTION}"], cwd=common_dir)
    tf_rec_v1_dir = os.path.join(script_path, "training/tf_rec_v1")
    subprocess.run(["python3", "setup.py", "bdist_wheel", f"--version={VERSION}",
                    f"--discription={LONG_DESCRIPTION}"], cwd=tf_rec_v1_dir)
    move_whl_script = os.path.join(script_path, "./build/move_whl_file_2_pkg_dir.sh")
    res = subprocess.run([move_whl_script, build_type], shell=False)
    if res.returncode:
        raise RuntimeError(f"move whl file to pkg dir failed!")