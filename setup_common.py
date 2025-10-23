#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

UNSUPPORTED_FILE_MODE_MASK = 0o022


def validate_read_file(read_file_path):
    """
    Validate file before reading，including validating soft link, file size
    :param read_file_path: the file path to be validated
    """
    # para type check
    if not isinstance(read_file_path, str):
        raise ValueError("parameter value's type is not str")

    # link file check
    if (os.path.abspath(read_file_path) != os.path.realpath(read_file_path)):
        raise ValueError(f"soft link or relative path: {read_file_path} should not be in the path parameter")

    stat_info = os.stat(read_file_path)
    # user group check
    process_uid = os.geteuid()
    process_gid = os.getegid()
    if not ((process_uid == stat_info.st_uid) or (process_gid == stat_info.st_gid)):
        raise ValueError(f"Invalid log file user or group, path: {read_file_path}.")

    # file mode check
    mode = stat.S_IMODE(stat_info.st_mode)
    if ((mode & UNSUPPORTED_FILE_MODE_MASK) != 0):
        raise ValueError(f"Current file:{read_file_path}, mode {oct(mode)} is unsupported")


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
    try:
        validate_read_file(build_script)
    except ValueError as e:
        raise ValueError(f"Build script validation failed: {e}") from e
    
    res = subprocess.run([build_script], shell=False)
    if res.returncode:
        raise RuntimeError("compile so files failed!")

    common_dir = os.path.join(script_path, "training/common")
    subprocess.run(["python3", "setup.py", "bdist_wheel", f"--version={VERSION}",
                    f"--discription={LONG_DESCRIPTION}"], cwd=common_dir, shell=False)
    tf_rec_v1_dir = os.path.join(script_path, "training/tf_rec_v1")
    subprocess.run(["python3", "setup.py", "bdist_wheel", f"--version={VERSION}",
                    f"--discription={LONG_DESCRIPTION}"], cwd=tf_rec_v1_dir, shell=False)
    move_whl_script = os.path.join(script_path, "./build/move_whl_file_2_pkg_dir.sh")
    res = subprocess.run([move_whl_script, build_type], shell=False)
    if res.returncode:
        raise RuntimeError(f"move whl file to pkg dir failed!")