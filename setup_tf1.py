#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
from setuptools import setup, find_packages
import pkg_resources
from setuptools.extern.packaging import version as packaging_version

script_path = os.getcwd()


# Patch Version class to preserve original version string
class NoNormalizeVersion(packaging_version.Version):
    def __init__(self, version):
        self._orig_version = version
        super().__init__(version)

    def __str__(self):
        return self._orig_version


def safe_version(v):
    return v


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
    VERSION = "7.1.RC1"

INIT_FILE = "mx_rec/__init__.py"
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
tf1_script = os.path.join(script_path, "./build/build_tf1.sh")
res = subprocess.run([tf1_script], shell=False)
if res.returncode:
    raise RuntimeError("compile so files failed!")

setup(
    name='mx_rec',
    version=VERSION,
    author='HUAWEI Inc',
    description='MindSDK Recommend',
    long_description=LONG_DESCRIPTION,
    # include mx_rec
    packages=find_packages(
        where='.',
        include=["mx_rec*"]
    ),
    # other file
    package_data={'': ['tools/*', 'tools/*/*', '*.yml', '*.sh', '*.so*']},
    # dependency
    python_requires='>=3.7.5'
)

move_whl_script = os.path.join(script_path, "./build/move_whl_file_2_pkg_dir.sh")
res = subprocess.run([move_whl_script, "tf1"], shell=False)
if res.returncode:
    raise RuntimeError(f"move tf1 whl file to pkg dir failed!")
