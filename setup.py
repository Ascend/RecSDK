#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
# Description: setup script.
# Author: MindX SDK
# Create: 2022
# History: NA

import os
import stat
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


packaging_version.Version = NoNormalizeVersion
# Patch safe_version() to prevent version normalization
pkg_resources.safe_version = lambda v: v

try:
    with open("README.md") as file:
        LONG_DESCRIPTION = file.read()
except IOError:
    LONG_DESCRIPTION = ""

env_version = os.getenv("VERSION")
VERSION = env_version if env_version is not None else '5.0.T104'

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

setup(
    name='mx_rec',
    version=VERSION,
    author='HUAWEI Inc',
    description='MindX SDK Recommend',
    long_description=LONG_DESCRIPTION,
    # include mx_rec
    packages=find_packages(
        where='.',
        include=["mx_rec*"]
    ),
    package_dir={},
    # other file
    package_data={'': ['tools/*', 'tools/*/*', '*.yml', '*.sh', '*.so*']},
    # dependency
    python_requires='>=3.7.5'
)
