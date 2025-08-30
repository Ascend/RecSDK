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
import argparse
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from packaging.version import Version
import shutil
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="7.2.RC1")
    parser.add_argument("--discription", default="")

    args, unknown = parser.parse_known_args()
    # 只允许特定的setuptools参数通过
    allowed_args = []
    for arg in unknown:
        if arg == "bdist_wheel":
            allowed_args.append(arg)

    sys.argv = [sys.argv[0]] + allowed_args
    return args

args = parse_args()

if os.path.exists("mx_rec"):
    shutil.rmtree("mx_rec")

if os.path.exists("rec_sdk_common"):
    shutil.rmtree("rec_sdk_common")

shutil.copytree("python", "mx_rec")

common_version = Version(args.version)

current_dir = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(current_dir, "..", "common", "rec_sdk_common")
dest_path = os.path.join(current_dir, "rec_sdk_common")

if(os.path.exists(source_path)):
    shutil.copytree(source_path, dest_path)
setup(
    name='mx_rec',
    version=args.version,
    author='HUAWEI Inc',
    description='MindSDK Recommend',
    long_description=args.discription,
    # include mx_rec
    packages=find_packages(
        where=".",
        include=["mx_rec*", "rec_sdk_common*"]
    ),
    # other file
    package_data={'': ['tools/*', 'tools/*/*', '*.yml', '*.sh', '*.so*']},
    # dependency
    python_requires='>=3.7.5'
)
shutil.rmtree("mx_rec")
shutil.rmtree("rec_sdk_common")