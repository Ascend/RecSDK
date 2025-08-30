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
import shutil


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

if os.path.exists("rec_sdk_common"):
    shutil.rmtree("rec_sdk_common")
shutil.copytree("python", "rec_sdk_common")
setup(
    name='rec_sdk_common',
    version=args.version,
    author='HUAWEI Inc',
    description='MindSDK Recommend',
    long_description=args.discription,
    # include mx_rec
    packages=find_packages(
        where=".",
        include=["rec_sdk_common*"]
    ),
    # other file
    package_data={'': ['tools/*', 'tools/*/*', '*.yml', '*.sh', '*.so*']},
    # dependency
    python_requires='>=3.7.5'
)