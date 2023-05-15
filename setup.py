# Copyright (c) Huawei Technologies Co., Ltd. 2021-2025. All rights reserved.
# Description: build script.
# Author: MindX SDK
# Create: 2022
# History: NA

import os
from setuptools import setup, find_packages

try:
    LONG_DESCRIPTION = open("README.md").read()
except IOError:
    LONG_DESCRIPTION = ""

env_version = os.getenv("VERSION")
VERSION = env_version if env_version is not None else '5.0.T104'

setup(
    name='mx_rec',
    version=VERSION,
    author='HUAWEI Inc',
    url='https://www.hiascend.com/zh/software/mindx-sdk',
    description='MindX SDK Recommend',
    long_description=LONG_DESCRIPTION,
    # include mx_rec
    packages=find_packages(
        where='.',
        include=["mx_rec*"]
    ),
    package_dir={},
    # other file
    package_data={'': ['*.yml', '*.sh', '*.so*']},
    # dependency
    python_requires='>=3.7.5'
)
