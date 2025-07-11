#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import find_packages, setup

setup(
    name="torchrec_embcache",
    version="0.2.0",
    package_data={
        'torchrec_embcache': ['*.so*']
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.7",
    author="Huawei Inc",
    description="Embedding cache implementation",
    zip_safe=False,
)