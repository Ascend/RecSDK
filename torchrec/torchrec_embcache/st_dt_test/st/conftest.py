#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

from util import generate_test_cases, get_all_configs


CURRENT_FILE = Path(__file__).resolve()
MODULE_NAME = CURRENT_FILE.parent.name


ALL_CONFIGS = get_all_configs(MODULE_NAME)


def pytest_generate_tests(metafunc):
    """
    Pytest hook to generate tests dynamically based on the configurations.
    """
    generate_test_cases(metafunc, ALL_CONFIGS)
