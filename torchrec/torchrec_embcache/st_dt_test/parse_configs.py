#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import re
from collections import defaultdict

import yaml


def load_config(file_path: str) -> dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_all_configs(config_dir: str) -> dict:
    """
    Load all YAML configuration files from a specified directory.
    """
    configs = defaultdict(list)
    for filename in os.listdir(config_dir):
        if filename.endswith('.yaml'):
            config_path = os.path.join(config_dir, filename)
            config = load_config(config_path)
            filename = filename.split(".")[0]
            if not re.match(r".*_\d+$", filename):
                continue  # Skip files that do not match the expected format
            test_case_name, _, _ = re.split("_(\d+)$", filename)
            configs[test_case_name].append((filename, config))
    return configs