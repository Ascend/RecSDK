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
import glob
from typing import List, Dict

import tensorflow as tf

from mx_rec.util.initialize import ConfigInitializer


def check_files_in_directories(root_dir: str, file_patterns: List[str]) -> bool:
    """Check if there are files in the directory that match a specific pattern.

    Args:
        root_dir: Root directory.
        file_patterns: The list of file patterns to be checked.

    Returns: If files matching all patterns exist in each directory, return True; otherwise, return False.

    """
    directories = glob.glob(root_dir)
    is_dir = []
    for directory in directories:
        if not os.path.isdir(directory):
            is_dir.append(False)
            continue

        is_dir.append(True)
        found_patterns = set()
        for pattern in file_patterns:
            full_pattern = os.path.join(directory, "**", pattern)
            matched_files = glob.glob(full_pattern, recursive=True)
            if matched_files:
                found_patterns.add(pattern)

        if len(found_patterns) != len(file_patterns):
            return False

    if not any(is_dir):
        return False
    return True


def get_optimizer_dict_by_table_name(table_name: str) -> Dict[str, tf.Variable]:
    experimental_mode = ConfigInitializer.get_instance().train_params_config.experimental_mode
    if experimental_mode is None:
        return ConfigInitializer.get_instance().optimizer_config.get_optimizer_by_table_name(table_name)

    is_training = experimental_mode == tf.compat.v1.estimator.ModeKeys.TRAIN
    optimizer = ConfigInitializer.get_instance().optimizer_config.get_optimizer_by_table_name(
        table_name, is_training=is_training
    )
    return optimizer
