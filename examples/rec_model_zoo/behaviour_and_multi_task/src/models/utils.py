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
import stat
import re
import glob
import json
from typing import Dict, List


def get_third_nearest_checkpoint(path):
    """
    Retrieves the third most recent checkpoint file from the specified directory.

    This function searches for checkpoint files in the given directory that match the pattern
    'model.ckpt-*.index', extracts the version numbers from the filenames, sorts them in ascending
    order, and returns the path to the third most recent checkpoint file.

    Args:
        path (str): The directory path where checkpoint files are stored.

    Returns:
        str: The full path to the third most recent checkpoint file.

    Raises:
        IndexError: If there are fewer than three checkpoint files in the directory.

    Example:
        >>> get_third_nearest_checkpoint('/path/to/checkpoints')
        '/path/to/checkpoints/model.ckpt-12345'
    """
    filenames = glob.glob(os.path.join(path, 'model.ckpt-*.index'))
    pattern = re.compile(r'model.ckpt-(.*?).index', re.S)
    versions = []
    for filename in filenames:
        versions += [int(re.findall(pattern, filename)[0])]
    versions = sorted(versions)
    return os.path.join(path, 'model.ckpt-' + str(versions[-3]))


def json_file_load(json_name: str, json_path: str) -> dict:
    """
    Load a JSON file from the specified path.
    """
    flags = os.O_RDONLY
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    try:
        with os.fdopen(os.open(json_path, flags, modes), "r") as fp:
            json_re = json.load(fp)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{json_name} file not found: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading {json_name} file: {e}") from e

    return json_re


def dump_pred_prob(preds: List[Dict[str, float]], data_dir: str) -> None:
    """
    Dump the prediction results to a file.

    Args:
        preds (List[Dict[str, float]]): List of prediction results.
        data_dir (str): data save dir.

    Returns:
        None
    """
    flags = os.O_WRONLY | os.O_TRUNC | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    pred_path = os.path.join(data_dir, "pred.txt")
    with os.fdopen(os.open(pred_path, flags, modes), "w") as fo:
        for prob in preds:
            fo.write("%f\n" % (prob['prob']))


def dump_pred_multi(preds, data_dir):
    """
    Dump the prediction results to a file.
    """
    flags = os.O_WRONLY | os.O_TRUNC | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    pred_path = os.path.join(data_dir, "pred.txt")
    with os.fdopen(os.open(pred_path, flags, modes), "w") as fo:
        for prob in preds:
            fo.write("%f\t%f\t%f\n" % (prob['ctr'], prob['cvr'], prob['ctcvr']))
