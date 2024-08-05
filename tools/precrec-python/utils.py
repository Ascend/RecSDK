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

import logging
import sys
import os
import re
import json
from typing import Tuple

import numpy as np


DEBUG_LEVEL = -1
INFO_LEVEL = 0
WARNING_LEVEL = 1
ERROR_LEVEL = 2

KEY_NOT_FOUND = "key not found"

LOG_LEVEL_DICT = {
    DEBUG_LEVEL: logging.DEBUG,
    INFO_LEVEL: logging.INFO,
    WARNING_LEVEL: logging.WARNING,
    ERROR_LEVEL: logging.ERROR,
}


def init_logger(log_level: int = 0):
    """
    Init logger for given log level.
    """
    logging.basicConfig(
        level=LOG_LEVEL_DICT[log_level],
        format="[PrecCheck][%(levelname)s][%(asctime)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)
    logger.info("logger init succeed.\n")


def parse_dict(par_dict: dict, par_key: str):
    """
    Parse sub dict for given key value.
    """
    logging.debug("Paring dict for %s ......", par_key.upper())

    if not par_dict.get(par_key):
        raise ValueError(f"Parse dict failed because key {par_key} not exist!")

    cur_dict = par_dict[par_key]

    for key, value in cur_dict.items():
        logging.debug("%s: %s", key.upper(), value)

    return cur_dict


def parse_json_to_dict(json_path: str) -> any:
    """
    Parse json file to dict.
    """
    if not os.path.exists(json_path):
        raise ValueError(f"Parse json failed.{json_path} not exist!")

    with open(json_path, "r") as f:
        json_info = json.load(f)
    return json_info


def nested_dict_to_str(cur_item, pre_key_str=None):
    """
    Convert nested dict to str which is easier to understand for representation.
    """
    result = ""

    if isinstance(cur_item, dict):
        result = "\n"
        for key, value in cur_item.items():
            key = str(key)
            if pre_key_str:
                key = ".".join([pre_key_str, key])
            result += f"{key}: "
            result += nested_dict_to_str(value, key)
        return result

    if isinstance(cur_item, list):
        for item in cur_item:
            result += nested_dict_to_str(item, pre_key_str) + "\n"
        return result

    if isinstance(cur_item, np.ndarray) or isinstance(cur_item, tuple):
        result += f"\n{cur_item}\n\n"
    else:
        result += f"{cur_item}\n"

    return result


def parse_input_param() -> Tuple[str, str]:
    """
    Parse input param for python program.
    """
    # Check whether there are enough parameters.
    input_param_num = len(sys.argv)
    if input_param_num != 3:
        raise ValueError(
            f"Input param must have 2 values, but {input_param_num - 1} was given."
        )

    test_path = sys.argv[1]
    golden_path = sys.argv[2]

    logging.debug("Test_Path is: %s\n Golden_Path is: %s", test_path, golden_path)
    return test_path, golden_path


def validate_path(path: str, path_desc: str, regex_str: str = None):
    """
    Check path match given reget str and file whether exits.
    """
    # Check if the path is legal.
    if not os.path.isabs(path):
        raise ValueError(
            f"[{path_desc}] invalid: {path}."
            + f"The input is not a valid path, please check."
        )

    # Check if the file or path exits.
    if not os.path.exists(path):
        raise ValueError(f"[{path_desc}] does not exist: {path}, please check.")

    if regex_str:
        regex_pattern = re.compile(regex_str)
        dir_name = os.path.basename(path)
        if not regex_pattern.match(dir_name):
            raise ValueError(
                f"[{path_desc}] dir name invalid: {dir_name}"
                + "The file may have been tampered, please check."
            )


def sc_get_key_value(cur_dict: dict, cur_key):
    value = cur_dict.get(cur_key, KEY_NOT_FOUND)
    if value == KEY_NOT_FOUND:
        raise ValueError(f"Key：{cur_key} not found in current dict.")
    return value
