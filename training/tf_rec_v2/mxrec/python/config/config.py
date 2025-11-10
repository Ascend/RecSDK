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

from rec_sdk_common.validator.safe_checker import class_safe_check, str_safe_check
from mxrec.python.constants.constants import (
    CommNodeInfo,
    USE_RANKTABLE,
    USE_FUSION_OP,
    USE_LCCL_ALL2ALL_OP,
    FUSION_OP_TYPE,
    ALL2ALL_OP_TYPE,
)
from mxrec.python.config.parser import TomlParser, parse_env_field
from mxrec.python.config.comm_parser import parse_comm_info
from mxrec.python.config.log_parser import parse_log_level

_MAX_TOML_OPS_PARAMS_LEN = 128


def get_log_level() -> str:
    if TomlParser.get_instance().log_level is None:
        level = parse_log_level()
        TomlParser.get_instance().log_level = level

    return TomlParser.get_instance().log_level


def get_comm_node_info() -> CommNodeInfo:
    if TomlParser.get_instance().comm_node_info is None:
        info = parse_comm_info()
        TomlParser.get_instance().comm_node_info = info

    return TomlParser.get_instance().comm_node_info


def get_use_ranktable() -> bool:
    if TomlParser.get_instance().use_ranktable is None:
        use_ranktable = _parse_use_ranktable()
        TomlParser.get_instance().use_ranktable = use_ranktable

    return TomlParser.get_instance().use_ranktable


def _parse_use_ranktable() -> bool:
    config = TomlParser.get_instance().config
    use_ranktable = parse_env_field(config, USE_RANKTABLE)
    class_safe_check("use_ranktable", use_ranktable, (bool,))
    if not use_ranktable:
        raise ValueError(
            "currently, the use_ranktable only supports True, please check if the toml file is configured correctly"
        )

    return use_ranktable
