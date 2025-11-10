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

from rec_sdk_common.validator.safe_checker import str_safe_check
from mxrec.python.constants.constants import LogParams
from mxrec.python.config.parser import TomlParser, parse_env_field


def parse_log_level() -> str:
    config = TomlParser.get_instance().config
    log_level = parse_env_field(config, LogParams.LOG_LEVEL.value)
    str_safe_check("log_level", log_level)

    level_list = (
        LogParams.DEBUG.value,
        LogParams.INFO.value,
        LogParams.WARNING.value,
        LogParams.ERROR.value,
        LogParams.CRITICAL.value,
    )
    if log_level not in level_list:
        raise ValueError(
            f"log level is invalid, only {level_list} are allowed, but got {log_level}"
        )

    return log_level
