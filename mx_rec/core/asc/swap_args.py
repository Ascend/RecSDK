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
import functools
from collections import defaultdict
from enum import Enum


class SwapDataType(Enum):
    CONFIG = "config"
    CONTROL = "control"


def singleton(cls):
    _instance = {}

    @functools.wraps(cls)
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner


@singleton
class SwapArgs:
    def __init__(self):
        self.swap_config_dict = defaultdict(dict)
        self.swap_control_dict = defaultdict(dict)

    def set_data(self, data_type: str, **kwargs):
        if "var_name" not in kwargs:
            raise ValueError("Missing Required key: var_name")
        if "var_channel" not in kwargs:
            raise ValueError("Missing Required key: var_channel")
        var_name = kwargs.pop("var_name")
        var_channel = kwargs.pop("var_channel")

        if data_type == SwapDataType.CONFIG.value:
            self.swap_config_dict[var_name][var_channel] = kwargs
        elif data_type == SwapDataType.CONTROL.value:
            self.swap_control_dict[var_name][var_channel] = kwargs
        else:
            raise ValueError(f"Error data type in swap args: {data_type}")
