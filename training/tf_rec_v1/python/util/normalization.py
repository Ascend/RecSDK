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

import re

from rec_sdk_common.log import logger


def fix_invalid_table_name(name):
    """
    校验table name字符串中是否含有特殊字符，如有，替换为下划线
    :param name: table name
    :return : the fixed table name
    """
    pattern = "^[0-9A-Za-z_]+$"
    if re.match(pattern, name):
        return name
    fix_name = re.sub(r'\W+', '', name)
    if not fix_name:
        raise ValueError(f"The table name '{name}' doesn't contain valid character, "
                         f"according to the rule '{pattern}'")
    logger.warning("The table name '%s' contains invalid characters. The system automatically "
                   "remove invalid characters. The table name was changed to '%s'", name, fix_name)
    return fix_name
