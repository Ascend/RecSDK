#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import re
import logging


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
    logging.warning(f"The table name '%s' contains invalid characters. "
                    f"The system automatically remove invalid characters. "
                    f"The table name was changed to '%s'", name, fix_name)
    return fix_name
