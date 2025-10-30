#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def check(check_flag: bool, error_msg: str):
    """
    Check if check_flag is True or False. If check_flag is False, will raise an ValueError.
    """
    if not check_flag:
        raise ValueError(error_msg)
