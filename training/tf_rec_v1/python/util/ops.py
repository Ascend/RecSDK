#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
from types import ModuleType

import tensorflow as tf

from rec_sdk_common.log.log import LoggingProxy as logger
from rec_sdk_common.validator.validator import para_checker_decorator, StringValidator
from mx_rec.constants.constants import LIBREC_TF_NPU_OPS_SO


@para_checker_decorator(check_option_list=[
    ("so_pkg_name", StringValidator, {"min_len": 1, "max_len": 100}, ["check_string_length", "check_whitelist"])
])
def import_host_pipeline_ops(so_pkg_name: str = LIBREC_TF_NPU_OPS_SO) -> ModuleType:
    """
    导入so包.

    Args:
        so_pkg_name: so包的名称
    Returns: 返回用于调用op的module
    """

    so_pkg_path = 'mx_rec/libasc/' + so_pkg_name
    if os.path.exists(
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
                         so_pkg_path)):
        default_so_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
            so_pkg_path)
        logger.debug("Using the DEFAULT PATH '%s' to get ops lib.", default_so_path)
        return tf.load_op_library(default_so_path)
    else:
        raise ValueError(f"Please check if `{so_pkg_name}` exists (mxRec correctly installed).")
