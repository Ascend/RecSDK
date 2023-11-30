#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
from types import ModuleType

import tensorflow as tf

from mx_rec.util.log import logger
from mx_rec.constants.constants import LIBASC_OPS_SO


def import_host_pipeline_ops(so_pkg_name: str = LIBASC_OPS_SO) -> ModuleType:
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
