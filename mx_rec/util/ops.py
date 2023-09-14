#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os

import tensorflow as tf

from mx_rec.util.log import logger


def import_host_pipeline_ops():
    if os.path.exists(
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
                         'mx_rec/libasc/libasc_ops.so')):
        default_so_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
            'mx_rec/libasc/libasc_ops.so')
        logger.debug("Using the DEFAULT PATH '%s' to get ops lib.", default_so_path)
        return tf.load_op_library(default_so_path)
    else:
        raise ValueError("Please check if libasc_ops.so exists (mxRec correctly installed)")
