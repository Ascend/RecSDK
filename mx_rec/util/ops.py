#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import os

import tensorflow as tf

from mx_rec.constants.constants import HOST_PIPELINE_OPS_LIB_PATH


def import_host_pipeline_ops():
    if os.path.exists(
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
                         'mx_rec/libasc/libasc_ops.so')):
        default_so_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
            'mx_rec/libasc/libasc_ops.so')
        logging.debug(f"Using the DEFAULT PATH '{default_so_path}' to get ops lib.")
        return tf.load_op_library(default_so_path)
    else:
        raise ValueError("Please check if libasc_ops.so exists (mxRec correctly installed)")
