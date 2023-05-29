#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import os

import tensorflow as tf

from mx_rec.util.constants import HOST_PIPELINE_OPS_LIB_PATH


def import_host_pipeline_ops():
    host_pipeline_ops_lib_path = os.getenv(HOST_PIPELINE_OPS_LIB_PATH)
    if host_pipeline_ops_lib_path and os.path.exists(host_pipeline_ops_lib_path):
        logging.debug(f"Using the HOST_PIPELINE_OPS_LIB_PATH '{host_pipeline_ops_lib_path}' to get ops lib.")
        return tf.load_op_library(host_pipeline_ops_lib_path)
    elif os.path.exists(
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
                         'mx_rec/libasc/libasc_ops.so')):
        default_so_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
            'mx_rec/libasc/libasc_ops.so')
        logging.debug(f"Using the DEFAULT PATH '{default_so_path}' to get ops lib.")
        return tf.load_op_library(default_so_path)
    else:
        raise ValueError("Invalid host pipeline ops lib path. Please check if libasc_ops.so exists or corrected "
                         "configured")

