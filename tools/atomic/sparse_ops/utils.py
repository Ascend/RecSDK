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

"""
utils
"""
from __future__ import absolute_import
import tensorflow as tf
from mpi4py import rc

tf.get_logger().setLevel("ERROR")
rc.initialize = False  # if = True, The Init is done when "from mpi4py import MPI" is called


def ops():
    """
    返回emb相关的算子
    """
    return tf.load_op_library("libcust_ops.so")


def dataset_ops():
    """
    返回emb相关的算子
    """
    return tf.load_op_library("libasc_dataset_ops.so")
