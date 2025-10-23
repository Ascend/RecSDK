#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import logging
from enum import Enum
from types import ModuleType

import numpy as np
import tensorflow as tf
from npu_bridge.npu_init import npu_config_proto
import acl

CUSTOM_OPS_SO = "librecsdk_tf_npu_ops.so"


class MemcpyKind(Enum):
    ACL_MEMCPY_HOST_TO_HOST = 0
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2
    ACL_MEMCPY_DEVICE_TO_DEVICE = 3


def acl_init(device_id):
    ret = acl.init()
    if ret != 0:
        logging.info(f"acl init failed, ret: {ret}")
        return
    ret = acl.rt.set_device(device_id)
    if ret != 0:
        logging.info(f"acl set device {device_id} failed ret: {ret}")
        return
    logging.info("acl initialize success.")


def acl_finalize(device_id):
    ret = acl.rt.reset_device(device_id)
    if ret != 0:
        logging.info(f"acl reset device {device_id} failed, ret: {ret}")
        return
    ret = acl.finalize()
    if ret != 0:
        logging.info(f"acl finalize failed, ret: {ret}")
        return
    logging.info("acl finalize success.")


def bind_cpu(affinity):
    os.sched_setaffinity(0, affinity)


def save_data(array, save_path):
    array = np.array(array).flatten()
    array.tofile(save_path)


def read_bin_file(bin_file_path):
    array = np.fromfile(bin_file_path, dtype=np.float32)
    return array


def compare_bin_file(base_bin_file, output_bin_file):
    base = read_bin_file(base_bin_file)
    output = read_bin_file(output_bin_file)

    if base.size != output.size:
        logging.info(f"{output_bin_file} length[{output.size}] not equal {base_bin_file} length[{base.size}]")
        return False
    unequal_indices = np.where(np.abs(base - output) > 2**(-14))
    if len(unequal_indices[0]) == 0:
        logging.info("两个数组完全相等")
        return True
    else:
        # 获取第一个不相等的位置
        first_unequal_index =\
            (unequal_indices[0][0], unequal_indices[1][0]) if len(unequal_indices) > 1 else unequal_indices[0][0]
        logging.info(f"第一个不相等的位置: {first_unequal_index}")
        logging.info(f"arr1[{first_unequal_index}] = {base[first_unequal_index]}, "
              f"arr2[{first_unequal_index}] = {output[first_unequal_index]}")
        return False


def sess_config(execute_type, dump_data=False, dump_path="./dump_output", dump_steps="0|1|2"):
    if execute_type == 'ai_core':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_data_pre_proc"].b = True   # 开启数据预处理下沉到Device侧执行
        custom_op.parameter_map["mix_compile_mode"].b = True
        custom_op.parameter_map["use_off_line"].b = True     # True表示在昇腾AI处理器上执行训练
        custom_op.parameter_map["iterations_per_loop"].i = 100
        if dump_data:
            custom_op.parameter_map["enable_dump"].b = True
            custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_path)
            custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(dump_steps)
            custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
        session_config = npu_config_proto(config_proto=session_config)

    elif execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)

    return session_config


def import_ops(so_pkg_name: str = CUSTOM_OPS_SO) -> ModuleType:
    so_pkg_path = os.path.join("../../../../framework/tf_plugin/build/src/", so_pkg_name)
    if os.path.exists(so_pkg_path):
        logging.info(f"Using the DEFAULT PATH `{so_pkg_path}` to get ops lib.")
        return tf.load_op_library(so_pkg_path)
    else:
        raise ValueError(f"Please check if `{so_pkg_name}` exists.")


def format_size(size_bytes):
    """
    将字节数转换为带单位的字符串表示。

    参数:
        size_bytes (int): 数据量大小，以字节为单位。

    返回:
        str: 带有合适单位（B, KB, MB, GB）的字符串表示。
    """
    # 定义单位和换算比例
    units = ['B', 'KB', 'MB', 'GB']
    if size_bytes == 0:
        return "0B"  # 特殊处理零的情况

    unit_index = 0
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024.0
        unit_index += 1

    # 格式化输出，保留两位小数
    formatted_size = f"{size_bytes:.2f}".rstrip('0').rstrip('.')
    return f"{formatted_size} {units[unit_index]}"