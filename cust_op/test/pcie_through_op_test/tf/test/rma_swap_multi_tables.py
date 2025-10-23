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
import time
import argparse
import logging

import tensorflow as tf
from npu_bridge.estimator import npu_ops
from npu_bridge.npu_init import util
import numpy as np

import custom_pybind
from custom_pybind import ShmInfo, HostMgmt
from config import bind_cpu, sess_config, import_ops, format_size

rma_ops = import_ops("librecsdk_tf_npu_ops.so")

# 定义运行算子的Device ID
device_id = int(os.environ.get("RMA_DEVICE_ID"))
MAX_TABLE_NUM = 6


def shm_op_swap(emb_tables, emb_table_num, swap_in_index_arr, swap_out_index_arr, shm_swap_in_addr, shm_swap_out_addr):
    table_list = []
    for i in range(MAX_TABLE_NUM):
        if i < emb_table_num:
            table_list.append(emb_tables[i])
        else:
            table_list.append(emb_tables[0])
    shm_out_op = rma_ops.rma_swap_multi_tables(swap_in_index=swap_in_index_arr,
                                               swap_out_index=swap_out_index_arr,
                                               table_a=table_list[0],
                                               table_b=table_list[1],
                                               table_c=table_list[2],
                                               table_d=table_list[3],
                                               table_e=table_list[4],
                                               table_f=table_list[5],
                                               table_num=emb_table_num,
                                               shm_swap_in=shm_swap_in_addr,
                                               shm_swap_out=shm_swap_out_addr)
    return shm_out_op


def main(unused_argv):
    parser = argparse.ArgumentParser(description='命令行参数')
    parser.add_argument('--shape', type=int, help='数据量大小为(shape, shape), float32')
    parser.add_argument('--step', type=int, help='轮次')
    parser.add_argument('--table_num', type=int, help='emb表个数')
    args = parser.parse_args()
    shape_value = int(args.shape)
    step = int(args.step)
    table_num = int(args.table_num)
    logging.info(f"shape: ({shape_value}, {shape_value}), step: {step}, table_num: {table_num}")

    table_dim_split = 128
    table_size = 4000000
    table_dim = table_dim_split * table_num
    table_shape_params = (table_size, table_dim)
    table_shape_split_params = (table_size, table_dim_split)
    table_dtype_params = np.float32
    swap_num = (shape_value * shape_value) // table_dim

    affinity = set(range(0, 10))  # 请根据```npu-smi info -t topo```信息绑核
    bind_cpu(affinity)

    tables = []
    for _ in range(table_num):
        tables.append(np.random.uniform(-2, 2, size=table_shape_split_params).astype(table_dtype_params))
    tables = [tf.convert_to_tensor(table, tf.float32) for table in tables]
    with tf.device(f'/device:NPU:{device_id}'):
        tables = [tf.Variable(table) for table in tables]

    logging.info("====================")
    # 在昇腾AI处理器上运行单算子，得到实际运行结果
    index = np.random.choice(np.arange(0, table_size), size=swap_num, replace=False)

    shm_info = ShmInfo("rma_swap", 20 * 1024 * 1024 * 1024, 50, device_id, [swap_num, table_dim])
    host_mgmt = HostMgmt()
    host_mgmt.initialize([shm_info], step, 10086)
    swap_in_addr = custom_pybind.get_shm_mem("rma_swap_h2d", device_id, 50)
    shm_in_str = str(swap_in_addr)
    swap_out_addr = custom_pybind.get_shm_mem("rma_swap_d2h", device_id, 50)
    shm_out_str = str(swap_out_addr)
    rma_swap = shm_op_swap(emb_tables=tables,
                           emb_table_num=table_num,
                           swap_in_index_arr=index, swap_out_index_arr=index,
                           shm_swap_in_addr=shm_in_str, shm_swap_out_addr=shm_out_str)
    with tf.control_dependencies([rma_swap]):
        train_ops = tf.no_op()
    with tf.compat.v1.Session(config=sess_config('ai_core')) as session:
        session.run(tf.compat.v1.global_variables_initializer())
        train_op = util.set_iteration_per_loop(session, train_ops, 100)
        for i in range(step // 100):
            if i == 1:
                start_time = time.time()

            session.run(train_op)

        end_time = time.time()
        time_cost = (end_time - start_time) / (step - 100) * 1000 * 1000  # us
        logging.info(f"Data size: {format_size(swap_num * table_dim * 4)}, average time cost: {time_cost} us, "
                     f"bandwidth = {swap_num * table_dim * 4 / 1024 / 1024 / 1024 / time_cost * 1000 * 1000} GB/s")

    host_mgmt.destroy()
    logging.info('====================================')


if __name__ == "__main__":
    os.environ['WHICH_OP'] = 'GEOP'
    os.environ["DEVICE_ID"] = str(device_id)
    os.environ["ASCEND_DEVICE_ID"] = str(device_id)
    os.environ["RANK_ID"] = str(device_id)
    os.environ["JOB_ID"] = "10086"
    os.environ["SOC_VERSION"] = "Ascend910b"
    os.environ["GE_AICPU_FLAG"] = "1"
    os.environ["NEW_GE_FE_ID"] = "1"

    tf.compat.v1.app.run()
