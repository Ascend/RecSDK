#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
import numpy as np
from npu_bridge.estimator import npu_ops
from npu_bridge.npu_init import util

from custom_pybind import ShmInfo, HostMgmt
from config import bind_cpu, sess_config, import_ops, format_size

rma_ops = import_ops("librma_tf_ops.so")

# 定义运行算子的Device ID
device_id = int(os.environ.get("RMA_DEVICE_ID"))
MAX_TABLE_NUM = 6


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

    affinity = set(range(0, 10))
    bind_cpu(affinity)

    tables = []
    for _ in range(table_num):
        tables.append(np.random.uniform(-2, 2, size=table_shape_split_params).astype(table_dtype_params))
    tables = [tf.convert_to_tensor(table, tf.float32) for table in tables]
    with tf.device('/device:NPU:' + str(device_id)):
        tables = [tf.Variable(table) for table in tables]

    logging.info("====================")
    # 在昇腾AI处理器上运行单算子，得到实际运行结果
    index = np.random.choice(np.arange(0, table_size), size=swap_num, replace=False)

    shm_info = ShmInfo("tdt_swap", 20 * 1024 * 1024 * 1024, 50, device_id, [swap_num, table_dim])
    host_mgmt = HostMgmt()
    host_mgmt.initialize([shm_info], step, 10086, "TDT")

    def swap_op(swap_in_pos, swap_out_pos):
        with tf.compat.v1.variable_scope("h2d_emb"):
            h2d_emb = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.float32],
                output_shapes=[[swap_num, table_dim]],
                channel_name=f"tdt_swap_h2d",
            )[0]

        logging.info(f"h2d_emb shape: {h2d_emb}")

        swap_outs = [tf.gather(one_table, swap_out_pos) for one_table in tables]
        swap_out = tf.concat(swap_outs, axis=1)
        swap_out_op = npu_ops.outfeed_enqueue_op(
            channel_name=f"tdt_swap_d2h", inputs=[swap_out]
        )
        with tf.control_dependencies([swap_out_op]):
            nd_swap_pos = tf.expand_dims(swap_in_pos, 1)
            var_num = len(tables)
            h2d_emb_split = tf.split(h2d_emb, var_num, axis=1)

            swap_in_op = [
                tf.compat.v1.scatter_nd_update(tables[i], nd_swap_pos, h2d_emb_split[i])
                for i in range(var_num)]
        with tf.control_dependencies(swap_in_op):
            train_ops = tf.no_op()
        return train_ops
    test_op = swap_op(index, index)
    with tf.compat.v1.Session(config=sess_config('ai_core')) as session:
        session.run(tf.compat.v1.global_variables_initializer())
        test_op = util.set_iteration_per_loop(session, test_op, 100)
        for i in range(step // 100):
            if i == 1:
                start_time = time.time()

            session.run(test_op)

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