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

import argparse
import os
import time
import logging

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
# must load before mxrec_pybind
ops_so = tf.load_op_library("/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/librecsdk_tf_npu_ops.so")

import mxrec_pybind

tf.compat.v1.disable_eager_execution()
logging.basicConfig(level=logging.DEBUG)


def set_ascend_env(rank, rank_size, local_rank_size, file=None, dev_id=-1, dev_index=1):
    rank = str(rank)
    rank_size = str(rank_size)
    local_rank_size = int(local_rank_size)

    os.environ["MOX_USE_NPU"] = "1"
    os.environ["FUSION_TENSOR_SIZE"] = "2000000000"
    os.environ["MOX_USE_TF_ESTIMATOR"] = "0"
    os.environ["MOX_USE_TDT"] = "1"
    os.environ["HEARTBEAT"] = "1"
    os.environ["CONTINUE_TRAIN"] = "true"

    os.environ["RANK_ID"] = rank
    local_rank_id = int(rank) % int(local_rank_size)
    if dev_id != -1:
        os.environ["DEVICE_ID"] = str(dev_id)
        os.environ["ASCEND_DEVICE_ID"] = str(dev_id)
    else:
        os.environ["DEVICE_ID"] = str(local_rank_id)
        os.environ["ASCEND_DEVICE_ID"] = str(local_rank_id)
    if dev_index != -1:
        os.environ["DEVICE_INDEX"] = str(dev_index)
    else:
        os.environ["DEVICE_INDEX"] = str(local_rank_id)

    os.environ["RANK_SIZE"] = rank_size
    if file:
        os.environ["RANK_TABLE_FILE"] = file

    os.environ["HCCL_CONNECT_TIMEOUT"] = "600"

    os.environ["JOB_ID"] = "10086"
    os.environ["SOC_VERSION"] = "Ascend910"
    os.environ["GE_AICPU_FLAG"] = "1"
    os.environ["NEW_GE_FE_ID"] = "1"
    os.environ["EXPERIMENTAL_DYNAMIC_PARTITION"] = "1"
    os.environ["ENABLE_FORCE_V2_CONTROL"] = "1"


class WideDeep:
    def __init__(self, input_data, matrix, arr, restore):
        self.lbl_hldr = input_data
        self.matrix = matrix
        self.arr = arr
        self.restore = restore
        self.forward()

    def forward(self):
        with tf.control_dependencies([self.lbl_hldr]):
            alluss_result_ = ops_so.lccl_all_uss(
                send_data=self.lbl_hldr,
                send_count_matrix=self.matrix,
                shape_vec=self.arr,
                peer_mem=peer_mem,
                restore=self.restore,
                rank=rank_id,
                rank_size=rank_size,
                dim=dim)
            self.alluss_result = tf.reshape(alluss_result_, [-1, dim])
        return self.alluss_result


def verify_result(real_result:np.array, golden:np.array):
    loss = 1e-4
    minimum = 10e-10
    
    result = np.abs(real_result - golden)
    deno = np.maximum(np.abs(real_result), np.abs(golden))
    result_atol = np.less_equal(result, loss)
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss)
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == 0) > real_result.size * loss and \
            np.sum(result_atol == 0) > real_result.size * loss:
            raise ValueError("precision error")
    logging.info("AllUss precision test pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='base')
    parser.add_argument("--local_rank_size")
    parser.add_argument("--hccl_json")
    args = parser.parse_args()
    local_rank_size = int(args.local_rank_size)

    comm = MPI.COMM_WORLD
    rank_id = comm.Get_rank()
    comm_server_rank_id = 0  # select rank 0 as server for lccl meta info exchange node
    rank_size = comm.Get_size()
    logging.info(f"rank {rank_id}/{rank_size}")
    local_rank_id = rank_id % rank_size
    set_ascend_env(rank_id, rank_size, local_rank_size, file=args.hccl_json)

    peer_mem_ = mxrec_pybind.get_peer_mem(rank_id, comm_server_rank_id, rank_size)
    logging.info(f"python peer_mem_ = {peer_mem_}")
    peer_mem = tf.constant(peer_mem_, dtype=tf.int64)


    # create session
    sess_config = tf.compat.v1.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["mix_compile_mode"].b = True
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('must_keep_origin_dtype')
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    sess_config.gpu_options.allow_growth = True
    custom_op.parameter_map["hcom_parallel"].b = False
    custom_op.parameter_map["op_execute_timeout"].i = 500

    dim = 128
    emb_len = 128 * 512
    output_len = emb_len // 2
    restore_np = np.random.randint(0, output_len, size=emb_len)
    shape_np = np.random.randint(0, output_len, size=emb_len // 2)
    for i in range(emb_len):
        restore_np[i] = i % output_len

    random_matrix = np.full((rank_size, rank_size), emb_len // rank_size * dim)

    send_count = 0
    for i in range(rank_size):
        send_count += int(random_matrix[local_rank_id][i])
    rev_count = 0
    for i in range(rank_size):
        rev_count += int(random_matrix[i][local_rank_id])

    random_send_data_np = np.random.rand(send_count, 1).astype(np.float32).reshape(-1, dim)
    for i in range(random_send_data_np.shape[0]):
        for j in range(len(random_send_data_np[i])):
            random_send_data_np[i][j] = rank_id * 100000 + i
    
    restore = tf.convert_to_tensor(restore_np, dtype=tf.int32)
    shape = tf.convert_to_tensor(shape_np, dtype=tf.int32)
    random_send_data = tf.convert_to_tensor(random_send_data_np, dtype=tf.float32)
    random_matrix = tf.convert_to_tensor(random_matrix, dtype=tf.int64)
    peer_mem = tf.convert_to_tensor(peer_mem_, dtype=tf.int64)

    # model run parameter
    stop_steps = 1
    model = WideDeep(random_send_data, random_matrix, shape, restore)

    with tf.compat.v1.Session(config=sess_config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        logging.info("============start alluss test=============")
        # start run loop
        current_steps = 0
        train_finished = False
        while not train_finished:
            try:
                current_steps += 1
                logging.info(f"current step = {current_steps}")
                run_dict = {
                    "alluss_result": model.alluss_result,
                }
                start_time = time.time()
                results = sess.run(fetches=run_dict)
                end_time = time.time()
                logging.info(f"current steps: {current_steps}, step time:{(end_time - start_time) * 1000}")
                if current_steps >= stop_steps:
                    comm.Barrier()
                    logging.info("alluss finished")
                    train_finished = True
            except tf.errors.OutOfRangeError:
                comm.Barrier()
                logging.info("alluss test failed with error:{e}")
                train_finished = True
        MPI.Finalize()

    # check precision
    all2all = random_send_data_np
    send_row_per_rank = emb_len // rank_size
    for i in range(all2all.shape[0]):
        for j in range(len(all2all[i])):
            all2all[i][j] = int(i / send_row_per_rank) * 100000 + (i % send_row_per_rank) + send_row_per_rank * rank_id
    expect_uss = np.zeros((output_len, dim), dtype=np.float32)
    for i, value in enumerate(restore_np):
        expect_uss[value] += all2all[i]
    actual = np.array(results.get("alluss_result"))
    verify_result(actual, expect_uss)

    logging.info("============end alluss test=============")