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

import os
import sys
import time
import argparse
import logging

import numpy as np
import tensorflow as tf
from mpi4py import MPI  # must before emb_cache after SparseOps
import psutil
from sklearn.metrics import roc_auc_score
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.hccl import hccl_ops
from npu_bridge.estimator import npu_ops

from sparse_ops.config import set_ascend_env, AscendEnv
from sparse_ops.common import InputConfig, input_fn_tfrecord, USE_PIPELINE_TEST
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.asc.helper import FeatureSpec
from mx_rec.util.initialize import get_rank_size, init, clear_channel, get_rank_id, set_if_load, \
    terminate_config_initializer
from mx_rec.constants.constants import MxRecMode
from mx_rec.core.embedding import create_table, sparse_lookup
from mx_rec.util.initialize import get_ascend_global_hashtable_collection
from mx_rec.optimizers.lazy_adam import CustomizedLazyAdam

logging.getLogger().setLevel(logging.INFO)


def create_hash_optimizer():
    return CustomizedLazyAdam()


def get_sparse_optimizer():
    sparse_optimizer = create_hash_optimizer()
    return sparse_optimizer


class WideDeep:
    def __init__(self, input_data, feature_spec_list, hashtable):
        self.lbl_hldr = input_data["global_labels"][0]
        self.input_data = input_data
        self.feature_spec_list = feature_spec_list
        self.hash_table_list = hashtable
        self.forward()

    def forward(self):
        for feature, hash_table in zip(self.feature_spec_list, self.hash_table_list):
            self.embedding = sparse_lookup(hash_table, feature, 1024 * 1024 // rank_size, dim=None, is_train=True,
                                           name="merged_embedding_lookup", modify_graph=False, batch=self.input_data)
            self.loss = tf.reduce_mean(self.embedding, axis=0)
        with tf.control_dependencies([self.loss]):
            self.op = tf.no_op()
        return self.op


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base')
    parser.add_argument('--local_rank_size')
    parser.add_argument('--hosts')
    parser.add_argument('--hccl_json')
    parser.add_argument('--my_dim')
    parser.add_argument('--chongfudu')
    parser.add_argument('--new_key')
    parser.add_argument('--slp')
    args = parser.parse_args()
    local_rank_size = int(args.local_rank_size)
    comm = MPI.COMM_WORLD
    rank_id = comm.Get_rank()
    rank_size = comm.Get_size()
    logging.info(f"rank {rank_id}/{rank_size}")
    local_rank_id = rank_id % local_rank_size
    ascend_env = AscendEnv(rank_id, rank_size, local_rank_size, host=args.hosts, file=args.hccl_json)
    set_ascend_env(ascend_env)

    # create session
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["mix_compile_mode"].b = True
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('must_keep_origin_dtype')
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    sess_config.gpu_options.allow_growth = True
    custom_op.parameter_map["hcom_parallel"].b = False
    custom_op.parameter_map["HCCL_algorithm"].s = tf.compat.as_bytes("level0:fullmesh;level1:pairwise")

    custom_op.parameter_map["iterations_per_loop"].i = 5
    custom_op.parameter_map["enable_dump"].b = True
    custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("./dump")
    custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("1|2")
    custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
    custom_op.parameter_map["op_wait_timeout"].i = 500
    custom_op.parameter_map["op_execute_timeout"].i = 500
    custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes("op_impl_mode.ini")
    custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(30000000000))
    custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(30000000000))

    global_start_time = time.time()
    tf.set_random_seed(10086)
    np.random.seed(10086)

    my_dim = int(args.my_dim)
    logging.info("my_dim=%d", my_dim)

    hot_zhanbi = args.chongfudu
    hot_zhanbi = float(hot_zhanbi) / 10

    config = {
        "data_path": os.path.join(".", "data1", "data" + str(hot_zhanbi) + "_" + str(float(args.new_key))) + os.sep,
        "train_file_pattern": "tf",
        "test_file_pattern": "test",
        "batch_size": 1024 * 8,
        "field_num": 128,
        "send_count": 1024 * 1024 // rank_size,  # 65536 * 10 > 39(field num) * 16000(bz)
        "id_emb_dim": my_dim,
        "ext_emb_vec_size": my_dim,
        "train_epoch": 1,
        "dev_vocab_size": 5000001
    }

    # model run parameter
    print_steps = 300
    evaluate_stride = 80000  # eval every 200 steps
    eval_steps = -1  # 8 ranks 34
    stop_steps = 5
    # Hybrid step1.1: init cache
    emb_name = "wide_deep_emb"

    dev_vocab_size = config["dev_vocab_size"]  # 23120
    host_vocab_size = 0

    init(True, rank_id=rank_id, rank_size=local_rank_size, train_interval=100, eval_steps=-1,
         prefetch_batch_number=1, use_dynamic=0, use_dynamic_expansion=0)

    tf.disable_eager_execution()
    ######################################
    feature_spec_list = [
        FeatureSpec("feat_ids", feat_count=128, table_name="merged_sparse_embeddings", batch_size=config["batch_size"])]
    with tf.device('/cpu:0'):
        input_config = InputConfig(feature_spec_list=feature_spec_list,
                            rank_id=rank_id,
                            local_rank_id=local_rank_id,
                            rank_size=rank_size,
                            data_path=config["data_path"],
                            file_pattern=config["train_file_pattern"],
                            total_batch_size=int(rank_size * config["batch_size"]),
                            perform_shuffle=(not USE_PIPELINE_TEST),
                            num_epochs=config["train_epoch"])
        train_dataset = input_fn_tfrecord(input_config)
        train_iterator = train_dataset.make_initializable_iterator()
        train_next_iter = train_iterator.get_next()

        train_input_data = {"global_labels": train_next_iter["label"],
                            "feat_ids": train_next_iter["feat_ids"],
                            }

    sparse_optimizer_list = get_sparse_optimizer()

    sparse_hashtable = create_table(key_dtype=tf.int64,
                                    dim=tf.TensorShape([my_dim]),
                                    name="merged_sparse_embeddings",
                                    emb_initializer=tf.variance_scaling_initializer(mode="fan_avg",
                                                                                    distribution='normal', seed=0),
                                    device_vocabulary_size=dev_vocab_size * local_rank_size,
                                    mode=MxRecMode.mapping("ASC"))

    sparse_variables = tf.compat.v1.get_collection(get_ascend_global_hashtable_collection())
    model = WideDeep(train_input_data, feature_spec_list, [sparse_hashtable])

    train_ops = []
    for loss, sparse_optimizer in zip([model.loss], [sparse_optimizer_list]):
        sparse_grads = tf.gradients(loss, sparse_variables)
        grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, sparse_variables)]
        train_ops.append(sparse_optimizer.apply_gradients(grads_and_vars))

    MODIFY_GRAPH_FLAG = False
    if MODIFY_GRAPH_FLAG:
        modify_graph_and_start_emb_cache(dump_graph=False)
    else:
        start_asc_pipeline()

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([train_iterator.initializer])
        # build model
        logging.info("start build wdl(single domain) model")
        logging.info("=========start============")
        # start run loop
        total_start_time = time.time()
        current_steps = 0
        train_finished = False
        time.sleep(int(args.slp))
        while not train_finished:
            try:
                current_steps += 1
                logging.info("current step =%d", current_steps)
                #
                run_dict = {
                    "loss": model.op,
                    "adam": train_ops,
                    "lbl_hldr": model.lbl_hldr,
                }
                if current_steps == 1:
                    total_start_time = time.time()
                start_time = time.time()
                logging.info("start sess run")
                results = sess.run(fetches=run_dict)
                logging.info("start sess run 1")
                end_time = time.time()
                logging.info(f"current_steps: {current_steps} ,step time:{(end_time - start_time) * 1000}")
                if current_steps <= 5:
                    total_start_time = time.time()
                if current_steps % print_steps == 0:
                    logging.info("----------" * 10)
                    try:
                        logging.info(
                            f"current_steps: {current_steps} ,deep_loss:{results['deep_loss']},"
                            f"e2etime per step:{(end_time - start_time) * 1000}"
                        )
                    except KeyError:
                        logging.error(f"current_steps: {current_steps}")
                    logging.info("----------" * 10)

                if current_steps >= stop_steps:
                    train_finished = True

            except tf.errors.OutOfRangeError:
                train_finished = True

        # train_finished
        logging.info(
            f"training {current_steps} steps, consume time: "
            f"{(time.time() - total_start_time) / (current_steps - 5) * 1000} "
        )

        terminate_config_initializer()
        MPI.Finalize()