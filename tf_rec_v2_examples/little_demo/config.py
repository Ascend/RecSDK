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

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from rec_sdk_common.communication.hccl.hccl_info import get_rank_size, get_rank_id


@dataclass
class Config:
    log_level: str = "DEBUG"
    ckpt_name: str = "model"

    train_and_evaluate = "train_and_evaluate"
    load_and_train = "load_and_train"
    predict = "predict"
    user_ids = "user_ids"
    item_ids = "item_ids"
    label_0 = "label_0"
    label_1 = "label_1"

    random_seed: int = 128
    rank_size: int = get_rank_size()
    rank_id: int = get_rank_id()
    batch_size: int = 4096
    prefetch_num: int = 100
    learning_rate: float = 0.05
    item_feat_cnt: int = 16
    user_feat_cnt: int = 8
    key_type: tf.DType = tf.int64
    label_type: tf.DType = tf.float32
    value_type: tf.DType = tf.float32
    item_range: int = 80000 * rank_size
    user_range: int = 200000 * rank_size
    item_vocab_size: int = item_range * rank_size
    user_vocab_size: int = user_range * rank_size
    user_hashtable_dim: int = 32
    item_hashtable_dim: int = 8


def sess_config(
        dump_data: bool = False, dump_path: str = "./dump_output", dump_steps: str = "0|1|2",
        is_deterministic: bool = False
):
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["mix_compile_mode"].b = False
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["min_group_size"].b = 1
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["iterations_per_loop"].i = 10
    if is_deterministic:
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("must_keep_origin_dtype")
        custom_op.parameter_map["deterministic"].i = 1
    else:
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["hcom_parallel"].b = False
    custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes("op_precision.ini")
    custom_op.parameter_map["op_execute_timeout"].i = 2000
    custom_op.parameter_map["stream_max_parallel_num"].s = tf.compat.as_bytes("DNN_VM_AICPU:3,AIcoreEngine:3")

    if dump_data:
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(dump_steps)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    return session_config
