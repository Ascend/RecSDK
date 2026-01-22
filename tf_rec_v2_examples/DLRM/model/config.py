#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

from typing import Any, Tuple, Dict, Union

import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from rec_sdk_common.communication.hccl.hccl_info import get_rank_size, get_rank_id

from lr_scheduler import LearningRateScheduler


class Config:
    def __init__(self, data_path: str, toml_config: Dict[str, Dict[str, Union[str, int, float]]]):
        self.data_path = data_path
        self.rank_id = get_rank_id()
        self.rank_size = get_rank_size()
        self.train_file_pattern = "train"
        self.test_file_pattern = "test"
        self.train_steps = toml_config["model"]["train_steps"]
        self.train_interval = toml_config["model"]["train_interval"]
        self.eval_steps = toml_config["model"]["eval_steps"]
        self.prefetch_num = toml_config["model"]["prefetch_num"]
        self.line_per_sample = toml_config["model"]["line_per_sample"]
        self.loss_scale = toml_config["model"]["loss_scale"]
        self.weight_decay = toml_config["model"]["weight_decay"]
        self.batch_size = toml_config["model"]["batch_size"]
        self.train_epoch = toml_config["model"]["train_epoch"]
        self.test_epoch = toml_config["model"]["test_epoch"]
        self.dev_vocab_size = toml_config["model"]["dev_vocab_size"] * self.rank_size
        self.emb_dim = toml_config["model"]["emb_dim"]
        self.key_type = tf.int64
        self.value_type = tf.float32
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = self._get_lr(toml_config)

    def _get_lr(self, toml_config: Dict[str, Dict[str, Union[str, int, float]]]) -> Tuple[tf.Tensor, tf.Tensor]:
        global_batch_size = self.batch_size * self.rank_size
        lr_scheduler = LearningRateScheduler(
            toml_config["model"]["base_lr_dense"],
            toml_config["model"]["base_lr_sparse"],
            int(toml_config["model"]["warmup_steps"] / global_batch_size),
            int(toml_config["model"]["decay_start_step"] / global_batch_size),
            int(toml_config["model"]["decay_steps"] / global_batch_size),
        )
        return lr_scheduler.calc(self.global_step)


def sess_config(dump_data: bool = False, dump_path: str = "./dump_output", dump_steps: str = "0|1|2") -> Any:
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["mix_compile_mode"].b = False
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["min_group_size"].b = True
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["iterations_per_loop"].i = 10
    # 开启"must_keep_origin_dtype"精度模式和确定性计算以保证模型精度，若使用其他模式可能导致精度问题
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("must_keep_origin_dtype")
    custom_op.parameter_map["deterministic"].i = 1
    custom_op.parameter_map["hcom_parallel"].b = False
    custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes("op_precision.ini")
    custom_op.parameter_map["op_execute_timeout"].i = 2000
    # Total 31G, variable memory need 13G.
    custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(13 * 1024 * 1024 * 1024))
    # Total 31G, graph memory need 18G.
    custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(18 * 1024 * 1024 * 1024))
    custom_op.parameter_map["stream_max_parallel_num"].s = tf.compat.as_bytes("DNN_VM_AICPU:3,AIcoreEngine:3")

    if dump_data:
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(dump_steps)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    return session_config
