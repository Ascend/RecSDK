# coding=utf-8
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

import enum
import os

import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator.npu.npu_config import NPURunConfig

from mx_rec.constants.constants import CacheModeEnum

SSD_DATA_PATH = ["ssd_data"]


class LearningRateScheduler:
    """
    LR Scheduler combining Polynomial Decay with Warmup at the beginning.
    TF-based cond operations necessary for performance in graph mode.
    """

    def __init__(self, base_lr_dense, base_lr_sparse, warmup_steps, decay_start_step, decay_steps):
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self.decay_start_step = tf.constant(decay_start_step, dtype=tf.int32)
        self.decay_steps = tf.constant(decay_steps)
        self.decay_end_step = decay_start_step + decay_steps  # 65041
        self.poly_power = 2.0
        self.base_lr_dense = base_lr_dense
        self.base_lr_sparse = base_lr_sparse

    def calc(self, global_step):
        # used for the warmup stage
        warmup_step = tf.cast(1 / self.warmup_steps, tf.float32)
        lr_factor_warmup = 1 - tf.cast(self.warmup_steps - global_step, tf.float32) * warmup_step
        lr_factor_warmup = tf.cast(lr_factor_warmup, tf.float32)
        # used for the constant stage
        lr_factor_constant = tf.cast(1.0, tf.float32)
        
        lr_sparse = self.base_lr_sparse * lr_factor_constant
        lr_dense = self.base_lr_dense * lr_factor_constant
        return lr_dense, lr_sparse


class Config:
    def __init__(self, ):
        self.rank_id = int(os.getenv("OMPI_COMM_WORLD_RANK")) if os.getenv("OMPI_COMM_WORLD_RANK") else None
        tmp = os.getenv("TRAIN_RANK_SIZE")
        if tmp is None:
            raise ValueError("please export TRAIN_RANK_SIZE")
        self.rank_size = int(tmp)

        self.data_path = os.getenv("DLRM_CRITEO_DATA_PATH")
        self.train_file_pattern = "train"
        self.test_file_pattern = "test"

        self.batch_size = 4096
        self.line_per_sample = 1
        self.train_epoch = 1
        self.test_epoch = 9
        self.perform_shuffle = False

        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.int64

        self.feat_cnt = 26
        self.__set_emb_table_size()

        self.field_num = 26
        self.send_count = 46000 // self.rank_size

        self.emb_dim = 8
        self.hashtable_threshold = 1

        self.USE_PIPELINE_TEST = False

        # 动态学习率
        GLOBAL_BATCH_SIZE = 8192 * 8
        LR_SCHEDULE_STEPS = [
            int(2750 * 55296 / GLOBAL_BATCH_SIZE),
            int(49315 * 55296 / GLOBAL_BATCH_SIZE),
            int(27772 * 55296 / GLOBAL_BATCH_SIZE),
        ]
        self.global_step = tf.Variable(0, trainable=False)
        _lr_scheduler = LearningRateScheduler(
            0.001,
            0.001,
            LR_SCHEDULE_STEPS[0],
            LR_SCHEDULE_STEPS[1],
            LR_SCHEDULE_STEPS[2],
        )
        self.learning_rate = _lr_scheduler.calc(self.global_step)

    def __set_emb_table_size(self):
        self.cache_mode = os.getenv("CACHE_MODE")
        if self.cache_mode is None:
            raise ValueError("please export CACHE_MODE environment variable, support:[HBM, DDR, SSD]")

        if self.cache_mode == CacheModeEnum.HBM.value:
            self.dev_vocab_size = 14_000_000 * self.rank_size
            self.host_vocab_size = 0
        elif self.cache_mode == CacheModeEnum.DDR.value:
            self.dev_vocab_size = 500_000 * self.rank_size
            self.host_vocab_size = 24_000_000 * self.rank_size
        elif self.cache_mode == CacheModeEnum.SSD.value:
            self.dev_vocab_size = 100_000 * self.rank_size
            self.host_vocab_size = 2_000_000 * self.rank_size
            self.ssd_vocab_size = 24_000_000 * self.rank_size
        else:
            raise ValueError(f"get CACHE_MODE:{self.cache_mode}, expect in [HBM, DDR, SSD]")

    def get_emb_table_cfg(self):
        if self.cache_mode == CacheModeEnum.HBM.value:
            return {"device_vocabulary_size": self.dev_vocab_size}
        elif self.cache_mode == CacheModeEnum.DDR.value:
            return {"device_vocabulary_size": self.dev_vocab_size,
                    "host_vocabulary_size": self.host_vocab_size}
        elif self.cache_mode == CacheModeEnum.SSD.value:
            return {"device_vocabulary_size": self.dev_vocab_size,
                    "host_vocabulary_size": self.host_vocab_size,
                    "ssd_vocabulary_size": self.ssd_vocab_size,
                    "ssd_data_path": SSD_DATA_PATH}
        else:
            raise RuntimeError(f"get CACHE_MODE:{self.cache_mode}, check Config.__set_emb_table_size implementation")


def sess_config(dump_data=False, dump_path="./dump_output", dump_steps="0|1|2"):
    session_config = tf.ConfigProto(allow_soft_placement=False,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["mix_compile_mode"].b = False
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["min_group_size"].b = 1
    # 可选配置level0:pairwise;level1:pairwise
    custom_op.parameter_map["HCCL_algorithm"].s = tf.compat.as_bytes("level0:fullmesh;level1:fullmesh")
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["iterations_per_loop"].i = 10
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["hcom_parallel"].b = False
    custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes("op_impl_mode.ini")
    custom_op.parameter_map["op_execute_timeout"].i = 2000
    custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(
        str(13 * 1024 * 1024 * 1024))  # total 31 need 13;
    custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(18 * 1024 * 1024 * 1024))  # need 25
    custom_op.parameter_map["stream_max_parallel_num"].s = tf.compat.as_bytes("DNN_VM_AICPU:3,AIcoreEngine:3")

    if dump_data:
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(dump_steps)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    return session_config


def get_npu_run_config():
    session_config = tf.ConfigProto(allow_soft_placement=False,
                                    log_device_placement=False)

    session_config.gpu_options.allow_growth = True
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    run_config = NPURunConfig(
        save_summary_steps=1000,
        save_checkpoints_steps=100,
        keep_checkpoint_max=5,
        session_config=session_config,
        log_step_count_steps=20,
        precision_mode='allow_mix_precision',
        enable_data_pre_proc=True,
        iterations_per_loop=1,
        jit_compile=False,
        op_compiler_cache_mode="enable",
        HCCL_algorithm="level0:fullmesh;level1:fullmesh"  # 可选配置：level0:pairwise;level1:pairwise
    )
    return run_config
