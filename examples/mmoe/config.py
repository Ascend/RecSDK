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

    def __init__(self, base_lr_dense, base_lr_sparse):
        self.base_lr_dense = base_lr_dense
        self.base_lr_sparse = base_lr_sparse

    def calc(self):
        # used for the constant stage
        lr_factor_constant = tf.cast(1.0, tf.float32)
        
        lr_sparse = self.base_lr_sparse * lr_factor_constant
        lr_dense = self.base_lr_dense * lr_factor_constant
        return lr_dense, lr_sparse


class Config:
    def __init__(self, ) -> None:
        self.rank_id = int(os.getenv("OMPI_COMM_WORLD_RANK")) if os.getenv("OMPI_COMM_WORLD_RANK") else None
        tmp = os.getenv("TRAIN_RANK_SIZE")
        if tmp is None:
            raise ValueError("please export TRAIN_RANK_SIZE")
        self.rank_size = int(tmp)

        self.data_path = os.getenv("DLRM_CRITEO_DATA_PATH")
        self.train_file_pattern = "train"
        self.test_file_pattern = "test"

        self.batch_size = 32
        self.line_per_sample = 1
        self.train_epoch = 100
        self.test_epoch = 100
        self.expert_num = 8
        self.gate_num = 2
        self.expert_size = 16
        self.tower_size = 8
        
        self.perform_shuffle = False

        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.int64

        self.feat_cnt = 26
        self.__set_emb_table_size()

        self.field_num = 26
        self.send_count = self.get_send_count(self.rank_size)

        self.emb_dim = self.expert_num * self.expert_size + self.gate_num * self.expert_num
        self.hashtable_threshold = 1

        self.USE_PIPELINE_TEST = False

        self.global_step = tf.Variable(0, trainable=False)
        _lr_scheduler = LearningRateScheduler(
            0.001,
            0.001
        )
        self.learning_rate = _lr_scheduler.calc()
        
        
    @staticmethod
    def get_send_count(rank_size):
        try:
            return 46000 // rank_size
        except ZeroDivisionError as exp:
            raise ZeroDivisionError('Rank size can not be zero.') from exp
        

    def get_emb_table_cfg(self) -> None:
        if self.cache_mode not in [CacheModeEnum.HBM.value, CacheModeEnum.DDR.value, CacheModeEnum.SSD.value]:
            raise RuntimeError(f"Invalid MODE:{self.cache_mode}, check Config.__set_emb_table_size implementation")
        
        result = {"device_vocabulary_size": self.dev_vocab_size}

        if self.cache_mode == CacheModeEnum.HBM.value:
            return result

        result["host_vocabulary_size"] = self.host_vocab_size

        if self.cache_mode == CacheModeEnum.DDR.value:
            return result

        # At this point, we know it's SSD mode
        result["ssd_vocabulary_size"] = self.ssd_vocab_size
        result["ssd_data_path"] = SSD_DATA_PATH

        return result
    
    
    def __set_emb_table_size(self) -> None:
        self.cache_mode = os.getenv("CACHE_MODE")
        if self.cache_mode is None:
            raise ValueError("please export CACHE_MODE environment variable, support:[HBM, DDR, SSD]")

        if self.cache_mode == CacheModeEnum.HBM.value:
            self.dev_vocab_size = 1000 * self.rank_size
            self.host_vocab_size = 0
        elif self.cache_mode == CacheModeEnum.DDR.value:
            self.dev_vocab_size = 1000 * self.rank_size
            self.host_vocab_size = 1000 * self.rank_size
        elif self.cache_mode == CacheModeEnum.SSD.value:
            self.dev_vocab_size = 1000 * self.rank_size
            self.host_vocab_size = 1000 * self.rank_size
            self.ssd_vocab_size = 1000 * self.rank_size
        else:
            raise ValueError(f"get CACHE_MODE:{self.cache_mode}, expect in [HBM, DDR, SSD]")


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
