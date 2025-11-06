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
from enum import Enum
from typing import Union, Optional, List

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer as InitializerV1
from tensorflow.python.ops.init_ops_v2 import Initializer as InitializerV2


MXREC = "mxrec"
GRAPH_DEF = "graph_def"
USE_RANKTABLE = "use_ranktable"
USE_FUSION_OP = "use_fusion_op"
USE_LCCL_ALL2ALL_OP = "use_lccl_all2all_op"
FUSION_OP_TYPE = "fusion_op_type"
ALL2ALL_OP_TYPE = "all2all_op_type"

EMBEDDING_TABLE_COLLECTION = "EMBEDDING_TABLE_COLLECTION"
INIT_HASHTABLE_COLLECTION = "INIT_HASHTABLE_COLLECTION"
LOCAL_EMBEDDING_COLLECTION = "LOCAL_EMBEDDING_COLLECTION"


class EmbDistributionStrategy(Enum):
    # Model parallelism.
    MP = "MP"
    # Data parallelism.
    DP = "DP"


@dataclass
class EmbTableConfig:
    name: str
    dim: int
    dev_vocab_size: int = 0
    initializer: Union[
        InitializerV1, InitializerV2
    ] = tf.compat.v1.random_normal_initializer()
    key_dtype: tf.DType = tf.int64
    value_dtype: tf.DType = tf.float32
    dist_strategy: str = EmbDistributionStrategy.MP.value


@dataclass
class StaticEmbTableConfig(EmbTableConfig):
    min_used_times: Optional[int] = None
    max_cold_secs: Optional[int] = None


@dataclass
class MPLookupParams:
    local_ids: tf.Tensor
    local_ids_restore: tf.Tensor
    sorted_ids_indices: Optional[tf.Tensor] = None
    send_count_matrix: Optional[List[List[tf.Tensor]]] = None


@dataclass
class DPLookupParams:
    global_unique_ids: tf.Tensor
    max_uni_ids_count: Optional[tf.Tensor] = None
    local_uni_ids_count: Optional[tf.Tensor] = None
    ids_sc_all: Optional[tf.Tensor] = None
    rank_idx: Optional[tf.Tensor] = None
    local_unique_ids: Optional[tf.Tensor] = None
    local_unique_idx: Optional[tf.Tensor] = None
    global_unique_idx: Optional[tf.Tensor] = None


@dataclass
class CommNodeInfo:
    cm_chief_ip: str
    cm_chief_port: int
    cm_chief_device: int
    cm_worker_ip: str
    cm_worker_size: int


class CommParams(Enum):
    # The field name.
    CM_NODE_INFO = "cm-node-info"
    # The maximum/minimum length of an IPv4 address.
    MIN_IPV4_LEN = 7  # "0.0.0.0"
    MAX_IPV4_LEN = 15  # "255.255.255.255"
    # Used to configure the listening host ip of the master node.
    CM_CHIEF_IP = "CM_CHIEF_IP"
    # Used to configure the listening port of the master node.
    CM_CHIEF_PORT = "CM_CHIEF_PORT"
    MIN_CM_CHIEF_PORT = 0
    MAX_CM_CHIEF_PORT = 65520
    # Used to specify the logical id of the device for collecting cluster information of the server side
    # within the master node.
    CM_CHIEF_DEVICE = "CM_CHIEF_DEVICE"
    # Used to configure the network card ip used for information exchange between the current devices and
    # the master node.
    CM_WORKER_IP = "CM_WORKER_IP"
    # Used to configure the number of devices for this business communication domain.
    CM_WORKER_SIZE = "CM_WORKER_SIZE"
    MIN_CM_WORKER_SIZE = 0
    MAX_CM_WORKER_SIZE = 32768
    # The maximum value of the ranktable parameters.
    MAX_RANK_ID = 65535
    MAX_LOCAL_ID = 15


class LogParams(Enum):
    LOG_LEVEL = "log_level"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MPIParams(Enum):
    OMPI_COMM_WORLD_SIZE = "OMPI_COMM_WORLD_SIZE"
    OMPI_COMM_WORLD_LOCAL_SIZE = "OMPI_COMM_WORLD_LOCAL_SIZE"
    OMPI_COMM_WORLD_RANK = "OMPI_COMM_WORLD_RANK"


class ValidatorParams(Enum):
    MAX_INT32 = np.iinfo(np.int32).max
    MIN_INT32 = np.iinfo(np.int32).min
    MAX_UINT32 = np.iinfo(np.uint32).max
    MIN_UINT32 = np.iinfo(np.uint32).min
    MAX_INT64 = np.iinfo(np.int64).max
    MIN_INT64 = np.iinfo(np.int64).min
    MAX_UINT64 = np.iinfo(np.uint64).max
    MIN_UINT64 = np.iinfo(np.uint64).min
    MAX_FLOAT32 = np.finfo(np.float32).max
    MIN_FLOAT32 = np.finfo(np.float32).min
    STR_MAX_LENGTH = MAX_INT32
    STR_MIN_LENGTH = 0
    FILE_MIN_SIZE = 1
    FILE_MAX_SIZE = 1024 * 1024 * 1024 * 1024
    MAX_FILE_PATH_LENGTH = 1024


class NumCheckValueMethod(Enum):
    DEFAULT = "check_value"
    OPEN_INTERVAL = "check_value_for_open_interval"
    LEFT_OPEN_INTERVAL = "check_value_for_left_open_interval"
    RIGHT_OPEN_INTERVAL = "check_value_for_right_open_interval"
