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

from enum import Enum
import numpy as np


class RankTableParams(Enum):
    MAX_RANK_SIZE = 4095
    MIN_RANK_SIZE = 0


class FileParams(Enum):
    MIN_SIZE = 1
    MAX_SIZE = 1024 * 1024 * 1024 * 1024
    MAX_FILE_SIZE = 500 * 1024 * 1024 * 1024
    MAX_CONFIG_SIZE = 10 * 1024 * 1024


class CommParams(Enum):
    MAX_RANK_ID = 65535
    MAX_LOCAL_ID = 15
    MAX_DEVICE_ID = 15
    MAX_LOGIC_ID = 15


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
    MAX_UINT8 = np.iinfo(np.uint8).max
    MIN_VALUE = 1
    MIN_VALUE_NEGATIVE = -1
    STR_MIN_LENGTH = 0
    FILE_MIN_SIZE = 1
    FILE_MAX_SIZE = 1024 * 1024 * 1024 * 1024
    MAX_FILE_PATH_LENGTH = 1024


class MPIParams(Enum):
    OMPI_COMM_WORLD_SIZE = "OMPI_COMM_WORLD_SIZE"
    OMPI_COMM_WORLD_LOCAL_SIZE = "OMPI_COMM_WORLD_LOCAL_SIZE"
    OMPI_COMM_WORLD_RANK = "OMPI_COMM_WORLD_RANK"


class RankTableInfo(Enum):
    RANK_TABLE_FILE = "RANK_TABLE_FILE"
    RANK_LIST = "rank_list"
    SERVER_LIST = "server_list"
    RANK_ID = "rank_id"
    LOCAL_ID = "local_id"
    DEVICE_ID = "device_id"
    DEVICE = "device"


class NumCheckValueMethod(Enum):
    DEFAULT = "check_value"
    OPEN_INTERVAL = "check_value_for_open_interval"
    LEFT_OPEN_INTERVAL = "check_value_for_left_open_interval"
    RIGHT_OPEN_INTERVAL = "check_value_for_right_open_interval"


class ChipName(Enum):
    ASCEND_910B = "ASCEND_910B"
    NONE = "NONE"


class CommonEnv(Enum):
    CM_WORKER_SIZE = "CM_WORKER_SIZE"
    CM_CHIEF_DEVICE = "CM_CHIEF_DEVICE"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnvOptionCommon(Enum):
    RECSDK_LOG_LEVEL = "MXREC_LOG_LEVEL"
    DEVICE_TYPE = "TF_DEVICE"


class DeviceType(Enum):
    CPU = "CPU"
    NPU = "NPU"
    GPU = "GPU"
    NONE = "NONE"
