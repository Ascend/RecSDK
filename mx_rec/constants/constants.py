#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from enum import Enum
import numpy as np

ASCEND_GLOBAL_HASHTABLE_COLLECTION = "ASCEND_GLOBAL_HASHTABLE_COLLECTION"
ASCEND_CUTTING_POINT_INITIALIZER = "ASCEND_CUTTING_POINT_INITIALIZER"
ASCEND_CUTTING_POINT = "ASCEND_CUTTING_POINT"
ASCEND_SPARSE_LOOKUP_ENTRANCE = "ASCEND_SPARSE_LOOKUP_ENTRANCE"
ASCEND_SPARSE_LOOKUP_ID_OFFSET = "ASCEND_SPARSE_LOOKUP_ID_OFFSET"
ASCEND_SPARSE_LOOKUP_RESTORE_VECTOR = "ASCEND_SPARSE_LOOKUP_RESTORE_VECTOR"
ASCEND_SPARSE_LOOKUP_LOOKUP_RESULT = "ASCEND_SPARSE_LOOKUP_LOOKUP_RESULT"
# dynamic shape identity
ASCEND_SPARSE_LOOKUP_ALL2ALL_MATRIX = "ASCEND_SPARSE_LOOKUP_ALL2ALL_MATRIX"
# hot embed function identity
ASCEND_SPARSE_LOOKUP_HOT_POS = "ASCEND_SPARSE_LOOKUP_HOT_POS"
ASCEND_TIMESTAMP = "ASCEND_TIMESTAMP"
CUSTOMIZED_OPS_LIB_PATH = "CUSTOMIZED_OPS_LIB_PATH"
HOST_PIPELINE_OPS_LIB_PATH = "HOST_PIPELINE_OPS_LIB_PATH"
ASCEND_SPARSE_LOOKUP_LOCAL_EMB = "ASCEND_SPARSE_LOOKUP_LOCAL_EMB"

# the name of the embedding table merged by third party
ASCEND_TABLE_NAME_MUST_CONTAIN = None

# this number is a temp plan to solve a problem
# to avoid op "scatter_nd_update" may get a None tensor for input
AVOID_TENSOR_POS = 439999
LOCAL_RANK_SIZE = "LOCAL_RANK_SIZE"  # 训练时，当前服务器使用的NPU卡数
MAX_DEVICE_NUM_LOCAL_MACHINE = 16  # 单台服务器最大的卡数
DEFAULT_DEVICE_NUM_LOCAL_MACHINE = 8  # 单台服务器默认的卡数

MULTI_LOOKUP_TIMES = 128
DEFAULT_EVICT_TIME_INTERVAL = 60 * 60 * 24
TRAIN_CHANNEL_ID = 0
EVAL_CHANNEL_ID = 1
HASHTABLE_COLLECTION_NAME_LENGTH = 30

# RANK INFO
VALID_DEVICE_ID_LIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
MIN_SIZE = 1
MAX_CONFIG_SIZE = 10 * 1024 * 1024
MAX_SIZE = 1024 * 1024 * 1024 * 1024
MAX_DEVICE_NUM = 16
MAX_RANK_SIZE = 4095
MIN_DEVICE_NUM = 1
MIN_RANK_SIZE = 1

LOG_MAX_SIZE = 1024 * 1024

MAX_INT32 = np.iinfo(np.int32).max

DUMP_MIDIFY_GRAPH_FILE_MODE = 0o550
MAX_DEVICE_ID = 15


class BaseEnum(Enum):
    @classmethod
    def mapping(cls, key):
        for mode in cls:
            if isinstance(key, BaseEnum):
                key_value = key.value
            else:
                key_value = key
            if key_value == mode.value:
                return mode

        raise KeyError(f"Cannot find a corresponding mode in current Enum "
                       f"class {cls}, given parameter '{key}[{key.__class__}]' is illegal, "
                       f"please choose a valid one from "
                       f"'{list(map(lambda c: c.value, cls))}'.")


class DataName(Enum):
    KEY = "key"
    EMBEDDING = "embedding"
    FEATURE_MAPPING = "feature_mapping"
    OFFSET = "offset"
    THRESHOLD = "threshold"
    VALID_LEN = "valid_len"
    VALID_BUCKET_NUM = "valid_bucket_num"


class DataAttr(Enum):
    SHAPE = "shape"
    DATATYPE = "data_type"


class ASCAnchorAttr(Enum):
    TABLE_INSTANCE = "table_instance"
    IS_TRAINING = "is_training"
    RESTORE_VECTOR = "restore_vector"
    ID_OFFSETS = "id_offsets"
    FEATURE_SPEC = "feature_spec"
    ALL2ALL_MATRIX = "all2all_matrix"
    HOT_POS = "hot_pos"
    LOOKUP_RESULT = "lookup_result"
    MOCK_LOOKUP_RESULT = "mock_lookup_result"
    RESTORE_VECTOR_SECOND = "restore_vector_second"
    UNIQUE_KEYS = "unique_keys"
    GRADIENTS_STRATEGY = "gradients_strategy"


class MxRecMode(BaseEnum):
    ASC = "ASC"  # Ascend Sparse with Cpu-hashtable


class OptimizerType(Enum):
    LAZY_ADAM = "LazyAdam"
    SGD = "SGD"

    @staticmethod
    def get_optimizer_state_meta(mode):
        if mode in OPTIMIZER_STATE_META:
            return OPTIMIZER_STATE_META.get(mode)

        raise ValueError(f"Invalid mode value, please choose one from {list(map(lambda c: c.value, OptimizerType))}")


OPTIMIZER_STATE_META = {OptimizerType.LAZY_ADAM: ["momentum", "velocity"],
                        OptimizerType.SGD: []}


class All2allGradientsOp(BaseEnum):
    SUM_GRADIENTS = "sum_gradients"
    SUM_GRADIENTS_AND_DIV_BY_RANKSIZE = "sum_gradients_and_div_by_ranksize"


class ApplyGradientsStrategy(BaseEnum):
    DIRECT_APPLY = "direct_apply"
    SUM_SAME_ID_GRADIENTS_AND_APPLY = "sum_same_id_gradients_and_apply"
