#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from enum import Enum
import numpy as np

ASCEND_GLOBAL_HASHTABLE_COLLECTION = "ASCEND_GLOBAL_HASHTABLE_COLLECTION"
ASCEND_CUTTING_POINT_INITIALIZER = "ASCEND_CUTTING_POINT_INITIALIZER"
ASCEND_SPARSE_LOOKUP_ENTRANCE = "ASCEND_SPARSE_LOOKUP_ENTRANCE"
ASCEND_SPARSE_LOOKUP_ID_OFFSET = "ASCEND_SPARSE_LOOKUP_ID_OFFSET"
ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS = "ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS"
ASCEND_TIMESTAMP = "ASCEND_TIMESTAMP"
ASCEND_SPARSE_LOOKUP_LOCAL_EMB = "ASCEND_SPARSE_LOOKUP_LOCAL_EMB"
EMPTY_STR = ""

# 获取ConfigInitializer对象实例失败提示信息
GET_CONFIG_INSTANCE_ERR_MSG = "Please init the environment for mx_rec at first."

# 自动改图模式下从计算图中寻找dataset的锚点名称
ANCHOR_DATASET_NAME = "PrefetchDataset"

# the name of the embedding table merged by third party
ASCEND_TABLE_NAME_MUST_CONTAIN = None

# while循环最大深度
MAX_WHILE_SIZE = 800

# acl通道数据深度
DEFAULT_HD_CHANNEL_SIZE = 40
MAX_HD_CHANNEL_SIZE = 8192
MIN_HD_CHANNEL_SIZE = 2

# key process线程数
DEFAULT_KP_THREAD_NUM = 6
MIN_KP_THREAD_NUM = 1
MAX_KP_THREAD_NUM = 10

# Fast unique去重最大线程数
DEFAULT_FAST_UNIQUE_THREAD_NUM = 8
MIN_FAST_UNIQUE_THREAD_NUM = 1
MAX_FAST_UNIQUE_THREAD_NUM = 8

# Hot Embedding更新步数
DEFAULT_HOT_EMB_UPDATE_STEP = 1000
MIN_HOT_EMB_UPDATE_STEP = 1
MAX_HOT_EMB_UPDATE_STEP = 1000

MULTI_LOOKUP_TIMES = 128
DEFAULT_EVICT_TIME_INTERVAL = 60 * 60 * 24
TRAIN_CHANNEL_ID = 0
EVAL_CHANNEL_ID = 1
HASHTABLE_COLLECTION_NAME_LENGTH = 30
MAX_VOCABULARY_SIZE = 10**10

# RANK INFO
VALID_DEVICE_ID_LIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
MIN_SIZE = 1
MAX_CONFIG_SIZE = 10 * 1024 * 1024
MAX_SIZE = 1024 * 1024 * 1024 * 1024
MAX_FILE_SIZE = 500 * 1024 * 1024 * 1024
MAX_RANK_SIZE = 4095
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


class EnvOption(Enum):
    MXREC_LOG_LEVEL = "MXREC_LOG_LEVEL"
    SAVE_EASY = "SAVE_EASY"
    RANK_TABLE_FILE = "RANK_TABLE_FILE"
    ASCEND_VISIBLE_DEVICES = "ASCEND_VISIBLE_DEVICES"
    CM_CHIEF_DEVICE = "CM_CHIEF_DEVICE"
    CM_WORKER_SIZE = "CM_WORKER_SIZE"
    TF_DEVICE = "TF_DEVICE"
    APPLY_GRADIENTS_STRATEGY = "APPLY_GRADIENTS_STRATEGY"
    ACL_TIMEOUT = "AclTimeout"
    HD_CHANNEL_SIZE = "HD_CHANNEL_SIZE"
    KEY_PROCESS_THREAD_NUM = "KEY_PROCESS_THREAD_NUM"
    MAX_UNIQUE_THREAD_NUM = "MAX_UNIQUE_THREAD_NUM"
    FAST_UNIQUE = "FAST_UNIQUE"
    UPDATEEMB_V2 = "UpdateEmb_V2"
    HOT_EMB_UPDATE_STEP = "HOT_EMB_UPDATE_STEP"
    GLOG_STDERRTHREAHOLD = "GLOG_stderrthreshold"
    USE_COMBINE_FAAE = "USE_COMBINE_FAAE"
    STAT_ON = "STAT_ON"


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
    IS_GRAD = "is_grad"


class OptimizerType(Enum):
    LAZY_ADAM = "LazyAdam"
    SGD = "SGD"

    @staticmethod
    def get_optimizer_state_meta(mode):
        if mode in OPTIMIZER_STATE_META:
            return OPTIMIZER_STATE_META.get(mode)

        raise ValueError(f"Invalid mode value, please choose one from {list(map(lambda c: c.value, OptimizerType))}")


OPTIMIZER_STATE_META = {OptimizerType.LAZY_ADAM: ["momentum", "velocity"], OptimizerType.SGD: []}


class All2allGradientsOp(BaseEnum):
    SUM_GRADIENTS = "sum_gradients"
    SUM_GRADIENTS_AND_DIV_BY_RANKSIZE = "sum_gradients_and_div_by_ranksize"


class ApplyGradientsStrategy(BaseEnum):
    DIRECT_APPLY = "direct_apply"
    SUM_SAME_ID_GRADIENTS_AND_APPLY = "sum_same_id_gradients_and_apply"


class RecPyLogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    ERROR = "ERROR"


class RecCPPLogLevel(Enum):
    TRACE = "-2"
    DEBUG = "-1"
    INFO = "0"
    WARN = "1"
    ERROR = "2"


class TFDevice(Enum):
    CPU = "CPU"
    NPU = "NPU"
    GPU = "GPU"


class Flag(Enum):
    TRUE = "1"
    FALSE = "0"


