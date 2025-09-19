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
import dataclasses
from dataclasses import dataclass

from rec_sdk_common.constants.constants import (
    EnvOptionCommon,
    LogLevel,
    ValidatorParams,
    DeviceType,
    CommonEnv,
    RankTableInfo)
from rec_sdk_common.validator.validator import (
    para_checker_decorator,
    OptionValidator,
    DirectoryValidator,
    Convert2intValidator)
from mx_rec.constants.constants import EnvOption, Flag, EMPTY_STR, \
    DEFAULT_HD_CHANNEL_SIZE, DEFAULT_KP_THREAD_NUM, DEFAULT_FAST_UNIQUE_THREAD_NUM, RecCPPLogLevel, \
    MIN_HD_CHANNEL_SIZE, MAX_HD_CHANNEL_SIZE, MIN_KP_THREAD_NUM, MAX_KP_THREAD_NUM, \
    MIN_FAST_UNIQUE_THREAD_NUM, MAX_FAST_UNIQUE_THREAD_NUM, DEFAULT_HOT_EMB_UPDATE_STEP, MIN_HOT_EMB_UPDATE_STEP, \
    MAX_HOT_EMB_UPDATE_STEP, MAX_CM_WORKER_SIZE, MIN_CM_WORKER_SIZE, DEFAULT_CM_WORKER_SIZE, DEFAULT_CM_CHIEF_DEVICE, \
    SsdCompactLevel


@dataclass
class RecEnv:
    mxrec_log_level: str
    rank_table_file: str
    cm_chief_device: str
    cm_worker_size: str
    tf_device: str
    acl_timeout: str
    hd_channel_size: str
    key_process_thread_num: str
    max_unique_thread_num: str
    fast_unique: str
    hot_emb_update_step: str
    glog_stderrthreahold: str
    use_combine_faae: str
    record_key_count: str
    ssd_save_compact_level: str
    use_shm_swap: str


def get_global_env_conf() -> RecEnv:
    """
    获取Rec SDK全局环境变量，并做校验
    :return:
    """
    rec_env = RecEnv(
        mxrec_log_level=os.getenv(EnvOptionCommon.RECSDK_LOG_LEVEL.value, LogLevel.INFO.value),
        rank_table_file=os.getenv(RankTableInfo.RANK_TABLE_FILE.value, EMPTY_STR),
        cm_chief_device=os.getenv(CommonEnv.CM_CHIEF_DEVICE.value, DEFAULT_CM_CHIEF_DEVICE),
        cm_worker_size=os.getenv(CommonEnv.CM_WORKER_SIZE.value, DEFAULT_CM_WORKER_SIZE),
        tf_device=os.getenv(EnvOptionCommon.DEVICE_TYPE.value, DeviceType.NONE.value),
        acl_timeout=os.getenv(EnvOption.ACL_TIMEOUT.value, "-1"),
        hd_channel_size=os.getenv(EnvOption.HD_CHANNEL_SIZE.value, DEFAULT_HD_CHANNEL_SIZE),
        key_process_thread_num=os.getenv(EnvOption.KEY_PROCESS_THREAD_NUM.value, DEFAULT_KP_THREAD_NUM),
        max_unique_thread_num=os.getenv(EnvOption.MAX_UNIQUE_THREAD_NUM.value, DEFAULT_FAST_UNIQUE_THREAD_NUM),
        fast_unique=os.getenv(EnvOption.FAST_UNIQUE.value, Flag.FALSE.value),
        hot_emb_update_step=os.getenv(EnvOption.HOT_EMB_UPDATE_STEP.value, DEFAULT_HOT_EMB_UPDATE_STEP),
        glog_stderrthreahold=os.getenv(EnvOption.GLOG_STDERRTHREAHOLD.value, RecCPPLogLevel.INFO.value),
        use_combine_faae=os.getenv(EnvOption.USE_COMBINE_FAAE.value, Flag.FALSE.value),
        record_key_count=os.getenv(EnvOption.RECORD_KEY_COUNT.value, Flag.FALSE.value),
        ssd_save_compact_level=os.getenv(EnvOption.SSD_SAVE_COMPACT_LEVEL.value, SsdCompactLevel.FULL_COMPACT.value),
        use_shm_swap=os.getenv(EnvOption.USE_SHM_SWAP.value, Flag.FALSE.value)
    )

    return rec_env


@para_checker_decorator(check_option_list=[
    ("mxrec_log_level", OptionValidator, {"options": [i.value for i in list(LogLevel)]}),
    ("rank_table_file", DirectoryValidator, {}, ["check_exists_if_not_empty"]),
    ("cm_worker_size", Convert2intValidator, {"min_value": MIN_CM_WORKER_SIZE, "max_value": MAX_CM_WORKER_SIZE},
     ["check_value"]),
    ("cm_chief_device", Convert2intValidator, {"min_value": MIN_CM_WORKER_SIZE, "max_value": (MAX_CM_WORKER_SIZE - 1)},
     ["check_value"]),
    ("tf_device", OptionValidator, {"options": [i.value for i in list(DeviceType)]}),
    ("acl_timeout", Convert2intValidator, {"min_value": -1, "max_value": ValidatorParams.MAX_INT32.value}, ["check_value"]),
    ("hd_channel_size", Convert2intValidator,
     {"min_value": MIN_HD_CHANNEL_SIZE, "max_value": MAX_HD_CHANNEL_SIZE}, ["check_value"]),
    ("key_process_thread_num", Convert2intValidator,
     {"min_value": MIN_KP_THREAD_NUM, "max_value": MAX_KP_THREAD_NUM}, ["check_value"]),
    ("max_unique_thread_num", Convert2intValidator,
     {"min_value": MIN_FAST_UNIQUE_THREAD_NUM, "max_value": MAX_FAST_UNIQUE_THREAD_NUM}, ["check_value"]),
    ("fast_unique", OptionValidator, {"options": [i.value for i in list(Flag)]}),
    ("hot_emb_update_step", Convert2intValidator,
     {"min_value": MIN_HOT_EMB_UPDATE_STEP, "max_value": MAX_HOT_EMB_UPDATE_STEP}, ["check_value"]),
    ("glog_stderrthreahold", OptionValidator, {"options": [i.value for i in list(RecCPPLogLevel)]}),
    ("use_combine_faae", OptionValidator, {"options": [i.value for i in list(Flag)]}),
    ("record_key_count", OptionValidator, {"options": [i.value for i in list(Flag)]}),
    ("ssd_save_compact_level", Convert2intValidator,
     {"min_value": SsdCompactLevel.NO_COMPACT.value, "max_value": SsdCompactLevel.FULL_COMPACT.value}, ["check_value"]),
    ("use_shm_swap", OptionValidator, {"options": [i.value for i in list(Flag)]}),
])
def check_env(**kwargs):
    pass


global_env = get_global_env_conf()

check_env(**dataclasses.asdict(global_env))
