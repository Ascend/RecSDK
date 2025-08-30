#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
from typing import Optional, Dict

from rec_sdk_common.constants.constants import MPIParams, ValidatorParams
from rec_sdk_common.communication.hccl.hccl_mgmt import _get_rank_info_with_ranktable, _get_rank_info_without_ranktable
from rec_sdk_common.constants.constants import RankTableInfo
from rec_sdk_common.validator.validator import StringValidator
from rec_sdk_common.validator.safe_checker import str_safe_check


def _comm_env_value_str2int(value: str, greater_or_equal: int = 0) -> int:
    str_safe_check("communication environment value", value)

    if not StringValidator("communication environment value", value).can_be_transformed2int().is_valid():
        raise ValueError(f"the value should be number, but got the value: {value}")

    int_value = int(value)
    if int_value < greater_or_equal:
        raise ValueError(f"the value must be greater than or equal to {greater_or_equal}, but got {int_value}")
    return int_value


def get_rank_id() -> Optional[int]:
    """
    Get the rank id for the current device in the collective communication group
    Note: this method should be used after mpi init
    :return: int or None, the rank id of the calling process
    """
    rank_id = os.getenv(MPIParams.OMPI_COMM_WORLD_RANK.value)
    if rank_id is None:
        raise RuntimeError("Environment variable RANK_ID has not been exported, please init mpi/hccl first")

    return _comm_env_value_str2int(rank_id, 0)


def get_rank_size() -> Optional[int]:
    """
    Get the rank size of the default collective communication group
    Note: this method should be used after mpi init
    :return: int, the rank size of the group
    """
    rank_size = os.getenv(MPIParams.OMPI_COMM_WORLD_SIZE.value)
    if rank_size is None:
        raise RuntimeError("Environment variable RANK_SIZE has not been exported, please init mpi/hccl first")
    return _comm_env_value_str2int(rank_size, 1)


def get_local_rank_size() -> Optional[int]:
    """
    Get the local rank size of the default collective communication group
    Note: this method should be used after mpi init
    :return: int, the local rank size of the group
    """
    local_rank_size = os.getenv(MPIParams.OMPI_COMM_WORLD_LOCAL_SIZE.value)
    if local_rank_size is None:
        raise RuntimeError("Environment variable LOCAL_RANK_SIZE has not been exported, please init mpi/hccl first")
    return _comm_env_value_str2int(local_rank_size, 1)


def get_rank_to_device_dict() -> Dict[int, int]:
    rank_table_path = os.getenv(RankTableInfo.RANK_TABLE_FILE.value, "")
    if rank_table_path != "":
        return _get_rank_info_with_ranktable()
    
    return _get_rank_info_without_ranktable()


def get_device_id() -> Optional[int]:
    """
     Get the device logic id of the calling process
     Note: this method should be used after mpi init
    :return: int or None, the device logic id of the calling process
    """
    rank_to_device_dict = get_rank_to_device_dict()
    # 对local_rank_size取模适配多机场景
    device_id = rank_to_device_dict.get(get_rank_id() % get_local_rank_size())
    if device_id is None:
        raise RuntimeError("Environment variable DEVICE_ID has not been exported, please init mpi/hccl first")
    try:
        device_id_int = int(device_id)
    except ValueError as e:
        raise ValueError(f"Environment variable DEVICE_ID should be number, but got the type: "
                         f"{type(device_id)}.") from e
    return device_id_int



def get_min_device_id() -> Optional[int]:
    """
    Get the min device logic id of the calling process.
    Note: this method should be used after mpi init.
    :return: the min device id of the calling process.
    """
    rank_to_device_dict = get_rank_to_device_dict()
    return min(rank_to_device_dict)