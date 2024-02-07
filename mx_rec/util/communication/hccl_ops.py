#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import os
from typing import Optional

from mx_rec.constants.constants import EnvOption
from mx_rec.util.communication.hccl_mgmt import parse_hccl_json, set_hccl_info_without_json
from mx_rec.util.global_env_conf import global_env, get_global_env_conf
from mx_rec.util.log import logger


def get_rank_id() -> Optional[int]:
    """
    Get the rank id for the current device in the collective communication group
    Note: this method should be used after mpi init
    :return: int or None, the rank id of the calling process
    """
    rank_id = os.getenv(EnvOption.OMPI_COMM_WORLD_RANK.value)
    if rank_id is None:
        raise RuntimeError("Environment variable RANK_ID has not been exported, please init mpi/hccl first")
    try:
        rank_id_int = int(rank_id)
    except ValueError as e:
        raise ValueError(f"Environment variable RANK_ID should be number, but got the type: {type(rank_id)}") from e

    return rank_id_int


def get_device_id() -> Optional[int]:
    """
     Get the device id of the calling process
     Note: this method should be used after mpi init
    :return: int or None, the device id of the calling process
    """
    if global_env.rank_table_file:
        rank_to_device_dict = parse_hccl_json()
    else:
        rank_to_device_dict = set_hccl_info_without_json()
    device_id = rank_to_device_dict.get(get_rank_id())
    if device_id is None:
        raise RuntimeError("Environment variable DEVICE_ID has not been exported, please init mpi/hccl first")
    try:
        device_id_int = int(device_id)
    except ValueError as e:
        raise ValueError(f"Environment variable DEVICE_ID should be number, but got the type: "
                         f"{type(device_id)}.") from e
    return device_id_int


def get_rank_size() -> Optional[int]:
    """
    Get the rank size of the default collective communication group
    Note: this method should be used after mpi init
    :return: int, the rank size of the group
    """
    rank_size = os.getenv(EnvOption.OMPI_COMM_WORLD_LOCAL_SIZE.value)
    if rank_size is None:
        raise RuntimeError("Environment variable RANK_SIZE has not been exported, please init mpi/hccl first")
    try:
        rank_size_int = int(rank_size)
    except ValueError as e:
        raise ValueError(f"Environment variable RANK_SIZE should be number, but got the type: "
                         f"{type(rank_size)}.") from e

    return rank_size_int


def get_local_rank_size() -> Optional[int]:
    """
    Get the local rank size of the default collective communication group
    Note: this method should be used after mpi init
    :return: int, the local rank size of the group
    """
    local_rank_size = os.getenv(EnvOption.OMPI_COMM_WORLD_LOCAL_SIZE.value)
    if local_rank_size is None:
        raise RuntimeError("Environment variable LOCAL_RANK_SIZE has not been exported, please init mpi/hccl first")
    try:
        local_rank_size_int = int(local_rank_size)
    except ValueError as e:
        raise ValueError(f"Environment variable LOCAL_RANK_SIZE should be number, but got the type:"
                         f" {type(local_rank_size)}.") from e

    return local_rank_size_int
