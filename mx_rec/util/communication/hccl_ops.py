#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import os
from typing import Optional, Dict

from mx_rec.constants.constants import EnvOption
from mx_rec.util.communication.hccl_mgmt import parse_hccl_json, set_hccl_info_without_json
from mx_rec.util.global_env_conf import global_env


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


def get_rank_to_device_dict() -> Dict[int, int]:
    if global_env.rank_table_file:
        return parse_hccl_json()
    
    return set_hccl_info_without_json()


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


def get_rank_size() -> Optional[int]:
    """
    Get the rank size of the default collective communication group
    Note: this method should be used after mpi init
    :return: int, the rank size of the group
    """
    rank_size = os.getenv(EnvOption.OMPI_COMM_WORLD_SIZE.value)
    if rank_size is None:
        raise RuntimeError("Environment variable RANK_SIZE has not been exported, please init mpi/hccl first")
    try:
        rank_size_int = int(rank_size)
    except ValueError as e:
        raise ValueError(f"Environment variable RANK_SIZE should be number, but got the type: "
                         f"{type(rank_size)}.") from e
    if rank_size_int <= 0:
        raise ValueError("Rank size should be greater than 0.")
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
    if local_rank_size_int <= 0:
        raise ValueError("Local rank size should be greater than 0.")
    return local_rank_size_int
