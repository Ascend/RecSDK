#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import json
import os

from mxrec_pybind import get_logic_id
from mx_rec.constants.constants import MAX_CONFIG_SIZE, MAX_DEVICE_ID, MAX_RANK_SIZE, MIN_RANK_SIZE, MIN_SIZE, \
    VALID_DEVICE_ID_LIST
from mx_rec.util.global_env_conf import global_env
from mx_rec.validator.validator import Convert2intValidator, FileValidator, para_checker_decorator, StringValidator


def parse_hccl_json():
    rank_table_path = global_env.rank_table_file
    with open(rank_table_path, "r", encoding="utf-8"):
        # check whether json file is valid
        file_validator = FileValidator("RANK_TABLE_FILE", rank_table_path)
        # 1.check whether rank_table_path is soft link
        file_validator.check_not_soft_link()
        # 2.check json file size
        file_validator.check_file_size(MAX_CONFIG_SIZE, MIN_SIZE)
        file_validator.check()

    rank_table_path = os.path.realpath(global_env.rank_table_file)
    with open(rank_table_path, "r", encoding="utf-8") as file:
        try:
            table_hccl = json.load(file)
        except FileNotFoundError as e:
            raise ValueError("rank table file not found") from e
        except json.JSONDecodeError as e:
            raise ValueError("rank table file is unable to parse as json") from e
        if "server_list" not in table_hccl:
            raise AttributeError(f"Lack of attribute server_list.")
        if not table_hccl.get("server_list"):
            raise ValueError(f"Server_list is empty.")
        if "device" not in table_hccl.get("server_list")[0]:
            raise AttributeError(f"Lack of attribute device.")

    rank_to_device_dict = dict()
    local_rank_size = -1
    for server_list in table_hccl.get("server_list"):
        devices = server_list.get("device")
        if devices is None:
            raise ValueError("device is empty")

        local_rank_size = len(devices)
        for device in devices:
            if "rank_id" not in device or not device.get("rank_id").isdigit():
                raise ValueError(f"hccl_json rank_id wrong.")
            rank_id = int(device.get("rank_id"))
            if "device_id" not in device or not device.get("device_id").isdigit():
                raise ValueError(f"hccl_json device_id wrong.")

            res = get_logic_id(int(device.get("device_id")))
            if res < 0:
                raise RuntimeError(
                    f"get logic id from physic id fail, error code is {res}, please check if dsmi api is functional.")
            if res > MAX_DEVICE_ID:
                raise ValueError(f"get logic id from physic id fail, the device id is invalid.")
            rank_to_device_dict[rank_id] = res

    return rank_to_device_dict, local_rank_size


@para_checker_decorator(check_option_list=[
    ("visible_devices", StringValidator, {"msg": "please config ASCEND_VISIBLE_DEVICES in docker container start"}),
    ("rank_size", StringValidator, {"msg": "please config CM_WORKER_SIZE in docker container start"}),
    ("chief_device", StringValidator, {"msg": "please config CM_CHIEF_DEVICE in docker container start"}),
    ("rank_size", Convert2intValidator, {"min_value": MIN_RANK_SIZE, "max_value": MAX_RANK_SIZE,
                                         "constrained_options": [1, 2, 4, 8, 16]}, ["check_value"]),
    ("chief_device", Convert2intValidator, {"min_value": 0, "max_value": 15}, ["check_value"]),
])
def set_hccl_info_without_json(visible_devices: str, rank_size: str, chief_device: str):
    """
    Used for no rank table file configured training situation.
    Now, only less than or equal 8p training job is supported.
    :param visible_devices: 昇腾处理器可见的设备，来指定程序只使用其中的部分设备。
    :param rank_size: 参与集群训练的device数量。
    :param chief_device: 主节点device id。
    :return:
    """
    device_list = get_device_list(visible_devices)
    chief_device = int(chief_device)
    rank_size = int(rank_size)

    sorted_device_list = sorted(device_list)
    local_rank_size = len(sorted_device_list)

    if rank_size > local_rank_size:
        raise ValueError(f"Rank size {rank_size} is larger than local available devices: {local_rank_size}.")

    if chief_device not in sorted_device_list:
        raise ValueError(f"The environment variable CM_CHIEF_DEVICE {chief_device} is not in the local device list. ")


    rank_to_device_dict = {}
    chief_index = sorted_device_list.index(chief_device)
    sorted_device_list = sorted_device_list[chief_index:] + sorted_device_list[0: chief_index]
    sorted_device_list = sorted_device_list[:rank_size]

    for device_idx in sorted_device_list:
        res = get_logic_id(int(device_idx))
        if res < 0:
            raise RuntimeError(
                f"get logic id from physic id fail, error code is {res}, please check if dsmi api is functional.")

        if res > MAX_DEVICE_ID:
            raise ValueError(f"get logic id from physic id fail.")
        index = sorted_device_list.index(device_idx)
        rank_to_device_dict[index] = res
    return rank_to_device_dict, local_rank_size


def get_device_list(ascend_visible_devices):
    device_list = []
    try:
        if "-" in ascend_visible_devices:
            split_devices = ascend_visible_devices.strip().split("-")
            if split_devices:
                rank_start = int(split_devices[0])
                device_list = list(range(rank_start, int(ascend_visible_devices.strip().split("-")[-1]) + 1))
        elif "," in ascend_visible_devices:
            device_list = list(map(int, ascend_visible_devices.strip().split(",")))
        elif ascend_visible_devices in VALID_DEVICE_ID_LIST:
            device_list = [int(ascend_visible_devices.strip())]
        else:
            raise ValueError("invalid env variable ascend_visible_devices.")
    except ValueError as error:
        raise ValueError("Invalid env variable ascend_visible_devices, no valid device id is configured.") from error
    except IndexError as error:
        raise IndexError(
            f"Index of ascend_visible_devices {ascend_visible_devices.strip().split('-')[-1]} is out of range") \
            from error
    if not device_list:
        raise ValueError("No device is available in the environment.")
    return device_list
