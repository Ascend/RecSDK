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

import json
import os
import re

from mx_rec.constants.constants import VALID_DEVICE_ID_LIST, MIN_SIZE, MAX_CONFIG_SIZE, MAX_DEVICE_ID, \
    MIN_RANK_SIZE, MAX_RANK_SIZE
from mx_rec.validator.validator import FileValidator, para_checker_decorator, StringValidator, \
    Convert2intValidator
from mx_rec.util.global_env_conf import global_env


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
    for server_list in table_hccl.get("server_list"):
        devices = server_list.get("device")
        if devices is None:
            raise ValueError("device is empty")

        for device in devices:
            if "rank_id" not in device or not device.get("rank_id").isdigit():
                raise ValueError(f"hccl_json rank_id wrong.")
            rank_id = int(device.get("rank_id"))
            if "device_id" not in device or not device.get("device_id").isdigit():
                raise ValueError(f"hccl_json device_id wrong.")

            import mxrec_pybind
            res = mxrec_pybind.get_logic_id(int(device.get("device_id")))
            if res < 0:
                raise RuntimeError(
                    f"get logic id from physic id fail, error code is {res}, please check if dsmi api is functional.")
            if res > MAX_DEVICE_ID:
                raise ValueError(f"get logic id from physic id fail, the device id is invalid.")
            rank_to_device_dict[rank_id] = res

    return rank_to_device_dict


def set_hccl_info_without_json() -> dict:
    """
    Used for no rank table file configured training situation.
    :return: device_id and logic_id mapping.
    """
    visible_devices = global_env.ascend_visible_devices
    rank_size = global_env.cm_worker_size
    chief_device = global_env.cm_chief_device
    device_list = get_device_list(visible_devices)
    chief_device = int(chief_device)
    rank_size = int(rank_size)

    sorted_device_list = sorted(device_list)

    if chief_device not in sorted_device_list:
        raise ValueError(f"The environment variable CM_CHIEF_DEVICE {chief_device} is not in the local device list. ")

    rank_to_device_dict = {}
    chief_index = sorted_device_list.index(chief_device)
    sorted_device_list = sorted_device_list[chief_index:] + sorted_device_list[0: chief_index]
    sorted_device_list = sorted_device_list[:rank_size]

    for device_idx in sorted_device_list:
        import mxrec_pybind
        res = mxrec_pybind.get_logic_id(int(device_idx))
        if res < 0:
            raise RuntimeError(
                f"get logic id from physic id fail, error code is {res}, please check if dsmi api is functional.")

        if res > MAX_DEVICE_ID:
            raise ValueError(f"get logic id from physic id fail. res: {res}, chief_device: {chief_device}, "
                             f"device_idx: {device_idx}")
        index = sorted_device_list.index(device_idx)
        rank_to_device_dict[index] = res
    return rank_to_device_dict


def get_device_list(ascend_visible_devices):
    device_list = []
    try:
        nums = re.findall(r'\d+', ascend_visible_devices)
        # eg1:4-11, 则nums=['4', '11']   eg2:0-3,8-11  则nums['0', '3', '8', '11']
        if not all(int(i) <= MAX_DEVICE_ID for i in nums):
            raise ValueError("invalid env variable ascend_visible_devices.")
        ranges = re.findall(r'\d+-\d+', ascend_visible_devices)
        # eg1:4-11, 则ranges=['4-11']    eg2:0-3,8-11  则ranges['0-3', '8-11']
        for r in ranges:
            start, end = map(int, r.split('-'))  # '4-11', 则start 4, end 11.   ['0-3', '8-11']
            if start >= end:
                raise ValueError("invalid env variable ascend_visible_devices.")
            nums.extend(range(start, end + 1))
        device_list = sorted(list(set(map(int, nums))))
    except ValueError as error:
        raise ValueError("Invalid env variable ascend_visible_devices, no valid device id is configured.") from error

    if not device_list:
        raise ValueError("No device is available in the environment.")
    return device_list