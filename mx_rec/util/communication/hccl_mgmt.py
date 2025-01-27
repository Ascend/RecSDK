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
from typing import Dict, List

from mx_rec.constants.constants import MIN_SIZE, MAX_CONFIG_SIZE, MAX_DEVICE_ID
from mx_rec.validator.validator import FileValidator
from mx_rec.util.global_env_conf import global_env


def parse_hccl_json() -> Dict[int, int]:
    """
    Used for rank table file configured training situation.
    :return: rank_id to logic_id mapping dictionary.
    """
    rank_table_path = global_env.rank_table_file
    with open(rank_table_path, "r", encoding="utf-8") as file:
        # check whether json file is valid
        file_validator = FileValidator("RANK_TABLE_FILE", rank_table_path)
        # 1.check whether rank_table_path is soft link
        file_validator.check_not_soft_link()
        # 2.check json file size
        file_validator.check_file_size(MAX_CONFIG_SIZE, MIN_SIZE)
        # 3.check file mode
        file_validator.check_file_mode()
        file_validator.check()

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
            logic_id = mxrec_pybind.get_logic_id(int(device.get("device_id")))
            if logic_id < 0:
                raise RuntimeError(f"get logic id from physic id fail, error code is {logic_id}, "
                                   f"please check if dsmi api is functional.")
            if logic_id > MAX_DEVICE_ID:
                raise ValueError(f"get logic id from physic id fail, the device id is invalid.")
            rank_to_device_dict[rank_id] = logic_id
    return rank_to_device_dict


def set_hccl_info_without_json() -> Dict[int, int]:
    """
    Used for no rank table file configured training situation.
    :return: rank_id to logic_id mapping dictionary.
    """
    env_rank_size = global_env.cm_worker_size
    env_chief_device = global_env.cm_chief_device
    device_list = get_device_list()
    chief_device = int(env_chief_device)
    rank_size = int(env_rank_size)

    if chief_device not in device_list:
        raise ValueError(f"The environment variable CM_CHIEF_DEVICE {chief_device} is not in the local device list. ")

    rank_to_device_dict = {}
    chief_index = device_list.index(chief_device)
    device_list = device_list[chief_index:] + device_list[:chief_index]
    device_list = device_list[:rank_size]

    for rank_id, device_id in enumerate(device_list):
        rank_to_device_dict[rank_id] = device_id
    return rank_to_device_dict


def get_device_list() -> List[int]:
    """
    Obtain the number of visible Ascend devices in the environment.
    :return: the logic id list of visible Ascend devices .
    """
    import mxrec_pybind
    device_count = mxrec_pybind.get_device_count()
    device_list = [i for i in range(device_count)]
    return device_list