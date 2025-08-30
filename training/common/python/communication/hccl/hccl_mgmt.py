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
import json
from typing import Dict, List

from rec_sdk_common.constants.constants import RankTableInfo, ChipName, CommParams, CommonEnv, FileParams
from rec_sdk_common.validator.safe_checker import class_safe_check, int_safe_check


def _get_chip_name():
    import common_binding
    chipName = common_binding.get_chip_name(0)
    if "910B" in chipName:
        return ChipName.ASCEND_910B
    return ChipName.NONE


def _get_rank_info_with_ranktable() -> Dict[int, int]:
    """
    Used for rank table file configured training situation.
    :return: rank_id to logic_id mapping dictionary.
    """
    rank_table_path = os.getenv(RankTableInfo.RANK_TABLE_FILE.value, "")
    with open(rank_table_path, "r", encoding="utf-8") as file:
        try:
            ranktable_info = json.load(file)
        except FileNotFoundError as e:
            raise ValueError("rank table file not found") from e
        except json.JSONDecodeError as e:
            raise ValueError("rank table file is unable to parse as json") from e
        class_safe_check("ranktable_info", ranktable_info, (dict,))

        if RankTableInfo.SERVER_LIST.value not in ranktable_info:
            raise AttributeError(f"Lack of attribute server_list.")
        if not ranktable_info.get(RankTableInfo.SERVER_LIST.value):
            raise ValueError(f"Server_list is empty.")
        if RankTableInfo.DEVICE.value not in ranktable_info.get(RankTableInfo.SERVER_LIST.value)[0]:
            raise AttributeError(f"Lack of attribute device.")

    rank_to_device_dict = dict()
    for server_list in ranktable_info.get(RankTableInfo.SERVER_LIST.value):
        devices = server_list.get(RankTableInfo.DEVICE.value)
        if devices is None:
            raise ValueError("device is empty")

        for device in devices:
            if RankTableInfo.RANK_ID.value not in device or not device.get(RankTableInfo.RANK_ID.value).isdigit():
                raise ValueError(f"hccl_json rank_id wrong.")
            rank_id = int(device.get(RankTableInfo.RANK_ID.value))
            int_safe_check("rank_id", rank_id, min_value=0, max_value=CommParams.MAX_RANK_ID.value)
            if RankTableInfo.DEVICE_ID.value not in device or not device.get(RankTableInfo.DEVICE_ID.value).isdigit():
                raise ValueError(f"hccl_json device_id wrong.")
            import common_binding
            logic_id = common_binding.get_logic_id(int(device.get(RankTableInfo.DEVICE_ID.value)))
            int_safe_check("logic_id", logic_id, min_value=0, max_value=CommParams.MAX_LOGIC_ID.value)
            rank_to_device_dict[rank_id] = logic_id
    return rank_to_device_dict


def _get_rank_info_without_ranktable() -> Dict[int, int]:
    """
    Used for no rank table file configured training situation.
    :return: rank_id to logic_id mapping dictionary.
    """
    device_list = get_device_list()
    env_rank_size = os.getenv(CommonEnv.CM_WORKER_SIZE.value)
    env_chief_device = os.getenv(CommonEnv.CM_CHIEF_DEVICE.value)
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
    import common_binding
    device_count = common_binding.get_device_count()
    device_list = [i for i in range(device_count)]
    return device_list