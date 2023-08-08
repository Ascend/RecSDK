#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import json
import os

from mx_rec.constants.constants import VALID_DEVICE_ID_LIST, MIN_SIZE, MAX_CONFIG_SIZE, MAX_DEVICE_ID
from mx_rec.validator.validator import RankInfoValidator, FileValidator


def parse_hccl_json():
    rank_table_path = os.path.realpath(os.getenv("RANK_TABLE_FILE"))
    if not os.path.exists(rank_table_path):
        raise FileExistsError(f"Target_hccl_json_dir {rank_table_path} does not exist when reading.")

    with open(rank_table_path, "r", encoding="utf-8") as file:
        # check whether json file is valid
        file_validator = FileValidator(rank_table_path)
        # 1.check whether rank_table_path is soft link
        file_validator.check_not_soft_link()
        # 2.check json file size
        file_validator.check_file_size(file, MAX_CONFIG_SIZE, MIN_SIZE)
        file_validator.check()

        table_hccl = json.load(file)
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
            device_id = mxrec_pybind.get_logic_id(int(device.get("device_id")))
            if device_id > MAX_DEVICE_ID:
                raise ValueError(f"get logic id from physic id fail, the device id is invalid.")
            rank_to_device_dict[rank_id] = device_id

    return rank_to_device_dict


def set_hccl_info_without_json():
    """
    Used for no rank table file configured training situation.
    Now, only less than or equal 8p training job is supported.
    :return: None
    """
    RankInfoValidator().check_visible_devices()
    ascend_visible_devices = os.getenv("ASCEND_VISIBLE_DEVICES")
    device_list = get_device_list(ascend_visible_devices)

    chief_device = os.getenv("CM_CHIEF_DEVICE")
    rank_size = os.getenv("CM_WORKER_SIZE")
    sorted_device_list = sorted(device_list)
    if int(rank_size) != len(sorted_device_list):
        raise ValueError(f"Rank size {rank_size} is different from device num {len(sorted_device_list)}.")
    rank_to_device_dict = dict()
    try:
        rank_to_device_dict[0] = int(chief_device)
    except ValueError as err:
        raise ValueError("CM_WORKER_SIZE or CM_CHIEF_DEVICE uncorrected configured.") from err
    try:
        sorted_device_list.pop(int(chief_device) % len(sorted_device_list))
    except IndexError as err:
        raise IndexError(
            f"Config CM_CHIEF_DEVICE {chief_device} not in training container device list {sorted_device_list}.") \
            from err
    except ZeroDivisionError as err:
        raise ZeroDivisionError("sorted_device_list length can not equal to 0.") from err

    for device_idx in sorted_device_list:
        import mxrec_pybind

        try:
            device_id = mxrec_pybind.get_logic_id(int(device_idx))
            if device_id > MAX_DEVICE_ID:
                raise ValueError(f"get logic id from physic id fail.")
            index = sorted_device_list.index(device_idx)
            rank_to_device_dict[index + 1] = device_id
        except RuntimeError as exp:
            raise RuntimeError(f"get logic id from physic id fail. Possible reasons: 1) running user permission "
                               f"is not enough to call dsmi api 2) driver has been used by other process") from \
                exp
    return rank_to_device_dict


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
        raise ValueError("Invalid env variable ascend_visible_devices, no valid device id is configured. "
                         "Please refer to the document https://www.hiascend.com/document/detail/zh/"
                         "CANNCommunityEdition/63RC2alpha002/ptmoddevg/ptmigr/ptmigr_0151.html for "
                         "the correct configuration method.") from error
    except IndexError as error:
        raise IndexError(
            f"Index of ascend_visible_devices {ascend_visible_devices.strip().split('-')[-1]} is out of range") \
            from error
    return device_list
