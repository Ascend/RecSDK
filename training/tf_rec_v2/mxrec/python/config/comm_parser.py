#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import ipaddress

from rec_sdk_common.validator.safe_checker import str_safe_check, int_safe_check
from mxrec.python.constants.constants import CommNodeInfo, CommParams
from mxrec.python.config.parser import TomlParser, parse_field, parse_env_field


def parse_comm_info() -> CommNodeInfo:
    config = TomlParser.get_instance().config
    info = parse_field(config, CommParams.CM_NODE_INFO.value.lower())

    cm_chief_ip = parse_env_field(info, CommParams.CM_CHIEF_IP.value.lower())
    _check_ipv4("cm_chief_ip", cm_chief_ip)

    cm_chief_port = parse_env_field(info, CommParams.CM_CHIEF_PORT.value.lower())
    int_safe_check(
        "cm_chief_port",
        cm_chief_port,
        CommParams.MIN_CM_CHIEF_PORT.value,
        CommParams.MAX_CM_CHIEF_PORT.value,
    )

    cm_chief_device = parse_env_field(info, CommParams.CM_CHIEF_DEVICE.value.lower())
    int_safe_check("cm_chief_device", cm_chief_device, 0)

    cm_worker_ip = parse_env_field(info, CommParams.CM_WORKER_IP.value.lower())
    _check_ipv4("cm_worker_ip", cm_worker_ip)

    cm_worker_size = parse_env_field(info, CommParams.CM_WORKER_SIZE.value.lower())
    int_safe_check(
        "cm_worker_size",
        cm_worker_size,
        CommParams.MIN_CM_WORKER_SIZE.value,
        CommParams.MAX_CM_WORKER_SIZE.value,
    )

    comm_node_info = CommNodeInfo(
        cm_chief_ip=cm_chief_ip,
        cm_chief_port=cm_chief_port,
        cm_chief_device=cm_chief_device,
        cm_worker_ip=cm_worker_ip,
        cm_worker_size=cm_worker_size,
    )
    return comm_node_info


def _check_ipv4(name: str, ip: str):
    str_safe_check(
        name, ip, CommParams.MIN_IPV4_LEN.value, CommParams.MAX_IPV4_LEN.value
    )
    try:
        ipaddress.IPv4Address(ip)
    except ipaddress.AddressValueError as e:
        raise ValueError(
            f"the {name} {ip} is not a valid IPv4 address, please check"
        ) from e
