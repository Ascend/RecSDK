#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import os
from collections import defaultdict
import psutil

from rec_sdk_common.constants.constants import FileParams, EnvOptionCommon, LogLevel
from rec_sdk_common.communication.hccl.hccl_info import get_local_rank_size, get_rank_id
from rec_sdk_common.log.log import LoggingProxy


def get_available_cpu_num_and_range():
    """
    获取当前环境可用的cpu数量和numa范围
    Returns: cpu的数量，numa范围

    """
    cpu_available = os.sched_getaffinity(os.getpid())  # 获取可被绑定的核心

    is_ok = True
    cpu_pkg_id_file = "/sys/devices/system/cpu/cpu{}/topology/physical_package_id"
    pkg_id2cpu_list = defaultdict(list)
    for cpu in cpu_available:
        f_path = cpu_pkg_id_file.format(cpu)
        if not os.path.exists(f_path):
            LoggingProxy.warning("failed to get numa node of cpu: %s", cpu)
            is_ok = False
            break

        with open(f_path, "r", encoding="utf-8") as f_in:
            pkg_id = f_in.readline().strip()
            pkg_id2cpu_list[pkg_id].append(cpu)

    def parse_range(cpu_list, cpu_range):
        sorted_cpu_list = sorted(cpu_list)
        pre_cpu = sorted_cpu_list[0]
        cpu_range.append([pre_cpu])

        for sorted_cpu in sorted_cpu_list[1:]:
            if sorted_cpu - pre_cpu != 1:
                cpu_range[-1].append(pre_cpu)
                cpu_range.append([sorted_cpu])
            pre_cpu = sorted_cpu

        if len(cpu_range[-1]) == 1:
            cpu_range[-1].append(pre_cpu)

    valid_cpu_range_list = []
    if is_ok:
        LoggingProxy.info("available numa node num: %s", len(pkg_id2cpu_list))
        for _, part_cpu_list in pkg_id2cpu_list.items():
            parse_range(part_cpu_list, valid_cpu_range_list)
    else:
        parse_range(list(cpu_available), valid_cpu_range_list)
    return len(cpu_available), valid_cpu_range_list


def bind_cpu_task():
    """
    为每个进程绑定CPU
    """
    import math
    total_cpu, cpu_range_list = get_available_cpu_num_and_range()
    local_rank_size = get_local_rank_size()
    if local_rank_size <= 0:
        LoggingProxy.error(f"local rank size 's value less than or equal 0.")
        return

    avg_count = math.ceil(total_cpu / local_rank_size)
    while True:
        if avg_count == 0:
            LoggingProxy.warning(f"not enough cpu to bind. cpu num: %s, range: %s", total_cpu, cpu_range_list)
            return

        max_split = 0
        for cpu_range in cpu_range_list:
            max_split += (cpu_range[1] - cpu_range[0] + 1) // avg_count
        if max_split >= local_rank_size:
            break
        avg_count -= 1

    candidate_list = []
    for cpu_range in cpu_range_list:
        start = cpu_range[0]
        splits = (cpu_range[1] - cpu_range[0] + 1) // avg_count
        candidate_range = [list(range(start + i * avg_count, start + ((i + 1) * avg_count))) for i in range(splits)]
        candidate_list.extend(candidate_range)

    rank_id = get_rank_id()
    cpu_list = candidate_list[rank_id % local_rank_size]  # 取模适配多机

    process = psutil.Process()
    try:
        process.cpu_affinity(cpu_list)
    except IndexError:
        LoggingProxy.error("failed to bind cpu for rank %s: %s", rank_id, cpu_list)
    LoggingProxy.info("bind cpu for rank %s: %s", rank_id, cpu_list)


def bind_cpu(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        if kwargs.get("bind_cpu", True):
            bind_cpu_task()

    return wrapper
