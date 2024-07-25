#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import ctypes
import psutil

from mx_rec.util.log import logger

PCIE_INFO_ALL_RESERVE_LEN = 32
g_dcmi = None


class PcieInfo(ctypes.Structure):
    _fields_ = [
        ("venderid", ctypes.c_uint),
        ("subdeviceid", ctypes.c_uint),
        ("deviceid", ctypes.c_uint),
        ("subdeviceid", ctypes.c_uint),
        ("domain", ctypes.c_int),
        ("bdf_busid", ctypes.c_uint),
        ("bdf_deviceid", ctypes.c_uint),
        ("bdf_funcid", ctypes.c_uint),
        ("reserve", ctypes.c_char * PCIE_INFO_ALL_RESERVE_LEN)
    ]


def get_card_and_device(logic_id):
    """
    通过芯片逻辑id获取芯片的卡id和device id
    一张卡可能有多个芯片，对应多个device_id，但每个芯片的逻辑ID
    是不一样的
    """
    card = ctypes.c_int(0)
    device = ctypes.c_int(0)
    logic = ctypes.c_uint(logic_id)
    ret = g_dcmi.dcmi_get_card_id_device_id_from_logicid(ctypes.pointer(card),
                                                         ctypes.pointer(device),
                                                         logic)
    if ret != 0:
        raise OSError(f"logic id {logic_id} not exist")
    return card.value, device.value


def get_pcie_id(card_id, device_id):
    """
    get pcie <domain>:<bus>:<device_id>.<funcid>
    """
    info = PcieInfo()
    card = ctypes.c_int(card_id)
    dev = ctypes.c_int(device_id)
    ret = g_dcmi.dcmi_get_device_pcie_info_v2(card, dev, ctypes.pointer(info))
    if ret != 0:
        raise OSError(f"cant get pcie info of device {card_id}:{device_id}")
    pcie_id = f'{info.domain:04X}:{info.bdf_busid:02x}:'
    pcie_id += f'{info.bdf_deviceid:02x}.{info.bdf_funcid}'
    return pcie_id


def get_numa_by_pcie(pcie_id):
    """
    get numa node by pcie id
    """
    with open(f'/sys/bus/pci/devices/{pcie_id}/numa_node') as f:
        numa_node = f.read()
        return int(numa_node)


def get_cpu_list_by_numa(node):
    with open(f'/sys/devices/system/node/node{node}/cpulist') as f:
        cpulistinfo = f.read()
        cpulist_first = cpulistinfo.split(",")[0]
        [cpu_start, cpu_end] = cpulist_first.split("-")
        return list(range(int(cpu_start), int(cpu_end)))


def bind_cpu_by_device_logic_id(logic_id):
    global g_dcmi
    if g_dcmi is None:
        try:
            g_dcmi = ctypes.CDLL("libdcmi.so")
            if g_dcmi.dcmi_init() != 0:
                logger.error("dcmi init failed")
                return False
        except OSError as e:
            logger.error(e)
            return False
    try:
        card_id, device_id = get_card_and_device(logic_id)
        pcie_id = get_pcie_id(card_id, device_id)
        numa = get_numa_by_pcie(pcie_id)
        cpu_list = get_cpu_list_by_numa(numa)
        psutil.Process().cpu_affinity(cpu_list)
    except Exception as e:
        logger.error(e)
        return False
    return True
