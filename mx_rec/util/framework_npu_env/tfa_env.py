#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import os

from mx_rec.util.communication.hccl_ops import get_rank_id, get_rank_size, get_device_id
from mx_rec.util.global_env_conf import global_env
from mx_rec.util.log import logger


def set_ascend_env():
    """
    配置昇腾相关的参数和环境变量
    """
    logger.debug("Ascend env set start.")
    os.environ["RANK_ID"] = str(get_rank_id())

    device_id = str(get_device_id())
    os.environ["DEVICE_ID"] = device_id
    os.environ["ASCEND_DEVICE_ID"] = device_id
    os.environ["DEVICE_INDEX"] = device_id

    if global_env.rank_table_file:
        rank_size = get_rank_size()
        os.environ["RANK_SIZE"] = str(rank_size)

    os.environ["JOB_ID"] = "10086"
    logger.debug("Ascend env has been set.")
