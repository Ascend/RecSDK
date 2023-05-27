#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import logging


def get_log_level():
    env_log_level = os.getenv("MXREC_LOG_LEVEL")
    if env_log_level is None:
        env_log_level = "INFO"

    log_level = logging.getLevelName(env_log_level)
    if not isinstance(log_level, int):
        raise EnvironmentError("A wrong log level string was given.")

    log_format = "%(asctime)s\t%(levelname)s\t%(message)s"
    date_format = "%m/%d/%Y %H:%M:%S %p"

    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)
