#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import os
import logging

from mx_rec.constants.constants import RecPyLogLevel, EnvOption


def get_logger(log_level: str):
    options = [i.value for i in list(RecPyLogLevel)]
    if log_level not in options:
        raise ValueError(f"log level set for mxRec is not valid, only {options} are allowed, but got {log_level}")

    rec_logger = logging.getLogger("MxRec")
    formatter = logging.Formatter(fmt="[MxRec][%(asctime)s] [%(levelname)s] %(message)s",
                                  datefmt="%m/%d/%Y %H:%M:%S %p")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    rec_logger.addHandler(stream_handler)
    rec_logger.setLevel(log_level)
    return rec_logger


logger = get_logger(log_level=os.getenv(EnvOption.MXREC_LOG_LEVEL.value, RecPyLogLevel.INFO.value))
