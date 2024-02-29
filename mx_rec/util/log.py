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
