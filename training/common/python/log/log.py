#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

from rec_sdk_common.constants.constants import LogLevel, EnvOptionCommon


class LoggingProxy:
    _instance: logging.Logger = None

    @classmethod
    def set_instance(cls, log_level: str):
        if cls._instance is not None:
            raise RuntimeError("LoggingProxy has been initialized once, twice initialization was forbidden")

        cls._instance = _get_logger(log_level)

    @classmethod
    def info(cls, msg: str, *args, **kwargs):
        cls._instance.info(msg, *args, **kwargs)

    @classmethod
    def debug(cls, msg: str, *args, **kwargs):
        cls._instance.debug(msg, *args, **kwargs)

    @classmethod
    def error(cls, msg: str, *args, **kwargs):
        cls._instance.error(msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg: str, *args, **kwargs):
        cls._instance.warning(msg, *args, **kwargs)

    @classmethod
    def exception(cls, msg: str, *args, exc_info: bool = True, **kwargs):
        cls._instance.error(msg, *args, exc_info=exc_info, **kwargs)

    @classmethod
    def critical(cls, msg: str, *args, **kwargs):
        cls._instance.critical(msg, *args, **kwargs)


def _get_logger(log_level: str = LogLevel.INFO.value) -> logging.Logger:
    options = [i.value for i in list(LogLevel)]
    if log_level not in options:
        raise ValueError(f"log level set for mxRec is not valid, only {options} are allowed, but got {log_level}")

    rec_logger = logging.getLogger("RecSDK")
    formatter = logging.Formatter(fmt="[RecSDK][%(asctime)s] [%(levelname)s] %(message)s",
                                  datefmt="%m/%d/%Y %H:%M:%S %p")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    rec_logger.addHandler(stream_handler)
    rec_logger.setLevel(log_level)
    return rec_logger

LoggingProxy.set_instance(log_level=os.getenv(EnvOptionCommon.RECSDK_LOG_LEVEL.value, LogLevel.INFO.value))