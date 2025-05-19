#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

import logging
import sys
import os

def setup_logger(name, log_file=None, level=logging.INFO):
    """配置日志器"""
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(filename)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 文件输出 handler（可选）
    file_handler = None
    if log_file:
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

    # 初始化 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(console_handler)
    if file_handler:
        logger.addHandler(file_handler)

    return logger

# 实例化默认 logger
default_logger = setup_logger("PatternLogger", log_file="logs/pattern_output.log")