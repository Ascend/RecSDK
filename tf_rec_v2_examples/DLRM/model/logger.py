#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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


def _get_logger(log_level: str) -> logging.Logger:
    dlrm_logger = logging.getLogger("dlrm_model")
    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %H:%M:%S %p")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    dlrm_logger.addHandler(stream_handler)
    dlrm_logger.setLevel(log_level)
    return dlrm_logger


logger = _get_logger("DEBUG")
