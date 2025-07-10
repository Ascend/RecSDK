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

import os
import logging
from datetime import datetime

import pytz

BEHAVIOUR_AND_MULTI_TASK = {"BST", "CAN", "DFFM", "DIN", "DMT", "ESMM", "ETA", "MMOE", "PLE", "SharedBottom"}
FEATURE_INTERACTION = {"AFN_plus", "AFN", "AutoInt_plus", "AutoInt", "DCNv2", "DeepFM", "FFM", "FiBiNet",
                       "FM", "IPNN", "LR", "OPNN", "PNN", "WideDeep"}


def setup_logger(model_config, model_name):
    logger = logging.getLogger()
    log_level = getattr(logging, model_config.log_level.upper(), logging.DEBUG)
    logger.setLevel(log_level)
    console_hand = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(asctime)s: %(message)s")
    console_hand.setLevel(log_level)
    console_hand.setFormatter(formatter)
    logger.addHandler(console_hand)

    # Define the timezone for China Standard Time
    china_tz = pytz.timezone('Asia/Shanghai')
    logfile_na = model_name + "_" + datetime.now(china_tz).strftime("%Y_%m_%d_%H_%M_%S") + ".log"
    if model_name in BEHAVIOUR_AND_MULTI_TASK:
        logfile_path = os.path.join("../log/aliccp/", logfile_na)
    elif model_name in FEATURE_INTERACTION:
        logfile_path = os.path.join("../log/criteo/", logfile_na)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger