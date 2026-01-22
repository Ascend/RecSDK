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

import glob
import os
import shutil
import random
import argparse

import toml
import tensorflow as tf
import numpy as np
import mxrec

from config import Config
from runner import TaskRunner
from logger import logger


tf.compat.v1.disable_eager_execution()
RAND_SEED = 128
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
tf.compat.v1.set_random_seed(RAND_SEED)


def _del_related_dir(del_path: str):
    if not os.path.isabs(del_path):
        del_path = os.path.join(os.getcwd(), del_path)
    dirs = glob.glob(del_path)
    for sub_dir in dirs:
        shutil.rmtree(sub_dir, ignore_errors=True)
        logger.info("Delete dir: %s.", sub_dir)


if __name__ == "__main__":
    if not tf.__version__.startswith("1"):
        raise EnvironmentError("this model only supports tensorFlow 1.x")
    _del_related_dir("/root/ascend/log/*")
    _del_related_dir("kernel*")

    parser = argparse.ArgumentParser("DLRM for Rec SDK.")
    parser.add_argument("--data_path", type=str, default="", help="The dataset path.")
    parser.add_argument("--toml_path", type=str, default="./dlrm.toml", help="Path of toml file.")
    params = parser.parse_args()

    with open(params.toml_path, "r") as f:
        toml_config = toml.load(f)

    runner = TaskRunner(
        config=Config(params.data_path, toml_config),
        toml_config=toml_config,
        seed=RAND_SEED,
    )

    mxrec.init(params.toml_path)
    runner.run()
    logger.info("Demo done.")
