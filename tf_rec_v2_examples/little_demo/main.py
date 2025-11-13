#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
random.seed(Config.random_seed)
np.random.seed(Config.random_seed)
if tf.__version__.startswith("1"):
    tf.compat.v1.set_random_seed(Config.random_seed)
else:
    tf.random.set_seed(Config.random_seed)


def _del_related_dir(del_path: str) -> None:
    if not os.path.isabs(del_path):
        del_path = os.path.join(os.getcwd(), del_path)
    dirs = glob.glob(del_path)
    for sub_dir in dirs:
        shutil.rmtree(sub_dir, ignore_errors=True)
        logger.info("Delete dir: %s.", sub_dir)


def _clear_saved_model(mode: str, saved_path: str) -> None:
    _del_related_dir("/root/ascend/log/*")
    _del_related_dir("kernel*")

    if mode != Config.train_and_evaluate:
        return

    logger.info("Current mode is train, will delete previous saved model data if exist.")
    _del_related_dir(saved_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for Rec SDK.")
    parser.add_argument("-p", "--path", type=str, default="./demo.toml", help="Path of toml file.")
    params = parser.parse_args()

    with open(params.path, "r") as f:
        toml_config = toml.load(f)

    mode = toml_config["model"]["mode"]
    saved_path = toml_config["model"]["saved_path"]
    _clear_saved_model(mode, saved_path)
    train_steps = toml_config["model"]["train_steps"]
    eval_steps = toml_config["model"]["eval_steps"]
    train_interval = toml_config["model"]["train_interval"]
    batch_number = toml_config["model"]["batch_number"]
    is_deterministic = toml_config["model"]["deterministic"]

    runner = TaskRunner(
        train_steps=train_steps,
        train_interval=train_interval,
        eval_steps=eval_steps,
        batch_number=batch_number * Config.rank_size,
        is_deterministic=is_deterministic,
    )

    mxrec.init(params.path)
    runner.run(mode, saved_path)
    logger.info("Demo done.")
