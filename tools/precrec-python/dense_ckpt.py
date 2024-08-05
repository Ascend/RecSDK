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

import logging
import os

import numpy as np
import tensorflow as tf

DUMP_DENSE_STR = "02dump_model"
DENSE_ALLCLOSE_RTOL = 1e-10


class DenseModel:
    """
    This class is used to represent parsed dense model for choosen step and rank.
    """

    def __init__(self, data_dir: str, data_step: int):
        self.dense_path = os.path.join(data_dir, DUMP_DENSE_STR, f"model-{data_step}")

        var_list = tf.train.list_variables(self.dense_path)
        self.var_name_list = [var_item[0] for var_item in var_list]
        self.var_dict = {}

        for var_name in self.var_name_list:
            tensor = tf.train.load_variable(self.dense_path, var_name)
            self.var_dict[var_name] = tensor

    def __eq__(self, other) -> bool:
        if self.var_name_list != other.var_name_list:
            logging.error(
                "Dense ckpt var items not equal!\nTest var_name_list:%s\nGolden var_name_list:%s\n",
                self.var_name_list,
                other.var_name_list,
            )
            return False

        for var_name in self.var_name_list:
            test_var = self.var_dict[var_name]
            golden_var = other.var_dict[var_name]
            if test_var.shape != golden_var.shape:
                logging.error(
                    "[DenseModel]Test and Golden shape not equal!Variable name:%s\nTest:%s\nGolden:%s\n",
                    var_name,
                    test_var.shape,
                    golden_var.shape,
                )
                return False

            if not np.allclose(test_var, golden_var, rtol=DENSE_ALLCLOSE_RTOL):
                logging.error(
                    "[DenseModel]Test and Golden value not equal!Variable name:%s\n"
                    "Test var_name_list:\n%sGolden var_name_list:\n%s\n",
                    var_name,
                    test_var,
                    golden_var,
                )
                return False
        return True
