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

import copy
import logging
import os
import numpy as np
import tensorflow as tf
from utils import nested_dict_to_str


DENSE_ALLCLOSE_RTOL = 1e-10


class DenseModel:
    def __init__(self, data_dir:str, data_step:int):
        self.dense_path = os.path.join(data_dir, "02dump_model", f"model-{data_step}")

        var_list = tf.train.list_variables(self.dense_path)
        self.var_name_list = [var_item[0] for var_item in var_list]
        self.var_dict = {}

        for var_name in self.var_name_list:
            tensor = tf.train.load_variable(self.dense_path, var_name)
            self.var_dict[var_name] = tensor

    def __eq__(self, other) -> bool:
        if self.var_name_list != other.var_name_list:
            logging.error(
                f"Dense ckpt var items not equal!\n"
                f"Test var_name_list:{self.var_name_list}\n"
                f"Golden var_name_list:{other.var_name_list}\n"
            )
            return False

        for var_name in self.var_name_list:
            test_var = self.var_dict[var_name]
            golden_var = other.var_dict[var_name]
            if test_var.shape != golden_var.shape:
                logging.error(
                    f"[DenseModel]Test and Golden shape not equal!Variable name:{var_name}\n"
                    f"Test:{test_var.shape}\n"
                    f"Golden:{golden_var.shape}\n"
                )
                return False

            if not np.allclose(test_var, golden_var, rtol=DENSE_ALLCLOSE_RTOL):
                logging.error(
                    f"[DenseModel]Test and Golden value not equal!Variable name:{var_name}\n"
                    f"Test var_name_list:\n{test_var}"
                    f"Golden var_name_list:\n{golden_var}\n"
                )
                return False
        return True
