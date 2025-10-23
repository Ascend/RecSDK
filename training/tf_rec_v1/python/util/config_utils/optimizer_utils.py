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

from typing import Dict

import tensorflow as tf


class OptimizerConfig:
    def __init__(self):
        self._optimizer_instance = None
        # This true and false represent whether it is in training mode.
        self._table_optimizer_dict = {True: {}, False: {}}

    @property
    def optim_params_list(self):
        if not self._optimizer_instance:
            return []
        return self._optimizer_instance.optim_param_list

    @property
    def optimizer_instance(self):
        return self._optimizer_instance

    @optimizer_instance.setter
    def optimizer_instance(self, optimizer):
        self._optimizer_instance = optimizer

    def set_optimizer_for_table(
        self, table_name: str, optimizer_name: str, optimizer_dict: Dict[str, tf.Variable], is_training: bool = True
    ):
        self._table_optimizer_dict[is_training][table_name] = {optimizer_name: optimizer_dict}

    def get_optimizer_by_table_name(self, table_name: str, is_training: bool = True) -> Dict[str, tf.Variable]:
        if self._table_optimizer_dict.get(is_training) is None:
            raise KeyError(f"key `{is_training}` does not exist")
        return self._table_optimizer_dict.get(is_training).get(table_name)
