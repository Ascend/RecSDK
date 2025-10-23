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

from __future__ import absolute_import, division, print_function

from typing import List

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training import adagrad
from tensorflow.python.training.optimizer import Optimizer

from rec_sdk_common.validator.validator import (
    FloatValidator,
    StringValidator,
    para_checker_decorator
)
from mx_rec.validator.validator import LearningRateValidator
from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.ops import import_host_pipeline_ops


@para_checker_decorator(
    check_option_list=[
        ("learning_rate", LearningRateValidator, {"min_value": 0.0, "max_value": 10.0}, ["check_value"]),
        (
            "initial_accumulator_value",
            FloatValidator,
            {"min_value": 0.0, "max_value": 1.0},
            ["check_value_for_left_open_interval"],
        ),
        ("name", StringValidator, {"min_len": 1, "max_len": 200}, ["check_string_length"]),
    ]
)
def create_hash_optimizer_by_address(learning_rate=0.001, initial_accumulator_value=0.9, name="Adagrad") -> Optimizer:
    """Create an instance of adagrad hash optimizer.

    Args:
        learning_rate: A `Tensor` or a floating point value. The learning rate.
        initial_accumulator_value: A floating point value. Starting value for the accumulators, must be positive.
        name: Optional name prefix for the operations created when applying gradients. Defaults to "Adagrad".

    Returns:
        Adagrad hash optimizer instance

    Raises:
        ValueError: If `use_dynamic_expansion` was not set.
    """
    if not ConfigInitializer.get_instance().use_dynamic_expansion:
        raise ValueError(
            "dynamic expansion mode is not compatible with the optimizer, please config dynamic "
            "expansion mode and optimizer correctly"
        )
    optimizer = CustomizedAdagradByAddress(
        learning_rate=learning_rate,
        initial_accumulator_value=initial_accumulator_value,
        name=name,
    )
    ConfigInitializer.get_instance().optimizer_config.optimizer_instance = optimizer
    return optimizer


class CustomizedAdagradByAddress(adagrad.AdagradOptimizer, CustomizedOptimizer):
    def __init__(
        self,
        learning_rate: float,
        initial_accumulator_value: float,
        name="Adagrad",
    ):
        self.optimizer_type = "Adagrad"
        self.optim_param_list = ["accumulator"]
        super(CustomizedAdagradByAddress, self)._get_name(name=name)
        super(CustomizedAdagradByAddress, self).__init__(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            name=self.unique_name,
        )
        self._epsilon = 1e-7
        self._slot_num = 1
        self._derivative = 2

    def get_slot_init_values(self) -> List[float]:
        # return state value list of adagrad that needs to initialize in ASC DDR.
        return [self._initial_accumulator_value]

    def _apply_sparse(self, grad: tf.Tensor, var: tf.Tensor) -> tf.Operation:
        # The var tensor is used to obtain the table instance in dynamic expansion mode.
        var_tensor = var
        grad, var = self.sum_same_id_gradients(grad=grad, var=var, is_expansion=True)
        learning_rate_tensor = math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype)
        epsilon = math_ops.cast(self._epsilon, grad.dtype.base_dtype)

        host_pipeline_ops = import_host_pipeline_ops()
        dim = grad.shape.as_list()[-1]

        combined_tensor = host_pipeline_ops.embedding_lookup_by_address(var, embedding_dim=2 * dim, embedding_type=1)
        split_length = [dim] + [dim]
        split_tensors = tf.split(combined_tensor, split_length, axis=1)

        old_s_slice = split_tensors[1]
        s_t_slice = old_s_slice + math_ops.square(grad)

        denominator_slice = math_ops.sqrt(s_t_slice + epsilon)
        nd_value = tf.divide(-learning_rate_tensor * grad, denominator_slice)
        nd_value = self._process_grad_value_mask(var_tensor, nd_value)
        update_list = [nd_value] + [s_t_slice - old_s_slice]
        update_tensor = tf.concat(update_list, axis=1)
        var_update_op = host_pipeline_ops.embedding_update_by_address(var, update_tensor, update_type=0)

        return var_update_op

    def _create_slots(self, var_list: List[tf.Variable]):
        # slot变量由lookup算子控制 跳过父类的实现
        pass
