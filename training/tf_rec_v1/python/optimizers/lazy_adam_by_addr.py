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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training import adam

from rec_sdk_common.validator.validator import(
    para_checker_decorator,
    StringValidator,
    FloatValidator,
    LearningRateValidator)
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.ops import import_host_pipeline_ops


@para_checker_decorator(
    check_option_list=[
        ("learning_rate", LearningRateValidator, {"min_value": 0.0, "max_value": 10.0}, ["check_value"]),
        ("beta1", FloatValidator, {"min_value": 0.0, "max_value": 1.0}, ["check_value_for_open_interval"]),
        ("beta2", FloatValidator, {"min_value": 0.0, "max_value": 1.0}, ["check_value"]),
        ("epsilon", FloatValidator, {"min_value": 0.0, "max_value": 1.0}, ["check_value_for_left_open_interval"]),
        ("name", StringValidator, {"min_len": 1, "max_len": 200}, ["check_string_length"]),
    ]
)
def create_hash_optimizer_by_address(
    learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="LazyAdamByAddress"
):
    """
    Args:
        learning_rate: learning rate
        beta1:
        beta2:
        epsilon:
        name:

    Returns: a customized optimizer instance
    """
    if not ConfigInitializer.get_instance().use_dynamic_expansion:
        raise ValueError(
            "The dynamic expansion mode is not compatible with the optimizer, please config dynamic "
            "expansion mode and optimizer correctly."
        )

    optimizer_by_addr = CustomizedLazyAdamByAddress(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, name=name
    )
    ConfigInitializer.get_instance().optimizer_config.optimizer_instance = optimizer_by_addr
    return optimizer_by_addr


class CustomizedLazyAdamByAddress(adam.AdamOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(
        self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name="LazyAdamByAddress"
    ):
        self.optimizer_type = "LazyAdam"
        self.optim_param_list = ["momentum", "velocity"]
        super(CustomizedLazyAdamByAddress, self)._get_name(name=name)
        super(CustomizedLazyAdamByAddress, self).__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            use_locking=use_locking,
            name=self.unique_name,
        )

        self._slot_num = 2
        self._derivative = 2

    def get_slot_init_values(self):
        # return state value list of adam that needs to initialize in ASC DDR.
        initial_momentum_value = 0.0
        initial_velocity_value = 0.0
        return [initial_momentum_value, initial_velocity_value]

    def _create_slots(self, addr_list):
        first_addr = addr_list[0]
        self._create_non_slot_variable(initial_value=self._beta1, name="beta1_power", colocate_with=first_addr)
        self._create_non_slot_variable(initial_value=self._beta2, name="beta2_power", colocate_with=first_addr)

    def _apply_dense(self, grad, var):
        raise NotImplementedError("You are using a wrong type of variable.")

    def _cast_to_base_type(self, var):
        var_type = var.dtype.base_dtype
        temp_lr = math_ops.cast(self._lr_t, var_type)
        temp_b1 = math_ops.cast(self._beta1_t, var_type)
        temp_b2 = math_ops.cast(self._beta2_t, var_type)
        temp_epsilon = math_ops.cast(self._epsilon_t, var_type)
        temp = {
            "temp_lr": temp_lr,
            "temp_b1": temp_b1,
            "temp_b2": temp_b2,
            "temp_epsilon": temp_epsilon,
        }
        return temp

    def _apply_sparse(self, grad, addr):
        unique_local_grad, unique_addr = self.sum_same_id_gradients(grad=grad, var=addr, is_expansion=True)
        return self._apply_sparse_shared(unique_local_grad, unique_addr, addr)

    def _apply_sparse_shared(self, grad: tf.Tensor, addr: tf.Tensor, var_tensor: tf.Tensor) -> tf.Tensor:
        power_b1, power_b2 = self._get_beta_accumulators()
        power_b1 = math_ops.cast(power_b1, grad.dtype.base_dtype)
        power_b2 = math_ops.cast(power_b2, grad.dtype.base_dtype)
        temp = self._cast_to_base_type(grad)
        temp_lr = temp.get("temp_lr")
        temp_b1 = temp.get("temp_b1")
        temp_b2 = temp.get("temp_b2")
        temp_epsilon = temp.get("temp_epsilon")
        learning_rate = tf.divide(temp_lr * math_ops.sqrt(1 - power_b2), (1 - power_b1))

        host_pipeline_ops = import_host_pipeline_ops()
        dim = grad.shape.as_list()[-1]
        combined_tensor = host_pipeline_ops.embedding_lookup_by_address(addr, embedding_dim=3 * dim, embedding_type=1)

        split_length = [dim] + [dim] + [dim]
        split_tensors = tf.split(combined_tensor, split_length, axis=1)

        old_m_slice = split_tensors[1]
        m_t_slice = temp_b1 * old_m_slice + (1 - temp_b1) * grad

        old_v_slice = split_tensors[2]
        v_t_slice = temp_b2 * old_v_slice + (1 - temp_b2) * math_ops.square(grad)

        denominator_slice = math_ops.sqrt(tf.abs(v_t_slice)) + temp_epsilon
        nd_value = tf.divide(-learning_rate * m_t_slice, denominator_slice)
        nd_value = self._process_grad_value_mask(var_tensor, nd_value)
        update_list = [nd_value] + [m_t_slice - old_m_slice] + [v_t_slice - old_v_slice]
        update_tensor = tf.concat(update_list, axis=1)
        var_update_op = host_pipeline_ops.embedding_update_by_address(addr, update_tensor, update_type=0)

        return var_update_op
