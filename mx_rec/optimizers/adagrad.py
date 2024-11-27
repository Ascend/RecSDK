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

from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import adagrad, training_ops

from mx_rec.optimizers.base import CustomizedOptimizer, control_update_op_decorator
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.validator.validator import (
    para_checker_decorator,
    StringValidator,
    ClassValidator,
    FloatValidator,
    LearningRateValidator,
)


@para_checker_decorator(
    check_option_list=[
        (
            "learning_rate",
            LearningRateValidator,
            {"min_value": 0.0, "max_value": 10.0},
            ["check_value"],
        ),
        (
            "initial_accumulator_value",
            FloatValidator,
            {"min_value": 0.0, "max_value": 1.0},
            ["check_value_for_left_open_interval"],
        ),
        ("use_locking", ClassValidator, {"classes": (bool,)}),
        (
            "name",
            StringValidator,
            {"min_len": 1, "max_len": 200},
            ["check_string_length"],
        ),
    ]
)
def create_hash_optimizer(
    learning_rate=0.001,
    initial_accumulator_value=0.9,
    use_locking=False,
    name="Adagrad",
):
    """
    Create an instance of adagrad hash optimizer
    :param learning_rate: A `Tensor` or a floating point value.  The learning rate.
    :param initial_accumulator_value:  A floating point value. Starting value for the accumulators, must be positive.
    :param use_locking: If `True` use locks for update operations.
    :param name: Optional name prefix for the operations created when applying gradients.  Defaults to "Adagrad".
    :return: adagrad hash optimizer instance
    """
    if ConfigInitializer.get_instance().use_dynamic_expansion:
        raise ValueError(
            "The dynamic expansion mode is not compatible with the optimizer, please config dynamic "
            "expansion mode and optimizer correctly."
        )
    optimizer = CustomizedAdagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=initial_accumulator_value,
        use_locking=use_locking,
        name=name,
    )
    ConfigInitializer.get_instance().optimizer_config.optimizer_instance = optimizer
    return optimizer


class CustomizedAdagrad(adagrad.AdagradOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(
        self,
        learning_rate,
        initial_accumulator_value,
        use_locking=False,
        name="Adagrad",
    ):
        self.optimizer_type = "Adagrad"
        self.optim_param_list = ["accumulator"]
        super(CustomizedAdagrad, self)._get_name(name=name)
        super(CustomizedAdagrad, self).__init__(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            use_locking=use_locking,
            name=self.unique_name,
        )
        self._slot_num = 1
        self._derivative = 2

    def get_slot_init_values(self):
        # return state value list of adagrad that needs to initialize in ASC DDR.
        initial_accumulator_value = 0.0
        return [initial_accumulator_value]

    def _create_slots(self, var_list):
        for var in var_list:
            dtype = var.dtype.base_dtype
            if var.get_shape().is_fully_defined():
                init = init_ops.constant_initializer(self._initial_accumulator_value, dtype=dtype)
            else:
                init = self._init_constant_op(var, dtype)

            acc_state_name = self._name + "/" + "accumulator"
            self._get_or_make_slot_with_initializer(var, init, var.get_shape(), dtype, "acc", acc_state_name)

    def _apply_sparse_duplicate_indices(self, grad, var):
        #  _apply_sparse_duplicate_indices method include tf.unique and unsorted_segment_sum operations which may
        #  introduce dynamic shape problem, if encounter that, please de-annotation the method below.
        unique_local_grad, unique_keys = self.sum_same_id_gradients(grad=grad.values, var=var, is_expansion=False)
        gradient_no_duplicate_indices = ops.IndexedSlices(
            indices=unique_keys, values=unique_local_grad, dense_shape=grad.dense_shape
        )
        return self._apply_sparse(gradient_no_duplicate_indices, var)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        unique_local_grad, unique_keys = self.sum_same_id_gradients(grad=grad, var=handle, is_expansion=False)
        return self._resource_apply_sparse(unique_local_grad, handle, unique_keys)

    @control_update_op_decorator
    def _apply_sparse(self, grad, var):
        acc = self.get_slot(var, "acc")
        table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance(var)
        if table_instance.padding_keys_mask:
            mask_value = self._process_grad_value_mask(var, grad.values)
            grad = ops.IndexedSlices(
                values=mask_value,
                indices=grad.indices,
                dense_shape=grad.dense_shape,
            )
        return training_ops.sparse_apply_adagrad(
            var,
            acc,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad.values,
            grad.indices,
            use_locking=self._use_locking,
        )

    @control_update_op_decorator
    def _resource_apply_sparse(self, grad, var, indices):
        acc = self.get_slot(var, "acc")
        grad = self._process_grad_value_mask(var, grad)
        return training_ops.resource_sparse_apply_adagrad(
            var.handle,
            acc.handle,
            math_ops.cast(self._learning_rate_tensor, grad.dtype),
            grad,
            indices,
            use_locking=self._use_locking,
        )
