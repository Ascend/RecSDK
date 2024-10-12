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
from tensorflow.python.framework import ops
from tensorflow.python.training import gradient_descent

from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.tf_version_adapter import hccl_ops
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.validator.validator import (
    para_checker_decorator,
    StringValidator,
    ClassValidator,
    FloatValidator,
)


@para_checker_decorator(check_option_list=[
    ("learning_rate", FloatValidator, {"min_value": 0.0, "max_value": 10.0}, ["check_value"]),
    ("use_locking", ClassValidator, {"classes": (bool,)}),
    ("name", StringValidator, {"min_len": 1, "max_len": 200}, ["check_string_length"])
])
def create_hash_optimizer(learning_rate, use_locking=False, name="GradientDescent"):
    if ConfigInitializer.get_instance().use_dynamic_expansion:
        raise ValueError(
            "dynamic expansion mode is not compatible with the optimizer, please config dynamic "
            "expansion mode and optimizer correctly."
        )
    optimizer = CustomizedGradientDescent(
        learning_rate=learning_rate, use_locking=use_locking, name=name
    )
    ConfigInitializer.get_instance().optimizer_config.optimizer_instance = optimizer
    return optimizer


class CustomizedGradientDescent(
    gradient_descent.GradientDescentOptimizer, CustomizedOptimizer
):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
        self.optimizer_type = "gradient_descent"
        self.optim_param_list = []
        super(CustomizedGradientDescent, self)._get_name(name=name)
        super(CustomizedGradientDescent, self).__init__(
            learning_rate=learning_rate, use_locking=use_locking, name=self.unique_name
        )
        self._slot_num = 0
        self._derivative = 1

    def get_slot_init_values(self):
        return []

    def _apply_sparse_duplicate_indices(self, grad, var):
        table_instance = (
            ConfigInitializer.get_instance().sparse_embed_config.get_table_instance(var)
        )
        # The DP mode requires allreduce for gradients.
        if table_instance.is_dp:
            # In the DP mode, the second USS is used to align the length of the grad in the allreduce.
            unique_local_grad, unique_keys = self.sum_same_id_gradients(grad=grad.values, var=var, is_expansion=False)
            grad = ops.IndexedSlices(
                values=unique_local_grad,
                indices=unique_keys,
                dense_shape=grad.dense_shape,
            )
        nd_indices = tf.expand_dims(grad.indices, 1)
        nd_value = grad.values * math_ops.cast(
            self._learning_rate_tensor, var.dtype.base_dtype
        )
        var_update_op = tf.scatter_nd_add(
            var, nd_indices, -nd_value, use_locking=self._use_locking
        )
        return var_update_op

    def _apply_dense(self, grad, var):
        raise NotImplementedError("You are using a wrong type of variable.")
