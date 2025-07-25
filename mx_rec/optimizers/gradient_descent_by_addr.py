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

from tensorflow.python.ops import math_ops
from tensorflow.python.training import gradient_descent

from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.validator.validator import (
    para_checker_decorator,
    StringValidator,
    ClassValidator,
    FloatValidator,
    LearningRateValidator,
)


@para_checker_decorator(
    check_option_list=[
        ("learning_rate", LearningRateValidator, {"min_value": 0.0, "max_value": 10.0}, ["check_value"]),
        ("weight_decay", FloatValidator, {"min_value": 0.0, "max_value": 1.0}, ["check_value"]),
        ("use_locking", ClassValidator, {"classes": (bool,)}),
        ("name", StringValidator, {"min_len": 1, "max_len": 200}, ["check_string_length"]),
    ]
)
def create_hash_optimizer_by_addr(learning_rate, weight_decay=0.0001, use_locking=False, name="GradientDescentByAddr"):
    if not ConfigInitializer.get_instance().use_dynamic_expansion:
        raise ValueError(
            "The dynamic expansion mode is not compatible with the optimizer, please config dynamic "
            "expansion mode and optimizer correctly."
        )
    optimizer_by_addr = CustomizedGradientDescentByAddr(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_locking=use_locking,
        name=name,
    )
    ConfigInitializer.get_instance().optimizer_config.optimizer_instance = optimizer_by_addr
    return optimizer_by_addr


class CustomizedGradientDescentByAddr(gradient_descent.GradientDescentOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(
        self,
        learning_rate,
        weight_decay,
        use_locking=False,
        name="GradientDescentByAddr",
    ):
        self.optimizer_type = "gradient_descent"
        self.weight_decay = weight_decay
        self.optim_param_list = []
        super(CustomizedGradientDescentByAddr, self)._get_name(name=name)
        super(CustomizedGradientDescentByAddr, self).__init__(
            learning_rate=learning_rate, use_locking=use_locking, name=self.unique_name
        )

        self._slot_num = 0
        self._derivative = 1

    def get_slot_init_values(self):
        return []

    def _apply_sparse(self, grad, addr):
        # The var tensor is used to obtain the table instance in dynamic expansion mode.
        var_tensor = addr
        table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance(addr)
        # The DP mode requires allreduce for gradients.
        if table_instance.is_dp:
            # In the DP mode, the second USS is used to align the length of the grad in the allreduce.
            grad, addr = self.sum_same_id_gradients(grad=grad, var=addr, is_expansion=True)
        host_pipeline_ops = import_host_pipeline_ops()
        dim = grad.shape.as_list()[-1]
        if self.weight_decay is None:
            nd_value = grad * math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype)
        else:
            lookup_tensor = host_pipeline_ops.embedding_lookup_by_address(addr, embedding_dim=dim, embedding_type=1)
            nd_value = (grad + math_ops.cast(self.weight_decay, grad.dtype.base_dtype) * lookup_tensor) * math_ops.cast(
                self._learning_rate_tensor, grad.dtype.base_dtype
            )

        nd_value = self._process_grad_value_mask(var_tensor, nd_value)
        var_update_op = host_pipeline_ops.embedding_update_by_address(addr, -nd_value, update_type=0)

        return var_update_op

    def _apply_dense(self, grad, var):
        raise NotImplementedError("You are using a wrong type of variable.")
