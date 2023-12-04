#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from tensorflow.python.ops import math_ops
from tensorflow.python.training import gradient_descent

from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.initialize import get_host_pipeline_ops, insert_optimizer, get_use_dynamic_expansion
from mx_rec.constants.constants import MAX_INT32
from mx_rec.validator.validator import para_checker_decorator, StringValidator, ClassValidator, FloatValidator


@para_checker_decorator(check_option_list=[
    ("learning_rate", FloatValidator, {"min_value": -MAX_INT32, "max_value": MAX_INT32}, ["check_value"]),
    ("weight_decay", FloatValidator, {"min_value": 0, "max_value": 1}, ["check_value"]),
    ("use_locking", ClassValidator, {"classes": (bool,)}),
    ("name", StringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"])
])
def create_hash_optimizer_by_addr(learning_rate, weight_decay=0.0001, use_locking=False, name="GradientDescentByAddr"):
    if not get_use_dynamic_expansion():
        raise ValueError("dynamic expansion mode is not compatible with the optimizer, please config dynamic "
                         "expansion mode and optimizer correctly")
    optimizer_by_addr = CustomizedGradientDescentByAddr(learning_rate=learning_rate,
                                                        weight_decay=weight_decay,
                                                        use_locking=use_locking,
                                                        name=name)
    insert_optimizer(optimizer_by_addr)
    return optimizer_by_addr


class CustomizedGradientDescentByAddr(gradient_descent.GradientDescentOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate, weight_decay, use_locking=False, name="GradientDescentByAddr"):
        self.optimizer_type = "gradient_descent_by_addr"
        self.weight_decay = weight_decay
        super(CustomizedGradientDescentByAddr, self)._get_name(name=name)
        super(CustomizedGradientDescentByAddr, self).__init__(learning_rate=learning_rate, use_locking=use_locking,
                                                              name=self.unique_name)

        self._slot_num = 0

    @property
    def slot_num(self):
        return self._slot_num

    def initialize_slots(self, var, table_instance):
        return []

    def get_slot_init_values(self):
        return []

    def _apply_sparse(self, grad, addr):
        host_pipeline_ops = get_host_pipeline_ops()
        dim = grad.shape.as_list()[-1]
        if self.weight_decay is None:
            nd_value = grad * math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype)
        else:
            lookup_tensor = \
                host_pipeline_ops.embedding_lookup_by_address(addr, embedding_dim=dim, embedding_type=1)
            nd_value = (grad + math_ops.cast(self.weight_decay, grad.dtype.base_dtype) * lookup_tensor) * math_ops.cast(
                self._learning_rate_tensor, grad.dtype.base_dtype)
        var_update_op = host_pipeline_ops.embedding_update_by_address(addr, -nd_value, update_type=0)

        return var_update_op

    def _apply_dense(self, grad, var):
        raise NotImplementedError("You are using a wrong type of variable.")



