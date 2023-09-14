#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.training import gradient_descent

from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.constants.constants import MAX_INT32
from mx_rec.validator.validator import para_checker_decorator, StringValidator, ClassValidator, NumValidator


@para_checker_decorator(check_option_list=[
    ("learning_rate", NumValidator, {"min_value": -MAX_INT32, "max_value": MAX_INT32}, ["check_value"]),
    ("use_locking", ClassValidator, {"classes": (bool,)}),
    ("name", StringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"])
])
def create_hash_optimizer(learning_rate, use_locking=False, name="GradientDescent"):
    return CustomizedGradientDescent(learning_rate=learning_rate, use_locking=use_locking, name=name)


class CustomizedGradientDescent(gradient_descent.GradientDescentOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
        self.optimizer_type = "gradient_descent"
        super(CustomizedGradientDescent, self)._get_name(name=name)
        super(CustomizedGradientDescent, self).__init__(learning_rate=learning_rate, use_locking=use_locking,
                                                        name=self.unique_name)

    def initialize_slots(self, var, table_instance):
        return []

    def get_slot_init_values(self):
        return []

    def _apply_sparse_duplicate_indices(self, grad, var):
        nd_indices = tf.expand_dims(grad.indices, 1)
        nd_value = grad.values * math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        var_update_op = tf.scatter_nd_add(var, nd_indices, -nd_value, use_locking=self._use_locking)
        return var_update_op

    def _apply_dense(self, grad, var):
        raise NotImplementedError("You are using a wrong type of variable.")
