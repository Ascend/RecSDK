# coding=utf-8
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
from tensorflow.python.training import gradient_descent
from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.initialize import ConfigInitializer

from demo_logger import logger


def create_hash_optimizer(learning_rate, weight_decay=0.0001, use_locking=False, name="GradientDescent"):
    optimizer = CustomizedGradientDescentWithWeighDecay(learning_rate=learning_rate,
                                                        weight_decay=weight_decay,
                                                        use_locking=use_locking,
                                                        name=name)
    ConfigInitializer.get_instance().optimizer_config.optimizer_instance = optimizer
    return optimizer


class CustomizedGradientDescentWithWeighDecay(gradient_descent.GradientDescentOptimizer, CustomizedOptimizer):
    name_counter = defaultdict(int)

    def __init__(self, learning_rate, weight_decay, use_locking=False, name="GradientDescent"):
        self.optimizer_type = "gradient_descent_with_weight_decay"
        self.weight_decay = weight_decay
        super(CustomizedGradientDescentWithWeighDecay, self)._get_name(name=name)
        super(CustomizedGradientDescentWithWeighDecay, self).__init__(
            learning_rate=learning_rate, use_locking=use_locking, name=self.unique_name
        )
        self._slot_num = 0
        self._derivative = 1

    def get_slot_init_values(self):
        logger.info("no slot for gradient descent")
        return []

    def _apply_sparse_duplicate_indices(self, grad, var):
        logger.debug(">>>> Enter _apply_sparse_duplicate_indices")
        nd_indices = tf.expand_dims(grad.indices, 1)
        logger.info(f"weigh_decay={self.weight_decay}")
        if self.weight_decay is None:
            nd_value = grad.values * math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        else:
            nd_value = (grad.values + math_ops.cast(self.weight_decay, var.dtype.base_dtype) *
                        tf.gather(var, grad.indices)) * math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        var_update_op = tf.scatter_nd_add(var, nd_indices, -nd_value, use_locking=self._use_locking)
        return var_update_op

    def _apply_dense(self, grad, var):
        logger.debug(">>>> Enter _apply_dense")
        raise NotImplementedError("You are using a wrong type of variable.")
