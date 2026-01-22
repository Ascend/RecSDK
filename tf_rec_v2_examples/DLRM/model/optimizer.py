#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

from typing import Tuple, Optional, List

import tensorflow as tf
import mxrec

from config import Config


def get_dense_and_sparse_optimizer(cfg: Config) -> Tuple[tf.compat.v1.train.Optimizer, tf.compat.v1.train.Optimizer]:
    # Due to the unavailability of many operators, use the `Adam` optimizer to get the functionality working.
    dense_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.learning_rate[0])
    sparse_optimizer = mxrec.AdamWOptimizer(weight_decay=cfg.weight_decay, learning_rate=cfg.learning_rate[1])

    dense_optimizer = DenseLossScaleOptimizer(dense_optimizer, cfg.loss_scale)
    sparse_optimizer = SparseLossScaleOptimizer(sparse_optimizer, cfg.loss_scale)

    return dense_optimizer, sparse_optimizer


class SparseLossScaleOptimizer:
    """A custom optimizer that scales the loss before computing sparse gradients."""

    def __init__(self, opt: tf.compat.v1.train.Optimizer, loss_scale: int):
        if not isinstance(opt, tf.compat.v1.train.Optimizer):
            raise ValueError("`opt` must be an instance of Optimizer, but got: %s" % type(opt))
        self._optimizer = opt
        self._loss_scale = tf.convert_to_tensor(loss_scale, tf.float32)
        _scale_learning_rate(self._optimizer, loss_scale)

    def compute_gradients(self, loss: tf.Tensor, var_list: Optional[List[tf.Variable]] = None) -> List[tf.Tensor]:
        return tf.gradients(loss * self._loss_scale, var_list)

    def apply_gradients(self, grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]]) -> tf.Operation:
        return self._optimizer.apply_gradients(grads_and_vars)


class DenseLossScaleOptimizer:
    """A custom optimizer that scales the loss before computing dense gradients."""

    def __init__(self, opt: tf.compat.v1.train.Optimizer, loss_scale: int):
        if not isinstance(opt, tf.compat.v1.train.Optimizer):
            raise ValueError("`opt` must be an instance of Optimizer, but got: %s" % type(opt))
        self._optimizer = opt
        self._loss_scale = tf.convert_to_tensor(loss_scale, tf.float32)
        _scale_learning_rate(self._optimizer, loss_scale)

    def compute_gradients(self, loss: tf.Tensor, var_list: Optional[List[tf.Variable]] = None) -> List[tf.Tensor]:
        return self._optimizer.compute_gradients(loss * self._loss_scale, var_list=var_list)

    def apply_gradients(self, avg_grads: List[Tuple[tf.Tensor, tf.Variable]]) -> tf.Operation:
        return self._optimizer.apply_gradients(avg_grads)


def _scale_learning_rate(opt: tf.compat.v1.train.Optimizer, loss_scale: float):
    if loss_scale == 0:
        raise ValueError("loss scale can not be zero")

    if hasattr(opt, "_learning_rate"):
        # SGD or Adagrad optimizer.
        opt._learning_rate = opt._learning_rate / tf.convert_to_tensor(loss_scale, tf.float32)
    elif hasattr(opt, "_lr"):
        # Adam optimizer.
        opt._lr = opt._lr / tf.convert_to_tensor(loss_scale, tf.float32)
    else:
        raise ValueError("`opt` should have a `_learning_rate` or `_lr` named field")
