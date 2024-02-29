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

import tensorflow as tf
from tensorflow.python.training import optimizer


class DenseLossScaleOptimizer:
    def __init__(self, opt, loss_scale):
        if not isinstance(opt, optimizer.Optimizer):
            raise ValueError('"opt" must be an instance of Optimizer, but got: %s' % type(opt))
        self._optimizer = opt
        self._loss_scale = tf.convert_to_tensor(loss_scale, tf.float32)
        self._optimizer._learning_rate = self._optimizer._learning_rate / self._loss_scale

    def compute_gradients(self, loss, var_list=None):
        return self._optimizer.compute_gradients(loss*self._loss_scale, var_list=var_list)

    def apply_gradients(self, avg_grads):
        return self._optimizer.apply_gradients(avg_grads)


class SparseLossScaleOptimizer:
    def __init__(self, opt, loss_scale):
        if not isinstance(opt, optimizer.Optimizer):
            raise ValueError('"opt" must be an instance of Optimizer, but got: %s' % type(opt))
        self._optimizer = opt
        self._loss_scale = tf.convert_to_tensor(loss_scale, tf.float32)
        self._optimizer._learning_rate = self._optimizer._learning_rate / self._loss_scale

    def compute_gradients(self, loss, var_list=None):
        return tf.gradients(loss*self._loss_scale, var_list)

    def apply_gradients(self, grads_and_vars):
        return self._optimizer.apply_gradients(grads_and_vars)