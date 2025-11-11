#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

from enum import Enum
from typing import Union

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.framework import ops

from rec_sdk_common.util.tf_adapter import hccl_ops, gen_npu_cpu_ops
from rec_sdk_common.validator.safe_checker import str_safe_check
from mxrec.python.optimizer.utils import deduplicate_indexed_slices, check_optimizer_param_value
from mxrec.python.embedding.utils import EmbeddingResource, get_table_ins_by_local_embedding
from mxrec.python.constants.constants import NumCheckValueMethod, EmbDistributionStrategy


class _AdamWOptParams(Enum):
    LEARNING_RATE: float = 10.0
    WEIGHT_DECAY: float = 10.0
    BETA_1: float = 1.0
    BETA_2: float = 1.0
    EPSILON: float = 1.0
    MAX_NAME_LEN: int = 128


class AdamWOptimizer(optimizer.Optimizer):
    """The custom optimizer that implements the adam weight decay algorithm."""

    def __init__(
        self,
        learning_rate: Union[float, tf.Tensor] = 0.01,
        weight_decay: Union[float, tf.Tensor] = 0.004,
        beta_1: Union[float, tf.Tensor] = 0.9,
        beta_2: Union[float, tf.Tensor] = 0.999,
        epsilon: Union[float, tf.Tensor] = 1e-6,
        name: str = "AdamWOptimizer",
    ):
        super(AdamWOptimizer, self).__init__(False, name)

        check_optimizer_param_value(learning_rate, "learning_rate", max_value=_AdamWOptParams.LEARNING_RATE.value)
        check_optimizer_param_value(weight_decay, "weight_decay", max_value=_AdamWOptParams.WEIGHT_DECAY.value)
        check_optimizer_param_value(
            beta_1,
            "beta_1",
            max_value=_AdamWOptParams.BETA_1.value,
            check_value_method=NumCheckValueMethod.RIGHT_OPEN_INTERVAL.value,
        )
        check_optimizer_param_value(
            beta_2,
            "beta_2",
            max_value=_AdamWOptParams.BETA_2.value,
            check_value_method=NumCheckValueMethod.RIGHT_OPEN_INTERVAL.value,
        )
        check_optimizer_param_value(
            epsilon,
            "epsilon",
            max_value=_AdamWOptParams.EPSILON.value,
            check_value_method=NumCheckValueMethod.RIGHT_OPEN_INTERVAL.value,
        )
        str_safe_check("name", name, min_len=1, max_len=_AdamWOptParams.MAX_NAME_LEN.value)

        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self._amsgrad = False
        self._maximize = False

        self._beta1_power = tf.compat.v1.Variable(initial_value=0.9, name="beta1_power")
        self._beta2_power = tf.compat.v1.Variable(initial_value=0.9, name="beta2_power")
        self._lr_t = None
        self._weight_decay_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        weight_decay = self._call_if_callable(self._weight_decay)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._weight_decay_t = ops.convert_to_tensor(weight_decay, name="weight_decay")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

    def _apply_dense(self, grad: ops.IndexedSlices, var: tf.Variable) -> tf.Tensor:
        raise NotImplementedError("cannot use sparse optimizer to update dense gradient")

    def _resource_apply_dense(self, grad: tf.Tensor, handle: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("cannot use sparse optimizer to update dense gradient")

    def _apply_sparse(self, grad: ops.IndexedSlices, var: tf.Tensor) -> tf.Tensor:
        table_ins = get_table_ins_by_local_embedding(var)
        # Create slot.
        _m = tf.compat.v1.Variable(
            tf.compat.v1.random_uniform([table_ins.slice_dev_vocab_size, table_ins.dim], minval=1.0, maxval=1.0),
            name="m",
        )
        _v = tf.compat.v1.Variable(
            tf.compat.v1.random_uniform([table_ins.slice_dev_vocab_size, table_ins.dim], minval=1.0, maxval=1.0),
            name="v",
        )
        _max_grad_norm = tf.compat.v1.Variable(
            tf.compat.v1.random_uniform([table_ins.slice_dev_vocab_size, table_ins.dim], minval=1.0, maxval=1.0),
            name="max_grad_norm",
        )
        # Get table handle.
        var_ref = EmbeddingResource(table_ins.table_id)

        # Gradient deduplication.
        if table_ins.dist_strategy == EmbDistributionStrategy.MP.value:
            deduplicate_grad = deduplicate_indexed_slices(grad)
        else:
            raise ValueError(
                f"currently, the embedding table only support MP(model parallelism), "
                f"but got {table_ins.dist_strategy}"
            )
        grad_dtype = deduplicate_grad.values.dtype

        update_op = gen_npu_cpu_ops.embedding_hash_table_apply_adam_w(
            table_handle=var_ref.handle,
            m=_m,
            v=_v,
            beta1_power=self._beta1_power,
            beta2_power=self._beta2_power,
            lr=math_ops.cast(self._lr_t, grad_dtype),
            weight_decay=math_ops.cast(self._weight_decay_t, grad_dtype),
            beta1=math_ops.cast(self._beta1_t, grad_dtype),
            beta2=math_ops.cast(self._beta2_t, grad_dtype),
            epsilon=math_ops.cast(self._epsilon_t, grad_dtype),
            grad=deduplicate_grad.values,
            keys=deduplicate_grad.indices,
            max_grad_norm=_max_grad_norm,
            embedding_dim=table_ins.dim,
            bucket_size=table_ins.slice_dev_vocab_size,
            amsgrad=self._amsgrad,
            maximize=self._maximize,
        )

        return update_op
