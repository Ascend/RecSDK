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

from typing import Union

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training.optimizer import Optimizer, _TensorProcessor

from rec_sdk_common.validator.safe_checker import class_safe_check, float_safe_check
from mxrec.python.constants.constants import NumCheckValueMethod


def patch_for_update_op():
    def _update_op(self, optimizer: Optimizer, g: tf.Tensor):
        return optimizer._apply_sparse(g, self._v)

    _TensorProcessor.update_op = _update_op


def check_optimizer_param_value(
    value: Union[float, tf.Tensor],
    name: str,
    max_value: float,
    min_value: float = 0.0,
    check_value_method: str = NumCheckValueMethod.DEFAULT.value,
):
    class_safe_check(name, value, (float, tf.Tensor))
    if not isinstance(value, float):
        return
    float_safe_check(name, value, min_value=min_value, max_value=max_value, method=check_value_method)


def deduplicate_indexed_slices(grad: ops.IndexedSlices) -> ops.IndexedSlices:
    # Unique local ids, because there will be duplicate keys in the local ids after all-to-all communication.
    u_local_ids, u_local_idx = tf.unique(grad.indices)

    # To deduplicate gradients.
    unique_local_grad = tf.compat.v1.unsorted_segment_sum(
        data=grad.values,
        segment_ids=u_local_idx,
        num_segments=tf.shape(u_local_ids)[0],
        name="unique_local_grad",
    )
    deduplicate_grad = ops.IndexedSlices(
        values=unique_local_grad,
        indices=u_local_ids,
        dense_shape=grad.dense_shape,
    )

    return deduplicate_grad
