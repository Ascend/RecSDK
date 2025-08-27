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
from typing import Union

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training.optimizer import _TensorProcessor

from rec_sdk_common.log import logger
from rec_sdk_common.communication.hccl.hccl_info import get_rank_size
from rec_sdk_common.util.tf_adapter import hccl_ops, npu_ops
from mx_rec.core.asc.swap_args import SwapArgs
from mx_rec.constants.constants import ASCAnchorAttr
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.core.asc import get_restore_vector_second, get_unique_keys


def _get_padding_keys_offset_mask(table_name: str, max_lookup_vec_size: int) -> tf.Tensor:
    """
    Get the mask for the padding keys offset.

    Args:
        table_name: The table name.
        max_lookup_vec_size: The output shape of the get_next tensor.

    Returns: The get_next tensor.

    """

    channel_id = 0
    with tf.compat.v1.variable_scope(table_name, reuse=tf.compat.v1.AUTO_REUSE):
        padding_keys_offset_mask = npu_ops.gen_npu_ops.get_next(
            output_types=[tf.int32],
            output_shapes=[[max_lookup_vec_size]],
            channel_name=f"{table_name}_mask_{channel_id}",
        )[0]
    logger.debug(
        "Channel %s_mask_%d was built for getnext %s.",
        table_name,
        channel_id,
        padding_keys_offset_mask,
    )

    return padding_keys_offset_mask


class CustomizedOptimizer:
    MAX_COUNTER_COUNT = 1000
    name_counter = defaultdict(int)

    def __init__(self):
        self.unique_name = ""
        self.base_name = ""
        self._slot_num = 0  # 优化器对应slot的个数
        self._derivative = 1  # 优化器阶数，如果不做全局去重可以数学等价，则为1阶，其余2阶

    @property
    def slot_num(self):
        return self._slot_num

    @property
    def derivative(self):
        return self._derivative

    @staticmethod
    def sum_same_id_gradients(grad, var, is_expansion):
        table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance(var)
        table_name = table_instance.table_name

        max_lookup_vec_size = None
        use_static = ConfigInitializer.get_instance().use_static
        if use_static:
            send_count = table_instance.send_count
            rank_size = get_rank_size()
            max_lookup_vec_size = send_count * rank_size if send_count > 0 else None
            if table_instance.is_dp:
                max_lookup_vec_size = send_count if send_count > 0 else None

        with tf.compat.v1.variable_scope(str(ASCAnchorAttr.RESTORE_VECTOR_SECOND)):
            restore_vector_second = get_restore_vector_second(table_name, max_lookup_vec_size)

        with tf.compat.v1.variable_scope(str(ASCAnchorAttr.UNIQUE_KEYS)):
            unique_keys = get_unique_keys(table_name, max_lookup_vec_size, is_expansion)

        unique_local_grad = tf.compat.v1.unsorted_segment_sum(
            grad, restore_vector_second, array_ops.shape(unique_keys)[0]
        )

        # The DP mode requires allreduce for gradients.
        if table_instance.is_dp:
            unique_local_grad = hccl_ops.allreduce(unique_local_grad, "sum")

        return unique_local_grad, unique_keys

    @staticmethod
    def _process_grad_value_mask(var: Union[tf.Tensor, tf.Variable], grad_value: tf.Tensor) -> tf.Tensor:
        """
        Set the gradient of the padding keys to 0 on the embedding.

        Args:
            var: It's tf.Tensor in dynamic expansion mode and tf.Variable in normal mode.
            grad_value: The gradient of the embedding.

        Returns: The embedding gradient after mask.

        """

        table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance(var)
        if not table_instance.padding_keys_mask:
            return grad_value

        max_lookup_vec_size = None
        if ConfigInitializer.get_instance().use_static:
            send_count = table_instance.send_count
            rank_size = get_rank_size()
            max_lookup_vec_size = send_count * rank_size if send_count > 0 else None
        padding_keys_offset_mask = _get_padding_keys_offset_mask(table_instance.table_name, max_lookup_vec_size)

        indices_mask = tf.cast(padding_keys_offset_mask, tf.bool)
        if tf.__version__.startswith("2"):
            indices_mask = tf.compat.v1.expand_dims(indices_mask, axis=-1)
        zeros_value = tf.zeros_like(grad_value)
        grad_value_mask = tf.where(indices_mask, grad_value, zeros_value)

        return grad_value_mask

    def get_slot_init_values(self):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}.")

    def _get_name(self, name="CustomizedOptimizer"):
        if name in CustomizedOptimizer.name_counter:
            CustomizedOptimizer.name_counter[name] += 1
            count = CustomizedOptimizer.name_counter.get(name)

        elif len(CustomizedOptimizer.name_counter) <= CustomizedOptimizer.MAX_COUNTER_COUNT:
            count = CustomizedOptimizer.name_counter[name]
        else:
            raise ValueError("The optimizer exceeds the max num limitation.")

        self.unique_name = name + "_" + str(count)
        self.base_name = name


def custom_update_op(self, opt, grad):
    if isinstance(grad, ops.Tensor):
        update_op = opt._apply_sparse(grad, self._v)
        return update_op
    else:
        raise RuntimeError("Only support g with type Tensor.")


def control_update_op_decorator(apply_sparse):
    def wrapper(*args, **kwargs):
        second_arg = args[2] if len(args) > 2 else None  # index 2 input must be var
        slot_control_ops = tf.no_op(name="place_holder_slot_control_op")
        swap_args = SwapArgs()
        swap_args.set_slot_control(var_name=second_arg, control_ops=slot_control_ops)
        with tf.control_dependencies([slot_control_ops]):
            result = apply_sparse(*args, **kwargs)
        return result

    return wrapper


def patch_for_optimizer():
    _TensorProcessor.update_op = custom_update_op
    logger.debug("The update_op in Class optimizer._TensorProcessor has been patched.")
