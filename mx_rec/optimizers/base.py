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
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training.optimizer import _TensorProcessor

from mx_rec.util.tf_version_adapter import npu_ops
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.log import logger


class CustomizedOptimizer:

    name_counter = defaultdict(int)

    def __init__(self):
        self.unique_name = ""
        self.base_name = ""

    def initialize_slots(self, var, table_instance):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")

    def insert_slot(self, slot, named_slots_key, slot_name):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")

    def get_slot_init_values(self):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")

    def _get_name(self, name="CustomizedOptimizer"):
        if name in CustomizedOptimizer.name_counter:
            CustomizedOptimizer.name_counter[name] += 1
            count = CustomizedOptimizer.name_counter.get(name)

        else:
            count = CustomizedOptimizer.name_counter[name]
        self.unique_name = name + "_" + str(count)
        self.base_name = name

    def get_restore_vector_second(self, table_name: str) -> tf.Tensor:
        """
        Get restore vector which is calculated after the second all2all
        :param table_name: embedding table_name
        :return: the restore vector calculated after the second all2all
        """
        channel_id = 0
        logger.debug('Channel %s_restore_second_%s was built for getnext',
                     table_name, channel_id)
        with tf.compat.v1.variable_scope(table_name, reuse=tf.compat.v1.AUTO_REUSE):
            restore_vector_second = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.int32],
                output_shapes=[[None]],
                channel_name=f'{table_name}_restore_second_{channel_id}')[0]
        return restore_vector_second

    def get_unique_keys(self, table_name: str, is_expansion: bool) -> tf.Tensor:
        """
        Get the global unique keys which is calculated after the second all2all
        :param table_name: embedding table_name
        :param is_expansion: use dynamic expansion
        :return: the global unique keys calculated after the second all2all
        """
        channel_id = 0
        logger.debug('Channel %s_uniquekeys_%s was built for getnext', table_name, channel_id)
        with tf.compat.v1.variable_scope(table_name, reuse=tf.compat.v1.AUTO_REUSE):
            if is_expansion:
                unique_keys = npu_ops.gen_npu_ops.get_next(
                    output_types=[tf.int64],
                    output_shapes=[[None]],
                    channel_name=f'{table_name}_uniquekeys_{channel_id}')[0]
                return unique_keys

            unique_keys = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.int32],
                output_shapes=[[None]],
                channel_name=f'{table_name}_uniquekeys_{channel_id}')[0]
            return unique_keys

    def sum_same_id_gradients(self, grad, var, is_expansion):
        if isinstance(var, ops.Tensor):
            # 扩容模式从scope获取表名
            table_name = var.op.name.split('/')[0]
        else:
            table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance(var)
            table_name = table_instance.table_name
        with tf.compat.v1.variable_scope("restore_vector_second"):
            restore_vector_second = self.get_restore_vector_second(table_name)

        with tf.compat.v1.variable_scope("unique_keys"):
            unique_keys = self.get_unique_keys(table_name, is_expansion)

        unique_local_grad = tf.compat.v1.unsorted_segment_sum(grad,
                                                              restore_vector_second,
                                                              array_ops.shape(unique_keys)[0])
        return unique_local_grad, unique_keys


def custom_update_op(self, opt, grad):
    if isinstance(grad, ops.Tensor):
        update_op = opt._apply_sparse(grad, self._v)
        return update_op
    else:
        raise RuntimeError("Only support g with type Tensor.")


def patch_for_optimizer():
    _TensorProcessor.update_op = custom_update_op
    logger.debug("update_op in Class optimizer._TensorProcessor has been patched.")