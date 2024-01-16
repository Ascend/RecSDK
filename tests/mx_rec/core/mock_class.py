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

import tensorflow as tf
from tensorflow_core.python.training import slot_creator

from mx_rec.optimizers.lazy_adam import CustomizedLazyAdam


class MockSparseEmbedding:
    """
    原始SparseEmbedding会调用很多接口，用MockSparseEmbedding防止mock过多接口
    """

    def __init__(self, table_name="test_table", slice_device_vocabulary_size=10, embedding_size=5, init_param=1.,
                 emb_initializer=tf.zeros_initializer()):
        self.table_name = table_name
        self.slice_device_vocabulary_size = slice_device_vocabulary_size
        self.embedding_size = tf.TensorShape([embedding_size])
        self.init_param = init_param
        self.emb_initializer = emb_initializer
        self.variable = tf.compat.v1.get_variable(table_name,
                                                  shape=[slice_device_vocabulary_size, embedding_size],
                                                  trainable=False, initializer=tf.ones_initializer())


class MockHostPipeLineOps:
    """
    用于mock host_pipeline_ops，返回静态/动态 readEmbKey
    """

    def __init__(self):
        def _mock_read_emb_key_v2_fn(concat_tensor, **kwargs):
            return 0

        def _mock_read_emb_key_v2_dynamic_fn(concat_tensor, tensorshape_split_list, **kwargs):
            return 1

        self.read_emb_key_v2 = _mock_read_emb_key_v2_fn
        self.read_emb_key_v2_dynamic = _mock_read_emb_key_v2_dynamic_fn


class MockHcclOps:
    """
    用于mock hccl_ops
    """

    def __init__(self, shape=None):
        def _mock_all_to_all_v(send_data, send_counts, send_displacements, recv_counts, recv_displacements):
            if shape is None:
                return tf.constant(1, dtype=tf.int64, name="all_to_all_v")
            return tf.ones(shape, dtype=tf.int64, name="all_to_all_v")

        def _mock_all_to_all_v_c(send_data, send_count_matrix, rank):
            if shape is None:
                return tf.constant(1, dtype=tf.int64, name="all_to_all_v_c")
            return tf.ones(shape, dtype=tf.int64, name="all_to_all_v_c")

        self.all_to_all_v = _mock_all_to_all_v
        self.all_to_all_v_c = _mock_all_to_all_v_c


class MockOptimizer(CustomizedLazyAdam):
    """
    用于mock optimizer
    """

    def __init__(self):
        super(MockOptimizer, self)._get_name(name="MockLazyAdam")
        super(MockOptimizer, self).__init__(learning_rate=0.001, beta1=0.9, beta2=0.999,
                                            epsilon=1e-8, use_locking=False, name="MockLazyAdam")
        self.slot_num = 2

    def initialize_slots(self, var, table_instance):
        # Create slots for the first and second moments.
        def creat_one_single_slot(var, op_name):
            new_slot_variable = slot_creator.create_zeros_slot(var, op_name)
            return new_slot_variable

        momentum = creat_one_single_slot(var, self._name + "/" + "momentum")
        velocity = creat_one_single_slot(var, self._name + "/" + "velocity")
        named_slot_key = (var.op.graph, var.op.name)

        table_instance.set_optimizer(self._name, {"momentum": momentum, "velocity": velocity})
        return [{"slot": momentum, "named_slot_key": named_slot_key, "slot_name": "m", "optimizer": self},
                {"slot": velocity, "named_slot_key": named_slot_key, "slot_name": "v", "optimizer": self}]

    def insert_slot(self, slot, named_slots_key, slot_name):
        pass

    def get_slot_init_values(self):
        initial_momentum_value = 0.0
        initial_velocity_value = 0.0
        return [initial_momentum_value, initial_velocity_value]

    def update_op(self, optimizer, g):
        return super().update_op(optimizer, g)

    def _apply_spare_duplicate_indices(self, grad, var):
        return self._apply_sparse(grad, var)

    def _apply_sparse(self, grad, var):
        return super()._apply_sparse(grad, var)

    def _resource_apply_sparse(self, grad, handle, indices):
        return super()._resource_apply_sparse(grad, handle, indices)

    def _apply_dense(self, grad, var):
        return super()._apply_dense(grad, var)

    def _apply_sparse_duplicate_indices(self, grad, var):
        return self._apply_sparse(grad, var)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return self._resource_apply_sparse(grad, handle, indices)


class MockAscManager:
    """
    用于mock get_asc_manager()
    """

    def __init__(self):
        def _mock_get_table_size(self):
            return 0

        def _mock_get_table_capacity(self):
            return 1

        self.get_table_size = _mock_get_table_size
        self.get_table_capacity = _mock_get_table_capacity


class MockHybridMgmt:
    """
    用于mock HybridMgmt()
    """

    def __init__(self, is_initialized=True):
        def _mock_initialize(rank_info=0, emb_info=1, if_load=False, threshold_values=3):
            return is_initialized

        self.initialize = _mock_initialize
