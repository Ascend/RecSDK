#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import tensorflow as tf


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


class MockOptimizer:
    """
    用于mock optimizer
    """

    def __init__(self):
        def _mock_insert_slot(slot, named_slot_key, slot_name):
            return "mock_insert_slot"

        self.insert_slot = _mock_insert_slot


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
