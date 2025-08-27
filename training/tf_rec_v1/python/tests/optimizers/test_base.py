#!/usr/bin/env python3
# coding: UTF-8
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

import unittest
from unittest import mock

import tensorflow as tf

from mx_rec.optimizers.base import CustomizedOptimizer
from core.mock_class import MockConfigInitializer, MockSparseEmbedding


class TestCustomizedOptimizer(unittest.TestCase):
    def setUp(self):
        tf.compat.v1.reset_default_graph()

    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    def test_init_ok(self):
        opt = CustomizedOptimizer()
        self.assertEqual(opt.slot_num, 0)
        self.assertEqual(opt.derivative, 1)

    @mock.patch.multiple(
        "mx_rec.optimizers.base",
        get_rank_size=mock.MagicMock(return_value=1),
    )
    @mock.patch("mx_rec.optimizers.base.get_unique_keys")
    @mock.patch("mx_rec.optimizers.base.get_restore_vector_second")
    @mock.patch("mx_rec.optimizers.base.ConfigInitializer")
    def test_sum_same_id_gradients_with_static_and_no_dp(
        self, base_config_initializer, restore_vector_send, unique_keys
    ):
        test_table = MockSparseEmbedding()
        restore_vector_send.return_value = tf.constant([0, 1, 0], dtype=tf.int32)
        unique_keys.return_value = tf.constant([[0, 1, 0], [0, 1, 0]], dtype=tf.int32)
        mock_config_init = MockConfigInitializer(use_static=True, var=test_table)
        base_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        opt = CustomizedOptimizer()
        grad = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=tf.float32)
        res = opt.sum_same_id_gradients(grad, test_table.variable, is_expansion=False)
        self.assertIsNotNone(res)

    @mock.patch.multiple(
        "mx_rec.optimizers.base",
        get_rank_size=mock.MagicMock(return_value=1),
    )
    @mock.patch("mx_rec.optimizers.base.get_unique_keys")
    @mock.patch("mx_rec.optimizers.base.get_restore_vector_second")
    @mock.patch("mx_rec.optimizers.base.ConfigInitializer")
    def test_sum_same_id_gradients_with_static_and_dp(self, base_config_initializer, restore_vector_send, unique_keys):
        test_table = MockSparseEmbedding()
        test_table.is_dp = True
        restore_vector_send.return_value = tf.constant([0, 1, 0], dtype=tf.int32)
        unique_keys.return_value = tf.constant([[0, 1, 0], [0, 1, 0]], dtype=tf.int32)
        mock_config_init = MockConfigInitializer(use_static=True, var=test_table)
        base_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        opt = CustomizedOptimizer()
        grad = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=tf.float32)
        res = opt.sum_same_id_gradients(grad, test_table.variable, is_expansion=False)
        self.assertIsNotNone(res)
