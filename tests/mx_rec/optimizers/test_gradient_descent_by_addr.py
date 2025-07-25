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
from tensorflow.python.framework import ops

from mx_rec.optimizers.gradient_descent_by_addr import create_hash_optimizer_by_addr
from tests.mx_rec.core.mock_class import MockConfigInitializer, MockSparseEmbedding


class TestCreateHashOptimizerFunc(unittest.TestCase):
    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_use_dynamic_expansion_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate=0.01)

        self.assertIn("The dynamic expansion mode is not compatible with the optimizer", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_lr_type_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate="xxx")

        self.assertIn("is not float", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_lr_max_value_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate=10.1)

        self.assertIn("is bigger than", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_lr_min_value_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate=-0.01)

        self.assertIn("is less than", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_weight_decay_type_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate=0.1, weight_decay="xx")

        self.assertIn("is not float", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_weight_decay_max_value_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate=0.1, weight_decay=1.1)

        self.assertIn("is bigger than", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_weight_decay_min_value_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate=0.1, weight_decay=-0.1)

        self.assertIn("is less than", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_use_locking_type_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate=0.01, use_locking="xxx")

        self.assertIn("Invalid parameter type of para", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_name_type_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate=0.01, name=1)

        self.assertIn("is not str", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_name_min_len_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate=0.01, name="")

        self.assertIn("length is less than", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_name_max_len_err(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        name = "a" * 201
        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_addr(learning_rate=0.01, name=name)

        self.assertIn("length is bigger than", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_dense_grad_not_impl_err(self, sgd_config_initializer):
        with tf.compat.v1.Graph().as_default():
            dim = 8
            table = MockSparseEmbedding(
                table_name="test_table",
                embedding_size=dim,
                slice_device_vocabulary_size=2,
            )
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True, var=table)
            sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            sparse_lookup_res = table.variable

            # Model forward.
            dense_tensor = tf.compat.v1.get_variable(
                "dense_var", shape=(dim, 1), initializer=tf.constant_initializer(1), dtype=tf.float32
            )
            logits = tf.matmul(sparse_lookup_res, dense_tensor)
            logits = tf.reshape(logits, (-1,))

            labels = tf.constant([0, 1], dtype=tf.float32)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss)

            sparse_optimizer = create_hash_optimizer_by_addr(learning_rate=0.01)
            sparse_grads = tf.gradients(loss, [table.variable])
            grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, [table.variable])]

            with self.assertRaises(NotImplementedError) as e:
                sparse_optimizer.apply_gradients(grads_and_vars)

            self.assertIn("You are using a wrong type of variable", str(e.exception))

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_get_slot_init_values(self, sgd_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
        sparse_optimizer = create_hash_optimizer_by_addr(learning_rate=0.01)
        self.assertEqual(sparse_optimizer.get_slot_init_values(), [])

    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.import_host_pipeline_ops")
    @mock.patch("mx_rec.optimizers.base.ConfigInitializer")
    @mock.patch("mx_rec.optimizers.gradient_descent_by_addr.ConfigInitializer")
    def test_update_ok(self, sgd_config_initializer, base_config_initializer, mock_import_host_pipeline_ops):
        with tf.compat.v1.Graph().as_default():
            dim = 8
            table = MockSparseEmbedding(
                table_name="test_table",
                embedding_size=dim,
                slice_device_vocabulary_size=2,
            )
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True, var=table)
            sgd_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            base_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            mock_import_host_pipeline_ops_func = mock.MagicMock()
            mock_import_host_pipeline_ops_func.embedding_lookup_by_address.return_value = tf.constant(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
            )
            mock_import_host_pipeline_ops.return_value = mock_import_host_pipeline_ops_func
            sparse_lookup_res = table.variable

            # Model forward.
            dense_tensor = tf.compat.v1.get_variable(
                "dense_var", shape=(dim, 1), initializer=tf.constant_initializer(1), dtype=tf.float32
            )
            logits = tf.matmul(sparse_lookup_res, dense_tensor)
            logits = tf.reshape(logits, (-1,))

            labels = tf.constant([0, 1], dtype=tf.float32)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss)

            sparse_optimizer = create_hash_optimizer_by_addr(learning_rate=0.01)
            grads = tf.gradients(loss, [table.variable])
            sparse_grads = [
                ops.IndexedSlices(
                    values=tf.constant(
                        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
                    ),
                    indices=tf.constant([0, 1]),
                    dense_shape=tf.shape(grads[0]),
                )
            ]
            grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, [table.variable])]
            res = sparse_optimizer.apply_gradients(grads_and_vars)
            self.assertIsInstance(res, tf.Operation)
