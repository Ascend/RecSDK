#!/usr/bin/env python3
# coding: UTF-8
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

import unittest
from unittest import mock

import tensorflow as tf
from tensorflow.python.framework import ops

from mx_rec.optimizers.lazy_adam_by_addr import create_hash_optimizer_by_address
from core.mock_class import MockConfigInitializer, MockSparseEmbedding


class TestCreateHashOptimizerFunc(unittest.TestCase):
    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_use_dynamic_expansion_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(learning_rate=0.01)

        self.assertIn("The dynamic expansion mode is not compatible with the optimizer", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_lr_type_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(learning_rate="xxx")

        self.assertIn("is not float", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_lr_max_value_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(learning_rate=10.1)

        self.assertIn("is bigger than", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_lr_min_value_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(learning_rate=-0.01)

        self.assertIn("is less than", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_beta1_type_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(beta1="xxx")

        self.assertIn("is not float", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_beta1_max_value_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(beta1=10.1)

        self.assertIn("is bigger than", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_beta1_min_value_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(beta1=-0.01)

        self.assertIn("is less than", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_beta2_type_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(beta2="xxx")

        self.assertIn("is not float", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_beta2_max_value_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(beta2=10.1)

        self.assertIn("is bigger than", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_beta2_min_value_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(beta2=-0.01)

        self.assertIn("is less than", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_epsilon_type_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(epsilon="xxx")

        self.assertIn("is not float", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_epsilon_max_value_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(epsilon=10.1)

        self.assertIn("is bigger than", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_epsilon_min_value_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(epsilon=-0.01)

        self.assertIn("is less than", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_name_type_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(learning_rate=0.01, name=1)

        self.assertIn("is not str", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_name_min_len_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(learning_rate=0.01, name="")

        self.assertIn("length is less than", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_name_max_len_err(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        name = "a" * 201
        with self.assertRaises(ValueError) as e:
            create_hash_optimizer_by_address(learning_rate=0.01, name=name)

        self.assertIn("length is bigger than", str(e.exception))

    @mock.patch("mx_rec.optimizers.lazy_adam_by_addr.ConfigInitializer")
    def test_get_slot_init_values(self, adam_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
        adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
        sparse_optimizer = create_hash_optimizer_by_address(learning_rate=0.01)
        self.assertEqual(sparse_optimizer.get_slot_init_values(), [0.0, 0.0])
