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

from mx_rec.util.config_utils.embedding_utils import SparseEmbedConfig
from core.mock_class import MockConfigInitializer


class TestSparseEmbedConfig(unittest.TestCase):
    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    def test_init_ok(self):
        sparse_embed_config = SparseEmbedConfig()
        self.assertEqual(len(sparse_embed_config.table_instance_dict), 0)
        self.assertEqual(len(sparse_embed_config.dangling_table), 0)
        self.assertEqual(len(sparse_embed_config.table_name_set), 0)
        self.assertEqual(len(sparse_embed_config.name_to_var_dict), 0)
        self.assertEqual(len(sparse_embed_config.removing_var_list), 0)

    @mock.patch("mx_rec.util.initialize.ConfigInitializer")
    def test_get_table_instance_with_dynamic_ok(self, config_initializer):
        with tf.compat.v1.Graph().as_default():
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
            config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            sparse_embed_config = SparseEmbedConfig()
            test_tensor = tf.constant([0, 1], dtype=tf.float32)
            sparse_embed_config.insert_table_instance_to_tensor_dict(tensor=test_tensor, instance="xxx")
            self.assertEqual(sparse_embed_config.get_table_instance(key=test_tensor), "xxx")

    @mock.patch("mx_rec.util.initialize.ConfigInitializer")
    def test_get_table_instance_with_dynamic_err(self, config_initializer):
        with tf.compat.v1.Graph().as_default():
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
            config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            sparse_embed_config = SparseEmbedConfig()
            test_tensor = tf.constant([0, 1], dtype=tf.float32)
            sparse_embed_config.insert_table_instance_to_tensor_dict(tensor=test_tensor, instance="xxx")
            with self.assertRaises(KeyError):
                sparse_embed_config.get_table_instance(key=tf.constant([0], dtype=tf.float32))

    def test_insert_table_ins_to_tensor_err(self):
        with tf.compat.v1.Graph().as_default():
            sparse_embed_config = SparseEmbedConfig()
            test_tensor = tf.constant([0, 1], dtype=tf.float32)
            sparse_embed_config.insert_table_instance_to_tensor_dict(tensor=test_tensor, instance="xxx")
            with self.assertRaises(KeyError):
                sparse_embed_config.insert_table_instance_to_tensor_dict(tensor=test_tensor, instance="xxx")

    @mock.patch("mx_rec.util.initialize.ConfigInitializer")
    def test_get_table_instance_without_dynamic_ok(self, config_initializer):
        with tf.compat.v1.Graph().as_default():
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)
            config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            sparse_embed_config = SparseEmbedConfig()
            test_var = tf.compat.v1.get_variable(
                "test_var", shape=(1, 1), initializer=tf.constant_initializer(1), dtype=tf.float32
            )
            sparse_embed_config.insert_table_instance(name="test_name", key=test_var, instance="xxx", eval_flag=False)
            self.assertEqual(sparse_embed_config.get_table_instance(key=test_var), "xxx")

    @mock.patch("mx_rec.util.initialize.ConfigInitializer")
    def test_insert_table_instance_name_error(self, config_initializer):
        with tf.compat.v1.Graph().as_default():
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)
            config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            sparse_embed_config = SparseEmbedConfig()
            test_var = tf.compat.v1.get_variable(
                "test_var", shape=(1, 1), initializer=tf.constant_initializer(1), dtype=tf.float32
            )
            sparse_embed_config.insert_table_instance(name="test_name", key=test_var, instance="xxx", eval_flag=False)
            with self.assertRaises(ValueError):
                sparse_embed_config.insert_table_instance(name="test_name", key="xxx", instance="xxx", eval_flag=False)

    @mock.patch("mx_rec.util.initialize.ConfigInitializer")
    def test_insert_table_instance_key_error(self, config_initializer):
        with tf.compat.v1.Graph().as_default():
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)
            config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            sparse_embed_config = SparseEmbedConfig()
            test_var = tf.compat.v1.get_variable(
                "test_var", shape=(1, 1), initializer=tf.constant_initializer(1), dtype=tf.float32
            )
            sparse_embed_config.insert_table_instance(name="test_name", key=test_var, instance="xxx", eval_flag=False)
            with self.assertRaises(KeyError):
                sparse_embed_config.insert_table_instance(
                    name="test_name1", key=test_var, instance="xxx", eval_flag=False
                )

    @mock.patch("mx_rec.util.initialize.ConfigInitializer")
    def test_get_table_instance_eval_flag_ok(self, config_initializer):
        with tf.compat.v1.Graph().as_default():
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)
            config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            sparse_embed_config = SparseEmbedConfig()
            test_var = tf.compat.v1.get_variable(
                "test_var", shape=(1, 1), initializer=tf.constant_initializer(1), dtype=tf.float32
            )
            sparse_embed_config.insert_table_instance(name="test_name", key=test_var, instance="xxx", eval_flag=True)
            self.assertEqual(sparse_embed_config.get_table_instance(key=test_var), "xxx")

    def test_update_table_instance(self):
        with tf.compat.v1.Graph().as_default():
            sparse_embed_config = SparseEmbedConfig()
            test_var = tf.compat.v1.get_variable(
                "test_var", shape=(1, 1), initializer=tf.constant_initializer(1), dtype=tf.float32
            )
            sparse_embed_config.insert_table_instance(name="test_name", key=test_var, instance="xxx", eval_flag=False)
            update_var = tf.compat.v1.get_variable(
                "update_var", shape=(2, 1), initializer=tf.constant_initializer(1), dtype=tf.float32
            )
            sparse_embed_config.update_table_instance(
                table_name="update_var", emb_table="new_xxx", old_var=test_var, new_var=update_var
            )
            self.assertEqual(sparse_embed_config.table_instance_dict.get(update_var), "new_xxx")

    def test_export_table_num_not_empty(self):
        with tf.compat.v1.Graph().as_default():
            sparse_embed_config = SparseEmbedConfig()
            test_var = tf.compat.v1.get_variable(
                "test_var", shape=(1, 1), initializer=tf.constant_initializer(1), dtype=tf.float32
            )
            sparse_embed_config.insert_table_instance(name="test_name", key=test_var, instance="xxx", eval_flag=False)
            self.assertEqual(sparse_embed_config.export_table_num(), 1)

    def test_export_table_num_empty(self):
        sparse_embed_config = SparseEmbedConfig()
        self.assertEqual(sparse_embed_config.export_table_num(), 0)
