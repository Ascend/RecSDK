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

import os
from unittest import mock
import unittest

import tensorflow as tf
import numpy as np

from mx_rec.saver.saver import generate_file_name
from mx_rec.saver.sparse import export, check_table_param, SparseProcessor
from tests.mx_rec.core.mock_class import MockConfigInitializer


class TestSparseProcessor(unittest.TestCase):
    """
    Test the function of exporting sparse tables.
    """

    @mock.patch("mx_rec.saver.sparse.ConfigInitializer")
    def test_init_with_empty_list(self, sparse_config_initializer):
        mock_config_init = MockConfigInitializer()
        sparse_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        SparseProcessor.set_instance(table_list=[])
        self.assertIsNotNone(SparseProcessor.single_instance)

    @mock.patch("mx_rec.saver.sparse.ConfigInitializer")
    def test_init_with_list(self, sparse_config_initializer):
        mock_config_init = MockConfigInitializer()
        sparse_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        SparseProcessor.set_instance(table_list=["test_table"])
        self.assertIsNotNone(SparseProcessor.single_instance)

    def test_check_table_param(self):
        table_list = ["test_table_1", "test_table_0"]
        default_table_list = ["test_table_1", "test_table_2", "test_table_3"]
        expect_table_list = ["test_table_1"]
        result_table_list = check_table_param(table_list, default_table_list)
        self.assertEqual(result_table_list, expect_table_list)

    @mock.patch("mx_rec.saver.sparse.ConfigInitializer")
    def test_export_with_empty_table_list(self, sparse_config_initializer):
        mock_config_init = MockConfigInitializer()
        sparse_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        res = export()
        self.assertEqual(res, 0)

    @mock.patch("mx_rec.saver.sparse.ConfigInitializer")
    def test_export_ok(self, sparse_config_initializer):
        sparse_dir = "./tmp_export_sparse_data"
        if tf.io.gfile.isdir(sparse_dir):
            tf.io.gfile.rmtree(sparse_dir)
        mock_config_init = MockConfigInitializer(sparse_dir="./tmp_export_sparse_data", table_name_set={"test_table"})
        sparse_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)

        fake_emb = np.random.rand(5, 4).astype(np.float32)
        attribute = np.array([5, 4])
        table_dir = os.path.join(sparse_dir, "test_table")
        _write_host_data(fake_emb, attribute, "embedding", table_dir)
        fake_key = np.array([1, 2, 3, 4, 5])
        _write_host_data(fake_key, attribute, "key", table_dir)

        res = export(table_list=["test_table"])
        self.assertNotEqual(res, 0)
        tf.io.gfile.rmtree(sparse_dir)


def _write_host_data(data, attribute, data_type, table_dir):
    data_dir = os.path.join(table_dir, data_type)
    tf.io.gfile.makedirs(data_dir)
    data_file, attribute_file = generate_file_name(0)
    target_data_dir = os.path.join(data_dir, data_file)
    target_attribute_dir = os.path.join(data_dir, attribute_file)

    with tf.io.gfile.GFile(target_data_dir, "wb") as file:
        data = data.tostring()
        file.write(data)

    with tf.io.gfile.GFile(target_attribute_dir, "wb") as file:
        attribute = attribute.tostring()
        file.write(attribute)
