#!/usr/bin/env python3
# coding: UTF-8
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

import os
from unittest import mock
import unittest

import tensorflow as tf
import numpy as np

from mx_rec.saver.saver import write_binary_data, generate_file_name
from tests.mx_rec.core.mock_class import MockConfigInitializer
from tests.mx_rec.saver.sparse_embedding_mock import SparseEmbeddingMock
from mx_rec.saver.sparse import export, set_upper_dir, check_table_param
from mx_rec.constants.constants import DataAttr


class TestSparseProcessor(unittest.TestCase):
    """
    Test the function of exporting sparse tables.
    """

    def setUp(self):
        self.table_name = "test_table"
        self.device_dir_list = ["HashTable", "HBM"]
        self.host_dir_list = ["HashTable", "DDR"]
        self.fake_hbm_sparse_dir = "./test_export_hbm/sparse-model"
        self.fake_ddr_sparse_dir = "./test_export_ddr/sparse-model"
        self.hbm_npy_path = None
        self.ddr_npy_path = None

    def test_check_table_param(self):
        table_list = ["test_table_1", "test_table_0"]
        default_table_list = ["test_table_1", "test_table_2", "test_table_3"]
        expect_table_list = ["test_table_1"]
        result_table_list = check_table_param(table_list, default_table_list)
        self.assertEqual(result_table_list, expect_table_list)

    def build_fake_hbm_save(self):
        table_dir = os.path.join(set_upper_dir(self.fake_hbm_sparse_dir, self.device_dir_list), self.table_name)
        fake_key = np.array([1, 2, 3, 4, 5])
        fake_emb = np.random.rand(5, 4).astype(np.float32)
        # build HBM fake file
        self.write_device_data(fake_emb, table_dir)
        attribute = np.array([5, 1, 4])
        self.write_host_data(fake_key, attribute, "key", table_dir)

        self.hbm_npy_path = os.path.join(table_dir, "key-emb.npy")

    def build_fake_ddr_save(self):
        table_dir = os.path.join(set_upper_dir(self.fake_ddr_sparse_dir, self.host_dir_list), self.table_name)
        fake_key_offset_map = np.array([1, 0, 2, 6, 3, 2, 4, 9, 5, 4])
        key_offset_attribute = np.array([5, 2, 4])
        fake_embedding = np.random.rand(10, 4).astype(np.float32)
        embedding_attribute = np.array([10, 4, 4])
        self.write_host_data(fake_key_offset_map, key_offset_attribute, "embedding_hashmap", table_dir)
        self.write_host_data(fake_embedding, embedding_attribute, "embedding_data", table_dir)

        device_table_dir = os.path.join(set_upper_dir(self.fake_ddr_sparse_dir, self.device_dir_list), self.table_name)
        fake_device_emb = np.random.rand(5, 4).astype(np.float32)
        self.write_device_data(fake_device_emb, device_table_dir)

        self.ddr_npy_path = os.path.join(table_dir, "key-emb.npy")

    def write_device_data(self, embedding, table_dir):
        attribute = dict()
        attribute[DataAttr.DATATYPE.value] = embedding.dtype.name
        attribute[DataAttr.SHAPE.value] = embedding.shape

        embedding_dir = os.path.join(table_dir, "embedding")
        write_binary_data(embedding_dir, 0, embedding, attributes=attribute)

    def write_host_data(self, data, attribute, data_type, table_dir):
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
