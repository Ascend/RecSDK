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
import struct
import unittest
from collections import namedtuple
from unittest import mock

import tensorflow as tf

from mx_rec.saver.saver import Saver, SSD_SAVE_PATH_PREFIX, read_base_delta_and_write_for_ssd
from mx_rec.constants.constants import ASCEND_GLOBAL_HASHTABLE_COLLECTION
from tests.mx_rec.core.mock_class import MockConfigInitializer, MockSparseEmbedConfig
from tests.mx_rec.saver.sparse_embedding_mock import SparseEmbeddingMock

table_instance = SparseEmbeddingMock()


@mock.patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=mock.Mock(return_value=MockConfigInitializer()),
)
class TestSaver(unittest.TestCase):
    """
    Test the function of saving and loading sparse tables.
    """

    @mock.patch.multiple("mx_rec.saver.saver",
                         get_rank_id=mock.MagicMock(return_value=0),
                         get_rank_size=mock.MagicMock(return_value=1),
                         get_local_rank_size=mock.MagicMock(return_value=1))
    @mock.patch("mx_rec.saver.saver.ConfigInitializer")
    def test_save_and_load_is_consistent(self, saver_config_initializer):
        mock_config_initializer = \
            MockConfigInitializer(var=table_instance, asc_manager=True,
                                  use_dynamic_expansion=False,
                                  host_data=[0, 1, 4, 6, 8],
                                  ascend_global_hashtable_collection=ASCEND_GLOBAL_HASHTABLE_COLLECTION)
        saver_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        self.table_name = "test_table"
        self.optim_m_name = "test_table/LazyAdam/m"
        self.optim_v_name = "test_table/LazyAdam/v"
        self.graph = self.build_graph()

        with self.graph.as_default():
            self.saver = Saver()

        with tf.compat.v1.Session(graph=self.graph) as sess:
            embedding_directory = "./sparse-model-1/test_table/embedding"
            data_file = os.path.join(embedding_directory, "slice.data")
            attribute_file = os.path.join(embedding_directory, "slice.attribute")
            sess.run(tf.global_variables_initializer())
            origin_embedding = sess.run(self.var)[[0, 1, 4, 6, 8], :]

            self.saver.save(sess, save_path="model-1")
            self.assertTrue(os.path.exists(embedding_directory), "embedding目录已创建")
            self.assertTrue(os.path.exists(data_file), "embedding的data文件存储成功")
            self.assertTrue(os.path.exists(attribute_file), "embedding的attribute文件存储成功")

            tf.io.gfile.rmtree("./sparse-model-1")

    def build_graph(self):
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            self.shape = tf.TensorShape([10, 4])
            emb_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05, seed=128)
            initialized_tensor = emb_initializer(self.shape)
            self.var = tf.compat.v1.get_variable(self.table_name, trainable=False, initializer=initialized_tensor)

            optim_m_tensor = emb_initializer(self.shape)
            self.optimizer_m = tf.compat.v1.get_variable(self.optim_m_name, trainable=False, initializer=optim_m_tensor)
            optim_v_tensor = emb_initializer(self.shape)
            self.optimizer_v = tf.compat.v1.get_variable(self.optim_v_name, trainable=False, initializer=optim_v_tensor)

            tf.compat.v1.add_to_collection(ASCEND_GLOBAL_HASHTABLE_COLLECTION, self.var)
        return self.graph


class TestReadSSDModel(unittest.TestCase):
    """
    Test read base model and delta models for SSD.
    """
    def create_ssd_model_file(self, params):
        # Create 0th step model as base model.
        table_name_meta_file = os.path.join(SSD_SAVE_PATH_PREFIX + str(params.rank_id), params.table_name,
                                            params.table_name + ".meta." + params.base_model)
        tf.io.gfile.makedirs(os.path.join(SSD_SAVE_PATH_PREFIX + str(params.rank_id), params.table_name))
        with tf.io.gfile.GFile(table_name_meta_file, "wb") as file:
            file.write(struct.pack("I", len(params.table_name)))
            file.write(params.table_name.encode("utf-8"))
            file.write(struct.pack("Q", params.file_cnt))
            file.write(struct.pack("Q", params.file_id))

        table_meta_file = os.path.join(SSD_SAVE_PATH_PREFIX + str(params.rank_id), params.table_name,
                                       str(params.file_id) + ".meta." + params.base_model)
        table_data_file = os.path.join(SSD_SAVE_PATH_PREFIX + str(params.rank_id), params.table_name,
                                       str(params.file_id) + ".data." + params.base_model)
        with tf.io.gfile.GFile(table_meta_file, "wb") as f1, tf.io.gfile.GFile(table_data_file, "wb") as f2:
            f1.write("")
            f2.write("")

        # Create 1th step model as delta model.
        for delta in params.delta_models:
            table_name_meta_file = os.path.join(SSD_SAVE_PATH_PREFIX + str(params.rank_id), params.table_name,
                                                params.table_name + ".meta." + delta)
            tf.io.gfile.makedirs(os.path.join(SSD_SAVE_PATH_PREFIX + str(params.rank_id), params.table_name))
            with tf.io.gfile.GFile(table_name_meta_file, "wb") as file:
                file.write(struct.pack("I", len(params.table_name)))
                file.write(params.table_name.encode("utf-8"))
                file.write(struct.pack("Q", params.file_cnt))
                file.write(struct.pack("Q", params.file_id))

            meta_file = os.path.join(SSD_SAVE_PATH_PREFIX + str(params.rank_id), params.table_name,
                                     "delta-" + str(params.file_id) + ".meta." + delta)
            data_file = os.path.join(SSD_SAVE_PATH_PREFIX + str(params.rank_id), params.table_name,
                                     "delta-" + str(params.file_id) + ".data." + delta)
            with tf.io.gfile.GFile(meta_file, "wb") as file:
                for key, offset in zip(params.keys, params.offsets):
                    file.write(struct.pack("Q", key))
                    file.write(struct.pack("I", offset))
            with tf.io.gfile.GFile(data_file, "wb") as file:
                file.write(struct.pack("Q", params.emb_size))
                for emb in params.embedding:
                    file.write(struct.pack("f", emb))

    @mock.patch("mx_rec.saver.saver.ConfigInitializer")
    def test_read_base_delta_and_write_for_ssd(self, saver_config_initializer):
        params_for_create_ssd_data = namedtuple("params_for_create_ssd_data", ["rank_id", "table_name", "file_cnt",
                                                                               "file_id", "base_model", "delta_models",
                                                                               "keys", "offsets", "emb_size",
                                                                               "embedding"])
        params = params_for_create_ssd_data(rank_id=0, table_name="test_table", file_cnt=1, file_id=0, base_model="0",
                                            delta_models=["1"], keys=[1], offsets=[1], emb_size=8, embedding=[0.1] * 8)
        self.create_ssd_model_file(params)
        expected_meta_file = os.path.join(SSD_SAVE_PATH_PREFIX + str(params.rank_id), params.table_name,
                                          str(params.file_id) + ".meta." + params.delta_models[-1])
        expected_data_file = os.path.join(SSD_SAVE_PATH_PREFIX + str(params.rank_id), params.table_name,
                                          str(params.file_id) + ".data." + params.delta_models[-1])
        mock_config_initializer = MockConfigInitializer()
        mock_config_initializer.sparse_embed_config = MockSparseEmbedConfig(table_name_set=[params.table_name])
        saver_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
        read_base_delta_and_write_for_ssd("./tmp", params.base_model, params.delta_models, params.rank_id)
        self.assertTrue(tf.io.gfile.exists(expected_meta_file), "New meta file created.")
        self.assertTrue(tf.io.gfile.exists(expected_data_file), "New data file created.")
        tf.io.gfile.rmtree(SSD_SAVE_PATH_PREFIX + str(params.rank_id))


if __name__ == '__main__':
    unittest.main()
