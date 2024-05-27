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
import unittest
from unittest import mock

import tensorflow as tf

from mx_rec.saver.saver import Saver
from mx_rec.constants.constants import ASCEND_GLOBAL_HASHTABLE_COLLECTION
from mx_rec.util.initialize import ConfigInitializer
from tests.mx_rec.core.mock_class import MockConfigInitializer
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
            embedding_directory = "./sparse-model/test_table/embedding"
            data_file = os.path.join(embedding_directory, "slice.data")
            attribute_file = os.path.join(embedding_directory, "slice.attribute")
            sess.run(tf.global_variables_initializer())
            origin_embedding = sess.run(self.var)[[0, 1, 4, 6, 8], :]

            self.saver.save(sess)
            self.assertTrue(os.path.exists(embedding_directory), "embedding目录已创建")
            self.assertTrue(os.path.exists(data_file), "embedding的data文件存储成功")
            self.assertTrue(os.path.exists(attribute_file), "embedding的attribute文件存储成功")

            tf.io.gfile.rmtree("./sparse-model")

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


if __name__ == '__main__':
    unittest.main()
