#!/usr/bin/env python3
# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

import os
import unittest
from unittest import mock

import tensorflow as tf

from mx_rec.saver.saver import Saver
from mx_rec.constants.constants import ASCEND_GLOBAL_HASHTABLE_COLLECTION
from tests.mx_rec.saver.sparse_embedding_mock import SparseEmbeddingMock

table_instance = SparseEmbeddingMock()


class TestSaver(unittest.TestCase):
    """
    Test the function of saving and loading sparse tables.
    """

    @mock.patch.multiple("mx_rec.saver.saver",
                         get_rank_id=mock.MagicMock(return_value=0),
                         get_local_rank_size=mock.MagicMock(return_value=1),
                         get_ascend_global_hashtable_collection=mock.MagicMock(
                             return_value=ASCEND_GLOBAL_HASHTABLE_COLLECTION),
                         get_table_instance=mock.MagicMock(return_value=table_instance))
    def setUp(self):
        self.table_name = "test_table"
        self.optim_m_name = "test_table/LazyAdam/m"
        self.optim_v_name = "test_table/LazyAdam/v"
        self.graph = self.build_graph()

        with self.graph.as_default():
            self.saver = Saver()

    @mock.patch.multiple("mx_rec.saver.saver",
                         set_sparse_dir=mock.MagicMock(),
                         is_asc_manager_initialized=mock.MagicMock(return_value=True),
                         save_host_data=mock.MagicMock(),
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         get_table_instance_by_name=mock.MagicMock(return_value=table_instance),
                         get_host_data=mock.MagicMock(return_value=[0, 1, 4, 6, 8]),
                         restore_host_data=mock.MagicMock())
    def test_save_and_load_is_consistent(self):
        with tf.compat.v1.Session(graph=self.graph) as sess:
            embedding_directory = "./sparse-model/HashTable/HBM/test_table/embedding"
            data_file = os.path.join(embedding_directory, "slice_0.data")
            attribute_file = os.path.join(embedding_directory, "slice_0.attribute")
            sess.run(tf.global_variables_initializer())
            origin_embedding = sess.run(self.var)[[0, 1, 4, 6, 8], :]

            self.saver.save(sess)
            self.assertTrue(os.path.exists(embedding_directory), "embedding目录已创建")
            self.assertTrue(os.path.exists(data_file), "embedding的data文件存储成功")
            self.assertTrue(os.path.exists(attribute_file), "embedding的attribute文件存储成功")

            self.saver.restore(sess, "./model")
            load_embedding = sess.run(self.var)[:5, :]
            self.assertEqual(load_embedding.all(), origin_embedding.all())

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

            table_instance.set_optimizer("LazyAdam", {"momentum": self.optimizer_m, "velocity": self.optimizer_v})
            tf.compat.v1.add_to_collection(ASCEND_GLOBAL_HASHTABLE_COLLECTION, self.var)
        return self.graph


if __name__ == '__main__':
    unittest.main()
