#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import unittest
from unittest import mock
from unittest.mock import patch

import tensorflow as tf
from mx_rec.util.global_env_conf import global_env
from mx_rec.util.variable import check_and_get_config_via_var
from mx_rec.util.variable import get_dense_and_sparse_variable
from tests.mx_rec.core.mock_class import MockConfigInitializer


class MockTableInstance:
    def __init__(self):
        self.is_hbm = False
        self.optimizer = False


@patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=mock.Mock(return_value=MockConfigInitializer()),
)
class VariableTest(unittest.TestCase):
    def setUp(self):
        """
        准备步骤
        :return:无
        """
        self.cm_worker_size = global_env.cm_worker_size
        self.cm_chief_device = global_env.cm_chief_device
        global_env.cm_worker_size = "8"
        global_env.cm_chief_device = "0"

    def tearDown(self):
        """
        销毁步骤
        :return: 无
        """
        global_env.cm_worker_size = self.cm_worker_size
        global_env.cm_chief_device = self.cm_chief_device

    @mock.patch("mx_rec.util.variable.ConfigInitializer")
    def test_get_dense_and_sparse_variable(self, variable_config_initializer):
        mock_config_initializer = MockConfigInitializer(ascend_global_hashtable_collection="sparse_hastable")
        variable_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        dense_layer = tf.Variable([1, 2], trainable=True)
        sparse_emb = tf.Variable([4, 5], trainable=False)
        tf.compat.v1.add_to_collection("sparse_hastable", sparse_emb)
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, dense_layer)
        dense_variables, sparse_variables = get_dense_and_sparse_variable()
        with tf.Session() as sess:
            result = tf.reduce_all(tf.equal(dense_layer, dense_variables))
            sess.run(tf.compat.v1.global_variables_initializer())
            result_run = sess.run([result])

        self.assertTrue(result_run)
        tf.reset_default_graph()

    @mock.patch("mx_rec.util.variable.ConfigInitializer")
    def test_check_and_get_config_via_var_when_environment_error(self, variable_config_initializer):
        mock_config_initializer = MockConfigInitializer(var=MockTableInstance())
        variable_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        with self.assertRaises(EnvironmentError):
            self.assertEqual(MockTableInstance(), check_and_get_config_via_var("1", "optimize"))


if __name__ == '__main__':
    unittest.main()
