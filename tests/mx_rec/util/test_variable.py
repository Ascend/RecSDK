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


class MockTableInstance:
    def __init__(self):
        self.skip_emb_transfer = False
        self.optimizer = False


class VariableTest(unittest.TestCase):
    def setUp(self):
        """
        准备步骤
        :return:无
        """
        self.cm_worker_size = global_env.cm_worker_size
        self.cm_chief_device = global_env.cm_chief_device
        self.ascend_visible_devices = global_env.ascend_visible_devices
        global_env.cm_worker_size = "8"
        global_env.cm_chief_device = "0"
        global_env.ascend_visible_devices = "0-7"

    def tearDown(self):
        """
        销毁步骤
        :return: 无
        """
        global_env.cm_worker_size = self.cm_worker_size
        global_env.cm_chief_device = self.cm_chief_device
        global_env.ascend_visible_devices = self.ascend_visible_devices

    @patch.multiple("mx_rec.util.variable",
                    get_ascend_global_hashtable_collection=mock.MagicMock(return_value="sparse_hastable"))
    def test_get_dense_and_sparse_variable(self):
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

    @patch.multiple("mx_rec.util.variable",
                    get_table_instance=mock.MagicMock(return_value=MockTableInstance()))
    def test_check_and_get_config_via_var_when_environment_error(self):
        with self.assertRaises(EnvironmentError):
            self.assertEqual(MockTableInstance(), check_and_get_config_via_var("1", "optimize"))

    def test_check_and_get_config_via_var_when_success(self):
        table_instance = MockTableInstance()
        table_instance.skip_emb_transfer = True
        table_instance.optimizer = True
        with patch("mx_rec.util.variable.get_table_instance") as mock_get_table_instance:
            mock_get_table_instance.return_value = mock.MagicMock(table_instance)
            self.assertEqual(mock_get_table_instance.return_value, check_and_get_config_via_var("1", "optimize"))

if __name__ == '__main__':
    unittest.main()
