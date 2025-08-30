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

import time
import unittest
from unittest import mock

import tensorflow as tf

from mx_rec.core.feature_process import EvictHook
from mx_rec.core.asc.feature_spec import FeatureSpec
from core.mock_class import MockSparseEmbedding, MockConfigInitializer


class TestEvictHookClass(unittest.TestCase):
    """
    Test for 'mx_rec.core.feature_process.EvictHook'.
    """

    def test_init_case1(self):
        """
        case1: evict_step_interval为None
        """
        self.assertIsInstance(EvictHook(), EvictHook)

    def test_init_case2(self):
        """
        case2: evict_step_interval不为None
        """
        self.assertIsInstance(EvictHook(evict_step_interval=5), EvictHook)


@mock.patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=mock.Mock(return_value=MockConfigInitializer()),
)
@mock.patch.multiple(
    "tensorflow.compat.v1.train.Saver",
    __init__=mock.Mock(return_value=None),
    build=mock.Mock(),
)
class TestAfterRunFuncOfEvictHookClass(TestEvictHookClass):
    """
    Test for 'mx_rec.core.feature_process.EvictHook.after_run'.
    """

    def setUp(self):
        self.ori_var_assert = [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]
        self.evict_var_assert = [[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]

    @mock.patch("mx_rec.core.feature_process.npu_ops.gen_npu_ops.get_next")
    @mock.patch("mx_rec.core.feature_process.ConfigInitializer")
    def test_after_run_case1(self, feature_process_config_initializer, mock_get_next):
        """
        case1: evict_enable为True，python和C++侧正常触发淘汰
        """

        with tf.Graph().as_default():
            test_table = MockSparseEmbedding()
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False, var=test_table)
            feature_process_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            mock_config_initializer.get_instance().feature_spec_config.insert_feature_spec(
                FeatureSpec("test_spec", table_name="test_table", access_threshold=5, eviction_threshold=10), True)

            mock_get_next.return_value = [tf.constant([8, 9], dtype=tf.int32), tf.constant(2, dtype=tf.int32)]

            evict_hook = EvictHook(evict_enable=True, evict_time_interval=10)
            with tf.compat.v1.train.MonitoredSession(hooks=[evict_hook]) as sess:
                sess.graph._unsafe_unfinalize()
                sess.run(tf.compat.v1.global_variables_initializer())

                # sleep 1s 等待淘汰时间evict_time_interval
                time.sleep(10)

                # 获取原variable，淘汰会发生在此session run之后
                ori_variable = sess.run(test_table.variable)
                # ori_variable的9、10行值都为1
                self.assertListEqual(ori_variable[8:].tolist(), self.ori_var_assert)

                # 获取淘汰后variable
                evict_variable = sess.run(test_table.variable)
                # evict_variable的9、10行触发淘汰后都为0，其余行数的值不变
                self.assertListEqual(evict_variable[8:].tolist(), self.evict_var_assert)
                self.assertListEqual(evict_variable[:2].tolist(), self.ori_var_assert)

    @mock.patch("mx_rec.core.feature_process.npu_ops.gen_npu_ops.get_next")
    @mock.patch("mx_rec.core.feature_process.ConfigInitializer")
    def test_after_run_case2(self, feature_process_config_initializer, mock_get_next):
        """
        case2: evict_enable为True，C++侧异常
        """

        with tf.Graph().as_default():
            test_table = MockSparseEmbedding()
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False, trigger_evict=False,
                                                            var=test_table)
            feature_process_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            mock_config_initializer.get_instance().feature_spec_config.insert_feature_spec(
                FeatureSpec("test_spec", table_name="test_table", access_threshold=5, eviction_threshold=10), True)

            mock_get_next.return_value = [tf.constant([8, 9], dtype=tf.int32), tf.constant(2, dtype=tf.int32)]

            evict_hook = EvictHook(evict_enable=True, evict_time_interval=1)
            with tf.compat.v1.train.MonitoredSession(hooks=[evict_hook]) as sess:
                sess.graph._unsafe_unfinalize()
                sess.run(tf.compat.v1.global_variables_initializer())

                # sleep 1s 等待淘汰时间evict_time_interval
                time.sleep(1)

                # 获取原variable，淘汰会发生在此session run之后
                ori_variable = sess.run(test_table.variable)
                # ori_variable的9、10行值都为1
                self.assertListEqual(ori_variable[8:].tolist(), self.ori_var_assert)

                # 获取淘汰后variable
                evict_variable = sess.run(test_table.variable)
                # 此时C++侧异常，不执行淘汰，因此evict_variable后两行还是1
                self.assertListEqual(evict_variable[8:].tolist(), self.ori_var_assert)

    @mock.patch("mx_rec.core.feature_process.npu_ops.gen_npu_ops.get_next")
    @mock.patch("mx_rec.core.feature_process.ConfigInitializer")
    def test_after_run_case3(self, feature_process_config_initializer, mock_get_next):
        """
        case3: evict_enable为False
        """

        with tf.Graph().as_default():
            test_table = MockSparseEmbedding()
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False, var=test_table)
            feature_process_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            mock_config_initializer.get_instance().feature_spec_config.insert_feature_spec(
                FeatureSpec("test_spec", table_name="test_table", access_threshold=5, eviction_threshold=10), True)

            mock_get_next.return_value = [tf.constant([8, 9], dtype=tf.int32), tf.constant(2, dtype=tf.int32)]

            evict_hook = EvictHook(evict_enable=False, evict_time_interval=1)
            with tf.compat.v1.train.MonitoredSession(hooks=[evict_hook]) as sess:
                sess.graph._unsafe_unfinalize()
                sess.run(tf.compat.v1.global_variables_initializer())

                # sleep 1s 等待淘汰时间evict_time_interval
                time.sleep(1)

                # 获取原variable，淘汰会发生在此session run之后
                ori_variable = sess.run(test_table.variable)
                # ori_variable的9、10行值都为1
                self.assertListEqual(ori_variable[8:].tolist(), self.ori_var_assert)

                # 获取淘汰后variable
                evict_variable = sess.run(test_table.variable)
                # 此时evict_enable为False，不执行淘汰，因此evict_variable后两行还是1
                self.assertListEqual(evict_variable[8:].tolist(), self.ori_var_assert)


if __name__ == '__main__':
    unittest.main()
