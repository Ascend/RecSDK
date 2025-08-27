#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import unittest
from unittest import mock

import tensorflow as tf

import mx_rec.core.asc.merge_table
from mx_rec.core.asc.merge_table import find_dangling_table, check_dangling_table
from mx_rec.util.global_env_conf import global_env
from core.mock_class import MockSparseEmbedding, MockConfigInitializer, MockGlobalEnv


class TestAffirmFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.merge_table.affirm'.
    """

    def test_affirm_case1(self):
        """
        case1: reach_op为[]
        """

        from mx_rec.core.asc.merge_table import affirm

        self.assertTrue(affirm([]))

    def test_affirm_case2(self):
        """
        case2: reach_op中的算子没有("IdentityN", "Reshape", "Identity")中的类型
        """

        from mx_rec.core.asc.merge_table import affirm

        with tf.Graph().as_default():
            self.assertFalse(affirm([tf.no_op()]))


class TestCheckOpFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.merge_table.check_op'.
    """

    def test_check_op_case1(self):
        """
        case1: table_reachable_op的type为ApplyAdam
        """

        from mx_rec.core.asc.merge_table import check_op

        with tf.Graph().as_default():
            x = tf.Variable(0.0)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            apply_op = optimizer.apply_gradients([(tf.constant(1.0), x)])
            self.assertTrue(check_op(apply_op.control_inputs[0]))

    def test_check_op_case2(self):
        """
        case2: table_reachable_op是tf.no_op()
        """

        from mx_rec.core.asc.merge_table import check_op

        with tf.Graph().as_default():
            self.assertFalse(check_op(tf.no_op()))


class TestIsTrainTaskFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.merge_table.is_train_task'.
    """

    @mock.patch("mx_rec.graph.patch.ConfigInitializer")
    def setUp(self, graph_patch_config_initializer):
        mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)
        graph_patch_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        # 在tensorflow默认图中添加op
        a = tf.constant(1)
        b = tf.constant(2)
        c = tf.add(a, b)
        with tf.Session() as sess:
            sess.run(c)

    def tearDown(self):
        # 删除tensorflow默认图中添加的op
        tf.reset_default_graph()

    @mock.patch("mx_rec.core.asc.merge_table.ConfigInitializer")
    @mock.patch("mx_rec.graph.patch.ConfigInitializer")
    def test_is_train_task_case1(self, graph_patch_config_initializer, merge_table_config_initializer):
        """
        case1: bool_gauge_set为[]
        """

        from mx_rec.core.asc.merge_table import is_train_task

        mock_config_initializer = MockConfigInitializer(bool_gauge_set=[])
        graph_patch_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
        merge_table_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        self.assertFalse(is_train_task())

    @mock.patch.multiple("mx_rec.core.asc.merge_table", check_op=mock.MagicMock(return_value=True))
    @mock.patch("mx_rec.core.asc.merge_table.ConfigInitializer")
    @mock.patch("mx_rec.graph.patch.ConfigInitializer")
    def test_is_train_task_case2(self, graph_patch_config_initializer, merge_table_config_initializer):
        """
        case2: bool_gauge_set为["train"]，且check_op为True
        """

        from mx_rec.core.asc.merge_table import is_train_task

        mock_config_initializer = MockConfigInitializer(bool_gauge_set=["train"])
        graph_patch_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
        merge_table_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        self.assertTrue(is_train_task())


class TestFindDanglingTableFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.merge_table.find_dangling_table'.
    """

    @mock.patch("mx_rec.graph.patch.ConfigInitializer")
    def setUp(self, graph_patch_config_initializer):
        tf.compat.v1.reset_default_graph()
        mock_config_initializer = MockConfigInitializer()
        graph_patch_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        # 在tensorflow默认图中添加op
        a = tf.constant(1)
        b = tf.constant(2)
        c = tf.add(a, b)
        identity_n = tf.identity_n([c], name="test_table_identity_n")
        with tf.compat.v1.Session() as sess:
            sess.run(identity_n)

    def tearDown(self):
        # 删除tensorflow默认图中添加的op
        tf.compat.v1.reset_default_graph()

    @mock.patch.multiple("mx_rec.core.asc.merge_table", is_train_task=mock.MagicMock(return_value=False))
    def test_find_dangling_table_case1(self):
        """
        case1: is_train_task为False
        """

        self.assertListEqual(find_dangling_table([]), [])

    @mock.patch.multiple(
        "mx_rec.core.asc.merge_table",
        is_train_task=mock.MagicMock(return_value=True),
        global_env=mock.MagicMock(return_value=MockGlobalEnv(tf_device="GPU")),
    )
    def test_find_dangling_table_case2(self):
        """
        case2: is_train_task为True，merge为False
        """

        self.assertListEqual(find_dangling_table([]), [])

    @mock.patch.multiple(
        "mx_rec.core.asc.merge_table",
        is_train_task=mock.MagicMock(return_value=True),
    )
    @mock.patch.object(global_env, "tf_device", new="NPU")
    @mock.patch("mx_rec.core.asc.merge_table.ConfigInitializer")
    def test_merge_is_true(self, merge_table_config_init):
        test_table = MockSparseEmbedding(table_name="test_table")
        mock_config_init = MockConfigInitializer(table_instance_dict={"test_table": test_table})
        merge_table_config_init.get_instance = mock.Mock(return_value=mock_config_init)
        self.assertListEqual(find_dangling_table(["test_table"]), ["test_table"])


class TestShouldSkipFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.merge_table.should_skip'.
    """

    def tearDown(self):
        mx_rec.core.asc.merge_table.ASCEND_TABLE_NAME_MUST_CONTAIN = None

    def test_should_skip_case1(self):
        """
        case1: table_name不包含"merged"，ASCEND_TABLE_NAME_MUST_CONTAIN为str
        """

        from mx_rec.core.asc.merge_table import should_skip

        mx_rec.core.asc.merge_table.ASCEND_TABLE_NAME_MUST_CONTAIN = "merged"
        self.assertTrue(should_skip("test_table"))

    def test_should_skip_case2(self):
        """
        case2: table_name包含"merged"，ASCEND_TABLE_NAME_MUST_CONTAIN为str
        """

        from mx_rec.core.asc.merge_table import should_skip

        mx_rec.core.asc.merge_table.ASCEND_TABLE_NAME_MUST_CONTAIN = "merged"
        self.assertFalse(should_skip("merged_table"))

    def test_should_skip_case3(self):
        """
        case3: table_name包含"merged"，ASCEND_TABLE_NAME_MUST_CONTAIN为list
        """

        from mx_rec.core.asc.merge_table import should_skip

        mx_rec.core.asc.merge_table.ASCEND_TABLE_NAME_MUST_CONTAIN = ["merged"]
        self.assertFalse(should_skip("merged_table"))


class TestCheckDanglingTable(unittest.TestCase):
    def setUp(self):
        tf.compat.v1.reset_default_graph()

    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    @mock.patch("mx_rec.core.asc.merge_table.ConfigInitializer")
    def test_dangling_table_empty(self, merge_table_config_init):
        test_table = MockSparseEmbedding(table_name="test_table")
        mock_config_init = MockConfigInitializer(table_instance_dict={"test_table": test_table})
        merge_table_config_init.get_instance = mock.Mock(return_value=mock_config_init)
        self.assertListEqual(check_dangling_table(), [])

    @mock.patch("mx_rec.core.asc.merge_table.ConfigInitializer")
    def test_dangling_table_ok(self, merge_table_config_init):
        test_table = MockSparseEmbedding(table_name="test_table")
        mock_config_init = MockConfigInitializer(
            dangling_table=["test_table"], table_instance_dict={"test_table": test_table}
        )
        merge_table_config_init.get_instance = mock.Mock(return_value=mock_config_init)
        self.assertNotEqual(check_dangling_table(), [])


if __name__ == "__main__":
    unittest.main()
