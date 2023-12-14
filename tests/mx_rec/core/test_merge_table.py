#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import unittest
from unittest import mock

import tensorflow as tf

import mx_rec.core.asc.merge_table
from tests.mx_rec.core.mock_class import MockSparseEmbedding


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

    def setUp(self):
        # 在tensorflow默认图中添加op
        a = tf.constant(1)
        b = tf.constant(2)
        c = tf.add(a, b)
        with tf.Session() as sess:
            sess.run(c)

    def tearDown(self):
        # 删除tensorflow默认图中添加的op
        tf.reset_default_graph()

    @mock.patch.multiple("mx_rec.core.asc.merge_table",
                         get_bool_gauge_set=mock.MagicMock(return_value=[]))
    def test_is_train_task_case1(self):
        """
        case1: bool_gauge_set为[]
        """

        from mx_rec.core.asc.merge_table import is_train_task

        self.assertFalse(is_train_task())

    @mock.patch.multiple("mx_rec.core.asc.merge_table",
                         get_bool_gauge_set=mock.MagicMock(return_value=["train"]),
                         check_op=mock.MagicMock(return_value=True))
    def test_is_train_task_case2(self):
        """
        case2: bool_gauge_set为["train"]，且check_op为True
        """

        from mx_rec.core.asc.merge_table import is_train_task

        self.assertTrue(is_train_task())

    @mock.patch.multiple("mx_rec.core.asc.merge_table",
                         get_bool_gauge_set=mock.MagicMock(return_value=["train"]),
                         check_op=mock.MagicMock(return_value=False))
    def test_is_train_task_case3(self):
        """
        case3: bool_gauge_set为["train"]，且check_op为False
        """

        from mx_rec.core.asc.merge_table import is_train_task

        self.assertTrue(is_train_task())


class TestFindDanglingTableFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.merge_table.find_dangling_table'.
    """

    def setUp(self):
        # 在tensorflow默认图中添加op
        a = tf.constant(1)
        b = tf.constant(2)
        c = tf.add(a, b)
        with tf.Session() as sess:
            sess.run(c)

    def tearDown(self):
        # 删除tensorflow默认图中添加的op
        tf.reset_default_graph()

    @mock.patch.multiple("mx_rec.core.asc.merge_table",
                         is_train_task=mock.MagicMock(return_value=False))
    def test_find_dangling_table_case1(self):
        """
        case1: is_train_task为False
        """

        from mx_rec.core.asc.merge_table import find_dangling_table

        self.assertListEqual(find_dangling_table([]), [])

    @mock.patch.multiple("mx_rec.core.asc.merge_table",
                         is_train_task=mock.MagicMock(return_value=True),
                         get_enable_table_merge=mock.MagicMock(return_value=False))
    def test_find_dangling_table_case2(self):
        """
        case2: is_train_task为True，merge为False
        """

        from mx_rec.core.asc.merge_table import find_dangling_table

        self.assertListEqual(find_dangling_table([]), [])

    @mock.patch.multiple("mx_rec.core.asc.merge_table",
                         is_train_task=mock.MagicMock(return_value=True),
                         affirm=mock.MagicMock(return_value=True),
                         get_enable_table_merge=mock.MagicMock(return_value=True),
                         insert_dangling_table=mock.MagicMock(return_value=None))
    @mock.patch("mx_rec.core.asc.merge_table.export_table_instances")
    def test_find_dangling_table_case3(self, mock_export_table_instances):
        """
        case3: is_train_task为True，merge为True
        """

        from mx_rec.core.asc.merge_table import find_dangling_table

        mock_export_table_instances.return_value = {"table1": MockSparseEmbedding("table1")}
        dangling_table = find_dangling_table(["table2"])
        self.assertListEqual(dangling_table, ["table2", "table1"])


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


if __name__ == '__main__':
    unittest.main()
