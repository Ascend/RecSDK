#!/usr/bin/env python3
# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

import unittest
from unittest import mock

import tensorflow as tf

from mx_rec.util.global_env_conf import global_env


class TestGetRestoreVectorFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_restore_vector'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)

    def test_get_restore_vector_case1(self):
        """
        case1: HBM，emb_size不为int，抛出异常

        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        self.config["emb_size"] = "xxx"
        with self.assertRaises(TypeError):
            get_restore_vector(self.config)

    def test_get_restore_vector_case2(self):
        """
        case2: HBM，emb_size小于1，抛出异常
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        self.config["emb_size"] = 0
        with self.assertRaises(ValueError):
            get_restore_vector(self.config)

    def test_get_restore_vector_case3(self):
        """
        case3: 非HBM，ext_emb_size不为int，抛出异常
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        self.config["skip_emb_transfer"] = False
        self.config["ext_emb_size"] = "xxx"
        with self.assertRaises(TypeError):
            get_restore_vector(self.config)

    def test_get_restore_vector_case4(self):
        """
        case4: 非HBM，ext_emb_size小于1，抛出异常
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        self.config["skip_emb_transfer"] = False
        self.config["ext_emb_size"] = 0
        with self.assertRaises(ValueError):
            get_restore_vector(self.config)

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=True))
    @mock.patch("mx_rec.core.asc.build_graph.mxrec_pybind.get_ub_hot_size")
    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_restore_vector_case5(self, mock_get_next, mock_get_ub_hot_size):
        """
        case5: HBM，静态shape，hot emb
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        with tf.Graph().as_default():
            mock_get_next.return_value = [0, 1]
            mock_get_ub_hot_size.return_value = 8
            restore_vector, hot_pos = get_restore_vector(self.config)
            self.assertEqual(restore_vector, 0)
            self.assertEqual(hot_pos, 1)

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=True))
    @mock.patch("mx_rec.core.asc.build_graph.mxrec_pybind.get_ub_hot_size")
    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_restore_vector_case6(self, mock_get_next, mock_get_ub_hot_size):
        """
        case6: HBM，动态shape，hot emb
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        with tf.Graph().as_default():
            mock_get_next.return_value = [0, 1]
            mock_get_ub_hot_size.return_value = 8
            restore_vector, hot_pos = get_restore_vector(self.config)
            self.assertEqual(restore_vector, 0)
            self.assertEqual(hot_pos, 1)

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=True))
    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_restore_vector_case7(self, mock_get_next):
        """
        case7: HBM，静态shape
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        with tf.Graph().as_default():
            mock_get_next.return_value = [0]
            self.config["use_hot"] = False
            restore_vector, hot_pos = get_restore_vector(self.config)
            self.assertEqual(restore_vector, 0)
            self.assertIsNone(hot_pos)

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=False))
    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_restore_vector_case8(self, mock_get_next):
        """
        case8: HBM，动态shape
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        with tf.Graph().as_default():
            mock_get_next.return_value = [0]
            self.config["use_hot"] = False
            restore_vector, hot_pos = get_restore_vector(self.config)
            self.assertEqual(restore_vector, 0)
            self.assertIsNone(hot_pos)

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=False))
    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_restore_vector_case9(self, mock_get_next):
        """
        case9: 非HBM，动态shape
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        with tf.Graph().as_default():
            mock_get_next.return_value = [0]
            self.config["skip_emb_transfer"] = False
            self.config["use_hot"] = False
            restore_vector, hot_pos = get_restore_vector(self.config)
            self.assertEqual(restore_vector, 0)
            self.assertIsNone(hot_pos)


class TestGetIdOffsetsFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_id_offsets'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)
        self.max_lookup_vec_size = self.config.get("send_count") * self.config.get("rank_size")

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)

    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_id_offsets_case1(self, mock_get_next):
        """
        case1: 动态扩容
        """

        from mx_rec.core.asc.build_graph import get_id_offsets

        with tf.Graph().as_default():
            mock_get_next.return_value = [0]
            id_offsets, swap_pos, swap_len = get_id_offsets(self.max_lookup_vec_size, self.config)
            self.assertEqual(id_offsets, 0)
            self.assertListEqual(swap_pos, [])
            self.assertEqual(swap_len, 0)

    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_id_offsets_case2(self, mock_get_next):
        """
        case2: 非动态扩容，HBM
        """

        from mx_rec.core.asc.build_graph import get_id_offsets

        with tf.Graph().as_default():
            self.config["use_dynamic_expansion"] = False
            mock_get_next.return_value = [0]
            id_offsets, swap_pos, swap_len = get_id_offsets(self.max_lookup_vec_size, self.config)
            self.assertEqual(id_offsets, 0)
            self.assertListEqual(swap_pos, [])
            self.assertEqual(swap_len, 0)


class TestGetRestoreVectorSecondFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_restore_vector_second'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)
        self.max_lookup_vec_size = self.config.get("send_count") * self.config.get("rank_size")

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)

    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_restore_vector_second(self, mock_get_next):
        """
        case: test get_restore_vector_second
        """

        from mx_rec.core.asc.build_graph import get_restore_vector_second

        with tf.Graph().as_default():
            mock_get_next.return_value = [0]
            restore_vector_second = get_restore_vector_second(self.max_lookup_vec_size, self.config)
            self.assertEqual(restore_vector_second, 0)


class TestGetUniqueKeysFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_unique_keys'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)
        self.max_lookup_vec_size = self.config.get("send_count") * self.config.get("rank_size")

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)

    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_unique_keys_case1(self, mock_get_next):
        """
        case1: 动态扩容
        """

        from mx_rec.core.asc.build_graph import get_unique_keys

        with tf.Graph().as_default():
            mock_get_next.return_value = [0]
            unique_keys = get_unique_keys(self.max_lookup_vec_size, self.config)
            self.assertEqual(unique_keys, 0)

    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_unique_keys_case2(self, mock_get_next):
        """
        case2: 非动态扩容
        """

        from mx_rec.core.asc.build_graph import get_unique_keys

        with tf.Graph().as_default():
            self.config["use_dynamic_expansion"] = False
            mock_get_next.return_value = [1]
            unique_keys = get_unique_keys(self.max_lookup_vec_size, self.config)
            self.assertEqual(unique_keys, 1)


class TestGetAll2allArgsFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_all2all_args'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)

    def test_get_all2all_args_case1(self):
        """
        case1: 静态shape
        """

        from mx_rec.core.asc.build_graph import get_all2all_args

        with tf.Graph().as_default():
            all2all_args = get_all2all_args(True, self.config)
            self.assertIsNone(all2all_args)

    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_all2all_args_case2(self, mock_get_next):
        """
        case2: 动态shape
        """

        from mx_rec.core.asc.build_graph import get_all2all_args

        with tf.Graph().as_default():
            mock_get_next.return_value = [0]
            all2all_args = get_all2all_args(False, self.config)
            self.assertEqual(all2all_args, 0)


class TestGetSwapInfoFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_swap_info'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=True))
    def test_get_swap_info_case1(self):
        """
        case1: 静态shape，HBM
        """

        from mx_rec.core.asc.build_graph import get_swap_info

        with tf.Graph().as_default():
            swap_in = get_swap_info(self.config, None, None, None)
            self.assertIsInstance(swap_in[0], type(tf.no_op()))

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=True))
    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_swap_info_case2(self, mock_get_next):
        """
        case2: 静态shape，非HBM，table传入非list，抛出异常
        """

        from mx_rec.core.asc.build_graph import get_swap_info

        with tf.Graph().as_default():
            mock_get_next.return_value = tf.ones(shape=[8, 8], dtype=tf.float32)
            swap_pos = tf.constant([8, 9], dtype=tf.int32)
            swap_len = tf.constant(2, dtype=tf.int32)
            table = tf.compat.v1.get_variable("test_table", shape=[10, 8], initializer=tf.ones_initializer())
            self.config["skip_emb_transfer"] = False
            with self.assertRaises(RuntimeError):
                get_swap_info(self.config, swap_len, swap_pos, table)


class TestGetPreProcessedTensorForAscFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_preprocessed_tensor_for_asc'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, skip_emb_transfer=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_hot=True, use_dynamic_expansion=True)
        global_env.apply_gradients_strategy = "direct_apply"

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=True),
                         get_restore_vector=mock.MagicMock(return_value=[0, 0]),
                         get_id_offsets=mock.MagicMock(return_value=[0, 0, 0]),
                         get_all2all_args=mock.MagicMock(return_value=0),
                         get_swap_info=mock.MagicMock(return_value=0),
                         get_restore_vector_second=mock.MagicMock(return_value=0),
                         get_unique_keys=mock.MagicMock(return_value=0))
    def test_get_preprocessed_tensor_for_asc_case1(self):
        """
        case1: 静态shape，全局unique
        """

        from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc

        global_env.apply_gradients_strategy = "sum_same_id_gradients_and_apply"
        with tf.Graph().as_default():
            result = get_preprocessed_tensor_for_asc(None, self.config)
            self.assertIsNotNone(result.get("restore_vector"))
            self.assertIsNotNone(result.get("restore_vector_second"))
            self.assertIsNotNone(result.get("unique_keys"))

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=False),
                         get_restore_vector=mock.MagicMock(return_value=[0, 0]),
                         get_id_offsets=mock.MagicMock(return_value=[0, 0, 0]),
                         get_all2all_args=mock.MagicMock(return_value=0),
                         get_swap_info=mock.MagicMock(return_value=0),
                         get_restore_vector_second=mock.MagicMock(return_value=0),
                         get_unique_keys=mock.MagicMock(return_value=0))
    def test_get_preprocessed_tensor_for_asc_case2(self):
        """
        case2: 动态shape，全局unique
        """

        from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc

        global_env.apply_gradients_strategy = "sum_same_id_gradients_and_apply"
        with tf.Graph().as_default():
            result = get_preprocessed_tensor_for_asc(None, self.config)
            self.assertIsNotNone(result.get("restore_vector"))
            self.assertIsNotNone(result.get("restore_vector_second"))
            self.assertIsNotNone(result.get("unique_keys"))

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=False),
                         get_restore_vector=mock.MagicMock(return_value=[0, 0]),
                         get_id_offsets=mock.MagicMock(return_value=[0, 0, 0]),
                         get_all2all_args=mock.MagicMock(return_value=0),
                         get_swap_info=mock.MagicMock(return_value=0),
                         get_restore_vector_second=mock.MagicMock(return_value=0),
                         get_unique_keys=mock.MagicMock(return_value=0))
    def test_get_preprocessed_tensor_for_asc_case3(self):
        """
        case3: 动态shape，全局unique，channel_id=1
        """

        from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc

        global_env.apply_gradients_strategy = "sum_same_id_gradients_and_apply"
        with tf.Graph().as_default():
            self.config["channel_id"] = 1
            result = get_preprocessed_tensor_for_asc(None, self.config)
            self.assertIsNotNone(result.get("restore_vector"))
            self.assertIsNone(result.get("restore_vector_second"))

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_use_static=mock.MagicMock(return_value=False),
                         get_restore_vector=mock.MagicMock(return_value=[0, 0]),
                         get_id_offsets=mock.MagicMock(return_value=[0, 0, 0]),
                         get_all2all_args=mock.MagicMock(return_value=0),
                         get_swap_info=mock.MagicMock(return_value=0),
                         get_restore_vector_second=mock.MagicMock(return_value=0),
                         get_unique_keys=mock.MagicMock(return_value=0))
    def test_get_preprocessed_tensor_for_asc_case4(self):
        """
        case4: 动态shape
        """

        from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc

        with tf.Graph().as_default():
            result = get_preprocessed_tensor_for_asc(None, self.config)
            self.assertIsNotNone(result.get("restore_vector"))
            self.assertIsNone(result.get("restore_vector_second"))


if __name__ == '__main__':
    unittest.main()
