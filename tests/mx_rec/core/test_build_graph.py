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

import unittest
from unittest import mock

import tensorflow as tf

from mx_rec.util.global_env_conf import global_env
from mx_rec.core.asc.build_graph import SwapInfo
from tests.mx_rec.core.mock_class import MockConfigInitializer


class TestGetRestoreVectorFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_restore_vector'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, is_hbm=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_dynamic_expansion=True)

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, is_hbm=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_dynamic_expansion=True)

    def test_get_restore_vector_case1(self):
        """
        case1: HBM，emb_size不为int，抛出异常

        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        self.config["emb_size"] = "xxx"
        with self.assertRaises(TypeError):
            get_restore_vector(self.config)

    def test_get_restore_vector_case3(self):
        """
        case3: 非HBM，ext_emb_size不为int，抛出异常
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        self.config["is_hbm"] = False
        self.config["ext_emb_size"] = "xxx"
        with self.assertRaises(TypeError):
            get_restore_vector(self.config)

    @mock.patch("mx_rec.core.asc.build_graph.ConfigInitializer")
    @mock.patch("mx_rec.core.asc.build_graph.mxrec_pybind.get_ub_hot_size")
    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_restore_vector_case5(self, mock_get_next, mock_get_ub_hot_size, build_graph_config_initializer):
        """
        case5: HBM，静态shape，hot emb
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        with tf.Graph().as_default():
            mock_config_initializer = MockConfigInitializer(use_static=True)
            build_graph_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            mock_get_next.return_value = [0, 1]
            mock_get_ub_hot_size.return_value = 8
            restore_vector, hot_pos = get_restore_vector(self.config)
            self.assertEqual(restore_vector, 0)
            self.assertEqual(hot_pos, 1)

    @mock.patch("mx_rec.core.asc.build_graph.ConfigInitializer")
    @mock.patch("mx_rec.core.asc.build_graph.mxrec_pybind.get_ub_hot_size")
    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_restore_vector_case6(self, mock_get_next, mock_get_ub_hot_size, build_graph_config_initializer):
        """
        case6: HBM，动态shape，hot emb
        """

        from mx_rec.core.asc.build_graph import get_restore_vector

        with tf.Graph().as_default():
            mock_config_initializer = MockConfigInitializer(use_static=True)
            build_graph_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            mock_get_next.return_value = [0, 1]
            mock_get_ub_hot_size.return_value = 8
            restore_vector, hot_pos = get_restore_vector(self.config)
            self.assertEqual(restore_vector, 0)
            self.assertEqual(hot_pos, 1)


class TestGetIdOffsetsFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_id_offsets'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, is_hbm=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_dynamic_expansion=True)
        self.max_lookup_vec_size = self.config.get("send_count") * self.config.get("rank_size")

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, is_hbm=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_dynamic_expansion=True)

    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_id_offsets_case1(self, mock_get_next):
        """
        case1: 动态扩容
        """

        from mx_rec.core.asc.build_graph import get_id_offsets

        with tf.Graph().as_default():
            mock_get_next.return_value = [0]
            id_offsets, swap_info = get_id_offsets(self.max_lookup_vec_size, self.config)
            self.assertEqual(id_offsets, 0)
            self.assertListEqual(swap_info.swap_in_pos, [])
            self.assertEqual(swap_info.swap_in_len, 0)
            self.assertListEqual(swap_info.swap_out_pos, [])
            self.assertEqual(swap_info.swap_out_len, 0)

    @mock.patch("mx_rec.core.asc.build_graph.npu_ops.gen_npu_ops.get_next")
    def test_get_id_offsets_case2(self, mock_get_next):
        """
        case2: 非动态扩容，HBM
        """

        from mx_rec.core.asc.build_graph import get_id_offsets

        with tf.Graph().as_default():
            self.config["use_dynamic_expansion"] = False
            mock_get_next.return_value = [0]
            id_offsets, swap_info = get_id_offsets(self.max_lookup_vec_size, self.config)
            self.assertEqual(id_offsets, 0)
            self.assertListEqual(swap_info.swap_in_pos, [])
            self.assertEqual(swap_info.swap_in_len, 0)
            self.assertListEqual(swap_info.swap_out_pos, [])
            self.assertEqual(swap_info.swap_out_len, 0)


class TestGetAll2allArgsFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_all2all_args'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, is_hbm=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_dynamic_expansion=True)

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, is_hbm=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_dynamic_expansion=True)

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


class TestGetPreProcessedTensorForAscFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.build_graph.get_preprocessed_tensor_for_asc'.
    """

    def setUp(self):
        # 默认动态扩容、hot emb、HBM
        self.config = dict(table_name="test_table", channel_id=0, is_hbm=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_dynamic_expansion=True)

    def tearDown(self):
        # 恢复config
        self.config = dict(table_name="test_table", channel_id=0, is_hbm=True, emb_size=8, ext_emb_size=8,
                           feat_cnt=8, batch_size=32, rank_size=8, send_count=1, device_id=0,
                           use_dynamic_expansion=True)

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_restore_vector=mock.MagicMock(return_value=[0, 0]),
                         get_id_offsets=mock.MagicMock(return_value=[0, SwapInfo()]),
                         get_all2all_args=mock.MagicMock(return_value=0))
    @mock.patch("mx_rec.core.asc.build_graph.ConfigInitializer")
    def test_get_preprocessed_tensor_for_asc_case1(self, build_graph_config_initializer):
        """
        case1: 静态shape，全局unique
        """

        from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc

        with tf.Graph().as_default():
            mock_config_initializer = MockConfigInitializer(use_static=True)
            build_graph_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            result = get_preprocessed_tensor_for_asc(None, self.config)
            self.assertIsNotNone(result.get("restore_vector"))

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_restore_vector=mock.MagicMock(return_value=[0, 0]),
                         get_id_offsets=mock.MagicMock(return_value=[0, SwapInfo()]),
                         get_all2all_args=mock.MagicMock(return_value=0))
    @mock.patch("mx_rec.core.asc.build_graph.ConfigInitializer")
    def test_get_preprocessed_tensor_for_asc_case2(self, build_graph_config_initializer):
        """
        case2: 动态shape，全局unique
        """

        from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc

        with tf.Graph().as_default():
            mock_config_initializer = MockConfigInitializer()
            build_graph_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            result = get_preprocessed_tensor_for_asc(None, self.config)
            self.assertIsNotNone(result.get("restore_vector"))

    @mock.patch.multiple("mx_rec.core.asc.build_graph",
                         get_restore_vector=mock.MagicMock(return_value=[0, 0]),
                         get_id_offsets=mock.MagicMock(return_value=[0, SwapInfo]),
                         get_all2all_args=mock.MagicMock(return_value=0))
    @mock.patch("mx_rec.core.asc.build_graph.ConfigInitializer")
    def test_get_preprocessed_tensor_for_asc_case3(self, build_graph_config_initializer):
        """
        case3: 动态shape，全局unique，channel_id=1
        """

        from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc

        with tf.Graph().as_default():
            mock_config_initializer = MockConfigInitializer()
            build_graph_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            self.config["channel_id"] = 1
            result = get_preprocessed_tensor_for_asc(None, self.config)
            self.assertIsNotNone(result.get("restore_vector"))


if __name__ == '__main__':
    unittest.main()
