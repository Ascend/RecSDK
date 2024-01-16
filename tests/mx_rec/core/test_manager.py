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

import tensorflow as tf

from mx_rec.core.asc.feature_spec import FeatureSpec
from tests.mx_rec.core.mock_class import MockSparseEmbedding, MockOptimizer, MockHybridMgmt


class TestCheckDanglingTableFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.manager.check_dangling_table'.
    """

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         export_dangling_table=mock.MagicMock(return_value=[]),
                         export_table_instances=mock.MagicMock(return_value={}),
                         find_dangling_table=mock.MagicMock(return_value=["test_table"]))
    def test_check_dangling_table(self):
        """
        case: test check_dangling_table
        """

        from mx_rec.core.asc.manager import check_dangling_table

        self.assertListEqual(check_dangling_table(), ["test_table"])


class TestGenerateTableInfoListFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.manager.generate_table_info_list'.
    """

    @mock.patch("mx_rec.core.asc.manager.export_table_instances")
    def test_generate_table_info_list_case1(self, mock_export_table_instances):
        """
        case1: 一张表开DDR，一张表没开DDR，抛出异常
        """

        from mx_rec.core.asc.manager import generate_table_info_list

        with tf.Graph().as_default():
            test_table1 = MockSparseEmbedding("test_table1")
            test_table1.host_vocabulary_size = 1
            test_table2 = MockSparseEmbedding("test_table2")
            test_table2.host_vocabulary_size = 0
            mock_export_table_instances.return_value = {
                "test_table1": test_table1,
                "test_table2": test_table2
            }

            with self.assertRaises(ValueError):
                generate_table_info_list()

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         export_optimizer=mock.MagicMock(return_value=None),
                         should_skip=mock.MagicMock(return_value=True),
                         check_dangling_table=mock.MagicMock(return_value=["test_table"]))
    @mock.patch("mx_rec.core.asc.manager.export_table_instances")
    def test_generate_table_info_list_case2(self, mock_export_table_instances):
        """
        case2: test_table是dangling_table，skip为True
        """

        from mx_rec.core.asc.manager import generate_table_info_list

        with tf.Graph().as_default():
            test_table = MockSparseEmbedding("test_table")
            test_table.host_vocabulary_size = 1
            mock_export_table_instances.return_value = dict(test_table=test_table)

            table_info_list = generate_table_info_list()
            self.assertListEqual(table_info_list, [])

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         export_optimizer=mock.MagicMock(return_value=None),
                         should_skip=mock.MagicMock(return_value=True),
                         check_dangling_table=mock.MagicMock(return_value=[]))
    @mock.patch("mx_rec.core.asc.manager.export_table_instances")
    def test_generate_table_info_list_case3(self, mock_export_table_instances):
        """
        case3: test_table不是dangling_table，skip为True
        """

        from mx_rec.core.asc.manager import generate_table_info_list

        with tf.Graph().as_default():
            test_table = MockSparseEmbedding("test_table")
            test_table.host_vocabulary_size = 1
            mock_export_table_instances.return_value = dict(test_table=test_table)

            table_info_list = generate_table_info_list()
            self.assertListEqual(table_info_list, [])

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         export_optimizer=mock.MagicMock(return_value=None),
                         EmbInfoParams=mock.MagicMock(return_value=None),
                         EmbInfo=mock.MagicMock(return_value="test_table_info"),
                         matched_emb_initializer=mock.MagicMock(return_value=[]),
                         matched_opt_slot_initializers=mock.MagicMock(return_value=[]),
                         should_skip=mock.MagicMock(return_value=False),
                         get_use_static=mock.MagicMock(return_value=True),
                         check_dangling_table=mock.MagicMock(return_value=[]))
    @mock.patch("mx_rec.core.asc.manager.export_table_instances")
    def test_generate_table_info_list_case3(self, mock_export_table_instances):
        """
        case4: 静态shape，test_table不是dangling_table，skip为False
        """

        from mx_rec.core.asc.manager import generate_table_info_list

        with tf.Graph().as_default():
            test_table = MockSparseEmbedding("test_table")
            test_table.host_vocabulary_size = 8
            test_table.send_count = 1
            test_table.slice_device_vocabulary_size = 1
            test_table.slice_host_vocabulary_size = 1
            test_table.slice_ssd_vocabulary_size = 0
            test_table.is_grad = True
            test_table.is_save = True
            test_table.scalar_emb_size = 8
            test_table.ext_emb_size = 8
            test_table.ssd_data_path = ""
            mock_export_table_instances.return_value = dict(test_table=test_table)

            table_info_list = generate_table_info_list()
            self.assertListEqual(table_info_list, ["test_table_info"])


class TestMatchedConstantInitializerFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.manager.matched_constant_initializer'.
    """

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         InitializeInfo=mock.MagicMock(return_value=[]),
                         ConstantInitializerInfo=mock.MagicMock(return_value=[]))
    def test_matched_constant_initializer(self):
        """
        case: test matched_constant_initializer
        """

        from mx_rec.core.asc.manager import matched_constant_initializer

        with tf.Graph().as_default():
            table_info = MockSparseEmbedding("test_table")
            table_info.init_param = 1.
            table_info.scalar_emb_size = 8
            table_info.emb_initializer.value = 0

            self.assertListEqual(matched_constant_initializer(table_info), [])


class TestMatchedRandomNormalInitializerFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.manager.matched_random_normal_initializer'.
    """

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         InitializeInfo=mock.MagicMock(return_value=[]),
                         NormalInitializerInfo=mock.MagicMock(return_value=[]))
    def test_matched_random_normal_initializer_case1(self):
        """
        case1: emb_initializer.seed为None
        """

        from mx_rec.core.asc.manager import matched_random_normal_initializer

        with tf.Graph().as_default():
            table_info = MockSparseEmbedding("test_table")
            table_info.init_param = 1.
            table_info.scalar_emb_size = 8
            table_info.emb_initializer.seed = None
            table_info.emb_initializer.mean = 1
            table_info.emb_initializer.stddev = 1

            self.assertListEqual(matched_random_normal_initializer(table_info), [])

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         InitializeInfo=mock.MagicMock(return_value=[]),
                         NormalInitializerInfo=mock.MagicMock(return_value=[]))
    def test_matched_random_normal_initializer_case2(self):
        """
        case2: emb_initializer.seed非None
        """

        from mx_rec.core.asc.manager import matched_random_normal_initializer

        with tf.Graph().as_default():
            table_info = MockSparseEmbedding("test_table")
            table_info.init_param = 1.
            table_info.scalar_emb_size = 8
            table_info.emb_initializer.seed = 1
            table_info.emb_initializer.mean = 1
            table_info.emb_initializer.stddev = 1

            self.assertListEqual(matched_random_normal_initializer(table_info), [])


class TestMatchedTruncatedNormalInitializerFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.manager.matched_truncated_normal_initializer'.
    """

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         InitializeInfo=mock.MagicMock(return_value=[]),
                         NormalInitializerInfo=mock.MagicMock(return_value=[]))
    def test_matched_truncated_normal_initializer_case1(self):
        """
        case1: emb_initializer.seed为None
        """

        from mx_rec.core.asc.manager import matched_truncated_normal_initializer

        with tf.Graph().as_default():
            table_info = MockSparseEmbedding("test_table")
            table_info.init_param = 1.
            table_info.scalar_emb_size = 8
            table_info.emb_initializer.seed = None
            table_info.emb_initializer.mean = 1
            table_info.emb_initializer.stddev = 1

            self.assertListEqual(matched_truncated_normal_initializer(table_info), [])

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         InitializeInfo=mock.MagicMock(return_value=[]),
                         NormalInitializerInfo=mock.MagicMock(return_value=[]))
    def test_matched_random_normal_initializer_case2(self):
        """
        case2: emb_initializer.seed非None
        """

        from mx_rec.core.asc.manager import matched_truncated_normal_initializer

        with tf.Graph().as_default():
            table_info = MockSparseEmbedding("test_table")
            table_info.init_param = 1.
            table_info.scalar_emb_size = 8
            table_info.emb_initializer.seed = 1
            table_info.emb_initializer.mean = 1
            table_info.emb_initializer.stddev = 1

            self.assertListEqual(matched_truncated_normal_initializer(table_info), [])


class TestMatchedEmbInitializerFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.manager.matched_emb_initializer'.
    """

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         matched_constant_initializer=mock.MagicMock(return_value=1))
    def test_matched_emb_initializer_case1(self):
        """
        case1: 初始化器为 tf.constant_initializer
        """

        from mx_rec.core.asc.manager import matched_emb_initializer

        with tf.Graph().as_default():
            table_info = MockSparseEmbedding("test_table")
            table_info.scalar_emb_size = 8
            table_info.emb_initializer = tf.constant_initializer()

            self.assertEqual(matched_emb_initializer(table_info), 1)

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         matched_random_normal_initializer=mock.MagicMock(return_value=2))
    def test_matched_emb_initializer_case2(self):
        """
        case2: 初始化器为 tf.random_normal_initializer
        """

        from mx_rec.core.asc.manager import matched_emb_initializer

        with tf.Graph().as_default():
            table_info = MockSparseEmbedding("test_table")
            table_info.scalar_emb_size = 8
            table_info.emb_initializer = tf.random_normal_initializer()

            self.assertEqual(matched_emb_initializer(table_info), 2)

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         matched_truncated_normal_initializer=mock.MagicMock(return_value=3))
    def test_matched_emb_initializer_case3(self):
        """
        case3: 初始化器为 tf.truncated_normal_initializer
        """

        from mx_rec.core.asc.manager import matched_emb_initializer

        with tf.Graph().as_default():
            table_info = MockSparseEmbedding("test_table")
            table_info.scalar_emb_size = 8
            table_info.emb_initializer = tf.truncated_normal_initializer()

            self.assertEqual(matched_emb_initializer(table_info), 3)


class TestMatchedOptSlotInitializersFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.manager.matched_opt_slot_initializers'.
    """

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         InitializeInfo=mock.MagicMock(return_value="slot_initializer"))
    def test_matched_opt_slot_initializers(self):
        """
        case: test matched_opt_slot_initializers
        """

        from mx_rec.core.asc.manager import matched_opt_slot_initializers

        with tf.Graph().as_default():
            table_instance = MockSparseEmbedding("test_table")
            table_instance.scalar_emb_size = 8
            table_instance.ext_emb_size = 8
            table_instance.optimizer_instance_list = [MockOptimizer()]

            slot_initializers = matched_opt_slot_initializers(table_instance)
            self.assertListEqual(slot_initializers, ["slot_initializer", "slot_initializer"])


class TestGenerateThresholdListFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.manager.generate_threshold_list'.
    """

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         ThresholdValue=mock.MagicMock(return_value=0))
    @mock.patch("mx_rec.core.asc.manager.export_feature_spec")
    def test_generate_threshold_list(self, mock_export_feature_spec):
        """
        case: 有淘汰、准入
        """

        from mx_rec.core.asc.manager import generate_threshold_list

        with tf.Graph().as_default():
            test_feature_spec1 = FeatureSpec("test_feature_spec1",
                                             access_threshold=5, eviction_threshold=10, faae_coefficient=None)
            test_feature_spec2 = FeatureSpec("test_feature_spec2",
                                             access_threshold=5, faae_coefficient=None)
            mock_export_feature_spec.return_value = {
                "test_feature_spec1": test_feature_spec1,
                "test_feature_spec2": test_feature_spec2
            }

            self.assertListEqual(generate_threshold_list(), [0, 0])


class TestInitializeEmbCacheFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.manager.initialize_emb_cache'.
    """

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         get_rank_id=mock.MagicMock(return_value=0),
                         get_device_id=mock.MagicMock(return_value=0),
                         get_rank_size=mock.MagicMock(return_value=0),
                         get_train_steps=mock.MagicMock(return_value=0),
                         get_eval_steps=mock.MagicMock(return_value=0),
                         get_save_steps=mock.MagicMock(return_value=0),
                         get_if_load=mock.MagicMock(return_value=False),
                         get_use_static=mock.MagicMock(return_value=True),
                         get_use_hot=mock.MagicMock(return_value=True),
                         get_use_dynamic_expansion=mock.MagicMock(return_value=True),
                         USE_STATIC=mock.MagicMock(return_value=0),
                         USE_HOT=mock.MagicMock(return_value=1),
                         USE_DYNAMIC_EXPANSION=mock.MagicMock(return_value=2),
                         RankInfo=mock.MagicMock(return_value="mock_info"),
                         HybridMgmt=mock.MagicMock(return_value=MockHybridMgmt(is_initialized=False)))
    def test_initialize_emb_cache_case1(self):
        """
        case1: 初始化失败
        """

        from mx_rec.core.asc.manager import initialize_emb_cache

        with self.assertRaises(RuntimeError):
            initialize_emb_cache([], [])

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         get_rank_id=mock.MagicMock(return_value=0),
                         get_device_id=mock.MagicMock(return_value=0),
                         get_rank_size=mock.MagicMock(return_value=0),
                         get_train_steps=mock.MagicMock(return_value=0),
                         get_eval_steps=mock.MagicMock(return_value=0),
                         get_save_steps=mock.MagicMock(return_value=0),
                         get_if_load=mock.MagicMock(return_value=False),
                         get_use_static=mock.MagicMock(return_value=True),
                         get_use_hot=mock.MagicMock(return_value=True),
                         get_use_dynamic_expansion=mock.MagicMock(return_value=True),
                         USE_STATIC=mock.MagicMock(return_value=0),
                         USE_HOT=mock.MagicMock(return_value=1),
                         USE_DYNAMIC_EXPANSION=mock.MagicMock(return_value=2),
                         RankInfo=mock.MagicMock(return_value="mock_info"),
                         set_asc_manager=mock.MagicMock(return_value=None))
    @mock.patch("mx_rec.core.asc.manager.HybridMgmt")
    def test_initialize_emb_cache_case2(self, mock_hybrid_mgmt):
        """
        case2: 初始化成功
        """

        from mx_rec.core.asc.manager import initialize_emb_cache

        mock_mgmt = MockHybridMgmt(is_initialized=True)
        mock_hybrid_mgmt.return_value = mock_mgmt
        initialize_emb_cache([], [])
        self.assertTrue(mock_mgmt.initialize())


class TestStartAscPipeLineFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.manager.start_asc_pipeline'.
    """

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         generate_table_info_list=mock.MagicMock(return_value=[]),
                         generate_threshold_list=mock.MagicMock(return_value=[]))
    def test_start_asc_pipeline_case1(self):
        """
        case1: table_info_list为[]
        """

        from mx_rec.core.asc.manager import start_asc_pipeline

        with self.assertRaises(RuntimeError):
            start_asc_pipeline()

    @mock.patch.multiple("mx_rec.core.asc.manager",
                         generate_table_info_list=mock.MagicMock(return_value=["test_table"]),
                         generate_threshold_list=mock.MagicMock(return_value=[]),
                         get_stat_on=mock.MagicMock(return_value=True),
                         is_asc_manager_initialized=mock.MagicMock(return_value=False),
                         export_table_num=mock.MagicMock(return_value=None),
                         initialize_emb_cache=mock.MagicMock(return_value=None))
    def test_start_asc_pipeline_case2(self):
        """
        case2: table_info_list为["test_table"]，stat_on为True
        """

        from mx_rec.core.asc.manager import start_asc_pipeline

        # 该函数无返回值，且内部调用的函数已经被mock了，无异常、无返回值
        start_asc_pipeline()
        self.assertTrue(callable(start_asc_pipeline))


if __name__ == '__main__':
    unittest.main()
