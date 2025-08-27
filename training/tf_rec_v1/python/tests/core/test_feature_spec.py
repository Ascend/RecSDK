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
from functools import reduce

import tensorflow as tf

from mx_rec.core.asc.feature_spec import FeatureSpec
from core.mock_class import MockConfigInitializer


class TestFeatureSpecClass(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.feature_spec.FeatureSpec'.
    """

    def setUp(self):
        # 每个测试方法执行前，将FeatureSpec的静态成员设为默认值
        FeatureSpec.instance_count_train = 0
        FeatureSpec.instance_count_eval = 0
        FeatureSpec.use_timestamp_train = False
        FeatureSpec.use_timestamp_eval = False

    def test_init_case1(self):
        """
        case1: 初始化实例成功
        """
        self.assertIsInstance(FeatureSpec("case1"), FeatureSpec)

    def test_init_case2(self):
        """
        case2: 当access_threshold为None，eviction_threshold不为None时，初始化失败并抛出异常
        """
        with self.assertRaises(ValueError):
            FeatureSpec("case2", eviction_threshold=20)


class TestIncludeTimestampFuncOfFeatureSpecClass(TestFeatureSpecClass):
    """
    Test for 'mx_rec.core.asc.feature_spec.FeatureSpec.include_timestamp'.
    """

    def setUp(self):
        # 每个测试方法执行前，将FeatureSpec的静态成员设为默认值
        super().setUp()

    def test_include_timestamp_case1(self):
        """
        case1: is_training为False
        """
        case1_feature_spec = FeatureSpec("case1")
        is_training = False
        case1_feature_spec.include_timestamp(is_training)
        self.assertTrue(case1_feature_spec.use_timestamp_eval)

    def test_include_timestamp_case2(self):
        """
        case2: is_training为True
        """
        case2_feature_spec = FeatureSpec("case2")
        is_training = True
        case2_feature_spec.include_timestamp(is_training)
        self.assertTrue(case2_feature_spec.use_timestamp_train)

    def test_include_timestamp_case3(self):
        """
        case3: is_training为True并调用2次include_timestamp，抛出EnvironmentError异常
        """
        case3_feature_spec = FeatureSpec("case3")
        is_training = True
        case3_feature_spec.include_timestamp(is_training)
        with self.assertRaises(EnvironmentError):
            case3_feature_spec.include_timestamp(is_training)


class TestUseTimestampFuncOfFeatureSpecClass(TestFeatureSpecClass):
    """
    Test for 'mx_rec.core.asc.feature_spec.FeatureSpec.use_timestamp'.
    """

    def setUp(self):
        # 每个测试方法执行前，将FeatureSpec的静态成员设为默认值
        super().setUp()

    def test_use_timestamp_case1(self):
        """
        case1: is_training为False
        """

        case1_feature_spec = FeatureSpec("case1")
        is_training = False
        self.assertEqual(case1_feature_spec.use_timestamp(is_training), FeatureSpec.use_timestamp_eval)

    def test_use_timestamp_case2(self):
        """
        case2: is_training为True
        """

        case2_feature_spec = FeatureSpec("case2")
        is_training = True
        self.assertEqual(case2_feature_spec.use_timestamp(is_training), FeatureSpec.use_timestamp_train)


class TestSetFeatPosFuncOfFeatureSpecClass(TestFeatureSpecClass):
    """
    Test for 'mx_rec.core.asc.feature_spec.FeatureSpec.set_feat_pos'.
    """

    def setUp(self):
        # 每个测试方法执行前，将FeatureSpec的静态成员设为默认值
        super().setUp()

    def test_set_feat_pos_case1(self):
        """
        case1: is_training为False
        """

        case1_feature_spec = FeatureSpec("case1")
        is_training = False
        set_times = 5
        for _ in range(set_times):
            case1_feature_spec.set_feat_pos(is_training)
        # 因为feat_pos_eval/train = instance_count_eval/train ++，因此feat_pos_eval会少1
        self.assertEqual(case1_feature_spec.feat_pos_eval, set_times - 1)
        self.assertEqual(case1_feature_spec.instance_count_eval, set_times)

    def test_set_feat_pos_case2(self):
        """
        case2: is_training为True
        """

        case2_feature_spec = FeatureSpec("case2")
        is_training = True
        set_times = 5
        for _ in range(set_times):
            case2_feature_spec.set_feat_pos(is_training)
        # 因为feat_pos_eval/train = instance_count_eval/train ++，因此feat_pos_eval会少1
        self.assertEqual(case2_feature_spec.feat_pos_train, set_times - 1)
        self.assertEqual(case2_feature_spec.instance_count_train, set_times)


@mock.patch.multiple(
    "mx_rec.core.asc.feature_spec",
    ConfigInitializer=mock.Mock(return_value=MockConfigInitializer()),
)
class TestInsertPipelineModeFuncOfFeatureSpecClass(TestFeatureSpecClass):
    """
    Test for 'mx_rec.core.asc.feature_spec.FeatureSpec.insert_pipeline_mode'.
    """

    def setUp(self):
        # 每个测试方法执行前，将FeatureSpec的静态成员设为默认值
        super().setUp()

    def test_insert_pipeline_mode_case1(self):
        """
        case1: mode为非bool类型，抛出异常
        """

        case1_feature_spec = FeatureSpec("case1")
        mode = "xxx"
        with self.assertRaises(TypeError):
            case1_feature_spec.insert_pipeline_mode(mode)

    def test_insert_pipeline_mode_case2(self):
        """
        case2: mode为False
        """

        case2_feature_spec = FeatureSpec("case2")
        mode = False
        case2_feature_spec.insert_pipeline_mode(mode)
        self.assertSetEqual(case2_feature_spec.pipeline_mode, {False})

    def test_insert_pipeline_mode_case3(self):
        """
        case3: mode为True
        """

        case3_feature_spec = FeatureSpec("case3")
        mode = True
        case3_feature_spec.insert_pipeline_mode(mode)
        self.assertSetEqual(case3_feature_spec.pipeline_mode, {True})

    def test_insert_pipeline_mode_case4(self):
        """
        case4: mode为True，在已经设置过一次True的情况下，设置第二次后无报错
        """

        case4_feature_spec = FeatureSpec("case4")
        mode = True
        case4_feature_spec.insert_pipeline_mode(mode)
        case4_feature_spec.insert_pipeline_mode(mode)
        self.assertSetEqual(case4_feature_spec.pipeline_mode, {True})


@mock.patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=mock.Mock(return_value=MockConfigInitializer()),
)
class TestSetFeatAttributeFuncOfFeatureSpecClass(TestFeatureSpecClass):
    """
    Test for 'mx_rec.core.asc.feature_spec.FeatureSpec.set_feat_attribute'.
    """

    def setUp(self):
        self.is_training = True
        # 每个测试方法执行前，将FeatureSpec的静态成员设为默认值
        super().setUp()

    @mock.patch("mx_rec.core.asc.feature_spec.ConfigInitializer")
    def test_set_feat_attribute_case1(self, feature_spec_config_initializer):
        """
        case1: 未初始化initialized成员，静态shape，tensor的rank为0，抛出异常
        """

        mock_config_initializer = MockConfigInitializer(use_static=True)
        feature_spec_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        case1_tensor = tf.ones([], tf.int32)
        case1_feature_spec = FeatureSpec("case1")
        with self.assertRaises(ValueError):
            case1_feature_spec.set_feat_attribute(case1_tensor, self.is_training)

    @mock.patch("mx_rec.core.asc.feature_spec.ConfigInitializer")
    def test_set_feat_attribute_case2(self, feature_spec_config_initializer):
        """
        case2: 未初始化initialized成员，静态shape，tensor的rank等于1
        """

        mock_config_initializer = MockConfigInitializer(use_static=True)
        feature_spec_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        case2_tensor = tf.ones([32], tf.int32)
        case2_feature_spec = FeatureSpec("case2")
        case2_feature_spec.set_feat_attribute(case2_tensor, self.is_training)
        self.assertEqual(case2_feature_spec.feat_cnt, 1)

    @mock.patch("mx_rec.core.asc.feature_spec.ConfigInitializer")
    def test_set_feat_attribute_case3(self, feature_spec_config_initializer):
        """
        case3: 未初始化initialized成员，静态shape，tensor的rank大于1
        """

        mock_config_initializer = MockConfigInitializer(use_static=True)
        feature_spec_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        test_tensor_shape = [32, 16, 2]
        test_tensor = tf.ones(test_tensor_shape, tf.int32)
        case3_feature_spec = FeatureSpec("case3")
        case3_feature_spec.set_feat_attribute(test_tensor, self.is_training)
        reduce_shape = reduce(lambda x, y: x * y, test_tensor_shape[1:])
        self.assertTrue(case3_feature_spec.initialized)
        self.assertListEqual(case3_feature_spec.dims, test_tensor_shape)
        self.assertEqual(case3_feature_spec.tensor_rank, len(test_tensor_shape))
        self.assertEqual(case3_feature_spec.batch_size, test_tensor_shape[0])
        self.assertEqual(case3_feature_spec.feat_cnt, reduce_shape)
        self.assertEqual(case3_feature_spec.split, test_tensor_shape[0] * reduce_shape)

    @mock.patch("mx_rec.core.asc.feature_spec.ConfigInitializer")
    def test_set_feat_attribute_case4(self, feature_spec_config_initializer):
        """
        case4: 未初始化initialized成员，动态shape，tensor的rank大于1
        """
        mock_config_initializer = MockConfigInitializer(use_static=False)
        feature_spec_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        test_tensor_shape = [32, 3]
        test_tensor = tf.ones(test_tensor_shape, tf.int32)
        case4_feature_spec = FeatureSpec("case4")
        case4_feature_spec.set_feat_attribute(test_tensor, self.is_training)
        reduce_shape = reduce(lambda x, y: x * y, test_tensor_shape)
        with tf.Session() as sess:
            self.assertTrue(case4_feature_spec.initialized)
            self.assertEqual(sess.run(case4_feature_spec.dims), reduce_shape)
            self.assertEqual(case4_feature_spec.tensor_rank, 1)
            self.assertEqual(sess.run(case4_feature_spec.split), reduce_shape)
            self.assertEqual(sess.run(case4_feature_spec.batch_size), reduce_shape)
            self.assertEqual(case4_feature_spec.feat_cnt, 1)

    @mock.patch("mx_rec.core.asc.feature_spec.ConfigInitializer")
    def test_set_feat_attribute_case5(self, feature_spec_config_initializer):
        """
        case5: 静态shape，tensor的rank大于1，再次用不同tensor初始化initialized成员，抛出异常
        """
        mock_config_initializer = MockConfigInitializer(use_static=True)
        feature_spec_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

        case5_feature_spec = FeatureSpec("case5")
        case5_feature_spec.set_feat_attribute(tf.ones([32, 3], tf.int32), self.is_training)
        # 再次初始化
        with self.assertRaises(ValueError):
            case5_feature_spec.set_feat_attribute(tf.ones([64, 3], tf.int32), self.is_training)


class TestGetFeatureSpecFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.feature_spec.get_feature_spec'.
    """

    def test_get_feature_spec_case1(self):
        """
        case1: access_and_evict_config为None
        """

        from mx_rec.core.asc.feature_spec import get_feature_spec

        access_and_evict_config = None
        case1_results = get_feature_spec("fake_name", access_and_evict_config)
        self.assertIsInstance(case1_results, FeatureSpec)
        self.assertEqual(case1_results.access_threshold, None)
        self.assertEqual(case1_results.eviction_threshold, None)
        self.assertEqual(case1_results.faae_coefficient, None)

    def test_get_feature_spec_case2(self):
        """
        case2: access_and_evict_config不为None
        """

        from mx_rec.core.asc.feature_spec import get_feature_spec

        access_and_evict_config = dict(access_threshold=10, eviction_threshold=20, faae_coefficient=2)
        case2_results = get_feature_spec("fake_name", access_and_evict_config)
        self.assertIsInstance(case2_results, FeatureSpec)
        self.assertEqual(case2_results.access_threshold, access_and_evict_config.get("access_threshold"))
        self.assertEqual(case2_results.eviction_threshold, access_and_evict_config.get("eviction_threshold"))
        self.assertEqual(case2_results.faae_coefficient, access_and_evict_config.get("faae_coefficient"))


class TestSetTemporaryFeatureSpecAttributeFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.asc.feature_spec.set_temporary_feature_spec_attribute'.
    """

    def test_set_temporary_feature_spec_attribute(self):
        from mx_rec.core.asc.feature_spec import set_temporary_feature_spec_attribute

        user_ids_feature_spec = FeatureSpec("user_ids")
        total_feature_count = 200
        set_temporary_feature_spec_attribute(user_ids_feature_spec, total_feature_count)
        self.assertEqual(user_ids_feature_spec.batch_size, total_feature_count)
        self.assertEqual(user_ids_feature_spec.feat_cnt, 1)
        self.assertListEqual(user_ids_feature_spec.dims, [total_feature_count, 1])
        self.assertTrue(user_ids_feature_spec.initialized)
        self.assertSetEqual(user_ids_feature_spec.pipeline_mode, {True, False})


if __name__ == '__main__':
    unittest.main()