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

import os
import unittest
from unittest import mock

import tensorflow as tf

from mx_rec.core.asc import FeatureSpec
from mx_rec.core.asc.feature_spec import set_temporary_feature_spec_attribute
from mx_rec.core.embedding import SparseEmbedding
from mx_rec.constants.constants import All2allGradientsOp, ASCAnchorAttr
from tests.mx_rec.core.mock_class import MockSparseEmbedding, MockOptimizer, MockHcclOps, MockAscManager


class TestCreateTableFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.embedding.create_table'.
    """

    @mock.patch.multiple("mx_rec.core.embedding",
                         fix_invalid_table_name=mock.MagicMock(return_value="table1"),
                         SparseEmbedding=mock.MagicMock(return_value=MockSparseEmbedding()))
    def test_create_table(self):
        """
        case: test create_table
        """

        from mx_rec.core.embedding import create_table

        test_table = create_table(key_dtype=tf.int64,
                                  dim=tf.TensorShape([8]),
                                  name='test_table',
                                  emb_initializer=tf.compat.v1.truncated_normal_initializer())
        self.assertIsInstance(test_table, MockSparseEmbedding)


class TestSparseEmbeddingClass(unittest.TestCase):
    """
    Test for 'mx_rec.core.embedding.SparseEmbedding'.
    """

    def setUp(self):
        key_dtype = tf.int64
        dim = 8
        name = 'test_table'
        emb_initializer = tf.compat.v1.truncated_normal_initializer()
        optimizer_list = [MockOptimizer()]
        device_vocabulary_size = 1
        host_vocabulary_size = 2
        ssd_vocabulary_size = 3
        ssd_data_path = (os.getcwd(),)
        is_save = True
        init_param = 1.
        all2all_gradients_op = All2allGradientsOp.SUM_GRADIENTS.value

        self.config = dict(key_dtype=key_dtype, embedding_size=dim, table_name=name, emb_initializer=emb_initializer,
                           device_vocabulary_size=device_vocabulary_size, host_vocabulary_size=host_vocabulary_size,
                           ssd_vocabulary_size=ssd_vocabulary_size, ssd_data_path=ssd_data_path,
                           optimizer_list=optimizer_list, init_param=init_param, is_save=is_save,
                           all2all_gradients_op=all2all_gradients_op)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None))
    def test_init(self):
        """
        case: test create SparseEmbedding

        """

        with tf.Graph().as_default():
            test_sparse_emb = SparseEmbedding(self.config)
            self.assertIsInstance(test_sparse_emb, SparseEmbedding)


class TestGenerateLookupIdNotifyHybridFuncOfSparseEmbeddingClass(unittest.TestCase):
    """
    Test for 'mx_rec.core.embedding.SparseEmbedding.generate_lookup_id_notify_hybrid'.
    """

    def test_generate_lookup_id_notify_hybrid(self):
        """
        case: test generate_lookup_id_notify_hybrid
        """

        with tf.Graph().as_default():
            self.assertEqual(SparseEmbedding.generate_lookup_id_notify_hybrid(0).name, "d2h_notify_hybridmgmt_0")


class TestGetAnchorAttributeFuncOfSparseEmbeddingClass(unittest.TestCase):
    """
    Test for 'mx_rec.core.embedding.SparseEmbedding.get_anchor_attribute'.
    """

    def test_get_anchor_attribute_case1(self):
        """
        case1: 功能正常
        """

        with tf.Graph().as_default():
            anchor_ids = tf.constant(1, dtype=tf.int64)
            SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.IS_TRAINING] = True
            self.assertTrue(SparseEmbedding.get_anchor_attribute(anchor_ids, ASCAnchorAttr.IS_TRAINING))

    def test_get_anchor_attribute_case2(self):
        """
        case2: anchor_ids不是tensor，抛出异常
        """

        with tf.Graph().as_default():
            anchor_ids = 1
            SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.IS_TRAINING] = True
            with self.assertRaises(TypeError):
                SparseEmbedding.get_anchor_attribute(anchor_ids, ASCAnchorAttr.IS_TRAINING)

    def test_get_anchor_attribute_case3(self):
        """
        case3: attr不是ASCAnchorAttr，抛出异常
        """

        with tf.Graph().as_default():
            anchor_ids = tf.constant(1, dtype=tf.int64)
            SparseEmbedding.anchor_tensor_specs[anchor_ids]["xxx"] = True
            with self.assertRaises(ValueError):
                SparseEmbedding.get_anchor_attribute(anchor_ids, "xxx")

    def test_get_anchor_attribute_case4(self):
        """
        case4: 没有set直接get，抛出异常
        """

        with tf.Graph().as_default():
            anchor_ids = tf.constant(1, dtype=tf.int64)
            with self.assertRaises(KeyError):
                SparseEmbedding.get_anchor_attribute(anchor_ids, ASCAnchorAttr.IS_TRAINING)


class TestGetOwnEmbFuncOfSparseEmbeddingClass(unittest.TestCase):
    """
    Test for 'mx_rec.core.embedding.SparseEmbedding._get_own_emb'.
    """

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_rank_size=mock.MagicMock(return_value=1),
                         get_rank_id=mock.MagicMock(return_value=0),
                         hccl_ops=MockHcclOps())
    def test_get_own_emb_case1(self):
        """
        case1: rank=1，静态shape
        """

        with tf.Graph().as_default():
            src_emb = tf.constant([2, 1], dtype=tf.float32, name="src_emb")
            all2all_args = 2
            emb_size = 1
            use_static = True

            # reshape_info为[2, 1]
            own_emb = SparseEmbedding._get_own_emb(src_emb, all2all_args, emb_size, use_static)
            self.assertListEqual(own_emb.shape.as_list(), [2, 1])

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_rank_size=mock.MagicMock(return_value=8),
                         get_rank_id=mock.MagicMock(return_value=0),
                         hccl_ops=MockHcclOps(shape=[2 * 8, 1]))
    def test_get_own_emb_case2(self):
        """
        case2: rank=8，静态shape
        """

        with tf.Graph().as_default():
            src_emb = tf.constant([2, 1], dtype=tf.float32, name="src_emb")
            all2all_args = 2
            emb_size = 1
            use_static = True
            mock_shape = [all2all_args * 8, emb_size]

            own_emb = SparseEmbedding._get_own_emb(src_emb, all2all_args, emb_size, use_static)
            self.assertListEqual(own_emb.shape.as_list(), mock_shape)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_rank_size=mock.MagicMock(return_value=8),
                         get_rank_id=mock.MagicMock(return_value=0),
                         hccl_ops=MockHcclOps(shape=[2 * 8, 1]))
    def test_get_own_emb_case3(self):
        """
        case3: rank=8，动态shape
        """

        with tf.Graph().as_default():
            src_emb = tf.constant([2, 1], dtype=tf.float32, name="src_emb")
            all2all_args = 2
            emb_size = 1
            use_static = False
            mock_shape = [all2all_args * 8, emb_size]

            own_emb = SparseEmbedding._get_own_emb(src_emb, all2all_args, emb_size, use_static)
            self.assertListEqual(own_emb.shape.as_list(), mock_shape)


class TestSizeFuncOfSparseEmbeddingClass(TestSparseEmbeddingClass):
    """
    Test for 'mx_rec.core.embedding.SparseEmbedding.size'.
    """

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_asc_manager=mock.MagicMock(return_value=MockAscManager()))
    def test_size(self):
        """
        case: test size
        """

        with tf.Graph().as_default():
            test_sparse_emb = SparseEmbedding(self.config)
            self.assertEqual(test_sparse_emb.size(), 0)


class TestCapacityFuncOfSparseEmbeddingClass(TestSparseEmbeddingClass):
    """
    Test for 'mx_rec.core.embedding.SparseEmbedding.capacity'.
    """

    def tearDown(self):
        self.config["device_vocabulary_size"] = 1
        self.config["host_vocabulary_size"] = 2
        self.config["ssd_vocabulary_size"] = 3

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=True),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_asc_manager=mock.MagicMock(return_value=MockAscManager()))
    def test_capacity_case1(self):
        """
        case1: 开启动态扩容，HBM

        """

        with tf.Graph().as_default():
            self.config["host_vocabulary_size"] = 0
            self.config["ssd_vocabulary_size"] = 0
            test_sparse_emb = SparseEmbedding(self.config)
            self.assertEqual(test_sparse_emb.capacity(), 1)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None))
    def test_capacity_case2(self):
        """
        case2: 关闭动态扩容，HBM
        """

        with tf.Graph().as_default():
            self.config["host_vocabulary_size"] = 0
            self.config["ssd_vocabulary_size"] = 0
            test_sparse_emb = SparseEmbedding(self.config)
            self.assertEqual(test_sparse_emb.capacity(), 1)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None))
    def test_capacity_case3(self):
        """
        case3: 关闭动态扩容，DDR
        """

        with tf.Graph().as_default():
            self.config["ssd_vocabulary_size"] = 0
            test_sparse_emb = SparseEmbedding(self.config)
            self.assertEqual(test_sparse_emb.capacity(), 3)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None))
    def test_capacity_case4(self):
        """
        case4: 关闭动态扩容，SSD
        """

        with tf.Graph().as_default():
            test_sparse_emb = SparseEmbedding(self.config)
            self.assertEqual(test_sparse_emb.capacity(), 6)


class TestGetDefaultLookupNameFuncOfSparseEmbeddingClass(TestSparseEmbeddingClass):
    """
    Test for 'mx_rec.core.embedding.SparseEmbedding.get_default_lookup_name'.
    """

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None))
    def test_get_default_lookup_name(self):
        """
        case: test get_default_lookup_name
        """

        with tf.Graph().as_default():
            test_sparse_emb = SparseEmbedding(self.config)
            self.assertEqual(test_sparse_emb.get_default_lookup_name(), "sparse_lookup_0")


class TestLookupForAscFuncOfSparseEmbeddingClass(TestSparseEmbeddingClass):
    """
    Test for 'mx_rec.core.embedding.SparseEmbedding.lookup_for_asc'.
    """

    def tearDown(self):
        self.config["device_vocabulary_size"] = 1
        self.config["host_vocabulary_size"] = 2
        self.config["ssd_vocabulary_size"] = 3

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_use_static=mock.MagicMock(return_value=True))
    @mock.patch.multiple("mx_rec.core.embedding.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value=None))
    def test_lookup_for_asc_case1(self):
        """
        case1: test lookup_for_asc，静态shape
        """

        with tf.Graph().as_default():
            def _mock_lookup_for_asc_with_feature_spec_inner(feature_spec, send_count, **kwargs):
                return tf.constant(1, dtype=tf.int64)

            self.config["device_vocabulary_size"] = 100 * 8
            self.config["host_vocabulary_size"] = 100 * 8
            test_sparse_emb = SparseEmbedding(self.config)
            test_sparse_emb.lookup_for_asc_with_feature_spec_inner = _mock_lookup_for_asc_with_feature_spec_inner
            ids = tf.ones(shape=[2, 1], dtype=tf.int64, name="ids")
            send_count = 1
            kwargs = {"is_train": True}

            lookup_res = test_sparse_emb.lookup_for_asc(ids, send_count, **kwargs)
            with tf.Session() as sess:
                self.assertEqual(sess.run(lookup_res), 1)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=True),
                         get_name_to_var_dict=mock.MagicMock(return_value={"test_table": 1}),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_training_mode_channel_id=mock.MagicMock(return_value=None),
                         clear_channel=mock.MagicMock(return_value=None),
                         get_use_static=mock.MagicMock(return_value=False))
    @mock.patch.multiple("mx_rec.core.embedding.FeatureSpec",
                         set_feat_attribute=mock.MagicMock(return_value=None))
    def test_lookup_for_asc_case2(self):
        """
        case2: test lookup_for_asc，动态shape，is_training=False
        """

        with tf.Graph().as_default():
            def _mock_lookup_for_asc_with_feature_spec_inner(feature_spec, send_count, **kwargs):
                return tf.constant(1, dtype=tf.int64)

            self.config["device_vocabulary_size"] = 100 * 8
            self.config["host_vocabulary_size"] = 100 * 8
            test_sparse_emb = SparseEmbedding(self.config)
            test_sparse_emb.lookup_for_asc_with_feature_spec_inner = _mock_lookup_for_asc_with_feature_spec_inner
            ids = tf.ones(shape=[2, 1], dtype=tf.int64, name="ids")
            send_count = 1
            kwargs = {"is_train": False}

            lookup_res = test_sparse_emb.lookup_for_asc(ids, send_count, **kwargs)
            with tf.Session() as sess:
                self.assertEqual(sess.run(lookup_res), 1)


class TestLookupForAscWithFeatureSpecFuncOfSparseEmbeddingClass(TestSparseEmbeddingClass):
    """
    Test for 'mx_rec.core.embedding.SparseEmbedding.lookup_for_asc_with_feature_spec'.
    """

    def tearDown(self):
        self.config["device_vocabulary_size"] = 1
        self.config["host_vocabulary_size"] = 2
        self.config["ssd_vocabulary_size"] = 3

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_training_mode_channel_id=mock.MagicMock(return_value=None),
                         get_use_static=mock.MagicMock(return_value=True))
    @mock.patch("mx_rec.core.embedding.get_table_name_to_feature_spec")
    def test_lookup_for_asc_with_feature_spec_case1(self, mock_get_table_name_to_feature_spec):
        """
        case1: test lookup_for_asc_with_feature_spec，静态shape，len(same_table_feature_spec)=1
        """

        with tf.Graph().as_default():
            def _mock_lookup_for_asc_with_feature_spec_inner(feature_spec, send_count, **kwargs):
                return tf.constant(1, dtype=tf.int64)

            self.config["device_vocabulary_size"] = 100 * 8
            self.config["host_vocabulary_size"] = 100 * 8
            test_sparse_emb = SparseEmbedding(self.config)
            test_sparse_emb.lookup_for_asc_with_feature_spec_inner = _mock_lookup_for_asc_with_feature_spec_inner
            case1_feat = FeatureSpec("case1_feat", table_name="test_table")
            set_temporary_feature_spec_attribute(case1_feat, 1)
            mock_get_table_name_to_feature_spec.return_value = [case1_feat]
            send_count = 1
            kwargs = {"is_train": True}

            lookup_res = test_sparse_emb.lookup_for_asc_with_feature_spec(case1_feat, send_count, **kwargs)
            with tf.Session() as sess:
                self.assertEqual(sess.run(lookup_res), 1)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_training_mode_channel_id=mock.MagicMock(return_value=None),
                         get_use_static=mock.MagicMock(return_value=True))
    @mock.patch("mx_rec.core.embedding.get_table_name_to_feature_spec")
    def test_lookup_for_asc_with_feature_spec_case2(self, mock_get_table_name_to_feature_spec):
        """
        case2: test lookup_for_asc_with_feature_spec，静态shape，len(same_table_feature_spec)=0，抛出异常
        """

        with tf.Graph().as_default():
            def _mock_lookup_for_asc_with_feature_spec_inner(feature_spec, send_count, **kwargs):
                return tf.constant(1, dtype=tf.int64)

            self.config["device_vocabulary_size"] = 100 * 8
            self.config["host_vocabulary_size"] = 100 * 8
            test_sparse_emb = SparseEmbedding(self.config)
            test_sparse_emb.lookup_for_asc_with_feature_spec_inner = _mock_lookup_for_asc_with_feature_spec_inner
            case2_feat = FeatureSpec("case2_feat", table_name="test_table")
            set_temporary_feature_spec_attribute(case2_feat, 1)
            mock_get_table_name_to_feature_spec.return_value = []
            send_count = 1
            kwargs = {"is_train": True}

            with self.assertRaises(RuntimeError):
                test_sparse_emb.lookup_for_asc_with_feature_spec(case2_feat, send_count, **kwargs)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_training_mode_channel_id=mock.MagicMock(return_value=None),
                         clear_same_table_feature_spec=mock.MagicMock(return_value=None),
                         get_use_static=mock.MagicMock(return_value=True))
    @mock.patch("mx_rec.core.embedding.get_table_name_to_feature_spec")
    def test_lookup_for_asc_with_feature_spec_case3(self, mock_get_table_name_to_feature_spec):
        """
        case3: test lookup_for_asc_with_feature_spec，静态shape，len(same_table_feature_spec)>1
        """

        with tf.Graph().as_default():
            def _mock_lookup_for_asc_with_feature_spec_inner(feature_spec, send_count, **kwargs):
                return tf.ones(shape=[16, ], dtype=tf.int64)

            self.config["device_vocabulary_size"] = 100 * 8
            self.config["host_vocabulary_size"] = 100 * 8
            test_sparse_emb = SparseEmbedding(self.config)
            test_sparse_emb.lookup_for_asc_with_feature_spec_inner = _mock_lookup_for_asc_with_feature_spec_inner
            case3_feat = FeatureSpec("case3_feat", table_name="test_table")
            case3_feat_multi = FeatureSpec("case3_feat_multi", table_name="test_table")
            set_temporary_feature_spec_attribute(case3_feat, 1)
            set_temporary_feature_spec_attribute(case3_feat_multi, 1)
            case3_feat.split = 8
            case3_feat_multi.split = 8
            mock_get_table_name_to_feature_spec.return_value = [case3_feat, case3_feat_multi]
            send_count = 1
            kwargs = {"is_train": True}

            test_sparse_emb.lookup_for_asc_with_feature_spec(case3_feat, send_count, **kwargs)
            self.assertGreater(len(test_sparse_emb.lookup_result), 0)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_training_mode_channel_id=mock.MagicMock(return_value=None),
                         clear_same_table_feature_spec=mock.MagicMock(return_value=None),
                         get_use_static=mock.MagicMock(return_value=False))
    @mock.patch("mx_rec.core.embedding.get_table_name_to_feature_spec")
    def test_lookup_for_asc_with_feature_spec_case4(self, mock_get_table_name_to_feature_spec):
        """
        case4: test lookup_for_asc_with_feature_spec，动态shape，len(same_table_feature_spec)>1
        """

        with tf.Graph().as_default():
            def _mock_lookup_for_asc_with_feature_spec_inner(feature_spec, send_count, **kwargs):
                return tf.ones(shape=[16, ], dtype=tf.int64)

            self.config["device_vocabulary_size"] = 100 * 8
            self.config["host_vocabulary_size"] = 100 * 8
            test_sparse_emb = SparseEmbedding(self.config)
            test_sparse_emb.lookup_for_asc_with_feature_spec_inner = _mock_lookup_for_asc_with_feature_spec_inner
            case4_feat = FeatureSpec("case4_feat", table_name="test_table")
            case4_feat_multi = FeatureSpec("case4_feat_multi", table_name="test_table")
            set_temporary_feature_spec_attribute(case4_feat, 1)
            set_temporary_feature_spec_attribute(case4_feat_multi, 1)
            case4_feat.split = 8
            case4_feat_multi.split = 8
            mock_get_table_name_to_feature_spec.return_value = [case4_feat, case4_feat_multi]
            send_count = 1
            kwargs = {
                "is_train": True,
                "batch": {
                    "case4_feat": tf.ones(shape=[8, ], dtype=tf.int64),
                    "case4_feat_multi": tf.ones(shape=[8, ], dtype=tf.int64)
                }
            }

            test_sparse_emb.emb_size = 1
            test_sparse_emb.lookup_for_asc_with_feature_spec(case4_feat, send_count, **kwargs)
            self.assertGreater(len(test_sparse_emb.lookup_result), 0)


class TestLookupForAscWithFeatureSpecInnerFuncOfSparseEmbeddingClass(TestSparseEmbeddingClass):
    """
    Test for 'mx_rec.core.embedding.SparseEmbedding.lookup_for_asc_with_feature_spec_inner'.
    """

    def tearDown(self):
        self.config["device_vocabulary_size"] = 1
        self.config["host_vocabulary_size"] = 2
        self.config["ssd_vocabulary_size"] = 3

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         get_device_id=mock.MagicMock(return_value=0),
                         get_use_hot=mock.MagicMock(return_value=1),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_training_mode_channel_id=mock.MagicMock(return_value=None),
                         get_use_static=mock.MagicMock(return_value=True))
    @mock.patch("mx_rec.core.embedding.get_preprocessed_tensor_for_asc")
    def test_lookup_for_asc_with_feature_spec_inner_case1(self, mock_get_preprocessed_tensor_for_asc):
        """
        case1: test lookup_for_asc_with_feature_spec_inner，静态shape，关闭动态扩容
        """

        with tf.Graph().as_default():
            mock_get_preprocessed_tensor_for_asc.return_value = {
                "restore_vector": tf.ones(shape=[8, 8], dtype=tf.int64),
                "restore_vector_second": tf.ones(shape=[8, ], dtype=tf.int64),
                "unique_keys": tf.ones(shape=[8, ], dtype=tf.int64),
                "hot_pos": tf.ones(shape=[8, ], dtype=tf.int64),
                "id_offsets": tf.ones(shape=[8, ], dtype=tf.int64),
                "all2all_args": tf.ones(shape=[8, 8], dtype=tf.int64),
                "swap_in": [tf.no_op()]
            }

            self.config["device_vocabulary_size"] = 100 * 8
            self.config["host_vocabulary_size"] = 0
            test_sparse_emb = SparseEmbedding(self.config)
            case1_feat = FeatureSpec("case1_feat", table_name="test_table")
            set_temporary_feature_spec_attribute(case1_feat, 1)
            case1_feat.dims = [8, 8]
            send_count = 1
            kwargs = {"is_train": True}

            def _mock_get_own_emb(emb, all2all_args, emb_size, use_static):
                return test_sparse_emb.variable

            test_sparse_emb._get_own_emb = _mock_get_own_emb

            lookup_res = test_sparse_emb.lookup_for_asc_with_feature_spec_inner(case1_feat, send_count, **kwargs)
            self.assertIsInstance(lookup_res, tf.Tensor)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         get_device_id=mock.MagicMock(return_value=0),
                         get_use_hot=mock.MagicMock(return_value=1),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_training_mode_channel_id=mock.MagicMock(return_value=None),
                         get_use_static=mock.MagicMock(return_value=False))
    @mock.patch("mx_rec.core.embedding.get_preprocessed_tensor_for_asc")
    def test_lookup_for_asc_with_feature_spec_inner_case2(self, mock_get_preprocessed_tensor_for_asc):
        """
        case2: test lookup_for_asc_with_feature_spec_inner，动态shape，关闭动态扩容
        """

        with tf.Graph().as_default():
            mock_get_preprocessed_tensor_for_asc.return_value = {
                "restore_vector": tf.ones(shape=[8, 8], dtype=tf.int64),
                "restore_vector_second": tf.ones(shape=[8, ], dtype=tf.int64),
                "unique_keys": tf.ones(shape=[8, ], dtype=tf.int64),
                "hot_pos": tf.ones(shape=[8, ], dtype=tf.int64),
                "id_offsets": tf.ones(shape=[8, ], dtype=tf.int64),
                "all2all_args": tf.ones(shape=[8, 8], dtype=tf.int64),
                "swap_in": [tf.no_op()]
            }

            self.config["device_vocabulary_size"] = 100 * 8
            self.config["host_vocabulary_size"] = 0
            test_sparse_emb = SparseEmbedding(self.config)
            case2_feat = FeatureSpec("case2_feat", table_name="test_table")
            set_temporary_feature_spec_attribute(case2_feat, 1)
            case2_feat.dims = [8, 8]
            send_count = 1
            kwargs = {"is_train": True, "batch": {"case2_feat": tf.ones(shape=[8, 8], dtype=tf.int64)}}

            def _mock_get_own_emb(emb, all2all_args, emb_size, use_static):
                return test_sparse_emb.variable

            test_sparse_emb._get_own_emb = _mock_get_own_emb

            lookup_res = test_sparse_emb.lookup_for_asc_with_feature_spec_inner(case2_feat, send_count, **kwargs)
            self.assertIsInstance(lookup_res, tf.Tensor)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         get_device_id=mock.MagicMock(return_value=0),
                         get_use_hot=mock.MagicMock(return_value=1),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None),
                         get_training_mode_channel_id=mock.MagicMock(return_value=None),
                         get_use_static=mock.MagicMock(return_value=True))
    @mock.patch("mx_rec.core.embedding.get_preprocessed_tensor_for_asc")
    def test_lookup_for_asc_with_feature_spec_inner_case3(self, mock_get_preprocessed_tensor_for_asc):
        """
        case3: test lookup_for_asc_with_feature_spec_inner，静态shape，关闭动态扩容
            access_threshold > 0，覆盖 set_specific_value_for_non_valid_key()
        """

        with tf.Graph().as_default():
            mock_get_preprocessed_tensor_for_asc.return_value = {
                "restore_vector": tf.ones(shape=[8, 8], dtype=tf.int64),
                "restore_vector_second": tf.ones(shape=[8, ], dtype=tf.int64),
                "unique_keys": tf.ones(shape=[8, ], dtype=tf.int64),
                "hot_pos": tf.ones(shape=[8, ], dtype=tf.int64),
                "id_offsets": tf.ones(shape=[8, ], dtype=tf.int64),
                "all2all_args": tf.ones(shape=[8, 8], dtype=tf.int64),
                "swap_in": [tf.no_op()]
            }

            self.config["device_vocabulary_size"] = 100 * 8
            self.config["host_vocabulary_size"] = 0
            test_sparse_emb = SparseEmbedding(self.config)
            case3_feat = FeatureSpec("case3_feat", table_name="test_table", access_threshold=10)
            set_temporary_feature_spec_attribute(case3_feat, 1)
            case3_feat.dims = [8, 8]
            send_count = 1
            kwargs = {"is_train": True}

            def _mock_get_own_emb(emb, all2all_args, emb_size, use_static):
                return test_sparse_emb.variable

            test_sparse_emb._get_own_emb = _mock_get_own_emb

            lookup_res = test_sparse_emb.lookup_for_asc_with_feature_spec_inner(case3_feat, send_count, **kwargs)
            self.assertIsInstance(lookup_res, tf.Tensor)


class TestSparseLookupFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.embedding.sparse_lookup'.
    """

    def setUp(self):
        key_dtype = tf.int64
        dim = 8
        name = 'test_table'
        emb_initializer = tf.compat.v1.truncated_normal_initializer()
        optimizer_list = [MockOptimizer()]
        device_vocabulary_size = 1
        host_vocabulary_size = 2
        ssd_vocabulary_size = 3
        ssd_data_path = (os.getcwd(),)
        is_save = True
        init_param = 1.
        all2all_gradients_op = All2allGradientsOp.SUM_GRADIENTS.value

        self.config = dict(key_dtype=key_dtype, embedding_size=dim, table_name=name, emb_initializer=emb_initializer,
                           device_vocabulary_size=device_vocabulary_size, host_vocabulary_size=host_vocabulary_size,
                           ssd_vocabulary_size=ssd_vocabulary_size, ssd_data_path=ssd_data_path,
                           optimizer_list=optimizer_list, init_param=init_param, is_save=is_save,
                           all2all_gradients_op=all2all_gradients_op)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None))
    def test_sparse_lookup_case1(self):
        """
        case1: test sparse_lookup，FeatureSpec模式
        """

        from mx_rec.core.embedding import sparse_lookup

        def _mock_lookup_for_asc_with_feature_spec(ids, send_count, **kwargs):
            return 0

        with tf.Graph().as_default():
            case1_feat = FeatureSpec("case1_feat", table_name="test_table")
            test_sparse_emb = SparseEmbedding(self.config)
            test_sparse_emb.lookup_for_asc_with_feature_spec = _mock_lookup_for_asc_with_feature_spec

            self.assertEqual(sparse_lookup(test_sparse_emb, case1_feat), 0)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         set_modify_graph=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None))
    def test_sparse_lookup_case2(self):
        """
        case2: test sparse_lookup，自动改图模式
        """

        from mx_rec.core.embedding import sparse_lookup

        def _mock_lookup_for_asc(ids, send_count, **kwargs):
            return 1

        with tf.Graph().as_default():
            ids = tf.constant(1, tf.int64)
            test_sparse_emb = SparseEmbedding(self.config)
            test_sparse_emb.lookup_for_asc = _mock_lookup_for_asc

            self.assertEqual(sparse_lookup(test_sparse_emb, ids, modify_graph=True), 1)

    @mock.patch.multiple("mx_rec.core.embedding",
                         get_use_dynamic_expansion=mock.MagicMock(return_value=False),
                         is_asc_frozen=mock.MagicMock(return_value=False),
                         get_name_to_var_dict=mock.MagicMock(return_value=None),
                         get_ascend_global_hashtable_collection=mock.MagicMock(return_value="xxx"),
                         get_rank_size=mock.MagicMock(return_value=8),
                         insert_removing_var_list=mock.MagicMock(return_value=None),
                         set_modify_graph=mock.MagicMock(return_value=None),
                         insert_table_instance=mock.MagicMock(return_value=None))
    def test_sparse_lookup_case3(self):
        """
        case3: test sparse_lookup，自动改图模式，没传入modify_graph参数，抛出异常
        """

        from mx_rec.core.embedding import sparse_lookup

        with tf.Graph().as_default():
            ids = tf.constant(1, tf.int64)
            test_sparse_emb = SparseEmbedding(self.config)

            with self.assertRaises(ValueError):
                sparse_lookup(test_sparse_emb, ids)


if __name__ == '__main__':
    unittest.main()
