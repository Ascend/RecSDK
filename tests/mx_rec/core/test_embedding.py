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

from mx_rec.constants.constants import ASCAnchorAttr
from mx_rec.core.asc import FeatureSpec
from mx_rec.core.asc.feature_spec import set_temporary_feature_spec_attribute
from mx_rec.core.emb.dynamic_sparse_embedding import HBMDynamicSparseEmbedding
from mx_rec.core.emb.mergeable_sparse_embedding import MergeableSparseEmbedding
from mx_rec.core.emb.sparse_embedding import HBMSparseEmbedding, ExternalStorageSparseEmbedding
from tests.mx_rec.core.mock_class import MockConfigInitializer


class TestCreateTableFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.embedding.create_table'.
    """

    _mock_static_config_init = MockConfigInitializer(use_static=True)
    _mock_dynamic_expansion_config_init = MockConfigInitializer(use_static=True, use_dynamic_expansion=True)

    def setUp(self):
        tf.compat.v1.reset_default_graph()

    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=8),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer")
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer")
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    @mock.patch("mx_rec.core.util.ConfigInitializer")
    def test_create_table_case1(
        self,
        embedding_config_initializer,
        base_sparse_embedding_config_initializer,
        emb_validator_config_initializer,
        util_config_initializer,
    ):
        """
        case1: test create_table, 动态扩容
        """

        from mx_rec.core.embedding import create_table

        with tf.Graph().as_default():
            # mock
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
            embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            base_sparse_embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            emb_validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            util_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            # test
            test_table = create_table(
                key_dtype=tf.int64,
                dim=8,
                name="test_table",
                emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            )
            self.assertIsInstance(test_table, HBMDynamicSparseEmbedding)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=8),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer")
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer")
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    @mock.patch("mx_rec.core.util.ConfigInitializer")
    def test_create_table_case2(
        self,
        embedding_config_initializer,
        base_sparse_embedding_config_initializer,
        emb_validator_config_initializer,
        util_config_initializer,
    ):
        """
        case2: test create_table, 非动态扩容，HBM
        """

        from mx_rec.core.embedding import create_table

        with tf.Graph().as_default():
            # mock
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)
            embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            base_sparse_embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            emb_validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            util_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            # test
            test_table = create_table(
                key_dtype=tf.int64,
                dim=8,
                name="test_table",
                emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                device_vocabulary_size=8,
            )
            self.assertIsInstance(test_table, HBMSparseEmbedding)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=8),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer")
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer")
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    @mock.patch("mx_rec.optimizers.gradient_descent.ConfigInitializer")
    @mock.patch("mx_rec.core.util.ConfigInitializer")
    def test_create_table_case3(
        self,
        embedding_config_initializer,
        base_sparse_embedding_config_initializer,
        emb_validator_config_initializer,
        lazy_adam_config_initializer,
        util_config_initializer,
    ):
        """
        case3: test create_table, 非动态扩容，DDR/SSD
        """

        from mx_rec.core.embedding import create_table

        with tf.Graph().as_default():
            # mock
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)
            embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            base_sparse_embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            emb_validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            lazy_adam_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            util_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            # test
            test_table = create_table(
                key_dtype=tf.int64,
                dim=8,
                name="test_table",
                emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                device_vocabulary_size=8,
                host_vocabulary_size=8,
            )
            self.assertIsInstance(test_table, ExternalStorageSparseEmbedding)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=8),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.util.ConfigInitializer")
    @mock.patch("mx_rec.core.embedding.ConfigInitializer")
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer")
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_create_table_with_padding_keys(
        self,
        util_config_initializer,
        embedding_config_initializer,
        base_sparse_embedding_config_initializer,
        emb_validator_config_initializer,
    ):
        from mx_rec.core.embedding import create_table

        with tf.Graph().as_default():
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=True)
            mock_config_initializer.use_static = True
            util_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            base_sparse_embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            emb_validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            test_table = create_table(
                key_dtype=tf.int64,
                dim=8,
                name="test_table",
                emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                padding_keys=666,
                padding_keys_len=4096,
                padding_keys_mask=True,
            )
            self.assertIsInstance(test_table, HBMDynamicSparseEmbedding)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.optimizers.gradient_descent.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_static_config_init)
    def test_emb_init_ok(self):
        from mx_rec.core.embedding import create_table

        test_table = create_table(
            key_dtype=tf.int64,
            dim=8,
            name="test_table",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            device_vocabulary_size=8,
            host_vocabulary_size=16,
        )
        self.assertIsInstance(test_table, ExternalStorageSparseEmbedding)
        self.assertIsNotNone(test_table)
        self.assertEqual(test_table.emb_size, 8)
        self.assertIsInstance(test_table.key_dtype, tf.DType)
        self.assertIsNotNone(test_table.emb_initializer)
        self.assertTrue(test_table.is_save)
        self.assertEqual(test_table.init_param, 1.0)
        self.assertEqual(test_table.send_count, 0)
        self.assertEqual(test_table.rank_size, 1)
        self.assertEqual(test_table.ssd_vocabulary_size, 0)
        self.assertIsNotNone(test_table.ssd_data_path)
        self.assertEqual(test_table.slice_host_vocabulary_size, 16)
        self.assertEqual(test_table.slice_ssd_vocabulary_size, 0)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.optimizers.gradient_descent.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_static_config_init)
    def test_get_anchor_type_err(self):
        from mx_rec.core.embedding import create_table

        test_table = create_table(
            key_dtype=tf.int64,
            dim=8,
            name="test_table",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            device_vocabulary_size=8,
            host_vocabulary_size=16,
        )
        anchor = tf.constant(0.1)
        test_table.anchor_tensor_specs[anchor][ASCAnchorAttr.TABLE_INSTANCE] = test_table
        with self.assertRaises(TypeError):
            test_table.get_anchor_attribute("xxx", "xxx")

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.optimizers.gradient_descent.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_static_config_init)
    def test_get_anchor_value_err(self):
        from mx_rec.core.embedding import create_table

        test_table = create_table(
            key_dtype=tf.int64,
            dim=8,
            name="test_table",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            device_vocabulary_size=8,
            host_vocabulary_size=16,
        )
        anchor = tf.constant(0.1)
        test_table.anchor_tensor_specs[anchor][ASCAnchorAttr.TABLE_INSTANCE] = test_table
        with self.assertRaises(ValueError):
            test_table.get_anchor_attribute(anchor, "xxx")

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.optimizers.gradient_descent.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_static_config_init)
    def test_get_anchor_key_err(self):
        from mx_rec.core.embedding import create_table

        test_table = create_table(
            key_dtype=tf.int64,
            dim=8,
            name="test_table",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            device_vocabulary_size=8,
            host_vocabulary_size=16,
        )
        anchor = tf.constant(0.1)
        with self.assertRaises(KeyError):
            test_table.get_anchor_attribute(anchor, ASCAnchorAttr.TABLE_INSTANCE)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.optimizers.gradient_descent.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_static_config_init)
    def test_get_anchor_ok(self):
        from mx_rec.core.embedding import create_table

        test_table = create_table(
            key_dtype=tf.int64,
            dim=8,
            name="test_table",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            device_vocabulary_size=8,
            host_vocabulary_size=16,
        )
        anchor = tf.constant(0.1)
        test_table.anchor_tensor_specs[anchor][ASCAnchorAttr.TABLE_INSTANCE] = test_table
        res = test_table.get_anchor_attribute(anchor, ASCAnchorAttr.TABLE_INSTANCE)
        self.assertIsNotNone(res)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.optimizers.gradient_descent.ConfigInitializer", new=_mock_static_config_init)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_static_config_init)
    def test_emb_is_dp(self):
        from mx_rec.core.embedding import create_table

        test_table = create_table(
            key_dtype=tf.int64,
            dim=8,
            name="test_table",
            is_dp=True,
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            device_vocabulary_size=8,
            host_vocabulary_size=16,
        )
        self.assertIsInstance(test_table, ExternalStorageSparseEmbedding)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_dynamic_expansion_config_init)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_dynamic_expansion_config_init)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_dynamic_expansion_config_init)
    @mock.patch("mx_rec.optimizers.gradient_descent.ConfigInitializer", new=_mock_dynamic_expansion_config_init)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_dynamic_expansion_config_init)
    def test_emb_merge_is_true(self):
        from mx_rec.core.embedding import create_table

        test_table = create_table(
            key_dtype=tf.int64,
            dim=tf.TensorShape([8]),
            name="test_table",
            is_dp=True,
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            enable_merge=True,
            padding_keys=[8],
            padding_keys_mask=True,
            padding_keys_len=128,
        )
        test_table2 = create_table(
            key_dtype=tf.int64,
            dim=tf.TensorShape([8]),
            name="test_table2",
            is_dp=True,
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            enable_merge=True,
            padding_keys=[8],
            padding_keys_mask=True,
            padding_keys_len=128,
        )
        test_table.__str__()
        test_table.__repr__()
        self.assertIsInstance(test_table, MergeableSparseEmbedding)
        self.assertIsInstance(test_table2, MergeableSparseEmbedding)


class TestSparseLookupFunc(unittest.TestCase):
    """
    Test for 'mx_rec.core.embedding.sparse_lookup'.
    """

    _mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False, use_static=True)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=8),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer")
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.get_preprocessed_tensor_for_asc")
    @mock.patch("mx_rec.core.embedding.ConfigInitializer")
    @mock.patch("mx_rec.core.util.ConfigInitializer")
    def test_sparse_lookup_case1(
        self,
        util_config_initializer,
        embedding_config_initializer,
        mock_get_preprocessed_tensor_for_asc,
        emb_validator_config_initializer,
        base_sparse_embedding_config_initializer,
    ):
        """
        case1: test sparse_lookup
            表：非动态扩容，HBM
            ids：FeatureSpec模式，动态shape
        """

        from mx_rec.core.embedding import create_table, sparse_lookup

        with tf.Graph().as_default():
            # mock
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)

            util_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            emb_validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            base_sparse_embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            case1_feat = FeatureSpec("case1_feat", table_name="test_table")
            set_temporary_feature_spec_attribute(case1_feat, 1)
            case1_feat.dims = [8, 8]
            mock_config_initializer.get_instance().feature_spec_config.insert_feature_spec(case1_feat, True)
            batch = {"case1_feat": tf.ones(shape=[8, 8], dtype=tf.int64)}
            mock_get_preprocessed_tensor_for_asc.return_value = {
                "restore_vector": tf.ones(shape=[8, 8], dtype=tf.int64),
                "hot_pos": tf.ones(
                    shape=[
                        8,
                    ],
                    dtype=tf.int64,
                ),
                "id_offsets": tf.ones(
                    shape=[
                        8,
                    ],
                    dtype=tf.int64,
                ),
                "all2all_args": tf.ones(shape=[8, 8], dtype=tf.int64),
            }

            # test
            test_table = create_table(
                key_dtype=tf.int64,
                dim=8,
                name="test_table",
                emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                device_vocabulary_size=100 * 8,
            )
            self.assertIsInstance(test_table, HBMSparseEmbedding)

            res = sparse_lookup(test_table, case1_feat, batch=batch)
            self.assertIsInstance(res, tf.Tensor)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=8),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.asc.feature_spec.ConfigInitializer")
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.get_preprocessed_tensor_for_asc")
    @mock.patch("mx_rec.core.embedding.ConfigInitializer")
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer")
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    @mock.patch("mx_rec.core.util.ConfigInitializer")
    def test_sparse_lookup_case2(
        self,
        util_config_initializer,
        emb_validator_config_initializer,
        base_sparse_embedding_config_initializer,
        embedding_config_config_initializer,
        mock_get_preprocessed_tensor_for_asc,
        feature_spec_config_initializer,
    ):
        """
        case2: test sparse_lookup
            表：非动态扩容，HBM
            ids：自动改图模式，动态shape
        """

        from mx_rec.core.embedding import create_table, sparse_lookup

        with tf.Graph().as_default():
            # mock
            mock_config_initializer = MockConfigInitializer(use_dynamic_expansion=False)

            util_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            emb_validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            base_sparse_embedding_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            embedding_config_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)
            feature_spec_config_initializer.get_instance = mock.Mock(return_value=mock_config_initializer)

            case2_feat = tf.ones(shape=[8, 8], dtype=tf.int64)
            mock_get_preprocessed_tensor_for_asc.return_value = {
                "restore_vector": tf.ones(shape=[8, 8], dtype=tf.int64),
                "hot_pos": tf.ones(
                    shape=[
                        8,
                    ],
                    dtype=tf.int64,
                ),
                "id_offsets": tf.ones(
                    shape=[
                        8,
                    ],
                    dtype=tf.int64,
                ),
                "all2all_args": tf.ones(shape=[8, 8], dtype=tf.int64),
            }

            # test
            test_table = create_table(
                key_dtype=tf.int64,
                dim=8,
                name="test_table",
                emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                device_vocabulary_size=100 * 8,
            )
            self.assertIsInstance(test_table, HBMSparseEmbedding)

            res = sparse_lookup(test_table, case2_feat, modify_graph=True)
            self.assertIsInstance(res, tf.Tensor)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=8),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.asc.feature_spec.ConfigInitializer", new=_mock_config_initializer)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.get_preprocessed_tensor_for_asc")
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_config_initializer)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_config_initializer)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_config_initializer)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_config_initializer)
    def test_sparse_lookup_is_train_false(self, mock_get_preprocessed_tensor_for_asc):
        from mx_rec.core.embedding import create_table, sparse_lookup

        with tf.Graph().as_default():
            case2_feat = tf.ones(shape=[8, 8], dtype=tf.int64)
            mock_get_preprocessed_tensor_for_asc.return_value = {
                "restore_vector": tf.ones(shape=[8, 8], dtype=tf.int64),
                "hot_pos": tf.ones(
                    shape=[
                        8,
                    ],
                    dtype=tf.int64,
                ),
                "id_offsets": tf.ones(
                    shape=[
                        8,
                    ],
                    dtype=tf.int64,
                ),
                "all2all_args": tf.ones(shape=[8, 8], dtype=tf.int64),
            }

            test_table = create_table(
                key_dtype=tf.int64,
                dim=8,
                name="test_table",
                emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                device_vocabulary_size=100 * 8,
            )
            self.assertIsInstance(test_table, HBMSparseEmbedding)

            res = sparse_lookup(test_table, case2_feat, modify_graph=True, is_train=False, send_count=5000)
            self.assertIsInstance(res, tf.Tensor)


if __name__ == "__main__":
    unittest.main()
