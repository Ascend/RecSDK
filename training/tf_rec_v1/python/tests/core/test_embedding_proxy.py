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
from unittest.mock import MagicMock, patch

import tensorflow as tf

from mx_rec.constants import constants
from mx_rec.core import embedding_proxy
from mx_rec.core.emb import mergeable_sparse_embedding
from core import mock_class


class CreateMergeableEmbeddingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._config = dict(
            key_dtype=tf.int64,
            embedding_size=tf.TensorShape([8]),
            table_name="test_small_table_name",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            device_vocabulary_size=1,
            host_vocabulary_size=0,
            ssd_vocabulary_size=0,
            ssd_data_path="./",
            init_param=1.0,
            is_save=False,
            all2all_gradients_op=constants.All2allGradientsOp.SUM_GRADIENTS.value,
            is_dp=False,
        )

        self._union_key = embedding_proxy.UnionKey(
            key_dtype=tf.int64,
            emb_dim=8,
            initializer_type=type(tf.compat.v1.truncated_normal_initializer()),
            is_save=False,
            is_dp=False,
            init_param=1.0,
            all2all_gradients_op=constants.All2allGradientsOp.SUM_GRADIENTS.value,
            padding_keys_mask=True,
        )

    def tearDown(self) -> None:
        embedding_proxy.MergeableEmbeddingTableProxy().reset()

    @patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=MagicMock(return_value=8),
        get_rank_id=MagicMock(return_value=0),
        get_device_id=MagicMock(return_value=0),
    )
    @patch("mx_rec.core.embedding.ConfigInitializer")
    @patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer")
    @patch("mx_rec.validator.emb_validator.ConfigInitializer")
    @patch("mx_rec.core.util.ConfigInitializer")
    def test_ok(
        self,
        util_config_init,
        emb_validator_config_init,
        base_sparse_embedding_config_init,
        embedding_config_init,
    ):
        mock_config_init = mock_class.MockConfigInitializer(use_dynamic_expansion=True)

        util_config_init.get_instance = MagicMock(return_value=mock_config_init)
        emb_validator_config_init.get_instance = MagicMock(return_value=mock_config_init)
        base_sparse_embedding_config_init.get_instance = MagicMock(return_value=mock_config_init)
        embedding_config_init.get_instance = MagicMock(return_value=mock_config_init)

        with tf.compat.v1.Graph().as_default():
            table_name = self._config["table_name"]
            mergeable_table = embedding_proxy.create_mergeable_embedding(table_name, self._config, self._union_key)

        self.assertIsInstance(mergeable_table, mergeable_sparse_embedding.MergeableSparseEmbedding)
        self.assertIn("mergeable_table", mergeable_table.table_name)
        self.assertEqual(mergeable_table.merged_small_tables, ["test_small_table_name"])
        self.assertEqual(mergeable_table.slice_device_vocabulary_size, 1)
