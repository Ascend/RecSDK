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
from typing import Union
from unittest import TestCase
from unittest.mock import Mock, patch

import tensorflow as tf
from tensorflow import Tensor
import mx_rec.graph.merge_lookup as merge_lookup
from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_ENTRANCE, ASCAnchorAttr
from tests.mx_rec.core.mock_class import MockConfigInitializer


def mock_get_anchor_attribute(anchor: Tensor, attr: ASCAnchorAttr) -> Union[bool, Mock]:
    if attr == ASCAnchorAttr.IS_TRAINING:
        return True
    if attr == ASCAnchorAttr.IS_GRAD:
        return True
    if attr == ASCAnchorAttr.TABLE_INSTANCE:
        mock_table_instance = Mock()
        mock_table_instance.table_name = "mock_table_name"
        mock_table_instance.multi_lookup_times = {True: 2}
        mock_table_instance.send_count = 4096 * 8
        return mock_table_instance
    if attr == ASCAnchorAttr.FEATURE_SPEC:
        mock_feature_spec = Mock()
        mock_feature_spec.name = "mock_feature_spec_name"
        return mock_feature_spec

    raise ValueError(f"Unsupported param 'attr' for enum class 'ASCAnchorAttr': attr={attr}.")


class DoMergeLookupTest(TestCase):
    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    @patch.multiple(
        "mx_rec.graph.merge_lookup",
        replace_anchor_vec=Mock(),
    )
    @patch.multiple("mx_rec.graph.merge_lookup.BaseSparseEmbedding", get_anchor_attribute=mock_get_anchor_attribute)
    @patch("mx_rec.graph.merge_lookup.ConfigInitializer")
    def test_ok(self, merge_lookup_config_initializer):
        mock_config_initializer = MockConfigInitializer(modify_graph=True, merged_multi_lookup=False, use_static=False)
        merge_lookup_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        mock_cutting_point = tf.identity(tf.zeros(shape=(4096, 8)))
        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE, mock_cutting_point)
        merge_lookup.do_merge_lookup()

    @patch("mx_rec.graph.merge_lookup.ConfigInitializer")
    def test_ok_disable_modify_graph(self, merge_lookup_config_initializer):
        mock_config_initializer = MockConfigInitializer(modify_graph=False)
        merge_lookup_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        merge_lookup.do_merge_lookup()

    @patch("mx_rec.graph.merge_lookup.ConfigInitializer")
    def test_ok_already_exec_merged_lookup(self, merge_lookup_config_initializer):
        mock_config_initializer = MockConfigInitializer(modify_graph=True, merged_multi_lookup=True)
        merge_lookup_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        merge_lookup.do_merge_lookup()

    @patch("mx_rec.graph.merge_lookup.ConfigInitializer")
    def test_err_empty_cutting_point_list(self, merge_lookup_config_initializer):
        mock_config_initializer = MockConfigInitializer(modify_graph=True, merged_multi_lookup=False)
        merge_lookup_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        with self.assertRaises(RuntimeError):
            merge_lookup.do_merge_lookup()


if __name__ == "__main__":
    unittest.main()
