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
from collections import defaultdict
from unittest import TestCase, mock
from unittest.mock import patch, Mock, MagicMock
from typing import Union, Callable

import tensorflow as tf
from tensorflow import Tensor

from mx_rec.constants.constants import (
    ASCEND_CUTTING_POINT_INITIALIZER,
    ASCEND_SPARSE_LOOKUP_ENTRANCE,
    ASCEND_TIMESTAMP,
    ASCAnchorAttr,
)
from mx_rec.core.embedding import create_table
from mx_rec.core.asc.swap_args import SwapArgs
from mx_rec.graph.modifier import (
    GraphModifierHook,
    replace_anchor_for_ddr_ssd,
    _GraphModifier,
    _AnchorRecord,
    _get_input_index_list,
    _get_passing_tensor_list,
    _get_timestamp_index,
)
from tests.mx_rec.core.mock_class import MockConfigInitializer, MockSparseEmbedding, MockOptimizer
from tests.mx_rec.graph.mock_dataset import gen_mock_dataset


def _gen_mock_get_anchor_attribute(is_training: bool = True) -> Callable:
    def mock_get_anchor_attribute(anchor: Tensor, attr: ASCAnchorAttr) -> Union[bool, Mock]:
        if attr == ASCAnchorAttr.IS_TRAINING:
            return is_training
        if attr == ASCAnchorAttr.TABLE_INSTANCE:
            mock_table_instance = Mock()
            return mock_table_instance
        if attr == ASCAnchorAttr.FEATURE_SPEC:
            mock_feature_spec = Mock()
            mock_feature_spec.name = "mock_feature_spec_name"
            mock_feature_spec.table_name = "mock_table_name"
            return mock_feature_spec

        raise ValueError(f"Unsupported param 'attr' for enum class 'ASCAnchorAttr': attr={attr}.")

    return mock_get_anchor_attribute


class GetPreprocessingMapFuncTest(TestCase):
    def setUp(self) -> None:
        self._modifier = _GraphModifier()

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_get_map_func_success(self):
        mock_graph_def = self._modifier._full_graph.as_graph_def()
        mock_input_names = []
        mock_output_names = []

        self.assertTrue(
            callable(_GraphModifier._get_preprocessing_map_func(mock_graph_def, mock_input_names, mock_output_names))
        )


class GetInputIndexListTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_err_no_matched_cutting_point(self):
        mock_cutting_point_list = [tf.ones(shape=(4096, 8))]
        mock_replace_ment_specs = {}
        mock_mapping_name_list = []
        mock_base_count = 0

        with self.assertRaises(ValueError):
            _get_input_index_list(
                mock_cutting_point_list, mock_replace_ment_specs, mock_mapping_name_list, mock_base_count
            )


class GetPassingTensorList(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_tgt_op = mock_ids.op
        mock_cutting_point = tf.identity(mock_ids)
        mock_cutting_point_list = [mock_cutting_point]

        expected = {
            "passing_tensor_list": [mock_ids],
            "output_index_list": [0],
            "sub_src_tensors": mock_cutting_point_list,
        }
        passing_tensor_list, output_index_list, sub_src_tensors = _get_passing_tensor_list(
            mock_cutting_point_list, mock_tgt_op
        )
        self.assertEqual(passing_tensor_list, expected["passing_tensor_list"])
        self.assertEqual(output_index_list, expected["output_index_list"])
        self.assertEqual(sub_src_tensors, expected["sub_src_tensors"])


class GetSrcDatasetTest(TestCase):
    def setUp(self) -> None:
        self._modifier = _GraphModifier()

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    @patch("mx_rec.graph.modifier.ConfigInitializer")
    def test_ok_one_shot(self, modifier_config_initializer):
        mock_config_initializer = MockConfigInitializer(modify_graph=True)
        modifier_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        mock_dataset = gen_mock_dataset()
        mock_prefetch_dataset = mock_dataset.prefetch(10)
        mock_iterator = mock_prefetch_dataset.make_one_shot_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        src_dataset = self._modifier._get_src_dataset(mock_get_next_op, is_training=True)
        self.assertEqual(src_dataset, mock_dataset)


@patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=Mock(return_value=MockConfigInitializer()),
)
class GetTgtDatasetTest(TestCase):
    def setUp(self) -> None:
        self._modifier = _GraphModifier()

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    @patch.multiple(
        "mx_rec.graph.modifier",
        get_asc_insert_func=Mock(return_value=lambda x, y: x),
    )
    @patch.multiple("mx_rec.graph.modifier.BaseSparseEmbedding", get_anchor_attribute=_gen_mock_get_anchor_attribute())
    @patch("mx_rec.graph.modifier.ConfigInitializer")
    def test_ok(self, modifier_config_initializer):
        mock_config_initializer = MockConfigInitializer(modify_graph=True)
        modifier_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_sub_cutting_point_list = [mock_ids]
        mock_records = _AnchorRecord(defaultdict(), [], [], [], tf.compat.v1.GraphDef(), [], [], True)

        tgt_dataset = self._modifier._get_tgt_dataset(mock_dataset, mock_sub_cutting_point_list, mock_records)
        self.assertIsNotNone(tgt_dataset)


class ModifyGraphForAscTest(TestCase):
    def setUp(self) -> None:
        self._modifier = _GraphModifier()

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    @patch.multiple(
        "mx_rec.graph.modifier",
        get_asc_insert_func=Mock(return_value=lambda x, y: x),
    )
    @patch.multiple("mx_rec.graph.modifier.BaseSparseEmbedding", get_anchor_attribute=_gen_mock_get_anchor_attribute())
    @patch.multiple(
        "mx_rec.core.asc.manager",
        should_skip=MagicMock(return_value=True),
        check_dangling_table=MagicMock(return_value=["test_table"]),
    )
    @patch("mx_rec.graph.modifier.ConfigInitializer")
    def test_ok_train_mode(self, modifier_config_initializer):
        mock_config_initializer = MockConfigInitializer(modify_graph=True, merged_multi_lookup=True)
        modifier_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_cutting_point = tf.identity(mock_ids)

        test_table = MockSparseEmbedding("test_table")
        test_table.is_hbm = True
        mock_config_initializer.get_instance().sparse_embed_config.table_instance_dict = dict(test_table=test_table)

        mock_opt = MockOptimizer()
        modifier_config_initializer.get_instance().optimizer_config.optimizer_instance = mock_opt

        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE, mock_cutting_point)

        self._modifier.modify_graph_for_asc()

    @patch.multiple(
        "mx_rec.graph.modifier",
        get_asc_insert_func=Mock(return_value=lambda x, y: x),
        do_merge_lookup=Mock(),
    )
    @patch.multiple(
        "mx_rec.graph.modifier.BaseSparseEmbedding",
        get_anchor_attribute=_gen_mock_get_anchor_attribute(is_training=False),
    )
    @patch("mx_rec.graph.modifier.ConfigInitializer")
    def test_ok_eval_mode(self, modifier_config_initializer):
        mock_config_initializer = MockConfigInitializer(
            modify_graph=True, merged_multi_lookup=True, bool_gauge_set={"evaluate"}
        )
        modifier_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_cutting_point = tf.identity(mock_ids)

        test_table = MockSparseEmbedding("test_table")
        test_table.is_hbm = True
        mock_config_initializer.get_instance().sparse_embed_config.table_instance_dict = dict(test_table=test_table)

        mock_opt = MockOptimizer()
        modifier_config_initializer.get_instance().optimizer_config.optimizer_instance = mock_opt

        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE, mock_cutting_point)

        self._modifier.modify_graph_for_asc()

    @patch.multiple(
        "mx_rec.graph.modifier",
        get_asc_insert_func=Mock(return_value=lambda x, y: x),
    )
    @patch.multiple("mx_rec.graph.modifier.BaseSparseEmbedding", get_anchor_attribute=_gen_mock_get_anchor_attribute())
    @patch("mx_rec.graph.modifier.ConfigInitializer")
    def test_err_not_clear_flag(self, modifier_config_initializer):
        mock_config_initializer = MockConfigInitializer(modify_graph=True, merged_multi_lookup=False)
        modifier_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_cutting_point = tf.identity(mock_ids)

        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE, mock_cutting_point)

        with self.assertRaises(RuntimeError):
            self._modifier.modify_graph_for_asc()


class GetTimestampIndexTest(TestCase):
    def setUp(self) -> None:
        self._graph = tf.compat.v1.get_default_graph()

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    @patch.multiple(
        "mx_rec.graph.modifier.FeatureSpec",
        include_timestamp=Mock(),
        index_key=Mock(return_value=2),
    )
    @patch("mx_rec.graph.modifier.ConfigInitializer")
    def test_ok(self, modifier_config_initializer):
        mock_config_initializer = MockConfigInitializer()
        modifier_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_timestamp = mock_batch.get("mock_timestamp")
        mock_get_next_op = mock_timestamp.op

        tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, mock_timestamp)

        timestamp_index = _get_timestamp_index(self._graph, mock_get_next_op, is_training=True)
        self.assertEqual(timestamp_index, 2)


@patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=Mock(return_value=MockConfigInitializer()),
)
@patch.multiple(
    "tensorflow.compat.v1.train.Saver",
    __init__=Mock(return_value=None),
    build=Mock(),
)
class GraphModifierHookTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    @patch.multiple(
        "mx_rec.graph.modifier",
        modify_graph_and_start_emb_cache=Mock(),
        start_asc_pipeline=Mock(),
    )
    @patch("mx_rec.graph.modifier.ConfigInitializer")
    def test_ok(self, modifier_config_initializer):
        mock_config_initializer = MockConfigInitializer(
            modify_graph=True, is_graph_modify_hook_running=True, iterator_type="MakeIterator"
        )
        modifier_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_cutting_point = tf.identity(mock_ids)

        mock_new_iterator = mock_dataset.make_initializable_iterator()
        tf.compat.v1.add_to_collection(ASCEND_CUTTING_POINT_INITIALIZER, mock_new_iterator.initializer)

        with tf.compat.v1.train.MonitoredSession(hooks=[GraphModifierHook(modify_graph=True)]) as sess:
            sess.run(mock_iterator.initializer)
            sess.run(mock_cutting_point)

    @patch.multiple(
        "mx_rec.graph.modifier",
        modify_graph_and_start_emb_cache=Mock(),
        start_asc_pipeline=Mock(),
    )
    @patch("mx_rec.graph.modifier.ConfigInitializer")
    def test_err_invalid_iterator_type(self, modifier_config_initializer):
        mock_config_initializer = MockConfigInitializer(
            modify_graph=True, is_graph_modify_hook_running=True, iterator_type="InvalidIterator"
        )
        modifier_config_initializer.get_instance = Mock(return_value=mock_config_initializer)

        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_cutting_point = tf.identity(mock_ids)

        mock_new_iterator = mock_dataset.make_initializable_iterator()
        tf.compat.v1.add_to_collection(ASCEND_CUTTING_POINT_INITIALIZER, mock_new_iterator.initializer)

        with self.assertRaises(ValueError):
            with tf.compat.v1.train.MonitoredSession(hooks=[GraphModifierHook(modify_graph=True)]) as sess:
                sess.run(mock_iterator.initializer)
                sess.run(mock_cutting_point)


class TestReplaceAnchorForDDRSSD(unittest.TestCase):
    _mock_config_init_default = MockConfigInitializer(use_dynamic_expansion=False, use_static=True)

    def setUp(self):
        tf.compat.v1.reset_default_graph()

    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.graph.modifier.ConfigInitializer", new=_mock_config_init_default)
    def test_dev_continue(self):
        test_table = create_table(
            key_dtype=tf.int64,
            dim=8,
            name="test_table",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            device_vocabulary_size=8,
        )
        self._mock_config_init_default.get_instance().sparse_embed_config.kwargs = {"var": test_table}
        replace_anchor_for_ddr_ssd(tf.compat.v1.get_default_graph(), 1, 0)
        self.assertTrue(callable(replace_anchor_for_ddr_ssd))

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.graph.modifier.ConfigInitializer", new=_mock_config_init_default)
    def test_opt_is_none_and_train_mode_err(self):
        test_table = create_table(
            key_dtype=tf.int64,
            dim=8,
            name="test_table",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            device_vocabulary_size=8,
            host_vocabulary_size=16,
        )
        self._mock_config_init_default.get_instance().sparse_embed_config.kwargs = {"var": test_table}
        with self.assertRaises(RuntimeError):
            replace_anchor_for_ddr_ssd(tf.compat.v1.get_default_graph(), 1, 0)

    @mock.patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding",
        get_rank_size=mock.MagicMock(return_value=1),
        get_rank_id=mock.MagicMock(return_value=0),
        get_device_id=mock.MagicMock(return_value=0),
    )
    @mock.patch.multiple("mx_rec.graph.modifier.utils", replace_anchor_control=mock.MagicMock(return_value=None))
    @mock.patch.multiple("mx_rec.graph.modifier", _get_swap_info=mock.MagicMock(return_value=None))
    @mock.patch("mx_rec.core.embedding.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.core.emb.base_sparse_embedding.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.core.util.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.graph.modifier.ConfigInitializer", new=_mock_config_init_default)
    @mock.patch("mx_rec.graph.modifier.SwapArgs")
    def test_opt_is_none_and_eval_mode_ok(self, swap_args):
        test_table = create_table(
            key_dtype=tf.int64,
            dim=8,
            name="test_table",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
            device_vocabulary_size=8,
            host_vocabulary_size=16,
        )
        self._mock_config_init_default.get_instance().sparse_embed_config.kwargs = {"var": test_table}
        mock_swap_args = SwapArgs()
        mock_swap_args.set_data("control", var_name="test_table", var_channel=1, control_ops=tf.no_op())
        mock_swap_args.set_data("config", var_name="test_table", var_channel=1, swap_info=tf.no_op())
        mock_swap_args.set_slot_control(var_name="test_table", control_ops=tf.no_op())
        swap_args.return_value = mock_swap_args
        replace_anchor_for_ddr_ssd(tf.compat.v1.get_default_graph(), 0, 1)
        self.assertTrue(callable(replace_anchor_for_ddr_ssd))


if __name__ == "__main__":
    unittest.main()
