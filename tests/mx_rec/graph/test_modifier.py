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
from unittest import TestCase
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
from mx_rec.core.asc import FeatureSpec
from mx_rec.graph.modifier import (
    GraphModifierHook,
    AnchorRecord,
    find_make_iterator_op,
    find_target_dataset_op,
    find_target_instance_dataset,
    generate_get_next_op_specs,
    get_dataset_op,
    get_input_index_list,
    get_passing_tensor_list,
    get_preprocessing_map_func,
    get_src_dataset,
    get_tgt_dataset,
    get_timestamp_index,
    modify_graph_for_asc,
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
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_err_none_names_and_indexes(self):
        mock_graph_def = tf.compat.v1.GraphDef()
        mock_input_names = []
        mock_output_names = []

        with self.assertRaises(ValueError):
            get_preprocessing_map_func(mock_graph_def, mock_input_names, mock_output_names)


class GetInputIndexListTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_err_no_matched_cutting_point(self):
        mock_cutting_point_list = [tf.ones(shape=(4096, 8))]
        mock_replace_ment_specs = {}
        mock_mapping_name_list = []
        mock_base_count = 0

        with self.assertRaises(ValueError):
            get_input_index_list(
                mock_cutting_point_list, mock_replace_ment_specs, mock_mapping_name_list, mock_base_count
            )


class FindMakeIteratorOpTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")

        found_iter_op = find_make_iterator_op(mock_ids)
        self.assertEqual(found_iter_op.type, "MakeIterator")

    def test_err_no_tgt_dataset_op(self):
        mock_ids = tf.zeros(shape=(4096, 8))
        with self.assertRaises(ValueError):
            find_make_iterator_op(mock_ids)


class FindTargetDatasetOpTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_base_op = tf.identity(mock_ids).op

        found_tgt_dataset_op = find_target_dataset_op(base_ops=mock_base_op, op_type="IteratorGetNext")
        self.assertEqual(found_tgt_dataset_op, mock_ids.op)

    def test_err_no_tgt_op_type(self):
        mock_ids = tf.zeros(shape=(4096, 8))
        mock_base_op = mock_ids.op
        with self.assertRaises(ValueError):
            find_target_dataset_op(mock_base_op, "IteratorGetNext")


class GetDatasetOpTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        found_dataset_op = get_dataset_op(mock_get_next_op)
        self.assertEqual(found_dataset_op.type, "OptimizeDataset")

    def test_err_invalid_op_type(self):
        mock_get_next_op = tf.zeros(shape=(4096, 8)).op
        with self.assertRaises(TypeError):
            get_dataset_op(mock_get_next_op)


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
        passing_tensor_list, output_index_list, sub_src_tensors = get_passing_tensor_list(
            mock_cutting_point_list, mock_tgt_op
        )
        self.assertEqual(passing_tensor_list, expected["passing_tensor_list"])
        self.assertEqual(output_index_list, expected["output_index_list"])
        self.assertEqual(sub_src_tensors, expected["sub_src_tensors"])


class FindTargetInstanceDatasetTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_err_no_target_dataset_instance(self):
        with self.assertRaises(LookupError):
            find_target_instance_dataset(None)


class GetSrcDatasetTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok_one_shot(self):
        mock_dataset = gen_mock_dataset()
        mock_prefetch_dataset = mock_dataset.prefetch(10)
        mock_double_prefetch_dataset = mock_prefetch_dataset.prefetch(10)
        mock_iterator = mock_prefetch_dataset.make_one_shot_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        src_dataset = get_src_dataset(mock_get_next_op, is_training=True)
        self.assertEqual(src_dataset, mock_dataset)


@patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=Mock(return_value=MockConfigInitializer()),
)
class GetTgtDatasetTest(TestCase):
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
        mock_records = AnchorRecord(
            defaultdict(),
            [],
            [],
            [],
            tf.compat.v1.GraphDef(),
            [],
            [],
            True
        )

        tgt_dataset = get_tgt_dataset(mock_dataset, mock_sub_cutting_point_list, mock_records)
        self.assertIsNotNone(tgt_dataset)


class ModifyGraphForAscTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    @patch.multiple(
        "mx_rec.graph.modifier",
        get_asc_insert_func=Mock(return_value=lambda x, y: x),
    )
    @patch.multiple("mx_rec.graph.modifier.BaseSparseEmbedding", get_anchor_attribute=_gen_mock_get_anchor_attribute())
    @patch.multiple("mx_rec.core.asc.manager",
                         should_skip=MagicMock(return_value=True),
                         check_dangling_table=MagicMock(return_value=["test_table"]))
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

        modify_graph_for_asc()

    @patch.multiple(
        "mx_rec.graph.modifier",
        get_asc_insert_func=Mock(return_value=lambda x, y: x),
        do_merge_lookup=Mock(),
    )
    @patch.multiple(
        "mx_rec.graph.modifier.BaseSparseEmbedding",
        get_anchor_attribute=_gen_mock_get_anchor_attribute(is_training=False)
    )
    @patch("mx_rec.graph.modifier.ConfigInitializer")
    def test_ok_eval_mode(self, modifier_config_initializer):
        mock_config_initializer = MockConfigInitializer(modify_graph=True, merged_multi_lookup=True,
                                                        bool_gauge_set={"evaluate"})
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

        modify_graph_for_asc()

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
            modify_graph_for_asc()


class GetTimestampIndexTest(TestCase):
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

        timestamp_index = get_timestamp_index(mock_get_next_op, is_training=True)
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
        mock_config_initializer = MockConfigInitializer(modify_graph=True, is_graph_modify_hook_running=True,
                                                        iterator_type="MakeIterator")
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
        mock_config_initializer = MockConfigInitializer(modify_graph=True, is_graph_modify_hook_running=True,
                                                        iterator_type="InvalidIterator")
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


if __name__ == "__main__":
    unittest.main()
