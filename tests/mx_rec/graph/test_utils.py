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

import sys
import os
import pathlib
import shutil
import unittest
from unittest import TestCase

import tensorflow as tf
from tensorflow import Tensor, TensorSpec
from mx_rec.constants.constants import ASCAnchorAttr
from mx_rec.core.emb.base_sparse_embedding import BaseSparseEmbedding
from mx_rec.graph.utils import (
    check_input_list,
    find_parent_op,
    check_cutting_points,
    export_pb_graph,
    make_sorted_key_to_tensor_list,
    replace_anchor_vec,
)


class CheckInputListTest(TestCase):
    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    def test_ok_single_object(self):
        mock_obj = "obj"
        obj_type = str

        checked_objs = check_input_list(mock_obj, obj_type)
        self.assertEqual([mock_obj], checked_objs)

    def test_ok_object_list(self):
        mock_objs = ["obj1", "obj2", "ojb3"]
        obj_type = str

        checked_cutting_points = check_input_list(mock_objs, obj_type)
        self.assertEqual(mock_objs, checked_cutting_points)

    def test_err_inconsistent_object_and_type(self):
        mock_objs = ["obj1", "obj2", "ojb3"]
        obj_type = Tensor

        with self.assertRaises(ValueError):
            check_input_list(mock_objs, obj_type)


class FindParentOpTest(TestCase):
    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        tsr1 = tf.constant([1, 2, 3], dtype=tf.int64)
        mock_parent_op = tsr1.op
        tsr2 = tf.identity(tsr1)
        mock_child_op = tsr2.op

        parent_op = find_parent_op(mock_child_op)
        self.assertEqual([mock_parent_op], parent_op)


class CheckCuttingPointsTest(TestCase):
    def setUp(self):
        self._generator_iter_times = 3

    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_cutting_point_list = [tf.identity(tf.zeros(shape=(1,))) for _ in range(self._generator_iter_times)]
        check_cutting_points(mock_cutting_point_list)

    def test_err_invalid_cutting_point_list(self):
        mock_cutting_point_list = ["point" for _ in range(self._generator_iter_times)]
        with self.assertRaises(TypeError):
            check_cutting_points(mock_cutting_point_list)

    def test_err_invalid_cutting_point_operation(self):
        mock_cutting_point_list = [tf.zeros(shape=(1,)) for _ in range(self._generator_iter_times)]
        with self.assertRaises(ValueError):
            check_cutting_points(mock_cutting_point_list)


class ExportPBGraphTest(TestCase):
    def setUp(self) -> None:
        self._dir_name = "./export_graph"

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()
        if os.path.isdir(self._dir_name):
            shutil.rmtree(self._dir_name)

    def test_ok(self):
        mock_file_name = "test_graph.pbtxt"
        dump_graph = True
        mock_graph_def = tf.Graph().as_graph_def()
        as_text = True

        export_pb_graph(mock_file_name, dump_graph, mock_graph_def, self._dir_name, as_text)
        path = pathlib.Path(self._dir_name + "/" + mock_file_name)
        self.assertTrue(path.is_file())


class MakeSortedKeyToTensorListTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_batch = {
            "item_ids": TensorSpec(shape=(4096, 16), dtype=tf.int64),
            "user_ids": TensorSpec(shape=(4096, 8), dtype=tf.int64),
            "category_ids": TensorSpec(shape=(4096, 3), dtype=tf.int64),
            "label_0": TensorSpec(shape=(4096,), dtype=tf.int64),
            "label_1": TensorSpec(shape=(4096,), dtype=tf.int64),
            "user_ids_last_key": TensorSpec(shape=(4096, 16), dtype=tf.int64),
            "user_ids_last_key_last_key": TensorSpec(shape=(4096, 8), dtype=tf.int64),
        }
        mock_element_spec = [mock_batch]
        mock_sorted_keys = []
        mock_prefix = "mock_prefix"

        expected = [
            "mock_prefix_0_item_ids",
            "mock_prefix_0_item_ids_user_ids",
            "mock_prefix_0_item_ids_user_ids_category_ids",
            "mock_prefix_0_item_ids_user_ids_category_ids_label_0",
            "mock_prefix_0_item_ids_user_ids_category_ids_label_0_label_1",
            "mock_prefix_0_item_ids_user_ids_category_ids_label_0_label_1_user_ids_last_key",
            "mock_prefix_0_item_ids_user_ids_category_ids_label_0_label_1_user_ids_last_key_user_ids_last_key_last_key",
        ]
        sorted_batch_keys = make_sorted_key_to_tensor_list(mock_element_spec, mock_sorted_keys, mock_prefix)
        self.assertEqual(sorted_batch_keys, expected)


class ReplaceAnchorVecTest(TestCase):
    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_cutting_point = tf.zeros(shape=(4096, 8), dtype=tf.int64, name="ids")
        mock_attribute = ASCAnchorAttr.MOCK_LOOKUP_RESULT
        mock_anchor = tf.zeros(shape=(4096, 8), dtype=tf.float32, name="anchor")

        anchor_vec = tf.identity(mock_cutting_point, name="anchor_vec")
        anchor_vec_output = tf.identity(anchor_vec, name="anchor_vec_output")
        BaseSparseEmbedding.anchor_tensor_specs[mock_cutting_point][mock_attribute] = anchor_vec

        replace_anchor_vec(mock_cutting_point, mock_attribute, mock_anchor)
        self.assertEqual(anchor_vec_output.op.inputs[0], mock_anchor)


if __name__ == "__main__":
    unittest.main()
