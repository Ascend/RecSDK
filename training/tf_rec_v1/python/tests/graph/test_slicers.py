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
from unittest.mock import patch, Mock

import tensorflow as tf
from tensorflow import Graph

from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_ENTRANCE
from mx_rec.graph.constants import AnchorDatasetOp
from mx_rec.graph.slicers import NoGradSubgraphSlicer, LookupSubgraphSlicer, OrphanLookupKeySlicer
from graph.mock_dataset import gen_mock_dataset


class MockNoGradSubgraphSlicer(NoGradSubgraphSlicer):
    def __init__(self, full_graph: Graph = None, info_dir: str = "slicing") -> None:
        super().__init__(full_graph, info_dir)

    def summarize(self) -> None:
        pass

    def slice(self) -> None:
        pass


class NoGradSubgraphSlicerTestCase(unittest.TestCase):
    def test_ok_slice_ops(self):
        with tf.compat.v1.Graph().as_default():
            dataset = gen_mock_dataset()
            prefetch_dataset = dataset.prefetch(0)

            iterator = tf.compat.v1.data.make_initializable_iterator(prefetch_dataset)
            batch = iterator.get_next()

            mock_ids = batch["mock_ids"]
            mock_labels = batch["mock_labels"]

            inner_tensor = tf.identity(mock_ids)
            inner_op = inner_tensor.op

            tf.identity(inner_tensor)
            tf.identity(mock_labels)

            sliced_ops = {inner_op}
            MockNoGradSubgraphSlicer()._slice_ops(sliced_ops, is_training=True)

            g = tf.compat.v1.get_default_graph()
            prefetch_datasets = [op for op in g.get_operations() if AnchorDatasetOp.PREFETCH_DATASET.value in op.name]
            self.assertEqual(len(prefetch_datasets), 2)

    def test_ok_find_min_dep_ops(self):
        with tf.compat.v1.Graph().as_default():
            dataset = gen_mock_dataset()
            iterator = dataset.make_initializable_iterator()
            batch = iterator.get_next()
            ids = batch["mock_ids"]

            subgraph_in = tf.identity(ids)
            subgraph_out = tf.identity(subgraph_in)
            base_ops = {subgraph_out.op}

            min_dep_ops = NoGradSubgraphSlicer._find_min_dep_ops(base_ops)
            self.assertEqual(min_dep_ops, {subgraph_in.op, subgraph_out.op})

    def test_ok_validate_op(self):
        with tf.compat.v1.Graph().as_default():
            t = tf.constant(0)
            t = tf.add(t, 1)
            t = tf.subtract(t, 1)
            op = t.op

            is_valid = NoGradSubgraphSlicer._validate_op(op)
            self.assertTrue(is_valid, True)

    def test_ok_find_subgraph_in_and_out(self):
        with tf.compat.v1.Graph().as_default():
            dataset = gen_mock_dataset()
            iterator = dataset.make_initializable_iterator()
            batch = iterator.get_next()
            ids = batch.get("mock_ids")

            input_tensor = tf.identity(ids)
            inner_tensor = tf.identity(input_tensor)
            output_tensor = tf.identity(inner_tensor)
            subgraph_ops = {inner_tensor.op}

            (subgraph_in, subgraph_out) = MockNoGradSubgraphSlicer()._find_subgraph_in_and_out(subgraph_ops)
            self.assertEqual(subgraph_in, {input_tensor.op: {inner_tensor.op}})
            self.assertEqual(subgraph_out, {output_tensor.op: {inner_tensor.op}})

    def test_ok_find_old_dataset(self):
        with tf.compat.v1.Graph().as_default():
            dataset = gen_mock_dataset()
            iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]
            get_next = ids.op

            old_dataset = MockNoGradSubgraphSlicer()._find_old_dataset(get_next, is_training=True)
            self.assertEqual(old_dataset, dataset)

        with tf.compat.v1.Graph().as_default():
            dataset = gen_mock_dataset()
            prefetch_dataset = dataset.prefetch(0)
            iterator = tf.compat.v1.data.make_one_shot_iterator(prefetch_dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]
            get_next = ids.op

            old_dataset = MockNoGradSubgraphSlicer()._find_old_dataset(get_next, is_training=True)
            self.assertEqual(old_dataset, dataset)

        with tf.compat.v1.Graph().as_default():
            dataset = gen_mock_dataset()
            prefetch_dataset = dataset.prefetch(0)
            gen_mock_dataset().prefetch(0)

            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]
            get_next = ids.op

            old_dataset = MockNoGradSubgraphSlicer()._find_old_dataset(get_next, is_training=True)
            self.assertEqual(old_dataset, dataset)

        with tf.compat.v1.Graph().as_default():
            dataset = gen_mock_dataset()
            prefetch_dataset = dataset.prefetch(0)
            gen_mock_dataset().prefetch(0)

            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]
            get_next = ids.op

            old_dataset = MockNoGradSubgraphSlicer()._find_old_dataset(get_next, is_training=False)
            self.assertEqual(old_dataset, dataset)

    def test_ok_make_new_dataset(self):
        with tf.compat.v1.Graph().as_default():
            dataset = gen_mock_dataset()
            prefetch_dataset = dataset.prefetch(0)
            iterator = tf.compat.v1.data.make_initializable_iterator(prefetch_dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]

            in_op = ids.op
            inner_tensor = tf.identity(ids)
            inner_op = inner_tensor.op
            out_op = tf.identity(inner_tensor).op

            sliced_ops = {inner_op}
            in_op_to_edge_ops = {in_op: {inner_op}}
            out_op_to_edge_ops = {out_op: {inner_op}}

            new_dataset = MockNoGradSubgraphSlicer()._make_new_dataset(
                dataset, sliced_ops, in_op_to_edge_ops, out_op_to_edge_ops
            )
            new_prefetch_dataset = new_dataset
            new_iter = tf.compat.v1.data.make_initializable_iterator(new_prefetch_dataset)
            new_batch = new_iter.get_next()
            self.assertEqual(len(new_batch), 4)

    def test_ok_topo_sort_sliced_ops(self):
        with tf.compat.v1.Graph().as_default():
            t1 = tf.constant(0)
            t2 = tf.identity(t1)
            t3 = tf.identity(t2)
            ops = {t3.op, t2.op, t1.op}

            topo_sorted_ops = NoGradSubgraphSlicer._topo_sort_sliced_ops(ops)
            self.assertEqual(topo_sorted_ops, [t1.op, t2.op, t3.op])

    def test_ok_clone_subgraph_into_funcgraph(self):
        with tf.compat.v1.Graph().as_default():
            prefetch_dataset = gen_mock_dataset().prefetch(0)
            iterator = tf.compat.v1.data.make_initializable_iterator(prefetch_dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]

            in_op = ids.op
            inner_tensor = tf.identity(ids)
            inner_op = inner_tensor.op
            out_op = tf.identity(inner_tensor).op

            sliced_ops = {inner_op}
            in_op_to_edge_ops = {in_op: {inner_op}}
            out_op_to_edge_ops = {out_op: {inner_op}}

            with patch.object(tf.compat.v1.Graph, "get_tensor_by_name", return_value=tf.identity(inner_tensor)):
                new_batch = MockNoGradSubgraphSlicer()._clone_subgraph_into_funcgraph(
                    sliced_ops, in_op_to_edge_ops, out_op_to_edge_ops, batch
                )
            self.assertEqual(len(new_batch), 4)

    def test_ok_make_new_get_next(self):
        with tf.compat.v1.Graph().as_default():
            prefetch_dataset = gen_mock_dataset().prefetch(0)
            iterator = tf.compat.v1.data.make_initializable_iterator(prefetch_dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]

            old_get_next = ids.op
            new_dataset = gen_mock_dataset().prefetch(0)

            new_get_next = MockNoGradSubgraphSlicer()._make_new_get_next(old_get_next, new_dataset)
            self.assertIsNotNone(new_get_next)

        with tf.compat.v1.Graph().as_default():
            prefetch_dataset = gen_mock_dataset().prefetch(0)
            iterator = tf.compat.v1.data.make_one_shot_iterator(prefetch_dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]

            old_get_next = ids.op
            new_dataset = gen_mock_dataset().prefetch(0)

            new_get_next = MockNoGradSubgraphSlicer()._make_new_get_next(old_get_next, new_dataset)
            self.assertIsNotNone(new_get_next)


class LookupSubGraphSlicerTestCase(unittest.TestCase):
    def test_ok_find_all_tgt_ops(self):
        with tf.compat.v1.Graph().as_default():
            prefetch_dataset = gen_mock_dataset().prefetch(0)
            iterator = tf.compat.v1.data.make_initializable_iterator(prefetch_dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]

            inner_tensor = tf.identity(ids)
            tf.identity(inner_tensor)

            all_tgt_ops = LookupSubgraphSlicer(op_types=["Identity"])._find_all_tgt_ops()
            self.assertEqual(len(all_tgt_ops), 2)

    @patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding.BaseSparseEmbedding", get_anchor_attribute=Mock(return_value=True)
    )
    def test_ok_find_sliceable_tgt_ops(self):
        with tf.compat.v1.Graph().as_default():
            prefetch_dataset = gen_mock_dataset().prefetch(0)
            iterator = tf.compat.v1.data.make_initializable_iterator(prefetch_dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]

            inner_tensor = tf.identity(ids)
            lookup_key = tf.identity(inner_tensor)
            tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE, lookup_key)

            all_tgt_ops = LookupSubgraphSlicer(op_types=["Identity"])._find_sliceable_tgt_ops()
            self.assertEqual(len(all_tgt_ops), 2)


class OrphanLookupKeySlicerTestCase(unittest.TestCase):
    @patch.multiple("mx_rec.graph.slicers.utils", export_pb_graph=Mock(return_value=None))
    def test_ok_slice_ops(self):
        with tf.compat.v1.Graph().as_default():
            prefetch_dataset = gen_mock_dataset().prefetch(0)
            iterator = tf.compat.v1.data.make_initializable_iterator(prefetch_dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]

            inner_tensor = tf.constant(0, dtype=ids.dtype, shape=ids.shape)
            lookup_key = tf.identity(inner_tensor)
            tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE, lookup_key)

            sliceable_ops = {inner_tensor.op}
            OrphanLookupKeySlicer()._slice_ops(sliceable_ops, is_training=False)

            g = tf.compat.v1.get_default_graph()
            prefetch_datasets = [op for op in g.get_operations() if AnchorDatasetOp.PREFETCH_DATASET.value in op.name]
            self.assertEqual(len(prefetch_datasets), 2)

    @patch.multiple(
        "mx_rec.core.emb.base_sparse_embedding.BaseSparseEmbedding", get_anchor_attribute=Mock(return_value=True)
    )
    def test_ok_find_sliceable_tgt_ops(self):
        with tf.compat.v1.Graph().as_default():
            prefetch_dataset = gen_mock_dataset().prefetch(0)
            iterator = tf.compat.v1.data.make_initializable_iterator(prefetch_dataset)
            batch = iterator.get_next()
            ids = batch["mock_ids"]

            inner_tensor = tf.constant(0, dtype=ids.dtype, shape=ids.shape)
            lookup_key = tf.identity(inner_tensor)
            tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE, lookup_key)

            all_tgt_ops = OrphanLookupKeySlicer()._find_sliceable_tgt_ops()
            self.assertEqual(len(all_tgt_ops), 2)
