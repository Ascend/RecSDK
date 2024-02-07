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

from unittest import TestCase
from unittest.mock import patch, Mock

import tensorflow as tf
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.data.ops.dataset_ops import DatasetV1
from mx_rec.graph.acg_push_ops import (
    ACGPushOpsToDatasetHook,
    SubgraphInfo,
    _OP_NAME_CONTAIN_STRING_TO_PUSH,
    _ACG_NEW_INITIALIZER,
    _find_ops_to_be_pushed,
    _find_op_from_base_op,
    _find_subgraph_nodes,
    _get_mapping_tensor,
    _topo_subgraph,
    _get_dataset_op,
    _clone_subgraph_into_funcgraph,
    _update_subgraph_out_consumer,
    _get_src_dataset,
    _update_iterator_getnext,
    _find_subgraph_in_out,
    _push_subgraph_to_dataset,
    _warn_for_var_scope_nodes,
    _frozen_variable_node_to_func_const_node_def,
    _update_old_consumer,
    _get_mapping_for_subgraph,
    _get_mapping_for_subgraph_in,
    _ordered_output_from_subgraph,
    _replace_get_next_op,
    _patched_get_src_dataset,
)
from tests.mx_rec.core.mock_class import MockConfigInitializer
from tests.mx_rec.graph.mock_dataset import gen_mock_dataset


@patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=Mock(return_value=MockConfigInitializer(modify_graph=True, is_graph_modify_hook_running=True)),
)
@patch.multiple(
    "tensorflow.compat.v1.train.Saver",
    __init__=Mock(return_value=None),
    build=Mock(),
)
@patch.multiple("mx_rec.graph.acg_push_ops", _find_ops_to_be_pushed=Mock())
class ACGPushOpsToDatasetHookTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_cutting_point = tf.identity(mock_ids)

        mock_new_iterator = mock_dataset.make_initializable_iterator()
        tf.compat.v1.add_to_collection(_ACG_NEW_INITIALIZER, mock_new_iterator.initializer)

        with tf.compat.v1.train.MonitoredSession(hooks=[ACGPushOpsToDatasetHook()]) as sess:
            sess.run(mock_iterator.initializer)
            sess.run(mock_cutting_point)


@patch.multiple(
    "mx_rec.graph.acg_push_ops",
    _find_subgraph_nodes=Mock(return_value=set()),
    _push_subgraph_to_dataset=Mock(),
)
class FindOpsToBePushedTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok_op_contain_str_to_push(self):
        tensor = tf.constant(value=[1, 2, 3], name="MOCK" + list(_OP_NAME_CONTAIN_STRING_TO_PUSH)[0])
        mock_graph = tf.compat.v1.get_default_graph()
        _find_ops_to_be_pushed(mock_graph)

    def test_ok_op_type_to_push(self):
        const_tensor = tf.constant(value=[1, 2, 3], dtype=tf.int32)
        str_tensor = tf.compat.v1.as_string(const_tensor)
        num_tensor = tf.compat.v1.string_to_number(str_tensor)
        mock_graph = tf.compat.v1.get_default_graph()
        _find_ops_to_be_pushed(mock_graph)

    def test_ok_no_node_to_push(self):
        mock_graph = tf.compat.v1.get_default_graph()
        _find_ops_to_be_pushed(mock_graph)


class FindSubgraphNodesTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")

        tensor_in_subgraph = tf.identity(mock_ids)
        tensor_out_subgraph = tf.identity(tensor_in_subgraph)
        mock_base_nodes = {tensor_out_subgraph.op}

        subgraph_nodes = _find_subgraph_nodes(
            tf.compat.v1.get_default_graph(), mock_base_nodes, tgt_op_type="IteratorGetNext"
        )
        self.assertEqual(subgraph_nodes, {tensor_in_subgraph.op, tensor_out_subgraph.op})


class WarnForVarScopeNodesTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        with tf.compat.v1.variable_scope("mock_var_scope"):
            var1 = tf.compat.v1.get_variable("var", shape=(3, 3), initializer=tf.random_normal_initializer())

        mock_all_nodes = tf.compat.v1.get_default_graph().get_operations()
        mock_base_node = var1.op
        _warn_for_var_scope_nodes(mock_all_nodes, mock_base_node)


class FindOpFromBaseOpTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_err_no_tgt_op_type(self):
        parent_tensor = tf.ones(shape=(3, 3))
        child_tensor = tf.identity(parent_tensor)
        with self.assertRaises(ValueError):
            _find_op_from_base_op(child_tensor.op, "IteratorGetNext")


class GetDatasetOpTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_prefetch_dataset = mock_dataset.prefetch(buffer_size=10)
        mock_iterator = mock_prefetch_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        mock_graph = tf.compat.v1.get_default_graph()
        expected = mock_graph.get_operation_by_name("OptimizeDataset")

        tgt_dataset_op = _get_dataset_op(mock_graph, mock_get_next_op)
        self.assertEqual(tgt_dataset_op, expected)

    def test_err_invalid_get_next_op_type(self):
        mock_get_next_op = tf.zeros(shape=(3,)).op
        mock_graph = tf.compat.v1.get_default_graph()

        with self.assertRaises(TypeError):
            _get_dataset_op(mock_graph, mock_get_next_op)

    @patch.multiple("mx_rec.graph.acg_push_ops", _find_op_from_base_op=Mock(return_value=None))
    @patch.multiple("mx_rec.graph.acg_push_ops.modifier", find_parent_op=Mock(return_value=None))
    def test_err_no_tgt_op_found(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        mock_graph = tf.compat.v1.get_default_graph()

        with self.assertRaises(RuntimeError):
            _get_dataset_op(mock_graph, mock_get_next_op)


class OrderedOutputFromSubgraphTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next(name="IteratorGetNext")
        mock_ids = mock_batch.get("mock_ids")

        mock_subgraph_out = {tf.identity(mock_ids).op: {mock_ids.op}}

        addition_funcgraph_output_tensor = _ordered_output_from_subgraph(mock_subgraph_out)
        self.assertEqual(addition_funcgraph_output_tensor, [mock_ids])


class PushSubgraphToDatasetTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")

        tensor_in_subgraph = tf.identity(mock_ids)
        tensor_out_subgraph = tf.identity(tensor_in_subgraph)
        mock_subgraph_to_push = {tensor_in_subgraph.op}
        _push_subgraph_to_dataset(tf.compat.v1.get_default_graph(), mock_subgraph_to_push)


class FindSubgraphInOutTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")

        tensor_in_subgraph = tf.identity(mock_ids)
        tensor_out_subgraph = tf.identity(tensor_in_subgraph)
        mock_subgraph_nodes = {tensor_in_subgraph.op}

        (
            subgraph_in,
            subgraph_out,
        ) = _find_subgraph_in_out(mock_subgraph_nodes)
        self.assertEqual(subgraph_in, {mock_ids.op: {tensor_in_subgraph.op}})
        self.assertEqual(subgraph_out, {tensor_out_subgraph.op: {tensor_in_subgraph.op}})


class GetSrcDatasetTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok_make_iterator(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        src_dataset = _get_src_dataset(tf.compat.v1.get_default_graph(), mock_get_next_op)
        self.assertEqual(src_dataset, mock_dataset)

    def test_ok_one_shot_iterator(self):
        mock_dataset = gen_mock_dataset()
        mock_prefetch_dataset = mock_dataset.prefetch(10)
        mock_iterator = mock_prefetch_dataset.make_one_shot_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        src_dataset = _get_src_dataset(tf.compat.v1.get_default_graph(), mock_get_next_op)
        self.assertEqual(src_dataset, mock_dataset)

    def test_err_no_anchor_dataset(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_one_shot_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        with self.assertRaises(RuntimeError):
            _get_src_dataset(tf.compat.v1.get_default_graph(), mock_get_next_op)


class CloneSubgraphIntoFuncgraphTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")

        mock_subgraph_in = {mock_ids.op: {tf.identity(mock_ids).op}}
        mock_subgraph_out = {tf.identity(mock_ids).op: {mock_ids.op}}
        mock_subgraph_to_push = set()
        mock_subgraph_info = SubgraphInfo(mock_subgraph_in, mock_subgraph_out, mock_subgraph_to_push)

        mock_new_ids = tf.ones_like(mock_ids)
        mock_x = [mock_new_ids]
        mock_old_x = ({"mock_new_ids": mock_new_ids},)

        mock_defaultgraph = tf.compat.v1.get_default_graph()
        with tf.Graph().as_default():
            mock_funcgraph = tf.compat.v1.get_default_graph()
            _clone_subgraph_into_funcgraph(mock_funcgraph, mock_defaultgraph, mock_subgraph_info, mock_x, mock_old_x)


class GetMappingForSubgraphInTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_prefetch_dataset = mock_dataset.prefetch(10)
        mock_iterator = mock_prefetch_dataset.make_one_shot_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")

        mock_from_node = mock_ids.op
        mock_to_nodes = {tf.identity(mock_ids).op}
        mock_new_ids = tf.zeros_like(mock_ids)
        mock_x = [mock_new_ids]
        tensor_mapping = dict()

        _get_mapping_for_subgraph_in(mock_from_node, mock_to_nodes, mock_x, tensor_mapping)
        self.assertEqual(tensor_mapping, {mock_ids: mock_new_ids})


class GetMappingForSubgraphTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_defaultgraph = tf.compat.v1.get_default_graph()

        # NOTE: Simulate independent graph environment while executing `dataset.map()` method.
        with tf.Graph().as_default():
            key_tensor = tf.zeros(shape=(1))
            val_tensor = tf.zeros(shape=(1))
            mock_tensor_mapping = {key_tensor: val_tensor}

            mock_node_mapping = dict()
            mock_old_node = tf.identity(key_tensor).op
            mock_funcgraph = tf.compat.v1.get_default_graph()

            _get_mapping_for_subgraph(
                mock_funcgraph, mock_defaultgraph, mock_node_mapping, mock_old_node, mock_tensor_mapping
            )

        self.assertEqual(len(mock_node_mapping), 1)
        self.assertEqual(len(mock_tensor_mapping), 2)


@patch.multiple(
    "mx_rec.graph.patch",
    ConfigInitializer=Mock(return_value=MockConfigInitializer(modify_graph=True, is_graph_modify_hook_running=True)),
)
class FrozenVariableNodeToFuncConstNodeDefTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        var_tensor = tf.Variable(initial_value=[1], shape=(1,))
        tf.compat.v1.assign(ref=var_tensor, value=[1])

        mock_funcgraph = tf.Graph()
        mock_defaultgraph = tf.compat.v1.get_default_graph()
        new_const_node: node_def_pb2.NodeDef = _frozen_variable_node_to_func_const_node_def(
            var_tensor.op, mock_funcgraph, mock_defaultgraph
        )
        self.assertEqual(new_const_node.op, "Const")


class GetMappingTensorTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        key_tensor = tf.zeros(shape=(3, 3))
        val_tensor = tf.ones(shape=(3, 3))
        tsr2tsr = {key_tensor: val_tensor}
        keys = [key_tensor]

        mapped_tensors = _get_mapping_tensor(tsr2tsr, keys)
        self.assertEqual(mapped_tensors, [val_tensor])

    def test_err_key_tensor_not_exist(self):
        tsr2tsr = {tf.zeros(shape=(3, 3)): tf.ones(shape=(3, 3))}
        keys = [tf.ones(shape=(3, 3))]

        with self.assertRaises(KeyError):
            _get_mapping_tensor(tsr2tsr, keys)


class TopoSubgraphTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_prefetch_dataset = mock_dataset.prefetch(10)
        mock_iterator = mock_prefetch_dataset.make_one_shot_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        tensor1 = tf.identity(mock_ids)
        tensor2 = tf.add(tensor1, 1)
        mock_subgraph = {tensor1.op, tensor2.op}

        const_op_for_add = None
        for tensor in tensor2.op.inputs:
            if tensor.op.name != "Add/y":
                continue
            const_op_for_add = tensor.op

        if not const_op_for_add:
            self.fail(
                f"Failed to find input of add operation, input tensor of add op: {[x.op for x in tensor2.op.inputs]}"
            )

        topo_subgraph_list = _topo_subgraph(mock_subgraph)
        self.assertEqual(topo_subgraph_list, [tensor1.op, const_op_for_add, tensor2.op])


class UpdateIteratorGetNextTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_old_dataset = gen_mock_dataset()
        mock_old_iterator = mock_old_dataset.make_initializable_iterator()
        mock_old_batch = mock_old_iterator.get_next(name="OldIteratorGetNext")
        mock_old_ids = mock_old_batch.get("mock_ids")
        mock_old_get_next_op = mock_old_ids.op

        mock_new_dataset: DatasetV1 = mock_old_dataset.map(lambda x: x)
        mock_subgraph_out = {tf.identity(mock_old_ids).op: {mock_old_ids.op}}

        _update_iterator_getnext(
            graph=tf.compat.v1.get_default_graph(),
            get_next_op=mock_old_get_next_op,
            tgt_dataset=mock_new_dataset,
            subgraph_out=mock_subgraph_out,
            subgraph_to_push=set(),
        )


class UpdateOldConsumerTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next(name="NewIteratorGetNext")
        mock_ids = mock_batch.get("mock_ids")
        mock_new_get_next_op = mock_ids.op
        mock_output_tensor = tf.identity(mock_ids)

        _update_old_consumer(
            graph=tf.compat.v1.get_default_graph(),
            new_get_next_op=mock_new_get_next_op,
            output_tensor=mock_ids,
            subgraph_to_push=set(),
        )


class UpdateSubgraphOutConsumerTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_iterator = mock_dataset.make_initializable_iterator()
        mock_batch = mock_iterator.get_next(name="NewIteratorGetNext")
        mock_ids = mock_batch.get("mock_ids")
        mock_new_get_next_op = mock_ids.op
        mock_output_tensor = tf.identity(mock_ids)

        _update_subgraph_out_consumer(
            graph=tf.compat.v1.get_default_graph(),
            new_get_next_op=mock_new_get_next_op,
            offset=0,
            output_tensor=mock_ids,
        )


class PatchedGetSrcDatasetTest(TestCase):
    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_ok(self):
        mock_dataset = gen_mock_dataset()
        mock_prefetch_dataset = mock_dataset.prefetch(10)
        mock_double_prefetch_dataset = mock_prefetch_dataset.prefetch(10)
        mock_iterator = mock_prefetch_dataset.make_one_shot_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        src_dataset = _patched_get_src_dataset(mock_get_next_op, is_training=True)
        self.assertEqual(src_dataset, mock_prefetch_dataset)

    def test_err_single_prefetch_dataset(self):
        mock_dataset = gen_mock_dataset()
        mock_prefetch_dataset = mock_dataset.prefetch(10)
        mock_iterator = mock_prefetch_dataset.make_one_shot_iterator()
        mock_batch = mock_iterator.get_next()
        mock_ids = mock_batch.get("mock_ids")
        mock_get_next_op = mock_ids.op

        with self.assertRaises(RuntimeError):
            _patched_get_src_dataset(mock_get_next_op, is_training=True)
