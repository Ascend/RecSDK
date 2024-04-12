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

from typing import Dict, Tuple, List, Set

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter
from tensorflow.python.framework.ops import Operation
from tensorflow.python.util import nest as tf_nest
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util

from mx_rec.graph import modifier
from mx_rec.util.log import logger
from mx_rec.graph.utils import export_pb_graph
from mx_rec.graph.graph_typing import SubgraphInfo
from mx_rec.constants.constants import ASCEND_TIMESTAMP, ANCHOR_DATASET_NAME, MAX_WHILE_SIZE, AnchorIteratorOp
from mx_rec.validator.validator import para_checker_decorator, ClassValidator

tf.compat.v1.disable_eager_execution()

_ACG_NEW_NODE_PREFIX = "ACG_"
_ACG_NEW_ITERATOR = "ACG_NEW_ITERATOR"
_ACG_NEW_INITIALIZER = "ACG_NEW_INITIALIZER"

_OP_TYPE_TO_PUSH = frozenset(["StringSplit", "StringToNumber"])
_OP_TYPE_TO_IGNORE = frozenset([AnchorIteratorOp.ITERATOR_GET_NEXT])
_OP_TYPE_CONTAIN_STRING_TO_IGNORE = frozenset(["Dataset", "Summary"])
_OP_NAME_CONTAIN_STRING_TO_IGNORE = frozenset(["save", "report_", "loss"])
_OP_NAME_CONTAIN_STRING_TO_PUSH = frozenset(["ACG_PUSH_NODE"])

_TENSOR_TYPE_TO_IGNORE = frozenset([tf.variant, tf.resource])

_VARIABLE_TYPES = frozenset(["Variable", "VariableV2", "VarHandleOp"])
_IGNORE_REPLACE_NODE = frozenset(["Assign", "SaveV2"])


class ACGPushOpsToDatasetHook(tf.estimator.SessionRunHook):
    @para_checker_decorator(
        check_option_list=[
            ("dump_graph", ClassValidator, {"classes": (bool,)}),
        ]
    )
    def __init__(self, dump_graph: bool = False) -> None:
        super().__init__()
        self._dump_graph = dump_graph

        modifier.get_src_dataset = _patched_get_src_dataset
        logger.info("[ACGPushOpsToDatasetHook] The function `get_src_dataset` of modifier has been replaced!")

    def begin(self):
        logger.info("[ACGPushOpsToDataset] Trigger at beginning!")
        graph = tf.compat.v1.get_default_graph()
        _find_ops_to_be_pushed(graph=graph, dump_graph=self._dump_graph)

    def after_create_session(self, session, coord):
        logger.info("[ACGPushOpsToDatasetHook] Trigger after create session!")
        initializers = tf.compat.v1.get_collection(_ACG_NEW_INITIALIZER)
        logger.info("[ACGPushOpsToDatasetHook] Got new initialzers: %s.", initializers)
        session.run(initializers)

    def end(self, session):
        logger.info("[ACGPushOpsToDatasetHook] Trigger in the end!")


def _find_ops_to_be_pushed(graph: tf.Graph, dump_graph: bool = False):
    export_pb_graph("before_push_graph.pbtxt", dump_graph, graph_def=graph.as_graph_def())
    op_nodes = graph.get_operations()
    nodes_to_push = set()

    for op_node in op_nodes:
        if op_node.type in _OP_TYPE_TO_IGNORE:
            continue

        pushable = False
        if op_node.type in _OP_TYPE_TO_PUSH:
            pushable = True

        for ignore_type in _OP_TYPE_CONTAIN_STRING_TO_IGNORE:
            if ignore_type in op_node.type:
                pushable = False
            if not pushable:
                continue
        for ignore_name in _OP_NAME_CONTAIN_STRING_TO_IGNORE:
            if ignore_name in op_node.name:
                pushable = False
            if not pushable:
                continue
        for each_tensor in list(op_node.outputs) + list(op_node.inputs):
            if each_tensor.dtype in _TENSOR_TYPE_TO_IGNORE:
                pushable = False
            if not pushable:
                continue

        for push_name in _OP_NAME_CONTAIN_STRING_TO_PUSH:
            if push_name in op_node.name:
                pushable = True
                break

        if pushable:
            nodes_to_push.add(op_node)

    if not nodes_to_push:
        logger.info("No target op has to be pushed to dataset map func!")
        return

    logger.info("Found operations should be pushed: %s.", nodes_to_push)
    subgraph_nodes = _find_subgraph_nodes(
        graph, nodes_to_push, tgt_op_type=AnchorIteratorOp.ITERATOR_GET_NEXT.value, exclude_tgt_op=True
    )
    _push_subgraph_to_dataset(graph, subgraph_nodes, dump_graph)
    export_pb_graph("after_push_graph.pbtxt", dump_graph, graph_def=graph.as_graph_def())


def _find_subgraph_nodes(
    graph: tf.Graph,
    base_nodes: Set[tf.Operation],
    tgt_op_type: str,
    exclude_tgt_op: bool = True,
) -> Set[tf.Operation]:
    subgraph_nodes = set()
    visited_nodes = base_nodes
    found_nodes = base_nodes
    all_nodes = graph.get_operations()
    logger.info("Got base_nodes: %s.", base_nodes)

    loop_cnt = 0
    while len(found_nodes) > 0:
        loop_cnt += 1
        if loop_cnt > MAX_WHILE_SIZE:
            raise RuntimeError(f"In bfs_lookup function, the maximum cycle depth is greater than {MAX_WHILE_SIZE}.")

        base_nodes = set()
        for parent_node in found_nodes:
            if (not exclude_tgt_op) and parent_node.type == tgt_op_type:
                continue
            base_nodes.add(parent_node)
        found_nodes = set()
        for base_node in base_nodes:
            tmp_nodes = [x.op for x in base_node.inputs] + base_node.control_inputs
            _warn_for_var_scope_nodes(all_nodes, base_node)

            tmp_nodes = set(tmp_nodes) - visited_nodes
            if exclude_tgt_op:
                tmp_nodes = set(filter(lambda node: node.type != tgt_op_type, tmp_nodes))
            found_nodes.update(tmp_nodes)
            visited_nodes.update(tmp_nodes)

    subgraph_nodes.update(visited_nodes)
    logger.info("Found subgraph from nodes_to_push: %s.", subgraph_nodes)
    return subgraph_nodes


def _warn_for_var_scope_nodes(all_nodes: List[tf.Operation], base_node: tf.Operation):
    if base_node.type in _VARIABLE_TYPES:
        for x in base_node.outputs:
            varable_scope_node = [x for x in all_nodes if x.name.startswith(f"{base_node.name}/")]
            logger.warning("Got base_node: %s and varable_scope_node: %s.", base_node, varable_scope_node)


def _find_op_from_base_op(base_ops: tf.Operation, target_op_type: str) -> tf.Operation:
    base_ops = modifier.check_input_list(base_ops, tf.Operation)
    parent_ops = base_ops
    while True:
        for parent_op in parent_ops:
            if parent_op.type == target_op_type:
                return parent_op
        base_ops = parent_ops
        parent_ops = []
        for base_op in base_ops:
            parent_ops.extend(modifier.find_parent_op(base_op))
        if not parent_ops:
            raise ValueError(f"op {target_op_type} was not found.")


def _get_dataset_op(graph: tf.Graph, get_next_op: Operation) -> Operation:
    if get_next_op.type != AnchorIteratorOp.ITERATOR_GET_NEXT.value:
        raise TypeError(f"op '{get_next_op}' must be one instance of IteratorGetNext.")
    # looking for the MakeIterator operator which corresponds to given batch_tensor
    base_op = modifier.find_make_iterator_op(get_next_op.outputs[0])
    # looking for the op which is the one before OptimizeDataset operator
    if tf.__version__.startswith("1"):
        optimize_dataset_op = _find_op_from_base_op(base_op, "ModelDataset")
        target_op = modifier.find_parent_op(optimize_dataset_op)
        if not target_op:
            raise RuntimeError("the parent op for 'ModelDataset' op was not found.")
        if target_op[0].type != "OptimizeDataset":
            raise TypeError("op OptimizeDataset was not found.")
        target_op = target_op[0]
    else:
        # 'OptimizeDataset' is not available in TensorFlow2.X
        raise RuntimeError("Not supoprt tf2")
    return target_op


def _ordered_output_from_subgraph(subgraph_out: Dict[tf.Operation, Set[tf.Operation]]) -> List[tf.Tensor]:
    addition_funcgraph_output_tensor = []
    for k, v in sorted(subgraph_out.items(), key=lambda x: x[0].name):
        k_inputs = set(k.inputs)
        for node in v:
            _add_sorted_additional_tensors(addition_funcgraph_output_tensor, k_inputs, node)
    return addition_funcgraph_output_tensor


def _add_sorted_additional_tensors(addition_funcgraph_output_tensor, k_inputs, node):
    for each_tensor in sorted(node.outputs, key=lambda x: x.name):
        if each_tensor in k_inputs:
            addition_funcgraph_output_tensor.append(each_tensor)


def _get_tensor_consumers_unsafe(tensor: tf.Tensor) -> List[tf.Operation]:
    if isinstance(tensor, tf.Operation):
        raise RuntimeError(f"not support type: {node}")

    from tensorflow.python import pywrap_tensorflow as c_api

    consumer_names = c_api.TF_OperationOutputConsumers_wrapper(tensor._as_tf_output())
    graph = tensor.graph
    result = []
    for name in consumer_names:
        with graph._lock:
            if name in graph._nodes_by_name:  # ignore deleted node
                result.append(graph._nodes_by_name[name])

    return result


def _push_subgraph_to_dataset(graph: tf.Graph, subgraph_to_push: Set[tf.Operation], dump_graph: bool = False):
    subgraph_in, subgraph_out = _find_subgraph_in_out(subgraph_to_push)
    logger.info("Got input tensor of extracted subgraph: %s", subgraph_in)
    logger.info("Got output tensor of extracted subgraph: %s", subgraph_out)

    get_next_node = graph.get_operation_by_name(AnchorIteratorOp.ITERATOR_GET_NEXT.value)
    src_dataset = _get_src_dataset(graph, get_next_node)

    def acg_func(*x): # pragma: no cover
        old_x = x
        logger.debug("Got old batch layout: %s", x)

        x = tf_nest.flatten(x)
        for each_tensor in x:
            if not isinstance(each_tensor, tf.Tensor):
                raise RuntimeError(f"Expected tensor as input of mapfunc. but got: {x}!")

        funcgraph = tf.compat.v1.get_default_graph()
        subgraph_info = SubgraphInfo(subgraph_in, subgraph_out, subgraph_to_push)
        new_batch = _clone_subgraph_into_funcgraph(
            funcgraph,
            graph,
            subgraph_info,
            x,
            old_x,
        )

        logger.debug("Got new batch layout: %s.", new_batch)
        export_pb_graph("map_func_graph.pbtxt", dump_graph, graph_def=funcgraph.as_graph_def())
        return new_batch

    tgt_dataset = src_dataset.map(acg_func)
    tgt_dataset = tgt_dataset.prefetch(0)
    _update_iterator_getnext(
        graph=graph,
        get_next_op=get_next_node,
        tgt_dataset=tgt_dataset,
        subgraph_out=subgraph_out,
        subgraph_to_push=subgraph_to_push,
    )


def _find_subgraph_in_out(
    sub_graph_nodes: Set[tf.Operation],
) -> Tuple[Dict[tf.Operation, Set[tf.Operation]], Dict[tf.Operation, Set[tf.Operation]]]:
    relay_input_nodes = set()
    relay_output_nodes = set()
    input_to_subnodes = dict()
    output_to_subnodes = dict()

    for base_node in sub_graph_nodes:
        _update_subgraph_in(base_node, input_to_subnodes, relay_input_nodes, sub_graph_nodes)
        _update_subgraph_out(base_node, output_to_subnodes, relay_output_nodes, sub_graph_nodes)

    return input_to_subnodes, output_to_subnodes


def _update_subgraph_in(
    base_node: tf.Operation,
    input_to_subnodes: Dict[tf.Operation, Set[tf.Operation]],
    relay_input_nodes: Set[tf.Operation],
    sub_graph_nodes: Set[tf.Operation],
):
    for input_tensor in base_node.inputs:
        input_node = input_tensor.op
        if input_node not in sub_graph_nodes:
            relay_input_nodes.add(input_node)
            res = input_to_subnodes.get(input_node, set())
            res.add(base_node)
            input_to_subnodes[input_node] = res


def _update_subgraph_out(
    base_node: tf.Operation,
    output_to_subnodes: Dict[tf.Operation, Set[tf.Operation]],
    relay_output_nodes: Set[tf.Operation],
    sub_graph_nodes: Set[tf.Operation],
):
    for output_tensor in base_node.outputs:
        for output_consumer in output_tensor.consumers():
            if output_consumer not in sub_graph_nodes:
                relay_output_nodes.add(output_consumer)
                res = output_to_subnodes.get(output_consumer, set())
                res.add(base_node)
                output_to_subnodes[output_consumer] = res


def _get_src_dataset(graph: tf.Graph, get_next_op: Operation) -> DatasetV1Adapter:
    try:
        target_op = _get_dataset_op(graph, get_next_op)
    except (ValueError, TypeError, RuntimeError) as err:
        logger.warning("The dataset op was not found, the error is %s. Start to traverse the operations.", err)
        dataset_op_list = [op for op in graph.get_operations() if ANCHOR_DATASET_NAME in op.name]
        if len(dataset_op_list) != 1:
            raise RuntimeError(
                f"The `{ANCHOR_DATASET_NAME}` was not found from the operations, dataset_op_list: "
                f"{dataset_op_list}."
            ) from err
        target_op = dataset_op_list[0]
    except Exception as err:
        raise RuntimeError(f"The dataset was not found, the error is `{err}`.") from err
    if not target_op.outputs:
        raise ValueError(f"The length of the outputs of target op `{target_op}` is 0.")
    logger.info("Find target op `%s`, and output is `%s`.", target_op.name, target_op.outputs)
    src_dataset = modifier.find_target_instance_dataset(target_op.outputs[0])
    return src_dataset


def _clone_subgraph_into_funcgraph(
    funcgraph: tf.Graph,
    defaultgraph: tf.Graph,
    subgraph_info: SubgraphInfo,
    x: List[tf.Tensor],
    old_x: Tuple[Dict[str, tf.Tensor]],
) -> Dict[str, tf.Tensor]:
    topo_subgraph_list = _topo_subgraph(subgraph_info.subgraph_to_push)  # node
    tensor_mapping = {}  # subgraph-tensor -> funcgraph-tensor
    node_mapping = {}  # subgraph-node -> funcgraph-node
    for k, v in subgraph_info.subgraph_in.items():
        _get_mapping_for_subgraph_in(k, v, x, tensor_mapping)
    for old_node in topo_subgraph_list:
        _get_mapping_for_subgraph(funcgraph, defaultgraph, node_mapping, old_node, tensor_mapping)

    logger.info("Got node_mapping: %s", node_mapping)
    logger.info("Got tensor_mapping: %s", tensor_mapping)

    ordered_output_subgraph_tensors = _ordered_output_from_subgraph(subgraph_info.subgraph_out)
    addition_funcgraph_output_tensor = _get_mapping_tensor(tensor_mapping, ordered_output_subgraph_tensors)
    new_funcgraph_output_tensor = list(x) + addition_funcgraph_output_tensor
    logger.info("Got new_funcgraph_output_tensor: %s", new_funcgraph_output_tensor)

    new_x = old_x[0]
    for tensor in addition_funcgraph_output_tensor:
        last_key = f"{sorted(new_x)[-1]}_last_key"
        new_x[last_key] = tensor

    return new_x


def _get_mapping_for_subgraph_in(
    from_node: tf.Operation, to_nodes: Set[tf.Operation], x: List[tf.Tensor], tensor_mapping
):
    if from_node.type != AnchorIteratorOp.ITERATOR_GET_NEXT.value:
        raise RuntimeError(f"Expect IteratorGetNext for input tensor of subgraph, but got {from_node}")
    for node in to_nodes:
        for each_tensor in node.inputs:
            if each_tensor.op.type != AnchorIteratorOp.ITERATOR_GET_NEXT.value:
                continue
            old_tensor_name = each_tensor.name
            x_index = int(old_tensor_name.split(":")[-1])
            tensor_mapping[each_tensor] = x[x_index]


def _get_mapping_for_subgraph(
    funcgraph: tf.Graph,
    defaultgraph: tf.Graph,
    node_mapping: Dict[tf.Operation, tf.Operation],
    old_node: tf.Operation,
    tensor_mapping: Dict[tf.Tensor, tf.Tensor],
):
    logger.debug("old_node: %s \n old_node_inputs: %s", old_node, [x for x in old_node.inputs])
    node_def = old_node.node_def
    for each_tensor in old_node.inputs:
        if each_tensor not in tensor_mapping:
            raise RuntimeError(
                f"each_tensor(input) {each_tensor} need by {old_node.name} not in tensor_mapping.{tensor_mapping}"
            )
    new_inputs = _get_mapping_tensor(tensor_mapping, old_node.inputs)
    if old_node.type in _VARIABLE_TYPES:
        node_def = _frozen_variable_node_to_func_const_node_def(
            variable_node=old_node, funcgraph=funcgraph, defaultgraph=defaultgraph
        )
    node_def.name = _ACG_NEW_NODE_PREFIX + node_def.name
    new_node = tf.Operation(node_def=node_def, g=funcgraph, inputs=new_inputs)
    node_mapping[old_node] = new_node
    for old_out_tensor, new_out_tensor in zip(old_node.outputs, new_node.outputs):
        tensor_mapping[old_out_tensor] = new_out_tensor


def _frozen_variable_node_to_func_const_node_def(
    variable_node: tf.Operation, funcgraph: tf.Graph, defaultgraph: tf.Graph
) -> node_def_pb2.NodeDef:
    def create_const_node_def(node_name, dtype, data, data_shape=None):
        """Creates a Const op."""
        output_node = node_def_pb2.NodeDef()
        output_node.op = "Const"
        output_node.name = node_name
        output_node.attr["dtype"].CopyFrom(dtype)
        output_node.attr["value"].CopyFrom(
            attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(data, dtype=dtype.type, shape=data_shape))
        )
        return output_node

    # NOTE: Variable node type is readonly in funcgraph, all nodes of this type have to be fronzen.
    variable_name = variable_node.name
    if variable_node.type == "VarHandleOp":
        variable_name = f"{variable_name}/Read/ReadVariableOp:0"
    else:
        variable_name = f"{variable_name}:0"
    initializer = defaultgraph.get_operation_by_name(f"{variable_node.name}/Assign")
    logger.info(f"VariableV2: {variable_node.name}, initializer: {initializer.name} ")
    defaultsession = tf.compat.v1.Session(graph=defaultgraph)
    _ = defaultsession.run([initializer])
    logger.info(f"Start run variables data: {variable_name}")
    returned_variable_data = defaultsession.run(variable_name)
    logger.info(f"Start froze variables: {variable_name} {returned_variable_data}")
    new_const_node = create_const_node_def(
        variable_node.name, variable_node.node_def.attr["dtype"], returned_variable_data, returned_variable_data.shape
    )
    return new_const_node


def _get_mapping_tensor(tsr2tsr: Dict[tf.Tensor, tf.Tensor], keys: List[tf.Tensor]) -> List[tf.Tensor]:
    tensors = []
    for k in keys:
        if k not in tsr2tsr:
            raise KeyError(f"Failed to find key tensor: {k} from tensor map: {tsr2tsr}.")
        tensors.append(tsr2tsr[k])
    return tensors


def _topo_subgraph(subgraph: Set[tf.Operation]) -> List[tf.Operation]:
    topo_subgraph_list = []
    topo_subgraph_set = set()
    start_nodes = set()
    [start_nodes.add(x) for x in subgraph]
    logger.info("Got topo_subgraph start nodes: %s", start_nodes)

    def topo_subgraph_dfs(curr_node, output_list, output_set):
        if not isinstance(curr_node, tf.Operation):
            raise RuntimeError(f"topo_subgraph_dfs input should be node(aka. tf.Operator). {curr_node}")
        curr_inputs = curr_node.inputs
        logger.debug("Got topo_dfs: %s <- %s", curr_node.name, [x.name for x in curr_inputs])
        current_control_inputs = curr_node.control_inputs
        if len(current_control_inputs) > 0:
            raise RuntimeError(
                f"Control input are not supported: {curr_node.name}, control_inputs: {current_control_inputs}"
            )
        if curr_node in output_set:
            return
        output_set.add(curr_node)
        for tensor in curr_inputs:
            node = tensor.op
            if node.type != AnchorIteratorOp.ITERATOR_GET_NEXT.value and node not in output_set:
                topo_subgraph_dfs(node, output_list, output_set)
        output_list.append(curr_node)

    [topo_subgraph_dfs(x, topo_subgraph_list, topo_subgraph_set) for x in start_nodes]
    if len(topo_subgraph_list) != len(topo_subgraph_set):
        raise RuntimeError(f"Got duplicated topo node: {sorted(topo_subgraph_list, key=lambda x: x.name)}.")
    logger.info("Got topo_subgraph: %s", topo_subgraph_list)
    return topo_subgraph_list


def _update_iterator_getnext(
    graph: tf.Graph,
    get_next_op: Operation,
    tgt_dataset: DatasetV1Adapter,
    subgraph_out: Dict[tf.Operation, Set[tf.Operation]],
    subgraph_to_push: Set[tf.Operation],
):
    if not get_next_op.outputs:
        raise RuntimeError("there is no tensor in the dataset. Please check the dataset and data processing.")
    iterator_type = ""
    if get_next_op.inputs:
        iterator_type = get_next_op.inputs[0].op.type
    if iterator_type == "IteratorV2":
        iterator_type = modifier.find_make_iterator_op(get_next_op.outputs[0]).type
    if iterator_type not in (AnchorIteratorOp.MAKE_ITERATOR.value, AnchorIteratorOp.ONE_SHOT_ITERATOR.value):
        raise RuntimeError(
            f"Only iterators `MakeIterator` and `OneShotIterator` are supported in `graph modify` mode, "
            f"but the current iterator is `{iterator_type}`."
        )
    logger.info("The iterator type of dataset is %s.", iterator_type)
    if iterator_type == AnchorIteratorOp.MAKE_ITERATOR.value:
        new_iterator = tgt_dataset.make_initializable_iterator()
        logger.info("Got new_iterator: %s, new_iterator.initializer: %s.", new_iterator, new_iterator.initializer)
        graph.add_to_collection(_ACG_NEW_INITIALIZER, new_iterator.initializer)
    else:
        new_iterator = tgt_dataset.make_one_shot_iterator()
    new_batch = new_iterator.get_next(_ACG_NEW_ITERATOR)
    if "timestamp" in new_batch.keys():
        tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, new_batch["timestamp"])
    try:
        new_batch_tensor = new_batch
        while not isinstance(new_batch_tensor, tf.Tensor):
            if isinstance(new_batch_tensor, tuple):
                new_batch_tensor = new_batch_tensor[0]
            elif isinstance(new_batch_tensor, dict):
                new_batch_tensor = list(new_batch_tensor.values())
            elif isinstance(new_batch_tensor, list):
                new_batch_tensor = new_batch_tensor[0]
            elif isinstance(new_batch_tensor, tf.Tensor):
                break
            else:
                raise RuntimeError(
                    f"Need to support new_batch_tensor{new_batch_tensor}, type: {type(new_batch_tensor)}"
                )
    except IndexError as err:
        raise IndexError("Cannot find a tensor from given batch.") from err
    new_get_next_op = _find_op_from_base_op(new_batch_tensor.op, AnchorIteratorOp.ITERATOR_GET_NEXT.value)
    logger.info("Got new_get_next_op: %s.", new_get_next_op)
    _replace_get_next_op(graph, get_next_op, new_get_next_op, subgraph_out, subgraph_to_push)


def _replace_get_next_op(
    graph: tf.Graph,
    old_get_next_op: tf.Operation,
    new_get_next_op: tf.Operation,
    subgraph_out: Dict[tf.Operation, Set[tf.Operation]],
    subgraph_to_push: Set[tf.Operation],
):
    for output_tensor in old_get_next_op.outputs:
        _update_old_consumer(graph, new_get_next_op, output_tensor, subgraph_to_push)

    old_get_next_op_output_size = len(old_get_next_op.outputs)
    ordered_output_tensor = _ordered_output_from_subgraph(subgraph_out)

    for i, output_tensor in enumerate(ordered_output_tensor):
        offset = old_get_next_op_output_size + i
        _update_subgraph_out_consumer(graph, new_get_next_op, offset, output_tensor)


def _update_old_consumer(
    graph: tf.Graph, new_get_next_op: tf.Operation, output_tensor: tf.Tensor, subgraph_to_push: List[tf.Operation]
):
    old_tensor_name = output_tensor.name
    output_index = old_tensor_name.split(":")[-1]
    new_tensor_name = f"{new_get_next_op.name}:{output_index}"
    logger.info("Replace old_tensor_name: %s to new_tensor_name: %s", old_tensor_name, new_tensor_name)
    new_tensor = graph.get_tensor_by_name(new_tensor_name)
    for output_consumer in _get_tensor_consumers_unsafe(output_tensor):
        if output_consumer in subgraph_to_push:
            logger.info(
                "Ignore consumer in old subgraph %s, not let it connect to new IteratorGetNext.", output_consumer
            )
            continue
        for i, consumer_input in enumerate(output_consumer.inputs):
            if consumer_input != output_tensor:
                logger.debug("Not replace output_consumer: %s consumer_input: %s.", output_consumer, consumer_input)
                continue
            logger.info(
                "Success replace output_consumer: %s type: %s from consumer_input: %s to new_tensor: %s",
                output_consumer.name,
                output_consumer.type,
                consumer_input,
                new_tensor,
            )
            output_consumer._update_input(i, new_tensor)


def _update_subgraph_out_consumer(
    graph: tf.Graph, new_get_next_op: tf.Operation, offset: int, output_tensor: tf.Tensor
):
    new_tensor_name = f"{new_get_next_op.name}:{offset}"
    logger.info("Replace old_tensor_name: %s to new_tensor_name: %s.", output_tensor.name, new_tensor_name)
    new_tensor = graph.get_tensor_by_name(new_tensor_name)
    for output_consumer in _get_tensor_consumers_unsafe(output_tensor):
        if output_consumer.type in _IGNORE_REPLACE_NODE:
            logger.info("Ignore replace output_consumer: %s, it's of type: %s.", output_consumer, output_consumer.type)
            continue
        for j, consumer_input in enumerate(output_consumer.inputs):
            if consumer_input != output_tensor:
                logger.debug("Not replace output_consumer: %s consumer_input: %s.", output_consumer, consumer_input)
                continue
            logger.info(
                "Success replace output_consumer: %s type: %s from consumer_input: %s to new_tensor: %s",
                output_consumer.name,
                output_consumer.type,
                consumer_input,
                new_tensor,
            )
            output_consumer._update_input(j, new_tensor)


def _patched_get_src_dataset(get_next_op: Operation, is_training: bool) -> DatasetV1Adapter:
    try:
        target_op = modifier.get_dataset_op(get_next_op)
    except (ValueError, TypeError, RuntimeError) as err:
        logger.debug("In `OneShotIterator` mode, find `PrefetchDataset` from all ops in graph.")
        graph = tf.compat.v1.get_default_graph()
        dataset_op_list = [op for op in graph.get_operations() if ANCHOR_DATASET_NAME in op.name]
        dataset_op_list = sorted(dataset_op_list, key=lambda op: op.name)
        logger.debug("Got sorted dataset_op_list: %s.", dataset_op_list)
        if len(dataset_op_list) != 2:
            raise RuntimeError(
                f"Expect two `PrefetchDataset` ops in dataset_op_list, but got: {dataset_op_list}."
            ) from err
        target_op = dataset_op_list[1]
    except Exception as err:
        raise RuntimeError(f"The source dataset can't be found, got error: {err}.") from err

    if not target_op.outputs:
        raise ValueError(f"The length of the outputs of target op `{target_op}` is 0.")

    logger.debug("Find target dataset op: %s, and output is %s.", target_op, target_op.outputs)
    src_dataset = modifier.find_target_instance_dataset(target_op.outputs[0])

    return src_dataset
