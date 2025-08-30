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

import os
from collections import defaultdict
from typing import List, Dict, Set, Union, DefaultDict, Tuple

import tensorflow as tf
from tensorflow import Operation, Tensor, Graph
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.ops import control_flow_ops

from rec_sdk_common.log import logger
from mx_rec.graph.constants import AnchorDatasetOp, AnchorIteratorOp
from mx_rec.constants.constants import ASCAnchorAttr, SAVE_DIR_MODE
from mx_rec.core.embedding import BaseSparseEmbedding


def find_trans_dataset(graph: Graph, get_next: Operation) -> Operation:
    """Find the transformation dataset through 'get_next'.

    Args:
        get_next: The old 'IteratorGetNext' operation.

    Returns:
        trans_dataset: The target transformation dataset.
    """

    if get_next.type != AnchorIteratorOp.ITERATOR_GET_NEXT.value:
        raise TypeError(f"operation '{get_next}' must be one instance of 'IteratorGetNext'.")

    make_iter = find_make_iterator_op(graph, get_next.outputs[0])

    trans_dataset = None
    if tf.__version__.startswith("1"):
        optimize_dataset_op = upward_bfs_op(make_iter, AnchorDatasetOp.MODEL_DATASET.value)
        trans_dataset = find_parent_op(optimize_dataset_op)
        if not trans_dataset:
            raise RuntimeError("parent operation of 'ModelDataset' was not found.")
        if trans_dataset[0].type != AnchorDatasetOp.OPTIMIZE_DATASET.value:
            raise TypeError(f"operation 'OptimizeDataset' was not found.")
        trans_dataset = trans_dataset[0]
    else:
        trans_dataset = upward_bfs_op(make_iter, AnchorDatasetOp.PREFETCH_DATASET.value)

    return trans_dataset


def find_make_iterator_op(graph: Graph, batch_tensor: Tensor) -> Operation:
    operations = graph.get_operations()
    for each_op in operations:
        for input_tensor in batch_tensor.op.inputs:
            if (
                input_tensor.op.outputs
                and input_tensor.op.outputs[0] in list(each_op.inputs)
                and each_op.type == AnchorIteratorOp.MAKE_ITERATOR.value
            ):
                logger.debug("Op MakeIterator '%s' was found.", each_op.name)
                return each_op

    raise ValueError(f"operation `MakeIterator` cannot be found.")


def find_parent_op(operator: Operation) -> List[Operation]:
    parent_ops = []
    for input_tensor in operator.inputs:
        parent_op = input_tensor.op
        if isinstance(parent_op, Operation):
            parent_ops.append(parent_op)
    return parent_ops


def upward_bfs_op(base_ops: Union[Operation, Set[Operation], List[Operation]], tgt_op_type: str) -> Operation:
    if not isinstance(base_ops, (set, list)):
        base_ops = [base_ops]

    parent_ops = base_ops
    while True:
        for parent_op in parent_ops:
            if parent_op.type == tgt_op_type:
                return parent_op
        base_ops = parent_ops
        parent_ops = []
        for base_op in base_ops:
            parent_ops.extend(find_parent_op(base_op))
        if not parent_ops:
            raise ValueError(f"target operation '{tgt_op_type}'' was not found.")


def find_target_instance_dataset(graph: Graph, variant_tensor: Tensor) -> DatasetV1Adapter:
    dataset_instance_list = graph.get_collection("dataset_group")
    for ins in dataset_instance_list:
        if ins._variant_tensor == variant_tensor:
            if not isinstance(ins, DatasetV1Adapter):
                ins = ins._input_dataset
            logger.debug("Find target instance '%s', whose variant_tensor is '%s'.", ins, variant_tensor)
            if not isinstance(ins.element_spec, (list, tuple, dict)):
                raise NotImplementedError("the found dataset does not return a valid layout.")

            return ins

    raise LookupError(f"Can not find target instance, whose variant_tensor is '{variant_tensor}' respectively.")


def check_and_force_list(obj: Union[object, List[object]], obj_type: type) -> Union[object, List[object]]:
    if isinstance(obj, obj_type):
        obj = [obj]

    if isinstance(obj, list):
        for tensor in obj:
            if not isinstance(tensor, obj_type):
                raise ValueError(f"Given input parameter must be a {obj_type} or a list of {obj_type}")

    return obj


def check_cutting_points(cutting_point_list: List[Tensor]):
    for tensor in cutting_point_list:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Collection ASCEND_CUTTING_POINT can only contain Tensors, but '{tensor}' was found.")

        if tensor.op.type != "Identity":
            raise ValueError(f"Cutting point can only be the output of an Operator 'Identity'.")


def record_ops_to_replace(graph: Graph, src_op: Operation) -> DefaultDict[Tensor, List[Tuple[int, Operation]]]:
    replacement_specs = defaultdict(list)
    output_list = src_op.outputs
    op_list = graph.get_operations()
    for tensor in output_list:
        for operator in op_list:
            if tensor in operator.inputs:
                input_index = list(operator.inputs).index(tensor)
                replacement_specs[tensor].append((input_index, operator))

    return replacement_specs


def replace_anchor(replacement_specs: DefaultDict[Tensor, List[Tuple[int, Operation]]], new_tensor_list: List[Tensor]):
    if len(replacement_specs) != len(new_tensor_list):
        raise ValueError(
            f"Given replacement_specs and new_tensor_list must have the same length. "
            f"replacement_specs: {replacement_specs}, new_tensor_list: {new_tensor_list}"
        )

    for tensor_idx, (old_tensor, items) in enumerate(replacement_specs.items()):
        for input_idx, operator in items:
            try:
                operator._update_input(input_idx, new_tensor_list[tensor_idx])
            except InvalidArgumentError as err:
                logger.info(
                    "The replacement specs keys (old batch) is: %s. \n\t\t The new_tensor_list is: %s.",
                    replacement_specs.keys(),
                    new_tensor_list,
                )
                raise RuntimeError(
                    f"Cannot update edge, old tensor: {old_tensor}, " f"new tensor: {new_tensor_list[tensor_idx]}."
                ) from err


def replace_anchor_control(graph: Graph, place_holder_control: tf.Operation, real_anchor: Tensor):
    """
    将place_holder_control替换为入参real_anchor.

    Args:
        place_holder_control: control op
        real_anchor: 用来替换打桩节点的tensor

    Returns: None

    """

    if place_holder_control is None:
        raise RuntimeError(
            f"Node place_holder_control does not exist. Check whether the sparse lookup interface "
            f"is correctly invoked."
        )
    # find the op with stub node as the input
    replacement_specs_for_anchor_vec = record_control_to_replace(graph, place_holder_control)
    # replace anchor_vec with anchor
    replace_control_anchor(replacement_specs_for_anchor_vec, real_anchor)


def record_control_to_replace(graph: Graph, src_op: Operation) -> DefaultDict[Tensor, List[Tuple[int, Operation]]]:
    replacement_specs = defaultdict(list)
    op_list = graph.get_operations()
    for operator in op_list:
        if src_op in operator.control_inputs:
            input_index = operator.control_inputs.index(src_op)
            replacement_specs[src_op].append((input_index, operator))

    return replacement_specs


def replace_control_anchor(
    replacement_specs: DefaultDict[Tensor, List[Tuple[int, Operation]]], new_tensor_list: List[Tensor]
):

    for tensor_idx, (old_tensor, items) in enumerate(replacement_specs.items()):
        for _, operator in items:
            try:
                control_op = control_flow_ops.group(new_tensor_list)
                operator._add_control_input(control_op)
            except InvalidArgumentError as err:
                logger.info(
                    "The replacement control specs keys (old batch) is: %s. \n\t\t The new_tensor_list is: %s.",
                    replacement_specs.keys(),
                    new_tensor_list,
                )
                raise RuntimeError(
                    f"Cannot update edge, old tensor: {old_tensor}, " f"new tensor: {new_tensor_list[tensor_idx]}."
                ) from err


def replace_anchor_vec(graph: Graph, cutting_point: Tensor, attribute: ASCAnchorAttr, anchor: Tensor):
    """
    根据打桩节点的名字找到以此为输入的op，并将该op的输入替换为入参anchor.

    Args:
        cutting_point: sparse lookup查询的ids
        attribute: 被替换的打桩节点的名字
        anchor: 用来替换打桩节点的tensor

    Returns: None

    """

    # get stub node
    anchor_vec = BaseSparseEmbedding.get_anchor_attribute(cutting_point, attribute)
    if anchor_vec is None:
        raise RuntimeError(
            f"Node `{attribute.value}` does not exist. Check whether the sparse lookup interface "
            f"is correctly invoked."
        )
    # find the op with stub node as the input
    replacement_specs_for_anchor_vec = record_ops_to_replace(graph, anchor_vec.op)
    # replace anchor_vec with anchor
    replace_anchor(replacement_specs_for_anchor_vec, [anchor])


def make_sorted_key_to_tensor_list(
    element_spec: List[Dict[str, Tensor]], sorted_keys: List[str], prefix: str = ""
) -> List[str]:
    if isinstance(element_spec, tf.TensorSpec):
        sorted_keys.append(prefix)
        return sorted_keys
    elif isinstance(element_spec, dict):
        for key, item in element_spec.items():
            if not isinstance(key, str):
                raise TypeError(f"The key of element_spec must be a string.")

            prefix = "{0}_{1}".format(prefix, key)
            sorted_keys = make_sorted_key_to_tensor_list(item, sorted_keys, prefix=prefix)
            sorted_keys = sorted(sorted_keys)
        return sorted_keys

    elif isinstance(element_spec, (list, tuple)):
        for idx, item in enumerate(element_spec):
            prefix = "{0}_{1}".format(prefix, str(idx))
            sorted_keys = make_sorted_key_to_tensor_list(item, sorted_keys, prefix=prefix)
            sorted_keys = sorted(sorted_keys)
        return sorted_keys

    raise TypeError(f"Given element_spec, whose type is {type(element_spec)}, is invalid.")


def export_pb_graph(
    file_name: str,
    dump_graph: bool = False,
    graph_def: GraphDef = None,
    export_path: str = "./export_graph",
    as_text: bool = True,
):
    """
    Save tensorflow graph before and after modifier graph
    :param file_name: FileName of the graph
    :param dump_graph: Is serialize graph or not
    :param graph_def: A Graph or a GraphDef protocol buffer.
    :param export_path: Directory where to write the graph.
    This can refer to remote filesystems, such as Google Cloud Storage (GCS).
    :param as_text: If True, writes the graph as an ASCII proto
    :return: None
    """
    if dump_graph:
        dir_path = os.path.dirname(os.path.join(export_path, file_name))
        os.makedirs(dir_path, mode=SAVE_DIR_MODE, exist_ok=True)
        graph_def = graph_def if graph_def else tf.compat.v1.get_default_graph().as_graph_def()
        tf.io.write_graph(graph_def, export_path, file_name, as_text)
