#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from collections import defaultdict
import os

import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from mx_rec.constants.constants import ASCAnchorAttr, DUMP_MIDIFY_GRAPH_FILE_MODE
from mx_rec.core.embedding import SparseEmbedding
from mx_rec.util.log import logger


def check_input_list(objs, obj_type):
    if isinstance(objs, obj_type):
        objs = [objs]

    if isinstance(objs, list):
        for tensor in objs:
            if not isinstance(tensor, obj_type):
                raise ValueError(f"Given input parameter must be a {obj_type} or a list of {obj_type}")

    return objs


def find_parent_op(operator):
    parent_ops = []
    for input_tensor in operator.inputs:
        parent_op = input_tensor.op
        if isinstance(parent_op, tf.Operation):
            parent_ops.append(parent_op)
    return parent_ops


def check_cutting_points(cutting_point_list):
    for tensor in cutting_point_list:
        if not isinstance(tensor, tf.Tensor):
            raise TypeError(f"Collection ASCEND_CUTTING_POINT can only contain Tensors, but '{tensor}' was found.")

        if tensor.op.type != "Identity":
            raise ValueError(f"Cutting point can only be the output of an Operator 'Identity'.")


def record_ops_to_replace(src_op):
    replacement_specs = defaultdict(list)
    output_list = src_op.outputs
    op_list = tf.compat.v1.get_default_graph().get_operations()
    for tensor in output_list:
        for operator in op_list:
            if tensor in operator.inputs:
                input_index = list(operator.inputs).index(tensor)
                replacement_specs[tensor].append((input_index, operator))

    return replacement_specs


def replace_anchor(replacement_specs: defaultdict, new_tensor_list: list):
    if len(replacement_specs) != len(new_tensor_list):
        raise ValueError(f"Given replacement_specs and new_tensor_list must have the same length. "
                         f"replacement_specs: {replacement_specs}, new_tensor_list: {new_tensor_list}")

    for tensor_idx, (old_tensor, items) in enumerate(replacement_specs.items()):
        for input_idx, operator in items:
            try:
                operator._update_input(input_idx, new_tensor_list[tensor_idx])
            except InvalidArgumentError as err:
                logger.info("The replacement specs keys (old batch) is: %s. \n\t\t The new_tensor_list is: %s.",
                            replacement_specs.keys(), new_tensor_list)
                raise RuntimeError(f"Cannot update edge, old tensor: {old_tensor}, "
                                   f"new tensor: {new_tensor_list[tensor_idx]}.") from err


def export_pb_graph(file_name, dump_graph, graph_def=None, export_path="./export_graph", as_text=False):
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
        os.makedirs(dir_path, mode=DUMP_MIDIFY_GRAPH_FILE_MODE, exist_ok=True)
        graph_def = graph_def if graph_def else tf.compat.v1.get_default_graph().as_graph_def()
        tf.io.write_graph(graph_def, export_path, file_name, as_text)


def make_sorted_key_to_tensor_list(element_spec, sorted_keys, prefix=""):
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


def replace_anchor_vec(cutting_point: tf.Tensor, attribute: ASCAnchorAttr, anchor: tf.Tensor):
    """
    根据打桩节点的名字找到以此为输入的op，并将该op的输入替换为入参anchor.

    Args:
        cutting_point: sparse lookup查询的ids
        attribute: 被替换的打桩节点的名字
        anchor: 用来替换打桩节点的tensor

    Returns: None

    """

    # get stub node
    anchor_vec = SparseEmbedding.get_anchor_attribute(cutting_point, attribute)
    if anchor_vec is None:
        raise RuntimeError(f"Node `{attribute.value}` does not exist. Check whether the sparse lookup interface "
                           f"is correctly invoked.")
    # find the op with stub node as the input
    replacement_specs_for_anchor_vec = record_ops_to_replace(anchor_vec.op)
    # replace anchor_vec with anchor
    replace_anchor(replacement_specs_for_anchor_vec, [anchor])
