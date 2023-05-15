# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2025. All rights reserved.
# Description: build script.
# Author: MindX SDK
from collections import defaultdict

import tensorflow as tf


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
    op_list = tf.get_default_graph().get_operations()
    for tensor in output_list:
        for operator in op_list:
            if tensor in operator.inputs:
                input_index = list(operator.inputs).index(tensor)
                replacement_specs[tensor].append((input_index, operator))

    return replacement_specs


def replace_anchor(replacement_specs: defaultdict, new_tensor_list: list):
    # pylint: disable=W0212
    if len(replacement_specs) != len(new_tensor_list):
        raise ValueError("Given replacement_specs and new_tensor_list must have the same length.")

    for tensor_idx, (_, items) in enumerate(replacement_specs.items()):
        for input_idx, operator in items:
            operator._update_input(input_idx, new_tensor_list[tensor_idx])


def export_pb_graph(file_name, dump_graph, graph_def=None, export_path="./export_graph"):
    if dump_graph:
        graph_def = graph_def if graph_def else tf.get_default_graph().as_graph_def()
        tf.train.write_graph(graph_def, export_path, file_name, False)


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
