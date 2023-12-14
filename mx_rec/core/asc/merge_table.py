#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from typing import Dict, List

import tensorflow as tf
from tensorflow import Operation, Tensor

from mx_rec.constants.constants import MAX_WHILE_SIZE, ASCEND_TABLE_NAME_MUST_CONTAIN
from mx_rec.util.initialize import get_enable_table_merge, export_table_instances, insert_dangling_table, \
    get_bool_gauge_set
from mx_rec.util.log import logger


def affirm(reach_op: List[Operation]) -> bool:
    for node in reach_op:
        if node.type not in ("IdentityN", "Reshape", "Identity"):
            return False
    return True


def check_op(table_reachable_op: Operation) -> bool:
    """Check whether the tensor op is optimizer op or backward gradient.

    Args:
        table_reachable_tensor: tensor
    Returns:
        bool
    """
    if table_reachable_op.type == 'ApplyAdam':
        return True

    if 'gradients' in table_reachable_op.name and \
            table_reachable_op.type in ['UnsortedSegmentSum', 'TensorScatterUpdate']:
        return True

    return False


def is_train_task():
    bool_gauge_set = get_bool_gauge_set()
    if not bool_gauge_set:
        op_list = tf.compat.v1.get_default_graph().get_operations()
        for t_op in op_list:
            if check_op(t_op):
                return True

    if 'train' in bool_gauge_set or 'train_and_evaluate' in bool_gauge_set:
        return True

    return False


def find_dangling_table(table_names: List[str]) -> List[str]:
    """ Find the tables which are disconenct with the forward training graph. And
    these table will not be backward updated.

    :param table_names: list of all created tables' names
    :return: a list of dangling table names.
    """

    def find_table_op(table_name: str,
                      the_op: Operation,
                      table_lookup_op: Dict[str, List[Operation]],
                      table_reachable_tensor: Dict[str, List[Tensor]]) -> None:  # pragma: no cover
        """ find all the table lookup op.
        :param table_name: tables' names
        :param the_op: the op to be
        :param table_lookup_op: list of the table lookup ops
        :param table_reachable_tensor: the tensors which table lookup op can reach (
                here we just add the table lookup op's output tensors).
                The data structure is map, key is table_name, value is the output tensors of table lookup op.
        :return: None
        """
        if table_name in the_op.name and the_op.type == "IdentityN":
            if table_name not in table_lookup_op:
                table_lookup_op[table_name] = [the_op]
                table_reachable_tensor[table_name] = []
                table_reachable_tensor[table_name].extend(the_op.outputs)
            elif the_op not in table_lookup_op[table_name]:
                table_lookup_op[table_name].append(the_op)
                table_reachable_tensor[table_name].extend(the_op.outputs)

    def extend(op_list: List[Operation],
               tensor: Tensor,
               spread_tensors: List[Tensor]) -> None:  # pragma: no cover
        """extend the tensors which table lookup op can reach

        :param op_list: all op in the graph
        :param tensor: the tensor visited by bfs
        :param spread_tensors: the list of tensors which table lookup op can reach
        :return:
        """
        for the_op in op_list:
            if tensor in the_op.inputs:
                spread_tensors.extend(the_op.outputs)

    def bfs_lookup(next_to_visit: List[Tensor]) -> (set, bool):  # pragma: no cover
        """find all the tensors which table lookup op can reach

        :param next_to_visit: the tensor list to be visited by bfs
        :return: bool value indicate whether reached optimizer op or backward gradient op
        """
        tensors_visited = set()
        op_visited = set()
        while_num = 0
        while next_to_visit:
            while_num += 1
            if while_num > MAX_WHILE_SIZE:
                raise RuntimeError(f"In bfs_lookup function, the maximum cycle depth is greater than {MAX_WHILE_SIZE}.")
            spread_tensors = []
            for tensor in next_to_visit:
                if tensor in tensors_visited:
                    continue
                if check_op(tensor.op):
                    return op_visited, True
                tensors_visited.add(tensor)
                op_visited.add(tensor.op)
                extend(op_list, tensor, spread_tensors)
            next_to_visit = spread_tensors
        return op_visited, False

    if not is_train_task():
        logger.info("!!merge table only available in train task.")
        return []
    if not get_enable_table_merge():
        return []

    op_list = tf.compat.v1.get_default_graph().get_operations()

    table_lookup_op = {}
    table_reachable_tensor = {}

    for _, table_instance in export_table_instances().items():
        if table_instance.table_name not in table_names:
            table_names.append(table_instance.table_name)

    for the_op in op_list:
        for table_name in table_names:
            find_table_op(table_name, the_op, table_lookup_op, table_reachable_tensor)

    logger.debug("*********** find tables: %s ***********", table_lookup_op)
    dangling_table = []

    for table_name in table_names:
        if table_name not in table_lookup_op:
            logger.debug("*********** created table %s but never look up***********", table_name)
            dangling_table.append(table_name)
            insert_dangling_table(table_name)

    for table_name, table_op in table_reachable_tensor.items():
        reach_op, found = bfs_lookup(table_op)
        if not found and affirm(reach_op):
            dangling_table.append(table_name)
            insert_dangling_table(table_name)
    return dangling_table


def should_skip(table_name) -> bool:
    if ASCEND_TABLE_NAME_MUST_CONTAIN is not None \
            and isinstance(ASCEND_TABLE_NAME_MUST_CONTAIN, str) \
            and ASCEND_TABLE_NAME_MUST_CONTAIN not in table_name:
        return True
    if ASCEND_TABLE_NAME_MUST_CONTAIN is not None \
            and isinstance(ASCEND_TABLE_NAME_MUST_CONTAIN, list):
        skip = True
        for key_word in ASCEND_TABLE_NAME_MUST_CONTAIN:
            if isinstance(key_word, str) and key_word in table_name:
                skip = False
                break
        return skip
    return False
