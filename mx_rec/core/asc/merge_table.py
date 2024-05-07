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

from typing import Dict, List

import tensorflow as tf
from tensorflow import Operation, Tensor

from mx_rec.constants.constants import MAX_WHILE_SIZE, TFDevice, ASCEND_TABLE_NAME_MUST_CONTAIN
from mx_rec.util.global_env_conf import global_env
from mx_rec.util.initialize import ConfigInitializer
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
    bool_gauge_set = ConfigInitializer.get_instance().train_params_config.bool_gauge_set
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
                      table_reachable_tensor: Dict[str, List[Tensor]]) -> None:
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
               spread_tensors: List[Tensor]) -> None:
        """extend the tensors which table lookup op can reach

        :param op_list: all op in the graph
        :param tensor: the tensor visited by bfs
        :param spread_tensors: the list of tensors which table lookup op can reach
        :return:
        """
        for the_op in op_list:
            if tensor in the_op.inputs:
                spread_tensors.extend(the_op.outputs)

    def bfs_lookup(next_to_visit: List[Tensor]) -> (set, bool):
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

    enable_table_merge = True if global_env.tf_device == TFDevice.NPU.value else False
    if not enable_table_merge:
        return []

    op_list = tf.compat.v1.get_default_graph().get_operations()

    table_lookup_op = {}
    table_reachable_tensor = {}

    for _, table_instance in ConfigInitializer.get_instance().sparse_embed_config.table_instance_dict.items():
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
            ConfigInitializer.get_instance().sparse_embed_config.insert_dangling_table(table_name)

    for table_name, table_op in table_reachable_tensor.items():
        reach_op, found = bfs_lookup(table_op)
        if not found and affirm(reach_op):
            dangling_table.append(table_name)
            ConfigInitializer.get_instance().sparse_embed_config.insert_dangling_table(table_name)
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


def check_dangling_table():
    """
    If the dangling_table list is empty(maybe feature_spec mode), try to find again
    :return: list of dangling_table
    """
    config_instance = ConfigInitializer.get_instance()
    dangling_table = config_instance.sparse_embed_config.dangling_table
    if not dangling_table:
        table_names = []
        for _, table_instance in config_instance.sparse_embed_config.table_instance_dict.items():
            table_names.append(table_instance.table_name)
        dangling_table = find_dangling_table(table_names)

    return dangling_table
