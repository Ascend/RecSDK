#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging

import tensorflow as tf

from mx_rec.constants.constants import ASCAnchorAttr, ASCEND_SPARSE_LOOKUP_ENTRANCE
from mx_rec.core.embedding import SparseEmbedding
from mx_rec.graph.utils import check_cutting_points, replace_anchor_vec
from mx_rec.util.initialize import get_modify_graph, get_merged_multi_lookup, insert_merged_multi_lookup


def do_merge_lookup(is_train: bool = True):
    """
    自动改图一表一查/多查，添加前向和反向节点：
        1. 如果存在一表多查的情况，则对多查的表进行lookup合并操作，并用合并后的lookup result替换原来打桩的 mock lookup result.
        2. 若不存在一表多查，则无需合并，用sparse forward得到的lookup result替换原来打桩的 mock lookup result.
        3. 自动改图模式需要执行此函数，feature spec模式直接return.
        4. 此函数在Optimizer.compute_gradients()中利用patch执行，确保train时拥有正确的梯度和计算图；eval时在改图阶段执行.

    Args:
        is_train: 当前是否为训练模式，训练模式为True，否则为False

    Returns: None

    """

    if not get_modify_graph():
        logging.debug("The `do_merge_multi_lookup` function is called only for `modify graph` mode.")
        return
    if get_merged_multi_lookup(is_train):
        logging.debug("The merge multi lookup has been executed once and does not need to be executed again.")
        return
    logging.info("start to merge multi lookup, mode(train: True, eval: False): %s.", is_train)

    # get anchor ids
    cutting_point_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE)
    if not cutting_point_list:
        raise RuntimeError("The sparse table does not have sparse lookup.")
    check_cutting_points(cutting_point_list)

    # get lookup info
    sub_cutting_points_dict = dict()
    feature_spec_name_ids_dict = dict()
    for cutting_point in cutting_point_list:
        is_training = SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.IS_TRAINING)
        if is_training != is_train:
            logging.debug("Skip! The current mode(train: True, eval: False) is %s, but the mode of %s is %s.",
                          is_train, cutting_point, is_training)
            continue

        table_instance = SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.TABLE_INSTANCE)
        if len(table_instance.lookup_name_list) > 1:
            feature_spec = SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC)
            feature_spec_name_ids_dict[feature_spec.name] = cutting_point
        if sub_cutting_points_dict.get(is_training) is None:
            sub_cutting_points_dict[is_training] = []
        sub_cutting_points_dict[is_training].append(cutting_point)

    # merge or restore lookup
    sub_cutting_point_list = sub_cutting_points_dict.get(is_train)
    if not sub_cutting_point_list:
        raise RuntimeError(f"The current mode(train: True, eval: False) is {is_train}, and the sparse table does not "
                           f"have anchor ids.")
    for cutting_point in sub_cutting_point_list:
        feature_spec = SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.FEATURE_SPEC)
        table_instance = SparseEmbedding.get_anchor_attribute(cutting_point, ASCAnchorAttr.TABLE_INSTANCE)
        send_count = table_instance.send_count
        kwargs = dict(is_train=is_train, ids=cutting_point)
        if len(table_instance.lookup_name_list) > 1:
            kwargs["multi_lookup"] = True
            kwargs["feature_spec_name_ids_dict"] = feature_spec_name_ids_dict
        lookup_result = table_instance.lookup_for_asc_with_feature_spec(feature_spec, send_count, **kwargs)
        replace_anchor_vec(cutting_point, ASCAnchorAttr.MOCK_LOOKUP_RESULT, lookup_result)
        logging.debug("The mock lookup result of %s for %s was replaced.", feature_spec.name, table_instance.table_name)

    # records whether the current mode has been merged or restored lookup
    insert_merged_multi_lookup(is_train, True)
    logging.info("finish to merge multi lookup, mode(train: True, eval: False): %s.", is_train)
