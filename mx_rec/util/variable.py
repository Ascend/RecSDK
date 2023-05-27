#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import tensorflow as tf
from tensorflow.python.framework import ops

from mx_rec.util.initialize import get_table_instance, get_ascend_global_hashtable_collection


def get_dense_and_sparse_variable():
    dense_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    sparse_variables = tf.compat.v1.get_collection(get_ascend_global_hashtable_collection())
    return dense_variables, sparse_variables


def remove_saving_var(variable):
    global_variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    savable_objects = ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS)
    if variable in global_variables:
        global_variables.remove(variable)

    if variable in savable_objects:
        savable_objects.remove(variable)


def check_and_get_config_via_var(variable, optimizer_type: str):
    table_instance = get_table_instance(variable)

    if not table_instance.skip_emb_transfer and not table_instance.optimizer:
        raise EnvironmentError(f"When ASC with DDR, you must pass the '{optimizer_type}' optimizer instances to the"
                               f" init method of SparseEmbedding.")

    return table_instance


def check_param_range(name, value, min_border, max_border):
    if value > max_border or value < min_border:
        raise ValueError(f"Please offer a {name} between [{min_border}, {max_border}].")

    return


def check_param_type(name, value, legal_type):
    if not isinstance(value, legal_type):
        raise TypeError(f"Please offer a {name} within types: {legal_type}.")

    return

