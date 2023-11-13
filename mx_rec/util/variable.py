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


def check_and_get_config_via_var(variable, optimizer_type: str):
    table_instance = get_table_instance(variable)

    if not table_instance.skip_emb_transfer and not table_instance.optimizer:
        raise EnvironmentError(f"When ASC with DDR, you must pass the '{optimizer_type}' optimizer instances to the"
                               f" init method of SparseEmbedding.")

    return table_instance
