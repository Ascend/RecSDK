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
