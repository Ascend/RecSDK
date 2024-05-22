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

from mx_rec.util.initialize import ConfigInitializer


def get_dense_and_sparse_variable():
    dense_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    sparse_variables = tf.compat.v1.get_collection(
        ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection)
    return dense_variables, sparse_variables


def get_config_via_var(variable):
    table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance(variable)
    return table_instance
