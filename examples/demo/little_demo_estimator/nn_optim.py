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

import tensorflow as tf
from mx_rec.util.tf_version_adapter import hccl_ops
from mx_rec.util.initialize import get_rank_size, get_use_dynamic_expansion
from mx_rec.util.variable import get_dense_and_sparse_variable
from mx_rec.optimizers.gradient_descent import create_hash_optimizer
from mx_rec.optimizers.gradient_descent_by_addr import create_hash_optimizer_by_addr
from mx_rec.util.log import logger


def get_dense_and_sparse_optimizer(cfg):
    dense_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.learning_rate)
    if get_use_dynamic_expansion():
        sparse_optimizer = create_hash_optimizer_by_addr(learning_rate=cfg.learning_rate)
        logger.info("optimizer create_hash_optimizer_by_addr")
    else:
        sparse_optimizer = create_hash_optimizer(learning_rate=cfg.learning_rate)
        logger.info("optimizer create_hash_optimizer")

    return dense_optimizer, sparse_optimizer


def get_train_op_list(losses, learning_rate):
    train_ops_list = []
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    name = None

    dense_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    use_dynamic_expansion = get_use_dynamic_expansion()
    if use_dynamic_expansion:
        sparse_optimizer = create_hash_optimizer_by_addr(learning_rate=learning_rate)
    else:
        sparse_optimizer = create_hash_optimizer(learning_rate=learning_rate)

    dense_variables, sparse_variables = get_dense_and_sparse_variable()
    trainable_variables = [dense_variables]

    for i in range(len(losses)):
        name = losses[i][0]
        loss = losses[i][1]
        with tf.control_dependencies(update_ops):
            # do dense grad
            grads = dense_optimizer.compute_gradients(loss, var_list=trainable_variables)
            dense_grads = grads[:len(dense_variables)]
            avg_grads = []
            for grad, var in dense_grads:
                if get_rank_size() > 1:
                    grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
                if grad is not None:
                    avg_grads.append((grad, var))
            # apply gradients: update variables
            train_ops_list.append(dense_optimizer.apply_gradients(avg_grads, name="dense_optimizer"))

            # do sparse optimization
            if use_dynamic_expansion:
                from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_LOCAL_EMB, ASCEND_SPARSE_LOOKUP_ID_OFFSET, \
                    ApplyGradientsStrategy, ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS
                train_emb_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB)

                if (ApplyGradientsStrategy.mapping(os.getenv("APPLY_GRADIENTS_STRATEGY")) ==
                        ApplyGradientsStrategy.SUM_SAME_ID_GRADIENTS_AND_APPLY):
                    train_address_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS)
                else:
                    train_address_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET)

                local_grads = tf.gradients(loss, train_emb_list)  # local_embedding
                grads_and_vars = [(grad, address) for grad, address in zip(local_grads, train_address_list)]
                train_ops_list.append(sparse_optimizer.apply_gradients(grads_and_vars, name='hashtable_optimizer'))
            else:
                sparse_grads = tf.gradients(loss, sparse_variables)
                grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, sparse_variables)]
                train_ops_list.append(sparse_optimizer.apply_gradients(grads_and_vars, name="sparse_optimizer"))

    global_step_op = tf.compat.v1.assign_add(tf.compat.v1.train.get_global_step(), 1)
    train_ops_list.append(global_step_op)
    return train_ops_list, name


def get_train_op(params, losses):
    train_ops = []
    op_list, name = get_train_op_list(losses, params.learning_rate)
    train_ops.append([name + '_train', tf.group(*op_list)])
    ops = [loss[1] for loss in losses] + [train_op[1] for train_op in train_ops]
    return tf.group(*ops)
