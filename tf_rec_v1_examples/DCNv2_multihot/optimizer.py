# coding=utf-8
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

from delay_loss_scale import DenseLossScaleOptimizer, SparseLossScaleOptimizer

from mx_rec.util.initialize import ConfigInitializer
from rec_sdk_common.log import logger


def get_dense_and_sparse_optimizer_adam(cfg):
    from mx_rec.optimizers.lazy_adam import create_hash_optimizer
    from mx_rec.optimizers.lazy_adam_by_addr import create_hash_optimizer_by_address
    dense_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.learning_rate[0])
    use_dynamic_expansion = ConfigInitializer.get_instance().use_dynamic_expansion
    if use_dynamic_expansion:
        sparse_optimizer = create_hash_optimizer_by_address(learning_rate=cfg.learning_rate[1])
        logger.info("optimizer lazy_adam_by_addr")
    else:
        sparse_optimizer = create_hash_optimizer(learning_rate=cfg.learning_rate[1])
        logger.info("optimizer lazy_adam")
    sparse_optimizer = SparseLossScaleOptimizer(sparse_optimizer, cfg.loss_scale)
    dense_optimizer = DenseLossScaleOptimizer(dense_optimizer, cfg.loss_scale)
    return dense_optimizer, sparse_optimizer


def get_dense_and_sparse_optimizer_adagrad(cfg):
    from mx_rec.optimizers.adagrad import create_hash_optimizer
    from mx_rec.optimizers.adagrad_by_addr import create_hash_optimizer_by_address

    dense_optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=cfg.learning_rate[0],
                                                          initial_addumulator_value=0.0)
    use_dynamic_expansion = ConfigInitializer.get_instance().use_dynamic_expansion
    if use_dynamic_expansion:
        sparse_optimizer = create_hash_optimizer_by_address(learning_rate=cfg.learning_rate[1])
        logger.info("optimizer adagrad_by_addr")
    else:
        sparse_optimizer = create_hash_optimizer(learning_rate=cfg.learning_rate[1], initial_addumulator_value=0.0)
        logger.info("optimizer adagrad")
    sparse_optimizer = SparseLossScaleOptimizer(sparse_optimizer)
    dense_optimizer = DenseLossScaleOptimizer(dense_optimizer)
    return dense_optimizer, sparse_optimizer


def get_dense_and_sparse_optimizer(cfg):
    if cfg.optimizer == 'adagrad':
        return get_dense_and_sparse_optimizer_adagrad(cfg)
    elif cfg.optimizer == 'adam':
        return get_dense_and_sparse_optimizer_adam(cfg)
    else:
        raise "Not support optimize, please choose adam or adagrad"
