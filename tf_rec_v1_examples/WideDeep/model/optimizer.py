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
from mx_rec.optimizers.lazy_adam import create_hash_optimizer
from mx_rec.optimizers.lazy_adam_by_addr import create_hash_optimizer_by_address


def get_dense_and_sparse_optimizer(cfg):
    dense_optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate[0])
    use_dynamic_expansion = ConfigInitializer.get_instance().use_dynamic_expansion
    if use_dynamic_expansion:
        sparse_optimizer = create_hash_optimizer_by_address(learning_rate=cfg.learning_rate[1])
    else:
        sparse_optimizer = create_hash_optimizer(learning_rate=cfg.learning_rate[1])
    loss_scale = 1
    sparse_optimizer = SparseLossScaleOptimizer(sparse_optimizer, loss_scale)
    dense_optimizer = DenseLossScaleOptimizer(dense_optimizer, loss_scale)

    return dense_optimizer, sparse_optimizer
