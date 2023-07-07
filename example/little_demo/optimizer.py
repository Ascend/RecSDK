# coding: UTF-8

import logging

import tensorflow as tf

from mx_rec.optimizers.lazy_adam import create_hash_optimizer
from mx_rec.optimizers.lazy_adam_by_addr import create_hash_optimizer_by_address
from mx_rec.util.initialize import get_use_dynamic_expansion


def create_dense_and_sparse_optimizer(cfg):
    dense_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.learning_rate)
    use_dynamic_expansion = get_use_dynamic_expansion()
    if use_dynamic_expansion:
        sparse_optimizer = create_hash_optimizer_by_address(learning_rate=cfg.learning_rate)
        logging.info("optimizer lazy_adam_by_addr")
    else:
        sparse_optimizer = create_hash_optimizer(learning_rate=cfg.learning_rate)
        logging.info("optimizer lazy_adam")

    return dense_optimizer, sparse_optimizer
