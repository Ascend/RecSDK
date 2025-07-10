#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple

import tensorflow as tf

from utils import define_common_flag, build_estimator_spec, main
from examples.rec_model_zoo.common import setup_logger

MODEL_NAME = "FFM"


#################### CMD Arguments ####################
def define_flags():
    model_conf = tf.app.flags.FLAGS
    define_common_flag(MODEL_NAME)
    model_conf.embedding_size = 2

    return model_conf


def logistic_regression(feat_ids: tf.Tensor, feat_vals: tf.Tensor, feat_emb_lrb_lr: tf.Tensor,
                        field_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Build logistic regression part of the model.

    Args:
        feat_ids (tf.Tensor): Feature IDs.
        feat_vals (tf.Tensor): Feature values.
        feat_emb_lrb_lr (tf.Tensor): Logistic regression weights.
        field_size (int): Number of fields.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Logistic regression part of the model.
    """
    with tf.compat.v1.variable_scope("Logistic-Regression"):
        embeddings_origin_lr = tf.nn.embedding_lookup(feat_emb_lrb_lr, feat_ids)  # None * F * 1
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])  # None * F * 1
        embeddings_lr = tf.multiply(embeddings_origin_lr, feat_vals)
        lr_bias = tf.compat.v1.get_variable(name='lr_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        lr_part = tf.reduce_sum(embeddings_lr, axis=1) + lr_bias
    return lr_part, feat_vals


def field_factorization_machine(feat_ids: tf.Tensor, feat_vals: tf.Tensor, feat_emb_ffm: List[tf.Tensor],
                                field_size: int) -> tf.Tensor:
    """
    Build field factorization machine part of the model.

    Args:
        feat_ids (tf.Tensor): Feature IDs.
        feat_vals (tf.Tensor): Feature values.
        feat_emb_ffm (list): Field factorization machine weights.
        field_size (int): Number of fields.

    Returns:
        tf.Tensor: Field factorization machine part of the model.
    """
    with tf.compat.v1.variable_scope("Field-Factorization-Machine"):
        xs = [tf.multiply(tf.nn.embedding_lookup(feat_emb_ffm[i], feat_ids), feat_vals) for i in range(field_size)]
        xs_ = tf.concat(xs, axis=1)  # None * (F * F) * E
        row, col = [], []
        for i in range(field_size - 1):
            for j in range(i + 1, field_size):
                row.append(i * field_size + j)
                col.append(j * field_size + i)
        p = tf.gather(xs_, axis=1, indices=row)
        q = tf.gather(xs_, axis=1, indices=col)
        ffm_part = tf.reduce_sum(tf.reduce_sum(p * q, axis=1), axis=1, keepdims=True)
    return ffm_part


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params.field_size
    feature_size = params.feature_size
    embedding_size = params.embedding_size
    learning_rate = params.learning_rate

    # ------bulid weights------
    feat_emb_lrb_lr = tf.compat.v1.get_variable(name="emb_lr", shape=[feature_size, 1],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )

    feat_emb_ffm = []
    for field_i in range(field_size):
        feat_emb_ffm.append(tf.compat.v1.get_variable(name="emb_ffm_%d" % field_i, shape=[feature_size, embedding_size],
                                                    initializer=tf.random_normal_initializer(stddev=0.1), ))

    # ------build feature-------
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    # ------build f(x)------
    lr_part, feat_vals = logistic_regression(feat_ids, feat_vals, feat_emb_lrb_lr, field_size)
    ffm_part = field_factorization_machine(feat_ids, feat_vals, feat_emb_ffm, field_size)

    y = lr_part + ffm_part
    y = tf.reshape(y, shape=[-1])

    return build_estimator_spec(
        y_list=[y],
        mode=mode,
        labels=labels,
        params=params,
        learning_rate=learning_rate
    )


if __name__ == "__main__":
    model_config = define_flags()
    logger = setup_logger(model_config, MODEL_NAME)

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger), argv=[model_config])
