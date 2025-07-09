#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from utils import define_common_flag, build_estimator_spec, main, setup_logger

MODEL_NAME = "FM"


#################### CMD Arguments ####################
def define_flags():
    model_conf = tf.app.flags.FLAGS
    define_common_flag(MODEL_NAME)

    return model_conf


def build_logistic_regression(feat_ids: tf.Tensor, feat_vals: tf.Tensor, feat_emb_lr: tf.Tensor) -> tf.Tensor:
    """
    Build the logistic regression model.

    Args:
        feat_ids (tf.Tensor): Feature IDs.
        feat_vals (tf.Tensor): Feature values.
        feat_emb_lr (tf.Tensor): Embedding weights for logistic regression.

    Returns:
        tf.Tensor: Logistic regression output.
    """
    with tf.compat.v1.variable_scope("Logistic-Regression"):
        embeddings_origin_lr = tf.nn.embedding_lookup(feat_emb_lr, feat_ids)  # None * F * 1
        embeddings_lr = tf.multiply(embeddings_origin_lr, feat_vals)
        lr_bias = tf.compat.v1.get_variable(name='lr_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        first_order = tf.reduce_sum(embeddings_lr, axis=1) + lr_bias
    return first_order


def build_factorization_machine(embeddings_deep: tf.Tensor) -> tf.Tensor:
    """
    Build the factorization machine model.

    Args:
        embeddings_deep (tf.Tensor): Embedding layer output.

    Returns:
        tf.Tensor: Factorization machine output.
    """
    with tf.compat.v1.variable_scope("Factorization-Machine"):
        square_of_sum = tf.square(tf.reduce_sum(embeddings_deep, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(embeddings_deep * embeddings_deep, axis=1, keepdims=True)
        second_order = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=2, keepdims=False)
    return second_order


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params.field_size
    feature_size = params.feature_size
    embedding_size = params.embedding_size
    learning_rate = params.learning_rate

    # ------bulid weights------
    feat_emb_lr = tf.compat.v1.get_variable(name="emb_lr", shape=[feature_size, 1],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
    feat_emb_deep = tf.compat.v1.get_variable(name="emb_deep", shape=[feature_size, embedding_size],
                                              initializer=tf.random_normal_initializer(stddev=0.1), )

    # ------build feature-------
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    # ------build f(x)------
    with tf.compat.v1.variable_scope("Embedding-Layer"):
        embeddings_origin_deep = tf.nn.embedding_lookup(feat_emb_deep, feat_ids)  # None * F * E
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])  # None * F * 1
        embeddings_deep = tf.multiply(embeddings_origin_deep, feat_vals)

    first_order = build_logistic_regression(feat_ids, feat_vals, feat_emb_lr)
    second_order = build_factorization_machine(embeddings_deep)

    y = first_order + second_order
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
