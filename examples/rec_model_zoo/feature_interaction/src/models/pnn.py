#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple

import tensorflow as tf

from utils import define_common_flag, build_estimator_spec, main
from examples.rec_model_zoo.common import setup_logger

MODEL_NAME = "PNN"


#################### CMD Arguments ####################
def define_flags():
    model_conf = tf.app.flags.FLAGS
    define_common_flag(MODEL_NAME)

    return model_conf


def inner_outer_product(embeddings_deep: tf.Tensor, field_size: int, embedding_size: int) -> Tuple[
    tf.Tensor, tf.Tensor]:
    """
    Build the inner and outer product layers.

    Args:
        embeddings_deep (tf.Tensor): Embedding layer output.
        field_size (int): Number of fields.
        embedding_size (int): Embedding size.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Inner and outer product outputs.
    """
    row, col = [], []
    for i in range(field_size - 1):
        for j in range(i + 1, field_size):
            row.append(i)
            col.append(j)
    p = tf.gather(embeddings_deep, axis=1, indices=row)
    q = tf.gather(embeddings_deep, axis=1, indices=col)
    inner_product = tf.reduce_sum(p * q, axis=2)

    num_pairs = int(field_size * (field_size - 1) / 2)
    kernel = tf.compat.v1.get_variable("outer_product_kernel", shape=[embedding_size, num_pairs, embedding_size],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
    tmp = tf.expand_dims(p, axis=1) * kernel
    tmp = tf.reduce_sum(tmp, axis=-1)
    tmp = tf.transpose(tmp, perm=(0, 2, 1))
    outer_product = tf.reduce_sum(tmp * q, axis=-1)
    return inner_product, outer_product


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params.field_size
    feature_size = params.feature_size
    embedding_size = params.embedding_size
    learning_rate = params.learning_rate
    layers = list(map(int, params.deep_layers.split(',')))

    # ------bulid weights------
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

    inner_product, outer_product = inner_outer_product(embeddings_deep, field_size, embedding_size)

    with tf.compat.v1.variable_scope("Deep-Layer"):
        emb_for_deep_inputs = tf.reshape(embeddings_deep, shape=[-1, field_size * embedding_size])  # None * (F * E)
        deep_inputs = tf.concat([emb_for_deep_inputs, inner_product, outer_product], axis=1)

        for layer_i, _ in enumerate(layers):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[layer_i],
                                                            scope='mlp%d' % layer_i)

        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
                                                   scope='deep_out')

    y = tf.reshape(y_deep, shape=[-1])

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
