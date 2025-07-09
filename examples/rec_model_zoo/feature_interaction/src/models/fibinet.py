#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import tensorflow as tf

from utils import define_common_flag, build_estimator_spec, main, setup_logger

MODEL_NAME = "FiBiNet"


#################### CMD Arguments ####################
def define_flags():
    model_conf = tf.app.flags.FLAGS
    define_common_flag(MODEL_NAME)
    tf.app.flags.DEFINE_string("bilinear_type", 'interaction', "Type of bilinear interaction")
    tf.app.flags.DEFINE_integer("reduction_ratio", 3, "Reduction ratio of SE layers")

    return model_conf


def se_layer(inputs: tf.Tensor, num_fields: int, reduction_ratio: int = 3) -> tf.Tensor:
    """
    Squeeze-and-Excitation Layer.

    Args:
        inputs (tf.Tensor): Input tensor.
        num_fields (int): Number of fields.
        reduction_ratio (int): Reduction ratio for SE layers.

    Returns:
        tf.Tensor: Output tensor after applying SE layer.
    """
    reduced_size = max(1, int(num_fields / reduction_ratio))

    weights_1 = tf.compat.v1.get_variable(name="weights_1", shape=[num_fields, reduced_size],
                                          initializer=tf.random_normal_initializer(stddev=0.1))
    weights_2 = tf.compat.v1.get_variable(name="weights_2", shape=[reduced_size, num_fields],
                                          initializer=tf.random_normal_initializer(stddev=0.1))

    mean_tensor = tf.reduce_mean(inputs, axis=-1)

    activation_1 = tf.nn.relu(tf.tensordot(mean_tensor, weights_1, axes=(-1, 0)))
    activation_2 = tf.nn.relu(tf.tensordot(activation_1, weights_2, axes=(-1, 0)))

    output_tensor = tf.multiply(inputs, tf.expand_dims(activation_2, axis=2))

    return output_tensor


def bilinear_interaction(inputs: tf.Tensor, num_fields: int, embedding_dim: int, bilinear_type: str = 'interaction',
                         layer_name: str = 'se') -> tf.Tensor:
    """
    Build the bilinear interaction layer.

    Args:
        inputs (tf.Tensor): Input tensor.
        num_fields (int): Number of fields.
        embedding_dim (int): Embedding dimension.
        bilinear_type (str): Type of bilinear interaction.
        layer_name (str): Layer name.

    Returns:
        tf.Tensor: Bilinear interaction output.
    """
    row, col = [], []
    for i in range(num_fields - 1):
        for j in range(i + 1, num_fields):
            row.append(i)
            col.append(j)

    if bilinear_type == 'all':
        w_all = tf.compat.v1.get_variable(name="weight_all_%s" % layer_name,
                                          shape=[embedding_dim, embedding_dim],
                                          initializer=tf.random_normal_initializer(stddev=0.1))

        vidots = tf.matmul(inputs, w_all)

        p = tf.gather(vidots, axis=1, indices=row)
        q = tf.gather(inputs, axis=1, indices=col)

        result = tf.multiply(p, q)

    elif bilinear_type == 'each':
        w_each = tf.compat.v1.get_variable(name="weight_each_%s" % layer_name,
                                           shape=[(num_fields - 1), embedding_dim, embedding_dim],
                                           initializer=tf.random_normal_initializer(stddev=0.1))

        p = tf.gather(inputs, axis=1, indices=[i for i in range(num_fields - 1)])

        p_t = tf.transpose(p, perm=[1, 0, 2])

        vidots = tf.matmul(p_t, w_each)

        vidots_t = tf.transpose(vidots, perm=[1, 0, 2])

        v_g = tf.gather(vidots_t, axis=1, indices=row)
        q = tf.gather(inputs, axis=1, indices=col)

        result = tf.multiply(v_g, q)

    elif bilinear_type == 'interaction':
        interaction_num = (num_fields * (num_fields - 1)) / 2
        w_interaction = tf.compat.v1.get_variable(name="weight_interaction_%s" % layer_name,
                                                  shape=[interaction_num, embedding_dim, embedding_dim],
                                                  initializer=tf.random_normal_initializer(stddev=0.1))

        p = tf.gather(inputs, axis=1, indices=row)
        q = tf.gather(inputs, axis=1, indices=col)

        p_t = tf.transpose(p, perm=[1, 0, 2])

        vidots = tf.matmul(p_t, w_interaction)

        vidots_t = tf.transpose(vidots, perm=[1, 0, 2])

        result = tf.multiply(vidots_t, q)

    return result


def build_deep_layer(emb_p: tf.Tensor, se_p: tf.Tensor, layers: List[int]) -> tf.Tensor:
    """
    Build the deep layer.

    Args:
        emb_p (tf.Tensor): Embedding part.
        se_p (tf.Tensor): SE part.
        layers (List[int]): List of layer sizes.

    Returns:
        tf.Tensor: Deep layer output.
    """
    with tf.compat.v1.variable_scope("Deep-Layer"):
        deep_inputs = tf.concat([emb_p, se_p], axis=1)
        for layer_i, _ in enumerate(layers):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[layer_i],
                                                            scope='mlp%d' % layer_i)
        y = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
                                              scope='deep_out')
    return y


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    batch_size = params.batch_size
    field_size = params.field_size
    feature_size = params.feature_size
    embedding_size = params.embedding_size
    learning_rate = params.learning_rate
    reduction_ratio = params.reduction_ratio
    bilinear_type = params.bilinear_type
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

    with tf.compat.v1.variable_scope("BiInteraction-Layer"):
        se_part = se_layer(embeddings_deep, field_size, reduction_ratio)
        emb_p = tf.reshape(bilinear_interaction(embeddings_deep, field_size, embedding_size, bilinear_type, 'emb'),
                           (batch_size, -1))
        se_p = tf.reshape(bilinear_interaction(se_part, field_size, embedding_size, bilinear_type, 'se'),
                          (batch_size, -1))

    y = build_deep_layer(emb_p, se_p, layers)

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
