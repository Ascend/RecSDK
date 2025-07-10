#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from utils import build_estimator_spec, main
from autoint import define_flags, embedding_layer, multihead_attention
from examples.rec_model_zoo.common import setup_logger

MODEL_NAME = "AutoInt_plus"


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params.field_size
    feature_size = params.feature_size
    embedding_size = params.embedding_size
    learning_rate = params.learning_rate
    attention_layers = params.attention_layers
    heads_number = params.heads_num
    att_size = embedding_size / heads_number
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
    embeddings = embedding_layer(feat_ids, feat_vals, feat_emb_deep, field_size)
    with tf.compat.v1.variable_scope("Multihead-Attention-Layer", reuse=tf.compat.v1.AUTO_REUSE):
        attention_part = embeddings
        for i in range(attention_layers):
            attention_part = multihead_attention(x=attention_part, embedding_dim=embedding_size,
                                                 att_embedding_size=att_size, heads_num=heads_number,
                                                 layer_index=i)
    with tf.compat.v1.variable_scope("FC-Layer"):
        fc_inputs = tf.reshape(attention_part, shape=[-1, field_size * embedding_size])

        y = tf.contrib.layers.fully_connected(inputs=fc_inputs, num_outputs=1, activation_fn=tf.identity,
                                              scope='fc_out')

    with tf.compat.v1.variable_scope("Deep-Layer"):
        deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])  # None * (F * E)

        for layer_i, _ in enumerate(layers):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[layer_i],
                                                            scope='mlp%d' % layer_i)

        y_mlp = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
                                                  scope='mlp_out')

    y += y_mlp
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
    model_config.model_dir = "../checkpoint/criteo/{MODEL_NAME}/"
    logger = setup_logger(model_config, MODEL_NAME)

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger), argv=[model_config])
