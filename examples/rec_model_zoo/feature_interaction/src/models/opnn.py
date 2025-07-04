#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import tensorflow as tf

from utils import build_optimizer, main, setup_logger

MODEL_NAME = "OPNN"


#################### CMD Arguments ####################
def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("feature_size", 2100000, "Number of features")
    tf.app.flags.DEFINE_integer("field_size", 39, "Number of fields")
    tf.app.flags.DEFINE_integer("embedding_size", 10, "Embedding size")
    tf.app.flags.DEFINE_integer("train_size", 33003326, "Number of instances in the train set")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_string("deep_layers", '400,400,400', "deep layers")
    tf.app.flags.DEFINE_string("data_dir", '../data/criteo/', "data dir")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f'../checkpoint/criteo/{MODEL_NAME}/', "model check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
    tf.app.flags.DEFINE_string("task_type", 'train', "task type")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")

    return model_conf


def outer_product_layer(embeddings_deep: tf.Tensor, field_size: int, embedding_size: int) -> tf.Tensor:
    """
    Build the outer product layer.

    Args:
        embeddings_deep (tf.Tensor): Embedding layer output.
        field_size (int): Number of fields.
        embedding_size (int): Embedding size.

    Returns:
        tf.Tensor: Outer product output.
    """
    row, col = [], []
    for i in range(field_size - 1):
        for j in range(i + 1, field_size):
            row.append(i)
            col.append(j)
    p = tf.gather(embeddings_deep, axis=1, indices=row)
    q = tf.gather(embeddings_deep, axis=1, indices=col)
    num_pairs = int(field_size * (field_size - 1) / 2)
    kernel = tf.compat.v1.get_variable("outer_product_kernel", shape=[embedding_size, num_pairs, embedding_size],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
    tmp = tf.expand_dims(p, axis=1) * kernel
    tmp = tf.reduce_sum(tmp, axis=-1)
    tmp = tf.transpose(tmp, perm=(0, 2, 1))
    outer_product = tf.reduce_sum(tmp * q, axis=-1)
    return outer_product


def deep_layer(embeddings_deep: tf.Tensor, outer_product: tf.Tensor, field_size: int, embedding_size: int,
               layers: List[int]) -> tf.Tensor:
    """
    Build the deep layer.

    Args:
        embeddings_deep (tf.Tensor): Embedding layer output.
        outer_product (tf.Tensor): Outer product output.
        field_size (int): Number of fields.
        embedding_size (int): Embedding size.
        layers (List[int]): List of layer sizes.

    Returns:
        tf.Tensor: Deep layer output.
    """
    emb_for_deep_inputs = tf.reshape(embeddings_deep, shape=[-1, field_size * embedding_size])  # None * (F * E)
    deep_inputs = tf.concat([emb_for_deep_inputs, outer_product], axis=1)
    for layer_i, _ in enumerate(layers):
        deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[layer_i],
                                                        scope='mlp%d' % layer_i)
    y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
                                               scope='deep_out')
    return y_deep


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

    outer_product = outer_product_layer(embeddings_deep, field_size, embedding_size)
    y_deep = deep_layer(embeddings_deep, outer_product, field_size, embedding_size, layers)

    y = tf.reshape(y_deep, shape=[-1])

    pred = tf.sigmoid(y)
    predictions = {"prob": pred}
    export_outputs = {
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))

    log_loss = tf.compat.v1.losses.log_loss(labels, pred)
    auc_metric = tf.compat.v1.metrics.auc(labels, pred)
    loss_metric = tf.compat.v1.metrics.mean(log_loss)
    eval_metric_ops = {
        "auc": tf.compat.v1.metrics.auc(labels, pred),
        "logloss": tf.compat.v1.metrics.mean(log_loss),
        "stop_criterion": (auc_metric[0] - loss_metric[0], tf.group(auc_metric[1], loss_metric[1]))
    }

    # ------bulid optimizer------
    optimizer = build_optimizer(params.optimizer, learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)
    else:
        raise ValueError("Only support TRAIN, EVAL and PREDICT modes")


if __name__ == "__main__":
    model_config = define_flags()
    logger = setup_logger(model_config, MODEL_NAME)

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger), argv=[model_config])
