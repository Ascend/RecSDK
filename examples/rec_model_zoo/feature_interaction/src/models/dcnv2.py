#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import tensorflow as tf

from utils import build_optimizer, main, setup_logger

MODEL_NAME = "DCNv2"


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
    tf.app.flags.DEFINE_integer("cross_num", 3, "Number of cross layers")
    tf.app.flags.DEFINE_string("data_dir", '../data/criteo/', "data dir")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f'../checkpoint/criteo/{MODEL_NAME}/', "model check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
    tf.app.flags.DEFINE_string("task_type", 'train', "task type")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


def embedding_layer(feat_ids: tf.Tensor, feat_vals: tf.Tensor, feat_emb_deep: tf.Tensor, field_size: int,
                    embedding_size: int) -> tf.Tensor:
    """
    Build the embedding layer.

    Args:
        feat_ids (tf.Tensor): Feature IDs.
        feat_vals (tf.Tensor): Feature values.
        feat_emb_deep (tf.Tensor): Embedding weights.
        field_size (int): Number of fields.
        embedding_size (int): Embedding size.

    Returns:
        tf.Tensor: Embedding layer output.
    """
    embeddings_origin_deep = tf.nn.embedding_lookup(feat_emb_deep, feat_ids)  # None * F * E
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])  # None * F * 1
    embeddings_deep = tf.multiply(embeddings_origin_deep, feat_vals)
    return tf.reshape(embeddings_deep, shape=[-1, field_size * embedding_size])  # None * (F * E)


def cross_layer(cross_inputs: tf.Tensor, cross_num: int, field_size: int, embedding_size: int) -> tf.Tensor:
    """
    Build the cross layer.

    Args:
        cross_inputs (tf.Tensor): Input tensor for the cross layer.
        cross_num (int): Number of cross layers.
        field_size (int): Number of fields.
        embedding_size (int): Embedding size.

    Returns:
        tf.Tensor: Cross layer output.
    """
    cross_inputs_0 = cross_inputs
    for i in range(cross_num):
        layer_inputs = cross_inputs
        cross_inputs = tf.contrib.layers.fully_connected(inputs=cross_inputs,
                                                         num_outputs=field_size * embedding_size,
                                                         activation_fn=None, scope='cross_%d' % i)
        cross_inputs = tf.multiply(cross_inputs_0, cross_inputs)
        cross_inputs = tf.add(cross_inputs, layer_inputs)
    return cross_inputs


def deep_layer(deep_inputs: tf.Tensor, layers: List[int]) -> tf.Tensor:
    """
    Build the deep layer.

    Args:
        deep_inputs (tf.Tensor): Input tensor for the deep layer.
        layers (List[int]): List of layer sizes.

    Returns:
        tf.Tensor: Deep layer output.
    """
    for layer_i, _ in enumerate(layers):
        deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[layer_i],
                                                        scope='mlp%d' % layer_i)
    return deep_inputs


def prediction_layer(cross_inputs: tf.Tensor, deep_inputs: tf.Tensor) -> tf.Tensor:
    """
    Build the prediction layer.

    Args:
        cross_inputs (tf.Tensor): Cross layer output.
        deep_inputs (tf.Tensor): Deep layer output.

    Returns:
        tf.Tensor: Prediction layer output.
    """
    fc_inputs = tf.concat([cross_inputs, deep_inputs], axis=-1)
    y = tf.contrib.layers.fully_connected(inputs=fc_inputs, num_outputs=1, activation_fn=tf.identity, scope='fc_out')
    return tf.reshape(y, shape=[-1])


def model_fn(features, labels, mode, params):
    """
    Build Model function f(x) for Estimator.
    """
    # ------hyperparameters----
    field_size = params.field_size
    feature_size = params.feature_size
    embedding_size = params.embedding_size
    learning_rate = params.learning_rate
    cross_num = params.cross_num
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
    deep_inputs = embedding_layer(feat_ids, feat_vals, feat_emb_deep, field_size, embedding_size)
    cross_inputs = deep_inputs

    cross_inputs = cross_layer(cross_inputs, cross_num, field_size, embedding_size)
    deep_inputs = deep_layer(deep_inputs, layers)
    y = prediction_layer(cross_inputs, deep_inputs)

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

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))

    log_loss = tf.compat.v1.losses.log_loss(labels, pred)
    auc_metric = tf.compat.v1.metrics.auc(labels, pred)
    loss_metric = tf.compat.v1.metrics.mean(log_loss)
    eval_metric_ops = {
        "auc": tf.compat.v1.metrics.auc(labels, pred),
        "logloss": tf.compat.v1.metrics.mean(log_loss),
        "stop_criterion": (auc_metric[0] - loss_metric[0], tf.group(auc_metric[1], loss_metric[1]))
    }

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
        raise ValueError("Unsupported mode: {}".format(mode))


if __name__ == "__main__":
    model_config = define_flags()
    logger = setup_logger(model_config, MODEL_NAME)

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger), argv=[model_config])
