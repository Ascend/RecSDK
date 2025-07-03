#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from datetime import datetime

import pytz
import tensorflow as tf

from utils import build_optimizer, main

MODEL_NAME = "AutoInt_plus"


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
    tf.app.flags.DEFINE_integer("attention_layers", 3, "Number of attention layers")
    tf.app.flags.DEFINE_integer("att_embedding_size", 5, "Size of attention embedding")
    tf.app.flags.DEFINE_integer("heads_num", 2, "Number of attention heads")
    tf.app.flags.DEFINE_string("data_dir", '../data/criteo/', "data dir")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f'../checkpoint/criteo/{MODEL_NAME}/', "model check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
    tf.app.flags.DEFINE_string("task_type", 'train', "task type")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


def embedding_layer(feat_ids: tf.Tensor, feat_vals: tf.Tensor, feat_emb_deep: tf.Tensor, field_size: int,
                    ) -> tf.Tensor:
    """
    Build the embedding layer.

    Args:
        feat_ids (tf.Tensor): Feature IDs.
        feat_vals (tf.Tensor): Feature values.
        feat_emb_deep (tf.Tensor): Embedding weights.
        field_size (int): Number of fields.

    Returns:
        tf.Tensor: Embedding layer output.
    """
    embeddings_origin_deep = tf.nn.embedding_lookup(feat_emb_deep, feat_ids)  # None * F * E
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])  # None * F * 1
    embeddings = tf.multiply(embeddings_origin_deep, feat_vals)
    return embeddings


def multihead_attention(x: tf.Tensor, embedding_dim: int, att_embedding_size: int, heads_num: int,
                        layer_index: int) -> tf.Tensor:
    """
    Build the multihead attention layer.

    Args:
        x (tf.Tensor): Input tensor.
        embedding_dim (int): Embedding dimension.
        att_embedding_size (int): Attention embedding size.
        heads_num (int): Number of attention heads.
        layer_index (int): Layer index.

    Returns:
        tf.Tensor: Multihead attention layer output.
    """
    w_q = tf.compat.v1.get_variable(name="weight_Q_%d" % layer_index,
                                    shape=[embedding_dim, att_embedding_size * heads_num],
                                    initializer=tf.random_normal_initializer(stddev=0.1))
    w_k = tf.compat.v1.get_variable(name="weight_K_%d" % layer_index,
                                    shape=[embedding_dim, att_embedding_size * heads_num],
                                    initializer=tf.random_normal_initializer(stddev=0.1))
    w_v = tf.compat.v1.get_variable(name="weight_V_%d" % layer_index,
                                    shape=[embedding_dim, att_embedding_size * heads_num],
                                    initializer=tf.random_normal_initializer(stddev=0.1))
    w_res = tf.compat.v1.get_variable(name="weight_Res_%d" % layer_index,
                                      shape=[embedding_dim, att_embedding_size * heads_num],
                                      initializer=tf.random_normal_initializer(stddev=0.1))

    query = tf.tensordot(x, w_q, axes=(-1, 0))
    key = tf.tensordot(x, w_k, axes=(-1, 0))
    value = tf.tensordot(x, w_v, axes=(-1, 0))

    query = tf.stack(tf.split(query, heads_num, axis=2))
    key = tf.stack(tf.split(key, heads_num, axis=2))
    value = tf.stack(tf.split(value, heads_num, axis=2))

    inner_product = tf.matmul(query, key, transpose_b=True)
    inner_product /= att_embedding_size ** 0.5

    normalized_att_scores = tf.nn.softmax(inner_product, axis=-1)

    result = tf.matmul(normalized_att_scores, value)
    result = tf.concat(tf.split(result, heads_num), axis=-1)
    result = tf.squeeze(result, axis=0)

    result += tf.tensordot(x, w_res, axes=(-1, 0))
    result = tf.nn.relu(result)

    return result


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

    # Provide an estimator spec for `ModeKeys.EVAL`
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

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
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
    logger = logging.getLogger()
    log_level = getattr(logging, model_config.log_level.upper(), logging.DEBUG)
    logger.setLevel(log_level)
    console_hand = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(asctime)s: %(message)s")
    console_hand.setLevel(log_level)
    console_hand.setFormatter(formatter)
    logger.addHandler(console_hand)
    # Define the timezone for China Standard Time
    china_tz = pytz.timezone('Asia/Shanghai')
    logfile_na = MODEL_NAME + "_" + datetime.now(china_tz).strftime("%Y_%m_%d_%H_%M_%S") + ".log"
    logfile_path = os.path.join("../logs/criteo/", logfile_na)
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger), argv=[model_config])
