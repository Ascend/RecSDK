#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import List, Tuple
from datetime import datetime

import pytz
import tensorflow as tf

from utils import build_optimizer, main

MODEL_NAME = "FFM"


#################### CMD Arguments ####################
def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("feature_size", 2100000, "Number of features")
    tf.app.flags.DEFINE_integer("field_size", 39, "Number of fields")
    tf.app.flags.DEFINE_integer("embedding_size", 2, "Embedding size")
    tf.app.flags.DEFINE_integer("train_size", 33003326, "Number of instances in the train set")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_string("data_dir", '../data/criteo/', "data dir")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f'../checkpoint/criteo/{MODEL_NAME}/', "model check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
    tf.app.flags.DEFINE_string("task_type", 'train', "task type")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
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

    pred = tf.sigmoid(y)
    predictions = {"prob": pred}
    export_outputs = {
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

    # Provide an estimator spec for `ModeKeys.PREDICT`
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

    optimizer = build_optimizer(params.optimizer, learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            train_op=train_op)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)
    else:
        raise ValueError("Unsupported mode: {}".format(mode))


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
