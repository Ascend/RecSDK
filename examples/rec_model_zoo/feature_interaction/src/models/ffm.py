#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import stat
import glob
import shutil
import random
import logging
from typing import List, Tuple, Dict
from datetime import date, timedelta, datetime

import pytz
import tensorflow as tf
from npu_bridge.npu_init import NPUEstimator, NPURunConfig

from utils import (
    get_third_nearest_checkpoint,
    dump_pred,
    input_fn,
    build_optimizer
)

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


def main(model_cfg):
    # ------check Arguments------
    if model_cfg.dt_dir == "":
        model_cfg.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    model_cfg.model_dir = model_cfg.model_dir + model_cfg.dt_dir

    # ------init Envs------
    tr_files = glob.glob("%s/tr*tfrecords" % model_cfg.data_dir)
    random.shuffle(tr_files)
    va_files = glob.glob("%s/va*tfrecords" % model_cfg.data_dir)
    te_files = glob.glob("%s/te*tfrecords" % model_cfg.data_dir)

    train_size = model_cfg.train_size

    if model_cfg.clear_existing_model:
        if os.path.exists(model_cfg.model_dir):
            try:
                shutil.rmtree(model_cfg.model_dir)
            except PermissionError as e:
                raise PermissionError("Permission denied: {}".format(e)) from e
            except Exception as e:
                raise RuntimeError("Error clearing existing model: {}".format(e)) from e
        else:
            logger.warning("Model directory does not exist, skipping deletion.")

    # ------ for NPU  ------
    config = NPURunConfig(
        model_dir=model_cfg.model_dir,
        log_step_count_steps=100, save_summary_steps=100,
        save_checkpoints_steps=train_size // model_cfg.batch_size + 1,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    )
    estimator = NPUEstimator(model_fn=model_fn, model_dir=model_cfg.model_dir, params=model_cfg, config=config)

    hook = tf.estimator.experimental.stop_if_no_increase_hook(estimator, "stop_criterion",
    max_steps_without_increase=train_size // model_cfg.batch_size,
    run_every_secs=None, run_every_steps=10)
    hook_stop = tf.estimator.StopAtStepHook(last_step=200)
    os.makedirs(estimator.eval_dir())
    if model_cfg.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=None, batch_size=model_cfg.batch_size,
                                      field_size=model_cfg.field_size, perform_shuffle=True),
            hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                      field_size=model_cfg.field_size), steps=None,
            start_delay_secs=10, throttle_secs=0)
        logger.info("start train and evaluate")
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        logger.info("Early stopped, start evaluation...")
        estimator.evaluate(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                     field_size=model_cfg.field_size),
                           checkpoint_path=get_third_nearest_checkpoint(estimator.model_dir))

    elif model_cfg.task_type == 'eval':
        estimator.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                     field_size=model_cfg.field_size))

    elif model_cfg.task_type == 'infer':
        preds = estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                            field_size=model_cfg.field_size),
                                  predict_keys="prob")
        dump_pred(preds, model_cfg.data_dir)

    elif model_cfg.task_type == 'profiling_train':
        estimator.train(input_fn=lambda: input_fn(tr_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                  field_size=model_cfg.field_size, perform_shuffle=True),
                        hooks=[hook_stop])

    elif model_cfg.task_type == 'profiling_infer':
        preds = estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                            field_size=model_cfg.field_size),
                                  predict_keys="prob", hooks=[hook_stop])
        dump_pred(preds, model_cfg.data_dir)
    else:
        raise ValueError("Unsupported task type: {}".format(model_cfg.task_type))


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
    tf.compat.v1.app.run(main=lambda argv: main(argv[0]), argv=[model_config])
