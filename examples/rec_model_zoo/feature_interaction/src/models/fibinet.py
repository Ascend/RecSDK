#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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

MODEL_NAME = "FiBiNet"


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
    tf.app.flags.DEFINE_integer("reduction_ratio", 3, "Reduction ratio of SE layers")
    tf.app.flags.DEFINE_string("bilinear_type", 'interaction', "Type of bilinear interaction")
    tf.app.flags.DEFINE_string("deep_layers", '400,400,400', "deep layers")
    tf.app.flags.DEFINE_string("data_dir", '../data/criteo/', "data dir")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f'../checkpoint/criteo/{MODEL_NAME}/', "model check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
    tf.app.flags.DEFINE_string("task_type", 'train', "task type")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")

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
        raise ValueError("Only support TRAIN/EVAL/PREDICT mode")


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
    max_steps_without_increase=train_size // model_cfg.batch_size, run_every_secs=None, run_every_steps=10)
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
        raise ValueError("Invalid task_type: %s" % model_cfg.task_type)


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
