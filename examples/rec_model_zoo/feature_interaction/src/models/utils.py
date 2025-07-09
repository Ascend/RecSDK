#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import re
import os
import stat
import glob
import shutil
import random
from typing import Dict, List, Tuple
from datetime import date, timedelta, datetime
import logging

import pytz
import tensorflow as tf
from npu_bridge.npu_init import NPUEstimator, NPURunConfig


def define_common_flag(model_name):
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
    tf.app.flags.DEFINE_string("model_dir", f'../checkpoint/criteo/{model_name}/', "model check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
    tf.app.flags.DEFINE_string("task_type", 'train', "task type")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")


# ------ Load tfrecord dataset ------
def input_fn(filenames: List[str], batch_size: int = 32, field_size: int = 39, num_epochs: int = 1,
             perform_shuffle: bool = False) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """
    Input function for loading TFRecord dataset.

    Args:
        filenames (List[str]): List of TFRecord file paths.
        batch_size (int): Batch size.
        field_size (int): Number of fields.
        num_epochs (int): Number of epochs to repeat the dataset.
        perform_shuffle (bool): Whether to shuffle the dataset.

    Returns:
        Tuple[Dict[str, tf.Tensor], tf.Tensor]: Batch features and batch labels.
    """

    def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
            'ids': tf.io.FixedLenFeature(shape=(field_size,), dtype=tf.int64),
            'values': tf.io.FixedLenFeature(shape=(field_size,), dtype=tf.float32),
        }
        sample = tf.io.parse_example(data_record, features)
        sample['ids'] = tf.cast(sample['ids'], dtype=tf.int32)
        return {"feat_ids": sample['ids'], "feat_vals": sample['values']}, sample['label']

    dataset = tf.data.TFRecordDataset(filenames)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=500000)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True).map(extract_fn, num_parallel_calls=10).prefetch(100)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def build_optimizer(optimizer_name: str, learning_rate: float) -> tf.compat.v1.train.Optimizer:
    """
    Build the optimizer.

    Args:
        optimizer_name (str): Name of the optimizer.
        learning_rate (float): Learning rate.

    Returns:
        tf.compat.v1.train.Optimizer: Optimizer.
    """
    if optimizer_name == 'Adam':
        return tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer_name == 'Adagrad':
        return tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer_name == 'Momentum':
        return tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer_name == 'ftrl':
        return tf.compat.v1.train.FtrlOptimizer(learning_rate)
    else:
        raise ValueError("Unsupported optimizer: {}".format(optimizer_name))


def get_third_nearest_checkpoint(path):
    filenames = glob.glob(os.path.join(path, 'model.ckpt-*.index'))
    pattern = re.compile(r'model.ckpt-(.*?).index', re.S)
    versions = []
    for file in filenames:
        versions += [int(re.findall(pattern, file)[0])]
    versions_sorted = sorted(versions)
    return os.path.join(path, 'model.ckpt-' + str(versions_sorted[-3]))


def dump_pred(preds: List[Dict[str, float]], data_dir: str) -> None:
    """
    Dump the prediction results to a file.

    Args:
        preds (List[Dict[str, float]]): List of prediction results.
        data_dir (str): data save dir.

    Returns:
        None
    """
    flags = os.O_WRONLY | os.O_TRUNC | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    pred_path = os.path.join(data_dir, "pred.txt")
    with os.fdopen(os.open(pred_path, flags, modes), "w") as fo:
        for prob in preds:
            fo.write("%f\n" % (prob['prob']))


def setup_environment(model_cfg, logger):
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

    return tr_files, va_files, te_files, train_size


def setup_logger(model_config, model_name):
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
    logfile_na = model_name + "_" + datetime.now(china_tz).strftime("%Y_%m_%d_%H_%M_%S") + ".log"
    logfile_path = os.path.join("../log/criteo/", logfile_na)
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def build_estimator_spec(y_list, mode, labels, params, learning_rate):
    pred = tf.sigmoid(y_list[0])

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
    loss = 0.0
    for y in y_list:
        loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))

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


def main(model_cfg, model_fn, logger):
    tr_files, va_files, te_files, train_size = setup_environment(model_cfg, logger)

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
        raise ValueError("Unsupported task type: {}".format(model_cfg.task_type))