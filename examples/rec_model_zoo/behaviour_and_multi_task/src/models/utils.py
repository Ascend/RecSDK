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

import os
import stat
import re
import glob
import json
from typing import Dict, List
import logging
from datetime import datetime
from functools import partial
import shutil

import pytz
import tensorflow as tf
from npu_bridge.npu_init import NPUEstimator, NPURunConfig

tf.app.flags.DEFINE_string("data_dir", "../data/aliccp/cast50_padded/", "data dir")
tf.app.flags.DEFINE_string("task_type", "train", "task type")
tf.app.flags.DEFINE_integer("max_seq_len", 50, "max length of sequence")


def get_third_nearest_checkpoint(path):
    """
    Retrieves the third most recent checkpoint file from the specified directory.

    This function searches for checkpoint files in the given directory that match the pattern
    'model.ckpt-*.index', extracts the version numbers from the filenames, sorts them in ascending
    order, and returns the path to the third most recent checkpoint file.

    Args:
        path (str): The directory path where checkpoint files are stored.

    Returns:
        str: The full path to the third most recent checkpoint file.

    Raises:
        IndexError: If there are fewer than three checkpoint files in the directory.

    Example:
        >>> get_third_nearest_checkpoint('/path/to/checkpoints')
        '/path/to/checkpoints/model.ckpt-12345'
    """
    filenames = glob.glob(os.path.join(path, 'model.ckpt-*.index'))
    pattern = re.compile(r'model.ckpt-(.*?).index', re.S)
    versions = []
    for filename in filenames:
        versions += [int(re.findall(pattern, filename)[0])]
    versions = sorted(versions)
    return os.path.join(path, 'model.ckpt-' + str(versions[-3]))


def json_file_load(json_name: str, json_path: str) -> dict:
    """
    Load a JSON file from the specified path.
    """
    flags = os.O_RDONLY
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    try:
        with os.fdopen(os.open(json_path, flags, modes), "r") as fp:
            json_re = json.load(fp)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{json_name} file not found: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading {json_name} file: {e}") from e

    return json_re


def dump_pred_prob(preds: List[Dict[str, float]], data_dir: str) -> None:
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


def dump_pred_multi(preds, data_dir):
    """
    Dump the prediction results to a file.
    """
    flags = os.O_WRONLY | os.O_TRUNC | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    pred_path = os.path.join(data_dir, "pred.txt")
    with os.fdopen(os.open(pred_path, flags, modes), "w") as fo:
        for prob in preds:
            fo.write("%f\t%f\t%f\n" % (prob['ctr'], prob['cvr'], prob['ctcvr']))


def embedding_lookup_sparse_fake(params: tf.Tensor, ids: tf.Tensor, combiner: str = None,
                                 name: str = None) -> tf.Tensor:
    """
    Perform sparse embedding lookup and combine the results.

    Args:
        params (tf.Tensor): The embedding parameters.
        ids (tf.Tensor): The sparse IDs to lookup.
        combiner (str, optional): The combiner method ('sum' or 'mean'). Defaults to None.
        name (str, optional): The name for the operation. Defaults to None.

    Returns:
        tf.Tensor: The combined embedding results.

    Raises:
        ValueError: If the combiner is not 'sum' or 'mean'.
    """

    # Create a dense mask where valid IDs are marked as 1.0 and invalid IDs as 0.0
    dense_mask = tf.expand_dims(tf.cast(ids >= 0, tf.float32), axis=-1)

    # Replace invalid IDs (-1) with zeros
    ids = tf.where(tf.equal(ids, -1), tf.zeros_like(ids), ids)
    embedding = tf.nn.embedding_lookup(params, ids, name=name + "_dense_lookup") * dense_mask
    summed_embedding = tf.reduce_sum(embedding, axis=1)
    if combiner == "sum":
        return summed_embedding
    elif combiner == "mean":
        return summed_embedding / tf.reduce_sum(dense_mask, axis=1)
    else:
        raise ValueError("combiner only supports 'sum' or 'mean'")


def build_feature_descriptions():
    model_config = tf.app.flags.FLAGS
    spec_json_path = os.path.join(model_config.data_dir, "spec.json")
    local_spec = json_file_load("spec", spec_json_path)

    local_feature_descriptions = {}
    for mode_type in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        key_map = {
            tf.estimator.ModeKeys.TRAIN: "train",
            tf.estimator.ModeKeys.EVAL: "val",
            tf.estimator.ModeKeys.PREDICT: "test"
        }

        feature_description = {
            'y': tf.io.FixedLenFeature([], tf.float32),
            'z': tf.io.FixedLenFeature([], tf.float32),
            'one_hot_fields': tf.io.FixedLenFeature([len(local_spec["one_hot_fields"])], tf.int64)
        }
        for mul_fields in local_spec["multi_hot_fields"]:
            feature_description[mul_fields] = tf.io.FixedLenFeature(
                [local_spec.get(f"{key_map.get(mode_type)}_max_length").get(mul_fields)],
                tf.int64)
        for mul_fields in local_spec["special_fields"]:
            feature_description[mul_fields] = tf.io.FixedLenFeature(
                [local_spec.get(f"{key_map.get(mode_type)}_max_length").get(mul_fields)],
                tf.int64)
        local_feature_descriptions[mode_type] = feature_description

    return local_spec, local_feature_descriptions


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
    logfile_path = os.path.join("../log/aliccp/", logfile_na)
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, china_tz


def parse_example(mode, example):
    parsed_example = tf.io.parse_example(example, feature_descriptions.get(mode))
    input_data = {}
    target = {"y": parsed_example["y"], "z": parsed_example["z"]}
    for index, key in enumerate(spec["one_hot_fields"]):
        input_data[key] = parsed_example["one_hot_fields"][:, index]
    for key in spec["multi_hot_fields"]:
        input_data[key] = parsed_example[key]
    for key in spec["special_fields"]:
        input_data[key] = parsed_example[key]
    return input_data, target


def input_fn(filenames, mode, batch_size=32, num_epochs=1, perform_shuffle=False):
    dataset = tf.data.TFRecordDataset(filenames)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=500000)

    dataset = dataset.repeat(num_epochs).batch(batch_size, drop_remainder=True).map(
        partial(
            parse_example,
            mode,
        ),
        num_parallel_calls=10
    ).prefetch(100)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels


def build_optimizer(loss: tf.Tensor, model_cfg: object) -> tf.Operation:
    """
    Build the optimizer for training.

    Args:
        loss (tf.Tensor): The loss tensor to minimize.
        model_cfg (object): The model configuration object containing optimizer settings.

    Returns:
        tf.Operation: The operation for applying gradients.

    Raises:
        ValueError: If the optimizer type is not supported.
    """
    if model_cfg.optimizer == "Adam":
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=model_cfg.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8
        )
    elif model_cfg.optimizer == "Adagrad":
        optimizer = tf.compat.v1.train.AdagradOptimizer(
            learning_rate=model_cfg.learning_rate, initial_accumulator_value=1e-6
        )
    elif model_cfg.optimizer == "Momentum":
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=model_cfg.learning_rate, momentum=0.95
        )
    elif model_cfg.optimizer == "SGD":
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=model_cfg.learning_rate)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(model_cfg.optimizer))

    gvs = optimizer.compute_gradients(loss)

    def clip_if_not_none(grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)

    clipped_gradients = [(clip_if_not_none(grad), var) for grad, var in gvs]
    return optimizer.apply_gradients(clipped_gradients, global_step=tf.compat.v1.train.get_global_step())


def main(model_cfg, model_fn, logger, dump_func):
    if dump_func == "prob":
        dump_pred_func = dump_pred_prob
        predict_keys = "prob"
    elif dump_func == "multi":
        dump_pred_func = dump_pred_multi
        predict_keys = ["ctr", "cvr", "ctcvr"]
    else:
        raise ValueError(f"Unsupported dump_func: {dump_func}. Choose from ['prob', 'multi']")

    train_order = json_file_load("order", "./order.json")
    tr_files = []
    for index in train_order["reading_order"]:
        tr_files.append("%strain/data_train.csv.tfrecord.%s" % (model_cfg.data_dir, index))
    va_files = glob.glob("%sval/data_val.csv.tfrecord.*" % model_cfg.data_dir)
    te_files = glob.glob("%stest/data_test.csv.tfrecord.*" % model_cfg.data_dir)

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
        save_checkpoints_steps=spec["dataset_size"]["train"] // model_cfg.batch_size + 1,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    )
    model = NPUEstimator(model_fn=model_fn, model_dir=model_cfg.model_dir, config=config, params=model_cfg)

    hook = tf.estimator.experimental.stop_if_no_increase_hook(model, "auc_ctr",
    max_steps_without_increase=spec["dataset_size"]["train"] // model_cfg.batch_size,
    run_every_secs=None, run_every_steps=10)
    hook_stop = tf.estimator.StopAtStepHook(last_step=200)

    if model_cfg.task_type == "train":
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=None, batch_size=model_cfg.batch_size, perform_shuffle=True,
                                      mode=tf.estimator.ModeKeys.TRAIN),
            hooks=[hook]
        )

        test_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                      mode=tf.estimator.ModeKeys.EVAL),
            steps=None,
            start_delay_secs=10,
            throttle_secs=0
        )
        logger.info("start train and evaluate")
        tf.estimator.train_and_evaluate(model, train_spec, test_spec)
        logger.info("early stopped, start evaluating....")
        model.evaluate(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                      mode=tf.estimator.ModeKeys.PREDICT),
            checkpoint_path=get_third_nearest_checkpoint(model.model_dir))

    elif model_cfg.task_type == "eval":
        model.evaluate(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                      mode=tf.estimator.ModeKeys.EVAL),
        )

    elif model_cfg.task_type == 'infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                        mode=tf.estimator.ModeKeys.PREDICT),
                              predict_keys=predict_keys, hooks=[])
        dump_pred_func(preds, model_cfg.data_dir)

    elif model_cfg.task_type == 'profiling_train':
        model.train(
            input_fn=lambda: input_fn(tr_files, num_epochs=1, batch_size=model_cfg.batch_size, perform_shuffle=True,
                                      mode=tf.estimator.ModeKeys.TRAIN),
            hooks=[hook_stop])

    elif model_cfg.task_type == 'profiling_infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                        mode=tf.estimator.ModeKeys.PREDICT),
                              predict_keys=predict_keys, hooks=[hook_stop])
        dump_pred_func(preds, model_cfg.data_dir)
    else:
        raise ValueError("task_type should be 'train', 'eval', 'infer', 'profiling_train' or 'profiling_infer'")


spec, feature_descriptions = build_feature_descriptions()
