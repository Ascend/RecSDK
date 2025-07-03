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

import pytz
import tensorflow as tf


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


def build_feature_descriptions(model_config):
    spec_json_path = os.path.join(model_config.data_dir, "spec.json")
    spec = json_file_load("spec", spec_json_path)

    feature_descriptions = {}
    for mode_type in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
        key_map = {
            tf.estimator.ModeKeys.TRAIN: "train",
            tf.estimator.ModeKeys.EVAL: "val",
            tf.estimator.ModeKeys.PREDICT: "test"
        }

        feature_description = {
            'y': tf.io.FixedLenFeature([], tf.float32),
            'z': tf.io.FixedLenFeature([], tf.float32),
            'one_hot_fields': tf.io.FixedLenFeature([len(spec["one_hot_fields"])], tf.int64)
        }
        for mul_fields in spec["multi_hot_fields"]:
            feature_description[mul_fields] = tf.io.FixedLenFeature(
                [spec[f"{key_map[mode_type]}_max_length"][mul_fields]],
                tf.int64)
        for mul_fields in spec["special_fields"]:
            feature_description[mul_fields] = tf.io.FixedLenFeature(
                [spec[f"{key_map[mode_type]}_max_length"][mul_fields]],
                tf.int64)
        feature_descriptions[mode_type] = feature_description

    return spec, feature_descriptions


def setup_logger(model_config, MODEL_NAME):
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
    logfile_path = os.path.join("../log/aliccp/", logfile_na)
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, china_tz