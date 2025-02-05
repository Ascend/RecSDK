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
import glob
import json
import random
import shutil
import logging
from datetime import datetime
from functools import partial
from typing import Dict, Tuple

import pytz
import tensorflow as tf
from npu_bridge.npu_init import NPUEstimator, NPURunConfig

from utils import get_third_nearest_checkpoint

tf.compat.v1.enable_control_flow_v2()
tf.compat.v1.enable_resource_variables()
tf.compat.v1.set_random_seed(2024)
random.seed(2024)

MODEL_NAME = "MMOE"


def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_string("expert_layers", "512,256", "expert layers")
    tf.app.flags.DEFINE_string("tower_layers", "128,64", "tower layers")
    tf.app.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
    tf.app.flags.DEFINE_string("data_dir", "../data/aliccp/cast50_padded/", "data dir")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f"../checkpoint/aliccp/{MODEL_NAME}/", "code check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", f"../model/serving/{MODEL_NAME}/",
                               "export servable code for TensorFlow Serving")
    tf.app.flags.DEFINE_string("task_type", "train", "task type")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing code or not")
    tf.app.flags.DEFINE_integer("max_seq_len", 50, "max length of sequence")
    tf.app.flags.DEFINE_integer("task_num", 2, "task num")
    tf.app.flags.DEFINE_integer("experts_num", 8, "Number of experts")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


def parse_example(mode_type: str, example: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """
    Parse a single example for the given mode type.

    Args:
        mode_type (str): The mode type (e.g., TRAIN, EVAL, PREDICT).
        example (tf.Tensor): The serialized example to parse.

    Returns:
        Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]: A tuple containing the
        input dictionary and target dictionary.
    """
    # Parse the example using the feature descriptions for the given mode type
    parsed_example = tf.io.parse_example(example, feature_descriptions.get(mode_type))

    input_dict = {}
    target = {"y": parsed_example["y"], "z": parsed_example["z"]}

    # Extract one-hot fields from the parsed example
    for index, key in enumerate(spec["one_hot_fields"]):
        input_dict[key] = parsed_example["one_hot_fields"][:, index]

    # Extract multi-hot fields from the parsed example
    for key in spec["multi_hot_fields"]:
        input_dict[key] = parsed_example[key]

    # Extract special fields from the parsed example
    for key in spec["special_fields"]:
        input_dict[key] = parsed_example[key]

    return input_dict, target


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


def input_fn(filenames: list, mode_type: str, batch_size: int = 32, num_epochs: int = 1,
             perform_shuffle: bool = False) -> tuple:
    """
    Input function to create a dataset for training, evaluation, or prediction.
    """
    dataset = tf.data.TFRecordDataset(filenames)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=500000)

    dataset = dataset.repeat(num_epochs).batch(batch_size, drop_remainder=True).map(
        partial(
            parse_example,
            mode_type,
        ),
        num_parallel_calls=10
    ).prefetch(100)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels


def dump_pred(preds, model_cfg):
    """
    Dump the prediction results to a file.
    """
    flags = os.O_WRONLY | os.O_TRUNC
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    pred_path = os.path.join(model_cfg.data_dir, "pred.txt")
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


def build_embedding_layer(features: dict, spec: dict, model_cfg: object) -> tf.Tensor:
    """
    Build the embedding layer for the model.

    Args:
        features (dict): The input features.
        spec (dict): The specification dictionary containing vocab lengths and field names.
        model_cfg (object): The model configuration object containing embedding size.

    Returns:
        tf.Tensor: The concatenated and reshaped embedding tensor.
    """
    emb_weights = {}
    for key, vocab_len in spec["vocab_length"].items():
        emb_weights[key] = tf.compat.v1.get_variable(
            name=key + "_emb_wgts",
            shape=[vocab_len + 1, model_cfg.embedding_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
        )

    embeddings = {}
    for key in ["101", "121", "122", "124", "125", "126", "127", "128", "129", "205", "206", "207", "216", "508", "509",
                "702", "301"]:
        embeddings[key] = tf.nn.embedding_lookup(emb_weights[key], features[key], name=key + "_embedding_lookup")
        embeddings[key] = tf.reshape(embeddings[key], [-1, 1, model_cfg.embedding_size])
    for key in ["109_14", "110_14", "127_14", "150_14", "210", "853"]:
        embeddings[key] = tf.expand_dims(
            embedding_lookup_sparse_fake(emb_weights[key], features[key], combiner="sum",
                                         name=key + "_embedding_lookup"),
            axis=1
        )

    embedding = tf.concat(
        [embeddings.get(field_name) for field_name in spec.get("one_hot_fields")] +
        [embeddings.get(field_name) for field_name in spec.get("multi_hot_fields")] +
        [embeddings.get(field_name) for field_name in spec.get("special_fields")],
        axis=2,
    )

    # sparse feature class num
    embedding_feature_num = 23
    return tf.reshape(embedding, [-1, embedding_feature_num * model_cfg.embedding_size])


def build_experts(x_deep: tf.Tensor, model_cfg: object) -> tf.Tensor:
    """
    Build the experts for the model.

    Args:
        x_deep (tf.Tensor): The input tensor.
        model_cfg (object): The model configuration object containing expert layers and number of experts.

    Returns:
        tf.Tensor: The concatenated experts tensor.
    """
    experts = []
    expert_units = list(map(int, model_cfg.expert_layers.strip().split(',')))
    for expert_i in range(model_cfg.experts_num):
        y_dnn = x_deep
        for mlp_j, _ in enumerate(expert_units):
            y_dnn = tf.contrib.layers.fully_connected(inputs=y_dnn, num_outputs=expert_units[mlp_j],
                                                      activation_fn=tf.nn.relu,
                                                      scope='expert_%d_mlp_%d' % (expert_i, mlp_j))
        experts.append(tf.expand_dims(y_dnn, axis=1))
    return tf.concat(experts, axis=1)


def build_gate_networks(x_deep: tf.Tensor, model_cfg: object) -> list:
    """
    Build the gate networks for the model.

    Args:
        x_deep (tf.Tensor): The input tensor.
        model_cfg (object): The model configuration object containing task number and number of experts.

    Returns:
        list: A list of gate networks tensors.
    """
    gate_networks = []
    for i in range(model_cfg.task_num):
        gate_network = tf.contrib.layers.fully_connected(
            inputs=x_deep,
            num_outputs=model_cfg.experts_num,
            activation_fn=tf.nn.softmax,
            scope='gate_%d_mlp' % i)
        gate_network_shape = gate_network.get_shape().as_list()
        gate_network = tf.reshape(gate_network, shape=[-1, gate_network_shape[1], 1])
        gate_networks.append(gate_network)
    return gate_networks


def build_task_outputs(experts: tf.Tensor, gate_networks: list) -> list:
    """
    Build the task outputs by combining experts and gate networks.

    Args:
        experts (tf.Tensor): The tensor containing expert outputs.
        gate_networks (list): A list of gate network tensors.

    Returns:
        list: A list of reshaped task output tensors.
    """
    task_outputs = []
    for gate_network in gate_networks:
        task_out = tf.multiply(experts, gate_network)
        task_out_shape = task_out.get_shape().as_list()
        task_outputs.append(tf.reshape(task_out, shape=[-1, task_out_shape[1] * task_out_shape[2]]))
    return task_outputs


def build_tower(tower_input: tf.Tensor, name: str, model_cfg: object) -> tf.Tensor:
    """
    Build the tower network for a specific task.

    Args:
        tower_input (tf.Tensor): The input tensor for the tower.
        name (str): The name of the tower.
        model_cfg (object): The model configuration object containing tower layers.

    Returns:
        tf.Tensor: The output tensor of the tower network.
    """
    tower_units = list(map(int, model_cfg.tower_layers.strip().split(',')))
    y_tower = tower_input
    for tower_i, _ in enumerate(tower_units):
        y_tower = tf.contrib.layers.fully_connected(inputs=y_tower, num_outputs=tower_units[tower_i],
                                                    activation_fn=tf.nn.relu,
                                                    scope=name + '_tower_mlp_%d' % tower_i)
    return y_tower


def build_predictions(task_outputs: list, model_cfg: object) -> dict:
    """
    Build the predictions for the model.

    Args:
        task_outputs (list): A list of task output tensors.
        model_cfg (object): The model configuration object containing tower layers.

    Returns:
        dict: A dictionary containing the predictions for ctr, cvr, and ctcvr.
    """
    y_ctr = build_tower(task_outputs[0], name='ctr', model_cfg=model_cfg)
    y_ctr = tf.contrib.layers.fully_connected(inputs=y_ctr, num_outputs=1, activation_fn=None, scope='deep_out_click')
    y_ctr = tf.reshape(y_ctr, [-1, ])
    y_ctr_prediction = tf.sigmoid(y_ctr)

    y_cvr = build_tower(task_outputs[1], name='cvr', model_cfg=model_cfg)
    y_cvr = tf.contrib.layers.fully_connected(inputs=y_cvr, num_outputs=1, activation_fn=None,
                                              scope='deep_out_valid_play')
    y_cvr = tf.reshape(y_cvr, [-1, ])
    y_cvr_prediction = tf.sigmoid(y_cvr)

    y_ctcvr_prediction = y_ctr_prediction * y_cvr_prediction

    predictions = {
        "ctr": y_ctr_prediction,
        "cvr": y_cvr_prediction,
        "ctcvr": y_ctcvr_prediction
    }
    return predictions


def build_loss(labels: dict, y_ctr_prediction: tf.Tensor, y_ctcvr_prediction: tf.Tensor,
               model_cfg: object) -> tf.Tensor:
    """
    Build the loss function for the model.

    Args:
        labels (dict): A dictionary containing the true labels for ctr and ctcvr.
        y_ctr_prediction (tf.Tensor): The predicted ctr values.
        y_ctcvr_prediction (tf.Tensor): The predicted ctcvr values.
        model_cfg (object): The model configuration object containing loss weights.

    Returns:
        tf.Tensor: The combined loss tensor.
    """
    epsilon = 1e-7
    # Weight for the click-through rate (CTR) loss component
    click_weight = 0.14
    # Weight for the conversion rate (CVR) loss component
    conversion_weight = 0.023
    # Weight for the CTR task in the combined loss function
    ctr_task_wgt = model_cfg.ctr_task_wgt

    ctr_loss = - (1 - click_weight) / click_weight * labels['y'] * tf.math.log(y_ctr_prediction + epsilon) - \
               (1 - labels['y']) * tf.math.log(1 - y_ctr_prediction + epsilon)
    ctr_loss = tf.reduce_mean(ctr_loss)

    ctcvr_loss = - (1 - conversion_weight) / conversion_weight * labels['z'] * tf.math.log(
        y_ctcvr_prediction + epsilon) - \
                 (1 - labels['z']) * tf.math.log(1 - y_ctcvr_prediction + epsilon)
    ctcvr_loss = tf.reduce_mean(ctcvr_loss)

    return ctr_task_wgt * ctr_loss + (1 - ctr_task_wgt) * ctcvr_loss


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

    def clip_grad(grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)

    clipped_gradients = [(clip_grad(grad), var) for grad, var in gvs]
    return optimizer.apply_gradients(clipped_gradients, global_step=tf.compat.v1.train.get_global_step())


def model_fn(features: dict, labels: dict, mode: tf.estimator.ModeKeys,
             params: object) -> tf.estimator.EstimatorSpec:
    """
    Build the model function for the estimator.

    Args:
        features (dict): The input features.
        labels (dict): The true labels.
        mode (tf.estimator.ModeKeys): The mode (TRAIN, EVAL, PREDICT).
        params (object): The model configuration object.

    Returns:
        tf.estimator.EstimatorSpec: The EstimatorSpec object for the given mode.
    """
    # Build the embedding layer
    x_deep = build_embedding_layer(features, spec, params)

    # Build the experts
    experts = build_experts(x_deep, params)

    # Build the gate networks
    gate_networks = build_gate_networks(x_deep, params)

    # Build the task outputs
    task_outputs = build_task_outputs(experts, gate_networks)

    # Build the predictions
    predictions = build_predictions(task_outputs, params)

    # Define the export outputs for serving
    export_outputs = {
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)
    }
    loss = build_loss(labels, predictions["ctr"], predictions["ctcvr"], params)
    train_op = build_optimizer(loss, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
    elif mode == tf.estimator.ModeKeys.EVAL:
        ctr_mask = labels["y"] > 0
        cvr_labels = tf.boolean_mask(labels["z"], ctr_mask)
        cvr_pre = tf.boolean_mask(predictions["cvr"], ctr_mask)

        eval_metric_ops = {
            "auc_ctr": tf.compat.v1.metrics.auc(labels["y"], predictions["ctr"]),
            "auc_cvr": tf.compat.v1.metrics.auc(cvr_labels, cvr_pre),
            "auc_ctcvr": tf.compat.v1.metrics.auc(labels["z"], predictions["ctcvr"])
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
    else:
        raise ValueError("Unsupported mode: {}".format(mode))


def main(model_cfg):
    model_cfg.model_dir = model_cfg.model_dir + datetime.now(china_tz).strftime('%Y%m%d')

    train_order = json_file_load("train_order", "./order.json")

    tr_files = [
        "%strain/data_train.csv.tfrecord.%s" % (model_cfg.data_dir, index)
        for index in train_order["reading_order"]
    ]
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
                                      mode_type=tf.estimator.ModeKeys.TRAIN),
            hooks=[hook]
        )

        test_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                      mode_type=tf.estimator.ModeKeys.EVAL),
            steps=None,
            start_delay_secs=10,
            throttle_secs=0
        )
        logger.info("start train and evaluate")
        tf.estimator.train_and_evaluate(model, train_spec, test_spec)
        logger.info("early stopped, start evaluating....")
        model.evaluate(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                      mode_type=tf.estimator.ModeKeys.PREDICT),
            checkpoint_path=get_third_nearest_checkpoint(model.model_dir))

    elif model_cfg.task_type == "eval":
        model.evaluate(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size)
        )

    elif model_cfg.task_type == 'infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                        mode_type=tf.estimator.ModeKeys.PREDICT),
                              predict_keys=["ctr", "cvr", "ctcvr"], hooks=[])
        dump_pred(preds, model_cfg)

    elif model_cfg.task_type == 'profiling_train':
        model.train(
            input_fn=lambda: input_fn(tr_files, num_epochs=1, batch_size=model_cfg.batch_size, perform_shuffle=True,
                                      mode_type=tf.estimator.ModeKeys.TRAIN),
            hooks=[hook_stop])

    elif model_cfg.task_type == 'profiling_infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                        mode_type=tf.estimator.ModeKeys.PREDICT),
                              predict_keys=["ctr", "cvr", "ctcvr"], hooks=[hook_stop])

        dump_pred(preds, model_cfg)
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
    logfile_path = os.path.join("../logs/aliccp/", logfile_na)
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("FLAGS: " + str(model_config))

    spec_json_path = os.path.join(model_config.data_dir, "spec.json")
    spec = json_file_load("spec", spec_json_path)

    feature_descriptions = {}
    for mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
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
        try:
            for mul_fields in spec.get("multi_hot_fields"):
                feature_description[mul_fields] = tf.io.FixedLenFeature(
                    [spec[f"{key_map[mode]}_max_length"][mul_fields]],
                    tf.int64)
            for mul_fields in spec["special_fields"]:
                feature_description[mul_fields] = tf.io.FixedLenFeature(
                    [spec[f"{key_map[mode]}_max_length"][mul_fields]],
                    tf.int64)
        except KeyError as e_key:
            raise KeyError("Spec file Error, please check spec.json,  error description: {}".format(e_key)) from e_key
        except Exception as e_info:
            raise RuntimeError("Error loading feature description: {}".format(e_info)) from e_info

        feature_descriptions[mode] = feature_description

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0]), argv=[model_config])
