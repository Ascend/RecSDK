#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import random
import shutil
import logging
from datetime import datetime, date, timedelta
from functools import partial

import pytz
import tensorflow as tf
from npu_bridge.npu_init import NPUEstimator, NPURunConfig

from utils import (
    get_third_nearest_checkpoint,
    json_file_load,
    dump_pred_multi,
    embedding_lookup_sparse_fake
)

tf.compat.v1.enable_control_flow_v2()
tf.compat.v1.enable_resource_variables()
tf.compat.v1.set_random_seed(2024)
random.seed(2024)

MODEL_NAME = "SharedBottom"


def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_string("deep_layers", "512,256", "deep layers")
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
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


def parse_example(mode, example):
    parsed_exapmle = tf.io.parse_example(example, feature_descriptions[mode])
    input_data = {}
    target = {"y": parsed_exapmle["y"], "z": parsed_exapmle["z"]}
    for index, key in enumerate(spec["one_hot_fields"]):
        input_data[key] = parsed_exapmle["one_hot_fields"][:, index]
    for key in spec["multi_hot_fields"]:
        input_data[key] = parsed_exapmle[key]
    for key in spec["special_fields"]:
        input_data[key] = parsed_exapmle[key]
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


def build_optimizer(model_cfg) -> tf.compat.v1.train.Optimizer:
    """
    Build the optimizer based on the model configuration.

    Args:
        model_cfg: Model configuration containing optimizer type and learning rate.

    Returns:
        tf.compat.v1.train.Optimizer: The configured optimizer.

    Raises:
        ValueError: If the optimizer type is not supported.
    """

    if model_cfg.optimizer == "Adam":
        return tf.compat.v1.train.AdamOptimizer(
            learning_rate=model_cfg.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8
        )
    elif model_cfg.optimizer == "Adagrad":
        return tf.compat.v1.train.AdagradOptimizer(
            learning_rate=model_cfg.learning_rate, initial_accumulator_value=1e-6
        )
    elif model_cfg.optimizer == "Momentum":
        return tf.compat.v1.train.MomentumOptimizer(
            learning_rate=model_cfg.learning_rate, momentum=0.95
        )
    elif model_cfg.optimizer == "SGD":
        return tf.compat.v1.train.GradientDescentOptimizer(learning_rate=model_cfg.learning_rate)
    else:
        raise ValueError("Optimizer not supported: {}".format(model_cfg.optimizer))


def model_fn(features, labels, mode, params):
    """build Estimator model"""

    with tf.compat.v1.variable_scope("Embedding-Layer"):
        emb_weights = {}
        for key, vocab_len in spec["vocab_length"].items():
            emb_weights[key] = tf.compat.v1.get_variable(
                name=key + "_emb_wgts",
                shape=[vocab_len + 1, params.embedding_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
            )

        embeddings = {}
        for key in ["101", "121", "122", "124", "125", "126", "127", "128", "129",
                    "205", "206", "207", "216", "508", "509", "702", "301"]:
            embeddings[key] = tf.nn.embedding_lookup(emb_weights.get(key), features.get(key),
                                                     name=key + "_embedding_lookup")
            embeddings[key] = tf.reshape(embeddings[key], [-1, 1, params.embedding_size])
        for key in ["109_14", "110_14", "127_14", "150_14", "210", "853"]:
            embeddings[key] = tf.expand_dims(
                embedding_lookup_sparse_fake(emb_weights.get(key), features.get(key), combiner="sum",
                                             name=key + "_embedding_lookup"),
                axis=1
            )

    embedding = tf.concat(
        [embeddings.get(field_name) for field_name in spec["one_hot_fields"]] +
        [embeddings.get(field_name) for field_name in spec["multi_hot_fields"]] +
        [embeddings.get(field_name) for field_name in spec["special_fields"]],
        axis=2,
    )  # None * 1 * (23 * E)

    x_deep = tf.reshape(embedding, [-1, 23 * params.embedding_size])

    with tf.compat.v1.variable_scope("shared-bottom"):
        bottom_units = list(map(int, params.deep_layers.strip().split(',')))
        bottom_output = x_deep
        for bottom_i, _ in enumerate(bottom_units):
            bottom_output = tf.contrib.layers.fully_connected(inputs=bottom_output, num_outputs=bottom_units[bottom_i],
                                                              activation_fn=tf.nn.relu,
                                                              scope='bottom_mlp_%d' % bottom_i)

    with tf.compat.v1.variable_scope("tower"):
        tower_units = list(map(int, params.tower_layers.strip().split(',')))

        def build_tower(tower_input, name):
            y_tower = tower_input
            for tower_i, _ in enumerate(tower_units):
                y_tower = tf.contrib.layers.fully_connected(inputs=y_tower, num_outputs=tower_units[tower_i],
                                                            activation_fn=tf.nn.relu,
                                                            scope=name + '_tower_mlp_%d' % tower_i)
            return y_tower

        # CTR
        y_ctr = build_tower(bottom_output, name='ctr')
        y_ctr = tf.contrib.layers.fully_connected(inputs=y_ctr, num_outputs=1, activation_fn=None,
                                                  scope='deep_out_click')
        y_ctr = tf.reshape(y_ctr, [-1, ])
        y_ctr_prediction = tf.sigmoid(y_ctr)

        # CVR
        y_cvr = build_tower(bottom_output, name='cvr')
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

    export_outputs = {
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions
        )
    }
    # Estimator predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, export_outputs=export_outputs
        )

    # ------build loss function------
    with tf.compat.v1.variable_scope("loss-function-part"):
        epsilon = 1e-7
        click_weight = 0.14
        conversion_weight = 0.023
        ctr_task_wgt = params.ctr_task_wgt

        ctr_loss = - (1 - click_weight) / click_weight * labels['y'] * tf.math.log(y_ctr_prediction + epsilon) - \
                   (1 - labels['y']) * tf.math.log(1 - y_ctr_prediction + epsilon)
        ctr_loss = tf.reduce_mean(ctr_loss)

        ctcvr_loss = - (1 - conversion_weight) / conversion_weight * labels['z'] * tf.math.log(
            y_ctcvr_prediction + epsilon) - \
                     (1 - labels['z']) * tf.math.log(1 - y_ctcvr_prediction + epsilon)
        ctcvr_loss = tf.reduce_mean(ctcvr_loss)

        loss = ctr_task_wgt * ctr_loss + (1 - ctr_task_wgt) * ctcvr_loss

    # Provide an estimator spec for `ModeKeys.EVAL`
    if mode == tf.estimator.ModeKeys.EVAL:
        ctr_mask = labels["y"] > 0
        cvr_labels = tf.boolean_mask(labels["z"], ctr_mask)
        cvr_pre = tf.boolean_mask(y_cvr_prediction, ctr_mask)

        eval_metric_ops = {
            "auc_ctr": tf.compat.v1.metrics.auc(labels["y"], y_ctr_prediction),
            "auc_cvr": tf.compat.v1.metrics.auc(cvr_labels, cvr_pre),
            "auc_ctcvr": tf.compat.v1.metrics.auc(labels["z"], y_ctcvr_prediction)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
        )

    # ------bulid optimizer------
    optimizer = build_optimizer(params)

    gvs = optimizer.compute_gradients(loss)

    def clip_if_not_none(grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)

    clipped_gradients = [(clip_if_not_none(grad), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(clipped_gradients, global_step=tf.compat.v1.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def main(model_cfg):
    if model_cfg.dt_dir == "":
        model_cfg.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    model_cfg.model_dir = model_cfg.model_dir + (date.today() + timedelta(-1)).strftime('%Y%m%d')

    train_order = json_file_load("train_order", "./order.json")
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

    spec_json_path = os.path.join(model_config.data_dir, "spec.json")
    spec = json_file_load("spec", spec_json_path)

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
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size)
        )

    elif model_cfg.task_type == 'infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                        mode=tf.estimator.ModeKeys.PREDICT),
                              predict_keys=["ctr", "cvr", "ctcvr"], hooks=[])
        dump_pred_multi(preds, model_cfg.data_dir)

    elif model_cfg.task_type == 'profiling_train':
        model.train(
            input_fn=lambda: input_fn(tr_files, num_epochs=1, batch_size=model_cfg.batch_size, perform_shuffle=True,
                                      mode=tf.estimator.ModeKeys.TRAIN),
            hooks=[hook_stop])

    elif model_cfg.task_type == 'profiling_infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                        mode=tf.estimator.ModeKeys.PREDICT),
                              predict_keys=["ctr", "cvr", "ctcvr"], hooks=[hook_stop])
        dump_pred_multi(preds, model_cfg.data_dir)
    else:
        raise ValueError(f"Invalid task type: {model_cfg.task_type}")


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
        for mul_fields in spec["multi_hot_fields"]:
            feature_description[mul_fields] = tf.io.FixedLenFeature([spec[f"{key_map[mode]}_max_length"][mul_fields]],
                                                                    tf.int64)
        for mul_fields in spec["special_fields"]:
            feature_description[mul_fields] = tf.io.FixedLenFeature([spec[f"{key_map[mode]}_max_length"][mul_fields]],
                                                                    tf.int64)
        feature_descriptions[mode] = feature_description

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0]), argv=[model_config])
