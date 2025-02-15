#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import random
import logging
import stat
from datetime import date, timedelta, datetime

import pytz
import numpy as np
import tensorflow as tf
from npu_bridge.npu_init import NPUEstimator, NPURunConfig

from utils import get_third_nearest_checkpoint, dump_pred

MODEL_NAME = "AFN_plus"


def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("train_size", 33003326, "Number of instances in the train set")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_integer("feature_size", 2100000, "Number of features")
    tf.app.flags.DEFINE_integer("field_size", 39, "Number of fields")
    tf.app.flags.DEFINE_integer("embedding_size", 10, "Embedding size")
    tf.app.flags.DEFINE_integer("hidden_size", 1500, "hidden unit size")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_boolean("batch_norm", True, "perform batch normaization (True or False)")
    tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
    tf.app.flags.DEFINE_string("data_dir", '../data/criteo/', "data dir")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f'../checkpoint/criteo/{MODEL_NAME}/', "model check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
    tf.app.flags.DEFINE_string("task_type", 'train', "task type")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    tf.app.flags.DEFINE_string("deep_layers", '400,400,400', "deep layers")
    return model_conf


# ------ Load tfrecord dataset ------
def input_fn(filenames: list, batch_size: int = 32, field_size: int = 39, num_epochs: int = 1,
             perform_shuffle: bool = False) -> tuple:
    """
    Input function for loading TFRecord dataset.

    Args:
        filenames (list): List of TFRecord file paths.
        batch_size (int): Number of samples per batch.
        field_size (int): Number of fields in the dataset.
        num_epochs (int): Number of epochs to repeat the dataset.
        perform_shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tuple: A tuple containing batch features and batch labels.
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


def layer_first(embeddings_trans, field_size, hidden_size):
    """
    Apply the first layer transformation.

    Args:
        embeddings_trans (tf.Tensor): Transformed embeddings.
        field_size (int): Number of fields.
        hidden_size (int): Hidden layer size.

    Returns:
        tf.Tensor: Output of the first layer.
    """
    with tf.compat.v1.variable_scope("Layer_firtst"):
        weights = tf.compat.v1.get_variable("h_lr_weights", shape=[field_size, hidden_size],
                                            initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.compat.v1.get_variable('biases', [hidden_size], initializer=tf.constant_initializer(0))
        layer_out = tf.einsum('bkf,fo->bko', embeddings_trans, weights) + biases
    return layer_out


def plus_deep_layer(embeddings_deep, field_size, embedding_size, layers_dnn):
    """
    Apply the plus deep layer transformation.

    Args:
        embeddings_deep (tf.Tensor): Deep embeddings.
        field_size (int): Number of fields.
        embedding_size (int): Embedding size.
        layers_dnn (list): List of DNN layer sizes.

    Returns:
        tf.Tensor: Output of the plus deep layer.
    """
    with tf.compat.v1.variable_scope("Plus-Deep-Layer"):
        deep_inputs = tf.reshape(embeddings_deep, shape=[-1, field_size * embedding_size])  # None * (F * E)

        for dnn_i, _ in enumerate(layers_dnn):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers_dnn[dnn_i],
                                                            scope='plus_mlp%d' % dnn_i)

        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
                                                   scope='plus_deep_out')
        y_d = tf.reshape(y_deep, shape=[-1])
    return y_d


def combine_layers(y_d, y_afn):
    """
    Combine the outputs of the deep and plus deep layers.

    Args:
        y_d (tf.Tensor): Output of the plus deep layer.
        y_afn (tf.Tensor): Output of the deep layer.

    Returns:
        tf.Tensor: Combined output.
    """
    w1 = tf.compat.v1.get_variable(name='w1', shape=[1], initializer=tf.constant_initializer(0.5))
    w2 = tf.compat.v1.get_variable(name='w2', shape=[1], initializer=tf.constant_initializer(0.5))
    b_p = tf.compat.v1.get_variable(name='b_p', shape=[1], initializer=tf.constant_initializer(0.0))
    y_1 = w1 * tf.stop_gradient(y_d)
    y_2 = w2 * tf.stop_gradient(y_afn)
    y = y_1 + y_2 + b_p
    return y


def build_optimizer(loss: tf.Tensor, learning_rate: float, model_cfg: object) -> tf.Operation:
    """
    Build the optimizer.

    Args:
        loss (tf.Tensor): The loss value.
        learning_rate (float): The learning rate.
        model_cfg (object): The model configuration object.

    Returns:
        tf.Operation: The training operation.
    """
    if model_cfg.optimizer == 'Adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif model_cfg.optimizer == 'Adagrad':
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif model_cfg.optimizer == 'Momentum':
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif model_cfg.optimizer == 'ftrl':
        optimizer = tf.compat.v1.train.FtrlOptimizer(learning_rate)
    else:
        raise NotImplementedError("This optimizer is not implemented.")
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    return train_op


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params.field_size
    feature_size = params.feature_size
    embedding_size = params.embedding_size
    learning_rate = params.learning_rate
    layers = list(map(int, params.deep_layers.split(',')))
    layers_dnn = [400, 400, 400]

    # ------bulid weights------
    feat_emb = tf.compat.v1.get_variable(name="h_lr_emb", shape=[feature_size, embedding_size],
                                         initializer=tf.glorot_normal_initializer())
    feat_emb_deep = tf.compat.v1.get_variable(name="h_lr_emb_deep", shape=[feature_size, embedding_size],
                                              initializer=tf.random_normal_initializer(stddev=0.1), )
    feat_emb = tf.abs(feat_emb)
    feat_emb = tf.clip_by_value(feat_emb, 1e-4, np.infty)

    # ------build feature-------
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])
    feat_vals = tf.clip_by_value(feat_vals, 0.001, 1.)

    # ------build f(x)------
    with tf.compat.v1.variable_scope("Permutation-Layer"):
        embeddings_origin = tf.nn.embedding_lookup(feat_emb, feat_ids)
        embeddings_origin_deep = tf.nn.embedding_lookup(feat_emb_deep, feat_ids)
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings_origin, feat_vals)
        embeddings_deep = tf.multiply(embeddings_origin_deep, feat_vals)
        em_trans = tf.transpose(embeddings, perm=[0, 2, 1])
        em_trans = tf.math.log(em_trans, name="log_input")
        em_trans = tf.debugging.check_numerics(em_trans, "log2")

        train_phase = (mode == tf.estimator.ModeKeys.TRAIN)


        em_trans = batch_norm_layer(em_trans, train_phase=train_phase,
                                            scope_bn='bn_log', model_cfg=params)

    layer_out = layer_first(em_trans, field_size, params.hidden_size)
    with tf.compat.v1.variable_scope("Deep-Layer"):
        interactions = tf.exp(layer_out, name="restored_input")  # None * E * O
        interactions = batch_norm_layer(interactions, train_phase=train_phase,
                                        scope_bn='bn_inter', model_cfg=params)
        deep_inputs = tf.reshape(interactions, shape=[-1, embedding_size * params.hidden_size])  # None * (E * O)

        for layer_i, _ in enumerate(layers):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[layer_i],
                                                            scope='mlp%d' % layer_i)

        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
                                                   scope='deep_out')
        y_afn = tf.reshape(y_deep, shape=[-1])

    y_d = plus_deep_layer(embeddings_deep, field_size, embedding_size, layers_dnn)
    y = combine_layers(y_d, y_afn)

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

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) \
           + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_d, labels=labels)) \
           + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_afn, labels=labels))

    # Provide an estimator spec for `ModeKeys.EVAL`
    log_loss = tf.compat.v1.losses.log_loss(labels, pred)
    auc_metric = tf.compat.v1.metrics.auc(labels, pred)
    loss_metric = tf.compat.v1.metrics.mean(log_loss)
    eval_metric_ops = {
        "auc": tf.compat.v1.metrics.auc(labels, pred),
        "logloss": tf.compat.v1.metrics.mean(log_loss),
        "stop_criterion": (auc_metric[0] - loss_metric[0], tf.group(auc_metric[1], loss_metric[1]))
    }

    train_op = build_optimizer(loss, learning_rate, params)

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
        raise NotImplementedError("This mode is not implemented.")


def batch_norm_layer(x, train_phase, scope_bn, model_cfg):
    """
        Apply batch normalization to the input tensor.
    """
    bn_train = tf.contrib.layers.batch_norm(x, decay=model_cfg.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=model_cfg.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


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
            logger.info("Model directory does not exist, creating a new one...")

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
        raise ValueError(
            "Invalid task_type. task_type must be one of "
            "['train', 'eval', 'infer', 'profiling_train', 'profiling_infer']")


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
