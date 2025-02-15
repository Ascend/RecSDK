#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import random
import logging
import stat
from typing import List, Tuple, Dict, Any
from datetime import date, timedelta, datetime

import pytz
import tensorflow as tf
from npu_bridge.npu_init import NPUEstimator, NPURunConfig

from utils import get_third_nearest_checkpoint, dump_pred

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


# ------ Load tfrecord dataset ------
def input_fn(filenames, batch_size=32, field_size=39, num_epochs=1, perform_shuffle=False):
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
