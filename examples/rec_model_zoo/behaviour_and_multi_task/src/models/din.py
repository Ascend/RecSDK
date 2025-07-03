#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import random
import shutil
from datetime import date, timedelta
from functools import partial

import tensorflow as tf
from npu_bridge.npu_init import NPUEstimator, NPURunConfig

from utils import (
    get_third_nearest_checkpoint,
    json_file_load,
    dump_pred_prob,
    embedding_lookup_sparse_fake,
    build_feature_descriptions,
    setup_logger
)

tf.compat.v1.set_random_seed(2024)
random.seed(2024)

MODEL_NAME = "DIN"


def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_string("attention_layers", '80,40', "Attention Net mlp layers")
    tf.app.flags.DEFINE_string("deep_layers", "512,256,128,64", "deep layers")
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


def p_re_lu(_x, name=''):
    alphas = tf.compat.v1.get_variable('alpha_' + name, _x.get_shape()[-1],
                                       initializer=tf.constant_initializer(0.0),
                                       dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def dice(_x, axis=-1, epsilon=0.000000001, name='dice', training=True):
    alphas = tf.compat.v1.get_variable('alpha_' + name, _x.get_shape()[-1],
                                       initializer=tf.constant_initializer(0.0),
                                       dtype=tf.float32)
    inputs_normed = tf.layers.batch_normalization(
        inputs=_x,
        axis=axis,
        epsilon=epsilon,
        center=False,
        scale=False,
        training=training)
    x_p = tf.sigmoid(inputs_normed)
    return alphas * (1.0 - x_p) * _x + x_p * _x


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
        dense_ids = {}
        for key in ["101", "121", "122", "124", "125", "126", "127",
                    "128", "129", "205", "508", "509", "702", "301"]:
            embeddings[key] = tf.nn.embedding_lookup(emb_weights.get(key), features.get(key),
                                                     name=key + "_embedding_lookup")
            embeddings[key] = tf.reshape(embeddings[key], [-1, 1, params.embedding_size])

        embeddings["853"] = tf.expand_dims(
            embedding_lookup_sparse_fake(emb_weights.get("853"), features.get("853"), combiner="sum",
                                         name="853" + "_embedding_lookup"),
            axis=1
        )

        for key in ["206", "207", "216"]:
            embeddings[key] = tf.nn.embedding_lookup(emb_weights.get(key), features.get(key),
                                                     name=key + "_embedding_lookup")

        embeddings["210"] = embedding_lookup_sparse_fake(emb_weights.get("210"), features.get("210"), combiner="sum",
                                                         name="210" + "_embedding_lookup")

        for key in ["109_14", "110_14", "127_14", "150_14"]:
            feature_dense = features.get(key)
            dense_ids[key] = feature_dense
            dense_mask = tf.expand_dims(tf.cast(feature_dense >= 0, tf.float32), axis=-1)  # None * P * 1
            feature_dense = tf.where(tf.equal(feature_dense, -1), tf.zeros_like(feature_dense), feature_dense)
            emb = tf.nn.embedding_lookup(emb_weights.get(key), feature_dense,
                                         name=key + "_embedding_lookup")  # None * P * E
            emb = tf.multiply(emb, dense_mask)
            embeddings[key] = emb

    with tf.compat.v1.variable_scope("Field-wise-Pooling-layer", reuse=tf.compat.v1.AUTO_REUSE):
        attention_layers = list(map(int, params.attention_layers.strip().split(',')))

        def attention_unit(a_xx_emb, ub_dense_id, ub_emb, unit_name="targ_hist"):
            dense_mask = tf.expand_dims(tf.cast(ub_dense_id >= 0, tf.bool), axis=1)  # None * 1 * P
            padded_dim = tf.shape(ub_dense_id)[1]

            ax_emb = tf.reshape(tf.tile(a_xx_emb, [1, padded_dim]),
                                shape=[-1, padded_dim, params.embedding_size])  # None * E --> None * P * E
            x_inputs = tf.concat([ax_emb, ub_emb, ax_emb - ub_emb, ax_emb * ub_emb], axis=-1)  # None * P * 4E
            for att_i, _ in enumerate(attention_layers):
                x_inputs = tf.contrib.layers.fully_connected(inputs=x_inputs, num_outputs=attention_layers[att_i],
                                                             activation_fn=None,
                                                             scope="att_fc_%s_%d" % (unit_name, att_i))
                x_inputs = p_re_lu(x_inputs, name="att_fc_%s_%d" % (unit_name, att_i))
            att_wgt = tf.contrib.layers.fully_connected(inputs=x_inputs, num_outputs=1,
                                                        activation_fn=None,
                                                        scope="att_out_%s" % unit_name)  # None * P * 1
            att_wgt = tf.reshape(att_wgt, shape=[-1, 1, padded_dim])  # None * 1 * P
            paddings = tf.ones_like(att_wgt) * (-2 ** 32 + 1)  # None * 1 * P
            att_wgt = tf.where(dense_mask, att_wgt, paddings)  # None * 1 * P
            att_wgt = att_wgt / (ub_emb.get_shape().as_list()[-1] ** 0.5)
            att_wgt = tf.nn.softmax(att_wgt)  # None * 1 * P
            wgt_emb = tf.matmul(att_wgt, ub_emb)  # None * 1 * E

            return wgt_emb

        for target_key, his_key in zip(
                ["206", "207", "216", "210"],
                ["109_14", "110_14", "127_14", "150_14"]
        ):
            embeddings[his_key] = attention_unit(a_xx_emb=embeddings[target_key],
                                                 ub_dense_id=dense_ids[his_key],
                                                 ub_emb=embeddings[his_key],
                                                 unit_name="%s_%s" % (target_key, his_key))  # None * 1 * E

    for key in ["206", "207", "210", "216"]:
        embeddings[key] = tf.reshape(embeddings[key], [-1, 1, params.embedding_size])

    embedding = tf.concat(
        [embeddings[field_name] for field_name in spec["one_hot_fields"]] +
        [embeddings[field_name] for field_name in spec["multi_hot_fields"]] +
        [embeddings[field_name] for field_name in spec["special_fields"]],
        axis=2,
    )  # None * 1 * 23 * E)

    x_deep = tf.reshape(embedding, [-1, 23 * params.embedding_size])  # None * (23 * E)

    with tf.compat.v1.variable_scope("MLP-layer"):
        deep_layers = list(map(int, params.deep_layers.strip().split(',')))
        for layer_i, _ in enumerate(deep_layers):
            x_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=deep_layers[layer_i],
                                                       activation_fn=None, scope='mlp%d' % layer_i)
            x_deep = p_re_lu(x_deep, name='mlp%d' % layer_i)

    with tf.compat.v1.variable_scope("DIN-out"):
        y_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=1, activation_fn=tf.identity,
                                                   scope='din_out')
        y = tf.reshape(y_deep, shape=[-1])
        pred = tf.sigmoid(y)

    predictions = {"prob": pred}

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

        ctr_loss = - (1 - click_weight) / click_weight * labels['y'] * tf.math.log(pred + epsilon) - \
                   (1 - labels['y']) * tf.math.log(1 - pred + epsilon)
        loss = tf.reduce_mean(ctr_loss)

    # Provide an estimator spec for `ModeKeys.EVAL`
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "auc_ctr": tf.compat.v1.metrics.auc(labels["y"], pred),
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

    def clip_grad(grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)

    clipped_gradients = [(clip_grad(grad), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(clipped_gradients, global_step=tf.compat.v1.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op
        )


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
                              predict_keys="prob", hooks=[])
        dump_pred_prob(preds, model_cfg.data_dir)

    elif model_cfg.task_type == 'profiling_train':
        model.train(
            input_fn=lambda: input_fn(tr_files, num_epochs=1, batch_size=model_cfg.batch_size, perform_shuffle=True,
                                      mode=tf.estimator.ModeKeys.TRAIN),
            hooks=[hook_stop])

    elif model_cfg.task_type == 'profiling_infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=model_cfg.batch_size,
                                                        mode=tf.estimator.ModeKeys.PREDICT),
                              predict_keys="prob", hooks=[hook_stop])
        dump_pred_prob(preds, model_cfg.data_dir)
    else:
        raise ValueError("task type not supported: {}".format(model_cfg.task_type))


if __name__ == "__main__":
    model_config = define_flags()
    logger, china_tz = setup_logger(model_config, MODEL_NAME)
    spec, feature_descriptions = build_feature_descriptions(model_config)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0]), argv=[model_config])
