#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from datetime import date, timedelta

import tensorflow as tf

from utils import (
    embedding_lookup_sparse_fake,
    setup_logger,
    build_optimizer,
    main,
    spec
)

tf.compat.v1.enable_control_flow_v2()
tf.compat.v1.enable_resource_variables()
tf.compat.v1.set_random_seed(2024)
random.seed(2024)

MODEL_NAME = "ESMM"


def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_string("tower_layers", "512,256,128,64", "tower layers")
    tf.app.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f"../checkpoint/aliccp/{MODEL_NAME}/", "code check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", f"../model/serving/{MODEL_NAME}/",
                               "export servable code for TensorFlow Serving")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing code or not")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


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
        y_ctr = build_tower(x_deep, name='ctr')
        y_ctr = tf.contrib.layers.fully_connected(inputs=y_ctr, num_outputs=1, activation_fn=None,
                                                  scope='deep_out_click')
        y_ctr = tf.reshape(y_ctr, [-1, ])
        y_ctr_prediction = tf.sigmoid(y_ctr)

        # CVR
        y_cvr = build_tower(x_deep, name='cvr')
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
    train_op = build_optimizer(loss, params)

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op
        )
    else:
        raise ValueError("mode should be 'train' or 'eval' or 'infer'")


if __name__ == "__main__":
    model_config = define_flags()
    logger, china_tz = setup_logger(model_config, MODEL_NAME)
    if model_config.dt_dir == "":
        model_config.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    model_config.model_dir = model_config.model_dir + (date.today() + timedelta(-1)).strftime('%Y%m%d')

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger, "multi"), argv=[model_config])
