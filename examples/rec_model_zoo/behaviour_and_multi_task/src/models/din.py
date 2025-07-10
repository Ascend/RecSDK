#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from datetime import date, timedelta

import tensorflow as tf

from utils import embedding_lookup_sparse_fake, build_optimizer, main, spec
from examples.rec_model_zoo.common import setup_logger

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
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f"../checkpoint/aliccp/{MODEL_NAME}/", "code check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", f"../model/serving/{MODEL_NAME}/",
                               "export servable code for TensorFlow Serving")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing code or not")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


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
    train_op = build_optimizer(loss, params)

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op
        )


if __name__ == "__main__":
    model_config = define_flags()
    logger = setup_logger(model_config, MODEL_NAME)
    if model_config.dt_dir == "":
        model_config.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    model_config.model_dir = model_config.model_dir + (date.today() + timedelta(-1)).strftime('%Y%m%d')

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger, "prob"), argv=[model_config])
