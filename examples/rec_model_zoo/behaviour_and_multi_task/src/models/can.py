#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from datetime import datetime, date, timedelta

import tensorflow as tf

import utils
from utils import (
    embedding_lookup_sparse_fake,
    setup_logger,
    build_optimizer,
    main
)

tf.compat.v1.set_random_seed(2024)
random.seed(2024)

MODEL_NAME = "CAN"


def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_string("deep_layers", "512,256,128,64", "deep layers")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f"../checkpoint/aliccp/{MODEL_NAME}/", "code check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", f"../model/serving/{MODEL_NAME}/",
                               "export servable code for TensorFlow Serving")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing code or not")
    tf.app.flags.DEFINE_string("weight_emb_w", "[[12, 6], [6, 4]]", "shape of coaction weight")
    tf.app.flags.DEFINE_string("weight_emb_b", "[0, 0]", "width of coaction bias")
    tf.app.flags.DEFINE_integer("orders", 3, "coaction orders")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


def p_re_lu(_x, name=''):
    alphas = tf.compat.v1.get_variable('alpha_' + name, _x.get_shape()[-1],
                                       initializer=tf.constant_initializer(0.0),
                                       dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def build_embedding_layer(features: dict, model_cfg, spec: dict) -> dict:
    """
    Build the embedding layer for the model.

    Args:
        features (dict): Input features.
        model_cfg: Model configuration.
        spec (dict): Specification dictionary.

    Returns:
        dict: Dictionary of embeddings.
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
    for key in ["101", "121", "122", "124", "125", "126", "127", "128", "129",
                "205", "206", "207", "216", "508", "509", "702", "301"]:
        embeddings[key] = tf.nn.embedding_lookup(emb_weights[key], features[key], name=key + "_embedding_lookup")
        embeddings[key] = tf.reshape(embeddings[key], [-1, 1, model_cfg.embedding_size])
    for key in ["109_14", "110_14", "127_14", "150_14", "210", "853"]:
        embeddings[key] = tf.expand_dims(
            embedding_lookup_sparse_fake(emb_weights[key], features[key], combiner="sum",
                                         name=key + "_embedding_lookup"),
            axis=1
        )
    return embeddings


def build_coaction_layer(features: dict, model_cfg, spec: dict, embeddings: dict) -> tf.Tensor:
    """
    Build the coaction layer for the model.

    Args:
        features (dict): Input features.
        model_cfg: Model configuration.
        spec (dict): Specification dictionary.
        embeddings (dict): Dictionary of embeddings.

    Returns:
        tf.Tensor: Coaction layer tensor.
    """
    weight_emb_w = json.loads(model_cfg.weight_emb_w)
    weight_emb_b = json.loads(model_cfg.weight_emb_b)
    orders = model_cfg.orders
    weight_emb_dim = sum([w[0] * w[1] for w in weight_emb_w]) + sum(weight_emb_b)

    cate_weights = {}
    category_feats = []
    for key in ["101", "121", "122", "124", "125", "126", "127", "128", "129"]:
        cate_weights[key] = tf.compat.v1.get_variable(
            name=key + "_cate_emb_wgts",
            shape=[spec["vocab_length"][key] + 1, weight_emb_w[0][0]],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
        )
        cate = tf.nn.embedding_lookup(cate_weights[key], features[key], name=key + "_cate_lookup")
        category_feats.append(tf.reshape(cate, [-1, 1, weight_emb_w[0][0]]))
    category_feat = tf.concat(category_feats, axis=1)
    category_feat_orders = [tf.pow(category_feat, i + 1) for i in range(orders)]

    coaction_res = []

    for target_key, his_key in zip(
            ["206", "207", "216", "210"],
            ["109_14", "110_14", "127_14", "150_14"]
    ):
        target_coaction_weight = tf.compat.v1.get_variable(
            name=f"target_coaction_emb_weight_{target_key}",
            shape=[spec["vocab_length"][target_key] + 1, weight_emb_dim],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
        )
        if target_key in spec["special_fields"]:
            target_coaction_emb = embedding_lookup_sparse_fake(target_coaction_weight, features[target_key],
                                                               combiner="sum", name=target_key + "_cate_lookup")
        else:
            target_coaction_emb = tf.nn.embedding_lookup(target_coaction_weight, features[target_key],
                                                         name=target_key + "_cate_lookup")

        his_coaction_weight = tf.compat.v1.get_variable(
            name=f"his_coaction_emb_weight_{his_key}",
            shape=[spec["vocab_length"][his_key] + 1, weight_emb_w[0][0]],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
        )
        feature_dense = features[his_key]
        his_coaction_mask = tf.cast(feature_dense >= 0, tf.bool)
        feature_dense = tf.where(tf.equal(feature_dense, -1), tf.zeros_like(feature_dense), feature_dense)
        his_coaction_emb = tf.nn.embedding_lookup(his_coaction_weight, feature_dense, name=his_key + "_cate_lookup")

        weight, bias = [], []
        idx = 0
        for w, b in zip(weight_emb_w, weight_emb_b):
            weight.append(
                tf.reshape(
                    target_coaction_emb[:, idx: idx + w[0] * w[1]],
                    [-1, w[0], w[1]],
                )
            )
            idx += w[0] * w[1]
            if b == 0:
                bias.append(None)
            else:
                bias.append(
                    tf.reshape(target_coaction_emb[:, idx: idx + b], [-1, 1, b])
                )
                idx += b

        out_seq = []
        hh = [tf.pow(his_coaction_emb, i + 1) for i in range(orders)]
        for h in hh:
            for j, (w, b) in enumerate(zip(weight, bias)):
                h = tf.matmul(h, w)
                if b is not None:
                    h = h + b
                if j != len(weight) - 1:
                    h = tf.nn.tanh(h)
                out_seq.append(tf.where(
                    tf.tile(
                        tf.reshape(
                            his_coaction_mask, shape=[model_cfg.batch_size, -1, 1]),
                        [1, 1, int(h.shape[-1])]),
                    h, tf.zeros_like(h)))
        out_seq = tf.concat(out_seq, 2)
        out = tf.reduce_sum(out_seq, 1, keepdims=True)
        coaction_res.append(out)

        out_non_seq = []
        for h in category_feat_orders:
            for j, (w, b) in enumerate(zip(weight, bias)):
                h = tf.matmul(h, w)
                if b is not None:
                    h = h + b
                if j != len(weight) - 1:
                    h = tf.nn.tanh(h)
                out_non_seq.append(h)
        out_non_seq = tf.concat(out_non_seq, 2)
        out = tf.reshape(out_non_seq, shape=[-1, 1, int(out_non_seq.shape[-1] * out_non_seq.shape[-2])])
        coaction_res.append(out)

    embedding = tf.concat(
        [embeddings[field_name] for field_name in spec["one_hot_fields"]] +
        [embeddings[field_name] for field_name in spec["multi_hot_fields"]] +
        [embeddings[field_name] for field_name in spec["special_fields"]] +
        coaction_res,
        axis=2,
    )  # None * 1 * (?? * E)

    return embedding


def build_mlp_layer(x_deep: tf.Tensor, model_cfg) -> tf.Tensor:
    """
    Build the MLP layer for the model.

    Args:
        x_deep (tf.Tensor): Input tensor.
        model_cfg: Model configuration.

    Returns:
        tf.Tensor: Output tensor from the MLP layer.
    """
    x_deep = tf.layers.batch_normalization(inputs=x_deep, name='bn1')
    deep_layers = list(map(int, model_cfg.deep_layers.strip().split(",")))
    for layer_i, _ in enumerate(deep_layers):
        x_deep = tf.contrib.layers.fully_connected(
            inputs=x_deep,
            num_outputs=deep_layers[layer_i],
            activation_fn=None,
            scope="mlp%d" % layer_i,
        )
        x_deep = p_re_lu(x_deep, name='mlp%d' % layer_i)
    return x_deep


def build_loss_function(labels: dict, pred: tf.Tensor) -> tf.Tensor:
    """
    Build the loss function for the model.

    Args:
        labels (dict): Ground truth labels.
        pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: Loss tensor.
    """
    epsilon = 1e-7
    click_weight = 0.14

    ctr_loss = - (1 - click_weight) / click_weight * labels['y'] * tf.math.log(pred + epsilon) - \
               (1 - labels['y']) * tf.math.log(1 - pred + epsilon)
    loss = tf.reduce_mean(ctr_loss)
    return loss


def model_fn(features, labels, mode, params):
    """build Estimator model"""

    embeddings = build_embedding_layer(features, params, utils.spec)

    embedding = build_coaction_layer(features, params, utils.spec, embeddings)

    x_deep = tf.reshape(embedding,
                        [-1, 23 * params.embedding_size + 4 * params.orders * sum(
                            [w[1] for w in json.loads(params.weight_emb_w)]) * 10])

    with tf.compat.v1.variable_scope("MLP-layer"):
        x_deep = tf.layers.batch_normalization(inputs=x_deep, name='bn1')
        deep_layers = list(map(int, params.deep_layers.strip().split(",")))
        for layer_i, _ in enumerate(deep_layers):
            x_deep = tf.contrib.layers.fully_connected(
                inputs=x_deep,
                num_outputs=deep_layers[layer_i],
                activation_fn=None,
                scope="mlp%d" % layer_i,
            )
            x_deep = p_re_lu(x_deep, name='mlp%d' % layer_i)

    with tf.compat.v1.variable_scope("CAN-out"):
        y_deep = tf.contrib.layers.fully_connected(
            inputs=x_deep, num_outputs=1, activation_fn=tf.identity, scope="can_out"
        )
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

    loss = build_loss_function(labels, pred)

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
    else:
        raise ValueError("mode should be one of tf.estimator.ModeKeys.TRAIN, EVAL, PREDICT")


if __name__ == "__main__":
    model_config = define_flags()
    logger, china_tz = setup_logger(model_config, MODEL_NAME)
    if model_config.dt_dir == "":
        model_config.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    model_config.model_dir = model_config.model_dir + datetime.now(china_tz).strftime('%Y%m%d')

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger, "prob"), argv=[model_config])
