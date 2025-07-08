#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from dataclasses import dataclass
from datetime import date, timedelta

import tensorflow as tf

from utils import (
    embedding_lookup_sparse_fake,
    setup_logger,
    build_optimizer,
    main,
    spec
)

tf.compat.v1.set_random_seed(2024)
random.seed(2024)

MODEL_NAME = "ETA"


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
    tf.app.flags.DEFINE_integer("attention_dim", 4 * 4, "")
    tf.app.flags.DEFINE_integer("num_heads", 4, "")
    tf.app.flags.DEFINE_boolean("reuse_hash", True, "")
    tf.app.flags.DEFINE_integer("hash_bits", 32, "")
    tf.app.flags.DEFINE_integer("topk", 16, "")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


def model_fn(features, labels, mode, params):
    """build Estimator model"""

    hash_weights = tf.compat.v1.get_variable(
        name="hash_weight",
        shape=(params.embedding_size, params.hash_bits),
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(),
        trainable=False
    )

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
        masks = {}
        for key in ["101", "121", "122", "124", "125", "126", "127", "128", "129",
                    "205", "206", "207", "216", "508", "509", "702", "301"]:
            embeddings[key] = tf.nn.embedding_lookup(emb_weights.get(key), features.get(key),
                                                     name=key + "_embedding_lookup")
            embeddings[key] = tf.reshape(embeddings[key], [-1, 1, params.embedding_size])
        for key in ["109_14", "110_14", "127_14", "150_14"]:
            feature_dense = features.get(key)
            masks[key] = tf.expand_dims(tf.cast(feature_dense >= 0, tf.bool), axis=1)  # None * P * 1
            feature_dense = tf.where(tf.equal(feature_dense, -1), tf.zeros_like(feature_dense), feature_dense)
            embeddings[key] = tf.nn.embedding_lookup(emb_weights.get(key), feature_dense,
                                                     name=key + "_embedding_lookup")  # None * P * E
        for key in ["210", "853"]:
            embeddings[key] = tf.expand_dims(
                embedding_lookup_sparse_fake(emb_weights.get(key), features.get(key), combiner="sum",
                                             name=key + "_embedding_lookup"),
                axis=1
            )

    def long_emb_cat(field_name):
        dense_embedding = embeddings.get(field_name)
        dense_mask = masks.get(field_name)
        paddings = [[0, 0], [0, params.max_seq_len], [0, 0]]
        mask_paddings = [[0, 0], [0, 0], [0, params.max_seq_len]]
        dense_embedding = tf.pad(dense_embedding, paddings, mode="CONSTANT", constant_values=0)
        dense_mask = tf.pad(dense_mask, mask_paddings, mode="CONSTANT", constant_values=0)
        return (
            tf.slice(dense_embedding, [0, 0, 0], [-1, params.topk, -1]),
            tf.slice(dense_embedding, [0, 0, 0], [-1, params.max_seq_len, -1]),
            tf.slice(dense_mask, [0, 0, 0], [-1, -1, params.topk]),
            tf.slice(dense_mask, [0, 0, 0], [-1, -1, params.max_seq_len]),
        )

    @dataclass
    class ShortAttentionParams:
        attention_dim: int = 64
        num_heads: int = 1
        output_dim: int = 16
        index: int = 0
        att_name: str = "short"

    def short_attention(
            target_input,
            seq_input,
            mask,
            params_s: ShortAttentionParams
    ):
        attention_dim = params_s.attention_dim
        num_heads = params_s.num_heads
        output_dim = params_s.output_dim
        index = params_s.index
        att_name = params_s.att_name

        query = tf.contrib.layers.fully_connected(
            inputs=target_input,
            num_outputs=attention_dim,
            activation_fn=None,
            biases_initializer=None,
            scope="q_%s_%d" % (att_name, index),
        )
        key = tf.contrib.layers.fully_connected(
            inputs=seq_input,
            num_outputs=attention_dim,
            activation_fn=None,
            biases_initializer=None,
            scope="k_%s_%d" % (att_name, index),
        )
        value = tf.contrib.layers.fully_connected(
            inputs=seq_input,
            num_outputs=attention_dim,
            activation_fn=None,
            biases_initializer=None,
            scope="v_%s_%d" % (att_name, index),
        )

        d_model = query.shape[-1]
        key_dim = d_model // num_heads

        # Split heads
        query = tf.reshape(query, (-1, num_heads, query.shape[1], key_dim))
        key = tf.reshape(key, (-1, num_heads, key.shape[1], key_dim))
        value = tf.reshape(value, (-1, num_heads, value.shape[1], key_dim))

        # Scaled dot-product attention
        scores = tf.matmul(query, key, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(key_dim, tf.float32))

        paddings = tf.ones_like(scores) * (-(2 ** 32) + 1)
        scores = tf.where(tf.tile(tf.reshape(mask, [-1, 1, 1, seq_input.shape[1]]), [1, num_heads, 1, 1]), scores,
                          paddings)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        attention_output = tf.matmul(attention_weights, value)

        # Merge heads
        attention_output = tf.reshape(attention_output, (-1, query.shape[2], d_model))

        # Final linear projection
        attention_output = tf.contrib.layers.fully_connected(
            inputs=attention_output,
            num_outputs=output_dim,
            activation_fn=None,
            biases_initializer=None,
            scope="o_%s_%d" % (att_name, index),
        )

        return attention_output

    def lsh_hash(vecs, random_rotations):
        rotated_vecs = tf.matmul(vecs, random_rotations)  # B x seq_len x num_hashes
        hash_code = tf.nn.relu(tf.sign(rotated_vecs))
        return hash_code

    @dataclass
    class LongAttentionParams:
        topk: int = 10
        index: int = 0
        attention_dim: int = 64
        num_heads: int = 2

    def long_attention(target_input, seq_input, mask, params_l: LongAttentionParams):
        topk = params_l.topk
        index = params_l.index
        attention_dim = params_l.attention_dim
        num_heads = params_l.num_heads

        random_rotations = hash_weights if params.reuse_hash else tf.random.normal(
            shape=(target_input.shape[-1], 32), dtype=tf.float32
        )
        target_hash = lsh_hash(target_input, random_rotations)
        sequence_hash = lsh_hash(seq_input, random_rotations)
        hash_sim = -tf.reduce_sum(tf.abs(sequence_hash - target_hash), axis=-1)
        paddings = tf.zeros_like(hash_sim) + (-(2 ** 32) + 1)
        hash_sim = tf.where(tf.reshape(mask, [-1, hash_sim.shape[-1]]), hash_sim, paddings)
        _, topk_index = tf.nn.top_k(hash_sim, k=topk, sorted=True)

        topk_emb = tf.gather(
            seq_input, topk_index[..., tf.newaxis], axis=1, batch_dims=1
        )
        topk_mask = tf.gather(mask, topk_index[..., tf.newaxis], axis=-1, batch_dims=1)

        params_s = ShortAttentionParams(
            attention_dim=attention_dim,
            num_heads=num_heads,
            index=index,
            att_name="long"
        )

        return short_attention(target_input, topk_emb, topk_mask, params_s=params_s)

    emb_cats = [long_emb_cat(field) for field in ["109_14", "110_14", "127_14", "150_14"]]

    target_field_name = ["206", "207", "216", "210"]

    with tf.variable_scope("short-Attention"):
        short_attentions_arr = []
        for index, (emb_cat, target_name) in enumerate(
                zip(emb_cats, target_field_name)
        ):
            emb_target = embeddings[target_name]
            emb_short = emb_cat[0]
            mask_short = emb_cat[2]
            params_s = ShortAttentionParams(
                attention_dim=params.attention_dim,
                num_heads=params.num_heads,
                index=index,
                att_name="short"
            )
            short_attentions_arr.append(
                short_attention(emb_target, emb_short, mask=mask_short, params_s=params_s)
            )

    with tf.variable_scope("long-Attention"):
        long_attentions_arr = []
        for index, (emb_cat, target_name) in enumerate(
                zip(emb_cats, target_field_name)
        ):
            emb_target = embeddings[target_name]
            emb_long = emb_cat[1]
            mask_long = emb_cat[3]
            params_l = LongAttentionParams(
                topk=params.topk,
                index=index,
                attention_dim=params.attention_dim,
                num_heads=params.num_heads
            )
            long_attentions_arr.append(
                long_attention(emb_target, emb_long, mask=mask_long, params_l=params_l)
            )

    embedding = tf.concat(
        [embeddings[field_name] for field_name in spec["one_hot_fields"]] +
        [embeddings[field_name] for field_name in spec["special_fields"]] +
        short_attentions_arr +
        long_attentions_arr,
        axis=-1,
    )  # None * 1 * (27 * E)

    x_deep = tf.reshape(embedding, [-1, (23 + 4) * params.embedding_size])  # None * (27 * E)

    with tf.variable_scope("MLP-layer"):
        deep_layers = list(map(int, params.deep_layers.strip().split(",")))
        for layer_i, _ in enumerate(deep_layers):
            x_deep = tf.contrib.layers.fully_connected(
                inputs=x_deep,
                num_outputs=deep_layers[layer_i],
                activation_fn=tf.nn.relu,
                scope="mlp%d" % layer_i,
            )

    with tf.variable_scope("ETA-out"):
        y_deep = tf.contrib.layers.fully_connected(
            inputs=x_deep, num_outputs=1, activation_fn=tf.identity, scope="eta_out"
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
    else:
        raise ValueError("mode should be one of tf.estimator.ModeKeys.TRAIN, EVAL, PREDICT")


if __name__ == "__main__":
    model_config = define_flags()
    logger, china_tz = setup_logger(model_config, MODEL_NAME)
    if model_config.dt_dir == "":
        model_config.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    model_config.model_dir = model_config.model_dir + (date.today() + timedelta(-1)).strftime('%Y%m%d')

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger, "prob"), argv=[model_config])
