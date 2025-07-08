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

MODEL_NAME = "DFFM"


@dataclass
class DFUBLayerEmbeddings:
    q_tar_embedding: tf.Tensor
    k_tar_embedding: tf.Tensor
    v_tar_embedding: tf.Tensor


def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
    tf.app.flags.DEFINE_integer("internal_size", 8, "Internal size")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_integer("heads_num", 4, "Number of attention heads")
    tf.app.flags.DEFINE_string("deep_layers", "512,256,128,64", "deep layers")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f"../checkpoint/aliccp/{MODEL_NAME}/", "code check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", f"../model/serving/{MODEL_NAME}/",
                               "export servable code for TensorFlow Serving")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing code or not")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


def model_fn(features, labels, mode, params):
    """build Estimator model"""

    emb_weights, embeddings = build_embeddings(features, params)
    i_all_embeddings = build_dffi(embeddings, params)

    with tf.compat.v1.variable_scope("Inner-Product"):
        row, col = [], []
        for i in range(22 - 1):
            for j in range(i + 1, 22):
                row.append(i)
                col.append(j)
        p = tf.gather(i_all_embeddings, axis=1, indices=row)
        q = tf.gather(i_all_embeddings, axis=1, indices=col)
        inner_product = tf.reduce_sum(p * q, axis=2)

    d_layer_input = tf.concat([inner_product, tf.reshape(i_all_embeddings, [-1, 22 * params.embedding_size * 2])],
                              axis=-1)

    with tf.compat.v1.variable_scope("DFUB", reuse=tf.compat.v1.AUTO_REUSE):

        dfub_target_emb = {}
        for target_key in ["206", "207", "210", "216"]:
            target_dfub_weight = tf.compat.v1.get_variable(
                name=f"target_dfub_emb_weight_{target_key}",
                shape=[spec["vocab_length"][target_key] + 1, params.embedding_size * params.internal_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
            )
            if target_key == "210":
                dfub_target_emb[target_key] = tf.expand_dims(
                    embedding_lookup_sparse_fake(target_dfub_weight, features[target_key], combiner="sum",
                                                 name=target_key + "_dfub_lookup"),
                    axis=1
                )
            else:
                dfub_target_emb[target_key] = tf.reshape(
                    tf.nn.embedding_lookup(target_dfub_weight, features[target_key], name=target_key + "_dfub_lookup"),
                    [-1, 1, params.embedding_size * params.internal_size]
                )

        dfub_his_emb = {}
        dfub_his_len = {}
        for his_key in ["109_14", "110_14", "127_14", "150_14"]:
            feature_dense = features[his_key]
            dfub_his_len[his_key] = tf.reduce_sum(tf.cast(feature_dense >= 0, tf.int32), axis=1, keepdims=True)
            dense_mask = tf.expand_dims(tf.cast(feature_dense >= 0, tf.float32), axis=-1)  # None * P * 1
            feature_dense = tf.where(tf.equal(feature_dense, -1), tf.zeros_like(feature_dense), feature_dense)
            emb = tf.nn.embedding_lookup(emb_weights.get(his_key), feature_dense,
                                         name=his_key + "_dfub_lookup")  # None * P * E
            dfub_his_emb[his_key] = tf.multiply(emb, dense_mask)

        domain_dfub_weight = tf.compat.v1.get_variable(
            name=f"domain_dfub_emb_weight",
            shape=[spec["vocab_length"]["301"] + 1, params.embedding_size * params.internal_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
        )
        dfub_domain_emb = tf.reshape(
            tf.nn.embedding_lookup(domain_dfub_weight, features["301"], name="domain_dfub_lookup"),
            [-1, 1, params.embedding_size * params.internal_size]
        )

        dfub_output = []

        for target_key, his_key in zip(
                ["206", "207", "216", "210"],
                ["109_14", "110_14", "127_14", "150_14"]
        ):
            part_target_emb = dfub_target_emb[target_key]
            his_emb = dfub_his_emb[his_key]
            hist_mask = dfub_his_len[his_key]
            part1_num = int(0.5 * params.embedding_size * params.internal_size)
            part2_num = params.embedding_size * params.internal_size - part1_num
            target_emb = tf.concat([part_target_emb[:, :, :part1_num], dfub_domain_emb[:, :, :part2_num]], axis=-1)
            target_emb_c = DFUBLayerEmbeddings(q_tar_embedding=target_emb, k_tar_embedding=target_emb,
                                               v_tar_embedding=target_emb)

            his_emb_new = dfub_layer(params, his_emb, hist_mask, target_emb_c,
                                     scope="%s_%s" % (target_key, his_key))
            his_emb_new = tf.reduce_sum(his_emb_new, axis=1)
            dfub_output.append(his_emb_new)

        dfub_output_emb = tf.concat(dfub_output, axis=-1)

    d_layer_output = tf.concat([d_layer_input, dfub_output_emb], axis=-1)

    with tf.compat.v1.variable_scope("MLP-layer"):
        deep_layers = list(map(int, params.deep_layers.strip().split(',')))

        for layer_i, _ in enumerate(deep_layers):
            d_layer_output = tf.contrib.layers.fully_connected(inputs=d_layer_output, num_outputs=deep_layers[layer_i],
                                                               activation_fn=tf.nn.relu, scope='mlp%d' % layer_i)

    with tf.compat.v1.variable_scope("DFFM-out"):
        y_deep = tf.contrib.layers.fully_connected(inputs=d_layer_output, num_outputs=1, activation_fn=tf.identity,
                                                   scope='dffm_out')
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

    raise ValueError(f"Invalid mode received: {mode}")


def build_embeddings(features, params):
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
        for key in ["101", "121", "122", "124", "125", "126", "127", "128", "129",
                    "205", "206", "207", "216", "508", "509", "702", "301"]:
            embeddings[key] = tf.nn.embedding_lookup(emb_weights[key], features[key], name=key + "_embedding_lookup")

        for key in ["109_14", "110_14", "127_14", "150_14", "210", "853"]:
            embeddings[key] = embedding_lookup_sparse_fake(
                emb_weights[key], features[key], combiner="sum", name=key + "_embedding_lookup"
            )

        return emb_weights, embeddings


def build_dffi(embeddings, emb_weights, params):
    stack_emb = []
    for field_name in ["101", "109_14", "110_14", "127_14", "150_14", "121",
                       "122", "124", "125", "126", "127", "128", "129", "205",
                       "206", "207", "210", "216", "508", "509", "702", "853"]:
        stack_emb.append(embeddings.get(field_name))

    sparse_input = tf.stack(stack_emb, axis=1)  # None * 22 * E

    with tf.compat.v1.variable_scope("DFFI", reuse=tf.compat.v1.AUTO_REUSE):
        domain_emb = embeddings.get("301")
        domain_emb = tf.nn.relu(domain_emb)
        # map domain embedding
        meta_dnn_hidden_units = [16, 16]
        meta_dnn_hidden_units = [params.embedding_size] + meta_dnn_hidden_units

        meta_para = []
        for i in range(len(meta_dnn_hidden_units) - 1):
            meta_para.append(meta_dnn_hidden_units[i] * meta_dnn_hidden_units[i + 1])

        meta_param_size = sum(meta_para)
        domain_vec = tf.contrib.layers.fully_connected(inputs=domain_emb, num_outputs=meta_param_size,
                                                       activation_fn=tf.nn.relu, scope='domain_map_mlp')
        # MetaNet
        weight_list = []
        bias_list = []
        offset = 0
        for i, _ in enumerate(meta_dnn_hidden_units[:-1]):
            domain_weight = tf.reshape(
                domain_vec[:, offset:offset + meta_dnn_hidden_units[i] * meta_dnn_hidden_units[i + 1]],
                [-1, meta_dnn_hidden_units[i], meta_dnn_hidden_units[i + 1]]
            )
            offset += meta_dnn_hidden_units[i] * meta_dnn_hidden_units[i + 1]
            weight_list.append(domain_weight)

        bias_list = [0.0] * len(weight_list)

        dffi_output = sparse_input
        for weight_i, _ in enumerate(weight_list):
            dffi_output = tf.einsum('ijk,ikl->ijl', dffi_output, weight_list[weight_i]) + bias_list[weight_i]
            if weight_i < len(weight_list) - 1:
                dffi_output = tf.nn.relu(dffi_output)

        return tf.concat([dffi_output, sparse_input], axis=2)


def dfub_layer(params, seqs, masks, embeddings_t: DFUBLayerEmbeddings, scope="targ_hist"):
    q_tar_embedding = embeddings_t.q_tar_embedding
    k_tar_embedding = embeddings_t.k_tar_embedding
    v_tar_embedding = embeddings_t.v_tar_embedding
    w_q = tf.compat.v1.get_variable(name="weight_Q_%s" % scope,
                                    shape=[params.embedding_size, params.internal_size],
                                    initializer=tf.random_normal_initializer(stddev=0.1), )
    w_k = tf.compat.v1.get_variable(name="weight_K_%s" % scope,
                                    shape=[params.embedding_size, params.internal_size],
                                    initializer=tf.random_normal_initializer(stddev=0.1), )
    w_v = tf.compat.v1.get_variable(name="weight_V_%s" % scope,
                                    shape=[params.embedding_size, params.internal_size],
                                    initializer=tf.random_normal_initializer(stddev=0.1), )

    w_res = tf.compat.v1.get_variable(name="weight_Res_%s" % scope,
                                        shape=[params.embedding_size, params.embedding_size],
                                        initializer=tf.random_normal_initializer(stddev=0.1), )

    queries = tf.tensordot(seqs, w_q, axes=(-1, 0))
    keys = tf.tensordot(seqs, w_k, axes=(-1, 0))
    values = tf.tensordot(seqs, w_v, axes=(-1, 0))

    q_tar_embedding = tf.reshape(q_tar_embedding, [-1, params.internal_size, params.embedding_size])
    queries = tf.matmul(queries, q_tar_embedding)
    k_tar_embedding = tf.reshape(k_tar_embedding, [-1, params.internal_size, params.embedding_size])
    keys = tf.matmul(keys, k_tar_embedding)
    v_tar_embedding = tf.reshape(v_tar_embedding, [-1, params.internal_size, params.embedding_size])
    values = tf.matmul(values, v_tar_embedding)

    q_ = tf.concat(tf.split(queries, params.heads_num, axis=2), axis=0)
    k_ = tf.concat(tf.split(keys, params.heads_num, axis=2), axis=0)
    v_ = tf.concat(tf.split(values, params.heads_num, axis=2), axis=0)

    outputs = tf.matmul(q_, k_, transpose_b=True)

    query_masks = masks
    key_masks = masks

    query_masks = tf.sequence_mask(query_masks, tf.shape(seqs)[1], dtype=tf.float32)
    key_masks = tf.sequence_mask(key_masks, tf.shape(seqs)[1], dtype=tf.float32)
    query_masks = tf.squeeze(query_masks, axis=1)
    key_masks = tf.squeeze(key_masks, axis=1)

    key_masks = tf.tile(key_masks, [params.heads_num, 1])
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

    outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings, )
    outputs -= tf.reduce_max(outputs, axis=-1, keepdims=True)
    outputs = tf.nn.softmax(outputs, axis=-1)

    query_masks = tf.tile(query_masks, [params.heads_num, 1])
    query_masks = tf.tile(tf.expand_dims(
        query_masks, -1), [1, 1, tf.shape(keys)[1]])

    outputs *= query_masks

    result = tf.matmul(outputs, v_)
    result = tf.concat(tf.split(result, params.heads_num, axis=0), axis=2)
    result += tf.tensordot(seqs, w_res, axes=(-1, 0))
    result = tf.nn.relu(result)

    return result


if __name__ == "__main__":
    model_config = define_flags()
    logger, china_tz = setup_logger(model_config, MODEL_NAME)
    if model_config.dt_dir == "":
        model_config.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    model_config.model_dir = model_config.model_dir + (date.today() + timedelta(-1)).strftime('%Y%m%d')

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger, "prob"), argv=[model_config])
