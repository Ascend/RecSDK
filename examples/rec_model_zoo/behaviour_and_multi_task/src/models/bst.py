# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from dataclasses import dataclass
from typing import Tuple, Dict, Union, Any
from datetime import datetime

import numpy as np
import tensorflow as tf
import pytz

from utils import embedding_lookup_sparse_fake, build_optimizer, main, spec  
from examples.rec_model_zoo.common import setup_logger

tf.compat.v1.set_random_seed(2024)
np.random.seed(2024)
random.seed(2024)

MODEL_NAME = "BST"


def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_integer("transformer_layers", 1, "Number of transformer layers")
    tf.app.flags.DEFINE_integer("att_embedding_size", 4, "Size of attention embedding")
    tf.app.flags.DEFINE_integer("heads_num", 4, "Number of attention heads")
    tf.app.flags.DEFINE_string("attention_layers", '80,40', "Attention Net mlp layers")
    tf.app.flags.DEFINE_string("deep_layers", "512,256,128,64", "deep layers")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f"../checkpoint/aliccp/{MODEL_NAME}/", "code check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", f"../model/serving/{MODEL_NAME}/",
                               "export servable code for TensorFlow Serving")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing code or not")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


def build_embedding_layer(features: Dict[str, tf.Tensor], model_cfg) -> Tuple[
    Dict[Union[str, Any], Any], Dict[str, Any], Dict[str, Any]]:
    """
    Build the embedding layer.

    Args:
        features (Dict[str, tf.Tensor]): Input features.
        model_cfg: Model configuration.

    Returns:
        Dict[str, tf.Tensor]: Embeddings.
    """
    with tf.compat.v1.variable_scope("Embedding-Layer"):
        emb_weights = {}
        for key, vocab_len in spec["vocab_length"].items():
            emb_weights[key] = tf.compat.v1.get_variable(
                name=key + "_emb_wgts",
                shape=[vocab_len + 1, model_cfg.embedding_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
            )

        embeddings = {}
        dense_ids = {}
        dense_len = {}
        for key in ["101", "121", "122", "124", "125", "126", "127",
                    "128", "129", "205", "508", "509", "702", "301"]:
            embeddings[key] = tf.nn.embedding_lookup(emb_weights[key], features[key], name=key + "_embedding_lookup")
            embeddings[key] = tf.reshape(embeddings[key], [-1, 1, model_cfg.embedding_size])

        embeddings["853"] = tf.expand_dims(
            embedding_lookup_sparse_fake(emb_weights.get("853"), features.get("853"), combiner="sum",
                                         name="853" + "_embedding_lookup"),
            axis=1
        )

        for key in ["206", "207", "216"]:
            embeddings[key] = tf.nn.embedding_lookup(emb_weights[key], features[key], name=key + "_embedding_lookup")

        embeddings["210"] = embedding_lookup_sparse_fake(emb_weights.get("210"), features.get("210"), combiner="sum",
                                                         name="210" + "_embedding_lookup")

        for key in ["109_14", "110_14", "127_14", "150_14"]:
            feature_dense = features[key]
            dense_ids[key] = feature_dense
            dense_mask = tf.expand_dims(tf.cast(feature_dense >= 0, tf.float32), axis=-1)  # None * P * 1
            dense_len[key] = tf.reduce_sum(tf.cast(feature_dense >= 0, tf.int32), axis=1, keepdims=True)
            feature_dense = tf.where(tf.equal(feature_dense, -1), tf.zeros_like(feature_dense), feature_dense)
            emb = tf.nn.embedding_lookup(emb_weights[key], feature_dense,
                                         name=key + "_embedding_lookup")  # None * P * E
            emb = tf.multiply(emb, dense_mask)
            embeddings[key] = emb

    return embeddings, dense_ids, dense_len


def build_transformer_layer(embeddings: Dict[str, tf.Tensor], dense_len: Dict[str, tf.Tensor], model_cfg, mode) -> Dict[
    str, tf.Tensor]:
    """
    Build the transformer layer.

    Args:
        embeddings (Dict[str, tf.Tensor]): Embeddings.
        dense_len (Dict[str, tf.Tensor]): Dense lengths.
        model_cfg: Model configuration.

    Returns:
        Dict[str, tf.Tensor]: Updated embeddings.
    """
    with (((tf.compat.v1.variable_scope("Transformer-layer", reuse=tf.compat.v1.AUTO_REUSE)))):
        transformer_num = model_cfg.transformer_layers
        heads_num = model_cfg.heads_num

        def positional_encoding(inputs, maxlen):
            embedding_dim = inputs.get_shape().as_list()[-1]  # static
            batch_size, seq_length = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])  # None * T

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i - i % 2) / embedding_dim) for i in range(embedding_dim)]
                for pos in range(maxlen)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # maxlen * E

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            outputs = outputs * embedding_dim ** 0.5  # scale

            return outputs + inputs

        def positional_encoding_learn(inputs, maxlen, scope="position_encoding"):
            embedding_dim = inputs.get_shape().as_list()[-1]  # static
            batch_size, seq_length = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])  # None * T

            # First part of the PE function: sin and cos argument
            position_enc = tf.compat.v1.get_variable(scope, [maxlen, embedding_dim],  # W_shape
                                                     initializer=tf.contrib.layers.xavier_initializer())

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            outputs = outputs * embedding_dim ** 0.5  # scale

            return outputs + inputs

        def layer_normalization(inputs, scope="ln"):
            epsilon = 1e-9
            gamma = tf.compat.v1.get_variable(name="gamma_%s" % scope, shape=[inputs.get_shape().as_list()[-1], ],
                                              initializer=tf.ones_initializer())
            beta = tf.compat.v1.get_variable(name="beta_%s" % scope, shape=[inputs.get_shape().as_list()[-1], ],
                                             initializer=tf.zeros_initializer())
            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
            outputs = gamma * normalized + beta
            return outputs

        @dataclass
        class TransformerLayerParams:
            att_embedding_size: int
            heads_num: int
            dropout_rate: float = 0.2
            layer_name: str = "key"
            layer_index: int = 0

        def transformer_layer(inputs, masks, params_t: TransformerLayerParams):
            att_embedding_size = params_t.att_embedding_size
            heads_num = params_t.heads_num
            dropout_rate = params_t.dropout_rate
            layer_name = params_t.layer_name
            layer_index = params_t.layer_index
            num_units = att_embedding_size * heads_num
            embedding_size = int(inputs.get_shape().as_list()[-1])

            if num_units != embedding_size:
                raise ValueError(
                    "att_embedding_size * heads_num must equal the last dimension size of inputs,got %d * %d != %d" % (
                        att_embedding_size, heads_num, embedding_size))

            w_q = tf.compat.v1.get_variable(name="weight_Q_%s_%d" % (layer_name, layer_index),
                                            shape=[embedding_size, num_units],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            w_k = tf.compat.v1.get_variable(name="weight_K_%s_%d" % (layer_name, layer_index),
                                            shape=[embedding_size, num_units],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            w_v = tf.compat.v1.get_variable(name="weight_V_%s_%d" % (layer_name, layer_index),
                                            shape=[embedding_size, num_units],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )

            fw1 = tf.compat.v1.get_variable(name="fw1_%s_%d" % (layer_name, layer_index),
                                            shape=[num_units, 4 * num_units],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            fw2 = tf.compat.v1.get_variable(name="fw2_%s_%d" % (layer_name, layer_index),
                                            shape=[4 * num_units, num_units],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )

            queries = inputs
            keys = inputs
            query_masks = masks
            key_masks = masks

            query_masks = tf.sequence_mask(query_masks, tf.shape(inputs)[1], dtype=tf.float32)
            key_masks = tf.sequence_mask(key_masks, tf.shape(inputs)[1], dtype=tf.float32)
            query_masks = tf.squeeze(query_masks, axis=1)
            key_masks = tf.squeeze(key_masks, axis=1)

            q = tf.tensordot(queries, w_q, axes=(-1, 0))  # None * T_q * (D * h)
            k = tf.tensordot(keys, w_k, axes=(-1, 0))
            v = tf.tensordot(keys, w_v, axes=(-1, 0))

            # NOTE: Compute Expression : (h x None) x T_q x D
            q_ = tf.concat(tf.split(q, heads_num, axis=2), axis=0)
            k_ = tf.concat(tf.split(k, heads_num, axis=2), axis=0)
            v_ = tf.concat(tf.split(v, heads_num, axis=2), axis=0)

            # scaled_dot_product
            outputs = tf.matmul(q_, k_, transpose_b=True)
            outputs = outputs / (k_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.tile(key_masks, [heads_num, 1])

            # NOTE: Compute Expression : (h x None) x T_q x T_k
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

            # NOTE: Compute Expression : (h x None) x T_q x T_k
            outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings, )

            outputs -= tf.reduce_max(outputs, axis=-1, keepdims=True)
            outputs = tf.nn.softmax(outputs, axis=-1)
            query_masks = tf.tile(query_masks, [heads_num, 1])  # (h * None) * T_q
            # NOTE: Compute Expression : (h x None) x T_q x T_k
            query_masks = tf.tile(tf.expand_dims(
                query_masks, -1), [1, 1, tf.shape(keys)[1]])

            outputs *= query_masks

            if mode == tf.estimator.ModeKeys.TRAIN:
                outputs = tf.nn.dropout(outputs, rate=dropout_rate)

            # Weighted sum
            result = tf.matmul(outputs, v_)
            result = tf.concat(tf.split(result, heads_num, axis=0), axis=2)

            # res
            result += queries

            result = layer_normalization(result, scope="%s_%d_ln1" % (layer_name, layer_index))

            fw1 = tf.nn.leaky_relu(tf.tensordot(result, fw1, axes=[-1, 0]))
            if mode == tf.estimator.ModeKeys.TRAIN:
                fw1 = tf.nn.dropout(fw1, rate=dropout_rate)
            fw2 = tf.tensordot(fw1, fw2, axes=[-1, 0])

            result += fw2

            result = layer_normalization(result, scope="%s_%d_ln2" % (layer_name, layer_index))

            return result




        for key in ["109_14", "110_14", "127_14", "150_14"]:
            embeddings[key] = positional_encoding_learn(embeddings[key], maxlen=model_cfg.max_seq_len,
                                                        scope=key + "_pos")
            for i in range(transformer_num):
                params = TransformerLayerParams(
                    att_embedding_size=embeddings[key].get_shape().as_list()[-1] // heads_num,
                    heads_num=heads_num,
                    layer_name=key,
                    layer_index=i
                )
                embeddings[key] = transformer_layer(inputs=embeddings[key],
                                                    masks=dense_len[key],
                                                    params_t=params)

    return embeddings


def build_field_wise_pooling_layer(embeddings: Dict[str, tf.Tensor], dense_ids: Dict[str, tf.Tensor],
                                   dense_len: Dict[str, tf.Tensor], model_cfg) -> Dict[str, tf.Tensor]:
    """
    Build the field-wise pooling layer.

    Args:
        embeddings (Dict[str, tf.Tensor]): Embeddings.
        dense_ids (Dict[str, tf.Tensor]): Dense IDs.
        dense_len (Dict[str, tf.Tensor]): Dense lengths.
        model_cfg: Model configuration.

    Returns:
        Dict[str, tf.Tensor]: Updated embeddings.
    """
    with tf.compat.v1.variable_scope("Field-wise-Pooling-layer", reuse=tf.compat.v1.AUTO_REUSE):
        attention_layers = list(map(int, model_cfg.attention_layers.strip().split(',')))

        def attention_unit(a_xx_emb, ub_dense_id, ub_emb, masks, unit_name="targ_hist"):
            dense_mask = tf.sequence_mask(masks, tf.shape(ub_emb)[1])

            padded_dim = tf.shape(ub_dense_id)[1]

            ax_emb = tf.reshape(tf.tile(a_xx_emb, [1, padded_dim]),
                                shape=[-1, padded_dim, model_cfg.embedding_size])  # None * E --> None * P * E
            x_inputs = tf.concat([ax_emb, ub_emb, ax_emb - ub_emb, ax_emb * ub_emb], axis=-1)  # None * P * 4E
            for att_i, _ in enumerate(attention_layers):
                x_inputs = tf.contrib.layers.fully_connected(inputs=x_inputs, num_outputs=attention_layers[att_i],
                                                             activation_fn=tf.nn.leaky_relu,
                                                             scope="att_fc_%s_%d" % (unit_name, att_i))

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
                                                 masks=dense_len[his_key],
                                                 unit_name="%s_%s" % (target_key, his_key))  # None * 1 * E

    for key in ["206", "207", "210", "216"]:
        embeddings[key] = tf.reshape(embeddings[key], [-1, 1, model_cfg.embedding_size])

    return embeddings


def build_mlp_layer(embedding: tf.Tensor, model_cfg) -> tf.Tensor:
    """
    Build the MLP layer.

    Args:
        embedding (tf.Tensor): Input embedding.
        model_cfg: Model configuration.

    Returns:
        tf.Tensor: Output of the MLP layer.
    """
    x_deep = tf.reshape(embedding, [-1, 23 * model_cfg.embedding_size])  # None * (23 * E)

    with tf.compat.v1.variable_scope("MLP-layer"):
        deep_layers = list(map(int, model_cfg.deep_layers.strip().split(',')))

        for layer_i, _ in enumerate(deep_layers):
            x_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=deep_layers[layer_i],
                                                       activation_fn=tf.nn.leaky_relu, scope='mlp%d' % layer_i)

    return x_deep


def build_output_layer(x_deep: tf.Tensor) -> tf.Tensor:
    """
    Build the output layer.

    Args:
        x_deep (tf.Tensor): Input tensor from MLP layer.

    Returns:
        tf.Tensor: Output predictions.
    """
    with tf.compat.v1.variable_scope("BST-out"):
        y_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=1, activation_fn=tf.identity,
                                                   scope='bst_out')
        y = tf.reshape(y_deep, shape=[-1])
        pred = tf.sigmoid(y)

    return pred


def build_loss(labels: Dict[str, tf.Tensor], pred: tf.Tensor) -> tf.Tensor:
    """
    Build the loss function.

    Args:
        labels (Dict[str, tf.Tensor]): Ground truth labels.
        pred (tf.Tensor): Predictions.

    Returns:
        tf.Tensor: Loss value.
    """
    with tf.compat.v1.variable_scope("loss-function-part"):
        epsilon = 1e-7
        click_weight = 0.14

        ctr_loss = - (1 - click_weight) / click_weight * labels['y'] * tf.math.log(pred + epsilon) - \
                   (1 - labels['y']) * tf.math.log(1 - pred + epsilon)
        loss = tf.reduce_mean(ctr_loss)

    return loss


def model_fn(features, labels, mode, params):
    """build Estimator model"""

    embeddings, dense_ids, dense_len = build_embedding_layer(features, params)

    embeddings = build_transformer_layer(embeddings, dense_len, params, mode)

    embeddings = build_field_wise_pooling_layer(embeddings, dense_ids, dense_len, params)

    for key in ["206", "207", "210", "216"]:
        embeddings[key] = tf.reshape(embeddings[key], [-1, 1, params.embedding_size])

    embedding = tf.concat(
        [embeddings[field_name] for field_name in spec["one_hot_fields"]] +
        [embeddings[field_name] for field_name in spec["multi_hot_fields"]] +
        [embeddings[field_name] for field_name in spec["special_fields"]],
        axis=2,
    )  # None * 1 * 23 * E)

    x_deep = build_mlp_layer(embedding, params)

    pred = build_output_layer(x_deep)

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
    loss = build_loss(labels, pred)

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
        raise ValueError("Mode not supported: {}".format(mode))


if __name__ == "__main__":
    model_config = define_flags()
    logger = setup_logger(model_config, MODEL_NAME)
    china_tz = pytz.timezone('Asia/Shanghai')
    model_config.model_dir = model_config.model_dir + datetime.now(china_tz).strftime('%Y%m%d')

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger, "prob"), argv=[model_config])
