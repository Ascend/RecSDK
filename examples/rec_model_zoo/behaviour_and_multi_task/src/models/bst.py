# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import random
import shutil
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Union, Any
from datetime import datetime
from functools import partial

import pytz
import numpy as np
import tensorflow as tf
from npu_bridge.npu_init import NPUEstimator, NPURunConfig

from utils import get_third_nearest_checkpoint, dump_pred_prob, json_file_load

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


def parse_example(mode_type: str, example: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """
    Parse a single example for the given mode type.

    Args:
        mode_type (str): The mode type (e.g., TRAIN, EVAL, PREDICT).
        example (tf.Tensor): The serialized example to parse.

    Returns:
        Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]: A tuple containing
        the input dictionary and target dictionary.
    """
    # Parse the example using the feature descriptions for the given mode type
    parsed_example = tf.io.parse_example(example, feature_descriptions.get(mode_type))

    input_dict = {}
    target = {"y": parsed_example["y"], "z": parsed_example["z"]}

    # Extract one-hot fields from the parsed example
    for index, key in enumerate(spec["one_hot_fields"]):
        input_dict[key] = parsed_example["one_hot_fields"][:, index]

    # Extract multi-hot fields from the parsed example
    for key in spec["multi_hot_fields"]:
        input_dict[key] = parsed_example[key]

    # Extract special fields from the parsed example
    for key in spec["special_fields"]:
        input_dict[key] = parsed_example[key]

    return input_dict, target


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


def embedding_lookup_sparse_fake(params: tf.Tensor, ids: tf.Tensor, combiner: str = None,
                                 name: str = None) -> tf.Tensor:
    """
    Perform sparse embedding lookup and combine the results.

    Args:
        params (tf.Tensor): The embedding parameters.
        ids (tf.Tensor): The sparse IDs to lookup.
        combiner (str, optional): The combiner method ('sum' or 'mean'). Defaults to None.
        name (str, optional): The name for the operation. Defaults to None.

    Returns:
        tf.Tensor: The combined embedding results.

    Raises:
        ValueError: If the combiner is not 'sum' or 'mean'.
    """

    # Create a dense mask where valid IDs are marked as 1.0 and invalid IDs as 0.0
    dense_mask = tf.expand_dims(tf.cast(ids >= 0, tf.float32), axis=-1)

    # Replace invalid IDs (-1) with zeros
    ids = tf.where(tf.equal(ids, -1), tf.zeros_like(ids), ids)
    embedding = tf.nn.embedding_lookup(params, ids, name=name + "_dense_lookup") * dense_mask
    summed_embedding = tf.reduce_sum(embedding, axis=1)
    if combiner == "sum":
        return summed_embedding
    elif combiner == "mean":
        return summed_embedding / tf.reduce_sum(dense_mask, axis=1)
    else:
        raise ValueError("combiner only supports 'sum' or 'mean'")


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

            # Note: (h * None) * T_q * D
            q_ = tf.concat(tf.split(q, heads_num, axis=2), axis=0)
            k_ = tf.concat(tf.split(k, heads_num, axis=2), axis=0)
            v_ = tf.concat(tf.split(v, heads_num, axis=2), axis=0)

            # scaled_dot_product
            outputs = tf.matmul(q_, k_, transpose_b=True)
            outputs = outputs / (k_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.tile(key_masks, [heads_num, 1])

            # Note: (h * None) * T_q * T_k
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

            # Note: (h * None) * T_q * T_k
            outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings, )

            outputs -= tf.reduce_max(outputs, axis=-1, keepdims=True)
            outputs = tf.nn.softmax(outputs, axis=-1)
            query_masks = tf.tile(query_masks, [heads_num, 1])  # (h * None) * T_q
            # Note: (h * None) * T_q * T_k
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
        raise ValueError("Mode not supported: {}".format(mode))


def main(model_cfg):
    model_cfg.model_dir = model_cfg.model_dir + datetime.now(china_tz).strftime('%Y%m%d')

    train_order = json_file_load("order", "./order.json")

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
        raise ValueError("task_type should be 'train', 'eval', 'infer', 'profiling_train' or 'profiling_infer'")


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
    for mode_type in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
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
            feature_description[mul_fields] = tf.io.FixedLenFeature(
                [spec[f"{key_map[mode_type]}_max_length"][mul_fields]],
                tf.int64)
        for mul_fields in spec["special_fields"]:
            feature_description[mul_fields] = tf.io.FixedLenFeature(
                [spec[f"{key_map[mode_type]}_max_length"][mul_fields]],
                tf.int64)
        feature_descriptions[mode_type] = feature_description

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0]), argv=[model_config])
