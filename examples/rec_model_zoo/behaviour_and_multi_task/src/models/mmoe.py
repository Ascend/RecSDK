#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from datetime import datetime

import tensorflow as tf
import pytz

import utils
from utils import embedding_lookup_sparse_fake, build_optimizer, main
from examples.rec_model_zoo.common import setup_logger

tf.compat.v1.enable_control_flow_v2()
tf.compat.v1.enable_resource_variables()
tf.compat.v1.set_random_seed(2024)
random.seed(2024)

MODEL_NAME = "MMOE"


def define_flags():
    model_conf = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
    tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_string("expert_layers", "512,256", "expert layers")
    tf.app.flags.DEFINE_string("tower_layers", "128,64", "tower layers")
    tf.app.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", f"../checkpoint/aliccp/{MODEL_NAME}/", "code check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", f"../model/serving/{MODEL_NAME}/",
                               "export servable code for TensorFlow Serving")
    tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing code or not")
    tf.app.flags.DEFINE_integer("task_num", 2, "task num")
    tf.app.flags.DEFINE_integer("experts_num", 8, "Number of experts")
    tf.app.flags.DEFINE_string("log_level", "DEBUG", "log level {DEBUG, INFO, WARNING, ERROR, CRITICAL}")
    return model_conf


def build_embedding_layer(features: dict, spec: dict, model_cfg: object) -> tf.Tensor:
    """
    Build the embedding layer for the model.

    Args:
        features (dict): The input features.
        spec (dict): The specification dictionary containing vocab lengths and field names.
        model_cfg (object): The model configuration object containing embedding size.

    Returns:
        tf.Tensor: The concatenated and reshaped embedding tensor.
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
    for key in ["101", "121", "122", "124", "125", "126", "127", "128", "129", "205", "206", "207", "216", "508", "509",
                "702", "301"]:
        embeddings[key] = tf.nn.embedding_lookup(emb_weights[key], features[key], name=key + "_embedding_lookup")
        embeddings[key] = tf.reshape(embeddings[key], [-1, 1, model_cfg.embedding_size])
    for key in ["109_14", "110_14", "127_14", "150_14", "210", "853"]:
        embeddings[key] = tf.expand_dims(
            embedding_lookup_sparse_fake(emb_weights[key], features[key], combiner="sum",
                                         name=key + "_embedding_lookup"),
            axis=1
        )

    embedding = tf.concat(
        [embeddings.get(field_name) for field_name in spec.get("one_hot_fields")] +
        [embeddings.get(field_name) for field_name in spec.get("multi_hot_fields")] +
        [embeddings.get(field_name) for field_name in spec.get("special_fields")],
        axis=2,
    )

    # sparse feature class num
    embedding_feature_num = 23
    return tf.reshape(embedding, [-1, embedding_feature_num * model_cfg.embedding_size])


def build_experts(x_deep: tf.Tensor, model_cfg: object) -> tf.Tensor:
    """
    Build the experts for the model.

    Args:
        x_deep (tf.Tensor): The input tensor.
        model_cfg (object): The model configuration object containing expert layers and number of experts.

    Returns:
        tf.Tensor: The concatenated experts tensor.
    """
    experts = []
    expert_units = list(map(int, model_cfg.expert_layers.strip().split(',')))
    for expert_i in range(model_cfg.experts_num):
        y_dnn = x_deep
        for mlp_j, _ in enumerate(expert_units):
            y_dnn = tf.contrib.layers.fully_connected(inputs=y_dnn, num_outputs=expert_units[mlp_j],
                                                      activation_fn=tf.nn.relu,
                                                      scope='expert_%d_mlp_%d' % (expert_i, mlp_j))
        experts.append(tf.expand_dims(y_dnn, axis=1))
    return tf.concat(experts, axis=1)


def build_gate_networks(x_deep: tf.Tensor, model_cfg: object) -> list:
    """
    Build the gate networks for the model.

    Args:
        x_deep (tf.Tensor): The input tensor.
        model_cfg (object): The model configuration object containing task number and number of experts.

    Returns:
        list: A list of gate networks tensors.
    """
    gate_networks = []
    for i in range(model_cfg.task_num):
        gate_network = tf.contrib.layers.fully_connected(
            inputs=x_deep,
            num_outputs=model_cfg.experts_num,
            activation_fn=tf.nn.softmax,
            scope='gate_%d_mlp' % i)
        gate_network_shape = gate_network.get_shape().as_list()
        gate_network = tf.reshape(gate_network, shape=[-1, gate_network_shape[1], 1])
        gate_networks.append(gate_network)
    return gate_networks


def build_task_outputs(experts: tf.Tensor, gate_networks: list) -> list:
    """
    Build the task outputs by combining experts and gate networks.

    Args:
        experts (tf.Tensor): The tensor containing expert outputs.
        gate_networks (list): A list of gate network tensors.

    Returns:
        list: A list of reshaped task output tensors.
    """
    task_outputs = []
    for gate_network in gate_networks:
        task_out = tf.multiply(experts, gate_network)
        task_out_shape = task_out.get_shape().as_list()
        task_outputs.append(tf.reshape(task_out, shape=[-1, task_out_shape[1] * task_out_shape[2]]))
    return task_outputs


def build_tower(tower_input: tf.Tensor, name: str, model_cfg: object) -> tf.Tensor:
    """
    Build the tower network for a specific task.

    Args:
        tower_input (tf.Tensor): The input tensor for the tower.
        name (str): The name of the tower.
        model_cfg (object): The model configuration object containing tower layers.

    Returns:
        tf.Tensor: The output tensor of the tower network.
    """
    tower_units = list(map(int, model_cfg.tower_layers.strip().split(',')))
    y_tower = tower_input
    for tower_i, _ in enumerate(tower_units):
        y_tower = tf.contrib.layers.fully_connected(inputs=y_tower, num_outputs=tower_units[tower_i],
                                                    activation_fn=tf.nn.relu,
                                                    scope=name + '_tower_mlp_%d' % tower_i)
    return y_tower


def build_predictions(task_outputs: list, model_cfg: object) -> dict:
    """
    Build the predictions for the model.

    Args:
        task_outputs (list): A list of task output tensors.
        model_cfg (object): The model configuration object containing tower layers.

    Returns:
        dict: A dictionary containing the predictions for ctr, cvr, and ctcvr.
    """
    y_ctr = build_tower(task_outputs[0], name='ctr', model_cfg=model_cfg)
    y_ctr = tf.contrib.layers.fully_connected(inputs=y_ctr, num_outputs=1, activation_fn=None, scope='deep_out_click')
    y_ctr = tf.reshape(y_ctr, [-1, ])
    y_ctr_prediction = tf.sigmoid(y_ctr)

    y_cvr = build_tower(task_outputs[1], name='cvr', model_cfg=model_cfg)
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
    return predictions


def build_loss(labels: dict, y_ctr_prediction: tf.Tensor, y_ctcvr_prediction: tf.Tensor,
               model_cfg: object) -> tf.Tensor:
    """
    Build the loss function for the model.

    Args:
        labels (dict): A dictionary containing the true labels for ctr and ctcvr.
        y_ctr_prediction (tf.Tensor): The predicted ctr values.
        y_ctcvr_prediction (tf.Tensor): The predicted ctcvr values.
        model_cfg (object): The model configuration object containing loss weights.

    Returns:
        tf.Tensor: The combined loss tensor.
    """
    epsilon = 1e-7
    # Weight for the click-through rate (CTR) loss component
    click_weight = 0.14
    # Weight for the conversion rate (CVR) loss component
    conversion_weight = 0.023
    # Weight for the CTR task in the combined loss function
    ctr_task_wgt = model_cfg.ctr_task_wgt

    ctr_loss = - (1 - click_weight) / click_weight * labels['y'] * tf.math.log(y_ctr_prediction + epsilon) - \
               (1 - labels['y']) * tf.math.log(1 - y_ctr_prediction + epsilon)
    ctr_loss = tf.reduce_mean(ctr_loss)

    ctcvr_loss = - (1 - conversion_weight) / conversion_weight * labels['z'] * tf.math.log(
        y_ctcvr_prediction + epsilon) - \
                 (1 - labels['z']) * tf.math.log(1 - y_ctcvr_prediction + epsilon)
    ctcvr_loss = tf.reduce_mean(ctcvr_loss)

    return ctr_task_wgt * ctr_loss + (1 - ctr_task_wgt) * ctcvr_loss


def model_fn(features: dict, labels: dict, mode: tf.estimator.ModeKeys,
             params: object) -> tf.estimator.EstimatorSpec:
    """
    Build the model function for the estimator.

    Args:
        features (dict): The input features.
        labels (dict): The true labels.
        mode (tf.estimator.ModeKeys): The mode (TRAIN, EVAL, PREDICT).
        params (object): The model configuration object.

    Returns:
        tf.estimator.EstimatorSpec: The EstimatorSpec object for the given mode.
    """
    # Build the embedding layer
    x_deep = build_embedding_layer(features, utils.spec, params)

    # Build the experts
    experts = build_experts(x_deep, params)

    # Build the gate networks
    gate_networks = build_gate_networks(x_deep, params)

    # Build the task outputs
    task_outputs = build_task_outputs(experts, gate_networks)

    # Build the predictions
    predictions = build_predictions(task_outputs, params)

    # Define the export outputs for serving
    export_outputs = {
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    loss = build_loss(labels, predictions["ctr"], predictions["ctcvr"], params)
    train_op = build_optimizer(loss, params)

    if mode == tf.estimator.ModeKeys.EVAL:
        ctr_mask = labels["y"] > 0
        cvr_labels = tf.boolean_mask(labels["z"], ctr_mask)
        cvr_pre = tf.boolean_mask(predictions["cvr"], ctr_mask)

        eval_metric_ops = {
            "auc_ctr": tf.compat.v1.metrics.auc(labels["y"], predictions["ctr"]),
            "auc_cvr": tf.compat.v1.metrics.auc(cvr_labels, cvr_pre),
            "auc_ctcvr": tf.compat.v1.metrics.auc(labels["z"], predictions["ctcvr"])
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
    else:
        raise ValueError("Unsupported mode: {}".format(mode))


if __name__ == "__main__":
    model_config = define_flags()
    logger = setup_logger(model_config, MODEL_NAME)
    china_tz = pytz.timezone('Asia/Shanghai')
    model_config.model_dir = model_config.model_dir + datetime.now(china_tz).strftime('%Y%m%d')

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger, "multi"), argv=[model_config])
