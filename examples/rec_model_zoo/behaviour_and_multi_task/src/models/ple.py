#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from datetime import datetime

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

MODEL_NAME = "PLE"


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
    tf.app.flags.DEFINE_string("exp_per_task", '3,3', "number of experts per task")
    tf.app.flags.DEFINE_integer("shared_num", 2, "number of shared experts")
    tf.app.flags.DEFINE_integer("level_number", 2, "depth")
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
            embeddings[key] = tf.nn.embedding_lookup(emb_weights[key], features[key], name=key + "_embedding_lookup")
            embeddings[key] = tf.reshape(embeddings[key], [-1, 1, params.embedding_size])
        for key in ["109_14", "110_14", "127_14", "150_14", "210", "853"]:
            embeddings[key] = tf.expand_dims(
                embedding_lookup_sparse_fake(emb_weights[key], features[key], combiner="sum",
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

    with (tf.compat.v1.variable_scope("PLE-Net", reuse=tf.compat.v1.AUTO_REUSE)):
        exp_per_task = list(map(int, params.exp_per_task.strip().split(',')))
        expert_units = list(map(int, params.expert_layers.strip().split(',')))

        def ple_net(inputs, is_last, level):
            inputs_final = []
            for input_value in inputs:
                input_shape = input_value.get_shape().as_list()
                inputs_final.append(tf.reshape(input_value, shape=[-1, 1, input_shape[1]]))

            expert_outputs = []

            # task-specific expert part
            for i in range(0, params.task_num):
                for j in range(0, exp_per_task[i]):
                    inp = inputs_final[i]
                    for expert_k, _ in enumerate(expert_units):
                        inp = tf.contrib.layers.fully_connected(inputs=inp, num_outputs=expert_units[expert_k],
                                                                activation_fn=tf.nn.relu,
                                                                scope='level_%d_task_%d_expert_%d_mlp_%d' % (
                                                                    level, i, j, expert_k))
                    expert_outputs.append(inp)  # None * 1 * 256

            # shared expert part
            for i in range(0, params.shared_num):
                inp = inputs_final[-1]
                for expert_j, _ in enumerate(expert_units):
                    inp = tf.contrib.layers.fully_connected(inputs=inp, num_outputs=expert_units[expert_j],
                                                            activation_fn=tf.nn.relu,
                                                            scope='level_%d_shared_expert_%d_mlp_%d' % (
                                                                level, i, expert_j))
                expert_outputs.append(inp)  # None * 1 * 256

            outputs = []

            # cgc gate
            for i in range(0, params.task_num):
                cur_expert_num = exp_per_task[i] + params.shared_num

                cur_gate = tf.contrib.layers.fully_connected(inputs=inputs[i], num_outputs=cur_expert_num,
                                                             activation_fn=tf.nn.softmax,
                                                             scope='level_%d_task_%d_gate_mlp' % (
                                                                 level, i))  # None * cur_expert_num

                cur_gate_shape = cur_gate.get_shape().as_list()
                cur_gate = tf.reshape(cur_gate, shape=[-1, cur_gate_shape[1], 1])

                cur_experts = expert_outputs[i * exp_per_task[i]:(i + 1) * exp_per_task[i]] + \
                              expert_outputs[-int(params.shared_num):]
                expert_concat = tf.concat(cur_experts, axis=1)  # None * cur_expert_num * 256
                cur_gate_expert = tf.multiply(expert_concat, cur_gate)
                cur_gate_expert = tf.reduce_sum(cur_gate_expert, axis=1)  # None * 256
                outputs.append(cur_gate_expert)

            # shared gate
            if not is_last:
                all_expert_num = params.shared_num
                for expert_num in exp_per_task:
                    all_expert_num += expert_num

                cur_gate = tf.contrib.layers.fully_connected(inputs=inputs[-1], num_outputs=all_expert_num,
                                                             activation_fn=tf.nn.softmax,
                                                             scope='level_%d_shared_gate_mlp' % level)

                cur_gate_shape = cur_gate.get_shape().as_list()
                cur_gate = tf.reshape(cur_gate, shape=[-1, cur_gate_shape[1], 1])

                cur_experts = expert_outputs
                expert_concat = tf.concat(cur_experts, axis=1)  # None * all_expert_num * 256
                cur_gate_expert = tf.multiply(expert_concat, cur_gate)
                cur_gate_expert = tf.reduce_sum(cur_gate_expert, axis=1)  # None * 256
                outputs.append(cur_gate_expert)

            return outputs

        task_inputs = []
        for _ in range(params.task_num + 1):
            task_inputs.append(x_deep)

        for i in range(params.level_number):
            if i == params.level_number - 1:  # final layer
                task_outputs = ple_net(task_inputs, is_last=True, level=i)
            else:
                task_inputs = ple_net(task_inputs, is_last=False, level=i)

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
        y_ctr = build_tower(task_outputs[0], name='ctr')
        y_ctr = tf.contrib.layers.fully_connected(inputs=y_ctr, num_outputs=1, activation_fn=None,
                                                  scope='deep_out_click')
        y_ctr = tf.reshape(y_ctr, [-1, ])
        y_ctr_prediction = tf.sigmoid(y_ctr)

        # CVR
        y_cvr = build_tower(task_outputs[1], name='cvr')
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
        raise ValueError("mode should be 'train' or 'eval' or 'predict'.")


if __name__ == "__main__":
    model_config = define_flags()
    logger, china_tz = setup_logger(model_config, MODEL_NAME)
    if model_config.dt_dir == "":
        model_config.dt_dir = datetime.now(china_tz).strftime('%Y%m%d')
    model_config.model_dir = model_config.model_dir + datetime.now(china_tz).strftime('%Y%m%d')

    logger.info("FLAGS: " + str(model_config))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run(main=lambda argv: main(argv[0], model_fn, logger, "multi"), argv=[model_config])
