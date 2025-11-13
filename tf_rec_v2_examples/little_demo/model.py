#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import List

import tensorflow as tf

from config import Config


class Model:
    def __init__(self):
        self._layer_dims = [1024, 512, 256, 128]
        self._emb_dim = None
        self._loss = 0.0
        self._all_layer_dims = None
        self._h_w = []
        self._h_b = []
        self._h_w_head_0 = None
        self._h_w_head_1 = None
        self._h_b_head_0 = None
        self._h_b_head_1 = None

    def __call__(self, embedding_list: List[tf.Tensor], label_0: tf.Tensor, label_1: tf.Tensor):
        with tf.compat.v1.variable_scope("mlp", reuse=tf.compat.v1.AUTO_REUSE):
            embedding = tf.concat(embedding_list, axis=1)
            self._emb_dim = embedding.shape.as_list()[-1]
            self._all_layer_dims = [self._emb_dim] + self._layer_dims + [1]

            for i in range(len(self._all_layer_dims) - 2):
                self._h_w.append(
                    tf.compat.v1.get_variable(
                        "h%d_w" % (i + 1),
                        shape=self._all_layer_dims[i: i + 2],
                        initializer=tf.truncated_normal_initializer(-0.01, 0.01, Config.random_seed),
                        dtype=tf.float32,
                        collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_wts"],
                    )
                )
                self._h_b.append(
                    tf.compat.v1.get_variable(
                        "h%d_b" % (i + 1),
                        shape=[self._all_layer_dims[i + 1]],
                        initializer=tf.zeros_initializer,
                        dtype=tf.float32,
                        collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_bias"],
                    )
                )

            i += 1
            self._h_w_head_0 = tf.compat.v1.get_variable(
                "h_w_head_0",
                shape=self._all_layer_dims[i: i + 2],
                initializer=tf.truncated_normal_initializer(-0.01, 0.01, Config.random_seed),
                dtype=tf.float32,
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_wts"],
            )
            self._h_b_head_0 = tf.compat.v1.get_variable(
                "h_b_head_0",
                shape=[self._all_layer_dims[i + 1]],
                initializer=tf.zeros_initializer,
                dtype=tf.float32,
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_bias"],
            )
            self._h_w_head_1 = tf.compat.v1.get_variable(
                "h_w_head_1",
                shape=self._all_layer_dims[i: i + 2],
                initializer=tf.truncated_normal_initializer(-0.01, 0.01, Config.random_seed),
                dtype=tf.float32,
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_wts"],
            )
            self._h_b_head_1 = tf.compat.v1.get_variable(
                "h_b_head_1",
                shape=[self._all_layer_dims[i + 1]],
                initializer=tf.zeros_initializer,
                dtype=tf.float32,
                collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_bias"],
            )

            logit_list = self._forward(embedding)

            for logit, label in zip(logit_list, (label_0, label_1)):
                basic_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=label)
                deep_loss = tf.reduce_mean(basic_loss)
                self._loss += deep_loss

    @property
    def loss(self):
        return self._loss

    def _forward(self, embedding):
        hidden_output = tf.reshape(embedding, [-1, self._emb_dim])
        for i, h_w_var in enumerate(self._h_w):
            hidden_output = tf.nn.relu(hidden_output)
            hidden_output = tf.matmul(hidden_output, h_w_var)
            hidden_output = hidden_output + self._h_b[i]

        def _output_head(hidden_output, h_w, h_b):
            hidden_output = tf.nn.relu(hidden_output)
            hidden_output_branch = tf.matmul(hidden_output, h_w)
            logit = hidden_output_branch + h_b
            logit = tf.reshape(logit, [-1])
            return logit

        logit_0 = _output_head(hidden_output, self._h_w_head_0, self._h_b_head_0)
        logit_1 = _output_head(hidden_output, self._h_w_head_1, self._h_b_head_1)
        logit_list = [logit_0, logit_1]

        return logit_list
