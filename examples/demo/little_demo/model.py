# coding: UTF-8
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

from __future__ import print_function

import tensorflow as tf


class MyModel:
    def __init__(self):
        self.layer_dims = [1024, 512, 256, 128]
        self.act_func = 'relu'
        self.keep_prob = 0.8
        self._lambda = 4.91e-7
        self.emb_dim = None
        self.loss_list = []
        self.predict_list = []
        self.all_layer_dims = None
        self.h_w, self.h_b = [], []
        self.h_w_head_0, self.h_w_head_1, self.h_b_head_0, self.h_b_head_1 = None, None, None, None

    def __call__(self, embedding_list, label_0, label_1, is_training=True):
        with tf.compat.v1.variable_scope("mlp", reuse=tf.compat.v1.AUTO_REUSE):
            embedding = tf.concat(embedding_list, axis=1)
            self.emb_dim = embedding.shape.as_list()[-1]
            self.all_layer_dims = [self.emb_dim] + self.layer_dims + [1]

            with tf.compat.v1.variable_scope("mlp", reuse=tf.compat.v1.AUTO_REUSE):
                for i in range(len(self.all_layer_dims) - 2):
                    self.h_w.append(tf.compat.v1.get_variable('h%d_w' % (i + 1), shape=self.all_layer_dims[i: i + 2],
                                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                        dtype=tf.float32,
                                        collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_wts"]))
                    self.h_b.append(
                        tf.compat.v1.get_variable('h%d_b' % (i + 1), shape=[self.all_layer_dims[i + 1]],
                                        initializer=tf.zeros_initializer,
                                        dtype=tf.float32,
                                        collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_bias"]))
                i += 1
                self.h_w_head_0 = tf.compat.v1.get_variable('h_w_head_0', shape=self.all_layer_dims[i: i + 2],
                                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                        dtype=tf.float32,
                                        collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_wts"])
                self.h_b_head_0 = tf.compat.v1.get_variable('h_b_head_0', shape=[self.all_layer_dims[i + 1]],
                                        initializer=tf.zeros_initializer,
                                        dtype=tf.float32,
                                        collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_bias"])
                self.h_w_head_1 = tf.compat.v1.get_variable('h_w_head_1', shape=self.all_layer_dims[i: i + 2],
                                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                        dtype=tf.float32,
                                        collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_wts"])
                self.h_b_head_1 = tf.compat.v1.get_variable('h_b_head_1', shape=[self.all_layer_dims[i + 1]],
                                        initializer=tf.zeros_initializer,
                                        dtype=tf.float32,
                                        collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_bias"])

            logit_list = self.forward(embedding, self.act_func, self.keep_prob, training=is_training)

            for logit, label in zip(logit_list, (label_0, label_1)):
                train_preds = tf.sigmoid(logit)

                basic_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=label)

                deep_loss = tf.reduce_mean(basic_loss)  # + _lambda * tf.nn.l2_loss(embedding)
                self.predict_list.append(train_preds)
                self.loss_list.append(deep_loss)


    def forward(self, embedding, act_func, keep_prob, training):
        hidden_output = tf.reshape(embedding, [-1, self.emb_dim]) # *512
        for i, h_w_var in enumerate(self.h_w):
            hidden_output = tf.matmul(self.activate(act_func, hidden_output), h_w_var)
            hidden_output = hidden_output + self.h_b[i]

        def output_head(hidden_output, h_w, h_b):
            hidden_output_branch = tf.matmul(self.activate(act_func, hidden_output), h_w)
            logit = hidden_output_branch + h_b
            logit = tf.reshape(logit, [-1, ])

            return logit

        logit_0 = output_head(hidden_output, self.h_w_head_0, self.h_b_head_0)
        logit_1 = output_head(hidden_output, self.h_w_head_1, self.h_b_head_1)
        logit_list = [logit_0, logit_1]

        return logit_list

    @staticmethod
    def activate(act_func, input_x):
        if act_func == 'tanh':
            return tf.tanh(input_x)
        elif act_func == 'relu':
            return tf.nn.relu(input_x)
        else:
            return tf.sigmoid(input_x)
