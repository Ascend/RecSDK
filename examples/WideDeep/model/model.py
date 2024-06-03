# coding=utf-8
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

import time
from easydict import EasyDict as edict

import tensorflow as tf


model_cfg = edict()
model_cfg.loss_mode = "batch"
LOSS_OP_NAME = "loss"
LABEL_OP_NAME = "label"
VAR_LIST = "variable"
PRED_OP_NAME = "pred"


class MyModel:
    def __init__(self):
        self.kernel_init = None
        self._loss_fn = None
        self.is_training = None

    def build_model(self,model_args):
        wide_embedding = model_args.wide_embedding
        deep_embedding = model_args.deep_embedding
        label = model_args.label
        is_training = model_args.is_training
        seed = model_args.seed
        dropout_rate = model_args.dropout_rate
        batch_norm = model_args.batch_norm

        with tf.variable_scope("wide_deep", reuse=tf.AUTO_REUSE):
            self._loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.is_training = is_training

            # wide
            batch_size, wide_num, wide_emb_dim = wide_embedding.shape
            wide_input = tf.reshape(wide_embedding[:, :, 0], shape=(batch_size, wide_num * 1))
            wide_output = tf.reshape(tf.reduce_sum(wide_input, axis=1), shape=(-1, 1))

            # deep
            batch_size, deep_num, deep_emb_dim = deep_embedding.shape
            deep_input = tf.reshape(deep_embedding, shape=(batch_size, deep_num * deep_emb_dim))

            ## MLP
            hidden_units = [256, 128, 64]
            net = deep_input
            for i, unit in enumerate(hidden_units):

                net = tf.layers.dense(net, units=unit, activation='relu', name=f'hidden_layer_{i}',
                                      kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                                      bias_initializer=tf.zeros_initializer())

                if dropout_rate is not None and 0.0 < dropout_rate < 1.0:
                    net = tf.layers.dropout(net, dropout_rate, training=self.is_training)
                if batch_norm:
                    net = tf.layers.batch_normalization(net, training=self.is_training)

            deep_output = tf.layers.dense(net, units=1, activation=None, name='deep_output',
                                          kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                                          bias_initializer=tf.zeros_initializer())

            total_logits = 0.5 * tf.add(wide_output, deep_output, name='total_logits')
            loss = self._loss_fn(label, total_logits)
            prediction = tf.sigmoid(total_logits)
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide_deep')
            return {LOSS_OP_NAME: loss,
                    PRED_OP_NAME: prediction,
                    LABEL_OP_NAME: label,
                    VAR_LIST: trainable_variables}


my_model = MyModel()
