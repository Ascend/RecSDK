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

    @classmethod
    def _dot_interaction(cls, _input):
        num_features = tf.shape(_input)[1]
        batch_size = tf.shape(_input)[0]
        xactions = tf.matmul(_input, _input, transpose_b=True)
        ones = tf.ones_like(xactions, dtype=tf.float32)
        upper_tri_mask = tf.linalg.band_part(ones, 0, -1)

        activations = tf.where(condition=tf.cast(upper_tri_mask, tf.bool),
                                x=tf.zeros_like(xactions),
                                y=xactions)
        out_dim = num_features * num_features
        activations = tf.reshape(activations, (batch_size, out_dim))
        return activations

    def build_model(self,
                    embedding=None,
                    dense_feature=None,
                    label=None,
                    is_training=True,
                    seed=None):
        with tf.variable_scope("mlp", reuse=tf.AUTO_REUSE):
            self._loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.is_training = is_training
            dense_embedding_vec = self.bottom_stack(dense_feature, seed)
            dense_embedding = tf.expand_dims(dense_embedding_vec, 1)
            interaction_args = tf.concat([dense_embedding, embedding], axis=1)
            interaction_output = self._dot_interaction(interaction_args)
            feature_interaction_output = tf.concat([dense_embedding_vec, interaction_output], axis=1)
            # (8192, 857)
            logits = self.top_stack(feature_interaction_output, seed)
            loss = self._loss_fn(label, logits)
            prediction = tf.sigmoid(logits)
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mlp')
            return {LOSS_OP_NAME: loss,
                    PRED_OP_NAME: prediction,
                    LABEL_OP_NAME: label,
                    VAR_LIST: trainable_variables}

    def bottom_stack(self, _input, seed):
        dnn1 = tf.layers.dense(_input, 512, activation='relu', name='bs1',
                               kernel_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed),
                               bias_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed),
                               kernel_regularizer=tf.contrib.layers.l1_regularizer(1e-2))
        dnn2 = tf.layers.dense(dnn1, 256, activation='relu', name='bs2', kernel_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), bias_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), kernel_regularizer=tf.contrib.layers.l1_regularizer(1e-2))
        dnn3 = tf.layers.dense(dnn2, 128, activation='relu', name='bs3', kernel_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), bias_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), kernel_regularizer=tf.contrib.layers.l1_regularizer(1e-2))
        return dnn3

    def top_stack(self, _input, seed):
        dnn1 = tf.layers.dense(_input, 1024, activation='relu', name='ts1', kernel_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), bias_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), kernel_regularizer=tf.contrib.layers.l1_regularizer(1e-2))
        dnn2 = tf.layers.dense(dnn1, 1024, activation='relu', name='ts2', kernel_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), bias_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), kernel_regularizer=tf.contrib.layers.l1_regularizer(1e-2))
        dnn3 = tf.layers.dense(dnn2, 512, activation='relu', name='ts3', kernel_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), bias_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), kernel_regularizer=tf.contrib.layers.l1_regularizer(1e-2))
        dnn4 = tf.layers.dense(dnn3, 256, activation='relu', name='ts4', kernel_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), bias_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), kernel_regularizer=tf.contrib.layers.l1_regularizer(1e-2))
        dnn5 = tf.layers.dense(dnn4, 1, activation=None, name='ts5', kernel_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), bias_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution='normal', seed=seed), kernel_regularizer=tf.contrib.layers.l1_regularizer(1e-2))
        return dnn5


my_model = MyModel()
