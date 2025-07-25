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

from easydict import EasyDict as edict

import tensorflow as tf

model_cfg = edict()
model_cfg.loss_mode = "batch"
LOSS_OP_NAME = "loss"
LABEL_OP_NAME = "label"
VAR_LIST = "variable"
PRED_OP_NAME = "pred"


class Cross:

    def __init__(self, projection_dim=None, diag_scale=0.0):
        self._projection_dim = projection_dim
        self._diag_scale = diag_scale

    def __call__(self, x0, x=None, seed=0):
        result_shape = x0.shape[-1]
        dense_u = tf.layers.dense(x, self._projection_dim,
                                  kernel_initializer=tf.compat.v1.glorot_normal_initializer(seed),
                                  use_bias=False)

        dense_v = tf.layers.dense(dense_u, result_shape,
                                  kernel_initializer=tf.compat.v1.glorot_normal_initializer(seed),
                                  bias_initializer=tf.compat.v1.zeros_initializer(),
                                  use_bias=True
                                  )

        return x0 * dense_v + x


class CrossNetwork(tf.Module):
    def __init__(self, num_layers, projection_dim=None):
        self.cross_layers = []
        for _ in range(num_layers):
            self.cross_layers.append(Cross(projection_dim=projection_dim))

    def __call__(self, x0, seed):
        x = x0
        for cl in self.cross_layers:
            new_seed = seed + 1
            x = cl(x0=x0, x=x, seed=new_seed)
        return x



class MyModel:
    def __init__(self):
        self.kernel_init = None
        self._loss_fn = None
        self.is_training = None
        self.num_cross_layers = 3
        self.cross_layer_projection_dim = 512
        self.cross_interaction_op = CrossNetwork(num_layers=self.num_cross_layers,
                                                 projection_dim=self.cross_layer_projection_dim)

    @classmethod
    def bottom_stack(cls, _input, seed):
        dnn1 = tf.layers.dense(_input, 512, activation='relu', name='bs1',
                               kernel_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               bias_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed)
                               )
        dnn2 = tf.layers.dense(dnn1, 256, activation='relu', name='bs2',
                               kernel_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               bias_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               )
        dnn3 = tf.layers.dense(dnn2, 128, activation='relu', name='bs3',
                               kernel_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               bias_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               )
        return dnn3

    @classmethod
    def top_stack(cls, _input, seed):
        dnn1 = tf.layers.dense(_input, 1024, activation='relu', name='ts1',
                               kernel_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               bias_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               )
        dnn2 = tf.layers.dense(dnn1, 1024, activation='relu', name='ts2',
                               kernel_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               bias_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               )
        dnn3 = tf.layers.dense(dnn2, 512, activation='relu', name='ts3',
                               kernel_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               bias_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               )
        dnn4 = tf.layers.dense(dnn3, 256, activation='relu', name='ts4',
                               kernel_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               bias_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               )
        dnn5 = tf.layers.dense(dnn4, 1, activation=None, name='ts5',
                               kernel_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               bias_initializer=tf.variance_scaling_initializer
                               (scale=1.0, mode="fan_in", distribution='uniform', seed=seed),
                               )
        return dnn5

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
            interaction_args = tf.reshape(interaction_args, (tf.shape(interaction_args)[0], -1))
            interaction_output = self.cross_interaction_op(interaction_args, seed)
            logits = self.top_stack(interaction_output, seed)
            loss = self._loss_fn(label, logits)
            prediction = tf.sigmoid(logits)
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mlp')
            return {LOSS_OP_NAME: loss,
                    PRED_OP_NAME: prediction,
                    LABEL_OP_NAME: label,
                    VAR_LIST: trainable_variables}







my_model = MyModel()
