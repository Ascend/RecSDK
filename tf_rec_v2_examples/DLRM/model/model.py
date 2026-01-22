#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

from typing import Optional, Dict, Union

import tensorflow as tf


LOSS_OP_NAME = "loss"
LABEL_OP_NAME = "label"
PRED_OP_NAME = "pred"
SPARSE_FEATURE_LEN = 26
DENSE_FEATURE_LEN = 13


class DLRM:
    """Deep Learning Recommendation Model (DLRM) class."""

    def __init__(self, toml_config: Dict[str, Union[str, int, float]]):
        self._variable_scope_name = "mlp"
        self._l1_regularizer = toml_config["model"]["l1_regularizer"]
        self._bottom_stack_dnn1_shape = toml_config["model"]["bottom_stack_dnn1_shape"]
        self._bottom_stack_dnn2_shape = toml_config["model"]["bottom_stack_dnn2_shape"]
        self._bottom_stack_dnn3_shape = toml_config["model"]["bottom_stack_dnn3_shape"]
        self._top_stack_dnn1_shape = toml_config["model"]["top_stack_dnn1_shape"]
        self._top_stack_dnn2_shape = toml_config["model"]["top_stack_dnn2_shape"]
        self._top_stack_dnn3_shape = toml_config["model"]["top_stack_dnn3_shape"]
        self._top_stack_dnn4_shape = toml_config["model"]["top_stack_dnn4_shape"]
        self._top_stack_dnn5_shape = toml_config["model"]["top_stack_dnn5_shape"]

    def build_model(
        self,
        embedding: Optional[tf.Tensor] = None,
        dense_feature: Optional[tf.Tensor] = None,
        label: Optional[tf.Tensor] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, tf.Tensor]:
        """Builds the DLRM model.

        Args:
            embedding (Optional[tf.Tensor]): Tensor containing the embedding features.
            dense_feature (Optional[tf.Tensor]): Tensor containing the dense features.
            label (Optional[tf.Tensor]): Tensor containing the labels.
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            Dict[str, tf.Tensor]: A dictionary containing the loss, prediction, and label tensors.
        """
        with tf.variable_scope(self._variable_scope_name, reuse=tf.AUTO_REUSE):

            def _dot_interaction(input_tensor: tf.Tensor) -> tf.Tensor:
                num_features = tf.shape(input_tensor)[1]
                batch_size = tf.shape(input_tensor)[0]
                xactions = tf.matmul(input_tensor, input_tensor, transpose_b=True)
                ones = tf.ones_like(xactions, dtype=tf.float32)
                upper_tri_mask = tf.linalg.band_part(ones, 0, -1)

                # The Select operator has not been delivered, and a workaround method is used. The original code is:
                # `"tf.where(condition=tf.cast(upper_tri_mask, tf.bool), x=tf.zeros_like(xactions), y=xactions)"`.
                mask = tf.cast(tf.cast(upper_tri_mask, tf.bool), tf.float32)
                activations = mask * tf.zeros_like(xactions) + (1 - mask) * xactions

                out_dim = num_features * num_features
                activations = tf.reshape(activations, (batch_size, out_dim))
                return activations

            dense_embedding_vec = self._bottom_stack(dense_feature, seed)
            dense_embedding = tf.expand_dims(dense_embedding_vec, 1)
            interaction_args = tf.concat([dense_embedding, embedding], axis=1)
            interaction_output = _dot_interaction(interaction_args)
            feature_interaction_output = tf.concat([dense_embedding_vec, interaction_output], axis=1)
            # (8192, 857)
            logits = self._top_stack(feature_interaction_output, seed)
            label_float = tf.cast(label, dtype=tf.float32)
            per_sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_float, logits=logits)
            loss = tf.reduce_mean(per_sample_loss)
            prediction = tf.sigmoid(logits)
            return {LOSS_OP_NAME: loss, PRED_OP_NAME: prediction, LABEL_OP_NAME: label}

    def _bottom_stack(self, input_tensor: tf.Tensor, seed: int) -> tf.Tensor:
        dnn1 = tf.layers.dense(
            input_tensor,
            self._bottom_stack_dnn1_shape,
            activation="relu",
            name="bs1",
            use_bias=True,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            bias_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(self._l1_regularizer),
        )
        dnn2 = tf.layers.dense(
            dnn1,
            self._bottom_stack_dnn2_shape,
            activation="relu",
            name="bs2",
            use_bias=True,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            bias_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(self._l1_regularizer),
        )
        dnn3 = tf.layers.dense(
            dnn2,
            self._bottom_stack_dnn3_shape,
            activation="relu",
            name="bs3",
            use_bias=True,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            bias_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(self._l1_regularizer),
        )
        return dnn3

    def _top_stack(self, input_tensor: tf.Tensor, seed: int) -> tf.Tensor:
        dnn1 = tf.layers.dense(
            input_tensor,
            self._top_stack_dnn1_shape,
            activation="relu",
            name="ts1",
            use_bias=True,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            bias_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(self._l1_regularizer),
        )
        dnn2 = tf.layers.dense(
            dnn1,
            self._top_stack_dnn2_shape,
            activation="relu",
            name="ts2",
            use_bias=True,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            bias_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(self._l1_regularizer),
        )
        dnn3 = tf.layers.dense(
            dnn2,
            self._top_stack_dnn3_shape,
            activation="relu",
            name="ts3",
            use_bias=True,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            bias_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(self._l1_regularizer),
        )
        dnn4 = tf.layers.dense(
            dnn3,
            self._top_stack_dnn4_shape,
            activation="relu",
            name="ts4",
            use_bias=True,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            bias_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(self._l1_regularizer),
        )
        dnn5 = tf.layers.dense(
            dnn4,
            self._top_stack_dnn5_shape,
            activation=None,
            name="ts5",
            use_bias=True,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            bias_initializer=tf.compat.v1.variance_scaling_initializer(
                mode="fan_avg", distribution="normal", seed=seed
            ),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(self._l1_regularizer),
        )
        return dnn5
