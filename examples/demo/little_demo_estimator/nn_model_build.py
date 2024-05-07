#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import tensorflow as tf
from tensorflow import Tensor
from mx_rec.util.tf_version_adapter import npu_ops
from mx_rec.core.embedding import create_table, sparse_lookup
from mx_rec.constants.constants import ASCEND_TIMESTAMP

from utils import FeatureSpecIns


class LittleModel:
    def __init__(self, params, cfg, mode, features, create_fs_params=None, access_and_evict_config_dict=None):
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

        self.is_train = mode == tf.estimator.ModeKeys.TRAIN
        self.cfg = cfg
        self.params = params
        self.features = features
        self.create_fs_params = create_fs_params
        self.access_and_evict_config_dict = access_and_evict_config_dict

    @staticmethod
    def activate(act_func, input_x):
        if act_func == 'tanh':
            return tf.tanh(input_x)
        elif act_func == 'relu':
            return tf.nn.relu(input_x)
        else:
            return tf.sigmoid(input_x)

    def inference(self, label_0, label_1):
        with tf.compat.v1.variable_scope("mlp", reuse=tf.compat.v1.AUTO_REUSE):
            embedding_list = self._get_embedding_list()
            embedding = tf.concat(embedding_list, axis=1)
            self.emb_dim = embedding.shape.as_list()[-1]
            self.all_layer_dims = [self.emb_dim] + self.layer_dims + [1]

            with tf.compat.v1.variable_scope("mlp", reuse=tf.compat.v1.AUTO_REUSE):
                for i in range(len(self.all_layer_dims) - 2):
                    self.h_w.append(tf.compat.v1.get_variable('h%d_w' % (i + 1), shape=self.all_layer_dims[i: i + 2],
                                                              initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                                              dtype=tf.float32,
                                                              collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                           "deep", "mlp_wts"]))
                    self.h_b.append(
                        tf.compat.v1.get_variable('h%d_b' % (i + 1), shape=[self.all_layer_dims[i + 1]],
                                                  initializer=tf.compat.v1.zeros_initializer,
                                                  dtype=tf.float32,
                                                  collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "deep",
                                                               "mlp_bias"]))
                i += 1
                self.h_w_head_0 = tf.compat.v1.get_variable('h_w_head_0', shape=self.all_layer_dims[i: i + 2],
                                                            initializer=tf.compat.v1.random_uniform_initializer(-0.01,
                                                                                                                0.01),
                                                            dtype=tf.float32,
                                                            collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                         "deep", "mlp_wts"])
                self.h_b_head_0 = tf.compat.v1.get_variable('h_b_head_0', shape=[self.all_layer_dims[i + 1]],
                                                            initializer=tf.compat.v1.zeros_initializer,
                                                            dtype=tf.float32,
                                                            collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                         "deep", "mlp_bias"])
                self.h_w_head_1 = tf.compat.v1.get_variable('h_w_head_1', shape=self.all_layer_dims[i: i + 2],
                                                            initializer=tf.compat.v1.random_uniform_initializer(-0.01,
                                                                                                                0.01),
                                                            dtype=tf.float32,
                                                            collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                         "deep", "mlp_wts"])
                self.h_b_head_1 = tf.compat.v1.get_variable('h_b_head_1', shape=[self.all_layer_dims[i + 1]],
                                                            initializer=tf.compat.v1.zeros_initializer,
                                                            dtype=tf.float32,
                                                            collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                         "deep", "mlp_bias"])

            logit_list = self._forward(embedding, self.act_func, self.keep_prob, training=self.is_train)

            for logit, label in zip(logit_list, (label_0, label_1)):
                train_preds = tf.sigmoid(logit)
                basic_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=label)
                self.predict_list.append(train_preds)
                self.loss_list.append(basic_loss)

            loss = tf.reduce_mean(self.loss_list, keepdims=False)
            return loss, self.predict_list

    def _forward(self, embedding, act_func, keep_prob, training):
        hidden_output = tf.reshape(embedding, [-1, self.emb_dim])  # *512
        for i, h_w_var in enumerate(self.h_w):
            if training:
                hidden_output = tf.matmul(npu_ops.dropout(self.activate(act_func, hidden_output),
                                                          keep_prob=keep_prob), h_w_var)
            else:
                hidden_output = tf.matmul(self.activate(act_func, hidden_output), h_w_var)
            hidden_output = hidden_output + self.h_b[i]

        def output_head(hidden_output, h_w, h_b):
            if training:
                hidden_output_branch = tf.matmul(npu_ops.dropout(self.activate(act_func, hidden_output),
                                                                 keep_prob=keep_prob), h_w)
            else:
                hidden_output_branch = tf.matmul(self.activate(act_func, hidden_output), h_w)
            logit = hidden_output_branch + h_b
            logit = tf.reshape(logit, [-1, ])

            return logit

        logit_0 = output_head(hidden_output, self.h_w_head_0, self.h_b_head_0)
        logit_1 = output_head(hidden_output, self.h_w_head_1, self.h_b_head_1)
        logit_list = [logit_0, logit_1]

        return logit_list

    def _get_embedding_list(self):
        user_hashtable = create_table(key_dtype=tf.int64,
                                      dim=tf.TensorShape([self.cfg.user_hashtable_dim]),
                                      name='user_table',
                                      emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                                      device_vocabulary_size=self.cfg.user_vocab_size * 10,
                                      host_vocabulary_size=self.cfg.user_vocab_size * 0)
        item_hashtable = create_table(key_dtype=tf.int64,
                                      dim=tf.TensorShape([self.cfg.item_hashtable_dim]),
                                      name='item_table',
                                      emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                                      device_vocabulary_size=self.cfg.item_vocab_size * 10,
                                      host_vocabulary_size=self.cfg.item_vocab_size * 0)

        if self.params.modify_graph:
            if not self.params.enable_slicer_test:
                input_list = [[self.features["user_ids"], self.features["item_ids"]],
                              [user_hashtable, item_hashtable],
                              [self.cfg.user_send_cnt, self.cfg.item_send_cnt],
                              [True, True]]
            else:
                const_ids = _make_ids_with_const_ops(self.features["user_ids"])
                str_ids = _make_ids_with_str_ops(self.features["item_ids"])
                input_list = [[const_ids, str_ids],
                              [user_hashtable, item_hashtable],
                              [self.cfg.user_send_cnt, self.cfg.item_send_cnt],
                              [True, True]]

            if self.params.use_multi_lookup:
                # add `MULTI_LOOKUP_TIMES` times
                input_list[0].extend([self.features["user_ids"]] * self.params.multi_lookup_times)
                input_list[1].extend([user_hashtable] * self.params.multi_lookup_times)
                input_list[2].extend([self.cfg.user_send_cnt] * self.params.multi_lookup_times)
                input_list[3].extend([False] * self.params.multi_lookup_times)
            if self.params.use_timestamp:
                tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, self.features["timestamp"])
        else:
            if self.is_train:
                feature_spec_list = FeatureSpecIns.get_instance().get_train_feature_spec_list()
            else:
                feature_spec_list = FeatureSpecIns.get_instance().get_eval_feature_spec_list()
            input_list = [feature_spec_list,
                          [user_hashtable, item_hashtable],
                          [self.cfg.user_send_cnt, self.cfg.item_send_cnt],
                          [True, True]]
            if self.params.use_multi_lookup:
                # add `MULTI_LOOKUP_TIMES` times
                input_list[1].extend([user_hashtable] * self.params.multi_lookup_times)
                input_list[2].extend([self.cfg.user_send_cnt] * self.params.multi_lookup_times)
                input_list[3].extend([False] * self.params.multi_lookup_times)

        embedding_list = []
        feature_list, hash_table_list, send_count_list, is_grad_list = input_list
        for feature, hash_table, send_count, is_grad in zip(feature_list, hash_table_list, send_count_list, is_grad_list):
            access_and_evict_config = None
            if isinstance(self.access_and_evict_config_dict, dict):
                access_and_evict_config = self.access_and_evict_config_dict.get(hash_table.table_name)
            embedding = sparse_lookup(hash_table, feature, send_count, dim=None, is_train=self.is_train, is_grad=is_grad,
                                      name=hash_table.table_name + "_lookup", modify_graph=self.params.modify_graph,
                                      access_and_evict_config=access_and_evict_config, batch=self.features)

            reduced_embedding = tf.reduce_sum(embedding, axis=1, keepdims=False)
            embedding_list.append(reduced_embedding)

        return embedding_list


def _make_ids_with_const_ops(input_tensor: Tensor) -> Tensor:
    const_ids = tf.constant(1, shape=input_tensor.shape, dtype=input_tensor.dtype)
    const_ids = tf.compat.v1.add(const_ids, 1)
    const_ids = tf.compat.v1.subtract(const_ids, 1)

    return const_ids


def _make_ids_with_str_ops(input_tensor: Tensor) -> Tensor:
    str_ids = tf.compat.v1.strings.as_string(input_tensor)
    str_ids = tf.compat.v1.strings.to_number(str_ids)

    return str_ids
