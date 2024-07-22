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
    def __init__(self, expert_num=8, expert_size=16, tower_size=8, gate_num=2):

        self.expert_num = expert_num
        self.expert_size = expert_size
        self.tower_size = tower_size
        self.gate_num = gate_num

    
    def expert_layer(self, input):
        param_expert = []
        for i in range(0, self.expert_num):
            expert_linear = tf.layers.dense(input, units=self.expert_size, activation=None, name=f'expert_payer_{i}', 
                                            kernel_initializer=tf.constant_initializer(value=0.1), 
                                            bias_initializer=tf.constant_initializer(values = 0.1))
            
            param_expert.append(expert_linear)
        return param_expert
    
    
    def gate_layer(self, input):
        param_gate = []
        for i in range(0, self.gate_num):
            gate_linear = tf.layers.dense(input, units=self.gate_size, activation=None, name=f'gate_payer_{i}', 
                                            kernel_initializer=tf.constant_initializer(value=0.1), 
                                            bias_initializer=tf.constant_initializer(values = 0.1))
            
            param_gate.append(gate_linear)
        return param_gate
    
    
    def tower_layer(self, input, layer_name):
        tower_linear = tf.layers.dense(input, units=self.tower_size, activation=None, name=f'tower_payer_{layer_name}', 
                                            kernel_initializer=tf.constant_initializer(value=0.1), 
                                            bias_initializer=tf.constant_initializer(values = 0.1))
        
        tower_linear_out = tf.layers.dense(tower_linear, units=self.tower_size, activation=None, 
                                            name=f'tower_payer_out_{layer_name}', 
                                            kernel_initializer=tf.constant_initializer(value=0.1), 
                                            bias_initializer=tf.constant_initializer(values=0.1))
        
        return tower_linear_out
        
        

    
    def build_model(self,
                    embedding=None,
                    dense_feature=None,
                    label=None,
                    is_training=True,
                    seed=None):

        with tf.variable_scope("mmoe", reuse=tf.AUTO_REUSE):

            dense_expert = self.expert_layer(dense_feature)
            dense_gate = self.gate_layer(dense_feature)

            all_expert = []
            _slice_num = 0
            for i in range(0, self.expert_num):
                slice_num_end = _slice_num + self.expert_size
                cur_expert = tf.add(dense_expert[i], embedding[:, _slice_num:slice_num_end])
                cur_expert = tf.nn.relu(cur_expert)
                all_expert.append(cur_expert)
                _slice_num = slice_num_end

            expert_concat = tf.concat(all_expert, axis=1)
            expert_concat = tf.reshape(expert_concat, [-1, self.expert_num, self.expert_size])

            output_layers = []
            out_pred = []
            for i in range(0, self.gate_num):
                slice_gate_end = _slice_num + self.expert_num
                cur_gate = tf.add(dense_gate[i], embedding[:, _slice_num:slice_gate_end])
                cur_gate = tf.nn.softmax(cur_gate)

                cur_gate = tf.reshape(cur_gate, [-1, self.expert_num, 1])

                cur_gate_expert = tf.multiply(x=expert_concat, y=cur_gate)
                cur_gate_expert = tf.reduce_sum(cur_gate_expert, axis=1)
                out = self.tower_layer(cur_gate_expert, i)
                output_layers.append(out)
                out_pred.append(tf.nn.softmax(out[:, 1]))
                _slice_num = slice_num_end
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mmoe')

            label_income = label[:, 0:1]
            label_mat = label[:, 1:]

            pred_income_1 = tf.slice(output_layers[0], [0, 1], [-1, 1])
            pred_marital_1 = tf.slice(output_layers[1], [0, 1], [-1, 1])

            cost_income = tf.losses.log_loss(labels=tf.cast(label_income, tf.float32), predictions=pred_income_1,
                                             epsilon=1e-4)
            cost_marital = tf.losses.log_loss(labels=tf.cast(label_mat, tf.float32), predictions=pred_marital_1,
                                              epsilon=1e-4)

            avg_cost_income = tf.reduce_mean(cost_income)
            avg_cost_marital = tf.reduce_mean(cost_marital)

            loss = 0.5 * (avg_cost_income + avg_cost_marital)
            
            return {LOSS_OP_NAME: loss,
                    PRED_OP_NAME: out_pred,
                    LABEL_OP_NAME: label,
                    VAR_LIST: trainable_variables}
