# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import time

import numpy as np
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

tf.compat.v1.disable_eager_execution()

tf_dtype = tf.float32
np_dtype = np.float32

data_shape = [800, 5120]
expert_hidden1_shape = [5120, 512]
expert_hidden2_shape = [512, 128]

x = tf.compat.v1.placeholder(tf_dtype, shape=data_shape)

expert_hidden1_weight_np_1 = np.random.rand(*expert_hidden1_shape).astype(np_dtype)
expert_hidden1_weight_np_2 = np.random.rand(*expert_hidden1_shape).astype(np_dtype)
expert_hidden1_weight_np_3 = np.random.rand(*expert_hidden1_shape).astype(np_dtype)
expert_hidden1_weight_np_4 = np.random.rand(*expert_hidden1_shape).astype(np_dtype)

expert_hidden1_weight_tf_1 = tf.constant(expert_hidden1_weight_np_1, dtype=tf_dtype, shape=expert_hidden1_shape)
expert_hidden1_weight_tf_2 = tf.constant(expert_hidden1_weight_np_2, dtype=tf_dtype, shape=expert_hidden1_shape)
expert_hidden1_weight_tf_3 = tf.constant(expert_hidden1_weight_np_3, dtype=tf_dtype, shape=expert_hidden1_shape)
expert_hidden1_weight_tf_4 = tf.constant(expert_hidden1_weight_np_4, dtype=tf_dtype, shape=expert_hidden1_shape)

expert_hidden1_output_1 = tf.nn.relu(tf.matmul(x, expert_hidden1_weight_tf_1))
expert_hidden1_output_2 = tf.nn.relu(tf.matmul(x, expert_hidden1_weight_tf_2))
expert_hidden1_output_3 = tf.nn.relu(tf.matmul(x, expert_hidden1_weight_tf_3))
expert_hidden1_output_4 = tf.nn.relu(tf.matmul(x, expert_hidden1_weight_tf_4))

expert_hidden2_weight_np_1 = np.random.rand(*expert_hidden2_shape).astype(np_dtype)
expert_hidden2_weight_np_2 = np.random.rand(*expert_hidden2_shape).astype(np_dtype)
expert_hidden2_weight_np_3 = np.random.rand(*expert_hidden2_shape).astype(np_dtype)
expert_hidden2_weight_np_4 = np.random.rand(*expert_hidden2_shape).astype(np_dtype)

expert_hidden2_weight_tf_1 = tf.constant(expert_hidden2_weight_np_1, dtype=tf_dtype, shape=expert_hidden2_shape)
expert_hidden2_weight_tf_2 = tf.constant(expert_hidden2_weight_np_2, dtype=tf_dtype, shape=expert_hidden2_shape)
expert_hidden2_weight_tf_3 = tf.constant(expert_hidden2_weight_np_3, dtype=tf_dtype, shape=expert_hidden2_shape)
expert_hidden2_weight_tf_4 = tf.constant(expert_hidden2_weight_np_4, dtype=tf_dtype, shape=expert_hidden2_shape)

expert_hidden2_output_1 = tf.nn.relu(tf.matmul(expert_hidden1_output_1, expert_hidden2_weight_tf_1))
expert_hidden2_output_2 = tf.nn.relu(tf.matmul(expert_hidden1_output_2, expert_hidden2_weight_tf_2))
expert_hidden2_output_3 = tf.nn.relu(tf.matmul(expert_hidden1_output_3, expert_hidden2_weight_tf_3))
expert_hidden2_output_4 = tf.nn.relu(tf.matmul(expert_hidden1_output_4, expert_hidden2_weight_tf_4))

ret = tf.concat([
    expert_hidden2_output_1, 
    expert_hidden2_output_2, 
    expert_hidden2_output_3, 
    expert_hidden2_output_4], axis=0)
inputs_x = np.random.rand(*data_shape)
inputs_x = inputs_x.astype(np_dtype)

with tf.compat.v1.Session() as sess:
    result_cpu = sess.run(ret, feed_dict={x: inputs_x})

import npu_device
from npu_device.compat.v1.npu_init import RewriterConfig

session_config = tf.compat.v1.ConfigProto()
custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["graph_max_parallel_model_num"].i = 1
session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

with tf.compat.v1.Session(config=session_config) as sess:
    # warmup
    result_npu = sess.run(ret, feed_dict={x: inputs_x}) 

    start_time = time.time()
    for i in range(1000):
        sess.run(ret, feed_dict={x: inputs_x})
    end_time = time.time()
  
total_time_ms = (end_time - start_time) * 1000
logging.info(f"Total time of 1000 inference runs: {total_time_ms:.3f}")

max_rel_precision_error = np.max(np.abs(result_npu - result_cpu) / result_cpu)
logging.info(f"maximum relative precision error vs. the CPU is: {max_rel_precision_error:.6f}") 

