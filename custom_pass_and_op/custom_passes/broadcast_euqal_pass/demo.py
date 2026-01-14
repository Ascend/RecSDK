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

tf_dtype = tf.int64
np_dtype = np.int64

input0_shape = [100, 1, 1, 20]
input1_shape = [1, 1000, 20, 1]

x = tf.compat.v1.placeholder(tf_dtype, shape=input0_shape)
y = tf.compat.v1.placeholder(tf_dtype, shape=input1_shape)

equal_ret = tf.equal(x, y)
ret = tf.cast(tf.reduce_any(equal_ret, axis=-1, keepdims=False), dtype=tf_dtype)

inputs_x = np.random.rand(*input0_shape).astype(np_dtype)
inputs_y = np.random.rand(*input1_shape).astype(np_dtype)

with tf.compat.v1.Session() as sess:
    result_cpu = sess.run(ret, feed_dict={x: inputs_x, y: inputs_y})

import npu_device
from npu_device.compat.v1.npu_init import RewriterConfig

tf.config.optimizer.set_jit(False)
session_config = tf.compat.v1.ConfigProto()
optimizer = session_config.graph_options.rewrite_options.custom_optimizers.add()
optimizer.name = "NpuOptimizer"
optimizer.parameter_map["graph_max_parallel_model_num"].i = 1
session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

with tf.compat.v1.Session(config=session_config) as sess:
    #warmup
    sess.run(ret, feed_dict={x: inputs_x, y: inputs_y})

    start_time = time.time()
    result_npu = sess.run(ret, feed_dict={x: inputs_x, y: inputs_y})
    end_time = time.time()

total_time_ms = (end_time - start_time) * 1000
logging.info(f"Total time of inference runs: {total_time_ms:.3f}ms")

eps = 1e-10
precision_diff = np.abs(result_npu - result_cpu) / np.abs(result_cpu + eps)
max_rel_precision_error = np.max(precision_diff)
logging.info(f"maximum relative precision error vs. the CPU is: {max_rel_precision_error:.6f}")
