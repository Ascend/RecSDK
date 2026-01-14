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


def run_100_matmul(input0, input1, perm):
    npu_matmul_ret = []
    for _ in range(20):
        noise = tf.random.normal(shape=[1], mean=0.0, stddev=epsilon, dtype=input0.dtype)
        input1_noisy = input1 + noise
        input1_tile_ret = tf.tile(input1_noisy, perm)
        npu_matmul_input0 = tf.transpose(tf.reshape(input0, [100, 256, 8, 128]), perm=[0, 2, 1, 3])
        npu_matmul_input1 = tf.transpose(tf.reshape(input1_tile_ret, [100, 64, 8, 128]), perm=[0, 2, 1, 3])
        matmul_res = tf.matmul(npu_matmul_input0, npu_matmul_input1, transpose_b=True)
        npu_matmul_ret.append(tf.expand_dims(matmul_res, 0))
    return tf.reduce_mean(tf.concat(npu_matmul_ret, axis=0), axis=0)

tf.compat.v1.disable_eager_execution()

tf_dtype = tf.float32
np_dtype = np.float32

tile_input_shape = [100, 256, 1024]
bmm_input_shape = [1, 64, 1024]

x = tf.compat.v1.placeholder(tf_dtype, shape=tile_input_shape)
tile_perm = tf.constant([100, 1, 1])
y = tf.compat.v1.placeholder(tf_dtype, shape=bmm_input_shape)

y_tile_ret = tf.tile(y, tile_perm)
matmul_input0 = tf.transpose(tf.reshape(x, [100, 256, 8, 128]), perm=[0, 2, 1, 3])
matmul_input1 = tf.transpose(tf.reshape(y_tile_ret, [100, 64, 8, 128]), perm=[0, 2, 1, 3])
ret = tf.matmul(matmul_input0, matmul_input1, transpose_b=True)

inputs_x = np.random.rand(*tile_input_shape).astype(np_dtype)

inputs_y = np.random.rand(*bmm_input_shape).astype(np_dtype)

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

epsilon = tf.constant(1e-10, dtype=x.dtype)
npu_ret = run_100_matmul(x, y, tile_perm)

with tf.compat.v1.Session(config=session_config) as sess:
    #warmup
    sess.run(npu_ret, feed_dict={x: inputs_x, y: inputs_y})

    start_time = time.time()
    result_npu = sess.run(npu_ret, feed_dict={x: inputs_x, y: inputs_y})
    end_time = time.time()

total_time_ms = (end_time - start_time) * 1000
logging.info(f"Total time of 20 inference runs: {total_time_ms:.3f}ms")

max_rel_precision_error = np.max(np.abs(result_npu - result_cpu) / result_cpu)
logging.info(f"maximum relative precision error vs. the CPU is: {max_rel_precision_error:.6f}")
