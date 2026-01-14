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
    for _ in range(50):
        noise = tf.random.normal(shape=[1], mean=0.0, stddev=epsilon, dtype=input0.dtype)
        input0_noisy = input0 + noise
        matmul_res = tf.matmul(tf.tile(input0_noisy, perm), input1)
        npu_matmul_ret.append(tf.expand_dims(matmul_res, 0))
    return tf.reduce_mean(tf.concat(npu_matmul_ret, axis=0), axis=0)

tf.compat.v1.disable_eager_execution()

tf_dtype = tf.float32
np_dtype = np.float32

tile_input_shape = [1, 8, 256, 128]
bmm_input_shape = [100, 8, 128, 64]

x = tf.compat.v1.placeholder(tf_dtype, shape=tile_input_shape)
tile_perm = tf.constant([100, 1, 1, 1])
y = tf.compat.v1.placeholder(tf_dtype, shape=bmm_input_shape)

ret = tf.matmul(tf.tile(x, tile_perm), y)

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
logging.info(f"Total time of 50 inference runs: {total_time_ms:.3f}ms")

max_rel_precision_error = np.max(np.abs(result_npu - result_cpu) / result_cpu)
logging.info(f"maximum relative precision error vs. the CPU is: {max_rel_precision_error:.6f}")
