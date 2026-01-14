# Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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


def build_target_graph(input_ids, embedding_table):
    mask = tf.greater_equal(input_ids, 0)

    max_ids = tf.maximum(input_ids, 0)

    embeddings = tf.gather(embedding_table, max_ids)

    mask_expanded = tf.expand_dims(mask, axis=1)

    zero_vector = tf.zeros_like(embeddings)

    output = tf.where(mask_expanded, embeddings, zero_vector)

    return output

index_shape = [10000]
table_shape = [1000, 16]

index = tf.compat.v1.placeholder(tf.int64, shape=index_shape)
table_np = np.random.rand(*table_shape).astype(np.float32)
table = tf.constant(table_np, dtype=tf.float32, shape=table_shape)

ret = build_target_graph(index, table)

index_np = np.random.randint(low=-1, high=1000, size=index_shape).astype(np.int64)

with tf.compat.v1.Session() as sess:
    result_cpu = sess.run(ret, feed_dict={index: index_np})

import npu_device
from npu_device.compat.v1.npu_init import RewriterConfig

session_config = tf.compat.v1.ConfigProto()
optimizer = session_config.graph_options.rewrite_options.custom_optimizers.add()
optimizer.name = "NpuOptimizer"
optimizer.parameter_map["graph_max_parallel_model_num"].i = 1
session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

with tf.compat.v1.Session(config=session_config) as sess:
    sess.run(ret, feed_dict={index: index_np})

    start_time = time.time()
    result_npu = sess.run(ret, feed_dict={index: index_np})
    end_time = time.time()

total_time_ms = (end_time - start_time) * 1000
logging.info(f"Total time of 100 inference runs: {total_time_ms:.3f}ms")

eps = 1e-10
precision_diff = np.abs(result_npu - result_cpu) / np.abs(result_cpu + eps)
max_rel_precision_error = np.max(precision_diff)
logging.info(f"maximum relative precision error vs. the CPU is: {max_rel_precision_error:.6f}")