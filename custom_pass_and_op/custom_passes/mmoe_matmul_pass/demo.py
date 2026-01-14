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

data_on_left = False
data_shape = [800, 5120]
const_shape = [5120, 512]
transpose_data = False
transpose_const = True

if transpose_data == data_on_left:
    data_shape = data_shape[::-1]

if transpose_const == data_on_left:
    const_shape = const_shape[::-1]

if data_on_left:
    transpose_a = transpose_data
    transpose_b = transpose_const
    combined_dim = 1 if not transpose_b else 0
else:
    transpose_a = transpose_const
    transpose_b = transpose_data
    combined_dim = 1 if transpose_a else 0

x = tf.compat.v1.placeholder(tf_dtype, shape=data_shape)

inputs_y1 = np.random.rand(*const_shape).astype(np_dtype)
inputs_y2 = np.random.rand(*const_shape).astype(np_dtype)
inputs_y3 = np.random.rand(*const_shape).astype(np_dtype)
inputs_y4 = np.random.rand(*const_shape).astype(np_dtype)

y1 = tf.constant(inputs_y1, dtype=tf_dtype, shape=const_shape)
y2 = tf.constant(inputs_y2, dtype=tf_dtype, shape=const_shape)
y3 = tf.constant(inputs_y3, dtype=tf_dtype, shape=const_shape)
y4 = tf.constant(inputs_y4, dtype=tf_dtype, shape=const_shape)

if data_on_left:
    matmul_ret1 = tf.matmul(x, y1, transpose_a=transpose_data, transpose_b=transpose_const)
    matmul_ret2 = tf.matmul(x, y2, transpose_a=transpose_data, transpose_b=transpose_const)
    matmul_ret3 = tf.matmul(x, y3, transpose_a=transpose_data, transpose_b=transpose_const)
    matmul_ret4 = tf.matmul(x, y4, transpose_a=transpose_data, transpose_b=transpose_const)
else:
    matmul_ret1 = tf.matmul(y1, x, transpose_a=transpose_const, transpose_b=transpose_data)
    matmul_ret2 = tf.matmul(y2, x, transpose_a=transpose_const, transpose_b=transpose_data)
    matmul_ret3 = tf.matmul(y3, x, transpose_a=transpose_const, transpose_b=transpose_data)
    matmul_ret4 = tf.matmul(y4, x, transpose_a=transpose_const, transpose_b=transpose_data)

if data_on_left:
    ret_concat_dim = 1
else:
    ret_concat_dim = 0

ret = tf.concat([matmul_ret1, matmul_ret2, matmul_ret3, matmul_ret4], axis=ret_concat_dim)
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
