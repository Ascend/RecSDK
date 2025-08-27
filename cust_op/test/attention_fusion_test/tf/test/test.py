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

import os
import logging

import numpy as np
from tensorflow.python.framework import ops
import tensorflow as tf
import npu_device
from npu_device.compat.v1.npu_init import *

logging.getLogger().setLevel(logging.INFO)

os.environ["DEVICE_ID"] = str(0)
os.environ["ASCEND_DEVICE_ID"] = str(0)
os.environ["JOB_ID"] = "10086"

tf.compat.v1.disable_eager_execution()
tfOpLib = tf.load_op_library("/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/librecsdk_tf_npu_ops.so")

npu_device.compat.enable_v1()
npu_init = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()
config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF


def attention_fusion(q, k, v, mask=None):
    mask_on = 1
    if mask is None:
        mask = tf.zeros(())
        mask_on = 0
    attn_out_result, softmax_out_result = tfOpLib.attention_fusion(query=q, 
                                                                   key=k, value=v, atten_mask=mask, mask_on=mask_on)
    return attn_out_result, softmax_out_result


@ops.RegisterGradient("AttentionFusion")
def _npu_fusion_attention_grad(op, *grad):
    q = op.inputs[0]
    k = op.inputs[1]
    v = op.inputs[2]
    mask = op.inputs[3]
    
    softmax_output = op.outputs[1]
    dout = grad[0]
    d_q, d_k, d_v = tfOpLib.attention_fusion_grad(dout=dout, softmax_out=softmax_output, query=q, key=k, value=v)
    return (d_q, d_k, d_v, tf.zeros(tf.shape(mask)))


def param_attn_layer(q, k, v, m=None):
    # q k v 
    with tf.name_scope("param_attn_layer"):
        with tf.name_scope("qk_matmul"):
            qk = tf.matmul(q, k, transpose_b=True)
            
        with tf.name_scope("div"):
            sqrt_attndim = tf.sqrt(tf.cast(tf.shape(k)[2], tf.float32))
            if sqrt_attndim == 0 :
                qk_div = 0
            else:
                qk_div = qk / sqrt_attndim
            if m is not None:
                m = qk_div + (1 - m) * 10000
            else:
                m = qk_div
        with tf.name_scope("softmax_out"):
            softmax_output = tf.nn.softmax(m, axis=-1)

        with tf.name_scope("out_matmul"):    
            out = tf.matmul(softmax_output, v)

    return out, softmax_output


def generate_data_qk(batch_size, query_dim1, query_dim2, key_dim1, value_dim2):
    q = np.random.randn(batch_size, query_dim1, query_dim2).astype(np.float32)
    k = np.random.randn(batch_size, key_dim1, query_dim2).astype(np.float32)
    return (q, k)


def generate_data_vm(batch_size, query_dim1, query_dim2, key_dim1, value_dim2):
    v = np.random.randn(batch_size, key_dim1, value_dim2).astype(np.float32)
    m = np.random.randint(0, 2, size=(batch_size, query_dim1, key_dim1)).astype(np.float32)
    return (v, m)


query_ph = tf.placeholder(tf.float32, shape=[None, None, None], name="query")
key_ph = tf.placeholder(tf.float32, shape=[None, None, None], name="key")
value_ph = tf.placeholder(tf.float32, shape=[None, None, None], name="value")
mask_ph = tf.placeholder(tf.float32, shape=[None, None, None], name="mask")

# gloden
atten_out_gloden, softmax_out_gloden = param_attn_layer(query_ph, key_ph, value_ph, mask_ph)
loss_golden = tf.reduce_mean(atten_out_gloden, keep_dims=False)
grads_and_vars_golden = tf.gradients(loss_golden, [query_ph, key_ph, value_ph])

# fusion
atten_out, softmax_out = attention_fusion(q=query_ph, k=key_ph, v=value_ph, mask=mask_ph)
loss = tf.reduce_mean(atten_out, keep_dims=False)
grads_and_vars = tf.gradients(loss, [query_ph, key_ph, value_ph])

# test case
test_case = [(1024, 144, 64, 1000, 80)]

with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for dim0, dim1, dim2, dim3, dim4 in test_case:
        logging.info("===================test case %d, %d, %d, %d, %d, ===================",
                    dim0, dim1, dim2, dim3, dim4)
        query_np, key_np = generate_data_qk(dim0, dim1, dim2, dim3, dim4)
        value_np, mask_np = generate_data_vm(dim0, dim1, dim2, dim3, dim4)

        result_gloden = sess.run([loss_golden, grads_and_vars_golden, softmax_out_gloden],
                                    feed_dict={query_ph: query_np, key_ph:key_np, value_ph:value_np, mask_ph:mask_np})
        result = sess.run([loss, grads_and_vars, softmax_out],
                            feed_dict={query_ph: query_np, key_ph:key_np, value_ph:value_np, mask_ph:mask_np})
        
        logging.info(((result[0] - result[0]) < 1e-4).all())
        logging.info(((result[1][0] - result_gloden[1][0]) < 1e-4).all())
        logging.info(((result[1][1] - result_gloden[1][1]) < 1e-4).all())
        logging.info(((result[1][2] - result_gloden[1][2]) < 1e-4).all())
        logging.info("============ attention fusion end =============")
    





