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
import tensorflow as tf

import npu_device
from npu_device.compat.v1.npu_init import RewriterConfig

session_config = tf.compat.v1.ConfigProto()
optimizer = session_config.graph_options.rewrite_options.custom_optimizers.add()
optimizer.name = "NpuOptimizer"
optimizer.parameter_map["graph_max_parallel_model_num"].i = 1
session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF


def demo_load_custom_op():
    try:
        custom_op_lib = tf.load_op_library("./tf_ops/user_item_flash_attention_op.so")
    except Exception as e:
        logging.error("load tf so failed: ", e)
        return

    try:
        user_item_flash_attention = custom_op_lib.user_item_flash_attention
    except AttributeError as e:
        logging.error("get custom op failed: ", e)
        return

    dtype = tf.float32
    q_batch = 1
    kv_batch_size = 100
    q_seq_len = 300
    kv_seq_len = 1
    head_dim = 8
    hidden_dim = 64

    # 构造输入张量 1-4（固定个数输入）
    query = tf.ones(shape=[q_batch, q_seq_len, head_dim, hidden_dim], dtype=dtype, name="query")
    key = tf.ones(shape=[kv_batch_size, kv_seq_len, head_dim, hidden_dim], dtype=dtype, name="key_user")
    value = tf.ones(shape=[kv_batch_size, kv_seq_len, head_dim, hidden_dim], dtype=dtype, name="value_user")
    mask_len = tf.constant([q_seq_len], dtype=tf.int32, name="mask_len")

    try:
        attention_out = user_item_flash_attention(
            query=query,
            key_user=key,
            value_user=value,
            mask_len=mask_len,
            key_item=[],
            value_item=[]
        )
        
        with tf.compat.v1.Session(config=session_config) as sess:
            output_value = sess.run(attention_out)
  
        logging.info(output_value)
                
    except Exception as e:
        logging.error("execute failed：", e)

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    
    demo_load_custom_op()
