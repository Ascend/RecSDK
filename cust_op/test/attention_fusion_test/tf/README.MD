`attention_fusion`目录包含了一个通过tensorflow调用AscendC自定义算子的开发示例。

## 适用场景说明
> 需要满足以下件的场景，适合使用本示例：
- 已经在/mxrec/cust_op/下安装完attention_fusion_grad和attention_fusion，希望在Tensorflow中调用该算子, 安装方法见mxrec/cust_op/ascendc_op/ai_core_op/attention_fusion_grad/v220/README.md。
- 本用例使用tensorflow 2.6.5为例。

## 目录结构说明
```
├── README.md # 说明文档
├── build_ops.sh # 编译tensorflow原型算子脚本
├── CMakeLists.txt # cmake编译文件
├── test # attention调用python侧实例
└── tf_ops # tensorflow原型注册
```
需要您阅读tf_custom_op和tf_parser目录下的README.md文件，以了解更多示例细节。本文档仅提供了示例的整体说明。

## 使用示例
1. 首先，安装AttentionFusion和AttentionFusionGrad的ascendc算子
2. 然后，编译安装tensorflow原型算子
3. 在test目录下执行python3 test.py

完成上述两步后，您可以在tf脚本中通过python API调用CANN中的Attention算子了，如同原生的tf算子一般。
> 注意，您仍然需要完成Tensorflow到NPU上的迁移动作。

### Tensorflow2.x使用示例（[session模式NPU迁移](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/moddevg/tfmigr2/tfmigr2_000001.html)）
```python
import os
import logging

import numpy as np
from tensorflow.python.framework import ops
import tensorflow as tf
import npu_device
from npu_device.compat.v1.npu_init import *

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
'''
后续按照tensorflow的方式进行session run和算子的调用
'''
```
