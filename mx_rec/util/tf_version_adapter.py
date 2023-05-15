# coding: UTF-8
import tensorflow as tf

if tf.__version__.startswith("1"):
    from npu_bridge.hccl import hccl_ops 
else:
    from npu_device.compat.v1.hccl import hccl_ops

if tf.__version__.startswith("1"):
    from npu_bridge.estimator import npu_ops 
else:
    from npu_device.compat.v1.estimator import npu_ops 