#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import tensorflow as tf

if tf.__version__.startswith("1"):
    from npu_bridge.hccl import hccl_ops
else:
    from npu_device.compat.v1.hccl import hccl_ops

if tf.__version__.startswith("1"):
    from npu_bridge.estimator import npu_ops
else:
    from npu_device.compat.v1.estimator import npu_ops

if tf.__version__.startswith("1"):
    from npu_bridge.estimator.npu.npu_hook import NPUCheckpointSaverHook
else:
    from npu_device.compat.v1.estimator.npu.npu_hook import NPUCheckpointSaverHook
