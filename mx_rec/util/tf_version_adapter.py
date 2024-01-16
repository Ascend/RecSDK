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
