#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""纯CPU环境下执行用例的统一NPU mock。

pytest会在导入任何测试模块之前先导入本文件，因此这里完成的mock
对目录下所有用例生效；spawn出的子进程仍需在脚本中手动mock。

"""

import os
import sys
from unittest.mock import MagicMock

# 必须在import torch之前设置，阻止torch自动加载真实torch_npu；
# 该环境变量会被mp.spawn的子进程继承，子进程import torch时同样生效
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

# 拦截后续所有显式的 import torch_npu（实例而非类，保证属性自动生成）
sys.modules.setdefault("torch_npu", MagicMock())

# 纯NPU算子包（依赖libtorch_npu.so），CPU环境mock掉；
# 用例路径走fbgemm_gpu的CPU实现，不依赖其NPU dispatch
sys.modules.setdefault("fbgemm_ascend", MagicMock())
