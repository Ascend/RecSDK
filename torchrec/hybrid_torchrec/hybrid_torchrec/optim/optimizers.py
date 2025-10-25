#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim import Adagrad, SGD, Adam


class AccumulateOptimizer:
    """梯度累积优化器"""
    def __init__(self, use_accumulate=False, accumulate_step=1):
        # 参数校验
        if not isinstance(accumulate_step, int) or accumulate_step <= 0:
            raise ValueError("accumulate_step must be positive integer")
        if not isinstance(use_accumulate, bool):
            raise TypeError("use_accumulate must be bool")
        # 梯度累积相关属性
        self.use_accumulate = use_accumulate
        self.accumulate_step = accumulate_step


class AccumulateAdagrad(Adagrad, AccumulateOptimizer):
    """支持梯度累积的Adagrad优化器"""
    def __init__(self, params, use_accumulate=False, accumulate_step=1, **kwargs):
        AccumulateOptimizer.__init__(self, use_accumulate=use_accumulate, accumulate_step=accumulate_step)
        Adagrad.__init__(self, params, **kwargs)


class AccumulateSGD(SGD, AccumulateOptimizer):
    """支持梯度累积的SGD优化器"""
    def __init__(self, params, use_accumulate=False, accumulate_step=1, **kwargs):
        AccumulateOptimizer.__init__(self, use_accumulate=use_accumulate, accumulate_step=accumulate_step)
        SGD.__init__(self, params, **kwargs)


class AccumulateAdam(Adam, AccumulateOptimizer):
    """支持梯度累积的Adam优化器"""
    def __init__(self, params, use_accumulate=False, accumulate_step=1, **kwargs):
        AccumulateOptimizer.__init__(self, use_accumulate=use_accumulate, accumulate_step=accumulate_step)
        Adam.__init__(self, params, **kwargs)