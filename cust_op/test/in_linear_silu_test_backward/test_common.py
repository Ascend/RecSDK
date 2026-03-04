#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import sysconfig
import logging
import random
import torch_npu
import torch

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def set_seed(seed=0):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def allclose(tensor: torch.Tensor, other: torch.Tensor, atol: float, ratio: float, name: str) -> bool:
    assert tensor.shape == other.shape
    # 使用 torch.isclose 进行逐元素比较
    
    close = torch.isclose(tensor.to(torch.float32), other.to(torch.float32), rtol=atol, atol=atol)
    diff_count = tensor.numel() - torch.sum(close)
    isPass = (diff_count / tensor.numel()) <= ratio
    # 如果有不满足精度的元素，打印它们的下标和值
    if not isPass:
        # 获取不满足精度的元素下标
        indices = torch.where(~close)
        # 转换为 CPU 上的 numpy 数组以便打印
        indices = [idx.cpu().numpy() for idx in indices]
        
        logger.info(f"\n========================================")
        logger.info(f"Found {diff_count} elements that do not meet the precision requirements (atol={atol}):")
        logger.info(f"{name} Tensor shape: {tensor.shape}")
        
        # 打印前10个不满足精度的元素（避免输出过多）
        max_print = min(10, diff_count)
        logger.info(f"\nFirst {max_print} mismatched indices:")
        for i in range(max_print):
            idx = tuple(idx[i] for idx in indices)
            actual = tensor[idx].item()
            expected = other[idx].item()
            diff = abs(actual - expected)
            logger.info(f"  Index {idx}: actual={actual:.5f}, expected={expected:.5f}, diff={diff:.5f}")
        logger.info(f"========================================\n")
    
    return isPass