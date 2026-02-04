#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import pytest
import torch
from dynamic_emb_extensions import (
    InitializerArgs,
    DynamicEmbTable,
    DynamicEmbDataType,
    EvictStrategy,
    SafeCheckMode,
    OptimizerType,
    find_pointers,
    load_from_pointer,
)


def get_dim_pointers_optimized(x_2d):
    """利用内存布局特性高效计算首地址"""
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    
    n, m = x_2d.size()
    element_size = x_2d.element_size()  # 每个元素的大小（字节）
    
    # 计算基地址
    base_ptr = x_2d.data_ptr()
    
    # 批量计算所有行的首地址
    stride = m * element_size  # 每行的字节跨度
    pointers = [base_ptr + i * stride for i in range(n)]
    
    return torch.tensor(pointers, dtype=torch.int64, device=x_2d.device)


@pytest.mark.parametrize("optimizer_type", [
    OptimizerType.Null,
    OptimizerType.SGD,
    OptimizerType.Adam,
    OptimizerType.AdaGrad,
    OptimizerType.RowWiseAdaGrad
]) 
@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("N", [1, 10, 100, 1000, 10000, 100000])
@pytest.mark.parametrize("vocabulary_size", [1024, 10240, 102400, 1024000, 10240000])
@pytest.mark.parametrize("dim", [8, 64, 128, 256, 512])
def test_find_pointers(optimizer_type, device, N, vocabulary_size, dim):
    torch.npu.set_device(device)
    table = DynamicEmbTable(
        DynamicEmbDataType.Int64,
        DynamicEmbDataType.Float32,
        EvictStrategy.kLru,
        dim,
        1024,
        vocabulary_size,
        1 * 1024 * 1024 * 1024,
        128,
        0.5,
        128,
        1024,
        device,
        False,
        False,
        0,
        1,
        InitializerArgs(),
        SafeCheckMode.IGNORE,
        optimizer_type
    )
    # 验证表创建成功
    assert table is not None
    keys = torch.randint(0, vocabulary_size, (N,), dtype=torch.int64, device=f'npu:{device}')
    founds = torch.empty(N, dtype=torch.bool, device=f'npu:{device}')
    pointers = torch.empty(N, dtype=torch.long, device=f'npu:{device}')
    find_pointers(table, N, keys, pointers, founds)
    torch.npu.empty_cache()


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("N", [1, 10, 100, 1000, 10000, 100000, 1000000, 10000001])
@pytest.mark.parametrize("dim", [8, 64, 128, 256, 512, 513, 1024, 1025])
def test_load_from_pointer(device, N, dim):
    torch.npu.set_device(device)
    inputs = torch.randn(N, dim, dtype=torch.float32, device=f'npu:{device}')
    pointers = get_dim_pointers_optimized(inputs)
    dst = torch.empty_like(inputs)
    
    result = load_from_pointer(pointers, dst)
    expected = inputs
    
    diff = torch.abs(result.cpu() - expected.cpu())
    max_diff = torch.max(diff).item()
    assert max_diff == 0
    torch.npu.empty_cache()