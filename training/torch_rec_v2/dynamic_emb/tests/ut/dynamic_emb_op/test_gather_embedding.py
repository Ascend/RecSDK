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

import logging
import time
import pytest
import torch
import dynamic_emb_extensions
logging.basicConfig(level=logging.INFO)



def get_result(inputs, indices):
    return torch.gather(inputs, 0, indices.to(torch.int64).unsqueeze(1).expand(-1, inputs.size(1)))


def get_ops_result(inputs, indices):
    return dynamic_emb_extensions.gather_embedding(inputs, indices)


@pytest.mark.parametrize("idx_type", [torch.int64, torch.uint64])
@pytest.mark.parametrize("N", [1, 10, 100, 1000, 10000])
@pytest.mark.parametrize("M", [5, 50, 500, 5000, 50000])
@pytest.mark.parametrize("dim", [8, 64, 128])
@pytest.mark.parametrize('val_type', [torch.float32, torch.float16, torch.bfloat16])
def test_gather_embedding(idx_type, N, M, dim, val_type):
    torch.npu.set_device(0)
    inputs = torch.randn(N, dim, dtype=val_type, device=f'npu:0')
    indices = torch.randint(0, N, (M, ), device=f'npu:0').to(idx_type)
    result = get_ops_result(inputs, indices)
    expected = get_result(inputs, indices)
    diff = torch.abs(result.cpu() - expected.cpu())
    max_diff = torch.max(diff).item()
    assert max_diff == 0

    # 性能对比测试
    # 预热
    for _ in range(10):
        _ = get_ops_result(inputs, indices)
        _ = get_result(inputs, indices)

    # 测试自定义op性能
    start_time = time.time()
    for _ in range(100):
        _ = get_ops_result(inputs, indices)
    custom_time = time.time() - start_time

    # 测试基准方法性能
    start_time = time.time()
    for _ in range(100):
        _ = get_result(inputs, indices)
    base_time = time.time() - start_time

    # 计算加速比
    speedup = base_time / custom_time if custom_time > 0 else float('inf')
    logging.info(f"\nN={N}, M={M}, dim={dim}, dtype={val_type}")
    logging.info(f"自定义op时间: {custom_time:.6f}s")
    logging.info(f"基准方法时间: {base_time:.6f}s")
    logging.info(f"加速比: {speedup:.2f}x")
    torch.npu.empty_cache()