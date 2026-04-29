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
import dynamic_emb_extensions as demb

from dynamic_emb_extensions import (
    DynamicEmbDataType,
    dynamic_emb_sgd_with_pointer,
    dynamic_emb_sgd_with_table,
    dynamic_emb_sgd_fused,
    find_pointers,
    load_from_pointer,
)


def get_dim_pointers_optimized(x_2d):
    """利用内存布局特性高效计算每行首地址"""
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    n, m = x_2d.size()
    elem_size = x_2d.element_size()
    row_stride = m * elem_size
    pointers = [x_2d.data_ptr() + i * row_stride for i in range(n)]
    return torch.tensor(pointers, dtype=torch.int64, device=x_2d.device)


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("batch_size", [1, 1024, 4096, 8192, 10240, 102400])
@pytest.mark.parametrize("embedding_dim", [8, 64, 128, 256, 512, 1024, 31, 1023])
@pytest.mark.parametrize("lr", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("iter_num", [10, 100])
def test_dynamic_emb_sgd_with_pointer(device, batch_size, embedding_dim, lr, iter_num):
    """
    1. 使用 PyTorch 模拟生成真实参数。
    2. 将状态同步给 Custom 算子。
    3. 执行单步更新对比。
    """
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    params_init = torch.randn(batch_size, embedding_dim, device=f'npu:{device}')
    param_torch = torch.nn.Parameter(params_init.clone())
    optimizer_torch = torch.optim.SGD([param_torch], lr=lr)

    for _ in range(iter_num - 1):
        dummy_grad = torch.randn_like(param_torch)
        optimizer_torch.zero_grad()
        param_torch.grad = dummy_grad
        optimizer_torch.step()

    current_params = param_torch.data.clone()
    values = current_params.contiguous()
    val_pointers = get_dim_pointers_optimized(values)

    test_grad = torch.randn_like(current_params)
    dynamic_emb_sgd_with_pointer(
        test_grad,
        val_pointers,
        DynamicEmbDataType.Float32,
        lr,
    )
    torch.npu.synchronize()

    optimizer_torch.zero_grad()
    param_torch.grad = test_grad.clone()
    optimizer_torch.step()
    torch.npu.synchronize()

    custom_param_result = values
    torch_param_result = param_torch.data

    torch.testing.assert_close(custom_param_result, torch_param_result, rtol=1e-5, atol=1e-5)


def build_dynamic_emb_table(dim, device):
    torch.npu.set_device(device)
    return demb.DynamicEmbTable(
        demb.DynamicEmbDataType.Int64,
        demb.DynamicEmbDataType.Float32,
        demb.EvictStrategy.kLru,
        dim,
        1024,
        2048,
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
        demb.InitializerArgs(),
        demb.SafeCheckMode.IGNORE,
        demb.OptimizerType.SGD,
    )


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("num_keys", [1, 8, 31, 64])
@pytest.mark.parametrize("embedding_dim", [8, 64, 128, 256, 31, 1023])
@pytest.mark.parametrize("lr", [0.001, 0.01, 0.1])
def test_dynamic_emb_sgd_with_table(device, num_keys, embedding_dim, lr):
    """
    1. 将一组键值对加载到 DynamicEmbTable。
    2. 调用 dynamic_emb_sgd_with_table 更新表中的向量。
    3. 与 PyTorch SGD 结果对比。
    """
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    n = num_keys
    keys = torch.randperm(1000, device=f'npu:{device}')[:n].to(torch.int64)
    values = torch.randn(n, embedding_dim, device=f'npu:{device}')
    grads = torch.randn(n, embedding_dim, device=f'npu:{device}')
    expected = values - lr * grads

    table = build_dynamic_emb_table(embedding_dim, device)
    table.load(n, keys, values, None, True, False)

    dynamic_emb_sgd_with_table(table, n, keys, grads, lr, DynamicEmbDataType.Float32)
    torch.npu.synchronize()

    pointers = torch.empty(n, dtype=torch.int64, device=f'npu:{device}')
    founds = torch.empty(n, dtype=torch.bool, device=f'npu:{device}')
    find_pointers(table, n, keys, pointers, founds)

    values_out = torch.empty_like(values)
    load_from_pointer(pointers, values_out)

    assert founds.all().item(), "所有 key 都应能在表中找到"
    torch.testing.assert_close(values_out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("batch_size", [1, 1024, 4096, 8192])
@pytest.mark.parametrize("embedding_dim", [8, 64, 128, 256, 31, 1023])
@pytest.mark.parametrize("lr", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("iter_num", [10, 100])
def test_dynamic_emb_sgd_fused(device, batch_size, embedding_dim, lr, iter_num):
    """
    1. 使用 PyTorch SGD 生成当前参数。
    2. 调用 dynamic_emb_sgd_fused 更新 values。
    3. 与 PyTorch 结果对比。
    """
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    params_init = torch.randn(batch_size, embedding_dim, device=f'npu:{device}')
    param_torch = torch.nn.Parameter(params_init.clone())
    optimizer_torch = torch.optim.SGD([param_torch], lr=lr)

    for _ in range(iter_num - 1):
        dummy_grad = torch.randn_like(param_torch)
        optimizer_torch.zero_grad()
        param_torch.grad = dummy_grad
        optimizer_torch.step()

    current_params = param_torch.data.clone()
    values = current_params.contiguous()

    test_grad = torch.randn_like(current_params)
    dynamic_emb_sgd_fused(test_grad, values, lr)
    torch.npu.synchronize()

    optimizer_torch.zero_grad()
    param_torch.grad = test_grad.clone()
    optimizer_torch.step()
    torch.npu.synchronize()

    custom_param_result = values
    torch_param_result = param_torch.data

    torch.testing.assert_close(custom_param_result, torch_param_result, rtol=1e-5, atol=1e-5)
