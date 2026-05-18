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

import pytest
import torch

from dynamic_emb_extensions import (
    DynamicEmbDataType,
    DynamicEmbTable,
    EvictStrategy,
    InitializerArgs,
    OptimizerType,
    SafeCheckMode,
    dynamic_emb_adagrad_fused,
    dynamic_emb_adagrad_with_pointer,
    dynamic_emb_adagrad_with_table,
    find_pointers,
    load_from_pointer,
)


class OptimizerParams:
    """优化器参数
    lr: 学习率
    eps: 小常数，防止除零
    """

    def __init__(self, lr, eps):
        self.lr = lr
        self.eps = eps


@pytest.fixture
def optimizer_params(lr, eps):
    return OptimizerParams(lr=lr, eps=eps)


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
@pytest.mark.parametrize("embedding_dim", [8, 64, 128, 256, 512, 1024, 8192, 31, 1023])
@pytest.mark.parametrize("lr", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("iter_num", [10, 100])
@pytest.mark.parametrize(
    ("grad_type", "eps"),
    [
        (torch.float32, 1e-8),
        (torch.float16, 1e-4),
        (torch.bfloat16, 1e-8),
    ],
)
def test_dynamic_emb_adagrad_with_pointer(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type):
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    lr = optimizer_params.lr
    eps = optimizer_params.eps

    params_init = torch.randn(batch_size, embedding_dim, dtype=grad_type, device=f"npu:{device}")
    param_torch = torch.nn.Parameter(params_init.clone())
    optimizer_torch = torch.optim.Adagrad([param_torch], lr=lr, eps=eps)

    for _ in range(iter_num - 1):
        dummy_grad = torch.randn_like(param_torch, dtype=grad_type)
        optimizer_torch.zero_grad()
        param_torch.grad = dummy_grad
        optimizer_torch.step()

    current_params = param_torch.data.clone()
    if len(optimizer_torch.state) == 0:
        sum_init = torch.zeros_like(current_params)
    else:
        state = optimizer_torch.state[param_torch]
        sum_init = state["sum"].to(dtype=grad_type, device=current_params.device).clone()

    values = torch.cat([current_params, sum_init], dim=1).contiguous()
    val_pointers = get_dim_pointers_optimized(values)

    test_grad = torch.randn_like(current_params, dtype=grad_type)
    dtype_to_dynamic_emb = {
        torch.float32: DynamicEmbDataType.Float32,
        torch.float16: DynamicEmbDataType.Float16,
        torch.bfloat16: DynamicEmbDataType.BFloat16,
    }
    val_dynamic_type = dtype_to_dynamic_emb[grad_type]
    dynamic_emb_adagrad_with_pointer(
        test_grad,
        val_pointers,
        val_dynamic_type,
        embedding_dim,
        lr,
        eps,
    )
    torch.npu.synchronize()

    optimizer_torch.zero_grad()
    param_torch.grad = test_grad.clone()
    optimizer_torch.step()
    torch.npu.synchronize()

    custom_param_result = values[:, :embedding_dim]
    torch_param_result = param_torch.data
    tol_map = {
        torch.float32: (1e-5, 1e-5),
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }
    rtol, atol = tol_map[grad_type]
    torch.testing.assert_close(custom_param_result, torch_param_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("batch_size", [1, 1024, 4096, 102400])
@pytest.mark.parametrize("embedding_dim", [64, 128, 8192, 31])
@pytest.mark.parametrize("lr", [0.001, 0.01])
@pytest.mark.parametrize("iter_num", [10, 100])
@pytest.mark.parametrize(
    ("grad_type", "eps"),
    [
        (torch.float32, 1e-8),
        (torch.float16, 1e-4),
        (torch.bfloat16, 1e-8),
    ],
)
def test_dynamic_emb_adagrad_with_table(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type):
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    lr = optimizer_params.lr
    eps = optimizer_params.eps

    params_init = torch.randn(batch_size, embedding_dim, dtype=grad_type, device=f"npu:{device}")
    param_torch = torch.nn.Parameter(params_init.clone())
    optimizer_torch = torch.optim.Adagrad([param_torch], lr=lr, eps=eps)
    for _ in range(iter_num - 1):
        dummy_grad = torch.randn_like(param_torch, dtype=grad_type)
        optimizer_torch.zero_grad()
        param_torch.grad = dummy_grad
        optimizer_torch.step()

    current_params = param_torch.data.clone()
    if len(optimizer_torch.state) == 0:
        sum_init = torch.zeros_like(current_params)
    else:
        state = optimizer_torch.state[param_torch]
        sum_init = state["sum"].to(dtype=grad_type, device=current_params.device).clone()

    values = torch.cat([current_params, sum_init], dim=1).contiguous()
    keys = torch.arange(batch_size, dtype=torch.int64, device=f"npu:{device}")
    dtype_to_dynamic_emb = {
        torch.float32: DynamicEmbDataType.Float32,
        torch.float16: DynamicEmbDataType.Float16,
        torch.bfloat16: DynamicEmbDataType.BFloat16,
    }
    val_dynamic_type = dtype_to_dynamic_emb[grad_type]
    # 向量池需落在 HBM；按容量预留 w+acc，避免回落 host 触发 HKV 报错
    vector_capacity = max(2048, batch_size * 2)
    bytes_per_vector = embedding_dim * 2 * values.element_size()
    max_hbm_for_vectors = max(1 * 1024 * 1024 * 1024, int(vector_capacity * bytes_per_vector * 2))
    table = DynamicEmbTable(
        DynamicEmbDataType.Int64,
        val_dynamic_type,
        EvictStrategy.kLru,
        embedding_dim * 2,
        1024,
        vector_capacity,
        max_hbm_for_vectors,
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
        OptimizerType.Null,
    )
    table.load(batch_size, keys, values, None, True, False)
    torch.npu.synchronize()

    pointers_before = torch.empty(batch_size, dtype=torch.int64, device=f"npu:{device}")
    founds_before = torch.empty(batch_size, dtype=torch.bool, device=f"npu:{device}")
    find_pointers(table, batch_size, keys, pointers_before, founds_before)
    assert torch.all(founds_before).item(), "find_pointers failed before update"
    values_before = torch.zeros(batch_size, embedding_dim * 2, dtype=grad_type, device=f"npu:{device}")
    values_before = load_from_pointer(pointers_before, values_before)

    test_grad = torch.randn_like(current_params, dtype=grad_type)
    dynamic_emb_adagrad_with_table(
        table,
        batch_size,
        keys,
        test_grad,
        lr,
        eps,
        val_dynamic_type,
    )
    torch.npu.synchronize()

    optimizer_torch.zero_grad()
    param_torch.grad = test_grad.clone()
    optimizer_torch.step()
    torch.npu.synchronize()

    pointers = torch.empty(batch_size, dtype=torch.int64, device=f"npu:{device}")
    founds = torch.empty(batch_size, dtype=torch.bool, device=f"npu:{device}")
    find_pointers(table, batch_size, keys, pointers, founds)
    assert torch.all(founds).item()

    values_after = torch.zeros(batch_size, embedding_dim * 2, dtype=grad_type, device=f"npu:{device}")
    values_after = load_from_pointer(pointers, values_after)
    custom_param_result = values_after[:, :embedding_dim]
    torch_param_result = param_torch.data
    tol_map = {
        torch.float32: (1e-5, 1e-5),
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }
    rtol, atol = tol_map[grad_type]

    # Pointer path sanity check with same init state and same test_grad.
    mirror_values = values_before.clone()
    mirror_pointers = get_dim_pointers_optimized(mirror_values)
    dynamic_emb_adagrad_with_pointer(
        test_grad.clone(),
        mirror_pointers,
        val_dynamic_type,
        embedding_dim,
        lr,
        eps,
    )
    torch.npu.synchronize()
    pointer_param_result = mirror_values[:, :embedding_dim]

    try:
        torch.testing.assert_close(custom_param_result, torch_param_result, rtol=rtol, atol=atol)
    except AssertionError as err:
        diff_table_torch = (custom_param_result - torch_param_result).abs()
        diff_ptr_torch = (pointer_param_result - torch_param_result).abs()
        diff_table_ptr = (custom_param_result - pointer_param_result).abs()
        raise AssertionError(
            "with_table mismatch. "
            f"dtype={grad_type}, batch={batch_size}, dim={embedding_dim}, iter={iter_num}; "
            f"max|table-torch|={diff_table_torch.max().item():.6e}, "
            f"max|pointer-torch|={diff_ptr_torch.max().item():.6e}, "
            f"max|table-pointer|={diff_table_ptr.max().item():.6e}"
        ) from err


@pytest.mark.parametrize("device", [0])
@pytest.mark.parametrize("batch_size", [1, 1024, 4096, 8192, 102400])
@pytest.mark.parametrize("embedding_dim", [8, 64, 128, 256, 8192, 31, 1023])
@pytest.mark.parametrize("lr", [0.001, 0.01])
@pytest.mark.parametrize("iter_num", [10, 100])
@pytest.mark.parametrize(
    ("grad_type", "eps"),
    [
        (torch.float32, 1e-8),
        (torch.float16, 1e-4),
        (torch.bfloat16, 1e-8),
    ],
)
def test_dynamic_emb_adagrad_fused(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type):
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    lr = optimizer_params.lr
    eps = optimizer_params.eps

    params_init = torch.randn(batch_size, embedding_dim, dtype=grad_type, device=f"npu:{device}")
    param_torch = torch.nn.Parameter(params_init.clone())
    optimizer_torch = torch.optim.Adagrad([param_torch], lr=lr, eps=eps)
    for _ in range(iter_num - 1):
        dummy_grad = torch.randn_like(param_torch, dtype=grad_type)
        optimizer_torch.zero_grad()
        param_torch.grad = dummy_grad
        optimizer_torch.step()

    current_params = param_torch.data.clone()
    if len(optimizer_torch.state) == 0:
        sum_init = torch.zeros_like(current_params)
    else:
        state = optimizer_torch.state[param_torch]
        sum_init = state["sum"].to(dtype=grad_type, device=current_params.device).clone()

    values = torch.cat([current_params, sum_init], dim=1).contiguous()
    test_grad = torch.randn_like(current_params, dtype=grad_type)
    dynamic_emb_adagrad_fused(test_grad, values, lr, eps)
    torch.npu.synchronize()

    optimizer_torch.zero_grad()
    param_torch.grad = test_grad.clone()
    optimizer_torch.step()
    torch.npu.synchronize()

    custom_param_result = values[:, :embedding_dim]
    torch_param_result = param_torch.data
    tol_map = {
        torch.float32: (1e-5, 1e-5),
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }
    rtol, atol = tol_map[grad_type]
    torch.testing.assert_close(custom_param_result, torch_param_result, rtol=rtol, atol=atol)

