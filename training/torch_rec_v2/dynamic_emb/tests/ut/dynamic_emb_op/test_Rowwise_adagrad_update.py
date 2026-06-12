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
    dynamic_emb_rowwise_adagrad_fused,
    dynamic_emb_rowwise_adagrad_fused_hybrid,
    dynamic_emb_rowwise_adagrad_with_pointer,
    dynamic_emb_rowwise_adagrad_with_pointer_hybrid,
    dynamic_emb_rowwise_adagrad_with_table,
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


@pytest.fixture(name="opt_params")
def _opt_params_fixture(lr, eps):
    return OptimizerParams(lr=lr, eps=eps)


def get_dim_pointers_optimized(x_2d):
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    n, m = x_2d.size()
    elem_size = x_2d.element_size()
    row_stride = m * elem_size
    pointers = [x_2d.data_ptr() + i * row_stride for i in range(n)]
    return torch.tensor(pointers, dtype=torch.int64, device=x_2d.device)


def get_rowwise_state_dim(dtype: torch.dtype) -> int:
    # 与 csrc/dynamic_variable_base.h 中 RowWiseAdaGrad 的定义一致：16 / sizeof(T)
    return 16 // torch.tensor([], dtype=dtype).element_size()


def rowwise_adagrad_step(param: torch.Tensor, state: torch.Tensor, grad: torch.Tensor, lr: float, eps: float):
    # 参考 RowWiseAdaGradVecOptimizer：每行累计 grad^2 的均值
    row_accum = state[:, :1].float()
    row_accum = row_accum + (grad.float() * grad.float()).mean(dim=1, keepdim=True)
    denom = torch.sqrt(row_accum) + eps
    new_param = param.float() - lr * grad.float() / denom
    param.copy_(new_param.to(param.dtype))
    state[:, 0:1].copy_(row_accum.to(state.dtype))


def get_rowwise_adagrad_tol(grad_type: torch.dtype) -> tuple[float, float]:
    tol_map = {
        torch.float32: (1e-5, 1e-5),
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }
    return tol_map[grad_type]


def assert_rowwise_adagrad_close(
    custom_values: torch.Tensor,
    ref_params: torch.Tensor,
    ref_state: torch.Tensor,
    embedding_dim: int,
    rtol: float,
    atol: float,
) -> None:
    custom_param_result = custom_values[:, :embedding_dim]
    custom_state_result = custom_values[:, embedding_dim:]
    try:
        torch.testing.assert_close(custom_param_result, ref_params, rtol=rtol, atol=atol)
    except AssertionError:
        max_diff = (custom_param_result.float() - ref_params.float()).abs().max().item()
        print(f"param max_diff={max_diff}")
        raise
    try:
        torch.testing.assert_close(custom_state_result, ref_state, rtol=rtol, atol=atol)
    except AssertionError:
        max_diff = (custom_state_result.float() - ref_state.float()).abs().max().item()
        print(f"state max_diff={max_diff}")
        raise


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
def test_dynamic_emb_rowwise_adagrad_with_pointer(device, batch_size, embedding_dim, opt_params, iter_num, grad_type):
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    lr = opt_params.lr
    eps = opt_params.eps
    state_dim = get_rowwise_state_dim(grad_type)

    params = torch.randn(batch_size, embedding_dim, dtype=grad_type, device=f"npu:{device}")
    state = torch.zeros(batch_size, state_dim, dtype=grad_type, device=f"npu:{device}")
    for _ in range(iter_num - 1):
        dummy_grad = torch.randn_like(params, dtype=grad_type)
        rowwise_adagrad_step(params, state, dummy_grad, lr, eps)

    values = torch.cat([params.clone(), state.clone()], dim=1).contiguous()
    val_pointers = get_dim_pointers_optimized(values)
    test_grad = torch.randn_like(params, dtype=grad_type)

    dtype_to_dynamic_emb = {
        torch.float32: DynamicEmbDataType.Float32,
        torch.float16: DynamicEmbDataType.Float16,
        torch.bfloat16: DynamicEmbDataType.BFloat16,
    }
    val_dynamic_type = dtype_to_dynamic_emb[grad_type]
    dynamic_emb_rowwise_adagrad_with_pointer(
        test_grad,
        val_pointers,
        val_dynamic_type,
        state_dim,
        lr,
        eps,
    )
    torch.npu.synchronize()

    ref_params = params.clone()
    ref_state = state.clone()
    rowwise_adagrad_step(ref_params, ref_state, test_grad, lr, eps)
    torch.npu.synchronize()

    rtol, atol = get_rowwise_adagrad_tol(grad_type)
    assert_rowwise_adagrad_close(values, ref_params, ref_state, embedding_dim, rtol, atol)


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
def test_dynamic_emb_rowwise_adagrad_with_pointer_hybrid(
    device, batch_size, embedding_dim, opt_params, iter_num, grad_type
):
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    lr = opt_params.lr
    eps = opt_params.eps
    state_dim = get_rowwise_state_dim(grad_type)

    params = torch.randn(batch_size, embedding_dim, dtype=grad_type, device=f"npu:{device}")
    state = torch.zeros(batch_size, state_dim, dtype=grad_type, device=f"npu:{device}")
    for _ in range(iter_num - 1):
        dummy_grad = torch.randn_like(params, dtype=grad_type)
        rowwise_adagrad_step(params, state, dummy_grad, lr, eps)

    values = torch.cat([params.clone(), state.clone()], dim=1).contiguous()
    val_pointers = get_dim_pointers_optimized(values)
    test_grad = torch.randn_like(params, dtype=grad_type)

    dtype_to_dynamic_emb = {
        torch.float32: DynamicEmbDataType.Float32,
        torch.float16: DynamicEmbDataType.Float16,
        torch.bfloat16: DynamicEmbDataType.BFloat16,
    }
    val_dynamic_type = dtype_to_dynamic_emb[grad_type]
    dynamic_emb_rowwise_adagrad_with_pointer_hybrid(
        test_grad,
        val_pointers,
        val_dynamic_type,
        state_dim,
        lr,
        eps,
    )
    torch.npu.synchronize()

    ref_params = params.clone()
    ref_state = state.clone()
    rowwise_adagrad_step(ref_params, ref_state, test_grad, lr, eps)
    torch.npu.synchronize()

    rtol, atol = get_rowwise_adagrad_tol(grad_type)
    assert_rowwise_adagrad_close(values, ref_params, ref_state, embedding_dim, rtol, atol)


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
def test_dynamic_emb_rowwise_adagrad_with_table(device, batch_size, embedding_dim, opt_params, iter_num, grad_type):
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    lr = opt_params.lr
    eps = opt_params.eps
    state_dim = get_rowwise_state_dim(grad_type)

    params = torch.randn(batch_size, embedding_dim, dtype=grad_type, device=f"npu:{device}")
    state = torch.zeros(batch_size, state_dim, dtype=grad_type, device=f"npu:{device}")
    for _ in range(iter_num - 1):
        dummy_grad = torch.randn_like(params, dtype=grad_type)
        rowwise_adagrad_step(params, state, dummy_grad, lr, eps)

    values = torch.cat([params.clone(), state.clone()], dim=1).contiguous()
    keys = torch.arange(batch_size, dtype=torch.int64, device=f"npu:{device}")
    dtype_to_dynamic_emb = {
        torch.float32: DynamicEmbDataType.Float32,
        torch.float16: DynamicEmbDataType.Float16,
        torch.bfloat16: DynamicEmbDataType.BFloat16,
    }
    val_dynamic_type = dtype_to_dynamic_emb[grad_type]
    # 向量池需落在 HBM；按容量预留 w+state，避免回落 host 触发 HKV 报错
    vector_capacity = max(2048, batch_size * 2)
    bytes_per_vector = (embedding_dim + state_dim) * values.element_size()
    max_hbm_for_vectors = max(1 * 1024 * 1024 * 1024, int(vector_capacity * bytes_per_vector * 2))
    table = DynamicEmbTable(
        DynamicEmbDataType.Int64,
        val_dynamic_type,
        EvictStrategy.kLru,
        embedding_dim + state_dim,
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
    values_before = torch.zeros(batch_size, embedding_dim + state_dim, dtype=grad_type, device=f"npu:{device}")
    values_before = load_from_pointer(pointers_before, values_before)

    test_grad = torch.randn_like(params, dtype=grad_type)
    dynamic_emb_rowwise_adagrad_with_table(
        table,
        batch_size,
        keys,
        test_grad,
        lr,
        eps,
        val_dynamic_type,
    )
    torch.npu.synchronize()

    ref_params = params.clone()
    ref_state = state.clone()
    rowwise_adagrad_step(ref_params, ref_state, test_grad, lr, eps)

    pointers = torch.empty(batch_size, dtype=torch.int64, device=f"npu:{device}")
    founds = torch.empty(batch_size, dtype=torch.bool, device=f"npu:{device}")
    find_pointers(table, batch_size, keys, pointers, founds)
    assert torch.all(founds).item()

    values_after = torch.zeros(batch_size, embedding_dim + state_dim, dtype=grad_type, device=f"npu:{device}")
    values_after = load_from_pointer(pointers, values_after)
    rtol, atol = get_rowwise_adagrad_tol(grad_type)

    mirror_values = values_before.clone()
    mirror_pointers = get_dim_pointers_optimized(mirror_values)
    dynamic_emb_rowwise_adagrad_with_pointer(
        test_grad.clone(),
        mirror_pointers,
        val_dynamic_type,
        state_dim,
        lr,
        eps,
    )
    torch.npu.synchronize()

    try:
        assert_rowwise_adagrad_close(values_after, ref_params, ref_state, embedding_dim, rtol, atol)
    except AssertionError as err:
        custom_param = values_after[:, :embedding_dim]
        custom_state = values_after[:, embedding_dim:]
        pointer_param = mirror_values[:, :embedding_dim]
        pointer_state = mirror_values[:, embedding_dim:]
        diff_param_table_ref = (custom_param - ref_params).abs()
        diff_state_table_ref = (custom_state - ref_state).abs()
        diff_param_ptr_ref = (pointer_param - ref_params).abs()
        diff_state_ptr_ref = (pointer_state - ref_state).abs()
        diff_param_table_ptr = (custom_param - pointer_param).abs()
        diff_state_table_ptr = (custom_state - pointer_state).abs()
        raise AssertionError(
            "with_table mismatch. "
            f"dtype={grad_type}, batch={batch_size}, dim={embedding_dim}, iter={iter_num}; "
            f"max|param_table-ref|={diff_param_table_ref.max().item():.6e}, "
            f"max|state_table-ref|={diff_state_table_ref.max().item():.6e}, "
            f"max|param_pointer-ref|={diff_param_ptr_ref.max().item():.6e}, "
            f"max|state_pointer-ref|={diff_state_ptr_ref.max().item():.6e}, "
            f"max|param_table-pointer|={diff_param_table_ptr.max().item():.6e}, "
            f"max|state_table-pointer|={diff_state_table_ptr.max().item():.6e}"
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
def test_dynamic_emb_rowwise_adagrad_fused(device, batch_size, embedding_dim, opt_params, iter_num, grad_type):
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    lr = opt_params.lr
    eps = opt_params.eps
    state_dim = get_rowwise_state_dim(grad_type)

    params = torch.randn(batch_size, embedding_dim, dtype=grad_type, device=f"npu:{device}")
    state = torch.zeros(batch_size, state_dim, dtype=grad_type, device=f"npu:{device}")
    for _ in range(iter_num - 1):
        dummy_grad = torch.randn_like(params, dtype=grad_type)
        rowwise_adagrad_step(params, state, dummy_grad, lr, eps)

    values = torch.cat([params.clone(), state.clone()], dim=1).contiguous()
    test_grad = torch.randn_like(params, dtype=grad_type)
    dynamic_emb_rowwise_adagrad_fused(test_grad, values, lr, eps)
    torch.npu.synchronize()

    ref_params = params.clone()
    ref_state = state.clone()
    rowwise_adagrad_step(ref_params, ref_state, test_grad, lr, eps)

    rtol, atol = get_rowwise_adagrad_tol(grad_type)
    assert_rowwise_adagrad_close(values, ref_params, ref_state, embedding_dim, rtol, atol)


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
def test_dynamic_emb_rowwise_adagrad_fused_hybrid(device, batch_size, embedding_dim, opt_params, iter_num, grad_type):
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)

    lr = opt_params.lr
    eps = opt_params.eps
    state_dim = get_rowwise_state_dim(grad_type)

    params = torch.randn(batch_size, embedding_dim, dtype=grad_type, device=f"npu:{device}")
    state = torch.zeros(batch_size, state_dim, dtype=grad_type, device=f"npu:{device}")
    for _ in range(iter_num - 1):
        dummy_grad = torch.randn_like(params, dtype=grad_type)
        rowwise_adagrad_step(params, state, dummy_grad, lr, eps)

    values = torch.cat([params.clone(), state.clone()], dim=1).contiguous()
    test_grad = torch.randn_like(params, dtype=grad_type)
    dynamic_emb_rowwise_adagrad_fused_hybrid(test_grad, values, lr, eps)
    torch.npu.synchronize()

    ref_params = params.clone()
    ref_state = state.clone()
    rowwise_adagrad_step(ref_params, ref_state, test_grad, lr, eps)
    torch.npu.synchronize()

    rtol, atol = get_rowwise_adagrad_tol(grad_type)
    assert_rowwise_adagrad_close(values, ref_params, ref_state, embedding_dim, rtol, atol)
