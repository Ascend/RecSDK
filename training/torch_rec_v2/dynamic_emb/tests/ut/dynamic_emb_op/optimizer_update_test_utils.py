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

import torch

from dynamic_emb_extensions import (
    DynamicEmbDataType,
    DynamicEmbTable,
    EvictStrategy,
    InitializerArgs,
    OptimizerType,
    SafeCheckMode,
    dynamic_emb_adamW_fused,
    dynamic_emb_adamW_fused_hybrid,
    dynamic_emb_adamW_with_pointer,
    dynamic_emb_adamW_with_pointer_hybrid,
    dynamic_emb_adamW_with_table,
    dynamic_emb_sgd_fused,
    dynamic_emb_sgd_fused_hybrid,
    dynamic_emb_sgd_with_pointer,
    dynamic_emb_sgd_with_pointer_hybrid,
    dynamic_emb_sgd_with_table,
    find_pointers,
    load_from_pointer,
)

DTYPE_TO_DYNAMIC_EMB = {
    torch.float32: DynamicEmbDataType.Float32,
    torch.float16: DynamicEmbDataType.Float16,
    torch.bfloat16: DynamicEmbDataType.BFloat16,
}

TOL_MAP = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


class OptimizerParams:
    """优化器参数容器，用于 pytest fixture 与 kernel 调用。"""

    def __init__(self, lr, beta1, beta2, eps, weight_decay):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay


def create_adamw_torch_optimizer(param_torch, opt_params):
    return torch.optim.AdamW(
        [param_torch],
        lr=opt_params.lr,
        betas=(opt_params.beta1, opt_params.beta2),
        eps=opt_params.eps,
        weight_decay=opt_params.weight_decay,
    )


def run_adamw_with_pointer(test_grad, val_pointers, val_dynamic_type, state_dim, opt_params, iter_num):
    dynamic_emb_adamW_with_pointer(
        test_grad,
        val_pointers,
        val_dynamic_type,
        state_dim,
        opt_params.lr,
        opt_params.beta1,
        opt_params.beta2,
        opt_params.eps,
        opt_params.weight_decay,
        iter_num,
    )


def run_adamw_with_table(table, batch_size, keys, test_grad, opt_params, iter_num, val_dynamic_type):
    dynamic_emb_adamW_with_table(
        table,
        batch_size,
        keys,
        test_grad,
        opt_params.lr,
        opt_params.beta1,
        opt_params.beta2,
        opt_params.eps,
        opt_params.weight_decay,
        iter_num,
        val_dynamic_type,
    )


def run_adamw_fused(test_grad, values, opt_params, iter_num):
    dynamic_emb_adamW_fused(
        test_grad,
        values,
        opt_params.lr,
        opt_params.beta1,
        opt_params.beta2,
        opt_params.eps,
        opt_params.weight_decay,
        iter_num,
    )


def run_adamw_with_pointer_hybrid(test_grad, val_pointers, val_dynamic_type, state_dim, opt_params, iter_num):
    dynamic_emb_adamW_with_pointer_hybrid(
        test_grad,
        val_pointers,
        val_dynamic_type,
        state_dim,
        opt_params.lr,
        opt_params.beta1,
        opt_params.beta2,
        opt_params.eps,
        opt_params.weight_decay,
        iter_num,
    )


def run_adamw_fused_hybrid(test_grad, values, opt_params, iter_num):
    dynamic_emb_adamW_fused_hybrid(
        test_grad,
        values,
        opt_params.lr,
        opt_params.beta1,
        opt_params.beta2,
        opt_params.eps,
        opt_params.weight_decay,
        iter_num,
    )


def setup_npu_test(device: int) -> None:
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    torch.npu.set_device(device)


def get_dim_pointers_optimized(x_2d: torch.Tensor) -> torch.Tensor:
    """利用内存布局特性高效计算每行首地址"""
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    n, m = x_2d.size()
    elem_size = x_2d.element_size()
    row_stride = m * elem_size
    pointers = [x_2d.data_ptr() + i * row_stride for i in range(n)]
    return torch.tensor(pointers, dtype=torch.int64, device=x_2d.device)


def build_test_dynamic_emb_table(
    value_dim: int,
    device: int,
    value_type: DynamicEmbDataType = DynamicEmbDataType.Float32,
    max_capacity: int = 2048,
    optimizer_type: OptimizerType = OptimizerType.SGD,
) -> DynamicEmbTable:
    torch.npu.set_device(device)
    return DynamicEmbTable(
        DynamicEmbDataType.Int64,
        value_type,
        EvictStrategy.kLru,
        value_dim,
        1024,
        max_capacity,
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
        optimizer_type,
    )


def adamw_state_dim(embedding_dim: int) -> int:
    return embedding_dim * 2


def adamw_value_dim(embedding_dim: int) -> int:
    return embedding_dim * 3


def build_adamw_test_table(
    embedding_dim: int,
    batch_size: int,
    device: int,
    value_type: DynamicEmbDataType = DynamicEmbDataType.Float32,
    element_size: int = 4,
) -> DynamicEmbTable:
    torch.npu.set_device(device)
    value_dim = adamw_value_dim(embedding_dim)
    vector_capacity = max(2048, batch_size * 2)
    bytes_per_vector = value_dim * element_size
    max_hbm_for_vectors = max(1 * 1024 * 1024 * 1024, int(vector_capacity * bytes_per_vector * 2))
    return DynamicEmbTable(
        DynamicEmbDataType.Int64,
        value_type,
        EvictStrategy.kLru,
        value_dim,
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


def build_sgd_test_table(
    embedding_dim: int,
    batch_size: int,
    device: int,
    value_type: DynamicEmbDataType = DynamicEmbDataType.Float32,
    element_size: int = 4,
) -> DynamicEmbTable:
    """与 Adagrad table 测试一致：向量池落在 HBM，容量按 batch 预留。"""
    torch.npu.set_device(device)
    vector_capacity = max(2048, batch_size * 2)
    bytes_per_vector = embedding_dim * element_size
    max_hbm_for_vectors = max(1 * 1024 * 1024 * 1024, int(vector_capacity * bytes_per_vector * 2))
    return DynamicEmbTable(
        DynamicEmbDataType.Int64,
        value_type,
        EvictStrategy.kLru,
        embedding_dim,
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


def run_warmup_steps(param_torch, optimizer_torch, iter_num, grad_type=None) -> None:
    for _ in range(iter_num - 1):
        if grad_type is None:
            dummy_grad = torch.randn_like(param_torch)
        else:
            dummy_grad = torch.randn_like(param_torch, dtype=grad_type)
        optimizer_torch.zero_grad()
        param_torch.grad = dummy_grad
        optimizer_torch.step()


def run_reference_optimizer_step(param_torch, optimizer_torch, test_grad) -> None:
    optimizer_torch.zero_grad()
    param_torch.grad = test_grad.clone()
    optimizer_torch.step()
    torch.npu.synchronize()


def build_adamw_value_buffer(param_torch, optimizer_torch, grad_type, iter_num):
    run_warmup_steps(param_torch, optimizer_torch, iter_num, grad_type)

    current_params = param_torch.data.clone()
    if len(optimizer_torch.state) == 0:
        m_init = torch.zeros_like(current_params)
        v_init = torch.zeros_like(current_params)
    else:
        state = optimizer_torch.state[param_torch]
        m_init = state["exp_avg"].to(dtype=grad_type, device=current_params.device).clone()
        v_init = state["exp_avg_sq"].to(dtype=grad_type, device=current_params.device).clone()

    values = torch.cat([current_params, m_init, v_init], dim=1).contiguous()
    return values, current_params


def prepare_sgd_state(param_torch, optimizer_torch, grad_type, iter_num):
    """Run warmup steps and return (values, test_grad) for a single update comparison."""
    run_warmup_steps(param_torch, optimizer_torch, iter_num, grad_type)
    current_params = param_torch.data.clone()
    values = current_params.contiguous()
    test_grad = torch.randn_like(current_params, dtype=grad_type)
    return values, test_grad


def prepare_adamw_state(param_torch, optimizer_torch, grad_type, iter_num):
    """Run warmup steps and return (values, test_grad) for a single update comparison."""
    values, current_params = build_adamw_value_buffer(param_torch, optimizer_torch, grad_type, iter_num)
    test_grad = torch.randn_like(current_params, dtype=grad_type)
    return values, test_grad


def assert_close_by_grad_type(custom_result, torch_result, grad_type) -> None:
    rtol, atol = TOL_MAP[grad_type]
    torch.testing.assert_close(custom_result, torch_result, rtol=rtol, atol=atol)


def make_sequential_table_keys(batch_size: int, device: int) -> torch.Tensor:
    return torch.arange(batch_size, dtype=torch.int64, device=f"npu:{device}")


def load_values_into_table(table, batch_size: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    table.load(batch_size, keys, values, None, True, False)
    torch.npu.synchronize()


def read_table_values(
    table,
    batch_size: int,
    keys: torch.Tensor,
    row_dim: int,
    grad_type,
    device: int,
    *,
    founds_error_msg: str = "find_pointers failed",
) -> torch.Tensor:
    pointers = torch.empty(batch_size, dtype=torch.int64, device=f"npu:{device}")
    founds = torch.empty(batch_size, dtype=torch.bool, device=f"npu:{device}")
    find_pointers(table, batch_size, keys, pointers, founds)
    assert torch.all(founds).item(), founds_error_msg
    values = torch.zeros(batch_size, row_dim, dtype=grad_type, device=f"npu:{device}")
    return load_from_pointer(pointers, values)


def extract_weight_from_table_row(values: torch.Tensor, embedding_dim: int, table_row_dim: int) -> torch.Tensor:
    if table_row_dim == embedding_dim:
        return values
    return values[:, :embedding_dim]


def run_optimizer_table_update_test(
    *,
    device: int,
    batch_size: int,
    embedding_dim: int,
    grad_type,
    iter_num: int,
    param_torch,
    optimizer_torch,
    test_grad: torch.Tensor,
    values: torch.Tensor,
    table,
    table_row_dim: int,
    run_table_update,
    run_pointer_mirror,
) -> None:
    """Table 更新 + PyTorch 参考 + pointer 镜像校验（SGD / AdamW 共用）。"""
    keys = make_sequential_table_keys(batch_size, device)
    load_values_into_table(table, batch_size, keys, values)
    values_before = read_table_values(
        table,
        batch_size,
        keys,
        table_row_dim,
        grad_type,
        device,
        founds_error_msg="find_pointers failed before update",
    )

    run_table_update(keys, test_grad)
    torch.npu.synchronize()
    run_reference_optimizer_step(param_torch, optimizer_torch, test_grad)

    values_after = read_table_values(table, batch_size, keys, table_row_dim, grad_type, device)
    custom_param_result = extract_weight_from_table_row(values_after, embedding_dim, table_row_dim)

    mirror_values = values_before.clone()
    mirror_pointers = get_dim_pointers_optimized(mirror_values)
    run_pointer_mirror(test_grad.clone(), mirror_pointers, mirror_values)
    torch.npu.synchronize()
    pointer_param_result = extract_weight_from_table_row(mirror_values, embedding_dim, table_row_dim)

    assert_optimizer_table_matches_reference(
        custom_param_result,
        param_torch.data,
        pointer_param_result,
        grad_type,
        batch_size,
        embedding_dim,
        iter_num,
    )


def assert_optimizer_table_matches_reference(
    custom_param_result,
    torch_param_result,
    pointer_param_result,
    grad_type,
    batch_size,
    embedding_dim,
    iter_num,
) -> None:
    try:
        assert_close_by_grad_type(custom_param_result, torch_param_result, grad_type)
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


def _make_trainable_param(batch_size: int, embedding_dim: int, grad_type, device: int):
    setup_npu_test(device)
    params_init = torch.randn(batch_size, embedding_dim, dtype=grad_type, device=f"npu:{device}")
    return torch.nn.Parameter(params_init.clone())


def _run_single_step_compare(
    param_torch,
    optimizer_torch,
    values: torch.Tensor,
    test_grad: torch.Tensor,
    grad_type,
    embedding_dim: int,
    run_custom_update,
    *,
    weight_slice=None,
) -> None:
    run_custom_update()
    torch.npu.synchronize()
    run_reference_optimizer_step(param_torch, optimizer_torch, test_grad)
    custom_result = values if weight_slice is None else values[:, weight_slice]
    assert_close_by_grad_type(custom_result, param_torch.data, grad_type)


def run_sgd_pointer_test(device, batch_size, embedding_dim, lr, iter_num, grad_type) -> None:
    param_torch = _make_trainable_param(batch_size, embedding_dim, grad_type, device)
    optimizer_torch = torch.optim.SGD([param_torch], lr=lr)
    values, test_grad = prepare_sgd_state(param_torch, optimizer_torch, grad_type, iter_num)
    val_dynamic_type = DTYPE_TO_DYNAMIC_EMB[grad_type]
    val_pointers = get_dim_pointers_optimized(values)

    def run_custom_update():
        dynamic_emb_sgd_with_pointer(test_grad, val_pointers, val_dynamic_type, lr)

    _run_single_step_compare(
        param_torch, optimizer_torch, values, test_grad, grad_type, embedding_dim, run_custom_update
    )


def run_sgd_table_test(device, batch_size, embedding_dim, lr, iter_num, grad_type) -> None:
    param_torch = _make_trainable_param(batch_size, embedding_dim, grad_type, device)
    optimizer_torch = torch.optim.SGD([param_torch], lr=lr)
    values, test_grad = prepare_sgd_state(param_torch, optimizer_torch, grad_type, iter_num)
    val_dynamic_type = DTYPE_TO_DYNAMIC_EMB[grad_type]
    table = build_sgd_test_table(embedding_dim, batch_size, device, val_dynamic_type, values.element_size())

    def run_table_update(keys, grad):
        dynamic_emb_sgd_with_table(table, batch_size, keys, grad, lr, val_dynamic_type)

    def run_pointer_mirror(grad, mirror_pointers, _mirror_values):
        dynamic_emb_sgd_with_pointer(grad, mirror_pointers, val_dynamic_type, lr)

    run_optimizer_table_update_test(
        device=device,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        grad_type=grad_type,
        iter_num=iter_num,
        param_torch=param_torch,
        optimizer_torch=optimizer_torch,
        test_grad=test_grad,
        values=values,
        table=table,
        table_row_dim=embedding_dim,
        run_table_update=run_table_update,
        run_pointer_mirror=run_pointer_mirror,
    )


def run_sgd_fused_test(device, batch_size, embedding_dim, lr, iter_num, grad_type) -> None:
    param_torch = _make_trainable_param(batch_size, embedding_dim, grad_type, device)
    optimizer_torch = torch.optim.SGD([param_torch], lr=lr)
    values, test_grad = prepare_sgd_state(param_torch, optimizer_torch, grad_type, iter_num)

    def run_custom_update():
        dynamic_emb_sgd_fused(test_grad, values, lr)

    _run_single_step_compare(
        param_torch, optimizer_torch, values, test_grad, grad_type, embedding_dim, run_custom_update
    )


def run_sgd_pointer_hybrid_test(device, batch_size, embedding_dim, lr, iter_num, grad_type) -> None:
    param_torch = _make_trainable_param(batch_size, embedding_dim, grad_type, device)
    optimizer_torch = torch.optim.SGD([param_torch], lr=lr)
    values, test_grad = prepare_sgd_state(param_torch, optimizer_torch, grad_type, iter_num)
    val_dynamic_type = DTYPE_TO_DYNAMIC_EMB[grad_type]
    val_pointers = get_dim_pointers_optimized(values)

    def run_custom_update():
        dynamic_emb_sgd_with_pointer_hybrid(test_grad, val_pointers, val_dynamic_type, lr)

    _run_single_step_compare(
        param_torch, optimizer_torch, values, test_grad, grad_type, embedding_dim, run_custom_update
    )


def run_sgd_fused_hybrid_test(device, batch_size, embedding_dim, lr, iter_num, grad_type) -> None:
    param_torch = _make_trainable_param(batch_size, embedding_dim, grad_type, device)
    optimizer_torch = torch.optim.SGD([param_torch], lr=lr)
    values, test_grad = prepare_sgd_state(param_torch, optimizer_torch, grad_type, iter_num)

    def run_custom_update():
        dynamic_emb_sgd_fused_hybrid(test_grad, values, lr)

    _run_single_step_compare(
        param_torch, optimizer_torch, values, test_grad, grad_type, embedding_dim, run_custom_update
    )


def run_adamw_pointer_test(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type) -> None:
    param_torch = _make_trainable_param(batch_size, embedding_dim, grad_type, device)
    optimizer_torch = create_adamw_torch_optimizer(param_torch, optimizer_params)
    values, test_grad = prepare_adamw_state(param_torch, optimizer_torch, grad_type, iter_num)
    val_dynamic_type = DTYPE_TO_DYNAMIC_EMB[grad_type]
    val_pointers = get_dim_pointers_optimized(values)
    state_dim = adamw_state_dim(embedding_dim)

    def run_custom_update():
        run_adamw_with_pointer(test_grad, val_pointers, val_dynamic_type, state_dim, optimizer_params, iter_num)

    _run_single_step_compare(
        param_torch,
        optimizer_torch,
        values,
        test_grad,
        grad_type,
        embedding_dim,
        run_custom_update,
        weight_slice=slice(embedding_dim),
    )


def run_adamw_table_test(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type) -> None:
    param_torch = _make_trainable_param(batch_size, embedding_dim, grad_type, device)
    optimizer_torch = create_adamw_torch_optimizer(param_torch, optimizer_params)
    values, test_grad = prepare_adamw_state(param_torch, optimizer_torch, grad_type, iter_num)
    val_dynamic_type = DTYPE_TO_DYNAMIC_EMB[grad_type]
    value_dim = adamw_value_dim(embedding_dim)
    table = build_adamw_test_table(embedding_dim, batch_size, device, val_dynamic_type, values.element_size())

    def run_table_update(keys, grad):
        run_adamw_with_table(table, batch_size, keys, grad, optimizer_params, iter_num, val_dynamic_type)

    def run_pointer_mirror(grad, mirror_pointers, _mirror_values):
        run_adamw_with_pointer(
            grad,
            mirror_pointers,
            val_dynamic_type,
            adamw_state_dim(embedding_dim),
            optimizer_params,
            iter_num,
        )

    run_optimizer_table_update_test(
        device=device,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        grad_type=grad_type,
        iter_num=iter_num,
        param_torch=param_torch,
        optimizer_torch=optimizer_torch,
        test_grad=test_grad,
        values=values,
        table=table,
        table_row_dim=value_dim,
        run_table_update=run_table_update,
        run_pointer_mirror=run_pointer_mirror,
    )


def run_adamw_fused_test(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type) -> None:
    param_torch = _make_trainable_param(batch_size, embedding_dim, grad_type, device)
    optimizer_torch = create_adamw_torch_optimizer(param_torch, optimizer_params)
    values, test_grad = prepare_adamw_state(param_torch, optimizer_torch, grad_type, iter_num)

    def run_custom_update():
        run_adamw_fused(test_grad, values, optimizer_params, iter_num)

    _run_single_step_compare(
        param_torch,
        optimizer_torch,
        values,
        test_grad,
        grad_type,
        embedding_dim,
        run_custom_update,
        weight_slice=slice(embedding_dim),
    )


def run_adamw_pointer_hybrid_test(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type) -> None:
    param_torch = _make_trainable_param(batch_size, embedding_dim, grad_type, device)
    optimizer_torch = create_adamw_torch_optimizer(param_torch, optimizer_params)
    values, test_grad = prepare_adamw_state(param_torch, optimizer_torch, grad_type, iter_num)
    val_dynamic_type = DTYPE_TO_DYNAMIC_EMB[grad_type]
    val_pointers = get_dim_pointers_optimized(values)
    state_dim = adamw_state_dim(embedding_dim)

    def run_custom_update():
        run_adamw_with_pointer_hybrid(test_grad, val_pointers, val_dynamic_type, state_dim, optimizer_params, iter_num)

    _run_single_step_compare(
        param_torch,
        optimizer_torch,
        values,
        test_grad,
        grad_type,
        embedding_dim,
        run_custom_update,
        weight_slice=slice(embedding_dim),
    )


def run_adamw_fused_hybrid_test(device, batch_size, embedding_dim, optimizer_params, iter_num, grad_type) -> None:
    param_torch = _make_trainable_param(batch_size, embedding_dim, grad_type, device)
    optimizer_torch = create_adamw_torch_optimizer(param_torch, optimizer_params)
    values, test_grad = prepare_adamw_state(param_torch, optimizer_torch, grad_type, iter_num)

    def run_custom_update():
        run_adamw_fused_hybrid(test_grad, values, optimizer_params, iter_num)

    _run_single_step_compare(
        param_torch,
        optimizer_torch,
        values,
        test_grad,
        grad_type,
        embedding_dim,
        run_custom_update,
        weight_slice=slice(embedding_dim),
    )
