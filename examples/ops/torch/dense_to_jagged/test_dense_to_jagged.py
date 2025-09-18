#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
import itertools
import logging
import sysconfig

import pytest
import fbgemm_gpu
import numpy as np
import torch_npu
import torch

DEVICE = "npu:0"
logging.getLogger().setLevel(logging.INFO)
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DENSE_DIM0 = [128, 40] # 测试不同batch大小
DENSE_DIM1 = [210]     # 固定特征维度1
DENSE_DIM2 = [1, 8]    # 固定特征维度2
DIM_LIST = list(itertools.product(DENSE_DIM0, DENSE_DIM1, DENSE_DIM2))

DENSE_DATATYPE = [torch.float32, torch.int64] # 测试不同数据类型
OFFSET_DATATYPE = [torch.int32, torch.int64]  # 偏移量数据类型
TYPE_LIST = list(itertools.product(DENSE_DATATYPE, OFFSET_DATATYPE))


def dense_to_jagged_wrapper(dense, offsets, total_L=None):
    return DenseToJagged.apply(dense, offsets, total_L)


def jagged_to_padded_dense(values, offsets, max_lengths, padding_value):
    return torch.ops.mxrec.jagged_to_padded_dense(
        values=values.to(DEVICE),
        offsets=offsets,
        max_lengths=max(max_lengths),
        padding_value=padding_value,
    )


class DenseToJagged(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dense, offsets, total_L=None):
        ctx.save_for_backward(*offsets)
        if total_L is None:
            total_L = offsets[0][-1].item()
        out0, out1 = torch.ops.mxrec.dense_to_jagged(dense.to(DEVICE), offsets, total_L)
        ctx.dense_shape = dense.shape
        return out0.to(DEVICE), out1

    @staticmethod
    def backward(ctx, grad_out0, grad_out1):
        offsets = list(ctx.saved_tensors)
        max_len = ctx.dense_shape[1]
        grad_dense = torch.ops.mxrec.jagged_to_padded_dense(
            values=grad_out0.to(DEVICE),
            offsets=offsets,
            max_lengths=max([max_len]),
            padding_value=0.0,
        )
        return grad_dense, None, None


def get_result(device, denses, offsets, types, output_size=None):
    dense_datatype, offset_datatype = types
    dense_torch = torch.from_numpy(denses).to(dense_datatype).to(device)
    offsets_torch = torch.from_numpy(offsets).to(offset_datatype).to(device)

    # 计算累积偏移量
    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)

    # 执行核心操作：稠密张量→不规则张量
    jagged_embedding = torch.ops.fbgemm.dense_to_jagged(dense_torch, [jagged_id_offset], output_size)[0]
    return jagged_embedding.cpu()


@pytest.mark.parametrize("dims", DIM_LIST)
@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("output_size_type", ["none", "exact"])  # 测试不同output_size场景
def test_dense_to_jagged(dims, types, output_size_type):
    dense_dim0, dense_dim1, dense_dim2 = dims
    # 1. 生成随机输入数据
    denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0) # 生成随机偏移量

    # 计算实际的output_size
    actual_size = np.sum(offsets)

    # 根据测试类型设置output_size
    output_size = None
    if output_size_type == "exact":
        output_size = actual_size

    # 2. 分别获取CPU和NPU结果
    golden_result = get_result(torch.device("cpu"), denses, offsets, types, output_size)
    npu_result = get_result(torch.device(DEVICE), denses, offsets, types, output_size)

    # 3. 结果比对（允许1e-4的误差）
    # 正常情况应该完全匹配
    result_forward = torch.abs(golden_result[0] - npu_result[0]) < 1e-4
    assert result_forward.all().item()

    # ===== 反向传播验证 =====
    # 6. 准备可训练参数
    dense_datatype, offset_datatype = types
    dense_torch = torch.from_numpy(denses).to(dense_datatype).to(DEVICE)
    offsets_torch = torch.from_numpy(offsets).to(offset_datatype).to(DEVICE)

    # 计算累积偏移量
    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)

    input_dense_npu = dense_torch.clone().to(torch.float32).to(DEVICE).requires_grad_(True)
    input_dense_npu_py = dense_torch.clone().to(torch.float32).to(DEVICE).requires_grad_(True)

    # 7. 计算NPU前向传播
    npu_jagged_for_grad = torch.ops.mxrec.dense_to_jagged(
        input_dense_npu,
        [jagged_id_offset.to(DEVICE)],
        output_size
    )[0]

    # 8. 计算NPU python实现前向传播
    npu_py_jagged_for_grad = dense_to_jagged_wrapper(
        input_dense_npu_py,
        [jagged_id_offset.to(DEVICE)],
        output_size
    )[0]

    # 9. 生成随机梯度(与输出形状相同)
    grad_output = torch.randn_like(npu_jagged_for_grad)

    # 10. NPU反向传播
    npu_jagged_for_grad.backward(grad_output.to(DEVICE))
    npu_grad_input = input_dense_npu.grad

    # 11. NPU python反向传播
    npu_py_jagged_for_grad.backward(grad_output.to(DEVICE))
    npu_py_grad_input = input_dense_npu_py.grad

    # 12. 梯度比对
    assert torch.allclose(
        npu_py_grad_input.cpu(),
        npu_grad_input.cpu(),
        atol=1e-4,
        rtol=1e-4
    ), f"NPU python梯度与NPU梯度不匹配\nNPU python梯度:\n{npu_py_grad_input.cpu()}\nNPU梯度:\n{npu_grad_input.cpu()}"


# 专门测试异常情况的测试用例
@pytest.mark.parametrize("dims", [(128, 210, 8)])  # 固定维度简化测试
@pytest.mark.parametrize("types", [(torch.float32, torch.int32)])  # 固定类型简化测试
def test_dense_to_jagged_edge_cases(dims, types):
    dense_dim0, dense_dim1, dense_dim2 = dims
    # 1. 生成随机输入数据
    denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0)

    # 计算实际的output_size
    actual_size = np.sum(offsets)

    # 测试output_size为0的情况
    with pytest.raises(RuntimeError):
        _ = get_result(torch.device(DEVICE), denses, offsets, types, 0)

    # 测试output_size为负数的情况
    with pytest.raises(RuntimeError):
        _ = get_result(torch.device(DEVICE), denses, offsets, types, -1)

    # 测试大于actual_size的output_size情况
    with pytest.raises(RuntimeError):
        _ = get_result(torch.device(DEVICE), denses, offsets, types, actual_size + 10)

    # 测试小于actual_size的output_size情况
    with pytest.raises(RuntimeError):
        _ = get_result(torch.device(DEVICE), denses, offsets, types, max(1, actual_size - 10))