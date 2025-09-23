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

import sysconfig
import pytest
import fbgemm_gpu
import numpy as np
import torch_npu
import torch

# 加载NPU自定义算子库
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
# 设置用的卡号
DEVICE = "npu:0"


def jagged_to_padded_dense_wrapper(values, offsets, max_lengths, padding_value, is_mxrec):
    return JaggedToPaddedDense.apply(values, offsets, max_lengths, padding_value, is_mxrec)


class JaggedToPaddedDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, offsets, max_lengths, padding_value, is_mxrec):
        ctx.save_for_backward(*offsets)
        ctx.total_L = values.shape[0]
        ctx.is_mxrec = is_mxrec
        if is_mxrec:
            return torch.ops.mxrec.jagged_to_padded_dense_forward(
                values=values.to(DEVICE),
                offsets=offsets,
                max_lengths=max(max_lengths),
                padding_value=padding_value,
            )
        else:
            return torch.ops.fbgemm.jagged_to_padded_dense_forward(
                values=values.to(DEVICE),
                offsets=offsets,
                max_lengths=max(max_lengths),
                padding_value=padding_value,
            )

    @staticmethod
    def backward(ctx, grad_output):
        offsets = list(ctx.saved_tensors)
        total_L = ctx.total_L
        is_mxrec = ctx.is_mxrec
        if total_L is None:
            total_L = offsets[0][-1].item()
        if is_mxrec:
            grad_values = torch.ops.mxrec.jagged_to_padded_dense_backward(grad_output.to(DEVICE), offsets, total_L)
        else:
            grad_values = torch.ops.fbgemm.jagged_to_padded_dense_backward(grad_output.to(DEVICE), offsets, total_L)
        return grad_values, None, None, None, None


def generate_jagged_tensor(batch_size, max_seq_len, num_heads, attention_dim):
    """
    生成不规则(Jagged)张量测试数据
    Args:
        batch_size: 批处理大小
        max_seq_len: 单个样本最大序列长度
        num_heads: 注意力头数量
        attention_dim: 每个注意力头的维度

    Returns:
        jagged_tensor: 不规则数据张量，形状为(total_sequences, num_heads, attention_dim)
        seq_offsets: 序列偏移量数组，表示每个样本在jagged_tensor中的起始位置
        total_sequences: 所有样本的序列总长度
    """
    # 为每个样本随机生成序列长度(1到max_seq_len之间)
    seq_lens = np.random.randint(1, max_seq_len + 1, batch_size)

    # 计算累积偏移量(前面补0)
    seq_offsets = torch.concat((
        torch.zeros((1,), dtype=torch.int64),
        torch.cumsum(torch.from_numpy(seq_lens), dim=0)
    )).numpy()

    total_sequences = np.sum(seq_lens)

    # 生成随机数据(-1到1均匀分布)
    jagged_tensor = torch.rand(
        total_sequences, num_heads, attention_dim,
        dtype=torch.float32
    ).uniform_(-1, 1)

    return jagged_tensor, seq_offsets, total_sequences


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("max_seq_len", [128, 256])
@pytest.mark.parametrize("num_heads", [2, 8])
@pytest.mark.parametrize("attention_dim", [32])
@pytest.mark.parametrize("use_list_max_lengths", [True, False])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_jagged_to_padded_dense(batch_size,
                                max_seq_len,
                                num_heads,
                                attention_dim,
                                use_list_max_lengths,
                                is_mxrec):
    """
    测试不规则张量到填充密集张量的转换算子
    测试逻辑:
    1. 生成随机测试数据
    2. 使用FBGEMM的CPU实现计算基准结果
    3. 调用NPU算子计算结果
    4. 对比两者差异(允许1e-4的误差)
    5. 新增: 验证自动求导功能
    """
    # 1. 生成测试数据
    jagged_tensor, seq_offsets, total_sequences = generate_jagged_tensor(
        batch_size, max_seq_len, num_heads, attention_dim)

    # 2. 准备FBGEMM算子输入(需要展平最后两个维度)
    input_flat = jagged_tensor.reshape(total_sequences, num_heads * attention_dim)
    fbgemm_offsets = torch.from_numpy(seq_offsets)

    # ===== 前向传播验证 =====
    # 3. 调用FBGEMM CPU实现
    fbgemm_dense = torch.ops.fbgemm.jagged_to_padded_dense(
        input_flat,
        [fbgemm_offsets],
        [max_seq_len],
        0.0  # 填充值
    )

    # 4. 调用NPU算子
    if is_mxrec:
        npu_dense = torch.ops.mxrec.jagged_to_padded_dense(
            input_flat.to(DEVICE),
            [fbgemm_offsets.to(DEVICE)],
            [max_seq_len] if use_list_max_lengths else max_seq_len,
            0.0
        )
    else:
        npu_dense = torch.ops.fbgemm.jagged_to_padded_dense(
            input_flat.to(DEVICE),
            [fbgemm_offsets.to(DEVICE)],
            [max_seq_len] if use_list_max_lengths else max_seq_len,
            0.0
        )

    # 5. 前向传播结果比对
    assert torch.allclose(
        fbgemm_dense.reshape(-1),
        npu_dense.cpu().reshape(-1),
        atol=1e-4,
        rtol=1e-4
    ), f"NPU结果与FBGEMM CPU结果不匹配\nFBGEMM:\n{fbgemm_dense}\nNPU:\n{npu_dense.cpu()}"

    # ===== 反向传播验证 =====
    # 6. 准备可训练参数
    input_flat_npu = input_flat.clone().to(DEVICE).requires_grad_(True)
    input_flat_npu_py = input_flat.clone().to(DEVICE).requires_grad_(True)

    # 7. 计算NPU前向传播
    if is_mxrec:
        npu_dense_for_grad = torch.ops.mxrec.jagged_to_padded_dense(
            input_flat_npu,
            [fbgemm_offsets.to(DEVICE)],
            [max_seq_len] if use_list_max_lengths else max_seq_len,
            0.0
        )
    else:
        npu_dense_for_grad = torch.ops.fbgemm.jagged_to_padded_dense(
            input_flat_npu,
            [fbgemm_offsets.to(DEVICE)],
            [max_seq_len] if use_list_max_lengths else max_seq_len,
            0.0
        )

    # 8. 计算NPU python实现前向传播
    npu_py_dense_for_grad = jagged_to_padded_dense_wrapper(
        input_flat_npu_py,
        [fbgemm_offsets.to(DEVICE)],
        [max_seq_len],
        0.0,
        is_mxrec
    )

    # 9. 生成随机梯度(与输出形状相同)
    grad_output = torch.randn_like(npu_dense_for_grad)

    # 10. NPU反向传播
    npu_dense_for_grad.backward(grad_output.to(DEVICE))
    npu_grad_input = input_flat_npu.grad

    # 11. NPU python反向传播
    npu_py_dense_for_grad.backward(grad_output.to(DEVICE))
    npu_py_grad_input = input_flat_npu_py.grad

    # 12. 梯度比对
    assert torch.allclose(
        npu_py_grad_input.cpu(),
        npu_grad_input.cpu(),
        atol=1e-4,
        rtol=1e-4
    ), f"NPU python梯度与NPU梯度不匹配\nNPU python梯度:\n{npu_py_grad_input.cpu()}\nNPU梯度:\n{npu_grad_input.cpu()}"