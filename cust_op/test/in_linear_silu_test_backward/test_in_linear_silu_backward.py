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
import sysconfig
import random

import pytest
import numpy as np
import torch_npu
import torch

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
    return isPass


def get_data(total_seqs, embeddim_dim, hidden_size, data_type, is_diff_dim, enable_bias=True):
    h = hidden_size // 4
    if is_diff_dim:
        split_args = [h - 32, h - 32, h + 32, h + 32]
    else:
        split_args = [h] * 4
    N = sum(split_args)
    
    # 基础数据
    x = torch.rand(total_seqs, embeddim_dim, dtype=data_type).uniform_(-1, 1)
    weight = torch.rand(N, embeddim_dim, dtype=data_type).uniform_(-1, 1)
    
    # 可选的偏置
    if enable_bias:
        bias = torch.rand(N, dtype=data_type).uniform_(-1, 1)
    else:
        bias = None
    
    return x, weight, bias, split_args


def ln_linear_silu_backward_golden_cpu(mixed_uvqk_grad, normed_input, uvqk, bias, linear_output):
    # 1. 反向通过 F.silu 激活函数
    sigmoid_val = torch.sigmoid(linear_output.to(torch.float32))

    # silu的梯度: d(silu)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    silu_grad = linear_output * sigmoid_val * (1 - sigmoid_val)

    silu_grad = sigmoid_val + silu_grad
    # 应用链式法则
    linear_output_grad = mixed_uvqk_grad * silu_grad # [SEQ_LEN, H]

    # 2. 反向通过线性层 (WX + b)
    # 对于线性变换 output = input @ weight + bias
    # a) 计算偏置的梯度: dL/db = sum(dL/d(output), dim=0)
    # 因为 bias 会广播到每个样本，所以梯度是所有样本梯度的和
    if bias is not None:
        bias_grad = torch.sum(linear_output_grad, dim=0)
    else:
        bias_grad = None

    # b) 计算权重的梯度: dL/dW = input^T @ dL/d(output)
    # c) 计算输入的梯度: dL/dX = dL/d(output) @ weight^T
    linear_output_grad = linear_output_grad.to(uvqk.dtype)
    normed_input_grad = torch.mm(linear_output_grad, uvqk)
    uvqk_grad = torch.mm(linear_output_grad.t(), normed_input)  # [seq_len, H].t [seq_len, d]> [H, d]

    return normed_input_grad.to(uvqk.dtype), uvqk_grad.to(uvqk.dtype), bias_grad


def ln_linear_silu_backward_npu(u_grad, v_grad, q_grad, k_grad, normed_input, uvqk, bias, linear_output, split_args):
    normed_input_grad, uvqk_grad, bias_grad, _ = torch.ops.mxrec.in_linear_silu_backward(normed_input.npu(
    ), uvqk.npu(), bias, u_grad.npu(), v_grad.npu(), q_grad.npu(), k_grad.npu(), linear_output.npu(), split_args)
    return normed_input_grad, uvqk_grad, bias_grad


def get_data(total_seqs, embeddim_dim, hidden_size, enable_bias, data_type, is_diff_dim):
    h = hidden_size // 4
    if (is_diff_dim):
        split_args = [h - 32, h - 32, h + 32, h + 32]
    else:
        split_args = [h] * 4
    N = sum(split_args)
    u_grad = torch.rand(total_seqs, split_args[0], dtype=data_type).uniform_(-1, 1)
    v_grad = torch.rand(total_seqs, split_args[1], dtype=data_type).uniform_(-1, 1)
    q_grad = torch.rand(total_seqs, split_args[2], dtype=data_type).uniform_(-1, 1)
    k_grad = torch.rand(total_seqs, split_args[3], dtype=data_type).uniform_(-1, 1)

    x = torch.rand(total_seqs, embeddim_dim, dtype=data_type).uniform_(-1, 1)
    weight = torch.rand(N, embeddim_dim, dtype=data_type).uniform_(-1, 1)
    linear_output = torch.rand(total_seqs, N, dtype=data_type).uniform_(-1, 1)

    if enable_bias:
        bias = torch.rand(N, dtype=torch.float32).uniform_(-1, 1).npu()
    else:
        bias = None
    return u_grad.npu(), v_grad.npu(), q_grad.npu(), k_grad.npu(), x.npu(), \
        weight.npu(), bias, linear_output.npu(), split_args


def excute(total_seqs, dim, enable_bias, data_type, is_diff_dim):
    if dim[0] <= 32 and is_diff_dim:
        return
    u_grad, v_grad, q_grad, k_grad, x, weight, bias, linear_output, split_args = get_data(total_seqs,
                                                                                          dim[0],
                                                                                          dim[1],
                                                                                          enable_bias,
                                                                                          data_type,
                                                                                          is_diff_dim)

    mixed_uvqk_grad = torch.cat((u_grad, v_grad, q_grad, k_grad), dim=-1)
   
    x_grad, weight_grad, bias_grad = ln_linear_silu_backward_npu(u_grad, v_grad, q_grad, k_grad,
                                                                 x, weight, bias, linear_output, split_args)
    

    x_grad_golden, weight_grad_golden, bias_grad_golden = ln_linear_silu_backward_golden_cpu(
        mixed_uvqk_grad, x, weight, bias, linear_output
    )
    torch.npu.synchronize()
    loss = 1e-4
    if data_type == torch.float16:
        loss = 1e-3
    elif data_type == torch.bfloat16:
        loss = 5e-3
    
    x_res = allclose(x_grad, x_grad_golden, loss, loss, "x_grad")
    weight_res = allclose(weight_grad, weight_grad_golden, loss, loss, "weight_grad")
    bias_res = not enable_bias or allclose(bias_grad, bias_grad_golden, loss, loss, "bias_grad")
    
    assert x_res and weight_res and bias_res


@pytest.mark.parametrize("total_seqs", [1, 1123, 3201, 7687, 20480])
@pytest.mark.parametrize("dim", [(16, 128), (64, 512), (128, 2048)])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_diff_dim", [False, True])
def test_in_linear_silu_backward(total_seqs, dim, enable_bias, data_type, is_diff_dim):
    excute(total_seqs, dim, enable_bias, data_type, is_diff_dim)


@pytest.mark.parametrize("total_seqs", [9, 511])
@pytest.mark.parametrize("dim", [(8160, 32640), (512, 32768), (512, 6144)])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_diff_dim", [False])
def test_in_linear_silu_backward_large(total_seqs, dim, enable_bias, data_type, is_diff_dim):
    excute(total_seqs, dim, enable_bias, data_type, is_diff_dim)


if __name__ == "__main__":
    set_seed(0)
    test_in_linear_silu_backward(total_seqs=20480,
                                 dim=(16, 128),
                                 enable_bias=True,
                                 data_type=torch.float16,
                                 is_diff_dim=False)