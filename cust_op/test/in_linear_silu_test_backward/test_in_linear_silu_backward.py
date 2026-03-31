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
import random
import sys
import os

import pytest
import numpy as np
import torch_npu
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../common')))
from test_common import set_seed, allclose
from utils import compare_data_with_double_pole

torch.npu.config.allow_internal_format = False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ln_linear_silu_backward_npu(mixed_uvqk_grad,
                                   normed_input,
                                   uvqk,
                                   bias,
                                   linear_output,
                                   dtype):
    # 1. 反向通过 F.silu 激活函数
    sigmoid_val = torch.sigmoid(linear_output.to(torch.float32))
    # silu的梯度: d(silu)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    silu_grad = sigmoid_val * (1 + linear_output * (1 - sigmoid_val))
  
    # 应用链式法则
    linear_output_grad = mixed_uvqk_grad * silu_grad # [SEQ_LEN, H]

    # 2. 反向通过线性层 (WX + b)
    # 对于线性变换 output = input @ weight + bias
    # a) 计算权重的梯度: dL/dW = input^T @ dL/d(output)
    # b) 计算输入的梯度: dL/dX = dL/d(output) @ weight^T
    # c) 计算偏置的梯度: dL/db = sum(dL/d(output), dim=0)
    if bias is not None:
        bias_grad = torch.sum(linear_output_grad, dim=0)
    else:
        bias_grad = None
    linear_output_grad = linear_output_grad.to(uvqk.dtype)
    normed_input_grad = torch.mm(linear_output_grad, uvqk)
    uvqk_grad = torch.mm(linear_output_grad.t(), normed_input)

    return normed_input_grad.to(uvqk.dtype), uvqk_grad.to(uvqk.dtype), bias_grad


def ln_linear_silu_backward_golden_cpu(mixed_uvqk_grad, normed_input, uvqk, bias, linear_output):
    sigmoid_val = sigmoid(linear_output.to(torch.float32).cpu().numpy())
    linear_output = linear_output.to(torch.float32).cpu().numpy()
    uvqk = uvqk.to(torch.float32).cpu().numpy()
    mixed_uvqk_grad = mixed_uvqk_grad.to(torch.float32).cpu().numpy()
    normed_input = normed_input.to(torch.float32).cpu().numpy()

    silu_grad = linear_output * sigmoid_val * (1 - sigmoid_val)

    silu_grad = sigmoid_val + silu_grad
    linear_output_grad = mixed_uvqk_grad * silu_grad # [SEQ_LEN, H]

    if bias is not None:
        bias_grad = np.sum(linear_output_grad, axis=0).astype(np.float32)
        bias_grad = torch.from_numpy(bias_grad)
    else:
        bias_grad = None

    normed_input_grad = np.matmul(linear_output_grad, uvqk)
    uvqk_grad = np.matmul(linear_output_grad.T, normed_input)

    return torch.from_numpy(normed_input_grad), torch.from_numpy(uvqk_grad), bias_grad


def ln_linear_silu_backward_fused(u_grad, v_grad, q_grad, k_grad, normed_input, uvqk, bias, linear_output, split_args):
    if bias is not None:
        bias = bias.npu()
    normed_input_grad, uvqk_grad, bias_grad, _ = torch.ops.mxrec.in_linear_silu_backward(normed_input.npu(
    ), uvqk.npu(), bias, u_grad.npu(), v_grad.npu(), q_grad.npu(), k_grad.npu(), linear_output.npu(), split_args)
    return normed_input_grad, uvqk_grad, bias_grad


def get_data(total_seqs, embeddim_dim, hidden_size, enable_bias, data_type, is_diff_dim):
    h = hidden_size // 4
    if (is_diff_dim):
        split_args = [h - 32, h - 16, h + 16, h + 32]
    else:
        split_args = [h] * 4
    N = sum(split_args)
    u_grad = torch.rand(total_seqs, split_args[0], dtype=data_type)
    v_grad = torch.rand(total_seqs, split_args[1], dtype=data_type)
    q_grad = torch.rand(total_seqs, split_args[2], dtype=data_type)
    k_grad = torch.rand(total_seqs, split_args[3], dtype=data_type)

    x = torch.rand(total_seqs, embeddim_dim, dtype=data_type).uniform_(-1, 1)
    weight = torch.rand(N, embeddim_dim, dtype=data_type).uniform_(-1, 1)
    linear_output = torch.rand(total_seqs, N, dtype=data_type).uniform_(-1, 1)

    if enable_bias:
        bias = torch.rand(N, dtype=torch.float32).uniform_(-1, 1)
    else:
        bias = None
    return u_grad, v_grad, q_grad, k_grad, x, \
        weight, bias, linear_output, split_args


def execute(total_seqs, dim, enable_bias, data_type, is_diff_dim):
    if dim[0] <= 32 and is_diff_dim:
        return
    u_grad, v_grad, q_grad, k_grad, x, weight, bias, linear_output, split_args = get_data(total_seqs,
                                                                                          dim[0],
                                                                                          dim[1],
                                                                                          enable_bias,
                                                                                          data_type,
                                                                                          is_diff_dim)

    mixed_uvqk_grad = torch.cat((u_grad, v_grad, q_grad, k_grad), dim=-1)

    x_grad_fused, weight_grad_fused, bias_grad_fused = ln_linear_silu_backward_fused(u_grad, v_grad, q_grad, k_grad,
                                                                 x, weight, bias, linear_output, split_args)

    x_grad_golden, weight_grad_golden, bias_grad_golden = ln_linear_silu_backward_golden_cpu(
        mixed_uvqk_grad, x, weight, bias, linear_output
    )
    if bias is not None:
        bias = bias.npu()

    x_grad_npu, weight_grad_npu, bias_grad_npu = ln_linear_silu_backward_npu(
        mixed_uvqk_grad.npu(), x.npu(), weight.npu(), bias, linear_output.npu(), x.dtype
    )

    compare_data_with_double_pole("x_grad", x_grad_fused, x_grad_npu, x_grad_golden)
    compare_data_with_double_pole("weight_grad", weight_grad_fused, weight_grad_npu, weight_grad_golden)
    if bias is not None:
        compare_data_with_double_pole("bias_grad", bias_grad_fused, bias_grad_npu, bias_grad_golden)
    


@pytest.mark.parametrize("total_seqs", [1, 1123, 2048, 3201, 7687, 20480])
@pytest.mark.parametrize("dim", [(16, 128), (64, 512), (240, 960), (128, 2048), (768, 768 * 4)])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_diff_dim", [False, True])
def test_in_linear_silu_backward(total_seqs, dim, enable_bias, data_type, is_diff_dim):
    set_seed(0)
    execute(total_seqs, dim, enable_bias, data_type, is_diff_dim)


@pytest.mark.parametrize("total_seqs", [511, 2048])
@pytest.mark.parametrize("dim", [(8160, 32640), (512, 6144), (1024, 16384)])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_diff_dim", [False])
def test_in_linear_silu_backward_large(total_seqs, dim, enable_bias, data_type, is_diff_dim):
    set_seed(0)
    execute(total_seqs, dim, enable_bias, data_type, is_diff_dim)


@pytest.mark.parametrize("invalid_case", [
    # splitList长度不是4
    ({"split_args": [64, 64, 64]}),
    # splitList元素不在[16, 8192]范围内
    ({"split_args": [8, 64, 64, 64]}),
    ({"split_args": [8200, 64, 64, 64]}),
    # splitList元素不是16的倍数
    ({"split_args": [17, 64, 64, 64]}),
    # x.dim[1]不在[16, 8192]范围内
    ({"dim": (8, 256)}),
    ({"dim": (8200, 256)}),
    # x.dim[1]不是16的倍数
    ({"dim": (17, 256)}),
    # weight.dim[0]不等于sum(splitList)
    ({"split_args": [64, 64, 64, 64], "weight_dim0": 257}),
    # weight.dim[0]不在[16, 8192]范围内
    ({"split_args": [2, 2, 2, 2]}),  # sum=8 < 64
    ({"split_args": [2050, 2050, 2050, 2050]}),  # sum=8200 > 8192
    # weight.dim[0]不是16的倍数
    ({"split_args": [17, 17, 17, 17]}),  # sum=68, 68%16=4
    # weight.dim[1]不等于x.dim[1]
    ({"dim": (64, 256), "weight_dim1": 257}),
    # weight.dim[1]不在[16, 8192]范围内
    ({"dim": (8, 256)}),
    ({"dim": (8200, 256)}),
    # weight.dim[1]不是16的倍数
    ({"dim": (17, 256)}),
    # split_args的dim与weight.dim[0]不匹配
    ({"split_args": [16, 16, 16, 16], "weight_dim0": 256}),
    # uvqk_total_seqs不等于total_seqs
    ({"uvqk_total_seqs": 1024}),
])
def test_in_linear_silu_backward_invalid_shape(invalid_case):
    """测试无效输入形状的情况"""
    total_seqs = 512
    uvqk_total_seqs = invalid_case.get("uvqk_total_seqs", total_seqs)
    dim = invalid_case.get("dim", (64, 256))
    enable_bias = False
    data_type = torch.float16

    split_args = invalid_case.get("split_args", [64, 64, 64, 64])
    N = sum(split_args)

    # 生成数据
    u_grad = torch.rand(uvqk_total_seqs, split_args[0] if len(split_args) > 0 else 64, dtype=data_type).uniform_(-1, 1)
    v_grad = torch.rand(uvqk_total_seqs, split_args[1] if len(split_args) > 1 else 64, dtype=data_type).uniform_(-1, 1)
    q_grad = torch.rand(uvqk_total_seqs, split_args[2] if len(split_args) > 2 else 64, dtype=data_type).uniform_(-1, 1)
    k_grad = torch.rand(uvqk_total_seqs, split_args[3] if len(split_args) > 3 else 64, dtype=data_type).uniform_(-1, 1)
    
    x = torch.rand(total_seqs, dim[0], dtype=data_type).uniform_(-1, 1)
    weight_dim0 = invalid_case.get("weight_dim0", N)
    weight_dim1 = invalid_case.get("weight_dim1", dim[0])
    weight = torch.rand(weight_dim0, weight_dim1, dtype=data_type).uniform_(-1, 1)
    
    linear_output = torch.rand(total_seqs, N, dtype=data_type).uniform_(-1, 1)
    
    if enable_bias:
        bias = torch.rand(N, dtype=torch.float32).uniform_(-1, 1).npu()
    else:
        bias = None

    u_grad = u_grad.npu()
    v_grad = v_grad.npu()
    q_grad = q_grad.npu()
    k_grad = k_grad.npu()
    x = x.npu()
    weight = weight.npu()
    linear_output = linear_output.npu()

    with pytest.raises(Exception):
        ln_linear_silu_backward_npu(u_grad, v_grad, q_grad, k_grad, x, weight,
            bias, linear_output, split_args)


@pytest.mark.parametrize("total_seqs", [1024])
@pytest.mark.parametrize("dim", [(128, 2048)])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize("data_type", [torch.float64])
@pytest.mark.parametrize("is_diff_dim", [True, False])
def test_in_linear_silu_backward_invalid_dtype(total_seqs, dim, enable_bias, data_type, is_diff_dim):
    u_grad, v_grad, q_grad, k_grad, x, weight, bias, linear_output, split_args = get_data(total_seqs,
                                                                                          dim[0],
                                                                                          dim[1],
                                                                                          enable_bias,
                                                                                          data_type,
                                                                                          is_diff_dim)
    with pytest.raises(Exception):
        ln_linear_silu_backward_npu(u_grad, v_grad, q_grad, k_grad, x, weight,
            bias, linear_output, split_args)


if __name__ == "__main__":
    set_seed(0)
    test_in_linear_silu_backward_large(total_seqs=3201,
                                 dim=(192, 768),
                                 enable_bias=True,
                                 data_type=torch.float16,
                                 is_diff_dim=True)