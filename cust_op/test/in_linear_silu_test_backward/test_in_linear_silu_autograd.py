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
import itertools
import logging
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import sysconfig
from pathlib import Path
import pytest
import fbgemm_gpu
import numpy as np
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F

from test_common import set_seed, allclose


def get_data_autograd(total_seqs, embeddim_dim, hidden_size, data_type, is_diff_dim, enable_bias=True):
    h = hidden_size // 4
    if is_diff_dim:
        split_args = [h - 32, h - 32, h + 32, h + 32]
    else:
        split_args = [h] * 4
    N = sum(split_args)

    # torch.rand初始化避免梯度为0
    x = torch.rand(total_seqs, embeddim_dim, dtype=data_type)
    weight = torch.rand(N, embeddim_dim, dtype=data_type)
    target = torch.rand(total_seqs, split_args[0], dtype=data_type)
    # 可选的偏置
    if enable_bias:
        bias = torch.rand(N, dtype=data_type)
    else:
        bias = None

    return x, weight, bias, target, split_args


def ln_linear_silu_backward_npu(x, weight, bias, target, split_args):
    x = torch.nn.Parameter(torch.Tensor(x), requires_grad=True).to("npu")
    weight = torch.nn.Parameter(torch.Tensor(weight), requires_grad=True).to("npu")
    bias = torch.nn.Parameter(torch.Tensor(bias).to(torch.float32), requires_grad=True).to("npu")

    user, value, query, key, _ = torch.ops.mxrec.in_linear_silu(
      x, weight, bias, split_args
    )
    x.retain_grad()
    weight.retain_grad()
    bias.retain_grad()
    output = (user + value + query + key) / 4
    loss = nn.MSELoss()(output, target.npu())
    loss.backward()
    x_grad = x.grad.detach().cpu().clone().to(torch.float32)
    weight_grad = weight.grad.detach().cpu().clone().to(torch.float32)
    bias_grad = bias.grad.detach().cpu().clone().to(torch.float32)
    return x_grad, weight_grad, bias_grad


def ln_linear_silu_backward_golden(x, weight, bias, target, split_arg_list):
    x = torch.nn.Parameter(torch.Tensor(x).to(torch.float32), requires_grad=True)
    weight = torch.nn.Parameter(torch.Tensor(weight).to(torch.float32), requires_grad=True)
    if bias is not None:
        bias = torch.nn.Parameter(torch.Tensor(bias).to(torch.float32), requires_grad=True)
        mixed_uvqk = F.linear(x, weight, bias)
    else:
        mixed_uvqk = F.linear(x, weight)
    
    # 应用silu激活函数
    mixed_uvqk = F.silu(mixed_uvqk)
    
    # 分割为user, value, query, key
    (user, value, query, key) = torch.split(
        mixed_uvqk,
        split_arg_list,
        dim=-1,
    )

    output = (user + value + query + key) / 4
    loss = nn.MSELoss()(output, target.to(output.dtype))
    loss.backward()
    # 获取梯度
    x_grad = x.grad.detach().cpu().clone().to(torch.float32)
    weight_grad = weight.grad.detach().cpu().clone().to(torch.float32)
    bias_grad = bias.grad.detach().cpu().clone().to(torch.float32)
    
    return x_grad, weight_grad, bias_grad


def excute(total_seqs, dim, data_type, is_diff_dim):
    if dim[0] <= 32 and is_diff_dim:
        return
    x, weight, bias, target, split_args = get_data_autograd(total_seqs,
                                        dim[0],
                                        dim[1],
                                        data_type,
                                        is_diff_dim)
    x_grad_golden, weight_grad_golden, bias_grad_golden = ln_linear_silu_backward_golden(
        x, weight, bias, target, split_args
    )
   
    x_grad, weight_grad, bias_grad = ln_linear_silu_backward_npu(x, weight, bias, target, split_args)

    torch.npu.synchronize()
    loss = 1e-4
    if data_type == torch.float16:
        loss = 1e-3
    elif data_type == torch.bfloat16:
        loss = 5e-3
    
    x_res = allclose(x_grad, x_grad_golden, loss, loss, "x")
    weight_res = allclose(weight_grad, weight_grad_golden, loss, loss, "weight")
    bias_res = allclose(bias_grad, bias_grad_golden, loss, loss, "bias")
    
    assert x_res and weight_res and bias_res


@pytest.mark.parametrize("total_seqs", [1, 1123, 20480, 32001, 7680])
@pytest.mark.parametrize("dim", [(16, 128), (64, 512), (128, 2048)])
@pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_diff_dim", [False])
def test_in_linear_silu_backward(total_seqs, dim, data_type, is_diff_dim):
    if dim[0] < 32 and is_diff_dim:
        return
    excute(total_seqs, dim, data_type, is_diff_dim)


@pytest.mark.parametrize("total_seqs", [9, 1221])
@pytest.mark.parametrize("dim", [(8160, 32640), (512, 32768), (512, 6144)])
@pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_diff_dim", [False])
def test_in_linear_silu_backward_large(total_seqs, dim, data_type, is_diff_dim):
    excute(total_seqs, dim, data_type, is_diff_dim)


if __name__ == "__main__":
    set_seed(0)
    test_in_linear_silu_backward(total_seqs=1024,
                                dim=(16, 64),
                                data_type=torch.float16,
                                is_diff_dim=True)