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
import itertools
import logging
import sysconfig

from pathlib import Path
import pytest
import fbgemm_gpu
import numpy as np
import torch_npu
import torch

torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(str(CURR_DIR.parent.parent.parent / 
    "cust_op/framework/torch_plugin/torch_library/in_linear_silu/build/libin_linear_silu.so"))


def get_user_value_query_key_tensors_uniattn(x, split_arg_list, in_linear_uvqk):
    mixed_uvqk = in_linear_uvqk(x)
    linear_output = mixed_uvqk
    mixed_uvqk = torch.nn.functional.silu(mixed_uvqk)
    (user, value, query, key) = torch.split(
        mixed_uvqk,
        split_arg_list,
        dim=-1,
    )
    value = value.view(-1, split_arg_list[1]).contiguous()
    query = query.view(-1, split_arg_list[2]).contiguous()
    key = key.view(-1, split_arg_list[3]).contiguous()
    user = user.contiguous()
    return user, value, query, key, linear_output


def get_user_value_query_key_tensors_fused(x, split_arg_list, in_linear_uvqk):
    result = torch.ops.mxrec.distance_in_linear_silu(
        x,
        in_linear_uvqk.weight,
        in_linear_uvqk.bias.to(torch.float32).contiguous(),
        split_arg_list
    )
    return result


def in_linear_silu_test(m, n, k, device, dtype):
    # 构造数据
    seg_len = int(n / 4)
    split_arg_list = [seg_len, seg_len, seg_len, seg_len]
    x = torch.randint(low=-100, high=101, size=(m, k), dtype=dtype, device="npu")
    w = torch.randint(low=-100, high=101, size=(n, k), dtype=dtype, device="npu")
    w = torch.nn.init.xavier_uniform_(w)
    bias = torch.ones(n, dtype=dtype, device="npu")

    # golden
    x_golden = x.clone().contiguous().requires_grad_(True)
    w_golden = w.clone().contiguous().requires_grad_(True)
    bias_golden = bias.clone().contiguous().requires_grad_(True)

    def init_linear_golden(m: torch.nn.Module):
        m.weight.data = w_golden
        m.bias.data = bias_golden

    in_linear_golden = torch.nn.Linear(
        128,
        w_golden.shape[0],
        bias=True,
        device="cpu",
        dtype=x_golden.dtype
    ).apply(init_linear_golden)

    golden_input_grads = {}

    def save_x_grad(grad):
        golden_input_grads["x"] = grad.clone()

    def save_w_grad(grad):
        golden_input_grads["w"] = grad.clone()

    def save_bias_grad(grad):
        golden_input_grads["bias"] = grad.clone()

    x_golden.register_hook(save_x_grad)
    in_linear_golden.weight.register_hook(save_w_grad)
    in_linear_golden.bias.register_hook(save_bias_grad)

    golden = get_user_value_query_key_tensors_uniattn(x_golden, split_arg_list, in_linear_golden)

    # ops result
    x_result = x.clone().contiguous().requires_grad_(True)
    w_result = w.clone().contiguous().requires_grad_(True)
    bias_result = bias.clone().contiguous().requires_grad_(True)

    def init_linear_result(m: torch.nn.Module):
        m.weight.data = w_result
        m.bias.data = bias_result
    
    in_linear_result = torch.nn.Linear(
        128,
        w_result.shape[0],
        bias=True,
        device="npu",
        dtype=x_result.dtype
    ).apply(init_linear_result)

    result_input_grads = {}

    def save_x_r_grad(grad):
        result_input_grads["x"] = grad.clone()

    def save_w_r_grad(grad):
        result_input_grads["w"] = grad.clone()

    def save_bias_r_grad(grad):
        result_input_grads["bias"] = grad.clone()

    x_result.register_hook(save_x_r_grad)
    in_linear_result.weight.register_hook(save_w_r_grad)
    in_linear_result.bias.register_hook(save_bias_r_grad)
    result = get_user_value_query_key_tensors_fused(x_result, split_arg_list, in_linear_result)

    atol = 1 / 2048
    rtol = 1 / 2048
    if dtype == torch.float16:
        atol = 1 / 256
        rtol = 1 / 256
    elif dtype == torch.bfloat16:
        atol = 1 / 128
        rtol = 1 / 128
    assert torch.allclose(result[0], golden[0], atol=atol, rtol=rtol)
    assert torch.allclose(result[1], golden[1], atol=atol, rtol=rtol)
    assert torch.allclose(result[2], golden[2], atol=atol, rtol=rtol)
    assert torch.allclose(result[3], golden[3], atol=atol, rtol=rtol)


@pytest.mark.parametrize("m", [12892, 16384, 13582, 6450, 2789, 16196, 11873, 4387, 3677])
@pytest.mark.parametrize("n", [9216, 16384])
@pytest.mark.parametrize("k", [48, 64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_in_linear_silu(m, n, k, device, dtype):
    in_linear_silu_test(m, n, k, device, dtype)


@pytest.mark.parametrize("m", [256, 512, 2048, 20480])
@pytest.mark.parametrize("n", [1024, 4096, 9216, 16384])
@pytest.mark.parametrize("k", [16, 32, 48, 64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_in_linear_silu_normal(m, n, k, device, dtype):
    in_linear_silu_test(m, n, k, device, dtype)