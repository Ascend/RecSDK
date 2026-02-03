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
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def get_user_value_query_key_tensors_uniattn(x, weight, bias, split_arg_list):
    mixed_uvqk = torch.addmm(bias, x, weight.t())
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


def get_user_value_query_key_tensors_fused(x, weight, bias, split_arg_list):
    result = torch.ops.mxrec.distance_in_linear_silu(
        x,
        weight,
        bias.to(torch.float32).contiguous(),
        split_arg_list
    )
    return result


def get_common_input():
    dtype = torch.float32
    split_arg_list = [16, 16, 16, 16]
    x = torch.randint(low=-100, high=101, size=(100, 16), dtype=dtype, device="npu")
    w = torch.randint(low=-100, high=101, size=(64, 16), dtype=dtype, device="npu")
    w = torch.nn.init.xavier_uniform_(w)
    bias = torch.ones(w.shape[0], dtype=dtype, device="npu")
    return x, w, bias, split_arg_list


def compare_result(result, golden, dtype):
    atol = 1e-4
    rtol = 1e-4
    if dtype == torch.float16:
        atol = 1e-3
        rtol = 1e-3
    elif dtype == torch.bfloat16:
        atol = 5e-3
        rtol = 5e-3
    for res, gold in zip(result, golden):
        assert torch.allclose(res, gold, atol=atol, rtol=rtol)


def in_linear_silu_diff_dim_test(m, n, k, device, dtype):
    # 构造数据
    seq_len = int(n / 4)
    split_arg_list = [seq_len - 16, seq_len - 16, seq_len + 16, seq_len + 16]
    x = torch.randint(low=-100, high=101, size=(m, k), dtype=dtype, device="npu")
    w = torch.randint(low=-100, high=101, size=(n, k), dtype=dtype, device="npu")
    w = torch.nn.init.xavier_uniform_(w)
    bias = torch.ones(n, dtype=dtype, device="npu")

    golden = get_user_value_query_key_tensors_uniattn(x, w, bias, split_arg_list)
    result = get_user_value_query_key_tensors_fused(x, w, bias, split_arg_list)
    compare_result(golden, result, dtype)


def in_linear_silu_test(m, n, k, device, dtype):
    # 构造数据
    seq_len = int(n / 4)
    split_arg_list = [seq_len, seq_len, seq_len, seq_len]
    x = torch.randint(low=-100, high=101, size=(m, k), dtype=dtype, device="npu")
    w = torch.randint(low=-100, high=101, size=(n, k), dtype=dtype, device="npu")
    w = torch.nn.init.xavier_uniform_(w)
    bias = torch.ones(n, dtype=dtype, device="npu")

    golden = get_user_value_query_key_tensors_uniattn(x, w, bias, split_arg_list)
    result = get_user_value_query_key_tensors_fused(x, w, bias, split_arg_list)
    compare_result(golden, result, dtype)


@pytest.mark.parametrize("m", [12892, 16384, 13582, 6450, 2789, 16196, 11873, 4387, 3677])
@pytest.mark.parametrize("n", [9216, 16384])
@pytest.mark.parametrize("k", [48, 64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_in_linear_silu(m, n, k, device, dtype):
    if (n != 16384 and k != 48):
        in_linear_silu_test(m, n, k, device, dtype)


@pytest.mark.parametrize("m", [256, 512, 2048, 20480])
@pytest.mark.parametrize("n", [1024, 4096, 9216, 16384])
@pytest.mark.parametrize("k", [16, 32, 48, 64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_in_linear_silu_normal(m, n, k, device, dtype):
    if (k != 48 and (n != 4096 or n != 16384)):
        in_linear_silu_test(m, n, k, device, dtype)


@pytest.mark.parametrize("m", [2561, 12048])
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("k", [64, 128, 256])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_in_linear_silu_diff_dim(m, n, k, device, dtype):
    in_linear_silu_diff_dim_test(m, n, k, device, dtype)


def test_invalid_split_args():
    x, weight, bias, split_arg_list = get_common_input()
    with pytest.raises(RuntimeError):
        split_arg_list = [16, 17, 15, 16]
        _ = get_user_value_query_key_tensors_fused(x, weight, bias, split_arg_list)
    with pytest.raises(RuntimeError):
        split_arg_list = [16, 16, 32, 32]
        _ = get_user_value_query_key_tensors_fused(x, weight, bias, split_arg_list)
    with pytest.raises(RuntimeError):
        split_arg_list = [16, 16, 32]
        _ = get_user_value_query_key_tensors_fused(x, weight, bias, split_arg_list)


def test_invalid_bias():
    x, weight, _, split_arg_list = get_common_input()
    bias_invalid = x.clone()
    with pytest.raises(RuntimeError):
        _ = get_user_value_query_key_tensors_fused(x, weight, bias_invalid, split_arg_list)


def test_invalid_x_weight():
    x, weight, bias, split_arg_list = get_common_input()
    with pytest.raises(RuntimeError):
        _ = get_user_value_query_key_tensors_fused(x.t(), weight, bias, split_arg_list)