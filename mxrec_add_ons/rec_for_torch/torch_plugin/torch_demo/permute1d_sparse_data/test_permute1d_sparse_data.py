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
import torch
import torch_npu
import fbgemm_gpu

# 定义用到的卡和so位置
DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

# 定义参数数据类型列表
lengths_type = [torch.int64, torch.int32, torch.int64, torch.int32]
values_type = [torch.int64, torch.int32, torch.float32, torch.float32]


def create_permute_tensor(cnt):
    return torch.arange(cnt, dtype=torch.int32)[torch.randperm(cnt)]


def create_lengths_tensor(cnt, dtype, max_value=10):
    return torch.randint(1, max_value, size=[cnt], dtype=dtype)


def create_values_tensor(cnt, dtype):
    values_construct_param = dict(size=[cnt], dtype=dtype)
    if dtype.is_floating_point:
        input_values = torch.rand(**values_construct_param)
    else:
        input_values = torch.randint(-10, 10, **values_construct_param)
    return input_values


# CPU调用permute_1D_sparse_data算子
def get_result(permute, lengths, values):
    (permuted_lengths, permuted_values, permuted_weights) = (
        torch.ops.fbgemm.permute_1D_sparse_data(permute, lengths, values)
    )

    return permuted_lengths.cpu(), permuted_values.cpu()


# NPU调用permute_1D_sparse_data算子
def get_result_npu(permute, lengths, values):
    torch.npu.set_device(DEVICE)
    input_permute_torch = permute.to(DEVICE)
    input_lengths_torch = lengths.to(DEVICE)
    input_values_torch = values.to(DEVICE)

    torch.npu.synchronize()
    (permuted_lengths, permuted_values, permuted_weights) = (
        torch.ops.fbgemm.permute_1D_sparse_data(
            input_permute_torch, input_lengths_torch, input_values_torch,
        )
    )
    torch.npu.synchronize()
    return permuted_lengths.cpu(), permuted_values.cpu()


@pytest.mark.parametrize("type_list", zip(lengths_type, values_type))
@pytest.mark.parametrize("permute_len", [2, 3, 5, 16, 64, 256, 1024])
@pytest.mark.parametrize("max_lengths", [2048, 20480, 204800])
def test_permute1d_sparse_data(type_list, permute_len, max_lengths):
    ltype, vtype = type_list
    input_permute = create_permute_tensor(permute_len)
    input_lengths = create_lengths_tensor(permute_len, dtype=ltype, max_value=max_lengths)
    input_values = create_values_tensor(sum(input_lengths), dtype=vtype)

    golden = get_result(input_permute, input_lengths, input_values)
    result = get_result_npu(input_permute, input_lengths, input_values)

    assert torch.allclose(golden[0], result[0], atol=1e-5)
    assert torch.allclose(golden[1], result[1], atol=1e-5)