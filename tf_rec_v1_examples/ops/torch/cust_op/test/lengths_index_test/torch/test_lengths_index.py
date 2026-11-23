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
import torch_npu
import fbgemm_gpu
import sysconfig

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

def get_complete_result(t_in, shape=None):
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(t_in)
    output_size = int(offsets[-1].item())

    output = torch.zeros(output_size, dtype=t_in.dtype)
    for seq_idx in range(len(t_in)):
        start = int(offsets[seq_idx].item())
        end = int(offsets[seq_idx + 1].item()) if seq_idx < len(t_in) - 1 else output_size
        output[start:end] = seq_idx
    return output

def get_exclusive_result(t_in, shape):
    # 输入shape时：使用 exclusive_cumsum
    offsets = torch.ops.fbgemm.asynchronous_exclusive_cumsum(t_in)
    output_size = int(shape)
    output = torch.zeros(output_size, dtype=t_in.dtype)
    for seq_idx in range(len(t_in)):
        start = int(offsets[seq_idx].item())
        end = int(offsets[seq_idx + 1].item()) if seq_idx < len(t_in) - 1 else output_size
        output[start:end] = seq_idx

    return output

def get_ops_result(t_in, shape):
    return torch.ops.fbgemm.lengths_index(t_in, shape).cpu()


# 不输入shape,使用非排他累加和
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0"]*5)
@pytest.mark.parametrize("lengths", [0, 1, 10, 100, 1000, 10000, 100000])
def test_lengths_index_complete_cumsum(dtype, device, lengths):
    t_int = torch.randint(0, 100, (lengths,), dtype=dtype)
    golden = get_complete_result(t_int)
    result = get_ops_result(t_int.to(device), None)
    assert torch.equal(result, golden)


# # 输入shape,使用排他累加和
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0"]*5)
@pytest.mark.parametrize("lengths", [0, 1, 10, 100, 1000, 10000, 100000])
def test_lengths_index_exclusive_cumsum(dtype, device, lengths):
    t_int = torch.randint(0, 100, (lengths,), dtype=dtype)
    shape = int(torch.sum(t_int).item())
    golden = get_exclusive_result(t_int, shape)
    result = get_ops_result(t_int.to(device), shape)
    assert torch.equal(result, golden)


# debug:确定性用例
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize(
    "lengths, shape, expected",
    [
        ([1], None, [0]),
        ([5], None, [0, 0, 0, 0, 0]),
        ([10], None, [0] * 10),
        ([4, 0, 2], None, [0, 0, 0, 0, 2, 2]),
        ([0, 0, 0], None, []),
        ([2, 0, 3], 5, [0, 0, 2, 2, 2]),
    ],
)
def test_lengths_index_reference_edges(dtype, device, lengths, shape, expected):
    t = torch.tensor(lengths, dtype=dtype)
    if shape is None:
        golden = get_complete_result(t)
        result = get_ops_result(t.to(device), None)
    else:
        golden = get_exclusive_result(t, shape)
        result = get_ops_result(t.to(device), shape)
    exp = torch.tensor(expected, dtype=dtype) if expected else torch.tensor([], dtype=dtype)
    assert torch.equal(golden, exp)
    assert torch.equal(result, golden)