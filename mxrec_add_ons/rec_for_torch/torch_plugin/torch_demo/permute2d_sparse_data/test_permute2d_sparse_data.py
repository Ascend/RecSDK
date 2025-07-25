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
import numpy as np

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

lengths_type = [np.int64, np.int32, np.int64, np.int32]
values_type = [np.int64, np.int32, np.float32, np.float32]


def get_result(permute, lengths, values):
    input_permute_torch = torch.from_numpy(permute)
    input_lengths_torch = torch.from_numpy(lengths)
    input_values_torch = torch.from_numpy(values)

    (permuted_lengths, permuted_values, permuted_weights) = (
        torch.ops.fbgemm.permute_2D_sparse_data(
            input_permute_torch,
            input_lengths_torch,
            input_values_torch,
        )
    )

    return permuted_lengths.cpu(), permuted_values.cpu()


def get_result_npu(permute, lengths, values, device="npu:0"):
    torch.npu.set_device(device)
    input_permute_torch = torch.from_numpy(permute).to(device)
    input_lengths_torch = torch.from_numpy(lengths).to(device)
    input_values_torch = torch.from_numpy(values).to(device)

    (permuted_lengths, permuted_values, permuted_weights) = (
        torch.ops.fbgemm.permute_2D_sparse_data(
            input_permute_torch, input_lengths_torch, input_values_torch, None
        )
    )
    torch.npu.synchronize()
    return permuted_lengths.cpu(), permuted_values.cpu()


@pytest.mark.parametrize("device", ["npu:0", "npu:5"])
@pytest.mark.parametrize("type_list", zip(lengths_type, values_type))
@pytest.mark.parametrize("permute_dim", np.random.randint(2, 30, 4).tolist())
@pytest.mark.parametrize("lengths", [2048, 20480, 204800])
def test_permute2d_sparse_data(type_list, device, permute_dim, lengths):
    ltype, vtype = type_list
    input_permute = np.arange(permute_dim).astype(np.int32)
    np.random.shuffle(input_permute)
    input_lengths = np.ones((permute_dim, lengths), dtype=ltype)
    input_values = np.arange(0, permute_dim * lengths).astype(vtype)

    golden = get_result(input_permute, input_lengths, input_values)
    result = get_result_npu(input_permute, input_lengths, input_values, device)

    assert torch.allclose(golden[0], result[0], atol=1e-5)
    assert torch.allclose(golden[1], result[1], atol=1e-5)
