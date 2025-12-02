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
import random
import sysconfig
import pytest
import torch
import torch_npu
import numpy as np
import fbgemm_gpu


torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(10000)


def generate_data(length, dtype):
    tensor = torch.arange(0, length).to(dtype)
    shuffled_tensor = tensor[torch.randperm(tensor.size(0))]
    return shuffled_tensor


def get_result(permute, device):
    if "cpu" in device:
        result = torch.ops.fbgemm.invert_permute(permute)
    elif "npu" in device:
        permute = permute.to(device)
        result = torch.ops.mxrec.invert_permute(permute)
        torch.npu.synchronize()
    else:
        raise ValueError(f"Unsupported device: {device}")

    return result.cpu().detach().numpy()


@pytest.mark.parametrize("length", [i for i in range(5, 1025, 16)])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_invert_permute_small(length, device, dtype):
    permute = generate_data(length, dtype)
    result_cpu = get_result(permute, "cpu")
    result_npu = get_result(permute, device)
    assert np.array_equal(result_cpu, result_npu)


@pytest.mark.parametrize("length", [i for i in range(1031, 16385, 256)])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_invert_permute_middle(length, device, dtype):
    permute = generate_data(length, dtype)
    result_cpu = get_result(permute, "cpu")
    result_npu = get_result(permute, device)
    assert np.array_equal(result_cpu, result_npu)


@pytest.mark.parametrize("length", [i for i in range(16964, 1048577, 8192)])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_invert_permute_large(length, device, dtype):
    permute = generate_data(length, dtype)
    result_cpu = get_result(permute, "cpu")
    result_npu = get_result(permute, device)
    assert np.array_equal(result_cpu, result_npu)