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


def large_differences(a, b, threshold=0.001):
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()

    rel_diff = torch.abs(a_flat - b_flat) / (torch.abs(b_flat) + 1e-7)

    large_indices = torch.where(rel_diff > threshold)[0]
    return len(large_indices) == 0


def get_global_loss(x_np, weight_np, bias_np, attr_dict):
    x = torch.from_numpy(x_np)
    weight = torch.from_numpy(weight_np)
    bias = torch.from_numpy(bias_np)
    mixed_uvqk = torch.addmm(bias, x, weight.T)
    mixed_uvqk = torch.nn.functional.silu(mixed_uvqk)
    (user, value, query, key) = torch.split(
        mixed_uvqk,
        attr_dict,
        dim=-1,
    )
    return user, value, query, key


def get_op_loss(x_np, weight_np, bias_np, attr_dict, device):
    torch.npu.set_device(device)
    x = torch.from_numpy(x_np)
    weight = torch.from_numpy(weight_np)
    bias = torch.from_numpy(bias_np)
    result = torch.ops.mxrec.distance_in_linear_silu(
        x.npu(), weight.npu(), bias.npu(), attr_dict
    )
    return result[0].cpu()


@pytest.mark.parametrize("M", [256, 512, 2048, 20480])
@pytest.mark.parametrize("K,N", [(16, 256), (32, 1024), (48, 2304), (64, 4096)])
@pytest.mark.parametrize("device", ["npu:0"])
def test_in_linear_silu(M, K, N, device):
    x = np.random.uniform(0, 1, [M, K]).astype(np.float32)
    weight = np.random.uniform(0, 1, [4 * N, K]).astype(np.float32)
    bias = np.random.uniform(0, 1, [4 * N]).astype(np.float32)
    attr_dict = [N, N, N, N]
    (user, value, query, key) = get_global_loss(x, weight, bias, attr_dict)
    user1 = get_op_loss(x, weight, bias, attr_dict, device)
    assert large_differences(user, user1, threshold=2 ** -11)